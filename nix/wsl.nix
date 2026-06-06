{
  config,
  lib,
  pkgs,
  ccproxyPackage,
  ...
}:

let
  distroName = "ccproxy";
  nixosWslChannel = "https://github.com/nix-community/NixOS-WSL/archive/refs/heads/main.tar.gz";
  wslDistributionConf = pkgs.writeText "wsl-distribution.conf" (
    lib.generators.toINI { } {
      oobe.defaultName = distroName;
    }
  );
  defaultConfig = pkgs.writeText "configuration.nix" ''
    # This file is the mutable in-distro NixOS configuration.
    # The release artifact itself is built from ccproxy's flake.
    { config, pkgs, ... }:

    {
      imports = [
        <nixos-wsl/modules>
      ];

      wsl.enable = true;
      wsl.defaultUser = "${distroName}";

      environment.systemPackages = with pkgs; [
        cacert
        curl
        iproute2
        iptables
        jq
        procps
        slirp4netns
        util-linux
        wireguard-tools
      ];

      nix.settings.experimental-features = [
        "nix-command"
        "flakes"
      ];

      system.stateVersion = "${config.system.nixos.release}";
    }
  '';
in
{
  wsl.enable = true;
  wsl.defaultUser = distroName;

  networking.hostName = "ccproxy-wsl";

  environment.systemPackages = with pkgs; [
    ccproxyPackage
    cacert
    curl
    iproute2
    iptables
    jq
    procps
    slirp4netns
    util-linux
    wireguard-tools
  ];

  environment.variables.SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";

  nix.settings.experimental-features = [
    "nix-command"
    "flakes"
  ];

  system.stateVersion = config.system.nixos.release;

  system.build.tarballBuilder = lib.mkForce (
    pkgs.writeShellApplication {
      name = "ccproxy-wsl-tarball-builder";
      runtimeInputs = with pkgs; [
        coreutils
        e2fsprogs
        gnutar
        nixos-install-tools
        pigz
        config.nix.package
      ];
      text = ''
        usage() {
          echo "Usage: $0 [output.wsl]"
          exit 1
        }

        if ! [ "$EUID" -eq 0 ]; then
          echo "This script must be run as root"
          exit 1
        fi

        out="ccproxy.wsl"
        if [ "$#" -gt 1 ]; then
          usage
        fi
        if [ "$#" -eq 1 ]; then
          out="$1"
        fi

        root="$(mktemp -p "''${TMPDIR:-/tmp}" -d ccproxy-wsl-tarball.XXXXXXXXXX)"
        cleanup() {
          chattr -Rf -i "$root" >/dev/null 2>&1 || true
          rm -rf "$root" || true
        }
        trap cleanup INT TERM EXIT

        chmod o+rx "$root"

        echo "[ccproxy-wsl] Installing NixOS closure"
        nixos-install \
          --root "$root" \
          --no-root-passwd \
          --system ${config.system.build.toplevel} \
          --substituters ""

        ${lib.optionalString config.nix.channel.enable ''
          echo "[ccproxy-wsl] Adding NixOS-WSL channel"
          nixos-enter --root "$root" --command 'HOME=/root nix-channel --add ${nixosWslChannel} nixos-wsl'
        ''}

        echo "[ccproxy-wsl] Installing WSL distribution metadata"
        install -Dm644 ${wslDistributionConf} "$root/etc/wsl-distribution.conf"

        echo "[ccproxy-wsl] Installing default NixOS configuration"
        install -Dm644 ${defaultConfig} "$root/etc/nixos/configuration.nix"

        echo "[ccproxy-wsl] Compressing $out"
        tar -C "$root" \
          -c \
          --sort=name \
          --mtime="@1" \
          --numeric-owner \
          --hard-dereference \
          . \
          | pigz > "$out"
      '';
    }
  );
}
