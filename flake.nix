{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    nixos-wsl = {
      url = "github:nix-community/NixOS-WSL/main";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      nixos-wsl,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs [ "x86_64-linux" "aarch64-linux" ];

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
      defaultSettings = import ./nix/defaults.nix;

      perSystem = forAllSystems (system: let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python313;

        # Rust/C extension wheels that need autoPatchelf fixes
        pyprojectOverrides = final: prev: {
          tokenizers = prev.tokenizers.overrideAttrs (old: {
            buildInputs = (old.buildInputs or []) ++ [ pkgs.stdenv.cc.cc.lib ];
          });
          mitmproxy-rs = prev.mitmproxy-rs.overrideAttrs {
            autoPatchelfIgnoreMissingDeps = true;
          };
          tiktoken = prev.tiktoken.overrideAttrs {
            autoPatchelfIgnoreMissingDeps = true;
          };
          curl-cffi = prev.curl-cffi.overrideAttrs (old: {
            buildInputs = (old.buildInputs or []) ++ [ pkgs.stdenv.cc.cc.lib ];
          });
          # Suppress uv's "Ignoring invalid SSL_CERT_FILE" warning: stdenv sets
          # SSL_CERT_FILE=/no-cert-file.crt to block network access; uv warns on
          # the missing path even though the install is --offline --no-cache.
          ai-ccproxy = prev.ai-ccproxy.overrideAttrs (old: {
            preInstall = (old.preInstall or "") + ''
              export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            '';
          });
        };

        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
            (
              lib.composeManyExtensions [
                pyproject-build-systems.overlays.default
                overlay
                pyprojectOverrides
              ]
            );

        venv = pythonSet.mkVirtualEnv "ccproxy-env" workspace.deps.default;

        yaml = pkgs.formats.yaml { };

        mkConfig =
          {
            settings ? { },
            litellmConfig ? { },
            configDir ? ".ccproxy",
          }:
          let
            deepMerged = lib.recursiveUpdate defaultSettings.settings settings;
            # Provider entries carry a discriminated `auth` union; merge per
            # provider shallowly so a user override replaces the entire entry
            # instead of mixing exclusive auth keys.
            providers =
              (defaultSettings.settings.providers or { })
              // (settings.providers or { });
            mergedSettings = deepMerged // { inherit providers; };
            mergedLiteLLMConfig = lib.recursiveUpdate defaultSettings.litellmConfig litellmConfig;
            ccproxyYaml = yaml.generate "ccproxy.yaml" { ccproxy = mergedSettings; };
            configYaml = yaml.generate "config.yaml" mergedLiteLLMConfig;
          in
          {
            inherit ccproxyYaml configYaml;

            shellHook = ''
              mkdir -p "${configDir}"
              ln -sfn ${ccproxyYaml} "${configDir}/ccproxy.yaml"
              ln -sfn ${configYaml} "${configDir}/config.yaml"
              export CCPROXY_CONFIG_DIR="$PWD/${configDir}"
            '';
          };

        templateCcproxyYaml = yaml.generate "ccproxy.yaml" {
          ccproxy = defaultSettings.settings;
        };
        templateConfigYaml = yaml.generate "config.yaml" defaultSettings.litellmConfig;

        devConfig = mkConfig {
          settings = {
            port = 4001;
            inspector = {
              port = 8084;
              cert_dir = "./.ccproxy";
              mitmproxy = {
                web_password.command = "opc secret op://dev/ccproxy/web_password";
                ignore_hosts = [
                  "oauth2\\.googleapis\\.com"
                  "accounts\\.google\\.com"
                ];
              };
            };
            mcp = {
              http = {
                port = 4031;
              };
            };
            otel = {
              enabled = false;
              endpoint = "http://localhost:4317";
            };
          };
        };
        inspectorRuntimeDeps = with pkgs; [
          slirp4netns
          wireguard-tools
          iproute2
          iptables
          util-linux
          procps
        ];
        inspectorPacketDeps = with pkgs; [
          tcpdump
          wireshark-cli
        ];
        inspectDeps = pkgs.lib.makeBinPath inspectorRuntimeDeps;
        devInspectorDeps = inspectorRuntimeDeps ++ inspectorPacketDeps;
        releaseTestDeps = with pkgs; [
          qemu_kvm
          cloud-utils
          python3
          socat
          xorriso
        ];
        syncCcproxyTemplate = pkgs.writeShellApplication {
          name = "sync-ccproxy-template";
          runtimeInputs = with pkgs; [
            coreutils
            git
          ];
          text = ''
            repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
            install -m 644 ${templateCcproxyYaml} "$repo_root/src/ccproxy/templates/ccproxy.yaml"
            install -m 644 ${templateConfigYaml} "$repo_root/src/ccproxy/templates/config.yaml"
          '';
        };
        wslArtifactValidator = pkgs.writeShellApplication {
          name = "ccproxy-validate-wsl-artifact";
          runtimeInputs = with pkgs; [
            bash
            git
            uv
          ];
          text = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.file
              pkgs.stdenv.cc.cc.lib
            ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            exec bash ${./scripts/validate_wsl_artifact.sh} "$@"
          '';
        };
        wslKvmSmoke = pkgs.writeShellApplication {
          name = "ccproxy-wsl-kvm-smoke";
          runtimeInputs = with pkgs; [
            coreutils
            curl
            gnugrep
            gnused
            jq
            python3
            qemu_kvm
            socat
            xorriso
          ];
          text = ''
            export OVMF_CODE="${pkgs.OVMF.fd}/FV/OVMF_CODE.fd"
            export OVMF_VARS_TEMPLATE="${pkgs.OVMF.fd}/FV/OVMF_VARS.fd"
            exec bash ${./scripts/wsl_kvm_smoke.sh} "$@"
          '';
        };
      in {
        packages = {
          default = pkgs.writeShellScriptBin "ccproxy" ''
            export PATH="${venv}/bin:${inspectDeps}:$PATH"
            exec ${venv}/bin/ccproxy "$@"
          '';
          sync-ccproxy-template = syncCcproxyTemplate;
          inherit wslArtifactValidator;
          inherit wslKvmSmoke;
        };

        apps.sync-ccproxy-template = {
          type = "app";
          program = "${syncCcproxyTemplate}/bin/sync-ccproxy-template";
          meta.description = "Regenerate ccproxy's packaged YAML templates";
        };

        devShells = {
          default = pkgs.mkShell {
            packages = with pkgs; [
              python313
              uv
              ruff
              pyright
              pre-commit
              jq
              git
              just
              process-compose
            ]
            ++ devInspectorDeps
            ++ releaseTestDeps;

            shellHook = ''
              # Nix's python setup hook aggregates every python package in the
              # shell (pre-commit, release-test python3, ...) into PYTHONPATH —
              # a python 3.14 closure that shadows the project's 3.13 venv and
              # breaks ABI-sensitive imports (mypy/librt). The uv venv owns the
              # Python environment; nix supplies self-contained tool wrappers.
              # mypy itself is a uv dev dependency, invoked via `uv run mypy`.
              unset PYTHONPATH
              ${devConfig.shellHook}
              ${syncCcproxyTemplate}/bin/sync-ccproxy-template
              if git rev-parse --git-dir >/dev/null 2>&1; then
                repo_root="$(git rev-parse --show-toplevel)"
                hook_path="$(git rev-parse --git-path hooks/pre-commit)"
                managed_hook="$repo_root/.githooks/pre-commit"
                if [ ! -e "$hook_path" ] || grep -q "ccproxy managed pre-commit hook" "$hook_path" || grep -q "pre-commit.com" "$hook_path"; then
                  mkdir -p "$(dirname "$hook_path")"
                  ln -sfn "$managed_hook" "$hook_path"
                else
                  echo "ccproxy: existing custom pre-commit hook left untouched at $hook_path" >&2
                fi
              fi
              export CCPROXY_BASE_URL="http://127.0.0.1:4001"
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
              ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              export UV_PYTHON_PREFERENCE="only-system"
              export UV_PYTHON_DOWNLOADS="never"
              export UV_PYTHON="${python}"
              uv sync --extra sdk --quiet 2>/dev/null || true
              export VIRTUAL_ENV="$PWD/.venv"
              export PATH="$PWD/.venv/bin:$PWD/result/bin:$PATH"
            '';
          };
        };

        lib = {
          inherit defaultSettings mkConfig;
        };
      });
    in
    {
      packages = lib.mapAttrs (_: v: v.packages) perSystem;
      apps = lib.mapAttrs (_: v: v.apps) perSystem;
      devShells = lib.mapAttrs (_: v: v.devShells) perSystem;
      lib = lib.mapAttrs (_: v: v.lib) perSystem;

      inherit defaultSettings;
      homeModules.ccproxy = import ./nix/module.nix;
      nixosConfigurations.ccproxy-wsl = nixpkgs.lib.nixosSystem {
        system = "x86_64-linux";
        specialArgs = {
          ccproxyPackage = self.packages.x86_64-linux.default;
          nixosWslIcon = "${nixos-wsl}/assets/NixOS-WSL.ico";
        };
        modules = [
          nixos-wsl.nixosModules.default
          ./nix/wsl.nix
        ];
      };
    };
}
