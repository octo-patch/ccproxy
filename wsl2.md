# WSL2 Strategy for `ccproxy`

Research date: 2026-06-05

## Recommendation

`ccproxy` should support Windows by supporting **WSL2**, not by attempting a native Windows reimplementation of the namespace path.

The best out-of-box shape is:

1. Ship a **`ccproxy.wsl`** artifact.
2. Build it on top of **NixOS-WSL**, not a raw NixOS rootfs.
3. Validate it with Microsoft's **modern `.wsl` validator**.
4. Test it locally on **real Windows** with **PowerShell Core + Pester** by importing an ephemeral distro, running `ccproxy` inside it, and unregistering it afterward.

The lower-effort fallback path is to support **Ubuntu on WSL2 + systemd + Determinate Nix Installer**, but that is not the best out-of-box story.

## Why WSL2 Is The Right Windows Target

`ccproxy`'s Linux-only path already depends on Linux primitives:

- user and network namespaces via `unshare` and `nsenter`
- `slirp4netns`
- `ip`, `iptables`, and routing changes
- WireGuard

WSL2 gives us a real Linux kernel, so the correct move is to run the existing Linux design **inside WSL2**, not to translate it into Windows-specific APIs.

The current Microsoft WSL kernel source config is a strong fit for the namespace path. The current `config-wsl` in `microsoft/WSL2-Linux-Kernel` enables:

- `CONFIG_USER_NS=y`
- `CONFIG_NET_NS=y`
- `CONFIG_NF_TABLES=y`
- `CONFIG_IP_NF_IPTABLES=m`
- `CONFIG_IP6_NF_IPTABLES=m`
- `CONFIG_NETFILTER_XT_TARGET_REDIRECT=m`
- `CONFIG_NETFILTER_XT_MATCH_OWNER=m`
- `CONFIG_WIREGUARD=m`
- `CONFIG_TUN=m`
- `CONFIG_VETH=y`

That is not a guarantee about every machine's loaded modules, but it means the current WSL kernel line is architecturally compatible with what `ccproxy` already does. The runtime truth should still be established by `ccproxy namespace status` and `ccproxy namespace doctor`.

## Current WSL Baseline

As of 2026-06-05:

- The latest `microsoft/WSL` release is **2.7.3**, published on **2026-04-25**.
- Microsoft's modern `.wsl` distro packaging requires **WSL 2.4.4 or newer**.
- Microsoft documents **systemd** support for WSL and says current `wsl --install` Ubuntu defaults to systemd.
- Microsoft documents **mirrored networking** on **Windows 11 22H2 and newer**.

For `ccproxy`, that implies the support target should be:

- **Tier 1**: Windows 11 22H2+ with Store WSL updated to latest stable, systemd enabled, mirrored networking recommended.
- **Tier 2**: Windows 10 / older WSL networking model, best-effort only.

I would define the official Windows support boundary as:

- **Supported**: Store-distributed WSL2
- **Not supported**: WSL1
- **Not supported**: native Windows without WSL

## Why NixOS-WSL Should Be The Base

The strongest upstream precedent is `nix-community/NixOS-WSL`.

Reasons:

- It is explicitly **tested with the Windows Store version of WSL2**.
- It ships a modern **`nixos.wsl`** artifact.
- Its docs already support both:
  - `wsl --install --from-file nixos.wsl`
  - `wsl --import ... nixos.wsl --version 2`
- It has a **tarball builder** that produces a `.wsl` artifact directly.
- It already handles WSL-specific NixOS integration details:
  - a shim before systemd activation
  - WSL-required FHS symlinks
- Its CI validates the tarball with Microsoft's **`validate-modern.py`** script and then runs **Windows-local Pester tests** against imported WSL distros.

This matters because "plain NixOS rootfs in WSL" is not the same thing as "NixOS packaged correctly for WSL". NixOS-WSL already absorbed that complexity.

I would not build a raw NixOS tarball from scratch unless there is a very specific reason to diverge from NixOS-WSL.

## Why Not Stop At Ubuntu + Nix

`DeterminateSystems/nix-installer` is a good fallback and explicitly treats **WSL2 as stable**, with a strong recommendation to enable **systemd** first. That makes it a credible fallback for advanced users.

But it is still a fallback:

- users must start from a distro we do not control
- package prerequisites still need to be assembled correctly
- WSL-specific runtime behavior is less reproducible
- the install story is weaker than "download `ccproxy.wsl`, install, run"

If the goal is real Windows out-of-box support, the distro artifact is the better end state.

## Networking Implications For `ccproxy`

The most important distinction is:

- the **namespace jail itself** runs entirely inside the WSL Linux kernel
- Windows only matters at the boundary where Windows-hosted tools talk to the WSL-hosted proxy

That means:

- the current namespace design should not need a Windows-native rewrite
- the main Windows concern is connectivity and user ergonomics, not feature parity for namespaces

On Microsoft's current docs:

- default WSL2 NAT already lets **Windows -> WSL** clients reach Linux services through `localhost`
- **mirrored networking** improves compatibility, especially for VPNs, IPv6, LAN access, and **WSL -> Windows** `localhost`

So mirrored networking is **recommended**, but it is not the core reason the namespace jail can work.

## Best Packaging Shape

The best packaging direction is:

1. Add a NixOS-WSL-based system definition that includes:
   - `ccproxy`
   - `slirp4netns`
   - `wireguard-tools`
   - `iproute2`
   - `iptables`
   - `util-linux` for `unshare` and `nsenter`
   - `procps`, `curl`, `jq`, `ca-certificates`
2. Enable systemd in the distro.
3. Build a release artifact named something like `ccproxy.wsl`.
4. Make Windows support mean "install this distro and run `ccproxy` inside it".

This lets us fully control the userland that `ccproxy` expects while still relying on the upstream WSL kernel.

## Best Local Test Shape

The best test model is the one `NixOS-WSL` already uses:

- run tests on **Windows**
- use **PowerShell Core**
- use **Pester**
- create a **temporary imported distro**
- run commands through `wsl.exe -d <temp-id> -- ...`
- unregister the distro after the test

Their helper does exactly this with:

- `wsl.exe --import <guid> <tempdir> <tarball> --version 2`
- `wsl.exe -d <guid> -- ...`
- `wsl.exe --unregister <guid>`

That is the right pattern for `ccproxy` too.

I would adapt that model directly and make the local Windows harness authoritative. If we later add CI, CI should run the **same PowerShell test entrypoint**, not a different fake path.

## Concrete Test Plan For `ccproxy`

For a first real WSL2 harness, I would validate:

1. `wsl --update`
2. `wsl --version`
3. import `ccproxy.wsl` into a temporary distro name
4. `systemctl is-system-running`
5. `ccproxy namespace status --json`
6. `ccproxy namespace doctor --json`
7. a minimal `ccproxy run --inspect -- ...` execution
8. unregister the distro and delete the temp directory

The important part is that the tests should exercise the same Linux-only path that Windows users will actually use.

For `ccproxy` specifically, the high-signal checks are:

- required tools are present
- user namespaces are available
- `slirp4netns` works
- the WireGuard config can be consumed
- DNS and IPv4 egress work inside the namespace
- namespace-localhost reachability works for the proxy

That aligns directly with the repo's existing:

- `ccproxy namespace status`
- `ccproxy namespace doctor`

Those commands should become the backbone of WSL validation.

## Artifact Validation

NixOS-WSL's current workflow does something worth copying exactly:

- clone `microsoft/WSL`
- install `distributions/requirements.txt`
- run `distributions/validate-modern.py --tar <artifact>`

That validator checks a lot of packaging correctness we should not reinvent:

- `.wsl` structure
- required `/etc/wsl-distribution.conf` and `/etc/wsl.conf`
- systemd-related rules
- passwd/shadow expectations
- discouraged WSL units
- file ownership and modes
- absence of packaging mistakes like embedded kernel/initramfs

If we ship a `ccproxy.wsl`, this validator should be part of the build/test loop, including local pre-release validation.

## Why Real Windows Testing Matters

`NixOS-WSL` explicitly removed support for running its tests in an emulated WSL environment through Docker. Their tests now require **real Windows**.

That is the correct lesson for `ccproxy`:

- Linux CI can validate Linux semantics
- it cannot prove the Windows + WSL integration boundary
- a real Windows-local test harness is necessary

This matches your stated preference to run the validation locally instead of treating GitHub CI as the primary proof.

## Useful Upstream Precedents

- `nix-community/NixOS-WSL`
  - modern `.wsl` packaging
  - Store WSL2 as the main target
  - tarball builder
  - Windows Pester tests using ephemeral imported distros
  - Microsoft validator in CI

- `microsoft/WSL`
  - latest WSL release line
  - official docs for systemd, networking, custom `.wsl` distros
  - authoritative `validate-modern.py`

- `DeterminateSystems/nix-installer`
  - mature WSL2 Nix support
  - recommends enabling systemd first
  - good fallback path for stock Ubuntu WSL2 users

- `podman-container-tools/podman-machine-os`
  - precedent for shipping a Linux image artifact and verifying it with a Windows-side script
  - the older `containers/podman-machine-wsl-os` repo is now deprecated because the WSL image build moved into the main machine OS repo

## Proposed Staging

### Stage 1: declare the support boundary

- Windows support means **WSL2**
- Store WSL2 only
- systemd required
- mirrored networking recommended

### Stage 2: internal WSL test harness

- build/import temporary distro
- run `namespace status` and `namespace doctor`
- exercise `ccproxy run --inspect`

### Stage 3: ship `ccproxy.wsl`

- base on NixOS-WSL
- include all namespace prerequisites
- validate with Microsoft's script

### Stage 4: make `ccproxy.wsl` the default Windows story

- keep "Ubuntu + systemd + Nix" as an advanced fallback
- do not make it the primary documented path

## Bottom Line

The best implemented WSL2 strategy for `ccproxy` is not "teach Windows how to do Linux namespaces". It is:

- keep the Linux design
- run it inside WSL2
- package the environment as a `.wsl` distro
- validate the artifact with Microsoft's tooling
- test it on real Windows with a local PowerShell Core/Pester harness

If the goal is genuine out-of-box Windows support for the existing namespace jail, **a NixOS-WSL-based `ccproxy.wsl` plus a Windows-local import/test/unregister harness is the strongest path**.

## Sources

- Microsoft WSL latest release: <https://github.com/microsoft/WSL/releases/tag/2.7.3>
- Microsoft WSL install docs: <https://learn.microsoft.com/en-us/windows/wsl/install>
- Microsoft WSL systemd docs: <https://learn.microsoft.com/en-us/windows/wsl/systemd>
- Microsoft WSL networking docs: <https://learn.microsoft.com/en-us/windows/wsl/networking>
- Microsoft custom distro / `.wsl` packaging docs: <https://learn.microsoft.com/en-us/windows/wsl/build-custom-distro>
- Microsoft WSL kernel source: <https://github.com/microsoft/WSL2-Linux-Kernel>
- NixOS-WSL repo: <https://github.com/nix-community/NixOS-WSL>
- NixOS-WSL install docs: <https://nix-community.github.io/NixOS-WSL/install.html>
- NixOS-WSL build docs: <https://nix-community.github.io/NixOS-WSL/building.html>
- Determinate Nix Installer repo: <https://github.com/DeterminateSystems/nix-installer>
- Podman machine OS repo: <https://github.com/podman-container-tools/podman-machine-os>
