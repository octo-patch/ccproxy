# Development

test:
    uv run pytest

lint:
    uv run ruff check .

fmt:
    uv run ruff format .

typecheck:
    uv run mypy src/ccproxy tests --no-incremental
    uv run ty check src tests --output-format concise

package-mflows *ARGS:
    uv run python scripts/package_mflows.py {{ARGS}}

e2e-packaged-mflows:
    tmp=$(mktemp -d); \
    trap 'CCPROXY_CONFIG_DIR="'"$tmp"'" process-compose down >/dev/null 2>&1 || true; rm -rf "'"$tmp"'"' EXIT; \
    cp src/ccproxy/templates/ccproxy.yaml "$tmp/ccproxy.yaml"; \
    cp src/ccproxy/templates/config.yaml "$tmp/config.yaml"; \
    mkdir -p "$tmp/shapes"; \
    uv run python -c 'import sys, yaml; p=sys.argv[1]; shapes=sys.argv[2]; data=yaml.safe_load(open(p)); cc=data["ccproxy"]; cc["port"]=4001; cc["inspector"]["port"]=8084; cc["mcp"]["http"]["port"]=4031; cc["inspector"]["cert_dir"]=sys.argv[3]; cc["shaping"]["shapes_dir"]=shapes; open(p, "w").write(yaml.safe_dump(data, sort_keys=False))' "$tmp/ccproxy.yaml" "$tmp/shapes" "$tmp"; \
    CCPROXY_CONFIG_DIR="$tmp" process-compose down >/dev/null 2>&1 || true; \
    CCPROXY_CONFIG_DIR="$tmp" process-compose up --detached; \
    for i in $(seq 1 60); do CCPROXY_CONFIG_DIR="$tmp" uv run ccproxy status --proxy >/dev/null 2>&1 && break; sleep 1; done; \
    CCPROXY_CONFIG_DIR="$tmp" uv run ccproxy status --proxy; \
    CCPROXY_CONFIG_DIR="$tmp" CCPROXY_E2E_PACKAGED_SHAPES=1 CCPROXY_E2E_URL=http://127.0.0.1:4001 uv run pytest --no-cov -rs -m e2e tests/e2e/test_packaged_mflows_e2e.py

e2e-openai-conversations:
    # Credential-gated smoke for the openai_conversations provider. Requires a
    # running ccproxy with providers.openai_conversations configured (creds +
    # fresh cookies). The tests self-skip when creds/proxy are absent.
    CCPROXY_E2E_URL=${CCPROXY_E2E_URL:-http://127.0.0.1:4001} uv run pytest --no-cov -rs -m e2e tests/e2e/test_openai_conversations_e2e.py

e2e-litellm-config-frontend:
    tmp=$(mktemp -d); \
    trap 'PC_SOCKET_PATH="'"$tmp"'/process-compose.sock" CCPROXY_CONFIG_DIR="'"$tmp"'" process-compose down >/dev/null 2>&1 || true; rm -rf "'"$tmp"'"' EXIT; \
    cp tests/e2e/fixtures/litellm_config_frontend/ccproxy.yaml "$tmp/ccproxy.yaml"; \
    cp tests/e2e/fixtures/litellm_config_frontend/config.yaml "$tmp/config.yaml"; \
    uv run python -c 'import sys, yaml; p=sys.argv[1]; data=yaml.safe_load(open(p)); data["ccproxy"]["inspector"]["cert_dir"]=sys.argv[2]; open(p, "w").write(yaml.safe_dump(data, sort_keys=False))' "$tmp/ccproxy.yaml" "$tmp"; \
    PC_SOCKET_PATH="$tmp/process-compose.sock" CCPROXY_CONFIG_DIR="$tmp" CCPROXY_E2E_LOCAL_KEY=local-e2e-secret process-compose up --detached; \
    for i in $(seq 1 60); do PC_SOCKET_PATH="$tmp/process-compose.sock" CCPROXY_CONFIG_DIR="$tmp" uv run ccproxy status --proxy >/dev/null 2>&1 && break; sleep 1; done; \
    PC_SOCKET_PATH="$tmp/process-compose.sock" CCPROXY_CONFIG_DIR="$tmp" uv run ccproxy status --proxy; \
    CCPROXY_E2E_LITELLM_FRONTEND=1 CCPROXY_E2E_URL=http://127.0.0.1:4011 uv run pytest --no-cov -rs -m e2e tests/e2e/test_litellm_config_frontend_e2e.py

e2e-namespace-observe:
    command -v slirp4netns >/dev/null
    command -v unshare >/dev/null
    command -v nsenter >/dev/null
    command -v ip >/dev/null
    command -v wg >/dev/null
    command -v iptables >/dev/null
    command -v sysctl >/dev/null
    tmp=$(mktemp -d); \
    trap 'CCPROXY_CONFIG_DIR="'"$tmp"'" process-compose down >/dev/null 2>&1 || true; rm -rf "'"$tmp"'"' EXIT; \
    cp src/ccproxy/templates/ccproxy.yaml "$tmp/ccproxy.yaml"; \
    cp src/ccproxy/templates/config.yaml "$tmp/config.yaml"; \
    mkdir -p "$tmp/shapes"; \
    uv run python -c 'import sys, yaml; p=sys.argv[1]; shapes=sys.argv[2]; data=yaml.safe_load(open(p)); cc=data["ccproxy"]; cc["port"]=4001; cc["inspector"]["port"]=8084; cc["mcp"]["http"]["port"]=4031; cc["inspector"]["cert_dir"]=sys.argv[3]; cc["shaping"]["shapes_dir"]=shapes; open(p, "w").write(yaml.safe_dump(data, sort_keys=False))' "$tmp/ccproxy.yaml" "$tmp/shapes" "$tmp"; \
    CCPROXY_CONFIG_DIR="$tmp" process-compose down >/dev/null 2>&1 || true; \
    CCPROXY_CONFIG_DIR="$tmp" process-compose up --detached; \
    for i in $(seq 1 60); do test -s "$tmp/.inspector-wireguard-client.conf" && break; sleep 1; done; \
    test -s "$tmp/.inspector-wireguard-client.conf"; \
    CCPROXY_CONFIG_DIR="$tmp" uv run ccproxy namespace status --json; \
    CCPROXY_CONFIG_DIR="$tmp" uv run ccproxy namespace doctor --json

# Process management
up:
    process-compose up --detached

down:
    process-compose down

restart:
    process-compose down
    process-compose up --detached

logs *ARGS:
    process-compose process logs ccproxy {{ARGS}}

# Build wheel for pip-install validation (mirrors the GHA build-wheel job)
build-wheel:
    rm -rf dist
    uv build --wheel

# Release-gate: boot a vanilla cloud VM and validate the install end-to-end.
# Pre-req: `just build-wheel`.
#
# Usage: just release-test-qemu debian-12 | ubuntu-24.04 | fedora-44
release-test-qemu DISTRO="debian-12":
    test -d dist || just build-wheel
    scripts/qemu_release_test.sh {{DISTRO}}

# Run release-gate test against every supported distro sequentially.
release-test-qemu-all:
    just build-wheel
    scripts/qemu_release_test.sh debian-12
    scripts/qemu_release_test.sh ubuntu-24.04
    scripts/qemu_release_test.sh fedora-44

# Build the x86_64 NixOS-WSL release artifact.
build-wsl ARTIFACT="ccproxy.wsl":
    sudo nix run .#nixosConfigurations.ccproxy-wsl.config.system.build.tarballBuilder -- {{ARTIFACT}}

# Validate a .wsl artifact with Microsoft's modern distro validator.
validate-wsl-artifact ARTIFACT="ccproxy.wsl":
    nix run .#wslArtifactValidator -- {{ARTIFACT}}

# Run the Windows-local WSL2 import/probe/unregister harness.
test-wsl ARTIFACT="ccproxy.wsl":
    pwsh -File scripts/test_wsl.ps1 -Artifact {{ARTIFACT}}

# Build/run a disposable Windows 11 KVM VM and execute the WSL2 harness inside it.
test-wsl-kvm ARTIFACT="tmp/ccproxy-wsl-smoke/ccproxy.wsl":
    nix run .#wslKvmSmoke -- {{ARTIFACT}}
