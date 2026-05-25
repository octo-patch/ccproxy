# Development

test:
    uv run pytest

lint:
    uv run ruff check .

fmt:
    uv run ruff format .

typecheck:
    uv run mypy src/ccproxy

package-mflows *ARGS:
    uv run python scripts/package_mflows.py {{ARGS}}

e2e-packaged-mflows:
    tmp=$$(mktemp -d); \
    trap 'CCPROXY_CONFIG_DIR="'"$$tmp"'" process-compose down >/dev/null 2>&1 || true; rm -rf "'"$$tmp"'"' EXIT; \
    cp src/ccproxy/templates/ccproxy.yaml "$$tmp/ccproxy.yaml"; \
    mkdir -p "$$tmp/shapes"; \
    uv run python -c 'import sys, yaml; p=sys.argv[1]; shapes=sys.argv[2]; data=yaml.safe_load(open(p)); cc=data["ccproxy"]; cc["port"]=4001; cc["inspector"]["port"]=8084; cc["mcp"]["http"]["port"]=4031; cc["inspector"]["cert_dir"]=sys.argv[3]; cc["shaping"]["shapes_dir"]=shapes; open(p, "w").write(yaml.safe_dump(data, sort_keys=False))' "$$tmp/ccproxy.yaml" "$$tmp/shapes" "$$tmp"; \
    CCPROXY_CONFIG_DIR="$$tmp" process-compose down >/dev/null 2>&1 || true; \
    CCPROXY_CONFIG_DIR="$$tmp" process-compose up --detached; \
    CCPROXY_CONFIG_DIR="$$tmp" CCPROXY_E2E_PACKAGED_SHAPES=1 CCPROXY_E2E_URL=http://127.0.0.1:4001 uv run pytest -m e2e tests/e2e/test_packaged_mflows_e2e.py

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
