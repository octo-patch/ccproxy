#!/usr/bin/env bash
set -euo pipefail

artifact="${1:-${ARTIFACT:-ccproxy.wsl}}"
validator_ref="${WSL_VALIDATOR_REF:-2.7.3}"
validator_dir="${WSL_VALIDATOR_DIR:-tmp/wsl-validator/microsoft-WSL}"

if [[ ! -f "$artifact" ]]; then
  echo "ERROR: WSL artifact not found: $artifact" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required" >&2
  exit 1
fi

mkdir -p "$(dirname "$validator_dir")"

if [[ -d "$validator_dir/.git" ]]; then
  git -C "$validator_dir" fetch --tags --prune origin
else
  git clone https://github.com/microsoft/WSL "$validator_dir"
fi

git -C "$validator_dir" checkout --detach "$validator_ref"

uv run \
  --with-requirements "$validator_dir/distributions/requirements.txt" \
  python "$validator_dir/distributions/validate-modern.py" --tar "$artifact"
