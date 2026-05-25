# Fingerprint Capture

`ccproxy` has three different views of a provider request, and fingerprint work
has to keep them separate:

- **Client reference traffic**: the original tool inside the WireGuard namespace.
- **Provider-visible traffic**: the TLS connection made by ccproxy to the real provider.
- **Mitmproxy flow data**: HTTP semantics after TLS has already been terminated.

The TLS fingerprint is treated as an inherent property of every captured
shape: `ccproxy flows shape <provider>` writes the JA3/JA4 material parsed
from the originating ClientHello into the same `.mflow` it persists. At
runtime, any provider whose shape carries an embedded fingerprint
automatically replays through the impersonating sidecar — no explicit
`providers.<name>.fingerprint_profile` is required.

The active code path:

1. [`FingerprintCaptureAddon`](../src/ccproxy/inspector/fingerprint_capture.py)
   reads mitmproxy's TLS ClientHello event, computes JA3/JA4 material, and
   stores it on the later HTTP flow as `metadata_from_flow(flow).fingerprint.client`
   (`ccproxy.fingerprint.client` in serialized flow metadata). This fires
   for both reverse-proxy and WireGuard listeners, so any traffic that
   reaches mitmproxy contributes a fingerprint.
2. [`ShapeCaptureAddon`](../src/ccproxy/inspector/shape_capturer.py) embeds
   that profile into `shapes/{provider}.mflow` metadata as
   `ccproxy.fingerprint.profile` when `ccproxy flows shape {provider}` is run.
   Bundled fallbacks carry the same metadata in
   `ccproxy/templates/shapes/{provider}.mflow`.
3. [`forward_oauth`](../src/ccproxy/hooks/forward_oauth.py) detects the
   `sk-ant-oat-ccproxy-anthropic` sentinel and stores `ctx.metadata.oauth_provider`.
4. [`transform`](../src/ccproxy/inspector/routes/transform.py) rewrites the
   reverse-proxy request to `https://api.anthropic.com/v1/messages`.
5. [`TransportOverrideAddon`](../src/ccproxy/inspector/transport_override_addon.py)
   resolves the fingerprint by precedence: an explicit
   `providers.<name>.fingerprint_profile` wins; otherwise it calls
   `ShapeStore.pick_fingerprint(provider.type)` and engages the sidecar with
   `provider.type` as the impersonate key when the shape carries a captured
   profile. Either way it stores the real target URL in
   `X-CCProxy-Target-Url`, the profile in `X-CCProxy-Impersonate`, and
   rewrites the mitmproxy destination to the localhost sidecar.
6. [`sidecar`](../src/ccproxy/transport/sidecar.py) forwards the request through
   [`httpx-curl-cffi`](../src/ccproxy/transport/dispatch.py). Browser profile
   names use curl-cffi impersonation directly; shape-backed names such as
   `anthropic` load the captured JA3/signature-algorithm/http-version profile.

Set `providers.<name>.fingerprint_profile` only as an override — either to
force a `curl-cffi` browser name (e.g. `chrome131` for `perplexity_pro`,
which has no captured shape counterpart) or to reuse another provider's
captured shape.

## Capture a Profile From Your CLI

Any HTTP client that can be driven through `ccproxy run --inspect` becomes a
source of TLS fingerprints. The WireGuard namespace terminates TLS on the
mitmproxy side, so `FingerprintCaptureAddon` sees the real ClientHello and
attaches it to the flow as `ccproxy.fingerprint.client`.

```bash
# 1. Drive your CLI through the namespaced jail.
ccproxy run --inspect -- <your-tool> <args>

# 2. Find the captured flow for the provider you want to shape.
ccproxy flows list --jq '
  .[] | select(.request.pretty_host == "api.anthropic.com"
            and (.request.path | startswith("/v1/messages"))) | .id
'

# 3. Persist it as the provider's shape (--mflow writes the full flow,
#    embedding ccproxy.fingerprint.profile in its metadata).
ccproxy flows shape anthropic --jq 'map(select(.id == "<flow-id>"))' --mflow

# 4. Done. The next outbound request that ccproxy routes through this
#    provider replays the captured JA3 + signature algorithms via the
#    in-process curl-cffi sidecar. Verify with the tshark recipes below.
```

Substitute `anthropic` for any provider declared in `ccproxy.yaml` (e.g.
`openai`, `deepseek`, a custom provider you added). The provider does not
need an explicit `fingerprint_profile` — the shape's embedded fingerprint
drives the runtime impersonation automatically.

Per-CLI fingerprinting means you can:

- Capture from a vendor's official SDK and route arbitrary harnesses
  through ccproxy as that SDK.
- Swap impersonation by replacing
  `~/.config/ccproxy/shapes/<provider>.mflow` — no daemon restart, no
  config change.
- A/B different clients by capturing each into a distinct provider entry
  that shares the same upstream host.

WireGuard reference traffic also remains useful for comparing against the
real client, even when not shaped — `tls_clienthello` always populates
`ccproxy.fingerprint.client` so the inspector and MCP tools can read it.

## Bundled vs personal shapes

There are two on-disk tiers, with deliberately different fidelity:

- **Personal shapes** at `~/.config/ccproxy/shapes/<provider>.mflow` —
  written by `ccproxy flows shape <provider>` from a real captured
  request. Capture is **deliberately generous**: every observed header
  (except actual auth tokens), the full body, and the
  `ccproxy.fingerprint.profile` metadata all persist. The runtime
  selectively applies fields per `shaping.providers.<name>` config —
  saving more on disk costs nothing and gives future apply-time policy
  changes room to work without recapture.
- **Bundled shapes** at `src/ccproxy/templates/shapes/<provider>.mflow` —
  shipped in the public repo as the working baseline. They MUST NOT
  carry any capturer identity (UUIDs, `metadata.user_id` real values,
  `diagnostics.previous_message_id`, ccproxy-internal correlation
  headers). `scripts/package-mflows.py` is the one-way distillation:

  ```bash
  # capture a fresh shape, then package it for the public bundle:
  ccproxy flows shape anthropic --mflow            # → ~/.config/...
  uv run python scripts/package-mflows.py \
      ~/.config/ccproxy/shapes/anthropic.mflow \
      --out src/ccproxy/templates/shapes/anthropic.mflow

  # pre-commit gate runs in --verify mode:
  uv run python scripts/package-mflows.py --verify
  ```

  The pre-commit hook (`.pre-commit-config.yaml` → `package-mflows-verify`)
  blocks commits if a bundled `.mflow` contains a header in the scrubber's
  drop list, a non-placeholder `metadata.user_id`, a non-null
  `diagnostics.previous_message_id`, a non-empty `tools[]`, or any
  flow-metadata key other than `ccproxy.fingerprint.profile`.

**Degradation note.** The bundled shape's `metadata.user_id` is an
all-zero UUID triple. If Anthropic ever turns identity-presence in
`metadata.user_id` into a detection vector, every install relying on the
bundled fallback will be flagged uniformly. The cure is per-user
capture: `ccproxy flows shape anthropic` → personal shape carries your
real `device_id` / `account_uuid` and survives this class of detection.
The same applies to any future identity-bearing field that gets added to
the scrubber's drop list.

## Tooling

The dev shell includes the packet tools used here:

```bash
nix develop --command bash -lc 'command -v tcpdump; command -v tshark; command -v dumpcap'
```

Host captures need packet-capture privileges. On this workstation, `sudo -n`
is enough for `tcpdump`.

`ccproxy run --inspect` writes TLS key material to `.ccproxy/tls.keylog`; see
[`cli.py`](../src/ccproxy/cli.py) and
[`namespace.py`](../src/ccproxy/inspector/namespace.py). Use that keylog when
decrypting namespace captures.

## Capture Provider-Visible Traffic

Start from the project root with the dev daemon running:

```bash
just restart
ccproxy status --json
```

Capture the host's provider-visible traffic while sending a sentinel-routed
Anthropic request through the reverse proxy:

```bash
mkdir -p .ccproxy/captures
stamp=$(date -u +%Y%m%dT%H%M%SZ)
pcap=".ccproxy/captures/anthropic_provider_${stamp}.pcap"
log=".ccproxy/captures/anthropic_provider_${stamp}.tcpdump.log"

sudo -n tcpdump -i any -s 0 -U -w "$pcap" 'tcp port 443' >"$log" 2>&1 &
pid=$!
sleep 1

curl -sS http://127.0.0.1:4001/v1/messages \
  -H 'content-type: application/json' \
  -H 'x-api-key: sk-ant-oat-ccproxy-anthropic' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 24,
    "stream": false,
    "messages": [
      {
        "role": "user",
        "content": "Reply with exactly: ccproxy anthropic fingerprint probe"
      }
    ]
  }'

sleep 2
sudo -n kill -INT "$pid" 2>/dev/null || true
wait "$pid" || true
printf 'PCAP=%s\n' "$pcap"
```

Extract the provider ClientHello:

```bash
tshark -r "$pcap" \
  -Y 'tls.handshake.type == 1 && tls.handshake.extensions_server_name == api.anthropic.com' \
  -T fields \
  -E header=y \
  -E separator=$'\t' \
  -E occurrence=f \
  -e frame.number \
  -e frame.time_relative \
  -e ip.src \
  -e tcp.srcport \
  -e ip.dst \
  -e tcp.dstport \
  -e tls.handshake.extensions_server_name \
  -e tls.handshake.extensions_alpn_str \
  -e tls.handshake.ja3 \
  -e tls.handshake.ja3_full \
  -e tls.handshake.ja4 \
  -e tls.handshake.ja4_r
```

## Capture Client Reference Traffic

Capture inside the WireGuard namespace to see the real CLI's fingerprint before
ccproxy terminates TLS:

```bash
mkdir -p .ccproxy/captures
stamp=$(date -u +%Y%m%dT%H%M%SZ)
pcap="$PWD/.ccproxy/captures/anthropic_client_${stamp}.pcap"
log="$PWD/.ccproxy/captures/anthropic_client_${stamp}.tcpdump.log"

ccproxy run --inspect -- bash -lc "
  set -euo pipefail
  tcpdump -i any -s 0 -U -w '$pcap' 'tcp port 443' >'$log' 2>&1 &
  pid=\$!
  sleep 1
  claude --model haiku -p 'Reply with exactly: ccproxy anthropic client fingerprint probe'
  sleep 2
  kill -INT \$pid 2>/dev/null || true
  wait \$pid || true
"
printf 'PCAP=%s\n' "$pcap"
```

Use the same ClientHello extraction command against the new pcap.

To persist the captured profile for replay, shape the Anthropic request flow:

```bash
ccproxy flows list --json | jq '.[] | select(.request.pretty_host == "api.anthropic.com" and (.request.path | startswith("/v1/messages"))) | .id'
ccproxy flows shape anthropic --jq 'map(select(.id == "<flow-id>"))'
uv run python - <<'PY'
from pathlib import Path
from mitmproxy import http
from mitmproxy.io import FlowReader
from ccproxy.inspector.fingerprint import REPLAY_FINGERPRINT_METADATA

path = Path.home() / ".config/ccproxy/shapes/anthropic.mflow"
with path.open("rb") as fo:
    flows = [flow for flow in FlowReader(fo).stream() if isinstance(flow, http.HTTPFlow)]
fingerprint = flows[-1].metadata[REPLAY_FINGERPRINT_METADATA]
print({key: fingerprint[key] for key in ("ja3", "ja4", "ja4_r", "http_version", "alpn_protocols")})
PY
```

To inspect decrypted HTTP/1.1 request fields:

```bash
tshark -o tls.keylog_file:.ccproxy/tls.keylog \
  -r "$pcap" \
  -Y 'http.request && http.host == api.anthropic.com' \
  -T fields \
  -E header=y \
  -E separator=$'\t' \
  -E occurrence=f \
  -e frame.number \
  -e frame.time_relative \
  -e ip.src \
  -e tcp.srcport \
  -e http.request.method \
  -e http.host \
  -e http.request.uri \
  -e http.request.version \
  -e http.user_agent
```

## Current Baseline

Measured with Claude Code `2.1.150` against Anthropic:

| Path | JA3 | JA4 | ALPN |
| --- | --- | --- | --- |
| Claude Code inside WireGuard | `d871d02cecbde59abbf8f4806134addf` | `t13d1714h1_5b57614c22b0_43ade6aba3df` | `http/1.1` |
| Shape-backed `anthropic` sidecar | `d871d02cecbde59abbf8f4806134addf` | `t13d1714h1_5b57614c22b0_43ade6aba3df` | `http/1.1` |
| Native mitmproxy provider leg | `5659c10619c455ea477287b12cf3f7e7` | `t13d2812h1_a01be8c064b6_8e6e362c5eac` | `http/1.1` |

Use `tshark` to compare `ALPN + JA3 + JA4 + JA4_r`; that tuple is the repeatable
verification target for sidecar replay.
