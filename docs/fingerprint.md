# Fingerprint Capture

`ccproxy` has three different views of a provider request, and fingerprint work
has to keep them separate:

- **Client reference traffic**: the original tool inside the WireGuard namespace.
- **Provider-visible traffic**: the TLS connection made by ccproxy to the real provider.
- **Mitmproxy flow data**: HTTP semantics after TLS has already been terminated.

For the Anthropic path, `providers.anthropic.fingerprint_profile` opts routed
reverse-proxy traffic into the in-process sidecar. The active code path is:

1. [`FingerprintCaptureAddon`](../src/ccproxy/inspector/fingerprint_capture.py)
   reads mitmproxy's TLS ClientHello event, computes JA3/JA4 material, and
   stores it on the later HTTP flow as `ccproxy.fingerprint.client`.
2. [`ShapeCaptureAddon`](../src/ccproxy/inspector/shape_capturer.py) writes
   that profile into `shapes/{provider}.mflow` metadata as
   `ccproxy.fingerprint.profile` when `ccproxy flows shape {provider}` is run.
   Bundled fallbacks carry the same metadata in
   `ccproxy/templates/shapes/{provider}.mflow`.
3. [`forward_oauth`](../src/ccproxy/hooks/forward_oauth.py) detects the
   `sk-ant-oat-ccproxy-anthropic` sentinel and stores `ccproxy.oauth_provider`.
4. [`transform`](../src/ccproxy/inspector/routes/transform.py) rewrites the
   reverse-proxy request to `https://api.anthropic.com/v1/messages`.
5. [`TransportOverrideAddon`](../src/ccproxy/inspector/transport_override_addon.py)
   sees the provider's `fingerprint_profile`, stores the real target URL in
   `X-CCProxy-Target-Url`, stores the profile in `X-CCProxy-Impersonate`, and
   rewrites the mitmproxy destination to the localhost sidecar.
6. [`sidecar`](../src/ccproxy/transport/sidecar.py) forwards the request through
   [`httpx-curl-cffi`](../src/ccproxy/transport/dispatch.py). Browser profile
   names use curl-cffi impersonation directly; shape-backed names such as
   `anthropic` load the captured JA3/signature-algorithm/http-version profile.

Captured shape metadata is preserved in the `.mflow` artifact. Runtime shape
application stamps only request headers, query parameters, and body content
onto the active provider request; captured `.mflow` metadata is not copied onto
the active request flow unless code explicitly asks for a specific metadata
entry such as the embedded fingerprint profile.

WireGuard reference traffic is still useful for comparing against the real
client, but it does not automatically exercise the sidecar. It is normally
passed through as already-addressed upstream traffic.

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
