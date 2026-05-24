# Fingerprint Capture

`ccproxy` has three different views of a provider request, and fingerprint work
has to keep them separate:

- **Client reference traffic**: the original tool inside the WireGuard namespace.
- **Provider-visible traffic**: the TLS connection made by ccproxy to the real provider.
- **Mitmproxy flow data**: HTTP semantics after TLS has already been terminated.

For the Anthropic path, `providers.anthropic.fingerprint_profile` opts routed
reverse-proxy traffic into the in-process sidecar. The active code path is:

1. [`forward_oauth`](../src/ccproxy/hooks/forward_oauth.py) detects the
   `sk-ant-oat-ccproxy-anthropic` sentinel and stores `ccproxy.oauth_provider`.
2. [`transform`](../src/ccproxy/inspector/routes/transform.py) rewrites the
   reverse-proxy request to `https://api.anthropic.com/v1/messages`.
3. [`TransportOverrideAddon`](../src/ccproxy/inspector/transport_override_addon.py)
   sees the provider's `fingerprint_profile`, stores the real target URL in
   `X-CCProxy-Target-Url`, stores the profile in `X-CCProxy-Impersonate`, and
   rewrites the mitmproxy destination to the localhost sidecar.
4. [`sidecar`](../src/ccproxy/transport/sidecar.py) forwards the request through
   [`httpx-curl-cffi`](../src/ccproxy/transport/dispatch.py), which applies the
   selected curl-cffi impersonation profile.

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
| Claude Code inside WireGuard | `d871d02cecbde59abbf8f4806134addf` | `13d1714h1_5b57614c22b0_43ade6aba3df` | `http/1.1` |
| Native mitmproxy provider leg | `5659c10619c455ea477287b12cf3f7e7` | `13d2812h1_a01be8c064b6_8e6e362c5eac` | `http/1.1` |

`chrome131` is expected to change the provider-visible leg from mitmproxy's
native OpenSSL profile to curl-cffi's Chrome-like profile. It is not expected
to match Claude Code's native Node/Bun TLS fingerprint exactly unless curl-cffi
adds a matching impersonation profile.
