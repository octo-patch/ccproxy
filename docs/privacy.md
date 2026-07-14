# Privacy Guide

ccproxy is a development interceptor. It is designed to make LLM client traffic
observable, debuggable, and transformable while keeping the default proxy path
permissive enough for normal development tools to keep working.

This guide explains what ccproxy's privacy-related features do, what they do
not do, which local artifacts are sensitive, and how to inspect the current
runtime behavior without relying on undocumented assumptions.

## 1. Privacy Model

ccproxy is not an anonymity system, a policy firewall, a sandbox escape
mitigation layer, or a substitute for provider-side privacy controls.

The privacy model is:

- ccproxy keeps traffic inspection local to the ccproxy process and local
  config directory unless you explicitly export, copy, upload, or forward the
  captured data.
- ccproxy can run a client in a Linux network namespace and route that client's
  network traffic through mitmproxy's WireGuard listener for transparent local
  inspection.
- ccproxy does not block arbitrary destinations by default. Unmatched
  WireGuard-captured traffic passes through to the original destination.
- ccproxy deliberately exposes its runtime inputs, generated WireGuard config,
  slirp4netns topology, and live namespace probe results so users can see what
  is happening instead of trusting a handmade security profile.
- ccproxy strips ccproxy-internal correlation headers before upstream egress so
  provider APIs do not receive headers such as `x-ccproxy-flow-id`.

The short version:

```
ccproxy privacy = local transparent inspection + explicit diagnostics + egress hygiene
ccproxy privacy ≠ default network denial policy or provider anonymity
```

## 2. Entry Points And Their Privacy Implications

ccproxy accepts traffic through two different paths. They are intentionally
different.

### Reverse Proxy

The reverse proxy path is used when an SDK or tool points its API base URL at
ccproxy:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4000
export OPENAI_BASE_URL=http://127.0.0.1:4000
```

or:

```bash
ccproxy run -- my-tool
```

Privacy implications:

- The client intentionally talks to ccproxy as its configured API endpoint.
- Only traffic addressed to ccproxy is intercepted.
- Other network traffic from the process is unaffected.
- Unmatched reverse-proxy requests do not have a real default upstream. They
  fail instead of being forwarded to an arbitrary placeholder backend.
- This path is easiest to reason about because only explicitly configured API
  calls enter ccproxy.

Use this path when the tool supports base URL configuration and you only need to
inspect LLM API traffic.

### WireGuard Namespace Capture

The transparent capture path runs a command inside a rootless Linux user+network
namespace:

```bash
ccproxy start
ccproxy run --capture -- claude -p "hello"
```

Privacy implications:

- The child process gets its own network namespace.
- ccproxy configures a WireGuard client inside that namespace.
- The namespace default route goes through the WireGuard interface.
- mitmproxy receives the decrypted traffic through its WireGuard listener.
- ccproxy injects a combined CA bundle into the child process environment so
  TLS clients can trust mitmproxy's local interception certificate.
- Unmatched WireGuard traffic is permissive by default and passes through to the
  original destination.
- Namespace localhost routing is intentionally ergonomic: tools that hardcode
  `127.0.0.1:4000` can still reach the host-side ccproxy listener through
  slirp4netns gateway DNAT.
- A port-forwarding helper watches for local listening ports inside the
  namespace and forwards them through the slirp4netns API. This supports
  development workflows such as OAuth callback listeners.

Use this path when the tool does not support a base URL, when you need to
observe the native CLI's provider traffic, or when you need a reference capture
for shaping/fingerprint work.

Do not treat this path as a privacy firewall. Its job is transparent capture for
development. It is deliberately permissive.

## 3. Namespace Transparency Commands

ccproxy exposes namespace inspection commands so users can examine the current
runtime state without reading source code or inferring behavior from log lines.

### `ccproxy namespace status`

```bash
ccproxy namespace status
ccproxy namespace status --json
```

This command reports static and file-system-observable inputs for the namespace
capture path:

- `mode`: currently `permissive`
- `privacy_claim`: currently `false`
- `runner`: the built-in namespace runner
- `wireguard_config.path`: where mitmproxy's generated client config is stored
- `wireguard_config.present`: whether that file exists
- `topology`: the slirp4netns and WireGuard addresses ccproxy uses
- `tools`: whether required tools are visible on `PATH`

Example JSON shape:

```json
{
  "mode": "permissive",
  "privacy_claim": false,
  "runner": "builtin-unshare-slirp4netns-wireguard",
  "wireguard_config": {
    "path": "/home/user/.config/ccproxy/.inspector-wireguard-client.conf",
    "present": true
  },
  "topology": {
    "guest_ip": "10.0.2.100",
    "gateway_ip": "10.0.2.2",
    "slirp_dns_ip": "10.0.2.3",
    "wireguard_client_ip": "10.0.0.1/32"
  }
}
```

Interpretation:

- `privacy_claim: false` is intentional. ccproxy reports observations and
  implementation facts; it does not claim that the namespace is a restrictive
  privacy boundary.
- `wireguard_config.present: false` usually means `ccproxy start` is not
  running, failed before mitmproxy generated the config, or is using a different
  `CCPROXY_CONFIG_DIR`.
- Tool paths are reported as local diagnostics. They are not sent anywhere by
  the status command.

### `ccproxy namespace doctor`

```bash
ccproxy namespace doctor
ccproxy namespace doctor --json
```

This command creates the same permissive namespace path used by
`ccproxy run --capture`, runs a small probe inside it, then tears the namespace
down.

It checks:

- DNS lookup from inside the namespace
- public IPv4 TCP reachability
- public IPv6 TCP reachability
- reachability of ccproxy on namespace localhost
- the route table observed inside the namespace
- `/etc/resolv.conf` as seen by the namespace process

Doctor fails only for operational problems in the current development path:

- DNS lookup failed
- public IPv4 reachability failed
- ccproxy localhost reachability failed

IPv6 is reported but not considered a failure. Many development machines and
networks do not provide working IPv6, and ccproxy does not currently claim an
IPv6 privacy policy.

Example:

```bash
ccproxy namespace doctor --json | jq '.failures'
```

Expected healthy output for the current permissive path:

```json
[]
```

A healthy doctor run means "the transparent capture path works." It does not
mean "the child process cannot reach anything except provider APIs."

### `ccproxy namespace wireguard-config`

```bash
ccproxy namespace wireguard-config
```

This prints mitmproxy's generated WireGuard client configuration.

That output is sensitive. It can include private key material for the local
WireGuard tunnel. Use it for inspection and debugging, but do not paste it into
issues, chat logs, or public bug reports.

## 4. Network Topology

The current namespace topology is intentionally simple and derived from
slirp4netns plus mitmproxy's WireGuard mode.

```
  ┌─ child process ─────────────────────────────────────┐
  │                                                     │
  │  lo:   127.0.0.1                                   │
  │  tap0: 10.0.2.100/24                               │
  │  wg0:  10.0.0.1/32                                 │
  │                                                     │
  │  default route → wg0                               │
  └──────────────────────┬──────────────────────────────┘
                         │ WireGuard endpoint via slirp
                         ▼
  ┌─ slirp4netns ───────────────────────────────────────┐
  │  gateway: 10.0.2.2                                  │
  │  DNS:     10.0.2.3                                  │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─ mitmproxy WireGuard listener ──────────────────────┐
  │  decrypts tunnel and emits normal HTTPFlow objects  │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
                 ccproxy addon pipeline
```

Important details:

- `10.0.2.100` is the namespace TAP address configured by slirp4netns.
- `10.0.2.2` is the slirp4netns host gateway and the rewritten WireGuard
  endpoint.
- `10.0.2.3` is the slirp/libslirp DNS forwarder address.
- `10.0.0.1/32` is the WireGuard client interface address.
- The default route inside the namespace points at `wg0`.
- ccproxy rewrites mitmproxy's WireGuard endpoint to the slirp gateway because
  `127.0.0.1` inside the namespace is the namespace loopback, not the host
  loopback.

## 5. What Is Kept Local

The following items are local to your machine unless you explicitly move them:

- ccproxy config files
- generated WireGuard client config
- mitmproxy certificate authority files
- TLS and WireGuard keylogs
- captured flows in mitmweb memory
- exported HAR files
- ccproxy log files
- shape captures and packaged-shape development artifacts
- local OpenTelemetry spans before export, when OTel export is disabled

ccproxy does not upload its flow store, logs, keylogs, or generated configs to a
ccproxy service. There is no ccproxy-hosted privacy backend.

Provider APIs still receive whatever request ccproxy ultimately forwards to
them. Transforming a request does not make its prompt, metadata, tool schemas,
or attachments private from the destination provider.

## 6. Sensitive Local Artifacts

Treat the config directory as sensitive. By default it is:

```bash
${XDG_CONFIG_HOME:-$HOME/.config}/ccproxy
```

The project dev shell may instead set:

```bash
CCPROXY_CONFIG_DIR=$PWD/.ccproxy
```

### Configuration files

`ccproxy.yaml` can contain provider definitions, auth source commands, auth
source file paths, explicit transform rules, shaping settings, and MCP settings.
The sibling LiteLLM-compatible `config.yaml` can contain literal API keys,
`os.environ/NAME` references, destinations, headers, organizations, model
metadata, and request defaults.

Even when credentials are loaded through commands or external files, the config
can reveal where secrets live and which providers/accounts are in use.
Protect both files with restrictive permissions. Environment references inside
opaque `model_info` are deliberately left unexpanded so their values cannot be
exposed through the synthetic `/v1/models` response.

### `.inspector-wireguard-client.conf`

This file is generated from mitmproxy's running WireGuard listener. ccproxy uses
it to configure the namespace-side WireGuard client.

It is sensitive because it can contain WireGuard private key material. The
`namespace status` command reports only its path and whether it exists.
`namespace wireguard-config` prints the raw file and should be handled
accordingly.

### `tls.keylog`

At inspector startup, ccproxy sets:

```bash
MITMPROXY_SSLKEYLOGFILE=$CCPROXY_CONFIG_DIR/tls.keylog
SSLKEYLOGFILE=$CCPROXY_CONFIG_DIR/tls.keylog
```

That file lets Wireshark decrypt TLS sessions for intercepted traffic. It is
excellent for local debugging and extremely sensitive for sharing.

Anyone with the packet capture and the matching TLS keylog can decrypt the HTTP
payloads for those sessions.

### `wg.keylog`

ccproxy also writes a WireGuard keylog for decrypting the outer WireGuard tunnel
in packet captures.

Anyone with the packet capture and the matching WireGuard keylog can inspect the
tunnel layer. Combined with `tls.keylog`, the full captured traffic path can be
reconstructed.

### mitmproxy CA Files

mitmproxy generates a local certificate authority for TLS interception.
ccproxy's inspect path injects a combined CA bundle into the child process so
clients can trust locally re-signed certificates.

Do not install the mitmproxy CA into global trust stores unless you understand
the implications. Prefer ccproxy's per-command injected bundle for development
capture.

### Flow Exports

`ccproxy flows dump` emits a HAR file. HAR files can contain:

- prompts
- system prompts
- tool definitions and tool arguments
- image/file references
- provider responses
- request and response headers
- authorization-like headers when present in the captured material
- cookies for browser-shaped traffic
- model names and account/project identifiers

HAR files are debugging artifacts, not safe public logs.

### Logs

ccproxy logs are intended for operational diagnostics, but logs can still reveal
provider names, routes, model names, local file paths, and failure details. Read
logs before sharing them.

## 7. Flow Privacy And Inspection

ccproxy stores recent flow records in memory so CLI and MCP tools can inspect
them.

The flow store:

- is process-local
- is protected by a thread lock
- expires entries after a TTL
- can be cleared through the flows CLI

List flows:

```bash
ccproxy flows list
ccproxy flows list --json
```

Compare what the client sent with what ccproxy forwarded:

```bash
ccproxy flows compare
```

Export flows to HAR:

```bash
ccproxy flows dump > flows.har
```

Clear flows:

```bash
ccproxy flows clear --all
```

Privacy guidance:

- Use `flows compare` locally when debugging transformations. It is often safer
  than exporting a full HAR.
- Prefer jq filters when exporting:

  ```bash
  ccproxy flows dump --jq 'map(select(.request.pretty_host == "api.anthropic.com"))' > anthropic.har
  ```

- Clear captured flows after debugging sensitive sessions:

  ```bash
  ccproxy flows clear --all
  ```

- Treat MCP flow-inspection tools the same as the CLI. MCP clients can see the
  flow data returned by ccproxy's MCP server.

## 8. Egress Hygiene

ccproxy adds internal headers while processing flows. These headers are
implementation details, not provider API inputs.

Examples:

- `x-ccproxy-flow-id`
- `x-ccproxy-hooks`
- `x-ccproxy-auth-injected`

`EgressSanitizerAddon` runs at the end of the mitmproxy addon chain and strips
those ccproxy-internal correlation headers before the request reaches the next
hop.

Two sidecar headers are intentionally excluded from this strip step:

- `x-ccproxy-target-url`
- `x-ccproxy-impersonate`

Those headers are part of the local loopback contract between mitmproxy and the
in-process transport sidecar. The sidecar consumes and strips them before
forwarding to the real upstream provider.

This is egress hygiene, not a general content redaction feature. Request bodies,
tool schemas, prompts, and response bodies still go to the selected provider
unless a hook or transform explicitly changes them.

## 9. Auth And Sentinel Keys

ccproxy's preferred API key surface is the sentinel key:

```text
sk-ant-oat-ccproxy-{provider}
```

When a request uses a sentinel key, the `inject_auth` hook resolves the real
credential from the matching `providers.{provider}.auth` entry and injects it
into the outbound request.

Privacy benefits:

- SDK configs and MCP server configs can contain sentinel keys instead of raw
  provider credentials.
- Per-provider auth resolution stays in ccproxy config.
- OAuth-capable auth sources can refresh tokens inside ccproxy instead of
  requiring clients to manage them.

Limits:

- The real credential is still present in the final outbound request to the
  provider.
- If a client uses a raw provider key directly against ccproxy, it can bypass
  the sentinel-key auth path.
- Flow captures and logs should still be treated as sensitive.

## 10. Shape Artifacts

Shape replay is used to reproduce known-good provider request envelopes while
injecting live request content. Packaged defaults are public distribution
artifacts and are expected to be minimal request-only `.mflow` files.

Packaged shape files must not contain:

- responses
- websocket state
- errors
- ccproxy flow records
- client request snapshots
- provider response snapshots
- auth tokens
- cookies
- captured TLS fingerprint metadata

For local development, shape capture is still sensitive. A locally captured
shape can contain request headers, request bodies, provider-specific envelope
details, and local metadata unless it is explicitly prepared and audited.

Audit packaged shapes with:

```bash
uv run ccproxy shapes audit
```

## 11. OpenTelemetry

OpenTelemetry is optional. When enabled, ccproxy exports spans to the configured
OTLP endpoint.

Span attributes can include:

- request method
- URL
- server address
- provider/model classification
- ccproxy direction/source metadata
- session/conversation identifiers derived from request content

Do not enable OTel export to a third-party collector unless that collector is
allowed to receive operational metadata about your LLM traffic.

Configuration:

```yaml
otel:
  enabled: true
  endpoint: "http://localhost:4317"
  service_name: "ccproxy"
```

## 12. Recommended Workflows

### Inspect The Current Namespace Path

```bash
ccproxy start
ccproxy namespace status
ccproxy namespace doctor
```

Use this before debugging a transparent capture session. It tells you whether
the generated WireGuard config exists, which tools are on `PATH`, and whether
the namespace path can resolve DNS, reach public IPv4, and reach ccproxy on
localhost.

### Capture A Development Session

```bash
ccproxy start
ccproxy run --capture -- claude -p "hello"
ccproxy flows list
ccproxy flows compare
```

Clear flows when done:

```bash
ccproxy flows clear --all
```

### Export A Minimal HAR

Prefer filtered exports:

```bash
ccproxy flows dump \
  --jq 'map(select(.request.path | startswith("/v1/messages")))' \
  > llm-flows.har
```

Review the HAR before sharing it.

### Inspect WireGuard Details

Use status first:

```bash
ccproxy namespace status --json
```

Only print the raw WireGuard config when you need the actual INI:

```bash
ccproxy namespace wireguard-config
```

Do not share the raw output.

### Packet Capture Debugging

When you intentionally need packet-level debugging:

```bash
sudo tcpdump -i any -w ccproxy.pcap
```

Then load:

- `$CCPROXY_CONFIG_DIR/wg.keylog` into Wireshark's WireGuard keylog setting
- `$CCPROXY_CONFIG_DIR/tls.keylog` into Wireshark's TLS keylog setting

Delete or tightly control the resulting files after use. The packet capture plus
keylogs can expose plaintext traffic.

## 13. What ccproxy Does Not Currently Provide

ccproxy intentionally does not expose a user-managed privacy policy DSL.

There is currently no public config for:

- `strict` mode
- `balanced` mode
- firewall backend selection
- DNS policy mode
- IPv6 policy mode
- host access allow/deny policy
- persistent namespace jails

This is deliberate. The namespace path uses existing implementation formats and
observable runtime behavior:

- mitmproxy's generated WireGuard client config
- slirp4netns topology and API behavior
- Linux namespace execution via `unshare` and `nsenter`
- ccproxy's existing transform and hook configuration

The privacy interface is therefore transparent and diagnostic rather than a
custom security configuration surface.

## 14. Troubleshooting

### `wireguard_config.present` Is False

Start ccproxy first:

```bash
ccproxy start
```

Also verify that `CCPROXY_CONFIG_DIR` is the same for `ccproxy start` and
`ccproxy namespace status`.

### `namespace doctor` Fails DNS

Check:

- host DNS works
- `slirp4netns` is installed
- the namespace route table in the doctor JSON
- `/etc/resolv.conf` in the doctor JSON

### `namespace doctor` Fails IPv4

Check:

- host internet access
- WireGuard listener startup logs
- required tools on `PATH`
- firewall rules on the host that may block slirp or UDP loopback traffic

### `namespace doctor` Fails `ccproxy_port_ok`

Check:

- `ccproxy start` is running
- the configured ccproxy port
- whether the command and daemon use the same config directory
- namespace localhost DNAT warnings in `ccproxy logs`

### IPv6 Is Not Reachable

This is reported but not treated as a doctor failure. Many local development
networks lack working IPv6. ccproxy does not currently claim or enforce an IPv6
privacy policy.

### A Tool Still Leaks Data To A Provider

ccproxy is permissive by default. If a tool sends data to a provider and the
traffic is not blocked by your own external controls, ccproxy will generally let
that traffic proceed.

Use:

```bash
ccproxy flows list
ccproxy flows compare
ccproxy flows dump
```

to inspect what happened, then adjust the tool, provider config, transform
rules, hooks, or external network policy as appropriate.

## 15. Sharing Checklist

Before sharing diagnostics, review and redact:

- raw provider API keys
- OAuth access tokens and refresh tokens
- cookies
- `Authorization` headers
- `x-api-key` headers
- prompts and system prompts
- tool call arguments
- file URLs and uploaded file identifiers
- account, project, or workspace IDs
- `.inspector-wireguard-client.conf`
- `tls.keylog`
- `wg.keylog`
- packet captures
- HAR files
- local config paths that reveal secret locations

Prefer sharing command output from:

```bash
ccproxy namespace status --json
```

over raw configs or packet captures. Status output intentionally avoids printing
WireGuard private key material.
