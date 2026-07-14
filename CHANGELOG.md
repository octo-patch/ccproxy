# Changelog

All notable changes to ccproxy are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Added LiteLLM-compatible `config.yaml` model declarations compiled into
  ccproxy's provider, authentication, transform, and `/v1/models` services.
- Added exact and wildcard aliases, deterministic alias chains, includes,
  `api_base`/`base_url`, runtime environment-key sources, metadata, headers,
  organization, and client-wins request defaults for the faithfully supported
  subset.

### Security

- Explicit model endpoints are isolated from native provider credentials,
  headers, query parameters, and fingerprint profiles. Provider transitions
  clear stale header and query credentials before applying the selected
  provider.

## [2.0.0] — 2026-06-12

ccproxy 2.0 is a clean break from 1.x: the LiteLLM proxy subprocess and
gateway architecture are gone, replaced by a single self-contained daemon —
an in-process mitmproxy interceptor with two listeners (reverse proxy +
rootless WireGuard namespace jail), a DAG-driven hook pipeline, and an
in-house cross-provider wire layer.

### Added

- **WireGuard namespace capture**: `ccproxy run --inspect -- <cmd>` runs any
  command in an unprivileged user+network namespace whose only egress is a
  WireGuard tunnel into ccproxy — transparent capture without root, proxy env
  vars, or certificate-pinning workarounds.
- **lightllm wire layer**: request-side `UIAdapter` IR projection plus
  response-side `pydantic_graph` SSE finite-state machines. Cross-format
  streaming between Anthropic, OpenAI Chat Completions, Google/Gemini,
  Perplexity, and the OpenAI Responses listener protocol (Codex CLI).
- **DAG hook pipeline**: hooks declare `reads`/`writes` glom paths and are
  topologically ordered; typed Pydantic AI access to messages/system/tools;
  per-request `x-ccproxy-hooks: +hook,-hook` overrides; pipeline rendering in
  `ccproxy status` (`--mermaid` emits stateDiagram-v2).
- **Request shaping**: provider identity replayed from captured `.mflow`
  shapes with declarative `content_fields` / `merge_strategies` profiles.
  Packaged defaults ship for Anthropic, Gemini, and OpenAI Responses
  (Codex CLI); `ccproxy shapes save` / `ccproxy shapes audit` manage local
  overrides and patch queues.
- **Unified `providers` map**: one sentinel key
  (`sk-ant-oat-ccproxy-{name}`) drives credential injection, header
  selection, routing, and wire-format dispatch. Auth sources: `command`,
  `file`, refresh-capable `anthropic_oauth` / `google_oauth`, and
  `codex_oauth` for Codex's ChatGPT OAuth file plus account-routing headers.
- **Gemini via cloudcode-pa**: OAuth subscription access with `v1internal`
  envelope wrap/unwrap, project prewarm, and capacity fallback (sticky
  retries on 429/503, then a configurable fallback-model walk). Enabled by
  default.
- **Perplexity Pro provider**: OpenAI chat-completions translated onto the
  web-subscription SSE endpoint — four SSE patch modes, three thread-resume
  modes, multimodal uploads through Perplexity's S3 chain, step/citation
  trails rendered into the OpenAI stream, 22 models.
- **TLS fingerprint impersonation**: per-provider `fingerprint_profile`
  routes flows through an in-process curl-cffi sidecar with genuine browser
  TLS+HTTP/2 fingerprints; captured JA3/JA4 material can be replayed from
  shapes.
- **In-daemon MCP server**: FastMCP streamable-HTTP with 22 tools (flow
  inspection, HAR export, diffing, shape capture, conversation grouping,
  model catalog, Perplexity quota + thread library). Reachable at
  `http://127.0.0.1:4030/mcp` or via `/mcp` on the proxy port (same
  in-process server).
- **MCP notification injection**: `POST /mcp/notify` on the proxy listener
  buffers external MCP terminal events (fire-and-forget, 200-always, lazy
  TTL expiry); the `inject_mcp_notifications` hook injects them as synthetic
  `tasks_get` tool exchanges before the final user message.
- **Privacy tooling**: `ccproxy namespace status|doctor|wireguard-config`,
  an egress sanitizer addon, `docs/privacy.md`, and TLS/WireGuard keylogs
  (`tls.keylog`, `wg.keylog`) for independent Wireshark audit.
- **Flow forensics**: `ccproxy flows list|dump|diff|compare|repl|clear` over
  jq filter chains; multi-page HAR 1.2 export; client-vs-forwarded compare;
  conversation grouping; synthetic OpenAI-compatible `/v1/models`.
- **OpenTelemetry**: per-flow spans with HTTP + GenAI semantic conventions
  (OTLP); Jaeger compose file included.
- **Distribution**: Nix flake + Home Manager module (`homeModules.ccproxy`),
  WSL2 `.wsl` artifact build (`nix/wsl.nix` + KVM smoke harness), PyPI
  install validation matrix, bundled Claude Code skills.

### Changed

- Inspector always on: `ccproxy start` runs the interceptor in the
  foreground under your supervisor; `--mitm`/`--detach`/`stop`/`restart`
  are gone.
- Config home moved to `$XDG_CONFIG_HOME/ccproxy/` (was `~/.ccproxy`);
  `--config-dir` became `--config`; `ccproxy install` became `ccproxy init`.
- `debug: true` replaced by `log_level` + `log_file`.
- Telemetry is OTel-only — Prisma/PostgreSQL storage and the `db-*`
  commands are removed.
- Failed response transforms now return an OpenAI-shape 500 error instead of
  passing the untransformed provider body through to the client.
- Auth config failures raise immediately (`AuthConfigError`) instead of
  silently forwarding unauthenticated requests toward a deferred 401.

### Breaking changes — 1.x → 2.0 migration

| 1.x | 2.0 |
| --- | --- |
| LiteLLM subprocess + `config.yaml` | Removed — in-process pipeline; `host`/`port` live in `ccproxy.yaml` |
| `oat_sources` + `inspector.transforms` (primary routing) | `providers` map; `lightllm.transforms` remains as an optional regex override layer |
| `ccproxy install` | `ccproxy init` |
| `--config-dir`; `~/.ccproxy` | `--config`; `$XDG_CONFIG_HOME/ccproxy/` |
| `ccproxy start --mitm` / `--detach`, `stop`, `restart` | Inspector always on; foreground daemon |
| `ccproxy db-sql` / `db-gql` / `db-prompt` | Removed (OTel spans) |
| `debug: true` | `log_level`, `log_file` |
| `ccproxy flows req` / `res` / `client`, `flows --clear` | `flows dump` / `compare` / `clear` |
| `ccproxy flows shape` | `ccproxy shapes save` |
| `ccproxy dag-viz` | `ccproxy status` (+ `--mermaid`) |
| `forward_oauth` hook | `inject_auth` hook |
| `inject_claude_code_identity` / `add_beta_headers` hooks | Shape replay (packaged defaults + local overrides) |
| `compliance` / seed / husk naming | `shaping` / shape; `seeds_dir` → `shapes_dir` |
| MCP over stdio (`ccproxy_mcp` script) | Streamable-HTTP MCP server in-daemon |
| `CCPROXY_MITM_*` env vars; `ccproxy.mitm.*` config | `CCPROXY_INSPECTOR_*`; `inspector.*` |
| `CCPROXY_OTEL_*` env vars | `otel:` config section |

### Security

- March 2026: pinned below LiteLLM 1.82.8 within hours of its
  credential-exfiltration compromise; 2.0 removes the dependency entirely.

### Known limitations

- Anthropic citation deltas (`BetaCitationsDelta`) are not yet forwarded
  through cross-format streaming transforms — they are dropped pending an
  upstream pydantic-ai citation IR part. Anthropic→Anthropic passthrough
  traffic is unaffected.
- Buffered (non-streaming) Gemini responses still unwrap envelopes via the
  legacy path; migration into the intake FSM is planned (Phase R).
- macOS supports the reverse-proxy listener only; the WireGuard namespace
  jail requires Linux (or the WSL2 artifact on Windows).

## [1.2.0] — 2025-12-05

Last release of the 1.x line (LiteLLM-based architecture).
