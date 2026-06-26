# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository.

## Project Overview

`ccproxy` is a transparent network interceptor for LLM tooling.
It accepts traffic at one of two listeners (a reverse proxy on port 4000, or a rootless WireGuard
namespace jail), feeds each request through a DAG-driven hook pipeline, and forwards directly to the
provider API. Cross-provider request/response transformation is handled by the `lightllm` subpackage
— request-side `UIAdapter` classes for wire ↔ IR projection plus `pydantic_graph` FSMs for SSE
streaming.

The package name is `ccproxy` (lowercase).
The PyPI distribution is `claude-ccproxy`. Python 3.13+. Console script: `ccproxy`
(`ccproxy.cli:entry_point`).

## Commands

```bash
just up          # Start dev services (process-compose, detached, port 4001)
just down        # Stop dev services
just test        # uv run pytest
just lint        # uv run ruff check .
just fmt         # uv run ruff format .
just typecheck   # uv run mypy src/ccproxy
just logs        # process-compose process logs ccproxy
```

```bash
uv run pytest tests/test_config.py            # Single test file
uv run pytest -k "test_token_count"           # Tests matching pattern
uv run pytest -m e2e                          # E2E tests (excluded by default)
```

Coverage threshold is 90% (`--cov-fail-under=90`). `-m "not e2e"` and
`--ignore=tests/test_shell_integration.py` are baked into pytest’s default `addopts`.

The `process-compose` socket is `/tmp/process-compose-ccproxy.sock` (set via `PC_SOCKET_PATH` in the
devShell).
Never run `ccproxy start` with `&`/`disown` — use `just up`/`just down` so process-compose
supervises it.

`just up` is idempotent — it does NOT restart an already-running dev daemon, so source changes won’t
be picked up. After editing ccproxy code, run `just restart` to load the new code.
Production’s systemd unit reloads automatically via `X-Restart-Triggers` only when the generated
YAML changes — code-only changes there require `systemctl --user restart ccproxy`.

### CLI

```bash
ccproxy start                          # Start proxy and inspector stack (foreground)
ccproxy run [--capture] -- <cmd>       # Run command with proxy env vars / WireGuard jail
ccproxy status [--proxy] [--inspect] [--mcp] [--mermaid]  # Health check (bitmask exit codes: 1=proxy, 2=inspect, 4=mcp); --mermaid emits hook DAGs as stateDiagram-v2
ccproxy init [--force]                 # Initialize ~/.config/ccproxy/ccproxy.yaml
ccproxy logs [-f] [-n LINES]           # Tail $CCPROXY_CONFIG_DIR/ccproxy.log
ccproxy flows {list,dump,diff,compare,repl,clear}  # Flow inspection
ccproxy shapes {save,audit}            # Save shape from captured flows / audit packaged .mflow artifacts
ccproxy namespace {status,doctor,wireguard-config}  # WireGuard namespace transparency tools
# MCP server: streamable-HTTP, hosted in-daemon on cfg.mcp.http.port (default 4030; dev 4031)
# clients connect to http://127.0.0.1:<port>/mcp with `Authorization: Bearer <token>`,
# or to /mcp on the proxy port itself (forwarded to the same in-process server)
```

### Smoke Test

```bash
ccproxy run --capture -- claude --model haiku -p "what's 2+2"
```

End-to-end check through the WireGuard namespace: TLS interception, hook pipeline, transform
dispatch, SSE streaming.

## Architecture

### Request/Response Flow

```
ccproxy start
  → mitmweb (reverse + WireGuard listeners, in-process via WebMaster API)
  → InspectorAddon.request() → FingerprintCaptureAddon → MultiHARSaver → ShapeCaptureAddon
    → inbound DAG → transform router (lightllm) → outbound DAG
    → TransportOverrideAddon → AuthAddon → GeminiAddon → PerplexityAddon → EgressSanitizerAddon
  → provider API directly
```

`InspectorAddon` owns OTel span lifecycle, FlowRecord creation, direction detection, and
pre-pipeline request snapshot.
`responseheaders()` sets `flow.response.stream` (either `True` for passthrough or an `SSEPipeline`
for cross-provider transform).
`AuthAddon` runs after the pipeline and detects 401s on flows where `inject_auth` injected a token,
refreshes, and replays.
`GeminiAddon` follows it and handles cloudcode-pa response unwrapping plus capacity (429/503)
sticky-retry and fallback-model walking.

There is no LiteLLM subprocess, no gateway namespace, no second WireGuard tunnel.
Two listeners are bound by mitmweb: `reverse:http://localhost:1@{port}` (placeholder backend,
overwritten by transform) and `wireguard:{conf}@{udp_port}`.

### Addon Chain (registered in `inspector/process.py:_build_addons`)

```
InspectorAddon → FingerprintCaptureAddon → MultiHARSaver → ShapeCaptureAddon
              → ccproxy_inbound (DAG) → ccproxy_transform → ccproxy_outbound (DAG)
              → TransportOverrideAddon → AuthAddon → GeminiAddon → PerplexityAddon
              → EgressSanitizerAddon
```

The pipeline routers are only added when their hook list is non-empty.
`TransportOverrideAddon` runs after the outbound DAG (so it sees ccproxy-finalized requests) and
before `AuthAddon` / `GeminiAddon` — it rewrites `flow.request.host/port/scheme` to the in-process
sidecar (`127.0.0.1:<sidecar_port>`) when the resolved Provider declares a `fingerprint_profile`.
`AuthAddon` and `GeminiAddon` sit after, so they see ccproxy-finalized requests/responses;
`AuthAddon.response` runs before `GeminiAddon.response`, so a 401 → refresh → replay → 429 sequence
cascades into capacity fallback.

### Key Subsystems (`src/ccproxy/`)

When working with `pydantic_graph`, reading all documentation before beginning is critical, and must
be read in their entirely when using the library:

- `./.kitstore/lib/pydantic-ai/docs/graph.md`

- `./.kitstore/lib/pydantic-ai/docs/graph/builder/index.md`

- `./.kitstore/lib/pydantic-ai/docs/graph/builder/steps.md`

- `./.kitstore/lib/pydantic-ai/docs/graph/builder/joins.md`

- `./.kitstore/lib/pydantic-ai/docs/graph/builder/decisions.md`

- `./.kitstore/lib/pydantic-ai/docs/graph/builder/parallel.md`

- `./.kitstore/lib/pydantic-ai/docs/examples/question-graph.md`

- **`lightllm/`** — IR ↔ wire translation.
  `adapters/` does request-side wire ↔ IR (`UIAdapter` subclasses: Anthropic, OpenAIChat
  bidirectional; Google, Perplexity outbound-only).
  `graph/` does response-side SSE streaming via `pydantic_graph` FSMs, plus
  `transform_buffered_response_sync` for non-streaming.
  **Canonical reference: `docs/lightllm.md`.**

- **`pipeline/`** — DAG-based hook execution engine.
  - `context.py` — `Context` wraps `HTTPFlow` (or bare `http.Request` for shapes).
    Typed content (`messages`, `system`, `tools`) is lazy-parsed into Pydantic AI objects; body
    mutations deferred until `commit()`; header mutations immediate.
  - `wire.py` — Bidirectional wire ↔ Pydantic AI conversion.
    Handles `CachePoint` round-trip; supports both Anthropic (`{type, text}`, `input_schema`) and
    OpenAI (`{function: {name, parameters}}`) tool formats.
  - `hook.py` / `dag.py` / `executor.py` — `@hook(reads=..., writes=...)` declares glom-dot-path
    dependencies; `HookDAG` does Kahn topo-sort on root fields; executor isolates errors except
    `AuthConfigError`. Sibling function `{name}_guard` auto-binds as the hook’s guard.
  - `loader.py`, `render.py`, `overrides.py` — Config-list-entry resolution; `rich` status
    rendering; `x-ccproxy-hooks: +hook,-hook` per-request override header.

- **`inspector/`** — mitmproxy addon layer.
  - `addon.py` — `InspectorAddon`: OTel + flow records + direction detection + pre-pipeline snapshot
    \+ provider response capture.
    Owns `responseheaders()` (xepor doesn’t implement it).
  - `auth_addon.py` / `gemini_addon.py` — 401-detect→refresh→replay and capacity
    fallback+envelope-unwrap respectively.
    `GeminiAddon` installs `EnvelopeUnwrapStream` in `responseheaders` for streaming flows.
  - `process.py` — In-process mitmweb via `WebMaster`. Two listeners (reverse + WireGuard);
    WireGuard UDP port found by binding to 0.
  - `pipeline.py` / `router.py` — Bridges hook registry with mitmproxy addons; `InspectorRouter` is
    a vendored xepor `InterceptedAPI` with mitmproxy 12.x compatibility fixes.
  - `routes/{transform,models,health}.py` — Three transform modes (`transform`/`redirect`/
    `passthrough`); synthetic `/v1/models` registered before transform routes.
  - `namespace.py` — Rootless user+net namespace via `unshare` + `slirp4netns` + WireGuard.
    TAP `10.0.2.100/24`, gateway `10.0.2.2`, DNS `10.0.2.3`.
  - `contentview.py`, `shape_capturer.py`, `multi_har_saver.py` — Custom mitmproxy contentviews +
    `ccproxy.shape` / `ccproxy.dump` commands.

- **`hooks/`** — Built-in pipeline hooks.
  Run `ccproxy status` for the live, authoritative view of which hooks are configured, in what
  order, and what each reads/writes.

| Hook | Stage | Purpose |
| --- | --- | --- |
| `inject_auth` | inbound | Substitute sentinel key (`sk-ant-oat-ccproxy-{provider}`); stamps `ctx.metadata.auth_provider` / `ctx.metadata.auth_injected`. |
| `extract_session_id` | inbound | `glom(body, "metadata.user_id")` → `ctx.metadata.session_id`. |
| `extract_pplx_files` | inbound | Upload Perplexity `image_url` parts via batch chain; write S3 URLs to body; strip non-text. Perplexity-guarded. |
| `pplx_thread_inject` | inbound | Three-mode Perplexity thread continuation (body session_id / L1 cache hit / pass-through). |
| `gemini_cli` | outbound | Wrap Gemini bodies in `v1internal` envelope; rewrite paths to `cloudcode-pa`; masquerade SDK UA; idempotent. |
| `pplx_stamp_headers` | outbound | Swap Bearer auth for browser-shape Cookie + UA + Origin + sec-fetch-* bundle. |
| `pplx_preflight` | outbound | Best-effort `GET /search/new?q=...` warm-up before `perplexity_ask`. |
| `inject_mcp_notifications` | outbound | Inject buffered MCP events as synthetic tool_use/tool_result pairs before final user message. |
| `verbose_mode` | outbound | Strip `redact-thinking-*` from `anthropic-beta`. |
| `shape` | outbound | Apply provider-specific packaged/local shape with `content_fields` injection. |
| `commitbee_compat` | outbound | commitbee compatibility shim; `isinstance(_body, dict)` short-circuit. |

- **`shaping/`** — Request shaping framework.

  **IMPERATIVE**: Shape replay is load-bearing for Anthropic identity.
  The previous `inject_claude_code_identity` hook has been removed; shape replay is now the only
  source of the Claude Code identity headers (user-agent, anthropic-beta, x-stainless-*, etc.)
  and the billing-header block.
  If a shape is missing or stale for the `anthropic` provider, requests will fail with 401/400 from
  Anthropic with no fallback.
  Normal users should consume the packaged defaults; do not direct users to capture their own shapes
  as a setup step. Refresh packaged defaults through `scripts/package_mflows.py` when provider SDK
  behavior changes. If a packaged default is stale and no fixed ccproxy release exists yet, point
  users to the manual shaping guide in `docs/shaping.md` as the temporary rescue path.

  A *shape* is a known-good `mitmproxy.http.HTTPFlow` persisted as a `{provider}.mflow`. At runtime,
  the working copy is configured via `http.Request.from_state()`, configured headers are stripped,
  `content_fields` from the provider’s profile are injected from the incoming request per
  `merge_strategies`, shape inner-DAG hooks run, then `apply_shape()` stamps headers + query params
  \+ body onto the outbound flow.
  Packaged defaults live in `src/ccproxy/templates/shapes/` and are public distribution artifacts.
  As of this repo state, `anthropic.mflow`, `gemini.mflow`, and `openai_responses.mflow` are
  packaged defaults. Codex/OpenAI Responses is supported through the default `codex` provider,
  `codex_oauth`, and same-format `openai_responses` redirect with shape replay.
  `scripts/package_mflows.py` is a dev artifact, not a public CLI command.
  It captures real CLI traffic through `ccproxy run --capture`, then prepares public `.mflow` files
  by reusing the same apply-time shaping machinery against canonical SDK requests.
  **IMPERATIVE**: Packaged default `.mflow` files must remain minimal request-only artifacts: no
  response, websocket, error, metadata, `ccproxy.record`, client request snapshot, provider response
  snapshot, auth token, cookie, or captured TLS fingerprint metadata.
  Implicit fingerprint replay from packaged defaults broke Gemini via the sidecar; browser/captured
  fingerprint use must remain an explicit Provider config choice.
  Validate packaged defaults with `uv run ccproxy shapes audit` and `just e2e-packaged-mflows`.
  - `caching/` — Composable glom-based cache control hooks for the shape inner DAG: `strip` (deletes
    via `glom.delete`) and `insert` (sets via `glom.assign`). Used to normalize Anthropic’s
    4-breakpoint `cache_control` limit after `prepend_shape:N` merges.
  - `regenerate.py` — Shape inner-DAG hooks: `regenerate_user_prompt_id`, `regenerate_session_id`,
    `regenerate_request_ids`, `regenerate_billing_header` (re-signs `x-anthropic-billing-header`).
  - `gemini.py` — Gemini-specific shape hook.

- **`flows/store.py`** — TTL store (3600s, lazy cleanup) keyed by `x-ccproxy-flow-id` for
  cross-addon state. `FlowRecord` carries client/forwarded/provider snapshots plus auth/otel/
  transform metadata plus `conversation_id` (SHA12 of first user text) and `system_prompt_sha`.
  `ctx.metadata` / `metadata_from_flow(flow)` are the supported ccproxy metadata access APIs;
  `flow.metadata` is only their mitmproxy backing store.

- **`transport/`** — Cached `httpx.AsyncClient` instances backed by `httpx-curl-cffi`’s
  `AsyncCurlTransport` for browser TLS+HTTP/2 fingerprint impersonation.
  `get_client(*, host, profile)` in `dispatch.py` is the entry point; profile names validate against
  curl-cffi’s `BrowserTypeLiteral`. `sidecar.py` runs an in-process Starlette+uvicorn server that
  `TransportOverrideAddon` redirects flows through via the two-header contract
  (`X-CCProxy-Target-Url` + `X-CCProxy-Impersonate`). `SSLKEYLOGFILE` + `MITMPROXY_SSLKEYLOGFILE`
  both route into `{config_dir}/tls.keylog` so Wireshark decrypts every leg from one file.
  Auth + Gemini retry paths call `get_client(...)` directly, bypassing the sidecar.

- **`auth/sources.py`** — `AuthFields` is the base.
  `CommandAuthSource` (`type: command`) and `FileAuthSource` (`type: file`) are static value
  loaders. `AuthSource(AuthFields)` is the refresh-capable base (60s expiry headroom, atomic
  write-back via tmp+fsync+rename+chmod0o600, glom-configurable
  `access_path`/`refresh_path`/`expiry_path`). `AnthropicAuthSource` and `GoogleAuthSource` extend
  it with provider-specific form refresh bodies.
  `CodexAuthSource` (`type: codex_oauth`) refreshes Codex ChatGPT JWTs from `~/.codex/auth.json` and
  exposes companion account-routing headers.
  `parse_auth_source` accepts bare strings, explicit `type:` discriminators, or `command`/`file` key
  inference.

- **`specs/`** — Vendored constants, Pydantic schemas, model catalog.
  - `claude_code_constants.py` — `BASE_BETAS`, `LONG_CONTEXT_BETAS` (vendored fact lists).
  - `claude_code_request.py` — `APIRequestParams` mirroring `/v1/messages` schema (`extra="allow"`).
  - `billing_salt.py` — Returns the configured `billing_salt` from `CCProxyConfig`. The salt is NOT
    vendored — user supplies via `ccproxy.yaml` `shaping.providers.anthropic.billing.salt` or
    `CCPROXY_BILLING_SALT` env var.
  - `model_catalog.py` — OpenAI-compatible `/v1/models` payload generator.
    `STATIC_MODEL_CATALOG` is the floor list; `build_catalog(refresh=True)` queries each provider’s
    upstream `/v1/models` and unions deduplicated results.

- **`mcp/`** — In-daemon FastMCP streamable-HTTP server (HTTP-only; stdio removed).
  - `server.py` — `FastMCP("ccproxy", stateless_http=True)` singleton with 22 tools spanning flow
    inspection, shape capture, conversation grouping, model catalog, Perplexity quota (60s TTL
    cache), and Perplexity Pro thread library curation (every mutation tool is slug-first).
    The `_MCP_INSTRUCTIONS` block reserves MCP tools for library curation + quota; normal Perplexity
    queries should hit `/v1/chat/completions`. Resources: `proxy://requests`, `proxy://status`. Auth
    via `configure_auth(token, base_url)` before `streamable_http_app()`. Uvicorn lifecycle is in
    `inspector/process.py:run_inspector()` — `log_config=None` + `lifespan="on"` are both mandatory.
  - `buffer.py` — `NotificationBuffer` singleton (default 65536 events/task, 600s TTL, lazy expiry
    on ingest). Ingestion lives on the proxy listener: `inspector/routes/mcp.py` registers
    `POST /mcp/notify` (fire-and-forget, 200-always, no auth) plus a `/mcp` rewrite that forwards
    proxy-listener flows to the in-process FastMCP server — MCP clients can use either
    `http://127.0.0.1:<mcp.http.port>/mcp` or `/mcp` on the proxy port.

- **`flows.py` (CLI)** — `Flows*` tyro subcommands plus `MitmwebClient` for programmatic mitmweb
  REST access. Auth is Bearer token resolved from `inspector.mitmproxy.web_password`. All subcommands
  operate on a resolved flow set:
  `GET /flows → config default_jq_filters → CLI --jq filters → final set`. Filters are jq
  expressions (subprocess; not a Python dependency); each must consume and produce a JSON array.
  Multiple `--jq` flags chain via `|`.

### Configuration

**Discovery**: `$CCPROXY_CONFIG_DIR` (default: `$XDG_CONFIG_HOME/ccproxy/`) is the single knob.
`ccproxy.yaml` is read from it.
The dev shell sets `CCPROXY_CONFIG_DIR=$PWD/.ccproxy` for a project-local config.

**Provenance**: `nix/defaults.nix` is the single source of truth.
`src/ccproxy/templates/ccproxy.yaml` is generated by `flake.nix` via `pkgs.formats.yaml.generate`
(`templateYaml`) and copied into the repo by the dev shell `shellHook` on shell entry.
**Do not edit the template directly**; edit `nix/defaults.nix` and re-enter the dev shell
(`nix develop` or `direnv reload`) to regenerate.
`flake.nix` exports `defaultSettings`, `lib.mkConfig`, and `homeModules.ccproxy`. The repo also has
a local pre-commit hook (`sync-ccproxy-template`) that runs the same refresh and stages
`src/ccproxy/templates/ccproxy.yaml`.

**Hook config format** — each entry is either a dotted module path or a `{hook, params}` dict:

```yaml
hooks:
  outbound:
    - ccproxy.hooks.gemini_cli
    - hook: ccproxy.hooks.shape
    - ccproxy.hooks.verbose_mode
```

**Transform matching** — `lightllm.transforms` is a list of `TransformOverride` rules layered on top
of sentinel-driven Provider routing.
Default is empty. Regex match fields: `match_host` (checked against `pretty_host` + Host +
X-Forwarded-Host), `match_path`, `match_model`. First match wins.
Actions: `redirect` (default), `transform`, `passthrough`. Auth resolves via `dest_provider` →
`config.providers[name]`; `dest_host`/`dest_path` are raw overrides.
Vertex AI: `dest_vertex_project`, `dest_vertex_location`.

**Shaping config** — per-provider profiles.
`content_fields` lists keys injected from the incoming request; everything else persists from the
shape. `merge_strategies` overrides the default `replace`: `prepend_shape`, `append_shape`, `drop`
(`:N` slices the shape’s array first).
`preserve_headers`, `strip_headers`, `capture.path_pattern` are self-explanatory.

### Singleton Patterns

`CCProxyConfig`, `NotificationBuffer`, `FlowStore`, `ShapeStore` are thread-safe singletons.
The `cleanup` autouse fixture in `tests/conftest.py` resets them: `clear_config_instance()`,
`clear_buffer()`, `clear_flow_store()`, `clear_store_instance()`, `clear_shape_hook_cache()`.

### Providers & Sentinel Keys

The sentinel key `sk-ant-oat-ccproxy-{name}` triggers a `providers[name]` lookup via the
`inject_auth` hook: token resolution, target auth header, and routing all flow from a single
`Provider` entry.
ALL API keys in MCP server configs and client environments must be ccproxy sentinel
keys — using raw provider keys bypasses the `inject_auth` hook and the shaping pipeline.
If a destination isn’t routable through a sentinel key, add a `providers` entry for it.

`providers` is a `dict[str, Provider]`. Each `Provider` carries `auth` (an `AnyAuthSource`
discriminated union — `command` / `file` / `anthropic_oauth` / `google_oauth` / `codex_oauth`; bare
YAML strings auto-coerce to `command`), `host` (single destination hostname), `path` (with `{model}`
/ `{action}` templating), `type` (an adapter-family name routed by
`lightllm/graph/__init__.py:dispatch_dump_sync` — `anthropic` / `openai` / `google` / `gemini` /
`vertex_ai` / `vertex_ai_beta` / `perplexity_pro`; Anthropic-compatible forks like `deepseek` and
`zai` use `type: anthropic`), and an optional `fingerprint_profile` (curl-cffi impersonate name,
e.g. `"chrome131"`, `"firefox144"`). `command` and `file` are static value loaders with no expiry
awareness; `anthropic_oauth`, `google_oauth`, and `codex_oauth` own the in-process refresh lifecycle
(60s headroom, atomic write-back to `file_path`). The optional `auth.header` field overrides the
target auth header (default `authorization` with `Bearer`; set to `x-api-key` for raw injection).
On 401, `AuthAddon` re-resolves the credential source; if the token changed, the request is
replayed.

When `fingerprint_profile` is set, `TransportOverrideAddon` rewrites `flow.request` to the
in-process sidecar transport which forwards via `httpx-curl-cffi` — the upstream sees a real browser
TLS+HTTP/2 fingerprint.
Default `None` keeps mitmproxy’s native transport.
The field is validated against `transport.VALID_PROFILES` at config load; invalid names fail-fast.
Opt in per Provider — impersonation has real costs (extra localhost hop, no HTTP/2 multiplexing
across the sidecar, mitmweb’s default view shows the rewritten-to-localhost request rather than the
upstream URL; use the `Forwarded-Request` contentview or `ccproxy flows compare` for the real
upstream intent, and Wireshark with the keylog for the on-the-wire bytes including Chrome-injected
headers).

**Iteration order is load-bearing.** `providers` iteration order determines the no-sentinel fallback
— the first provider with a cached token wins.

**Recommendation for Gemini**: use `type: google_oauth` (with gemini-cli’s installed-app `client_id`
/ `client_secret`, supplied by the user — ccproxy does not vendor them) so `_load_credentials()`
rotates an expired token before `prewarm_project()` POSTs to
`cloudcode-pa.../v1internal:loadCodeAssist` to resolve the `cloudaicompanionProject`. With
`type: command` there is no refresh — if the on-disk token is expired at startup,
`prewarm_project()` silently 401s and every Gemini request lacks the `project` field.

**Perplexity Pro (`perplexity_pro`)**: ccproxy-internal provider routed to
`www.perplexity.ai/rest/sse/perplexity_ask` via a `__Secure-next-auth.session-token` cookie + Chrome
browser-shape headers (stamped by `pplx_stamp_headers`). 22 models in
`specs/perplexity_models.json`. Token refresh via the `perplexity-webui-scraper` UV tool.

> **IMPERATIVE**: Before touching ANY code in `lightllm/pplx.py`, `lightllm/pplx_threads.py`,
> `hooks/pplx_*.py`, `hooks/extract_pplx_files.py`, `inspector/pplx_addon.py`, `mcp/server.py`
> (Perplexity tools), or anything else in the Perplexity surface — **READ `docs/pplx.md` IN ITS
> ENTIRETY**. The document is 1400 lines, covers the full hot path / four SSE patch modes / three
> resume modes / L1 cache lifecycle / multimodal upload chain / fingerprint impersonation / header
> semantics, and includes the troubleshooting catalogue for the specific bugs that surfaced during
> implementation (the `s 4.` truncation, the `equaluals 4.s 4.` doubling, the premature
> `finish_reason=stop`, etc.). Do NOT attempt to reconstruct mental models from this CLAUDE.md
> paragraph or from reading the source alone — the doc captures spec references
> (`~/dev/docs/man/pplx/*.md`), failure modes, and rationale that aren’t in the code comments.

Routing precedence per request: (1) `lightllm.transforms` regex match wins first; (2) sentinel
resolution via `ctx.metadata.auth_provider` / `metadata_from_flow(flow).auth_provider` set by
`inject_auth` resolves to a `providers[name]` lookup; (3) ReverseMode flows fall through to a 501
OpenAI-shape error, WireGuard flows pass through unchanged.
For sentinel-resolved Provider routing the action auto-derives: matching wire format → `redirect`,
otherwise cross-format `transform` via lightllm.

### Anthropic Billing Header

The `regenerate_billing_header` shape inner-DAG hook re-signs the shape’s
`x-anthropic-billing-header` against the incoming first user message.
The salt is a single static reverse-engineered constant and is **never committed to this repo** —
users supply it via `shaping.providers.anthropic.billing.salt` in `ccproxy.yaml` or the
`CCPROXY_BILLING_SALT` env var.
When unset, the hook no-ops with a warning.
Two-phase signing (typed `_body` + serialized wire layer with `xxhash64`): see the docstring in
`src/ccproxy/shaping/regenerate.py`.

### Key Constants (`src/ccproxy/constants.py`)

- `AUTH_SENTINEL_PREFIX` — `sk-ant-oat-ccproxy-`
- `SENSITIVE_PATTERNS` — regex patterns for header redaction
- `CLAUDE_CODE_SYSTEM_PREFIX` — required system prompt prefix for OAuth
- `AuthConfigError` — fatal exception that propagates through pipeline (not swallowed)

Vendored fact lists live separately in `src/ccproxy/specs/claude_code_constants.py`.

## Key Implementation Notes

- **TLS + WireGuard keylogs**: `MITMPROXY_SSLKEYLOGFILE` MUST be set before any mitmproxy import
  (evaluated at module import).
  Set in `_run_inspect()` (`cli.py`) before `run_inspector()`. Both `MITMPROXY_SSLKEYLOGFILE` and
  `SSLKEYLOGFILE` point at `{config_dir}/tls.keylog` (covers mitmproxy + curl-cffi sidecar legs).
  WireGuard tunnel keys go to `{config_dir}/wg.keylog`.
- **SSL CA bundle**: `_ensure_combined_ca_bundle()` combines mitmproxy CA with system CAs, injecting
  via `SSL_CERT_FILE` / `NODE_EXTRA_CA_CERTS` / `REQUESTS_CA_BUNDLE` / `CURL_CA_BUNDLE` for
  `ccproxy run --capture`.
- **Logging**: `FileHandler(cfg.resolved_log_file, mode="w")` truncated on each daemon start.
  Journal identifier from config-dir basename (`~/.config/ccproxy/` → `ccproxy`;
  `~/dev/projects/foo/.ccproxy/` → `ccproxy-foo`). `ccproxy logs` tails the log file.
- **Hook error isolation**: Errors in one hook don’t block others.
  `AuthConfigError` is the exception — it propagates through the pipeline (fatal).
- **Metadata access**: `ctx.metadata` is the ccproxy-owned flow metadata facade backed by
  mitmproxy’s `flow.metadata`. It never mutates request-body `metadata`. Hooks needing body-level
  metadata should use `ctx.extras.get("metadata.foo")`; hooks needing ccproxy flow state should use
  `ctx.metadata.foo` or nested dot access such as `ctx.metadata.pplx.resolved_via`.
- **Three-layer access model** for hooks:
  1. Header ops — `ctx.get_header()` / `ctx.set_header()`
  2. Typed ops — `ctx.system`, `ctx.messages`, `ctx.tools` (Pydantic AI objects)
  3. Raw body ops — `ctx.extras.get(path, default)` / `ctx.extras.set(path, value)` /
     `ctx.extras.delete(path)` / `ctx.extras.has(path)` for typed glom-pathed access;
     `from glom import glom, assign, delete` over `ctx._body` remains valid (the `extras` accessor
     is sugar over the same calls).
     Glom is the standard primitive; `reads`/`writes` declarations on `@hook` use glom dot-paths.
- **SSE streaming**: `flow.response.stream` MUST be set in `responseheaders` (before body arrives).
  xepor doesn’t implement `responseheaders` — that lives on `InspectorAddon` and `GeminiAddon`.
  Setting `stream` in `response` is too late.
- **Namespace localhost routing**: Inside the WireGuard namespace, `127.0.0.1` is isolated loopback
  — host services are at `10.0.2.2` (slirp4netns gateway).
  `route_localnet` sysctl + iptables OUTPUT DNAT rules transparently redirect namespace localhost →
  gateway so tools with hardcoded `127.0.0.1` base URLs work.
  A port remap rule maps the default ccproxy port (4000) to the running instance’s port when they
  differ.
- **Gemini caching + auth header**: Provider-side `cachedContents` caching is currently unsupported
  via the OAuth path (gemini-cli OAuth scopes don’t cover it).
  Gemini OAuth tokens (`ya29.*`) use `Authorization: Bearer`; API keys (`AIza*`) use `?key=` in the
  URL.

## Triage Principle

ALL failures through ccproxy are OUR bug until proven otherwise.
ccproxy is the intermediary — every header, token, body field, and user-agent passes through our
code. When a request fails (401/403/429/5xx), triage ccproxy first: check what we’re injecting,
stripping, mangling, or failing to masquerade before blaming the upstream provider.
For Gemini specifically: if all Gemini requests fail with 401, the in-process `GoogleAuthSource`
refresher should rotate the token automatically; if that fails, inspect `~/.gemini/oauth_creds.json`
(the refresh response sometimes omits `refresh_token` per gemini-cli #21691).

## Dev Instance vs Production Instance

Two ccproxy instances can run concurrently.
They differ only in `CCPROXY_CONFIG_DIR` and the YAML beneath it; `nix/defaults.nix` is the shared
floor.

### Dev (this repo)

`.ccproxy/ccproxy.yaml` is a **read-only symlink into the Nix store**. To change dev settings: edit
`devConfig` in `flake.nix`, then `direnv reload` and `just down && just up`. For one-off
experimental edits: replace the symlink with a real file (`direnv reload` will overwrite it back).
`process-compose` supervises via `just up`/`just down`; socket at
`/tmp/process-compose-ccproxy.sock`; logs at `.ccproxy/ccproxy.log` (truncated each start).

### Production (Home Manager module)

Distributed as `homeModules.ccproxy = import ./nix/module.nix` (re-exported from `flake.nix`).
Consumers import it as a Home Manager module and pass `programs.ccproxy.settings = { ... }` which
deep-merges over `nix/defaults.nix`. Lists (`hooks`, `transforms`, `shape_hooks`) replace wholesale;
only attrsets deep-merge.
`providers` merges per-provider shallowly because `auth` is a discriminated union — partial
overrides would mix exclusive auth keys.

After editing `nix/defaults.nix`, re-enter the dev shell (`nix develop` or `direnv reload`) to
refresh `src/ccproxy/templates/ccproxy.yaml` from `flake.nix`’s `templateYaml`.
