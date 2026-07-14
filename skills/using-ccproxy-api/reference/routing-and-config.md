# Model Routing & Configuration

## Contents

- [How routing works](#how-routing-works)
- [ccproxy.yaml configuration](#ccproxyyaml-configuration)
- [Transform rules](#transform-rules)
- [OAuth token management](#oauth-token-management)

---

## How routing works

Request flow through the three-stage addon chain:

```
Client request (model: "claude-sonnet-4-5-20250929")
  │
  ▼
ccproxy_inbound (DAG hooks)
  inject_auth: Detects sentinel key, substitutes real provider token.
  extract_session_id: Parses session_id from metadata.user_id.
  │
  ▼
ccproxy_transform (lightllm dispatch)
  Matches request against lightllm.transforms rules.
  First match wins, then sentinel-key Provider routing is used.
  Rewrites host/path/body to dest_provider format when routing applies.
  │
  ▼
ccproxy_outbound (DAG hooks)
  inject_mcp_notifications: Injects buffered MCP events.
  verbose_mode: Strips redact-thinking from beta header.
  shape: Stamps captured compliance envelopes onto proxied requests.
  │
  ▼
Provider API directly
```

---

## Configuration files

Configuration files are siblings beneath `~/.config/ccproxy/` or
`$CCPROXY_CONFIG_DIR`: `ccproxy.yaml` owns native providers, authentication,
hooks, shaping, inspection, and transform overrides; `config.yaml` owns
LiteLLM-compatible model aliases and destinations.

```yaml
# config.yaml
model_list:
  - model_name: requesty/*
    litellm_params:
      model: requesty/*
      api_base: https://router.requesty.ai/v1
      api_key: os.environ/REQUESTY_API_KEY
```

Routing order is explicit transform override, exact compiled model binding,
wildcard compiled model binding, then sentinel Provider fallback. See
`docs/configuration.md` for the complete compatibility boundary.

### Full OAuth configuration

```yaml
ccproxy:
  host: 127.0.0.1
  port: 4000
  log_level: INFO

  providers:
    anthropic:
      auth:
        type: command
        command: "jq -r '.claudeAiOauth.accessToken' ~/.claude/.credentials.json"
      base_url: https://api.anthropic.com
      path: /v1/messages
      type: anthropic

  hooks:
    inbound:
      - ccproxy.hooks.inject_auth
      - ccproxy.hooks.extract_session_id
    outbound:
      - ccproxy.hooks.inject_mcp_notifications
      - ccproxy.hooks.verbose_mode
      - ccproxy.hooks.shape

  shaping:
    enabled: true
    shapes_dir: ~/.config/ccproxy/shapes

  inspector:
    port: 8083

  lightllm:
    transforms:
      - match_host: cloudcode-pa.googleapis.com
        action: passthrough
      - match_path: /v1/chat/completions
        match_model: gpt-4o
        action: transform
        dest_provider: anthropic
        dest_model: claude-haiku-4-5-20251001
```

### Hook parameters

Hooks accept params via dict form:

```yaml
hooks:
  inbound:
    # Simple (no params)
    - ccproxy.hooks.inject_auth

    # With params
    - hook: ccproxy.hooks.some_hook
      params:
        key: value
```

---

## Transform rules

The default `lightllm.transforms` list is empty: sentinel-keyed flows route through `providers` automatically. Override rules cover edge cases — forcing a specific provider for a path/model combo, bypassing auth for a specific host, etc. Each rule is a `TransformOverride` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `action` | `redirect` \| `transform` \| `passthrough` | Default: `redirect`. Redirect rewrites host/auth only. Transform rewrites body format via lightllm. Passthrough forwards unchanged. |
| `match_host` | `str?` | Regex matched against `pretty_host`, `Host` header, and `X-Forwarded-Host`. |
| `match_path` | `str` | Regex matched against the request path. Default: `.*`. |
| `match_model` | `str?` | Regex matched against the `model` field in the request body. |
| `dest_provider` | `str?` | ccproxy provider name — resolves to a `providers[name]` entry (host/path/auth/format). |
| `dest_model` | `str?` | Rewrites `body['model']`. |
| `dest_base_url` | `str?` | Absolute destination base URL override. Bypasses provider lookup. |
| `dest_path` | `str?` | Raw path override. |
| `dest_vertex_project` | `str?` | GCP project ID for Vertex AI transforms. |
| `dest_vertex_location` | `str?` | GCP region for Vertex AI transforms. |

Auth is resolved via the `dest_provider` lookup: when a rule names `dest_provider: anthropic`, the auth comes from `providers.anthropic.auth` automatically — no separate auth-ref field is needed.

### Examples

```yaml
lightllm:
  transforms:
    # Gemini passthrough (don't transform)
    - action: passthrough
      match_host: cloudcode-pa.googleapis.com

    # Route OpenAI requests to Anthropic
    - match_path: /v1/chat/completions
      match_model: gpt-4o
      action: transform
      dest_provider: anthropic
      dest_model: claude-haiku-4-5-20251001

    # Route all /v1/messages to a different Anthropic model
    - match_path: /v1/messages
      match_model: claude-sonnet
      action: redirect
      dest_provider: anthropic
      dest_model: claude-opus-4-5-20251101
```

First regex match wins. Unmatched reverse proxy flows return a 501 error (OpenAI shape); unmatched WireGuard flows pass through unchanged.

---

## OAuth token management

### providers configuration

A `Provider` entry binds an auth source, a single destination (host + path), and an adapter-family `type` identifier under a sentinel-suffix key. The sentinel key `sk-ant-oat-ccproxy-{name}` resolves to `providers[name]` for token injection and routing.

**Compact form** (bare command string auto-coerces to a `command` auth):
```yaml
providers:
  anthropic:
    auth: "jq -r '.claudeAiOauth.accessToken' ~/.claude/.credentials.json"
    base_url: https://api.anthropic.com
    path: /v1/messages
    type: anthropic
```

**Explicit form**:
```yaml
providers:
  anthropic:
    auth:
      type: command
      command: "jq -r '.claudeAiOauth.accessToken' ~/.claude/.credentials.json"
    base_url: https://api.anthropic.com
    path: /v1/messages
    type: anthropic

  deepseek:
    auth:
      type: command
      command: "printenv DEEPSEEK_API_KEY"
      header: x-api-key       # custom auth header — defaults to Authorization: Bearer
    base_url: https://api.deepseek.com
    path: /anthropic/v1/messages
    type: anthropic           # destination format for lightllm dispatch
```

Provider fields:
- `auth` — discriminated union: `command`, `file`, `anthropic_oauth`, `google_oauth`. A bare string is coerced to `{type: command, command: <string>}`.
- `auth.header` — target header name; omit for the default `Authorization: Bearer {token}`.
- `base_url` — absolute HTTP(S) destination base URL, including an optional port and base path.
- `path` — destination path. Supports `{model}` and `{action}` templating substituted from glom-read body fields and URL captures.
- `type` — adapter-family identifier (`anthropic`, `gemini`, `openai`, `openai_responses`, `perplexity_pro`, …). Drives lightllm dispatch when the incoming format differs from what the destination speaks.

### Token refresh

OAuth-source providers (`anthropic_oauth`, `google_oauth`) refresh in-process via `AuthSource.resolve()` whenever the cached access token is within 60s of expiry — at startup (`_load_credentials()`) and on each header injection. On a 401 from upstream, `AuthAddon.response()` calls `config.resolve_auth_token(provider)` to re-resolve the credential source and replays the request with whatever token the resolver returns. Static `command` / `file` loaders have no refresh capability and rely on whichever secret manager owns rotation.

### Provider resolution

Provider resolution can come from a compiled `config.yaml` model binding or a
sentinel key. `inject_auth` parses `sk-ant-oat-ccproxy-{name}` and looks up
`providers[name]`; model bindings inject their effective Provider through the
same authentication service. `inspector.provider_map` is unrelated: it maps
hostnames to OTel `gen_ai.system` attribution only.
