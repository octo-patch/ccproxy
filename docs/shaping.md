# ccproxy Request Shaping

## Introduction

When ccproxy transforms LLM API traffic — rerouting an OpenAI-format request to Anthropic, or channeling a Gemini SDK call through a different endpoint — the resulting outbound request is structurally correct but potentially incomplete. The `lightllm` transform produces valid API payloads, but the non-obvious compliance metadata that makes a request indistinguishable from a native SDK call can be lost: beta headers, user-agent patterns, system prompt preambles, client identity markers, and session metadata.

ccproxy solves this through **request shaping**: it ships sanitized, known-good request templates for built-in providers, then injects the incoming request's content into the template's compliance envelope at runtime.

---

## Packaged Compliance Envelopes

### What a Shape Is

When ccproxy's lightllm transform converts a request, the outbound payload is API-correct but may lack the compliance metadata a native SDK request carries:

- **Beta headers**: `anthropic-beta: prompt-caching-2024-07-31,...`
- **Client identity**: `x-stainless-arch`, `x-stainless-os`, `x-stainless-runtime`
- **User-agent**: The exact UA string the target SDK sends
- **System prompt structure**: Claude Code's compliance preamble as the first system block
- **Metadata identity**: Nested JSON in `metadata.user_id` with `device_id`, `account_uuid`, `session_id`

A **shape** is a known-good request carrying this complete compliance envelope. Packaged defaults and explicit full overrides are stored as response-free `.mflow` files: request state plus preserved flow metadata. Advanced local customization is stored as a quilt-style patch queue against a deterministic `shape.json` projection of that request.

ccproxy ships sanitized default shapes for built-in shaping providers. These bundled shapes are read-only package assets and are used automatically; normal users do not need to capture their own shapes. They are prepared for public distribution as request-only `.mflow` files: no response body, no auth/cookie headers, and no ccproxy flow-record metadata. Advanced local overrides live as small `.patch` files under `$CCPROXY_CONFIG_DIR/shapes/{provider}/`.

Base resolution order is:

1. User full override: `{shapes_dir}/{provider}.mflow`
2. Bundled default: `ccproxy/templates/shapes/{provider}.mflow`
3. No shape: the shape hook no-ops and logs the missing provider shape

After the base is loaded, ccproxy applies the user patch queue from `{shapes_dir}/{provider}/series` if present.

## Manual Shaping When a Packaged Default Is Stale

Most users should never need this section. Use the packaged shapes first.
If a request used to work and now fails after the upstream CLI or SDK changed,
first upgrade ccproxy and try again. A newer ccproxy release may already ship a
refreshed packaged shape.

Use this manual guide when all of these are true:

- You are using a built-in shaped provider such as `anthropic`, `gemini`, or
  `openai_responses`.
- The packaged shape fails, usually with a provider-side 400, 401, or 403.
- There is not yet a ccproxy release with an updated packaged shape.
- The provider's official CLI still works on your machine when run normally.

In plain language: you will run the provider's real CLI once through ccproxy's
inspector, let ccproxy record the working request shape, and then ask ccproxy
to save only the useful request envelope. Your prompts and credentials are not
put into the packaged defaults; this creates a local override in your own
`$CCPROXY_CONFIG_DIR/shapes/` directory.

Use a boring test prompt, not private work. The local patch or `.mflow` can
include pieces of the captured request, and it is meant to stay on your machine.

### Before You Start

Make sure the real provider CLI is installed and logged in:

```bash
# Anthropic / Claude Code
claude -p "reply with ok"

# Gemini CLI
gemini -p "reply with ok"

# Codex CLI
codex exec --ephemeral --ignore-rules --skip-git-repo-check -C /tmp "reply with ok"
```

Make sure ccproxy is running in another terminal:

```bash
ccproxy start
```

For this repository's dev shell, use the supervised dev instance instead:

```bash
just up
```

Check that ccproxy is reachable:

```bash
ccproxy status --proxy --inspect
```

The manual capture command uses `ccproxy run --capture`. That mode requires the
WireGuard namespace prerequisites listed in the README. If `ccproxy run
--capture` reports missing system tools or namespace permissions, fix those
first; `ccproxy shapes save` cannot create a shape until ccproxy has inspected
one real CLI request.

### Step 1: Clear Old Captured Flows

This makes the next steps less confusing. It does not delete your saved shapes;
it only clears the temporary inspection history shown by `ccproxy flows`.

```bash
ccproxy flows clear --all
```

### Step 2: Run One Small Real CLI Request

Choose the provider you are fixing.

For Anthropic / Claude Code:

```bash
ccproxy run --capture -- claude --model haiku -p "Reply with exactly: manual shape ok"
```

For Gemini:

```bash
ccproxy run --capture -- gemini -m gemini-3.1-pro-preview -p "Reply with exactly: manual shape ok"
```

The important part is that the command succeeds. The exact wording of the
prompt is not special; it is just short and easy to recognize in the flow list.

### Step 3: Confirm ccproxy Saw the Provider Request

List the captured flows:

```bash
ccproxy flows list
```

For Anthropic, look for a successful request to `api.anthropic.com` whose path
starts with `/v1/messages`.

For Gemini, look for a successful request to `cloudcode-pa.googleapis.com`
whose path starts with `/v1internal:`.

If you do not see a matching 2xx flow, stop here. The shape would be based on a
failed or unrelated request. Check `ccproxy logs -f`, then run the CLI request
again.

### Step 4: Save the Local Shape Patch

Use the provider-specific command below. The `--jq` filter picks the newest
matching provider request from the flow list, so you do not need to copy a flow
ID by hand.

For Anthropic:

```bash
ccproxy shapes save anthropic \
  --jq 'map(select(.request.pretty_host == "api.anthropic.com" and (.request.path | startswith("/v1/messages")))) | .[-1:]'
```

For Gemini:

```bash
ccproxy shapes save gemini \
  --jq 'map(select(.request.pretty_host == "cloudcode-pa.googleapis.com" and (.request.path | startswith("/v1internal:")))) | .[-1:]'
```

Expected output looks like this:

```text
Saved shape patch for anthropic: /home/you/.config/ccproxy/shapes/anthropic/0001-local-shape.patch
```

or:

```text
Shape patch for anthropic is unchanged.
```

Both are acceptable. `unchanged` means your local capture already matches the
current base shape.

### Step 5: Test the SDK Path Again

Run the SDK, app, or harness that was failing. You do not need to restart
ccproxy; shape patches are read from disk when the shape hook picks the shape.

If you want a small direct check, use the same style as the packaged-shape E2E
tests: make one SDK request through ccproxy with the sentinel key and ask for a
short exact phrase.

For Anthropic SDK clients, the important settings are:

```python
api_key = "sk-ant-oat-ccproxy-anthropic"
base_url = "http://127.0.0.1:4000"
```

For Gemini SDK clients, the important settings are:

```python
api_key = "sk-ant-oat-ccproxy-gemini"
base_url = "http://127.0.0.1:4000/gemini"
```

Then compare the client request with the final request ccproxy forwarded:

```bash
ccproxy flows compare
```

You should see your actual prompt content plus the provider's native headers and
request structure in the forwarded request.

### If Patch Mode Fails

Patch mode is preferred because it keeps your local change small and layered on
top of the packaged default. If `ccproxy shapes save PROVIDER` says there is no
base shape, or if the upstream request changed so much that a patch is not
useful, save a full request-only local override instead:

```bash
ccproxy shapes save anthropic --mflow \
  --jq 'map(select(.request.pretty_host == "api.anthropic.com" and (.request.path | startswith("/v1/messages")))) | .[-1:]'
```

```bash
ccproxy shapes save gemini --mflow \
  --jq 'map(select(.request.pretty_host == "cloudcode-pa.googleapis.com" and (.request.path | startswith("/v1internal:")))) | .[-1:]'
```

`--mflow` writes a request-only local override such as
`~/.config/ccproxy/shapes/anthropic.mflow`. It is still local to your machine.

### Undo the Manual Shape

If the local shape makes things worse, delete it and ccproxy will fall back to
the packaged default on the next request:

```bash
rm -rf ~/.config/ccproxy/shapes/anthropic ~/.config/ccproxy/shapes/anthropic.mflow
rm -rf ~/.config/ccproxy/shapes/gemini ~/.config/ccproxy/shapes/gemini.mflow
```

Use only the provider line you actually changed.

### What to Send When Reporting the Stale Shape

If you open an issue or ask for help, include:

- The provider you refreshed: `anthropic` or `gemini`.
- The CLI version: `claude --version` or `gemini --version`.
- The ccproxy version.
- The upstream status code from the failing request.
- Whether `ccproxy shapes save PROVIDER` wrote a patch or required `--mflow`.

Do not paste auth tokens, cookies, full request bodies, or `.mflow` files into a
public issue.

## Advanced Reference: Local Override Internals

This reference explains what the commands in the manual guide write to disk and
how ccproxy uses those files at runtime.

### Under the Hood

`ccproxy shapes save` resolves the current flow set with the same `--jq`
filtering used by `ccproxy flows`, then invokes `MitmwebClient.save_shape()` →
`POST /commands/ccproxy.shape` → `ShapeCaptureAddon.save_shape_artifact()`
(`inspector/shape_capturer.py`). The addon validates the flow (POST method,
JSON content-type, `capture.path_pattern` regex), prepares a local shape by
removing response-side state and auth/transport/internal request headers,
preserves serializable flow metadata for local overrides, embeds any captured
replay fingerprint under `ccproxy.fingerprint.profile`, and then:

- Default mode: canonicalizes the selected request and provider base into `shape.json`, writes a standard unified diff as `{shapes_dir}/{provider}/0001-local-shape.patch`, and lists it in `{shapes_dir}/{provider}/series`.
- `--mflow` mode: writes a response-free `{shapes_dir}/{provider}.mflow` override via `FlowWriter`.

### Shape Storage

`ShapeStore` (`shaping/store.py`) maintains one shape root containing optional full overrides and patch queues:

```
~/.config/ccproxy/shapes/
├── anthropic.mflow
├── anthropic/
│   ├── series
│   └── 0001-local-shape.patch
├── openai_responses.mflow
├── openai_responses/
│   ├── series
│   └── 0001-local-shape.patch
├── gemini/
│   ├── series
│   └── 0001-local-shape.patch
└── ...

<package>/ccproxy/templates/shapes/
├── anthropic.mflow
├── gemini.mflow
├── openai_responses.mflow
└── ...
```

- **Append-only**: Each `add()` appends; previous shapes are preserved
- **User overrides win**: `pick()` returns the latest user shape first, then the bundled default
- **Patch queues apply last**: `{provider}/series` patches apply to either the user override or bundled default
- **Native format**: Inspectable via `mitmweb --rfile`
- **Thread-safe**: All operations under a threading lock
- **Clear means revert**: Clearing a user shape deletes the override and patch queue; the bundled default remains available

```yaml
shaping:
  enabled: true
  shapes_dir: ~/.config/ccproxy/shapes
```

The `series` file is a quilt-style ordered patch manifest:

```text
# applied top to bottom
0001-local-shape.patch
0002-another-change.patch -p1
```

Each patch is a standard unified diff against virtual `shape.json`. Git-style paths (`a/shape.json`, `b/shape.json`) use the default `-p1` strip level.

---

## The Shaping Pipeline

### Conceptual Model

The shape is the proven request envelope — a packaged or local flow carrying the full compliance metadata. At runtime, ccproxy creates a working copy, strips configured headers, injects the incoming request's content into declared fields, runs shape hooks (inner DAG) for dynamic operations, and stamps the result onto the outbound flow.

The identity/content boundary is declared per-provider in YAML config. `content_fields` lists the body keys that come from the incoming request. Everything NOT listed persists from the shape — compliance headers, beta flags, system prompt preamble, metadata skeleton, client identity markers. This inversion means the system doesn't need to enumerate what the envelope contains; it declares what it intends to inject.

```
Shape (packaged/local flow)
  │
  ▼
Deep copy shape.request → working Shape
  │
  ▼
┌──────────────────────────┐
│     STRIP phase          │  Strip headers (auth, transport)
│                          │  per profile.strip_headers
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   INJECT phase           │  Two-pass strip & fill of
│                          │  profile.content_fields using
│                          │  profile.merge_strategies
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  SHAPE HOOKS phase       │  Run profile.shape_hooks via
│                          │  inner DAG (e.g., UUID re-roll)
└──────────┬───────────────┘
           │
           ▼
shape_ctx.commit()            Flush body mutations to working.content
           │
           ▼
apply_shape(working, ctx,     Stamp shape headers + query params + body
  profile.preserve_headers)   onto outbound flow, preserving auth + host
           │
           ▼
Outbound flow carries shape's
compliance envelope with the
incoming request's content
```

### The Shape Hook

The `shape` hook (`hooks/shape.py`) runs last in the outbound pipeline. Its guard condition (`shape_guard`) ensures it only fires when:

- The flow entered via **reverse proxy** OR has the `ccproxy.auth_injected` flag
- AND the `FlowRecord` has a completed `TransformMeta`

WireGuard passthrough flows (already authentic) and flows without a transform are not shaped.

When it fires:

1. Gets the provider from `record.transform.provider_type`
2. Looks up `ProviderShapingConfig` from `config.shaping.providers[provider]`
3. `store.pick(provider)` — fetches the most recent user shape, falling back to the bundled default, then applies the provider patch queue
4. `http.Request.from_state(captured.request.get_state())` — deep-copies as a working `Shape`
5. `strip_headers(shape_ctx, profile.strip_headers)` — removes configured headers
6. `_inject_content(shape_ctx, incoming_ctx, profile)` — content injection per merge strategy
7. Runs shape hooks from `profile.shape_hooks` via inner `HookDAG`
8. `shape_ctx.commit()` — flushes body mutations to working request bytes
9. `apply_shape(working, ctx, profile.preserve_headers)` — stamps onto the outbound flow

### Content Injection

`_inject_content(shape_ctx, incoming_ctx, profile)` operates in two passes:

**Pass 1 — Strip**: For each key in `content_fields`, snapshot the shape's value (needed for non-replace strategies), then remove the key from the shape body. After this pass, the shape contains only envelope fields.

**Pass 2 — Fill**: For each key in `content_fields`, inject from the incoming request per the field's merge strategy:

| Strategy | Behavior | Use case |
|---|---|---|
| `replace` (default) | Incoming value replaces shape value. If incoming doesn't have the field, it stays absent. | model, messages, tools, stream, max_tokens |
| `prepend_shape` | Shape's original value prepended before incoming: `[*shape, *incoming]`. Strings auto-wrapped to `[{type: text, text: ...}]`. Append `:N` to keep only the first *N* shape elements (e.g. `prepend_shape:2`). | system (shape preamble + incoming prompt) |
| `append_shape` | Incoming first, shape appended: `[*incoming, *shape]`. Same string normalization. Append `:N` to keep only the first *N* shape elements. | Alternative system ordering |
| `drop` | Field removed entirely (already stripped in pass 1). | Suppress a field |

Null values from either side are coerced to empty lists for safe spreading.

### Shape Hooks (Inner DAG)

Shape hooks handle operations that can't be expressed as field injection — things that require cross-field logic, ID generation, or structural body mutations. They are standard `@hook(reads=..., writes=...)` decorated functions, DAG-ordered by their declarations and executed via `HookDAG` against the shape context. All raw body access uses glom (`glom()`, `assign()`, `delete()` from the `glom` package) — the standard primitive for `ctx._body` mutations across the entire hook system. The `reads`/`writes` declarations use glom dot-paths (e.g. `"metadata.user_id"`, `"system.*.cache_control"`) which the DAG resolves to root fields for dependency ordering.

Each hook has signature `(ctx: Context, params: dict) -> Context` where `ctx` is the shape context. The incoming pipeline context is available via `params["incoming_ctx"]`.

Shape hooks can be either bare module paths (all `@hook`-decorated functions in the module are loaded) or `{hook, params}` dicts for parameterized hooks with a `model=` Pydantic schema:

```yaml
shape_hooks:
  # Bare module path — loads all @hook functions from the module
  - ccproxy.shaping.regenerate
  # Parameterized hook — dict with hook path and params
  - hook: ccproxy.shaping.caching.strip
    params:
      paths: ["system.*.cache_control"]
```

#### Built-in Shape Hooks

| Hook | Module | Purpose |
|---|---|---|
| `regenerate_user_prompt_id` | `ccproxy.shaping.regenerate` | Re-rolls `user_prompt_id` via `glom()`/`assign()`. reads/writes=`["user_prompt_id"]`. |
| `regenerate_session_id` | `ccproxy.shaping.regenerate` | Parses nested JSON in `metadata.user_id` via `glom()`, re-rolls `session_id` into a fresh UUID4. reads/writes=`["metadata.user_id"]`. |
| `strip` | `ccproxy.shaping.caching.strip` | Deletes values at glom dot-paths via `delete()`. Parameterized via `StripParams(paths: list[str])`. reads/writes=`["system.*.cache_control", "tools.*.cache_control", "messages.*.content.*.cache_control"]`. |
| `insert` | `ccproxy.shaping.caching.insert` | Sets a value at a glom dot-path via `assign()`. Parameterized via `InsertParams(path: str, value: Any)`. Default value: `{"type": "ephemeral"}`. reads/writes=`["system.*.cache_control", "tools.*.cache_control"]`. |

### Cache Breakpoint Hooks

Anthropic limits explicit `cache_control` breakpoints to 4 per request. When `prepend_shape:2` merges the shape's system preamble (which carries its own `cache_control` annotations) with the incoming system prompt, the total breakpoint count can exceed this limit, causing API rejections.

The caching hooks in `ccproxy.shaping.caching` solve this by normalizing breakpoints after content injection: strip all existing breakpoints, then insert exactly one at the optimal position for prefix caching.

#### strip

Deletes values at one or more glom dot-paths using `glom.delete()` with `ignore_missing=True`. Non-existent paths are silently skipped.

```yaml
- hook: ccproxy.shaping.caching.strip
  params:
    paths: ["system.*.cache_control"]
```

**`StripParams` fields:**

| Field | Type | Description |
|---|---|---|
| `paths` | `list[str]` | Glom dot-paths to delete. Supports wildcards. |

#### insert

Sets a value at a single glom dot-path using `glom.assign()`. If the target path doesn't exist (e.g., empty list), the operation is silently skipped.

```yaml
- hook: ccproxy.shaping.caching.insert
  params:
    path: "system.-1.cache_control"
    value: {type: ephemeral}
```

**`InsertParams` fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | — | Glom dot-path target. |
| `value` | `Any` | `{"type": "ephemeral"}` | Value to set at the path. |

#### Default Anthropic Configuration

The default config strips all `cache_control` from system blocks, then inserts one on the last block (optimal for prefix caching — the longest shared prefix gets cached):

```yaml
shape_hooks:
  - ccproxy.shaping.regenerate
  - hook: ccproxy.shaping.caching.strip
    params:
      paths: ["system.*.cache_control"]
  - hook: ccproxy.shaping.caching.insert
    params:
      path: "system.-1.cache_control"
      value: {type: ephemeral}
```

**Before** (after `prepend_shape:2` merges system blocks):
```
system[0]: shape preamble    → cache_control: {type: ephemeral}  ← from shape
system[1]: shape preamble    → cache_control: {type: ephemeral}  ← from shape
system[2]: app system block  → (none)
system[3]: app system block  → cache_control: {type: ephemeral}  ← from client
system[4]: app system block  → cache_control: {type: ephemeral}  ← from client
```
Total: 4 breakpoints. Any additional client breakpoint exceeds the limit.

**After** (strip + insert):
```
system[0]: shape preamble    → (stripped)
system[1]: shape preamble    → (stripped)
system[2]: app system block  → (stripped)
system[3]: app system block  → (stripped)
system[4]: app system block  → cache_control: {type: ephemeral}  ← inserted
```
Total: 1 breakpoint. The last block is the optimal position because prefix caching benefits from caching the longest shared prefix.

#### Glom Dot-Path Syntax

All hooks that perform raw body mutations use [glom](https://glom.readthedocs.io/) as the standard primitive — both for runtime access (`glom()`, `assign()`, `delete()` over `ctx._body`) and for `reads`/`writes` declarations that drive DAG dependency ordering. The DAG extracts the root field from each dot-path (e.g. `"system.*.cache_control"` → `"system"`) for dependency resolution. Paths are dot-separated, with special syntax for list access:

| Pattern | Meaning | Example |
|---|---|---|
| `field.*.key` | Wildcard — iterates all items in the list | `system.*.cache_control` strips `cache_control` from every system block |
| `field.0.key` | Specific index | `system.0.cache_control` targets the first system block |
| `field.-1.key` | Negative index (last item) | `system.-1.cache_control` targets the last system block |
| `a.b.c` | Nested dict traversal | `metadata.user_id` reaches into nested dicts |

Numeric path segments auto-coerce to list indices. Non-numeric segments are dict key lookups.

### apply_shape()

`apply_shape(shape, ctx, preserve_headers)` (`shaping/models.py`) stamps the shape onto the outbound flow:

1. Snapshot `preserve_headers` values from the target flow (auth headers from `inject_auth`, host from redirect handler)
2. Clear ALL headers on the target flow
3. Copy ALL shape headers (compliance headers, user-agent, beta flags, x-stainless-*, etc.)
4. Restore the preserved headers (overwriting any shape values for those keys)
5. Merge query parameters from the shape (e.g. `?beta=true`)
6. Set `flow.request.content = shape.content`
7. Resync `ctx._body` from the shape content

Auth headers from `inject_auth` and the `host` from the transform router survive shaping. Everything else comes from the shape's compliance envelope. The `preserve_headers` list is configurable per-provider.

### Configuration

The shape hook reads its behavior entirely from the per-provider shaping profile in `config.shaping.providers`. The hook is a bare module path — no `{hook, params}` wrapper needed:

```yaml
hooks:
  outbound:
    - ccproxy.hooks.inject_mcp_notifications
    - ccproxy.hooks.verbose_mode
    - ccproxy.hooks.shape

shaping:
  enabled: true
  shapes_dir: ~/.config/ccproxy/shapes
  providers:
    anthropic:
      content_fields:
        - model
        - messages
        - tools
        - tool_choice
        - system
        - thinking
        - context_management
        - stream
        - max_tokens
        - temperature
        - top_p
        - top_k
        - stop_sequences
      merge_strategies:
        system: "prepend_shape:2"
      shape_hooks:
        - ccproxy.shaping.regenerate
        - hook: ccproxy.shaping.caching.strip
          params:
            paths: ["system.*.cache_control"]
        - hook: ccproxy.shaping.caching.insert
          params:
            path: "system.-1.cache_control"
            value: {type: ephemeral}
      preserve_headers:
        - authorization
        - x-api-key
        - x-goog-api-key
        - host
      strip_headers:
        - authorization
        - x-api-key
        - x-goog-api-key
        - content-length
        - host
        - transfer-encoding
        - connection
      capture:
        path_pattern: "^/v1/messages"
```

**Field reference (`ProviderShapingConfig`):**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `content_fields` | `list[str]` | `[]` | Body keys injected from incoming request |
| `merge_strategies` | `dict[str, str]` | `{}` | Per-field override: replace, prepend_shape[:N], append_shape[:N], drop |
| `shape_hooks` | `list[str \| dict]` | `[]` | Dotted module paths or `{hook, params}` dicts containing `@hook`-decorated functions, DAG-ordered |
| `preserve_headers` | `list[str]` | auth + host | Target headers apply_shape must NOT overwrite |
| `strip_headers` | `list[str]` | auth + transport | Shape headers to remove before stamping |
| `capture.path_pattern` | `str` | `""` | Regex for flow validation during `ccproxy shapes save` |

The packaged `openai_responses` profile for Codex includes
`ccproxy.shaping.codex`. That inner shape hook preserves the captured Codex
instruction envelope, converts public SDK string `input` into Codex's verbose
message-list form, and enforces `store: false`. The ChatGPT Codex backend does
not accept every public `/v1/responses` option; keep the default provider on
streaming requests and omit public-only fields such as `max_output_tokens`.

### Writing Custom Shape Hooks

Shape hooks use the standard `@hook` decorator with `reads`/`writes` for DAG ordering.

**Simple hook** (no parameters — registered as a bare module path):

```python
# myproject/shaping/custom.py
from typing import Any
from glom import assign, glom
from ccproxy.pipeline.context import Context
from ccproxy.pipeline.hook import hook

@hook(reads=["custom_tracking_id"], writes=["custom_tracking_id"])
def inject_custom_metadata(ctx: Context, params: dict[str, Any]) -> Context:
    """Add a custom tracking field from the incoming request into the shape."""
    incoming_ctx = params.get("incoming_ctx")
    if incoming_ctx is not None:
        value = glom(incoming_ctx._body, "custom_tracking_id", default=None)
        if value is not None:
            assign(ctx._body, "custom_tracking_id", value)
    return ctx
```

```yaml
shape_hooks:
  - myproject.shaping.custom
```

**Parameterized hook** (accepts config-driven parameters via a Pydantic model):

```python
# myproject/shaping/tag.py
from typing import Any
from glom import assign
from pydantic import BaseModel
from ccproxy.pipeline.context import Context
from ccproxy.pipeline.hook import hook

class TagParams(BaseModel):
    key: str
    value: str

@hook(reads=["metadata"], writes=["metadata"], model=TagParams)
def add_tag(ctx: Context, params: dict[str, Any]) -> Context:
    """Set a metadata tag from config params."""
    path = f"metadata.{params['key']}"
    assign(ctx._body, path, params["value"])
    return ctx
```

```yaml
shape_hooks:
  - hook: myproject.shaping.tag
    params:
      key: "environment"
      value: "production"
```

The `model=` kwarg on `@hook` declares a Pydantic model for parameter validation. When `load_hooks()` processes a `{hook, params}` entry, it validates `params` against the model and rejects invalid configurations at load time.

To add a new provider, add an entry under `shaping.providers` with the appropriate `content_fields` for that provider's API schema. No Python code changes required.

---

## End-to-End Workflow

```bash
# Fresh install: bundled defaults are used automatically
just up

# Check the packaged request-only artifacts
ccproxy shapes audit

# Verification
# Run a request through the reverse proxy with the sentinel key, then:
ccproxy flows compare
# The diff shows the forwarded request carrying shape compliance headers
# alongside your actual message content

# Advanced development override; see "Manual Shaping When a Packaged Default Is Stale" above:
ccproxy run --capture -- claude -p "shape refresh"
ccproxy shapes save anthropic

# Remove user customizations and return to the bundled default:
rm -rf ~/.config/ccproxy/shapes/anthropic ~/.config/ccproxy/shapes/anthropic.mflow
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "No shape available for provider X" in logs | No bundled default and no advanced local override for that provider | Install a ccproxy release that packages that provider shape; for custom-provider development, write an explicit `.mflow` override with `ccproxy shapes save X --mflow` |
| "No shaping profile for provider X" in logs | Missing provider config | Add `shaping.providers.X` to ccproxy.yaml |
| Shape hook not firing (no "Applied shape" log) | Guard condition not met: flow lacks transform, or entered via WireGuard passthrough | Verify transform/redirect routing exists; check that the flow entered through the reverse proxy or had auth injected |
| System prompt missing shape's preamble | `merge_strategies` misconfigured | Ensure `system: prepend_shape` is set in the provider's `merge_strategies` config |
| 400 "too many cache_control breakpoints" | Shape system blocks carry `cache_control` that survives `prepend_shape` merge | Add the `strip` and `insert` caching hooks to `shape_hooks` (see Cache Breakpoint Hooks) |
| 400/403 from provider after shaping | Stale packaged or local shape | Update ccproxy to a release with refreshed packaged defaults. If no fixed release exists yet, follow the manual shaping guide above. |
| Auth headers leaking from shape | `strip_headers` misconfigured | Ensure `authorization` and `x-api-key` are in the provider's `strip_headers` list |
