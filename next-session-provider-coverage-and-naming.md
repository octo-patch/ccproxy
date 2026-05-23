# ccproxy — next session: provider coverage + naming + IR consistency

## Context

Where we are: just committed
`feat(lightllm): Phase F/H — subgraph composition + typed tool promotion` (38ead67) and
`chore(lightllm): drop dead PerplexityProConfig + registry; doc cleanup` (8e5527a). Branch is 24
ahead of `origin/dev`, awaiting Kyle’s push + `nh os switch ~/.config/nixos`. 1663 pytest + mypy +
ruff all green. Live matrix rows 1, 2, 11, 12 pass.

Strategic threads from background research + live conversation:

1. **Naming inconsistency.** `inbound`/`outbound` is already the canonical hook-axis.
   But `ListenerFormat` (inbound wire format) and `Provider.provider: str` (outbound wire dialect)
   name the same axis with different words.
   Three different terms for two concepts.
2. **`Context.extras` first-class API.** `raw_extras: dict[str, Any]` is the dynamic-keys escape
   hatch but hooks still reach into `ctx._body` via raw glom calls.
3. **Codex / OpenAI Responses API parity.** ccproxy serves Claude Code via Anthropic listener +
   shape replay; for Codex parity we need `/v1/responses` listener + OpenAI Responses upstream.
4. **Unified dep-derived topology.** HookDAG synthesizes edges from `reads`/`writes` declarations;
   the FSM layer uses explicit `g.edge_from(A).to(B)`. Could they share a single dep-derived idiom?
5. **pydantic-ai Model shim.** Could ccproxy reuse pydantic-ai’s `Model` classes as the outbound
   wire-building layer?

Two background agents investigated #3 (OpenAI Responses scope), and #4/#5 (Opus, dep-topology +
Model shim). Their conclusions plus the live conversation produced the plan below.

## Core references - Use these heavily

Two sources shape this plan.

### `~/dev/src/pydantic-ai/` (we already depend on it)

Pydantic-ai ships `Model` classes per provider that already do the wire ↔ IR translation we’d
otherwise have to write.
Pinned at `>=1.99` in `pyproject.toml`. We can patch its private methods to expose the outbound
payload + intake parsers (the Step 5 trajectory).
13 free providers including `OpenAIResponsesModel`.

### `~/dev/src/gproxy/`

Two distinct pieces of gproxy are useful to ccproxy:

**A. `sdk/gproxy-channel/src/channels/chatgpt/`** (~4217 Rust LOC, 12 files) — the canonical
reference for **ChatGPT Pro WebUI access** (chatgpt.com/backend-api/f/conversation).
The only known OSS implementation of the 2026-04+ Sentinel anti-bot flow.
Files:
- `channel.rs` (1093) — Channel trait impl, refresh, classify
- `sentinel.rs` (227) — `prepare` → FNV-1a PoW → `finalize` → token cache
- `pow.rs` (116) — hashcash solver (~32-bit FNV-1a + xorshift-multiply avalanche)
- `prepare_p.rs` (292) — 25-slot browser fingerprint config + `gAAAAAC...` envelope
- `session.rs` (210) — Cloudflare `__cf_bm` warmup, header bundle, turn-context cache
- `request_builder.rs` (536) — OpenAI Chat body → chatgpt.com `/f/conversation` body
  (single-user-turn history flattening + `system_hints` mapping + `thinking_effort` mapping)
- `sse_v1.rs` (346) — JSON-Patch SSE delta decoder (5 event shapes)
- `sse_to_openai.rs` (357) — delta stream → `chat.completion.chunk` reshape
- `image.rs` + `image_edit.rs` (880) — DALL-E flows
- `models.rs` (131) + `models.json` (16 ids) — local model catalog

Architecturally analogous to ccproxy’s existing **Perplexity Pro** integration (`lightllm/pplx.py` +
intake FSM + the `pplx_*` outbound hooks).
Same shape: WebUI session-cookie auth, browser fingerprint impersonation via curl-cffi, custom SSE
format, custom payload builder, custom token-refresh flow.
This is what Step 6 below is about.

**B. `sdk/gproxy-protocol/src/openai/create_response/`** (~955 Rust LOC, 4 files) —
canonical type definitions for the **OpenAI Responses public API** wire format.
Plus `src/transform/*/openai_response/` for every cross-protocol transform bidirectionally.
Useful as reference for Step 4 (Codex parity) — but secondary to pydantic-ai’s
`OpenAIResponsesModel` which already covers most of what we need.

Rust → Python port is mechanical: `Option<T>` + serde tags map to Pydantic v2 `Field(default=None)`;
discriminated unions map to pydantic `Discriminator`. ~1.5-2x line expansion.

**The mental model:**
- pydantic-ai is the *outbound + intake shim* for upstream wire conversion (Step 5).
- gproxy-protocol is the *wire-format spec* for the OpenAI Responses listener side (Step 4) —
  reference for inbound parse + render where pydantic-ai’s client-only nature doesn’t help.
- gproxy’s chatgpt channel is the *reference implementation* for ChatGPT Pro WebUI access
  (Step 6) — a sibling upstream provider to ccproxy’s existing Perplexity Pro.

## Strategic trajectory — maximum pydantic-ai reuse

The through-line for this session and the ones after: **maximize reuse of pydantic-ai’s per-provider
work, including private APIs, even if it requires monkey-patches**. The tight coupling between
pydantic-ai’s wire-conversion code and the vendor SDK calls IS the value — pydantic-ai burns the
maintenance budget tracking every vendor’s wire shape so we don’t have to.

Per-provider, pydantic-ai’s `Model` classes expose ~10 hookable surfaces (4 high-value private + 4
public, plus capability profile + native-tool set).
We can hook all of them via the same monkey-patch pattern we already use for `_subgraph_patch.py`:

| # | pydantic-ai surface | Visibility | What it does | What ccproxy reuses it for |
| --- | --- | --- | --- | --- |
| 1 | `_messages_create` / `_completions_create` / `_generate_content` | private | Builds vendor SDK kwargs + calls `self.client.*.create(**kwargs)` | Outbound payload capture (the capture-dict trick) |
| 2 | `_map_messages` | private | IR `list[ModelMessage]` → vendor wire messages | Direct call alternative when signatures are stable |
| 3 | `_get_tools` / `_build_tools` | private | IR tools → vendor tool schemas (incl. Google `additionalProperties` strip, OpenAI tool-choice handling) | Direct call for the tools half of the payload |
| 4 | `_get_betas_and_extra_headers` | private | Vendor-specific outbound headers (Anthropic betas, OpenAI beta flags) | Direct call for outbound HTTP headers |
| 5 | `_process_response` | private | Vendor response object → `ModelResponse` IR | Buffered intake (replaces `transform_buffered_response_sync`) |
| 6 | `_process_streamed_response` | private | Vendor SSE async iterator → IR events | Streaming intake (replaces `*_intake.py` FSMs via `SSEPipeline` wrap) |
| 7 | `customize_request_parameters` | public | JSON schema transforms on tool defs per vendor quirk | Direct call during `_parse_tools` |
| 8 | `prepare_request` | public | Merge settings + apply customizations → prepared bundle | Direct call to normalize inputs |
| 9 | `Model.profile` | public | `ModelProfile` capability flags (supports_tools, supports_thinking, etc.) | Surface in `ccproxy status` per provider |
| 10 | `Model.supported_native_tools` | public classmethod | Set of `AbstractNativeTool` subclasses this Model supports | Drives `_tool_kinds.py` mapping (replaces our hand-maintained dict) |

What pydantic-ai does NOT cover (stays ccproxy’s responsibility per LISTENER format, not per
provider):
- **Inbound request parsing** (client wire → IR) — pydantic-ai is a client library; doesn’t receive
  requests.
- **Response wire rendering** (IR → listener SSE) — same reason.

Per LISTENER format we keep ~800 LOC (inbound parser + outbound render).
Per OUTBOUND-only provider we drop from ~700-900 LOC to ~120 LOC (shim wrappers + provider config).
Per LISTENER-format new provider we drop from ~1500-1800 LOC to ~800 LOC. The savings compound:
every new provider added via the shim costs ~120 LOC instead of ~800.

**Shipped pydantic-ai providers we’d get for free (or near-free):** AnthropicModel, OpenAIChatModel,
**OpenAIResponsesModel**, BedrockConverseModel, CerebrasModel, CohereModel, GoogleModel, GroqModel,
HuggingFaceModel, MistralModel, OpenRouterModel, XaiGrokModel, OllamaModel.
That’s 13 providers including OpenAI Responses (which is the core Codex parity target — and which
would cost ~1400 LOC of fresh code without the shim).

## Recommended Approach

Seven steps, ordered by combined ROI + dependencies.
Steps 1-3 are low-risk ergonomic wins.
Step 4 (OpenAI Responses listener) is the main public-API deliverable.
Step 5 is the **gating experiment** for the strategic trajectory — its outcome determines whether
Step 4 costs 1900 LOC of fresh code or ~900 LOC of shim-backed code.
Step 6 (ChatGPT Pro WebUI upstream) is the killer-feature payoff: it builds on Step 4’s listener —
same `/v1/responses` endpoint sentinel-routes between OpenAI public API and chatgpt.com WebUI, so
the user can negotiate between paid API tokens and ChatGPT Pro subscription seat per request.
Step 7 (dep-derived FSM topology) is a smaller intellectual experiment, lowest priority.

### Step 1 — Naming pass

Drop the redundant `Provider.provider: str` field name.
A `Provider` config object IS its wire dialect — there’s no second axis to disambiguate against, so
calling the field `provider` (or `outbound_format`) is over-explaining.
Use `type` to match the existing `AuthSource.type` discriminator pattern (`type: command` /
`type: file` / `type: anthropic_oauth`).

Also rename the inbound-format enum since `inbound`/`outbound` is already our canonical axis (it’s
how hooks are keyed).

| Current | New |
| --- | --- |
| `lightllm.parsed.ListenerFormat` (enum) | `lightllm.parsed.InboundFormat` |
| `ListenerFormat.ANTHROPIC_MESSAGES` etc. | `InboundFormat.ANTHROPIC_MESSAGES` (values unchanged) |
| `Provider.provider: str` (wire dialect field) | `Provider.type: str` |
| `dispatch_dump_sync(req, provider=…)` | `dispatch_dump_sync(req, provider_type=…)` |
| `dispatch_intake(upstream_provider=…)` | `dispatch_intake(provider_type=…)` |
| `Context._listener_format` | `Context._inbound_format` |
| Var names: `listener_format`, `upstream_provider` | `inbound_format`, `provider_type` |
| `_ANTHROPIC_COMPATIBLE`, `_GOOGLE_COMPATIBLE` (frozenset in `graph/__init__.py`) | unchanged (they’re sets of provider type values) |

YAML reads cleaner:

```yaml
providers:
  anthropic:
    auth: { type: anthropic_oauth, ... }
    host: api.anthropic.com
    path: /v1/messages
    type: anthropic    # was: provider: anthropic
  codex:
    auth: { type: file, file: ~/.opnix/secrets/openai-api-key }
    host: api.openai.com
    path: /v1/responses
    type: openai_responses
```

The outer key (`anthropic`, `codex`) is still the routing/sentinel name; the inner `type` field is
the wire dialect. Both `AuthSource.type` and `Provider.type` follow the same discriminator
convention, which is a nice symmetry.

Touch list (~25 files): `parsed.py`, `pipeline/context.py`, `pipeline/keyspace.py`, `config.py`,
`inspector/addon.py`, `inspector/routes/transform.py`, `inspector/routes/models.py`,
`lightllm/graph/__init__.py`, `lightllm/graph/buffered.py`, all `lightllm/graph/*_intake.py` +
`*_render.py` (only docstrings/comments), all `tests/test_lightllm_graph_*` + `tests/test_config.py`
\+ `tests/test_inspector_*`, `nix/defaults.nix` (if any string refs), `AGENTS.md`,
`docs/lightllm.md`, `docs/configuration.md`.

Risk: rename-pass-induced typo.
Mitigation: rely on mypy + ruff + the existing test suite; no behavior change.

Cost: ~1 hr mechanical refactor.

### Step 2 — Promote `raw_extras` to `Context.extras` glom-pathed accessor

Today hooks do this:

```python
from glom import glom, assign, delete
session_id = glom(ctx._body, "metadata.user_id", default=None)
assign(ctx._body, "pplx.attachments", [...], missing=dict)
```

Proposed: a small wrapper exposing the same glom verbs on `ctx.extras`:

```python
ctx.extras.get("metadata.user_id", default=None)
ctx.extras.set("pplx.attachments", [...])
ctx.extras.delete("tool_choice")
ctx.extras.has("metadata.user_id")  # bool
```

Implementation: ~50 LOC wrapper class in `pipeline/context.py`. `ctx.extras` returns a façade around
`ctx._cached_raw_extras` (or `ctx._body` for fields the IR doesn’t model — decide where the boundary
is).

Migration: optional.
Existing `glom(ctx._body, ...)` calls keep working; new code goes through `ctx.extras`. Migrate
hooks one at a time when touched.

Files: `src/ccproxy/pipeline/context.py` (+50 LOC), `docs/lightllm.md` (+section), `AGENTS.md`
(update three-layer access model note).

Cost: ~1 hr including doc + 2-3 unit tests.

### Step 3 — `HookDAG.render() -> str` mermaid output

Small ergonomic — matches the FSM mermaid render so hook + FSM graphs use the same visual language.

```python
class HookDAG:
    def render(self, *, title: str = "hook_dag", direction: str = "LR") -> str:
        """Render the topo-sorted hook DAG as mermaid stateDiagram-v2."""
        ...
```

Walks `self.execution_order`, emits one state node per hook, edges between hooks where one writes a
key the next reads. Uses the same `---\ntitle: ...\n---\nstateDiagram-v2\n direction LR\n ...`
envelope the FSM graphs use.

Wire into `ccproxy status` (already renders a hook pipeline visualization via rich — give it a
`--mermaid` flag) and the visualization snippet in `next.md` / `docs/lightllm.md`.

Files: `src/ccproxy/pipeline/dag.py` (+30 LOC), `src/ccproxy/pipeline/render.py` (existing — add
mermaid output mode).

Cost: ~30 min + one unit test asserting the rendered output for a fixed hook set.

### Step 4 — OpenAI Responses API support (Codex parity)

The main public-API deliverable.
Strategy is hybrid:

- **Wire-format types** — use OpenAI’s own SDK at
  `.venv/lib/python3.13/site-packages/openai/types/responses/` (TypedDicts) as the primary spec,
  falling back to **gproxy-protocol/src/openai/create_response/** as a secondary reference for
  any edge case the SDK’s TypedDicts don’t explain well (gproxy-protocol’s Rust types are more
  discriminated-union-aware than TypedDicts).
- **Listener-side inbound parse + render** — write `OpenAIResponsesAdapter.load_messages` + the
  render FSM by hand. ~700 LOC.
- **Outbound build + response intake** — delegate to pydantic-ai’s `OpenAIResponsesModel` via the
  Step 5 shim. Pydantic-ai already handles all 48 SSE event types and the request-payload assembly.

**Size depends on Step 5’s outcome:**
- **If Step 5 pilot succeeds** → outbound + intake come from the shim.
  Total Step 4 = ~900 LOC (listener parser + render FSM + shim glue).
- **If Step 5 fails or hasn’t run** → write fresh intake/render FSMs.
  Use gproxy-protocol’s `stream.rs` as the canonical 48-event spec.
  Total Step 4 = ~1900 LOC.

Run Step 5 BEFORE committing to Phase 4B implementation.
Phase 4A (listener-side parsing) is independent of Step 5; ship it either way.

**On gproxy-protocol as a port target:** gproxy-protocol gives us cleaner Pydantic-like types than
the OpenAI SDK’s loose TypedDicts, but it’s not strictly necessary for Step 4 — the SDK’s types
cover the wire shape, just less ergonomically.
Port gproxy-protocol’s files ONLY if we hit edge cases the SDK doesn’t disambiguate (discriminated
input items, server-side-tool result shapes, etc.). The cross-protocol transforms in gproxy-protocol
(`transform/*/openai_response/`) ARE the genuinely novel asset — those become Phase 4C when we want
cross-format routing.

From Sonnet agent A’s full scoping report (fresh-code estimate, used as the worst case):

**Background.** Codex CLI is a precompiled Rust agent binary that talks to OpenAI’s `/v1/responses`
endpoint. The Responses API is a NEW OpenAI API family (not just a Chat Completions version bump)
introducing:
- `input[]` heterogeneous items (message / function_call / reasoning / web_search_call /
  code_interpreter_call / mcp_call / apply_patch / shell / computer_use / file_search) vs Chat
  Completions’ role-based `messages[]`.
- Server-side conversation state via `previous_response_id` or `conversation: {id}`.
- Native `reasoning: {effort: low|medium|high}` for o-series / gpt-5 thinking budget.
- Built-in server-side tools unified: `web_search`, `file_search`, `code_interpreter`,
  `computer_use`, MCP server integrations.
- `prompt_cache_key` + `prompt_cache_retention: "in-memory" | "24h"` (OpenAI’s caching, different
  semantics from Anthropic’s block-level `cache_control`).
- `background: bool` mode (poll-based async response generation).
- 48 streaming event types vs Chat Completions’ ~15.

**Phase 4A — listener MVP** (~400 LOC):
- New `InboundFormat.OPENAI_RESPONSES` value.
- `_select_listener_format` in `pipeline/context.py` recognizes `/v1/responses` path.
- New `src/ccproxy/lightllm/adapters/openai_responses.py` with
  `load_messages(body: dict) -> list[ModelMessage]` parsing Responses’ `input[]` heterogeneous items
  into pydantic-ai IR. Uses the OpenAI SDK’s TypedDicts from `openai/types/responses/` as the
  wire-shape contract.
  `render` raises `NotImplementedError` for now.
- New `src/ccproxy/lightllm/adapters/_openai_responses_envelope.py` for `input[]` item
  discrimination + content-part parsing.
- Smoke test: `POST /v1/responses` with simple text input → route to Anthropic upstream via sentinel
  → return buffered Anthropic response.
  No streaming yet, no Responses upstream yet.

If the SDK TypedDicts prove ambiguous for any input item shape, port the specific type from
gproxy-protocol with a docstring attribution:

```python
class ResponseInputItem(BaseModel):
    """One item in the Responses ``input[]`` array.

    Ported from gproxy-protocol/src/openai/create_response/types.rs:N-M
    (commit f85f4e22de8556113684a6ee7ac42e81fc09f624) because the
    OpenAI SDK's TypedDict union doesn't preserve the discriminator we need.
    """
    ...
```

**Phase 4B — upstream support** (~1000 LOC with Step 5 shim, ~1900 LOC without):
- Port `gproxy-protocol/src/openai/create_response/{response,stream}.rs`:
  - `wire/responses/response.py` (~150 LOC) — `Response` wrapper, `ResponseError`, `ResponseUsage`,
    `IncompleteDetails`.
  - `wire/responses/stream.py` (~700 LOC) — all 48 SSE event types as discriminated union
    (`response.created`, `response.queued`, `response.in_progress`, `response.output_item.added`,
    `response.content_part.added`, `response.text.delta`, `response.text.done`,
    `response.reasoning.text.delta`, `response.function_call_arguments.delta`,
    `response.web_search_call.searching`, `response.code_interpreter_call.code.delta`,
    `response.mcp_call.*`, `response.computer_call.*`, `response.completed`, `response.failed`,
    `response.incomplete`, etc.).
- **If Step 5 shim landed**:
  - Bidirectional `OpenAIResponsesAdapter.render` — `list[ModelMessage]` → Responses `input[]` (uses
    ported wire models).
    ~200 LOC.
  - **Outbound build delegates** to pydantic-ai’s `OpenAIResponsesModel` via
    `get_outbound_payload(provider_type="openai_responses", ...)`. ~50 LOC shim glue.
  - **Streaming intake delegates** to pydantic-ai’s
    `OpenAIResponsesModel._process_streamed_response` via the shim’s `get_streaming_intake(...)`.
    ~50 LOC wrapper. No fresh 48-event FSM needed — pydantic-ai already handles it.
  - **Buffered intake** delegates similarly.
    ~50 LOC.
  - New `src/ccproxy/lightllm/graph/openai_responses_render.py` — listener-side IR → Responses SSE
    emitter, using ported `stream.py` types.
    ~400 LOC.
- **If Step 5 shim did NOT land**:
  - Write fresh 48-event FSM (`openai_responses_intake.py` ~700 LOC) using the ported `stream.py` as
    the per-event-type spec.
  - Write fresh outbound builder using the ported `request.py` types.
- Dispatch branches in `lightllm/graph/__init__.py:dispatch_dump_sync`
  (`provider_type == "openai_responses"`), `dispatch_intake`, `dispatch_render`.
- Provider config entry pattern:
  ```yaml
  providers:
    codex:
      auth: { type: file, file: ~/.opnix/secrets/openai-api-key }
      host: api.openai.com
      path: /v1/responses
      type: openai_responses
  ```
- Sentinel `sk-ant-oat-ccproxy-codex` routes via `forward_oauth` → Responses upstream.

**Phase 4C — cross-format transforms** (defer to follow-up session, but the SPEC is ready):

gproxy-protocol implements every cross-protocol transform between Responses and the other dialects
bidirectionally — and we already host the Rust source as a reference.
Subdirectories under `gproxy-protocol/src/transform/`:
- `openai/{generate_content,stream_generate_content}/openai_response/` — OpenAI Chat ↔ Responses
  (both directions).
- `claude/{generate_content,stream_generate_content}/openai_response/` — Claude ↔ Responses (both
  directions).
- `gemini/{generate_content,stream_generate_content}/openai_response/` — Gemini ↔ Responses (both
  directions).

Each subdirectory has `request.rs` + `response.rs` (sometimes `utils.rs`) implementing the `TryFrom`
mappings. Porting is bespoke per pair (~300-500 LOC each) but the algorithmic content is already
worked out — we’re translating logic, not designing it.

For MVP: mark Anthropic ↔ Responses + Chat ↔ Responses as initially-unsupported cross-format
transforms in `lightllm/graph/__init__.py:dispatch_intake/render`. When we want cross-format, port
one direction at a time.
The hardest case (Anthropic `thinking` ↔ Responses `reasoning`) is solved in gproxy-protocol; we
don’t need to invent the mapping.

Estimated ~3000-5000 LOC for full bidirectional coverage of all 3 pairs, ported one PR at a time.

**Shape replay for Codex** — skip.
No documented identity header requirements analogous to Anthropic’s `x-anthropic-billing-header`.
Revisit only if Codex requests start failing with 401/403; capture with Wireshark +
`ccproxy flows compare` then.

**Total estimate (with Step 5 shim):** Phase 4A is 2-3 days (mostly the port + listener parser);
Phase 4B is 2 days (mostly the render FSM + shim glue); tests add another 1 day.
~9 new files, four updated routing modules.

**Total estimate (without Step 5 shim):** Phase 4A unchanged; Phase 4B is 3-4 days (fresh intake
FSM); tests add another 1 day.
~10 new files.

Files (new):
- `src/ccproxy/lightllm/wire/__init__.py` (new package — wire-format type definitions ported from
  gproxy-protocol)
- `src/ccproxy/lightllm/wire/responses/__init__.py`
- `src/ccproxy/lightllm/wire/responses/request.py` (~200 LOC — port of
  `gproxy-protocol/src/openai/create_response/request.rs`)
- `src/ccproxy/lightllm/wire/responses/response.py` (~150 LOC — port of `response.rs`)
- `src/ccproxy/lightllm/wire/responses/stream.py` (~700 LOC — port of `stream.rs`, all 48 SSE event
  types)
- `src/ccproxy/lightllm/wire/responses/types.py` (~400 LOC — port of `types.rs`)
- `src/ccproxy/lightllm/adapters/openai_responses.py` (~300 LOC)
- `src/ccproxy/lightllm/adapters/_openai_responses_envelope.py` (~100 LOC)
- `src/ccproxy/lightllm/graph/openai_responses_intake.py` (~700 LOC — ONLY if Step 5 shim didn’t
  land; otherwise ~50 LOC shim wrapper)
- `src/ccproxy/lightllm/graph/openai_responses_render.py` (~400 LOC)
- `tests/test_wire_responses_models.py` (~150 LOC — round-trip serialization tests for the ported
  models)
- `tests/test_lightllm_graph_openai_responses_load.py` (~100 LOC)
- `tests/test_lightllm_graph_openai_responses_dump.py` (~100 LOC)
- `tests/test_lightllm_graph_intake_openai_responses.py` (~150 LOC)
- `tests/test_lightllm_graph_render_openai_responses.py` (~100 LOC)

Files (modified): `parsed.py` (enum value), `pipeline/context.py` (`_select_listener_format`),
`lightllm/graph/__init__.py` (3 dispatch branches), `lightllm/graph/buffered.py` (response
synthesis), `nix/defaults.nix` (no immediate change — config users add their own provider entry),
`docs/lightllm.md` (new `wire/` package documentation + cite gproxy-protocol attribution).

### Step 5 — pydantic-ai shim layer — 4-direction Mistral pilot

The strategic trajectory’s gating experiment.
Test all four reuse directions on one provider; if all four survive a pydantic-ai version bump,
commit to the aggressive-shim migration.

**Pilot target: Mistral.** Reasons:
- pydantic-ai has `pydantic_ai.models.mistral.MistralModel`; ccproxy doesn’t have its own Mistral
  adapter (zero migration cost — we’re not displacing anything).
- Mistral’s wire is OpenAI-compatible, so failure mode is easy to inspect (captured payload should
  look like OpenAI Chat; intake should yield IR events compatible with our existing
  OpenAIChatStreamedResponse parser).
- If all four directions pass, we ship Mistral as a sentinel-routable provider in ~150 LOC of
  ccproxy code total (the four shims + a dispatch branch + a provider entry).
- If any direction fails, fall back: ship Mistral as a `type: openai` provider entry routing through
  our existing `OpenAIChatAdapter` (Mistral is OpenAI wire-compatible).

**Shim module layout** (new files):

```
src/ccproxy/lightllm/
└── _pydantic_ai_shim/
    ├── __init__.py            # Public API: get_outbound_payload, get_buffered_intake,
    │                          # get_streaming_intake, get_capability_profile
    ├── _payload_patch.py      # Installs Model.build_request_payload via capture-dict trick
    ├── _intake_patch.py       # Wraps Model._process_response / _process_streamed_response
    │                          # into mitmproxy's chunk-callable / buffered shape
    ├── _profile.py            # Maps pydantic-ai's ModelProfile → ccproxy's capability surface
    └── _dispatch.py           # Maps provider_type string → pydantic-ai Model class + per-Model
                               #   client-method to patch (e.g. "anthropic" → AnthropicModel,
                               #   "client.beta.messages.create"; "openai_chat" → OpenAIChatModel,
                               #   "client.chat.completions.create"; etc.)
```

Public API the rest of ccproxy uses:

```python
# src/ccproxy/lightllm/_pydantic_ai_shim/__init__.py
def get_outbound_payload(
    provider_type: str, model: str, req: LLMRenderInput,
) -> dict[str, Any]:
    """Build the upstream wire-format payload via pydantic-ai's Model."""

def get_buffered_intake(
    provider_type: str, model: str,
) -> Callable[[bytes], ModelResponse]:
    """Return a callable that takes raw vendor response bytes and returns the IR."""

def get_streaming_intake(
    provider_type: str, model: str, request_params: ModelRequestParameters,
) -> StreamingIntakeFSM:
    """Return a feed(bytes) FSM wrapping pydantic-ai's _process_streamed_response."""

def get_capability_profile(provider_type: str, model: str) -> ModelProfile:
    """Return the ModelProfile pydantic-ai ships for the model."""
```

**The four directions tested independently on Mistral:**

1. **Outbound build** via capture-dict on `MistralModel._completions_create`:
   ```python
   payload = await get_outbound_payload(
       provider_type="mistral",
       model="mistral-large-latest",
       req=ctx,
   )
   wire_bytes = json.dumps(payload).encode()  # this is what mitmproxy forwards
   ```

   Test: assert payload structure matches Mistral’s OpenAI-compat wire (model + messages + tools +
   max_tokens).

2. **Buffered intake** via `MistralModel._process_response`:
   ```python
   parse = get_buffered_intake(provider_type="mistral", model="mistral-large-latest")
   ir_response: ModelResponse = parse(upstream_bytes)
   ```

   Test: feed a captured non-streaming Mistral response → assert IR has expected `TextPart` + usage.

3. **Streaming intake** via `MistralModel._process_streamed_response` wrapped in `SSEPipeline`:
   ```python
   fsm = get_streaming_intake(provider_type="mistral", model=..., request_params=...)
   for chunk in sse_chunks:
       for event in fsm.feed(chunk):
           ...  # IR events
   ```

   Test: feed captured Mistral SSE in chunked form → assert IR event sequence matches direct
   invocation of `_process_streamed_response` (we’re a faithful wrapper, not re-implementing).

4. **Capability profile** surfaced in `ccproxy status`:
   ```python
   profile = get_capability_profile("mistral", "mistral-large-latest")
   # ModelProfile{supports_tools=True, supports_thinking=False, ...}
   ```

   Test: assert the profile dict matches what pydantic-ai exposes; verify status display formatting.

**Total Mistral shim code:** ~50 LOC each direction × 4 = ~200 LOC + ~50 LOC of
`_pydantic_ai_shim/_dispatch.py` glue + ~30 LOC of provider config + dispatch branch = **~280 LOC
for Mistral end-to-end**. Compare to ~750 LOC for a fresh upstream-only adapter (per the current
per-provider cost).

**Version-bump CI guard.** Add a `tests/test_pydantic_ai_shim_pinning.py` that:
- Snapshots Mistral’s outbound payload bytes for a fixed IR input.
- Snapshots Mistral’s IR event sequence for a fixed SSE stream.
- Tests run against `pydantic-ai==1.99.x` (currently pinned floor).
- A separate matrix test (or pre-merge hook) re-runs against pydantic-ai’s latest available version
  on PyPI; if the snapshot diff is non-trivial, fail loudly so we know upstream changed something
  that affects us.

**Trajectory if Mistral pilot passes all 4 directions across one pydantic-ai version bump:**

1. Migrate existing **outbound-only** providers first (lowest blast radius): Google → drops
   `lightllm/adapters/google.py` (279 LOC) + `lightllm/graph/google_intake.py` (493 LOC) = ~770 LOC,
   replaced by ~80 LOC of shim glue.
2. Migrate **listener-role** providers (Anthropic, OpenAIChat) — the outbound + intake halves are
   replaced; the inbound parser + render FSM stay (those are listener-side, pydantic-ai doesn’t
   cover them). Net ~1000 LOC drop per provider in exchange for ~100 LOC of shim glue.
3. **Add OpenAI Responses** via the shim directly (this changes Step 4’s Phase 4B math from ~1100
   LOC to ~250 LOC).
4. Add Bedrock, Cohere, Groq, OpenRouter, Xai, Cerebras, HuggingFace, Ollama as free coverage — each
   is ~30-50 LOC of provider entry + a dispatch row.
5. Contribute `Model.build_request_payload` + `Model.parse_response_bytes` upstream as a PR so the
   capture-dict patch can be deleted.

**Trajectory if Mistral pilot fails any direction:**
- Document which direction(s) failed and why (likely candidates: private method signature changed,
  vendor SDK client structure too coupled to retry/error-handling to capture-dict cleanly,
  async-iterator shape doesn’t compose with `SSEPipeline`).
- Lock in the current per-provider adapter strategy.
- Ship Mistral via the existing `type: openai`-compatible route (zero new code).
- File the failure mode upstream as a pydantic-ai issue requesting a “build payload without send”
  API.

**Files (Mistral pilot):**
- `src/ccproxy/lightllm/_pydantic_ai_shim/__init__.py` (~80 LOC)
- `src/ccproxy/lightllm/_pydantic_ai_shim/_payload_patch.py` (~80 LOC — patches
  `MistralModel._completions_create`)
- `src/ccproxy/lightllm/_pydantic_ai_shim/_intake_patch.py` (~100 LOC — buffered + streaming
  wrappers)
- `src/ccproxy/lightllm/_pydantic_ai_shim/_profile.py` (~30 LOC)
- `src/ccproxy/lightllm/_pydantic_ai_shim/_dispatch.py` (~50 LOC — Mistral entry only at pilot
  stage)
- Dispatch branches in `lightllm/graph/__init__.py:dispatch_dump_sync` + `dispatch_intake` (~20 LOC)
- `nix/defaults.nix` provider entry for `mistral` (~10 LOC YAML)
- `tests/test_pydantic_ai_shim_mistral.py` (~150 LOC — 4-direction test suite + version-bump
  snapshot)

**Decision deferred to data.** Run the pilot; let the outcome dictate the trajectory.

### Step 6 — ChatGPT Pro WebUI as a Responses upstream (Codex ↔ GPT Pro negotiation)

The killer feature this whole plan enables: **same `/v1/responses` listener routes to EITHER OpenAI
public API OR chatgpt.com WebUI based on the sentinel key**. So a Codex CLI session can negotiate
between paid API tokens (for breadth + standard rate limits) and ChatGPT Pro subscription seat (for
premium models like gpt-5-pro / o3-pro that aren’t in the public API + flat monthly cost).

Requires Step 4 (the Responses listener) to exist first.
This step is the port of gproxy’s `chatgpt` channel into a sibling upstream provider, analogous
to ccproxy’s existing Perplexity Pro integration.
Realistic scope: **probably a follow-up session of its own**. Phase 6A (Sentinel + PoW + fingerprint
port, ~600 LOC) could fit at the end of this session as a feasibility pilot; Phases 6B-6E are
next-next-session work.

**Architecture parallel to Perplexity Pro:**

| Concern | Perplexity Pro (existing) | ChatGPT Pro (new) |
| --- | --- | --- |
| Inbound listener | OpenAI Chat (`/v1/chat/completions`) | OpenAI Responses (`/v1/responses`) |
| Auth | `__Secure-next-auth.session-token` cookie | chatgpt.com JWT + sentinel chat-requirements token + PoW token |
| Browser fingerprint | curl-cffi `chrome131` | curl-cffi `chrome136` (or `wreq` `Emulation::Chrome136` equivalent) |
| Pre-flight | `GET /search/new?q=...` warmup | Cloudflare warmup (`GET /`, `GET /backend-api/me`) + Sentinel `prepare`+PoW+`finalize` dance |
| Outbound wire | Perplexity 28-field `/rest/sse/perplexity_ask` body | chatgpt.com `/backend-api/f/conversation` body |
| Upstream SSE format | Perplexity’s custom JSON-per-event with `blocks`+`diff_block` patches | chatgpt.com’s SSE-v1 JSON-Patch delta encoding |
| Outbound header stamping | `pplx_stamp_headers` hook | `chatgpt_stamp_headers` hook (Cookie + 20+ sec-ch-ua-* + oai-* headers) |
| Token refresh | `uv tool run get-perplexity-session-token` (manual OTP) | `refresh_credential` lifecycle: re-run Sentinel flow + decode new JWT exp |

**Phase 6A — port the Sentinel + PoW + fingerprint subsystem** (~600 LOC):
- `src/ccproxy/lightllm/chatgpt_pro/sentinel.py` — port
  `sdk/gproxy-channel/src/channels/chatgpt/sentinel.rs` (227 LOC Rust).
  The `prepare → PoW → finalize → cache JWT exp` flow.
- `src/ccproxy/lightllm/chatgpt_pro/pow.py` — port `pow.rs` (116 LOC). FNV-1a + xorshift-multiply
  avalanche hashcash solver.
  Trivially ported.
- `src/ccproxy/lightllm/chatgpt_pro/prepare_p.py` — port `prepare_p.rs` (292 LOC). 25-slot browser
  fingerprint config + `gAAAAAC` envelope.
  Includes the from-scratch `Date.toString()` formatter — port Howard Hinnant’s algorithm directly.
- Unit tests asserting the JS-reference hash matches (gproxy ships known-good fixtures).

**Phase 6B — port the session, request builder, and SSE-v1 decoder** (~1200 LOC):
- `src/ccproxy/lightllm/chatgpt_pro/session.py` — port `session.rs` (210 LOC). Cloudflare warmup
  with 25-minute Mutex-cached `__cf_bm` cookie; standard header bundle.
- `src/ccproxy/lightllm/chatgpt_pro/request_builder.py` — port `request_builder.rs` (536 LOC).
  **Adaptation:** gproxy’s builder maps OpenAI Chat → `/f/conversation`; we map OpenAI Responses →
  `/f/conversation`. Reuse the history-flattening logic; remap `reasoning.effort` →
  `thinking_effort`; remap `tools: [web_search]` → `system_hints: ["search"]`; handle
  `previous_response_id` via the flattened-history path.
- `src/ccproxy/lightllm/chatgpt_pro/sse_v1.py` — port `sse_v1.rs` (346 LOC). Byte-streaming
  JSON-Patch SSE delta decoder, 5 event shapes (`delta_encoding`, typed, single-patch, batch,
  shorthand-batch). Direct port of `PatchKind` enum.

**Phase 6C — Responses-format intake FSM** (~700 LOC):
- `src/ccproxy/lightllm/graph/chatgpt_pro_intake.py` — pydantic-graph FSM that consumes
  `sse_v1.py`’s patch events and emits Responses-format `ModelResponseStreamEvent`s. **Adaptation:**
  gproxy’s `sse_to_openai.py` (357 LOC) maps to OpenAI **Chat Completions** chunk events; we
  re-target it to emit OpenAI **Responses** events (`response.text.delta`,
  `response.reasoning.text.delta`, `response.function_call_arguments.delta`, etc.). The channel-map
  state tracking (channel index → assistant message id) carries over directly.

**Phase 6D — outbound hook + provider config wiring** (~300 LOC):
- `src/ccproxy/hooks/chatgpt_stamp_headers.py` — outbound hook stamping the full chatgpt.com header
  bundle (Cookie + sec-ch-ua-* + oai-* + sentinel + PoW tokens + turn-trace-id).
  Runs after `forward_oauth` and before the request goes out, symmetric to `pplx_stamp_headers`.
- `src/ccproxy/lightllm/adapters/chatgpt_pro.py` — adapter with `render(req)` invoking the request
  builder. ~150 LOC.
- Dispatch branches in `lightllm/graph/__init__.py`:
  - `dispatch_dump_sync(req, provider_type="chatgpt_pro")` → `ChatGptProAdapter.render(req)`.
  - `dispatch_intake(provider_type="chatgpt_pro")` → `ChatGptProIntakeFSM`.
- Provider config entry pattern:
  ```yaml
  providers:
    chatgpt_pro:
      auth: { type: file, file: ~/.opnix/secrets/chatgpt-access-token }
      host: chatgpt.com
      path: /backend-api/f/conversation
      type: chatgpt_pro
      fingerprint_profile: chrome136
  ```
- Sentinel `sk-ant-oat-ccproxy-chatgpt_pro` routes via `forward_oauth` → ChatGPT Pro WebUI upstream.

**Phase 6E — Codex ↔ GPT Pro routing negotiation** (~200 LOC):
- The simplest negotiation surface: the client picks via sentinel key.
  `sk-ant-oat-ccproxy-codex` → OpenAI public API. `sk-ant-oat-ccproxy-chatgpt_pro` → WebUI. Already
  works at the `forward_oauth` layer; just needs both providers configured.
- Optional richer negotiation: per-model routing rules (`gpt-5-pro` always → `chatgpt_pro`;
  everything else → `codex`) via the existing `inspector.transforms` regex matcher with
  `match_model`.
- Optional capacity fallback: if WebUI returns Cloudflare `cf-mitigated` (warmup failed) or Sentinel
  rejected the PoW, fall back to public API. Same shape as the `GeminiAddon` capacity fallback —
  write `ChatGptProAddon` that detects the failure mode and rotates.

**Total Phase 6 estimate:** ~3000 LOC across all sub-phases + ~500 LOC tests.
4-6 days. Ship Phase 6A+7B+7C+7D incrementally as four PRs (each independently testable).
Phase 6E is a follow-up enhancement after the core upstream works.

**Risks specific to Phase 6:**
- **Sentinel flow stability.** OpenAI tightens the chatgpt.com anti-bot logic periodically.
  The 25-slot fingerprint shape and PoW algorithm have changed before.
  Mitigation: keep gproxy as the upstream reference; when it updates, port the diff.
  Set up a CI job that periodically runs the Sentinel `prepare`→`finalize` against the real
  chatgpt.com to detect breakage.
- **Cloudflare TLS fingerprint mismatch.** ccproxy’s existing fingerprint sidecar uses `chrome131`;
  gproxy uses `chrome136`. Either upgrade ccproxy’s default to a newer Chrome profile or override
  per-provider via `fingerprint_profile: chrome136`.
- **Single-turn flattening lossy.** chatgpt.com `/f/conversation` only accepts ONE user turn per
  call; gproxy concatenates history into the prompt.
  For Codex’s multi-turn agent loops this is potentially noisy — investigate `parent_message_id`
  threading as a follow-up if the flattened approach degrades reasoning quality.
- **No image-flow support in MVP.** gproxy’s `image.rs` + `image_edit.rs` (~880 LOC) are explicitly
  out-of-scope for Phase 6 unless Codex actually needs them.
  Defer.

### Step 7 — Dep-derived topology for FSMs — INVESTIGATE with one experiment

The user’s intuition: pydantic-graph IS a DAG, the HookDAG’s dep-derived topology pattern works, so
why not unify?
Annotate FSM steps with `reads`/`writes` on state fields, derive edges from data deps,
one consistent IR across hooks + FSMs.

Opus agent pushes back:
- FSM graphs are short (5-15 steps) and stable; HookDAG-style auto-derivation pays off when graphs
  are large or refactored often.
- Decision-routing (`g.decision().branch(g.match(Type).to(handler))`) is control flow, not data flow
  — reads/writes can’t express it.
- Hybrid (dep-derived linear segments + explicit decision routing in the same graph) mixes two
  idioms and is confusing.
- Stateless variant fights `parts_manager` continuity (which is inherently stateful across SSE
  chunks).

But the user’s framing has its own merits:
- Conceptual consistency across hooks + FSMs reduces cognitive load for new contributors.
- Annotating state-field reads/writes makes data flow self-documenting (mermaid render can show data
  deps as annotations).
- Auto-derivation prevents stale-edge bugs when refactoring.

**Don’t decide architecturally on theory.
Run one experiment.**

Pick the smallest FSM — google_intake’s inner `_chunk_dispatch_graph`
(`pop_next_part → classify_part → {handle_text_typed | handle_function_call_typed | handle_inline_data_typed | handle_function_response_typed | handle_unknown_part} → pop_next_part`).
It’s:
- Small (7 steps, 1 decision).
- Stable (Google’s wire format rarely changes).
- Has one decision (`classify_part`-driven) so we can test the hybrid model.

Rewrite it dep-derived.
Compare:

| Dimension | Explicit edges (today) | Dep-derived |
| --- | --- | --- |
| LOC for graph build | ~20 lines (one `_cg.add(...)` block) | ~5 lines (just `build_graph_from_deps([...])`) |
| Topology visibility | All edges in one block | Distributed across step decorators |
| Mermaid output | Identical | Identical |
| Refactoring resilience | Add new arm: edit `_cg.add` block | Add new arm: declare deps, auto-rewires |
| Decision routing | Native `g.match(Type).to(handler)` | Still explicit — hybrid |

If the experiment shows a meaningful win (e.g. half the LOC, clearer refactor story), port to
perplexity_intake’s inner subgraph next.
If marginal, lock in the two-idiom split and document.

**The dep-derivation helper** itself is ~50 LOC (Kahn’s algorithm already lives in
`pipeline/dag.py:HookDAG`; the new helper wraps pydantic-graph’s `GraphBuilder`):

```python
# src/ccproxy/lightllm/graph/_dep_builder.py (new, conditional on Step 7 experiment)
def build_graph_from_deps(
    state_type: type,
    steps: list[tuple[StepFn, set[str], set[str]]],  # (fn, reads, writes)
    *,
    input_type: type = NoneType,
    output_type: type = NoneType,
) -> Graph: ...
```

Files (if experiment proceeds): `src/ccproxy/lightllm/graph/_dep_builder.py` (new, ~50 LOC),
`src/ccproxy/lightllm/graph/google_intake.py` (modify inner subgraph), one test asserting the
derived topology matches the explicit one.

## Critical files

New for Step 4 (OpenAI Responses + gproxy-protocol port):
- `src/ccproxy/lightllm/wire/__init__.py` +
  `wire/responses/{__init__,request,response,stream,types}.py` (~1450 LOC ported from
  gproxy-protocol)
- `src/ccproxy/lightllm/adapters/openai_responses.py`
- `src/ccproxy/lightllm/adapters/_openai_responses_envelope.py`
- `src/ccproxy/lightllm/graph/openai_responses_intake.py` (skipped if Step 5 lands first — use shim
  instead)
- `src/ccproxy/lightllm/graph/openai_responses_render.py`
- `tests/test_wire_responses_models.py`
- `tests/test_lightllm_graph_{openai_responses_load,openai_responses_dump,intake_openai_responses,render_openai_responses}.py`

New for Step 5 (pydantic-ai shim, Mistral pilot):
- `src/ccproxy/lightllm/_pydantic_ai_shim/__init__.py`
- `src/ccproxy/lightllm/_pydantic_ai_shim/_payload_patch.py`
- `src/ccproxy/lightllm/_pydantic_ai_shim/_intake_patch.py`
- `src/ccproxy/lightllm/_pydantic_ai_shim/_profile.py`
- `src/ccproxy/lightllm/_pydantic_ai_shim/_dispatch.py`
- `tests/test_pydantic_ai_shim_mistral.py`
- `tests/test_pydantic_ai_shim_pinning.py` (version-bump snapshot guard)

New for Step 6 (ChatGPT Pro WebUI — only Phase 6A as feasibility pilot this session; 6B-6E
follow-up):
- `src/ccproxy/lightllm/chatgpt_pro/__init__.py`
- `src/ccproxy/lightllm/chatgpt_pro/sentinel.py` (port of
  `gproxy/sdk/gproxy-channel/src/channels/chatgpt/sentinel.rs`)
- `src/ccproxy/lightllm/chatgpt_pro/pow.py` (port of `pow.rs`)
- `src/ccproxy/lightllm/chatgpt_pro/prepare_p.py` (port of `prepare_p.rs`)
- `tests/test_chatgpt_pro_pow.py` (known-good JS-reference hash fixtures)
- `tests/test_chatgpt_pro_prepare_p.py` (deterministic fingerprint config)
- (Phases 6B-6E files deferred to next-next session: `session.py`, `request_builder.py`,
  `sse_v1.py`, `graph/chatgpt_pro_intake.py`, `adapters/chatgpt_pro.py`,
  `hooks/chatgpt_stamp_headers.py`)

Modified for Steps 1-3 + 4 + 5:
- `src/ccproxy/lightllm/parsed.py` (enum rename + new value)
- `src/ccproxy/pipeline/context.py` (rename + `_select_listener_format` extension + `Context.extras`
  accessor)
- `src/ccproxy/pipeline/dag.py` (mermaid render)
- `src/ccproxy/pipeline/render.py` (status integration + capability profile from shim)
- `src/ccproxy/config.py` (`Provider.type` rename)
- `src/ccproxy/lightllm/graph/__init__.py` (3 dispatch branches + param rename + Mistral branch +
  shim delegation)
- `src/ccproxy/lightllm/graph/buffered.py` (synthesis branch for Responses + param rename + shim
  delegation for buffered intake)
- `src/ccproxy/inspector/addon.py` + `routes/transform.py` + `routes/models.py` (param rename)
- `nix/defaults.nix` (Mistral provider entry; eventual provider list expansion contingent on pilot)
- `AGENTS.md`, `docs/lightllm.md`, `docs/configuration.md` (rename + Step 2 doc + Step 5 shim
  architecture doc)

Conditional on Step 7 experiment:
- `src/ccproxy/lightllm/graph/_dep_builder.py` (new)
- `src/ccproxy/lightllm/graph/google_intake.py` (inner subgraph rewrite)

## Reused patterns

- `HookDAG`’s Kahn topo-sort lives in `pipeline/dag.py` — Step 7’s dep helper reuses it.
- `_subgraph_patch.py:add_subgraph` — same monkey-patch idiom Step 5 uses to install
  `Model.build_request_payload` on each shipped pydantic-ai Model class.
  Both patches share: cited upstream TODO/gap, removable when upstream lands the equivalent, mypy
  override row in `pyproject.toml`.
- Adapter pattern (`AnthropicAdapter`, `OpenAIChatAdapter`) — Step 4’s listener-side parsers
  (inbound + render) copy the shape exactly.
  Outbound + intake halves come from the shim if Step 5 lands.
- `SSEPipeline`’s persistent asyncio loop — Step 5 reuses it to bridge pydantic-ai’s
  `AsyncIterator[ModelResponseStreamEvent]` into mitmproxy’s sync `feed(bytes) -> list[event]`
  shape.
- `_tool_kinds.py` mapping — Step 4 extends with OpenAI Responses native tool types if pydantic-ai
  adds new `ToolPartKind` values (e.g. `'tool-browse'`, `'tool-code'`) before Phase 4B lands.
  Step 5 eventually replaces the hand-maintained dict with `Model.supported_native_tools` lookups.
- **gproxy-protocol’s `openai/create_response/` Rust types** — secondary reference for Step 4 if the
  OpenAI SDK’s TypedDicts prove ambiguous on specific discriminated-union shapes.
  Each ported file cites its source file + commit SHA in the docstring.
- **gproxy-protocol’s `transform/*/openai_response/` Rust transforms** — reference (not ported in
  this session) for Phase 4C cross-format work.
  Each pair (Chat ↔ Responses, Claude ↔ Responses, Gemini ↔ Responses) has 300-500 LOC of `TryFrom`
  impls in Rust that map directly to Python conversion functions when we need them.
- **gproxy main workspace `sdk/gproxy-channel/src/channels/chatgpt/` Rust channel** — primary
  reference for Step 6’s ChatGPT Pro WebUI port.
  ccproxy’s existing Perplexity Pro architecture (provider config +
  outbound hooks + intake FSM + adapter) is the proven Python-side template; Step 6 fills in the
  chatgpt.com specifics by porting from this Rust source.
- ccproxy’s existing **Perplexity Pro** integration (`lightllm/pplx.py`, `hooks/pplx_*`,
  `lightllm/graph/perplexity_intake.py`) — the architectural template for Step 6. ChatGPT Pro
  implementation copies this shape exactly: WebUI cookie auth + browser fingerprint + custom SSE
  intake + outbound header-stamping hook.

## Verification

End-to-end signal that the session is done:

1. **Static gates** clean — pytest, mypy, ruff, deprecation warnings (per the standard suite in
   next.md).
2. **Rename pass** — `grep -rn 'ListenerFormat\|listener_format\|upstream_provider' src/ tests/`
   returns zero matches.
3. **`Context.extras` tests** — 2-3 unit tests for get/set/delete/has via glom paths.
4. **`HookDAG.render()` test** — assert rendered mermaid for a fixed 3-hook fixture matches a golden
   string.
5. **Phase 4A live test** —
   `curl -X POST http://127.0.0.1:4001/v1/responses -H 'Authorization: Bearer sk-ant-oat-ccproxy-anthropic' -d '{"model":"claude-sonnet-4-5-20250929","input":"hello","max_output_tokens":100}'`
   → 200 with buffered Anthropic response converted to Responses output shape.
   (Stretch goal for the session.)
6. **Phase 4B live test** (if it lands this session) — Codex CLI talking to `:4001/v1/responses`
   with sentinel key, routed to OpenAI upstream via `providers.codex`. Streaming flow visible in
   `ccproxy flows list`.
7. **Step 7 experiment outcome documented** — either:
   - “google_intake inner subgraph rewritten dep-derived; LOC delta -15, mermaid identical, refactor
     test passed. Port to perplexity next.”
   - OR “experiment showed marginal win; two-idiom split locked in.
     See `docs/lightllm.md` rationale section.”
8. **Step 5 pilot outcome documented** — either:
   - “Mistral payload patch landed against pydantic-ai >=1.99; `MistralModel.build_request_payload`
     returns the expected OpenAI-shape dict; one provider config entry + dispatch branch routes
     `sk-ant-oat-ccproxy-mistral` to Mistral.
     Next: add Groq/Cohere via the same patch; migrate existing adapters in a follow-up.”
   - OR “patch is too fragile to upstream internals; Mistral shipped as `type: openai` provider
     entry via the existing OpenAIChatAdapter.
     Lock in the current adapter strategy.”
9. **Plan file marked done** in `next.md`; “Outstanding / deferred” picks up any items that didn’t
   ship this session.

## Risk Notes

- **Rename pass churn.** ~25 files touched, mostly trivial.
  The risk is missing one and breaking imports.
  Mitigation: rely on mypy + the test suite; run
  `grep -rn 'ListenerFormat\|listener_format' src/ tests/` at the end.
- **OpenAI Responses streaming is complex.** 48 event types; the `response.output_item.added` event
  determines what subsequent deltas mean (text vs reasoning vs function call vs server-side tool).
  The intake FSM needs careful state threading.
  If we write it fresh: write Phase 4B’s intake test first using captured Responses SSE fixtures;
  build the FSM to match.
  If Step 5 lands first: skip this entirely and delegate to
  `OpenAIResponsesModel._process_streamed_response` via the shim — it already handles all 48 events.
- **`reasoning` blocks.** The IR doesn’t natively model OpenAI’s reasoning items.
  Approach: stash them in `raw_extras["cc:reasoning:N"]` for passthrough; on cross-format render to
  Anthropic, drop them (Anthropic’s `thinking` blocks aren’t structurally equivalent).
  Document the lossiness in `docs/lightllm.md` raw_extras conventions table.
- **Codex CLI gating.** If Codex CLI talking to OpenAI actually does require identity headers we
  don’t know about, the 401 path triggers shape replay scoping (defer to a follow-up).
- **Step 5 private-API fragility.** Patching `_messages_create` / `_completions_create` /
  `_process_streamed_response` etc.
  means pydantic-ai’s release notes become required reading.
  Mitigation: pin tight (`>=1.99,<2`); ship the `test_pydantic_ai_shim_pinning.py` snapshot guard so
  version bumps surface payload-shape diffs in CI before merge; document the shim’s pinned-version
  contract at the top of each `_pydantic_ai_shim/*.py` file.
- **Step 5 capture-dict edge cases.** The capture-and-raise trick assumes the SDK call IS the last
  meaningful step in `_messages_create`. If pydantic-ai later wraps the call in retry logic or
  post-processes the response, the capture exception might be caught in the wrong place.
  Mitigation: the wrapper catches `_PayloadCapture` specifically (not bare `Exception`) and
  re-raises anything else; the snapshot tests assert the captured kwargs match expected wire shape.
- **Step 5 async-shim composition.** Wrapping pydantic-ai’s
  `AsyncIterator[ModelResponseStreamEvent]` into our `feed(bytes) -> list[event]` interface requires
  routing chunks through `SSEPipeline`’s persistent loop.
  Test the streaming intake under chunk-boundary stress (1-byte, 16-byte, single-large-chunk) to
  ensure the wrapper preserves event sequence — same property the existing FSMs already have.
- **Step 7 experiment scope creep.** Keep the experiment bounded to ONE inner subgraph; don’t
  refactor the outer dispatch graph (which uses `g.decision().branch(g.match(Type).to(handler))` —
  explicit edges stay there regardless of experiment outcome).

## Stop conditions

- Steps 1-3 are independent and small; ship them all even if Steps 4/5 don’t land this session.
- **Step 5 (pydantic-ai shim) is the gating experiment** — run it BEFORE Phase 4B implementation.
  If it works, Phase 4B is ~250 LOC of shim glue.
  If it doesn’t, Phase 4B is ~1100 LOC of fresh code (or defer Phase 4B to a follow-up and ship 4A
  only this session).
- Step 4 ships as 4A first (the listener MVP). 4B’s scope depends on Step 5’s outcome.
- Step 5: budget 1 day for the Mistral pilot.
  If all 4 directions don’t pass within that window, lock in the fallback (Mistral as `type: openai`
  provider entry) and write up the failure mode for the upstream pydantic-ai issue.
- Step 6 (ChatGPT Pro WebUI) is too big for a single session.
  **Cap this session at Phase 6A** (port Sentinel + PoW + fingerprint, ~600 LOC) as a feasibility
  pilot — confirms the cryptographic + fingerprint pieces work against live chatgpt.com.
  Phases 6B-6E (request builder, SSE-v1 decoder, intake FSM, hooks, routing negotiation) belong in
  their own follow-up session(s). If Phase 6A doesn’t land cleanly, defer all of Step 6 to a fresh
  session.
- Step 7 experiment: if it takes more than 2 hours including the comparison write-up, time-box and
  pick a verdict on incomplete data.
  The goal is a decision, not a perfect implementation.

## What’s NOT in this plan

- Production rollout (Kyle-owned): push, `nh os switch ~/.config/nixos`.
- More live matrix coverage (rows 3, 4, 5, 6, 7, 8, 9, 10 + negative paths).
  Already covered at unit-test level; live verification can come opportunistically.
- **Phase 4C cross-format transforms** (Anthropic ↔ Responses, Chat ↔ Responses, Gemini ↔
  Responses). gproxy-protocol has the spec for all three pairs in Rust; we port one pair per
  follow-up PR after Phase 4B ships.
  Each pair is ~300-500 LOC of mechanical port.
- **Wholesale migration of existing providers** (Anthropic / OpenAIChat / Google / Perplexity) to
  pydantic-ai shims. Step 5 is the 4-direction Mistral pilot; migration of existing providers is
  contingent on the pilot’s stability outcome and gets its own follow-up session per provider.
  The trajectory is: Google first (outbound-only, lowest blast radius) → Anthropic → OpenAIChat.
  Perplexity stays as-is (pydantic-ai doesn’t have an equivalent for Perplexity Pro’s WebUI wire).
- **Free coverage expansion** (Bedrock, Cohere, Groq, OpenRouter, Xai, Cerebras, HuggingFace,
  Ollama) — each is ~30-50 LOC of provider config + dispatch row once the Step 5 shim is proven.
  Defer to a follow-up “free provider expansion” session that runs after Step 5 ships.
- **Porting gproxy-protocol’s Realtime API (`websocket/`) types** — Codex CLI doesn’t use Realtime;
  defer to a hypothetical Realtime-listener session.
- **Step 6 Phases 6B-6E** — request builder, SSE-v1 decoder, intake FSM, outbound hook + adapter,
  routing negotiation.
  The full ChatGPT Pro WebUI surface needs ~3000 LOC of porting; only Phase 6A (the cryptographic
  foundation) fits this session.
  The remaining work gets its own follow-up session(s) — likely two PRs: one for the wire/SSE layer
  (Phases 6B+6C), one for the integration layer (Phases 6D+6E).
- **ChatGPT image generation flows** — gproxy’s `image.rs` + `image_edit.rs` (~880 LOC) are
  explicitly out-of-scope.
  If Codex needs image tools, port later.
- **Contributing APIs upstream** — both `pydantic_graph.GraphBuilder.add_subgraph` and
  `pydantic_ai.Model.build_request_payload` should eventually go upstream as PRs so the patches can
  be deleted. Defer until the patches have stabilized across at least 2 version bumps.
- `kitstore.nix:lib/litellm` cleanup (cosmetic).
- OpenAI Chat Completions Responses-style tool support (`web_search_preview` etc.
  are Responses-only).
- Stateless FSM variant (Opus agent + practical analysis both reject).
