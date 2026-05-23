# Phase 4 — OpenAI Responses API (Codex parity)

## Context

OpenAI's Codex CLI is a precompiled Rust agent binary that talks exclusively
to OpenAI's `/v1/responses` endpoint — a new API family (not a version bump
of `/v1/chat/completions`) that ships:

- `input[]` heterogeneous items (message / function_call / reasoning /
  web_search_call / code_interpreter_call / mcp_call / apply_patch / shell /
  computer_use / file_search) instead of role-based `messages[]`
- Server-side conversation state via `previous_response_id` or
  `conversation: {id}`
- Native `reasoning: {effort: low|medium|high}` for o-series / gpt-5
  thinking-budget control
- Built-in server-side tools unified under one schema: `web_search`,
  `file_search`, `code_interpreter`, `computer_use`, MCP server integrations
- `prompt_cache_key` + `prompt_cache_retention: "in-memory" | "24h"` (OpenAI's
  caching — different semantics from Anthropic's block-level `cache_control`)
- `background: bool` mode (poll-based async response generation)
- 48 streaming event types (vs Chat Completions' ~15)

ccproxy currently terminates `/v1/chat/completions` (`InboundFormat.OPENAI_CHAT`)
and `/v1/messages` (`InboundFormat.ANTHROPIC_MESSAGES`). Codex CLI traffic
hits a sentinel-key URL but bounces because we don't recognize the
`/v1/responses` path or its request shape. **Phase 4 closes that gap.**

This is the main public-API deliverable in the master plan
`next-session-provider-coverage-and-naming.md` (Step 4). The previous
session (commit pending) shipped Step 1 (naming pass — `ListenerFormat` →
`InboundFormat`, `Provider.provider` → `Provider.type`), Step 2
(`Context.extras` accessor), Step 3 (`HookDAG.render()` mermaid).

### Architectural decisions inherited from prior planning

1. **Wire-format types come from the OpenAI SDK first, gproxy-protocol second.**
   `.venv/lib/python3.13/site-packages/openai/types/responses/` is the
   primary spec (TypedDicts). Where the SDK's loose TypedDict unions don't
   preserve the discriminator we need, port the specific Pydantic-equivalent
   type from `~/dev/src/gproxy/sdk/gproxy-protocol/src/openai/create_response/`
   with attribution in the docstring.

2. **Listener-side parse + render is hand-written.** ccproxy owns its inbound
   parser (`adapters/openai_responses.py`) and its render FSM
   (`graph/openai_responses_render.py`) — consistent with existing
   `AnthropicAdapter` + `anthropic_render.py` etc. Pydantic-ai is a CLIENT
   library; it doesn't receive requests.

3. **Outbound payload + intake FSM is written FRESH this phase.** The master
   plan's lift-and-patch shim trajectory (Step 5 / Mistral pilot) is
   explicitly deferred until after Phase 4 ships — per user direction in the
   previous session: "the Mistral stuff was gonna come after." Phase 4B
   writes a hand-coded 48-event intake FSM following the
   `anthropic_intake.py` / `openai_intake.py` pattern. When Step 5 lands in
   a future session, Phase 4B's intake can opportunistically migrate to a
   shim wrapper — but for now we ship without the dependency.

4. **Phase 4A's verification is cheap.** The existing
   `lightllm/graph/buffered.py` already does cross-format buffered transforms
   (per-upstream intake FSM → synthesize SSE → output-shape assembler). 4A
   only needs to add an `InboundFormat.OPENAI_RESPONSES → Responses JSON`
   output arm (~50 LOC). The smoke test (`POST /v1/responses` with any
   existing sentinel) works end-to-end without writing fresh intake.

5. **Cross-format transforms (Responses ↔ Anthropic, Responses ↔ Chat,
   Responses ↔ Gemini) are deferred to Phase 4C** — a follow-up after 4B.
   gproxy-protocol's `src/transform/*/openai_response/` directory has the
   spec for all three pairs (300-500 LOC of Rust `TryFrom` impls each); we
   port one pair per follow-up PR.

### Reference sources

The full context lives in `next-session-provider-coverage-and-naming.md`,
Section "Step 4 — OpenAI Responses API support (Codex parity)" (lines
246-427). Key external references:

- **OpenAI SDK TypedDicts**:
  `.venv/lib/python3.13/site-packages/openai/types/responses/` — wire-shape
  contract used by the official Python client. Source of truth for the
  permissive types we accept on inbound.
- **gproxy-protocol Rust types**:
  `~/dev/src/gproxy/sdk/gproxy-protocol/src/openai/create_response/{request,response,stream,types}.rs`
  — strongly-typed discriminated unions. Used as a secondary reference when
  the SDK's TypedDict unions lose information we need. Port specific files
  ONLY when a 4A or 4B file hits an edge case the SDK can't disambiguate.
  Cite source file + commit SHA in the docstring.
- **pydantic-ai's `OpenAIResponsesModel`** at
  `~/dev/src/pydantic-ai/pydantic_ai_slim/pydantic_ai/models/openai.py:1724`
  (with `OpenAIResponsesStreamedResponse` at `:3367`) — the eventual shim
  target. Not consumed this session, but the 48-event intake FSM we write
  can be compared against this implementation for parity hints.

---

## Scope

### In scope this session

**Phase 4A (mandatory):** listener-side parse + buffered output arm. End
state — `POST /v1/responses` with any existing sentinel
(`sk-ant-oat-ccproxy-anthropic`, etc.) routes to that provider's
upstream, the response comes back, gets converted into Responses-shape
JSON, returned to the client. Buffered (non-streaming) only. ~500 LOC
total.

**Phase 4B (stretch — only if 4A finishes early):** start the upstream
adapter (`OpenAIResponsesAdapter.render`) and the intake FSM scaffolding.
Don't attempt the full 48-event handler set in one session — pick the
5-10 highest-value events (`response.created`,
`response.output_item.added`, `response.content_part.added`,
`response.text.delta`, `response.text.done`, `response.completed`,
`response.failed`) for a first cut.

### Explicitly deferred

- **Full Phase 4B** (all 48 SSE event types, full streaming intake +
  render, Codex CLI end-to-end works against `api.openai.com/v1/responses`)
  — multi-session work, see follow-up section below
- **Phase 4C** (cross-format transforms: Responses ↔ Anthropic, ↔ Chat, ↔
  Gemini)
- **Step 5** (pydantic-ai shim / Mistral pilot) — original master plan
  Step 5, deferred per prior decision
- **ChatGPT Pro WebUI integration** — original Step 6
- **`background: bool` polling mode** — out of scope until Codex needs it
- **`conversation: {id}` server-side state** — out of scope; Codex's
  `previous_response_id` path covers the common case
- **OpenAI's `prompt_cache_key` / `prompt_cache_retention` semantics** —
  preserve via `raw_extras` for now; mapping to Anthropic's
  `cache_control` is a Phase 4C concern
- **Shape replay for Codex** — no documented identity-header requirements
  analogous to Anthropic's `x-anthropic-billing-header`. Revisit ONLY if
  Codex requests start failing 401/403; capture with Wireshark + `ccproxy
  flows compare` then

---

## Phase 4A — listener MVP

End state: `POST /v1/responses` is a routable inbound format. The listener
parses the request, runs the inbound DAG, dispatches to ANY existing
upstream provider, takes the buffered response, converts it to
Responses-shape JSON, returns it.

### Item 1 — `InboundFormat.OPENAI_RESPONSES` enum value

`src/ccproxy/lightllm/parsed.py`:

```python
class InboundFormat(StrEnum):
    UNKNOWN = "unknown"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"      # NEW
```

~5 LOC.

### Item 2 — Path detection in `_select_inbound_format`

`src/ccproxy/pipeline/context.py`:

```python
def _select_inbound_format(req: http.Request | None) -> InboundFormat:
    if req is None:
        return InboundFormat.UNKNOWN
    path = (req.path or "").split("?", 1)[0]
    if path.startswith("/v1/messages") or req.headers.get("anthropic-version"):
        return InboundFormat.ANTHROPIC_MESSAGES
    if path.startswith("/v1/chat/completions") or path.startswith("/chat/completions"):
        return InboundFormat.OPENAI_CHAT
    if path.startswith("/v1/responses") or path.startswith("/responses"):   # NEW
        return InboundFormat.OPENAI_RESPONSES                                 # NEW
    return InboundFormat.UNKNOWN
```

~5 LOC.

### Item 3 — `adapters/openai_responses.py` (load only; render raises)

New file. Parse `input[]` heterogeneous items into pydantic-ai IR
`list[ModelMessage]`. Uses OpenAI SDK TypedDicts from
`openai/types/responses/` as the wire-shape contract.

Skeleton:

```python
"""OpenAI Responses API listener-side adapter.

Inbound (wire → IR):
- ``load_messages(body, raw_extras)`` parses ``input[]`` heterogeneous
  items into pydantic-ai ``ModelMessage`` IR. Items not absorbed into the
  IR (reasoning blocks, server-side tool calls, file_search results, etc.)
  are preserved verbatim under conventional ``raw_extras`` keys for
  passthrough.

Outbound (IR → wire):
- ``render(req)`` raises ``NotImplementedError`` in Phase 4A. Phase 4B
  ships the render path.
"""

from __future__ import annotations
from typing import Any
from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ModelResponse,
    SystemPromptPart, UserPromptPart, ToolCallPart, ToolReturnPart,
    TextPart, ThinkingPart, ImageUrl, BinaryContent,
)
from ccproxy.lightllm.adapters._openai_responses_envelope import (
    parse_input_item,
)


class OpenAIResponsesAdapter:
    @classmethod
    def load_messages(
        cls,
        input_items: list[dict[str, Any]],
        *,
        instructions: str | None = None,
        raw_extras: dict[str, Any],
    ) -> list[ModelMessage]:
        """Parse Responses ``input[]`` items into pydantic-ai IR.

        ``instructions`` (the top-level system-prompt-equivalent) becomes
        a ``SystemPromptPart`` prepended to the first ``ModelRequest``.
        """
        ...

    @classmethod
    def render(cls, req) -> bytes:
        raise NotImplementedError("Phase 4B")
```

~300 LOC including the per-item-kind dispatch in `_openai_responses_envelope`
(see Item 4).

### Item 4 — `adapters/_openai_responses_envelope.py`

Per-item-kind dispatch helpers. The `input[]` array is a discriminated
union over `type`:

- `"message"` → role + content (text/image/file)
- `"function_call"` → `ToolCallPart`
- `"function_call_output"` → `ToolReturnPart`
- `"reasoning"` → `ThinkingPart` + `raw_extras["openai_responses:reasoning:N"]`
  for the structured blocks pydantic-ai can't model
- `"web_search_call"` / `"code_interpreter_call"` / `"mcp_call"` /
  `"computer_call"` / `"file_search_call"` / `"apply_patch"` / `"shell"` →
  stash in `raw_extras["openai_responses:server_tool:N"]` (these are
  server-side tool invocations; we preserve them but don't model them)

Skeleton:

```python
"""Per-item-kind parsers for OpenAI Responses ``input[]`` items.

The ``input[]`` array is a discriminated union over the item ``type``
field. Each branch extracts the IR-modellable fields and stashes the
remainder under a conventional ``raw_extras`` key for lossless passthrough.

Conventional ``raw_extras`` key scheme:

| Wire key | raw_extras key | Why |
|---|---|---|
| ``reasoning`` block | ``openai_responses:reasoning:{i}`` | pydantic-ai's ``ThinkingPart`` only carries content string; structured ``summary[]`` + ``encrypted_content`` not modeled |
| ``web_search_call`` etc. | ``openai_responses:server_tool:{i}`` | Server-side tool invocations have no IR equivalent |
| ``status``, ``id`` on items | ``openai_responses:item_id:{i}`` | Item IDs needed for ``previous_response_id`` continuation |
"""
```

~100 LOC.

### Item 5 — `buffered.py` OPENAI_RESPONSES output arm

`src/ccproxy/lightllm/graph/buffered.py` already synthesizes streaming
events from buffered upstream responses (Anthropic `BetaMessage`, OpenAI
`ChatCompletion`, Google `GenerateContentResponse`) and drives the
existing intake FSM. The output side has two arms today (OPENAI_CHAT,
ANTHROPIC_MESSAGES). Add a third:

```python
if inbound_format is InboundFormat.OPENAI_RESPONSES:
    out_dict = _parts_to_openai_responses(
        parts=parts,
        model=model,
        provider_response_id=_intake_provider_response_id(intake),
        finish_reason=_intake_finish_reason(intake),
    )
```

New helper `_parts_to_openai_responses` synthesizes the Responses
buffered shape:

```json
{
  "id": "resp_...",
  "object": "response",
  "model": "...",
  "status": "completed",
  "output": [
    {"type": "message", "role": "assistant", "content": [
      {"type": "output_text", "text": "..."},
      ...
    ]},
    {"type": "function_call", "call_id": "...", "name": "...", "arguments": "..."},
    ...
  ],
  "usage": {"input_tokens": ..., "output_tokens": ...}
}
```

~80 LOC (50 for `_parts_to_openai_responses`, 30 for dispatch wiring).

### Item 6 — Tests

- `tests/test_lightllm_graph_openai_responses_load.py` — parametrized
  cases: simple text input, multi-item input with function_call +
  function_call_output, image input, instructions field, reasoning items
  preserved via raw_extras. ~120 LOC.

- `tests/test_lightllm_graph_openai_responses_buffered_output.py` —
  feed canned IR parts through `_parts_to_openai_responses`, assert
  expected output shape with `output[0].content[0].text == "..."`,
  `usage.input_tokens == N`, etc. ~80 LOC.

### Phase 4A verification

```bash
just up
curl -sS -X POST http://127.0.0.1:4001/v1/responses \
  -H 'Authorization: Bearer sk-ant-oat-ccproxy-anthropic' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "input": "Say hello in one word.",
    "max_output_tokens": 50
  }' | jq .
```

Expected: 200 response, JSON in Responses buffered shape (`{"id": "resp_...",
"object": "response", "output": [{"type": "message", ...}]}`).

Inspect with `ccproxy flows list` then `ccproxy flows compare <flow_id>` to
verify the request went through the new listener, hit Anthropic upstream,
and the buffered transform stitched everything back together.

### Phase 4A LOC total

~510 LOC including tests. Single-session deliverable.

---

## Phase 4B — upstream support (stretch / next session)

End state: Codex CLI talks to `:4001/v1/responses` with sentinel
`sk-ant-oat-ccproxy-codex`, request flows through to `api.openai.com/v1/responses`,
streaming response works end-to-end.

### Files

```
src/ccproxy/lightllm/adapters/openai_responses.py  (extend with .render)
src/ccproxy/lightllm/graph/openai_responses_intake.py        (~700 LOC)
src/ccproxy/lightllm/graph/openai_responses_render.py        (~400 LOC)
src/ccproxy/lightllm/wire/__init__.py                        (~5 LOC)
src/ccproxy/lightllm/wire/responses/__init__.py              (~5 LOC)
src/ccproxy/lightllm/wire/responses/response.py              (~150 LOC, port of gproxy-protocol)
src/ccproxy/lightllm/wire/responses/stream.py                (~700 LOC, port of gproxy-protocol — 48 SSE event types as discriminated union)
src/ccproxy/lightllm/wire/responses/request.py               (~150 LOC, port of gproxy-protocol)
src/ccproxy/lightllm/wire/responses/types.py                 (~200 LOC, port of gproxy-protocol)
```

Plus dispatch branches in `lightllm/graph/__init__.py` (3 lines x 3 funcs
= ~15 LOC).

### `OpenAIResponsesAdapter.render` (outbound)

Map pydantic-ai IR → Responses request body:
- `list[ModelMessage]` → `input[]` (reverse of `load_messages`)
- `SystemPromptPart` → top-level `instructions` field
- `ToolDefinition.tool_kind == 'tool-search'` etc. → `tools: [{type: web_search}, ...]`
- `settings['reasoning_effort']` → `reasoning: {effort: ...}`
- Anything in `raw_extras["openai_responses:reasoning:N"]` /
  `raw_extras["openai_responses:server_tool:N"]` stitched back in order

~400 LOC.

### 48-event streaming intake FSM

Follow `anthropic_intake.py` / `openai_intake.py` pattern. Outer router
pops events; per-event-type handler step routes via `_g.decision()` →
`.branch(_g.match(ResponseTextDelta).to(handle_text_delta))` etc. 48
typed event classes (port from `gproxy-protocol/src/openai/create_response/stream.rs`
as Pydantic v2 discriminated union).

Critical event subset (the first cut would handle these and `raise
NotImplementedError` on the rest):

| Event | Handler |
|---|---|
| `response.created` | Stash `response.id` into state for `provider_response_id` |
| `response.in_progress` | No-op (status update) |
| `response.output_item.added` | Push new item onto `parts_manager`; type drives whether to make a `TextPart` / `ToolCallPart` / `ThinkingPart` |
| `response.output_item.done` | Close the current item |
| `response.content_part.added` | Begin a content part within the current item |
| `response.text.delta` | `parts_manager.handle_text_delta(vendor_part_id=...)` |
| `response.text.done` | Flush; finalize text part |
| `response.reasoning.text.delta` | `parts_manager.handle_thinking_delta(...)` |
| `response.reasoning.text.done` | Flush thinking |
| `response.function_call_arguments.delta` | `parts_manager.handle_tool_call_delta(args=...)` |
| `response.function_call_arguments.done` | Flush args; finalize tool call |
| `response.completed` | Pull final usage from `response.usage`; emit `FinalResultEvent` |
| `response.failed` | Set `state.error`; emit error event |
| `response.incomplete` | Set `finish_reason='length'` or similar; emit done |

Other 35-ish events (`response.queued`, `response.web_search_call.searching`,
`response.code_interpreter_call.code.delta`, `response.mcp_call.*`, etc.)
get stubbed `handle_ignored` first cut; we wire them as `_IgnoredEvent`
markers and add real handlers as Codex actually uses them.

~700 LOC fresh.

### Render FSM (listener-side IR → Responses SSE)

Consumes `ModelResponseStreamEvent` from the intake (or from other-format
intakes via cross-format Phase 4C); emits Responses SSE bytes. Inverse of
the intake's 48-event spec; can ship with a smaller surface (mirror the
critical-event subset from intake).

~400 LOC.

### Dispatch wiring

`src/ccproxy/lightllm/graph/__init__.py`:

```python
def dispatch_intake(*, provider_type: str, ...) -> AnyAsyncIntakeFSM:
    ...
    if provider_type == "openai_responses":
        return OpenAIResponsesIntakeFSM(model=model, request_params=request_params)
    ...

def dispatch_render(*, inbound_format: InboundFormat, ...) -> AnyAsyncRenderFSM:
    ...
    if inbound_format is InboundFormat.OPENAI_RESPONSES:
        return OpenAIResponsesRenderFSM(model=model)
    ...

def dispatch_dump_sync(req: "LLMRenderInput", *, provider_type: str) -> bytes:
    ...
    if provider_type == "openai_responses":
        from ccproxy.lightllm.adapters.openai_responses import OpenAIResponsesAdapter
        return OpenAIResponsesAdapter.render(req)
    ...
```

Add `OpenAIResponsesIntakeFSM` to `AnyAsyncIntakeFSM` union;
`OpenAIResponsesRenderFSM` to `AnyAsyncRenderFSM`.

### Provider config + sentinel

```yaml
# user's ccproxy.yaml (or nix/defaults.nix for shipped default)
providers:
  codex:
    auth: { type: file, file: ~/.opnix/secrets/openai-api-key }
    host: api.openai.com
    path: /v1/responses
    type: openai_responses
```

Sentinel `sk-ant-oat-ccproxy-codex` routes via `forward_oauth` →
Responses upstream.

### Phase 4B verification

```bash
# Phase 4B live test
codex --api-base http://127.0.0.1:4001 --api-key sk-ant-oat-ccproxy-codex \
  "Summarize this codebase in 5 bullets."

# Or curl-equivalent for the streaming path:
curl -sS -N -X POST http://127.0.0.1:4001/v1/responses \
  -H 'Authorization: Bearer sk-ant-oat-ccproxy-codex' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-5-pro",
    "input": "Count to 5.",
    "stream": true
  }'
```

Expected: real Codex CLI session works end-to-end; SSE stream visible in
`ccproxy flows list` with the right 48-event sequence.

---

## Phase 4C — cross-format transforms (deferred)

gproxy-protocol implements every cross-protocol transform between
Responses and the other dialects bidirectionally. Subdirectories under
`gproxy-protocol/src/transform/`:

- `openai/{generate_content,stream_generate_content}/openai_response/` —
  OpenAI Chat ↔ Responses (both directions)
- `claude/{generate_content,stream_generate_content}/openai_response/` —
  Claude ↔ Responses (both directions)
- `gemini/{generate_content,stream_generate_content}/openai_response/` —
  Gemini ↔ Responses (both directions)

Each subdirectory has `request.rs` + `response.rs` (sometimes `utils.rs`)
implementing the `TryFrom` mappings. Porting is bespoke per pair (~300-500
LOC each) but the algorithmic content is already worked out — we translate
logic, not design it.

The hardest case (Anthropic `thinking` ↔ Responses `reasoning`) is solved
in gproxy-protocol; we don't need to invent the mapping.

Until 4C lands, cross-format Responses↔X requests fail-loud with
`UnsupportedListenerError` / `UnsupportedUpstreamError`.

---

## Critical files

### New (Phase 4A)

```
src/ccproxy/lightllm/adapters/openai_responses.py
src/ccproxy/lightllm/adapters/_openai_responses_envelope.py
tests/test_lightllm_graph_openai_responses_load.py
tests/test_lightllm_graph_openai_responses_buffered_output.py
```

### Modified (Phase 4A)

```
src/ccproxy/lightllm/parsed.py                  (InboundFormat.OPENAI_RESPONSES enum value)
src/ccproxy/pipeline/context.py                 (_select_inbound_format + /v1/responses arm)
src/ccproxy/lightllm/graph/buffered.py          (+OPENAI_RESPONSES output arm + helper)
src/ccproxy/lightllm/adapters/__init__.py       (export OpenAIResponsesAdapter)
docs/lightllm.md                                (document the new listener format + raw_extras conventions)
```

### Conditional (Phase 4B if it lands this session)

```
src/ccproxy/lightllm/wire/__init__.py                       (new package)
src/ccproxy/lightllm/wire/responses/{__init__,request,response,stream,types}.py
src/ccproxy/lightllm/graph/openai_responses_intake.py
src/ccproxy/lightllm/graph/openai_responses_render.py
src/ccproxy/lightllm/graph/__init__.py                      (3 dispatch branches)
nix/defaults.nix                                            (optional: add `codex` Provider entry)
tests/test_lightllm_graph_intake_openai_responses.py
tests/test_lightllm_graph_render_openai_responses.py
tests/test_wire_responses_models.py                         (round-trip serialization for ported wire types)
```

---

## Reused patterns

- **`buffered.py` cross-format synthesis** — `lightllm/graph/buffered.py:1-56`
  doc and the existing `_parts_to_openai_chat_completion` /
  `_parts_to_anthropic_message` helpers are the template for Phase 4A's
  `_parts_to_openai_responses`. Pattern: pull
  `parts_manager.get_parts()` after intake drains, serialize each
  `TextPart` / `ToolCallPart` / `ThinkingPart` into the listener
  envelope's per-part shape.

- **Adapter envelope pattern** — `adapters/_anthropic_envelope.py` /
  `adapters/_openai_envelope.py` are the templates for
  `_openai_responses_envelope.py`. Pattern: per-content-kind dispatch
  helpers; absorbed-keys constant; `raw_extras` stitch-back.

- **`raw_extras` conventions** — see existing
  `docs/lightllm.md#raw_extras-contract` for the
  `cc:msg:N:block:M` / `unknown_block:msg:N:idx:M` /
  `image_detail:msg:N:block:M` patterns. Phase 4A introduces three new
  keys: `openai_responses:reasoning:{i}` (structured reasoning blocks
  pydantic-ai's `ThinkingPart` can't fully model),
  `openai_responses:server_tool:{i}` (web_search/code_interpreter/mcp/
  computer_use/file_search/apply_patch/shell call objects),
  `openai_responses:item_id:{i}` (item IDs needed for
  `previous_response_id` chaining).

- **`ModelResponsePartsManager`** — the intake state machine. Used by
  every existing `*_intake.py`. Phase 4B's
  `OpenAIResponsesIntakeFSM` uses it identically;
  `handle_text_delta` / `handle_thinking_delta` /
  `handle_tool_call_delta` / `handle_tool_call_part` do the same work
  for Responses event types as they do for Anthropic / OpenAI Chat /
  Google.

- **`_subgraph_patch.py`** monkey-patch precedent — if 4B's intake needs
  a two-level FSM (e.g., per-event subgraph that walks
  `response.output_item.added` sub-content), reuse the
  `GraphBuilder.add_subgraph` pattern. The Perplexity and Google intakes
  are the existing reference.

- **OpenAI SDK TypedDicts** —
  `.venv/lib/python3.13/site-packages/openai/types/responses/` covers
  the wire shape. Import `Response`, `ResponseInputItem`,
  `ResponseStreamEvent`, etc. directly for type-checking the boundary
  code.

- **gproxy-protocol Rust types** — port-on-demand reference. When the
  SDK's TypedDict union loses a discriminator we need, port the
  specific Pydantic-equivalent type from
  `~/dev/src/gproxy/sdk/gproxy-protocol/src/openai/create_response/`
  with attribution:

  ```python
  class ResponseInputItem(BaseModel):
      """One item in the Responses ``input[]`` array.

      Ported from gproxy-protocol/src/openai/create_response/types.rs:N-M
      (commit <SHA>) because the OpenAI SDK's TypedDict union doesn't
      preserve the discriminator we need.
      """
      ...
  ```

---

## Verification

End-of-session signal:

1. **Static gates clean** — pytest, mypy, ruff, no deprecation warnings.

2. **Phase 4A unit tests pass:**
   ```bash
   uv run pytest tests/test_lightllm_graph_openai_responses_load.py \
                 tests/test_lightllm_graph_openai_responses_buffered_output.py -v
   ```

3. **Phase 4A live smoke test** (curl from above) returns 200 with
   Responses-shaped JSON.

4. **Inspector trace clean:**
   ```bash
   ccproxy flows list
   ccproxy flows compare <flow_id>
   ```
   Forwarded request should be Anthropic-shape (going to api.anthropic.com);
   client response should be Responses-shape (coming back from
   buffered.py output arm).

5. **Documentation updated** — `docs/lightllm.md` mentions
   `InboundFormat.OPENAI_RESPONSES`, the three new `raw_extras` keys, and
   the buffered output arm.

6. **Phase 4B live test** (if 4B lands this session):
   - `codex` CLI talking to `:4001/v1/responses` with sentinel key works
     end-to-end
   - Streaming flow visible in `ccproxy flows list` with the expected
     SSE event sequence

7. **Plan-file outcome documented** at the bottom of this file:
   - "Phase 4A landed: listener format + load_messages + buffered output
     arm shipped. POST /v1/responses → Anthropic upstream works."
   - Per-direction outcome for Phase 4B if it landed (full vs. critical-events
     subset, what's stubbed `NotImplementedError`, etc.)

---

## Risk notes

- **OpenAI Responses streaming is complex.** 48 event types; the
  `response.output_item.added` event determines what subsequent deltas
  mean (text vs reasoning vs function call vs server-side tool). The
  intake FSM needs careful state threading.
  Mitigation: write Phase 4B's intake test first using captured Responses
  SSE fixtures; TDD the FSM against the fixtures.

- **`reasoning` blocks.** The IR doesn't natively model OpenAI's reasoning
  items (structured `summary[]` + `encrypted_content`). Pydantic-ai's
  `ThinkingPart` only carries a content string.
  Approach: stash reasoning items in `raw_extras["openai_responses:reasoning:N"]`
  for passthrough; on cross-format render to Anthropic (Phase 4C), drop
  the structured fields and emit only the text content (Anthropic's
  `thinking` blocks aren't structurally equivalent). Document the
  lossiness in `docs/lightllm.md` raw_extras conventions table.

- **Item IDs (`previous_response_id` continuation).** Responses items have
  `id` fields that Codex uses for conversation chaining. We need to
  preserve them through the round-trip.
  Approach: stash in `raw_extras["openai_responses:item_id:N"]` on inbound;
  re-stitch on outbound render.

- **Server-side tools (web_search, file_search, code_interpreter,
  computer_use, mcp_call, apply_patch, shell).** These are item kinds the
  IR doesn't model. They appear in `input[]` (assistant turn includes the
  call) AND in streaming output as their own event family.
  Approach: stash as `raw_extras["openai_responses:server_tool:N"]` for
  passthrough; never attempt to translate to other formats (Phase 4C
  cross-format rules will explicitly drop these from Anthropic/Chat
  output).

- **Codex CLI gating.** If Codex CLI talking to OpenAI actually does
  require identity headers we don't know about, the 401 path triggers
  shape-replay scoping (defer to a follow-up session if it bites). For
  Phase 4B's first cut, ship without shape replay and see if it works.

- **Cross-format `tool_choice` semantics.** Responses uses
  `tool_choice: {type: "function", name: "..."}` (object); Chat uses
  `tool_choice: {type: "function", function: {name: "..."}}` (nested
  object); Anthropic uses `tool_choice: {type: "tool", name: "..."}` (no
  nesting). Phase 4C concern, but flag here so it's not forgotten.

- **`prompt_cache_key` / `prompt_cache_retention`.** OpenAI's caching has
  different semantics from Anthropic's block-level `cache_control`. There's
  no clean mapping.
  Approach: preserve as `raw_extras["openai_responses:prompt_cache_*"]`;
  cross-format Anthropic ↔ Responses transform (Phase 4C) drops these
  fields in either direction.

- **`background: bool` polling mode.** Out of scope this phase entirely.
  If a request comes in with `background: true`, fail-loud with a 501.

- **gproxy-protocol port drift.** When we port specific Rust types, we
  freeze them against a commit SHA. If upstream gproxy-protocol moves on,
  our ports don't.
  Mitigation: cite the source commit SHA in the docstring; add a CI job
  that diffs against gproxy-protocol HEAD periodically (low priority —
  the wire format itself rarely changes, only the Rust expression of it).

---

## What's NOT in this plan

- **Migration of Phase 4B's intake to the pydantic-ai shim** — happens
  AFTER Step 5 (Mistral pilot) proves the shim trajectory works. Phase
  4B writes fresh code this phase; opportunistic migration is a
  follow-up.

- **`background: true` polling mode** — Codex CLI doesn't use it for
  interactive sessions; defer until requested.

- **`conversation: {id}` server-side state** — Codex's
  `previous_response_id` is the common path.

- **OpenAI Realtime API** (websocket types) — Codex CLI doesn't use
  Realtime; defer to a hypothetical Realtime-listener session.

- **OpenAI image generation flows** (`dall-e-*` via Responses) — out of
  scope unless Codex needs them.

- **Cross-format transforms** (Phase 4C: Responses ↔ Anthropic / Chat /
  Gemini) — explicitly deferred to follow-up PRs, one pair per PR. The
  spec exists in gproxy-protocol.

- **Shape replay for Codex** — no documented identity-header requirements;
  revisit only if requests start failing 401/403.

- **ChatGPT Pro WebUI as a Responses upstream** (master plan Step 6 /
  Phase 6A) — completely separate effort, multi-session.

- **Mistral pilot / pydantic-ai shim** (master plan Step 5) — deferred
  per prior session direction.

---

## Stop conditions

- **Phase 4A is mandatory.** End-of-session bar: items 1-5 shipped, unit
  tests green, live smoke test (curl `/v1/responses` with existing
  sentinel) returns 200 with Responses-shaped JSON.

- **Phase 4B is stretch.** Budget: if 4A finishes with >50% of the
  session remaining, start 4B with the critical-events subset (7 events
  listed above) and the OpenAIResponsesAdapter render path. If 4A
  consumes most of the session, defer ALL of 4B to a follow-up. Don't
  ship a half-implemented 48-event FSM that silently drops most events
  — it's worse than no implementation.

- **Wire-type porting from gproxy-protocol is on-demand.** Don't preemptively
  port `wire/responses/{request,response,stream,types}.py` until 4B
  actually needs them. The OpenAI SDK's TypedDicts cover 4A entirely.

- **If 4B hits an event-handling ambiguity** (e.g., what's the right IR
  shape for `response.code_interpreter_call.code.delta`?) — stash in
  `raw_extras["openai_responses:server_tool:..."]`, emit a DEBUG log,
  move on. Don't block 4B on getting every event perfect; ship the
  critical-event subset and iterate.

- **If Phase 4B's live test fails on Codex CLI specifically** (vs. raw
  curl) — diagnose via `ccproxy flows compare`. Likely cause:
  Codex expects an identity header we're not stamping. Defer shape replay
  to a follow-up session and document in the outcome.

---

## Outstanding / next session

After this phase lands, the immediate follow-up work in priority order:

1. **Complete Phase 4B's full 48-event handler set** if only the
   critical-event subset shipped — one session, ~600 LOC additional
   handlers.

2. **Mistral pilot / pydantic-ai shim** (master plan Step 5). Now that
   Phase 4 ships fresh, the shim becomes an architecture experiment that
   would let Phase 4B's intake become ~50 LOC of shim glue. Validate on
   Mistral first because it's OpenAI-compat and has zero migration cost.

3. **Phase 4C cross-format transforms** — port gproxy-protocol's
   `transform/*/openai_response/` one pair per PR. Start with the
   highest-traffic pair (likely Anthropic ↔ Responses since Codex CLI
   wants to route to Claude via ccproxy).

4. **ChatGPT Pro WebUI port** (master plan Step 6 / Phase 6A) — static
   port of `gproxy/sdk/gproxy-channel/src/channels/chatgpt/{sentinel,pow,prepare_p}.rs`.
   Independent of Phase 4 / 5; can run in parallel.
