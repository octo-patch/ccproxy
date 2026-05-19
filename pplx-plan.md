# Plan: Comprehensive Perplexity SSE Response Parser Overhaul

> **Phase relationship**: The existing plan in `~/.claude/plans/fix-pplx-md-reactive-lollipop.md`
> describes the **outbound prompt-injection tool calling** (shipped 2026-05-18; inert on every
> frontier model tested — Claude/GPT-5/DeepSeek/Grok all detect and refuse the injection). This
> plan is the **next phase**: comprehensive **response-side parser overhaul** that surfaces the
> rich step/tool data Perplexity emits via its native MCP-connector channel, plus all the other
> step types our parser silently drops today.

## Context

Audit (via captured GitHub-MCP probe response) and external research (8 OSS Perplexity parsers
+ Perplexity SPA bundle extraction in `~/dev/scratch/research/pplx/sse-research/`) revealed:

1. **ccproxy's `_extract_deltas` handles 1 of 68 `step_type` values** (`RESEARCH_CLARIFYING_QUESTIONS`).
   Everything else — `MCP_TOOL_INPUT`/`OUTPUT`, `SEARCH_WEB`, `READ_RESULTS`, `BROWSER_*`,
   `READ_CALENDAR`/`UPDATE_CALENDAR`, `GENERATE_IMAGE_RESULTS`, `FLIGHTS_*`, etc. — is **silently
   dropped**. CLAUDE.md flags this as the worst failure mode.

2. **The full SPA enum is now known** (extracted from `ThreadEntryContext-hgdcVwpW.js`'s `??`
   content-field fallback chain in `STEP_TYPE_ENUM.md`):

   | Category | step_types | OSS coverage |
   |---|---|---|
   | Core | 4 (INITIAL_QUERY, FINAL, TERMINATE, ATTACHMENT) | 2 |
   | Web search | 3 (SEARCH_WEB, WEB_RESULTS, SEARCH_RESULTS) | 2 |
   | Deep Research | 8 (ENTROPY_REQUEST, THOUGHT, *_CLARIFYING_QUESTIONS, COMET_AGENT_*) | 1 |
   | Browser agent | 19 (BROWSER_SEARCH, URL_NAVIGATE, BROWSER_GET_SITE_CONTENT, …) | 0 |
   | **MCP tool calls** | **2 (MCP_TOOL_INPUT, MCP_TOOL_OUTPUT)** | **0** |
   | Calendar/Email | 14 (READ_CALENDAR, SEND_EMAIL, GET_FREE_BUSY, …) | 0 |
   | Image/Video | 4 (GENERATE_IMAGE*, GENERATE_VIDEO*) | 1 |
   | Flights | 5 | 0 |
   | Productivity | 9 (CREATE_TASKS, CODE, CREATE_CHART, CANVAS_AGENT, …) | 0 |
   | **TOTAL** | **68** | **6 (9%)** |

3. **The naming convention is regular**: `UPPER_SNAKE_CASE` step_type ↔ `lower_snake_case_content`
   typed field. `MCP_TOOL_INPUT` → `mcp_tool_input_content`. Enables one generic dispatcher to
   handle the entire enum (including future additions) instead of 68 hardcoded branches.

4. **Two parallel channels carry the same step data** (per cross-repo analysis):
   - **Primary**: `blocks[].plan_block.steps[]` — structured, typed `*_content` fields, authoritative
   - **Fallback**: top-level `event.text` field — JSON-string of `step[]`, used when blocks empty
   - Reading both = double-counting. Use primary, fall back only when absent.

5. **Three new SSE endpoints discovered** (`pplx-unofficial-sdk/ANALYSIS.md` HAR analysis) — secondary
   channels alongside the main `/rest/sse/perplexity_ask`:
   - `/rest/sse/perplexity_mcp_response` — dedicated MCP tool response channel
   - `/rest/sse/handle_tool_user_approval_response` — interactive approval (blocking on user)
   - `/rest/sse/pro_search_step_result` — granular pro-search step results
   - Wire shapes unknown; live data not captured. **Probe-only this round**, real implementation deferred.

6. **Other silent drops** (audit from our own captured probe):
   - Bare `markdown_block` (no `diff_block` wrapper) — terminal events use this
   - `pending_followups_block.followups[]` — captured into `state.followups`, never emitted
   - `display_model` (top-level) — should populate `response.model` (we currently echo the requested model)
   - 30+ other top-level fields (mostly browser-UI metadata, but some semantically meaningful)

7. **The shipped prompt-injection approach is dead-on-arrival on frontier models** (confirmed across
   5 models in `examples/pplx.py` smoke test). The XML parser machinery itself works correctly;
   no model emits XML to feed it. Keep the code (regression-tested via 21 unit tests), gate the
   injection behind a config flag, default OFF.

**Intended outcome**: ccproxy clients see the same conceptual visibility into Perplexity's tool
use that Perplexity's own SPA renders. MCP tool calls appear as Claude-style reasoning_content
"thinking" blocks and as OpenAI `delta.tool_calls` (informational); structured per-step data
attaches via non-spec response fields for agentic clients. No step type is silently dropped.

## Locked Decisions

| ID | Decision | Rationale |
|---|---|---|
| E1 | Generic step dispatcher via lowercase content-field convention | One function handles all 68 step types + future additions. Each renderer is small and specialized. |
| E2 | `plan_block.steps[]` is the PRIMARY channel; `text`-field JSON is FALLBACK | Cross-repo evidence: structured channel is authoritative when present, text is used by some repos when blocks empty. Double-reading = double-counting. |
| E3 | All step types render as `delta.reasoning_content` (Claude-style thinking) | Universal UX value. Per-type rendering templates produce human-readable lines. Unknown step_types render with a generic fallback. |
| E4 | MCP_TOOL_INPUT/OUTPUT ALSO surface as informational `delta.tool_calls` | OpenAI clients with tool-aware UI render these as tool cards. `finish_reason="stop"` (NOT `"tool_calls"`) — execution is server-side, client should not re-execute. |
| E5 | Structured per-step data attaches as `pplx_mcp_steps` non-spec field | Pattern matches existing `pplx_thread_url_slug`. Agentic clients can introspect; standard clients just see content + reasoning. |
| E6 | `display_model` from response → `model_response.model` | Tells clients which actual upstream model fired (vs requested alias). |
| E7 | `state.followups` → `pplx_pending_followups` non-spec field | Already captured. Currently dead state. One line to surface. |
| E8 | Bare `markdown_block` (no diff_block wrapper) → handle like Mode A | Terminal events ship this shape. Currently dropped; usually no data loss because diff_block stream accumulated it, but fragile. |
| E9 | Catch-all DEBUG log for unknown `step_type` AND unknown `intended_usage` | Cheap insurance — next time Perplexity ships a new step type, our logs flag it within one run. |
| E10 | The 3 new SSE endpoints (`perplexity_mcp_response`, `handle_tool_user_approval_response`, `pro_search_step_result`) — **PROBE ONLY this round** | Capture live wire data, document shapes in `docs/pplx.md`; defer implementation to follow-up because we have zero captured payloads. Add a hook to log when these endpoints are accessed so we can discover them in the wild. |
| E11 | Split `pplx_tool_inject` hook into always-run-folding + gated-prompt-injection | `fold_tool_results` is universally useful (folds `role:tool` messages into Perplexity-readable text). `build_tool_prompt` is broken on frontier models — gate behind `pplx.experimental.tool_prompt: false` (default OFF). |
| E12 | Functional dispatch + nested dataclass state (matches existing codebase paradigm) | New state lives in extended `StreamState`. Renderers are free functions. No classes. |
| E13 | NO Pydantic models for the step types | The naming convention + opaque `*_content` dict is more flexible. Adding 68 Pydantic models is overengineering for read-only renderers. perplexity-cli's "all content is `dict[str, Any]`" pattern is the right floor. |

## Components

### New file: `src/ccproxy/lightllm/pplx_steps.py` (~300 LOC)

Pure functions + dataclasses for step rendering.

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class StepRenderResult:
    reasoning_text: str            # for delta.reasoning_content
    structured: dict[str, Any] | None  # for state.mcp_steps or state.steps
    tool_call: dict[str, Any] | None   # for delta.tool_calls (informational)

# Convention: UPPER_CASE step_type → lower_case_content field
def _content_field_for(step_type: str) -> str:
    return step_type.lower() + "_content"

def render_step(step: dict[str, Any]) -> StepRenderResult:
    """Dispatch a plan_block.steps[] entry to its renderer.

    Falls back to `_render_generic` for unknown step types so nothing is ever
    silently dropped. Reads content from the typed field (e.g.
    `mcp_tool_input_content` for `MCP_TOOL_INPUT`).
    """
    step_type = step.get("step_type") or "UNKNOWN"
    content_key = _content_field_for(step_type)
    content = step.get(content_key) or step.get("content") or {}  # tolerate text-field shape
    renderer = _RENDERERS.get(step_type, _render_generic)
    return renderer(step_type, content, step.get("uuid", ""))


# Specialized renderers (most common types):
def _render_initial_query(step_type, content, uuid): ...  # skip — already in user msg
def _render_search_web(step_type, content, uuid): ...    # "→ Web search: {queries}"
def _render_read_results(step_type, content, uuid): ...  # "← Read {N} results"
def _render_mcp_tool_input(step_type, content, uuid):
    """→ [GitHub] get_me({}): Getting authenticated user info"""
    app = content.get("app", "unknown")
    name = content.get("tool_name", "unknown")
    args = content.get("tool_args") or {}
    summary = content.get("tool_input_summary", "")
    args_repr = json.dumps(args, separators=(",", ":")) if args else "{}"
    text = f"\n→ [{app}] {name}({args_repr})"
    if summary:
        text += f": {summary}"
    text += "\n"
    structured = {
        "phase": "input", "step_uuid": uuid, "app": app, "tool_name": name,
        "tool_args": args, "goal_id": content.get("goal_id"),
        "request_user_approval": (content.get("request_user_approval") or {}).get("request_user_approval", False),
        "summary": summary,
    }
    tool_call = {
        "id": f"call_pplx_{uuid[:24]}" if uuid else f"call_pplx_{_short_uuid()}",
        "type": "function",
        "function": {"name": f"{app.lower()}_{name}", "arguments": json.dumps(args)},
    }
    return StepRenderResult(text, {"mcp_step": structured}, tool_call)

def _render_mcp_tool_output(step_type, content, uuid):
    """← get_me (success)"""
    name = (content.get("tool_name") or "tool")
    status = content.get("status", "unknown")
    text = f"← {name} ({status})\n"
    structured = {
        "phase": "output", "step_uuid": uuid, "status": status,
        "content": content.get("content"), "goal_id": content.get("goal_id"),
        "should_rerun_query": content.get("should_rerun_query", False),
    }
    return StepRenderResult(text, {"mcp_step": structured}, None)

def _render_final(step_type, content, uuid): ...           # skip — answer already in markdown_block
def _render_terminate(step_type, content, uuid): ...       # "✓ Done"
def _render_browser_search(step_type, content, uuid): ...  # "→ Browser: {query}"
def _render_read_calendar(step_type, content, uuid): ...   # "→ Calendar: read"
def _render_generate_image(step_type, content, uuid): ...  # "→ Generating image: {prompt}"
# … one per category (~10 total renderers cover ~80% of likely traffic)

def _render_generic(step_type, content, uuid):
    """Catch-all for unknown / unmapped step types. Logs at DEBUG."""
    summary = content.get("summary") or content.get("description") or content.get("query") or ""
    text = f"[{step_type}]" + (f" {summary}" if summary else "") + "\n"
    structured = {"step_type": step_type, "step_uuid": uuid, "content_keys": list(content.keys())}
    logger.debug("pplx_steps: unmapped step_type=%s (uuid=%s)", step_type, uuid)
    return StepRenderResult(text, {"unmapped_step": structured}, None)


_RENDERERS = {
    "INITIAL_QUERY": _render_initial_query,
    "FINAL": _render_final,
    "TERMINATE": _render_terminate,
    "SEARCH_WEB": _render_search_web,
    "READ_RESULTS": _render_read_results,
    "MCP_TOOL_INPUT": _render_mcp_tool_input,
    "MCP_TOOL_OUTPUT": _render_mcp_tool_output,
    "BROWSER_SEARCH": _render_browser_search,
    # … extend incrementally; unknowns hit _render_generic and log
}
```

### Modified: `src/ccproxy/lightllm/pplx.py`

**Extend `StreamState`** (line 289):
```python
@dataclass
class StreamState:
    answer_seen: str = ""
    reasoning_seen: str = ""
    ids: dict[str, str] = field(default_factory=dict)
    followups: list[str] = field(default_factory=list)
    final: bool = False
    tool_state: ToolCallState | None = None       # existing
    # NEW:
    mcp_steps: list[dict[str, Any]] = field(default_factory=list)
    all_steps: list[dict[str, Any]] = field(default_factory=list)   # full structured trail
    goals: list[dict[str, Any]] = field(default_factory=list)        # plan_block.goals snapshot
    seen_step_uuids: set[str] = field(default_factory=set)           # dedup across events
    pending_step_reasoning: str = ""                                  # drain → reasoning_delta
    pending_step_tool_calls: list[dict[str, Any]] = field(default_factory=list)  # drain → delta.tool_calls
```

**Extend `_extract_deltas`** (line 331):

Inside the existing `for block in blocks` loop, add a branch for `plan_block.steps[]`
(currently we only walk `plan_block.goals[]`):

```python
if intended_usage in ("pro_search_steps", "plan", "reasoning_plan_block"):
    plan_block = block.get("plan_block") or {}
    # EXISTING: walk goals[] for reasoning (keep as-is)
    ...
    # NEW: walk steps[] for full step coverage
    for step in (plan_block.get("steps") or []):
        if not isinstance(step, dict):
            continue
        uuid = step.get("uuid", "")
        # Dedup: same step uuid arrives in multiple cumulative events
        if uuid and uuid in state.seen_step_uuids:
            continue
        if uuid:
            state.seen_step_uuids.add(uuid)
        result = render_step(step)
        if result.reasoning_text:
            state.pending_step_reasoning += result.reasoning_text
            reasoning_delta = (reasoning_delta or "") + result.reasoning_text
        if result.structured:
            state.all_steps.append({"step_type": step.get("step_type"), **result.structured})
            if "mcp_step" in result.structured:
                state.mcp_steps.append(result.structured["mcp_step"])
        if result.tool_call:
            state.pending_step_tool_calls.append(result.tool_call)
    # NEW: capture goals snapshot (always overwrite — server sends cumulative)
    if (goals := plan_block.get("goals")):
        state.goals = list(goals)

# NEW: bare markdown_block (no diff_block wrapper)
mb = block.get("markdown_block")
if isinstance(mb, dict) and not block.get("diff_block"):
    answer_str = mb.get("answer")
    if isinstance(answer_str, str) and answer_str.startswith(state.answer_seen):
        delta = answer_str[len(state.answer_seen):]
        if delta:
            answer_delta = (answer_delta or "") + delta
        state.answer_seen = answer_str

# NEW: catch-all for unknown intended_usage (DEBUG log; once per stream)
elif intended_usage not in _KNOWN_INTENDED_USAGES:
    if intended_usage not in state.seen_step_uuids:  # reuse set as "logged" tracker
        state.seen_step_uuids.add(f"_iu:{intended_usage}")
        logger.debug("pplx: unhandled intended_usage=%s keys=%s", intended_usage, list(block.keys()))
```

**Extend text-field handling** (line 363) — current code only handles `RESEARCH_CLARIFYING_QUESTIONS`.
After E2 decision (structured channel is primary), the text field is a fallback for when no
`plan_block.steps[]` exists in this event. Logic:

```python
# Only walk text-field steps if this event has NO plan_block (avoid double-emit)
if isinstance(parsed, list) and not _event_has_plan_block(event):
    for step in parsed:
        if not isinstance(step, dict):
            continue
        st = step.get("step_type")
        if st == "RESEARCH_CLARIFYING_QUESTIONS":
            raise PerplexityClarifyingQuestionsError(_extract_clarifying_questions(step))
        # The text-field shape uses `content` instead of typed `*_content` fields.
        # render_step tolerates both.
        if step.get("uuid") in state.seen_step_uuids:
            continue
        result = render_step(step)
        # … same accumulation logic as above
```

**Update `chunk_parser`** (line 871) — drain `pending_step_reasoning` and
`pending_step_tool_calls`:

```python
if self._state.pending_step_reasoning:
    delta.reasoning_content = (getattr(delta, "reasoning_content", None) or "") + self._state.pending_step_reasoning
    self._state.pending_step_reasoning = ""

if self._state.pending_step_tool_calls:
    existing = getattr(delta, "tool_calls", None) or []
    delta.tool_calls = existing + self._state.pending_step_tool_calls
    self._state.pending_step_tool_calls = []

# On final chunk: attach non-spec fields + use display_model
if self._state.final:
    response.pplx_thread_url_slug = self._state.ids.get("thread_url_slug")
    if self._state.mcp_steps:
        response.pplx_mcp_steps = self._state.mcp_steps
    if self._state.followups:
        response.pplx_pending_followups = self._state.followups
    if self._state.goals:
        response.pplx_goals = self._state.goals
    if self._state.all_steps:
        response.pplx_steps = self._state.all_steps
```

**Update `transform_response`** (line 776) — non-streaming mirror:

```python
if state.mcp_steps:
    model_response.pplx_mcp_steps = state.mcp_steps
if state.followups:
    model_response.pplx_pending_followups = state.followups
if state.goals:
    model_response.pplx_goals = state.goals
if state.all_steps:
    model_response.pplx_steps = state.all_steps
display_model = state.ids.get("display_model")
if display_model:
    model_response.model = display_model
# Reasoning content from collected steps
if state.reasoning_seen or state.pending_step_reasoning:
    try:
        message.reasoning_content = (state.reasoning_seen or "") + state.pending_step_reasoning
    except Exception:
        pass
# Tool calls from MCP (informational, finish_reason stays "stop")
if state.pending_step_tool_calls:
    try:
        message.tool_calls = state.pending_step_tool_calls
    except Exception:
        pass
```

Note: `finish_reason` stays `"stop"` even when `pplx_mcp_steps` non-empty. The model already
finished using the tool server-side; the client must NOT re-execute. The existing
`finish_reason = "tool_calls"` promotion (from the prompt-injection path via
`state.tool_state.has_emitted`) stays — that's the user-defined-tools case, gated by the
experimental flag.

### Modified: `src/ccproxy/hooks/pplx_tool_inject.py`

Split into two distinct concerns:

```python
@hook(reads=["tools", "tool_choice", "messages"], writes=["messages"])
def pplx_tool_inject(ctx, _):
    body = ctx._body if isinstance(ctx._body, dict) else {}

    # ALWAYS: fold role:tool messages into Perplexity-readable user text
    messages = body.get("messages")
    if isinstance(messages, list):
        messages = fold_tool_results(messages)
        body["messages"] = messages

    # GATED: prompt-injection only when explicitly enabled
    if not get_config().pplx.experimental.tool_prompt:
        ctx._body = body
        return ctx

    tools = body.get("tools")
    tool_choice = body.get("tool_choice", "auto")
    if not tools or tool_choice == "none":
        ctx._body = body
        return ctx

    prompt = build_tool_prompt(tools, tool_choice)
    if not prompt:
        ctx._body = body
        return ctx

    messages = _prepend_to_last_user_message(messages, prompt)
    body["messages"] = messages
    ctx._body = body
    logger.info("pplx_tool_inject: experimental prompt-injection applied for %d tool(s)", len(tools))
    return ctx
```

### Modified: `src/ccproxy/config.py`

Add experimental section under existing `PplxConfig`:

```python
class PplxExperimentalConfig(BaseModel):
    tool_prompt: bool = False
    """Inject user-defined tools as XML protocol prompt. Defeated by frontier
    models in 2026; default OFF. See docs/pplx.md 'Tool calling' section."""

class PplxConfig(BaseModel):
    thread: PplxThreadConfig = Field(default_factory=PplxThreadConfig)
    experimental: PplxExperimentalConfig = Field(default_factory=PplxExperimentalConfig)
```

### Modified: `nix/defaults.nix`

```nix
pplx = {
  thread = { ... };
  experimental = { tool_prompt = false; };
};
```

Run `just sync-template` after edit.

### New file: `tests/test_pplx_steps.py` (~350 LOC, ~20 tests)

| Test | Verifies |
|---|---|
| `test_content_field_for_convention` | `MCP_TOOL_INPUT` → `mcp_tool_input_content` etc. |
| `test_render_step_dispatches_by_step_type` | Known types route to specialized renderer |
| `test_render_step_unknown_falls_through_to_generic` | Unmapped type doesn't crash; logs DEBUG; structured.unmapped_step populated |
| `test_render_step_text_field_shape_uses_content_key` | Tolerates `content` (text-field) shape vs typed `*_content` shape |
| `test_render_initial_query_emits_nothing` | INITIAL_QUERY is suppressed (redundant with user msg) |
| `test_render_final_emits_nothing` | FINAL suppressed (redundant with markdown_block) |
| `test_render_search_web` | "→ Web search: {queries}" format |
| `test_render_read_results` | "← Read {N} results" format |
| `test_render_mcp_tool_input_full` | Reasoning text + structured mcp_step + tool_call all populated |
| `test_render_mcp_tool_input_empty_args` | tool_args={} renders as `{}` |
| `test_render_mcp_tool_input_request_user_approval_captured` | structured.request_user_approval reflects gate |
| `test_render_mcp_tool_output_success` | "← {tool_name} (success)" |
| `test_render_mcp_tool_output_should_rerun_propagated` | structured.should_rerun_query |
| `test_render_browser_search` | Browser agent renderer |
| `test_render_terminate` | ✓ Done variant |
| `test_render_generate_image` | Image-gen renderer |
| `test_render_read_calendar` | Calendar renderer |

### Modified: `tests/test_lightllm_pplx.py` (~6 new tests)

| Test | Verifies |
|---|---|
| `test_extract_deltas_walks_plan_block_steps` | `plan_block.steps[]` is consumed (not just `goals[]`); state.mcp_steps populated for synthetic MCP step |
| `test_extract_deltas_dedups_step_uuid_across_events` | Same step uuid in 3 cumulative events emits reasoning only once |
| `test_extract_deltas_text_field_fallback_only_when_no_plan_block` | Avoids double-emit |
| `test_extract_deltas_handles_bare_markdown_block` | Block with `markdown_block` (no `diff_block`) extracts answer |
| `test_extract_deltas_logs_unknown_intended_usage` | DEBUG log fires once per unknown |
| `test_iterator_emits_mcp_step_reasoning_and_tool_calls` | Streaming chunk contains both `reasoning_content` and informational `tool_calls`; `finish_reason="stop"` |
| `test_iterator_attaches_pplx_mcp_steps_to_final_chunk` | `response.pplx_mcp_steps` populated on terminal chunk |
| `test_iterator_uses_display_model_for_response_model` | response.model = "claude46sonnet" when display_model that |
| `test_transform_response_attaches_pending_followups` | Non-streaming: `pplx_pending_followups` non-spec field |

## Implementation Phases

### Phase A — Step renderer module (foundational, no integration)
1. Create `src/ccproxy/lightllm/pplx_steps.py` with `StepRenderResult`, `_content_field_for`,
   `render_step`, generic + ~10 specialized renderers (covering MCP, web, browser, calendar,
   image generation), `_RENDERERS` registry.
2. Create `tests/test_pplx_steps.py` covering renderer dispatch + per-category renderers.
3. `nix develop --command bash -c 'uv run pytest tests/test_pplx_steps.py'` — iterate until green.

### Phase B — Wire renderer into `_extract_deltas` + StreamState
1. Extend `StreamState` with new fields (`mcp_steps`, `all_steps`, `goals`, `seen_step_uuids`,
   `pending_step_reasoning`, `pending_step_tool_calls`).
2. Add `plan_block.steps[]` walk inside the existing `pro_search_steps`/`plan` branch.
3. Add bare-`markdown_block` handling.
4. Add catch-all DEBUG log for unknown `intended_usage`.
5. Gate text-field step processing on "no plan_block in this event" to avoid double-emit.
6. Add unit tests for each new behavior in `tests/test_lightllm_pplx.py`.

### Phase C — Surface to OpenAI clients
1. Drain `pending_step_reasoning` into `delta.reasoning_content` in `chunk_parser`.
2. Drain `pending_step_tool_calls` into `delta.tool_calls` (additive — preserves any from the
   prompt-injection path).
3. Attach non-spec fields (`pplx_mcp_steps`, `pplx_pending_followups`, `pplx_goals`,
   `pplx_steps`) on the terminal chunk.
4. Use `state.ids["display_model"]` for `response.model` if present.
5. Mirror in `transform_response` for non-streaming.
6. Add iterator + transform_response integration tests.

### Phase D — Split & gate `pplx_tool_inject`
1. Refactor hook: `fold_tool_results` runs unconditionally; `build_tool_prompt` + prepend gated
   on `config.pplx.experimental.tool_prompt`.
2. Add `PplxExperimentalConfig` to `src/ccproxy/config.py`.
3. Add `experimental = { tool_prompt = false; };` to `nix/defaults.nix`; `just sync-template`.
4. Update `tests/test_pplx_tools.py` — verify gated behavior (mock get_config).
5. Update `docs/pplx.md` "Tool calling" section to describe the flag + state that injection
   is defeated on frontier models in 2026.

### Phase E — Probe & document new SSE endpoints
1. Write `examples/pplx_mcp_endpoints_probe.py` that explicitly exercises queries likely to
   trigger each endpoint:
   - `/rest/sse/perplexity_mcp_response` — MCP-heavy query, capture flow URLs
   - `/rest/sse/handle_tool_user_approval_response` — query likely to require approval (a
     GitHub write action, e.g., "create a branch")
   - `/rest/sse/pro_search_step_result` — pro search mode (`perplexity/best` with deep query)
2. Dump all flow URLs from `ccproxy flows list` after each probe; identify any URL outside
   the main `/rest/sse/perplexity_ask`.
3. If new endpoints fire: capture their SSE event shapes via `ccproxy flows dump`, document
   in `docs/pplx.md` as a new "Secondary SSE channels" section.
4. **Do not implement parsers** for these in this round — wire data needs to inform schema first.

### Phase F — Verification & docs
1. `nix develop --command just test` — full suite. Target: 55 prior pplx tests + ~25 new = 80+
   pass; full suite ≤ 2 pre-existing failures (documented).
2. `just lint` — no new errors beyond the documented pre-existing set.
3. `just typecheck` — same.
4. E2E re-run:
   - `examples/pplx.py` (custom-tool injection) with `pplx.experimental.tool_prompt: false` →
     verify no injection happens; model receives clean user query; folding still works.
   - `examples/pplx.py` with `pplx.experimental.tool_prompt: true` → verify injection happens
     (still won't trigger tool calls, but mechanically correct).
   - `examples/pplx_mcp_probe.py` → verify `pplx_mcp_steps` populated; `delta.reasoning_content`
     contains "→ [GitHub] get_me..." line; `delta.tool_calls` informational entry present.
5. Update `docs/pplx.md`:
   - Extend existing "Tool calling" section: clarify experimental flag + frontier-model limitation.
   - New "Step types & MCP" section: enumerate handled step types, link to STEP_TYPE_ENUM.md
     for the full SPA enum, describe the renderer convention, document the new non-spec
     response fields (`pplx_mcp_steps`, `pplx_pending_followups`, `pplx_goals`, `pplx_steps`).
   - If Phase E discovered new endpoints, add "Secondary SSE channels" section.

## Critical Files

### New
- `src/ccproxy/lightllm/pplx_steps.py` (~300 LOC, ~10 renderers + dispatcher)
- `tests/test_pplx_steps.py` (~350 LOC, ~20 tests)
- `examples/pplx_mcp_endpoints_probe.py` (~80 LOC) — Phase E

### Modified
- `src/ccproxy/lightllm/pplx.py` — `StreamState` extensions, `_extract_deltas` (steps walk +
  bare markdown_block + catch-all), `chunk_parser` (drain + non-spec fields), `transform_response`
  (mirror)
- `src/ccproxy/hooks/pplx_tool_inject.py` — split hook, gate prompt injection on config flag
- `src/ccproxy/config.py` — `PplxExperimentalConfig` with `tool_prompt: bool = False`
- `nix/defaults.nix` — `experimental = { tool_prompt = false; };`
- `src/ccproxy/templates/ccproxy.yaml` — regenerated via `just sync-template`
- `tests/test_lightllm_pplx.py` — ~9 new integration tests
- `tests/test_pplx_tools.py` — verify gated injection behavior
- `docs/pplx.md` — extend Tool calling section, new Step types & MCP section, possibly
  Secondary SSE channels section
- `CLAUDE.md` (pplx paragraph) — note the experimental flag + step renderer module

## Reused Existing Code

- `_extract_deltas` (pplx.py:331) — extend in place; existing Mode A/B/C/D answer parsing untouched.
- `_extract_clarifying_questions` (pplx.py:519) — keep; called from the text-field fallback path.
- `StreamState` (pplx.py:289) — extend; new fields default to empty so existing tests pass unchanged.
- `PerplexityProIterator` (pplx.py:846) — `chunk_parser` extended; init unchanged.
- `PerplexityProConfig.transform_response` (pplx.py:776) — extended in same shape as iterator.
- `fold_tool_results` (pplx_tools.py) — reused unchanged from existing plan.
- `build_tool_prompt` (pplx_tools.py) — reused; gated by config flag now.
- `extract_tool_deltas` (pplx_tools.py) — reused unchanged (still useful when injection enabled).
- `@hook` decorator (pipeline/hook.py) — same pattern, no changes.
- `get_config()` (config.py) — reused for reading the new experimental flag.

## Verification

### Unit tests (Phase A–C)
~20 new in `test_pplx_steps.py` + ~9 new in `test_lightllm_pplx.py` = ~29 new tests. All
synthetic SSE inputs, no network.

### E2E (Phase F)

**MCP probe path** (the core validation):
```bash
just up
nix develop --command bash -c 'uv run python examples/pplx_mcp_probe.py'

# Then dump and inspect:
nix develop --command bash -c 'ccproxy flows dump 2>/dev/null' | python3 -c "
import json,sys
d=json.load(sys.stdin)
e=[x for x in d['log']['entries'] if 'perplexity_ask' in x['request']['url'] and x['request']['method']=='POST'][-1]
ent=e['response']['content']['text']
# Find any 'pplx_mcp_steps' field on the rewritten client response
client_e=[x for x in d['log']['entries'] if 'chat/completions' in x['request']['url']][-1]
print(client_e['response']['content']['text'][-2000:])
"

# Expected (in the OpenAI response):
# - choices[0].message.reasoning_content contains "→ [GitHub] get_me" line
# - choices[0].message.tool_calls contains informational MCP tool call entries
# - choices[0].finish_reason == "stop"  (NOT "tool_calls" — server-side execution)
# - top-level pplx_mcp_steps: [{phase: "input", app: "GitHub", tool_name: "get_me", ...}, {phase: "output", ...}]
# - top-level pplx_thread_url_slug present
# - model field reflects display_model (e.g., "claude46sonnet")
```

**Custom-tool injection path** (regression for the existing flag-gated mechanism):
```bash
# Default: flag off
nix develop --command bash -c 'uv run python examples/pplx.py'
# Verify: forwarded query_str does NOT contain "Available tools" prompt
# Verify: response is normal Perplexity prose (model refuses or just answers)

# Flag on:
nix develop --command bash -c 'CCPROXY_PPLX_EXPERIMENTAL_TOOL_PROMPT=1 uv run python examples/pplx.py'
# Verify: forwarded query_str contains the injection prompt
# (Model will still refuse on frontier models — expected limitation, documented)
```

**Probe new endpoints** (Phase E):
```bash
nix develop --command bash -c 'uv run python examples/pplx_mcp_endpoints_probe.py'
nix develop --command bash -c 'ccproxy flows list --jq "map(.url) | unique"'
# Look for URLs other than /rest/sse/perplexity_ask:
#   /rest/sse/perplexity_mcp_response  (if MCP-heavy)
#   /rest/sse/handle_tool_user_approval_response  (if write action)
#   /rest/sse/pro_search_step_result  (if pro search mode)
# Document the actual shapes in docs/pplx.md
```

## Out of Scope (This Round)

- **Full parser implementation for the 3 secondary SSE endpoints** — probe & document only;
  defer until we have captured payloads. Trying to implement against unknown wire shapes is
  premature.
- **Pydantic models for individual step content types** — opaque `dict[str, Any]` per renderer
  is the right floor (matches perplexity-cli pattern). Strict typing 65 step types is
  overengineering for read-only renderers.
- **Approval-flow interactive handling** (`request_user_approval: true` blocking) — Phase E will
  document the shape; actually implementing user-approval intermediation needs UX design
  (how does an OpenAI client surface "Perplexity asked for approval"? A `tool_call` with a
  special id? A 4xx with a structured error? A separate WebSocket-style channel?). Out of scope.
- **Removing the prompt-injection code path** — keep it gated, default-off. The unit tests
  cover it. If we ever encounter a model that accepts the injection, it's ready.
- **Browser agent / Comet / Studio / Labs steps** — covered by the generic renderer (DEBUG log
  + structured field capture). Specialized renderers only for the ~10 most common categories.
- **Reconnect endpoint** (`/rest/sse/perplexity_ask/reconnect/{uuid}`, discovered by
  pplx-unofficial-sdk) — separate concern, not on the request hot path.

## Open Issues / Future Work

- **`request_user_approval: true`** semantics — captured into `pplx_mcp_steps[].request_user_approval`
  but currently no special handling. When Phase E captures wire data of the approval flow firing,
  we'll know whether to surface as a tool_call requiring response (OpenAI pattern) or as a 4xx
  blocking error.
- **`MCP_TOOL_OUTPUT.status != "success"`** — we only have positive cases. Once we observe an
  error, extend `_render_mcp_tool_output` to format it distinctly.
- **`pipedream_extra_args`** — field exists in MCP_TOOL_INPUT content; semantically unknown.
  Captured into structured field; no special handling.
- **Streaming args granularity for the informational `tool_calls`** — currently emit each
  MCP_TOOL_INPUT atomically. Same trade-off as the prompt-injection path (D3 from the prior plan).
- **`MCP_TOOL_OUTPUT.content` is a JSON-encoded string** — for very large outputs (file dumps,
  query results), this may bloat `pplx_mcp_steps`. Consider truncation policy or lazy
  attachment via `pplx_mcp_step_content_url` (require client to fetch on demand).
- **Model-routing implications of `display_model`** — if `response.model` reflects the actual
  routed model, clients chaining on the response might pick a different model for the next turn.
  Document this as deliberate transparency.
