"""Perplexity Pro SSE bytes → pydantic-ai IR events via FSM.

Pydantic-graph FSM port of
:class:`ccproxy.lightllm.response.intake_perplexity.PerplexityResponseIntake`.
One graph run per :meth:`PerplexityResponseIntakeFSM.feed` call: bytes are
appended to the SSE buffer, complete SSE frames are drained, each frame's
``data:`` payload is JSON-decoded into an event dict, wrapped in a
:class:`_PerplexityEventEnvelope`, and pushed onto an in-state queue. The
outer FSM router drains the queue dispatching each envelope into a nested
per-event subgraph that linearly absorbs IDs and ``has_plan_block``,
optionally walks the ``event.text`` mirror, then pops each ``blocks[]``
entry one at a time and routes it through three independent arms
(plan-block, bare markdown, diff-block) before flushing accumulated
reasoning + answer deltas via the ``ModelResponsePartsManager``.

The behavioral contract matches
:mod:`ccproxy.lightllm.response.intake_perplexity` byte-for-byte: same SSE
framing rules (``\\r\\n\\r\\n`` and ``\\n\\n`` separators, ``[DONE]``
silently ignored, ``data:``-prefix only), same prefix-diff semantics on
answer and reasoning, same ``ask_text`` skip filter, same step
deduplication via ``seen_step_uuids``, same ``RESEARCH_CLARIFYING_QUESTIONS``
silent suppression (the request-side surfaces it as a 400; intake's role is
emission only), same unknown-``intended_usage`` DEBUG dedup. The four
documented diff-block patch modes (Mode A root cumulative, Mode B
chunks-array, Mode C ``/chunks/N`` append, Mode D ``/markdown_block``) are
still handled by :func:`_apply_markdown_patch`. See ``docs/pplx.md`` for
the full wire-format reference.

The per-event subgraph composes into the outer graph via
:meth:`GraphBuilder.add_subgraph` (installed by
:mod:`ccproxy.lightllm.graph._subgraph_patch`). Shared state means
``state.answer_seen`` / ``state.reasoning_seen`` prefix accumulation
threads through both graphs unchanged. Per-event scratch fields
(``has_plan_block``, ``blocks_queue``, ``pending_*_delta``,
``current_event``) are reset by :func:`flush_event_deltas` so nothing
leaks across events.

The persistent-loop bridge between sync mitmproxy callables and this async
FSM lives in :class:`SSEPipeline` (Phase Q). For tests, the parametrize
fixture in ``tests/test_lightllm_response_intake_perplexity.py`` wraps the
async FSM in a one-loop-per-call sync adapter.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Private pydantic-ai import — same justification as the matching note in
# ``response/intake_perplexity.py``. We need byte-identical dispatch
# behavior and there is no public replacement.
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_graph import GraphBuilder, StepContext

import ccproxy.lightllm.graph._subgraph_patch  # noqa: F401  — installs add_subgraph
from ccproxy.lightllm.pplx_steps import _KNOWN_INTENDED_USAGES, render_step

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestParameters

logger = logging.getLogger(__name__)


_PPLX_ID_FIELDS: tuple[str, ...] = (
    "backend_uuid",
    "read_write_token",
    "context_uuid",
    "thread_url_slug",
    "thread_title",
    "display_model",
)
"""Top-level event fields captured into ``state.ids`` whenever they appear."""

_ANSWER_VENDOR_ID = "pplx-answer"
"""Stable vendor_part_id for the answer ``TextPart``."""

_REASONING_VENDOR_ID = "pplx-reasoning"
"""Stable vendor_part_id for the reasoning ``ThinkingPart``."""


# ── Dispatch envelopes ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class _PerplexityEventEnvelope:
    """Envelope wrapping one parsed Perplexity SSE event dict."""

    event: dict[str, Any]


@dataclass(frozen=True)
class _BlockDispatch:
    """Per-block dispatch envelope routed through the three independent arms."""

    block: dict[str, Any]


class _EventDone:
    """Sentinel — no more blocks left for the current event."""


class _FeedDone:
    """Marker returned by the outer router when the events queue is exhausted."""


# ── State ──────────────────────────────────────────────────────────────────


@dataclass
class _PerplexityIntakeState:
    """FSM state for one Perplexity intake graph run.

    The ``events_queue`` is the queue of dispatch envelopes drained from the
    SSE buffer *before* the outer graph run starts; the outer router pops
    from it. The ``out_events`` list accumulates
    :class:`ModelResponseStreamEvent` instances; the terminal outer step
    drains and returns it.

    The streaming state fields (``answer_seen``, ``reasoning_seen``, ``ids``,
    etc.) persist across feed calls so prefix-diffing and identifier capture
    work over the whole stream. The per-event scratch fields
    (``has_plan_block``, ``blocks_queue``, ``pending_*_delta``,
    ``current_event``) are reset at the end of each event by
    :func:`flush_event_deltas`.
    """

    parts_manager: ModelResponsePartsManager
    answer_seen: str = ""
    """Cumulative answer text seen so far — for prefix-diffing."""

    reasoning_seen: str = ""
    """Cumulative reasoning text from ``plan_block.goals[].description``."""

    ids: dict[str, str] = field(default_factory=dict)
    """Captured thread identifiers (last-write-wins)."""

    final: bool = False
    """``True`` once an event carries ``final_sse_message: true``."""

    seen_step_uuids: set[str] = field(default_factory=set)
    """Deduplication set for ``plan_block.steps[].uuid`` across cumulative events."""

    logged_unknown_intended_usages: set[str] = field(default_factory=set)
    """Per-stream dedup for the DEBUG log of unknown ``intended_usage`` values."""

    events_queue: deque[Any] = field(default_factory=deque)
    out_events: list[ModelResponseStreamEvent] = field(default_factory=list)

    # ── Per-event scratch (reset by flush_event_deltas) ────────────────────

    has_plan_block: bool = False
    """``True`` when any block in the current event has a ``plan_block`` dict."""

    blocks_queue: deque[dict[str, Any]] = field(default_factory=deque)
    """Per-event queue of block dicts; the per-event subgraph pops from it."""

    pending_reasoning_delta: str = ""
    """Reasoning text accumulated across the current event's blocks, flushed at event end."""

    pending_answer_delta: str = ""
    """Answer text accumulated across the current event's blocks, flushed at event end."""

    current_event: dict[str, Any] | None = None
    """The current event dict; populated by :func:`absorb_event`, cleared at flush."""


# ── Helpers (called from the FSM step bodies) ───────────────────────────────


def _consume_step(state: _PerplexityIntakeState, step: dict[str, Any]) -> str:
    """Render one ``plan_block.steps[]`` entry; return reasoning text to emit.

    Dedup across SSE events via ``state.seen_step_uuids``. Unlike the
    standalone iterator path, the intake doesn't accumulate structured
    ``state.all_steps`` / ``state.mcp_steps`` lists — those exist only
    for the non-spec OpenAI response-side surface, which the render layer
    owns. We emit only the reasoning text into the IR's ThinkingPart.
    """
    uuid_raw = step.get("uuid") or ""
    uuid_ = uuid_raw if isinstance(uuid_raw, str) else ""
    if uuid_ and uuid_ in state.seen_step_uuids:
        return ""
    if uuid_:
        state.seen_step_uuids.add(uuid_)

    result = render_step(step)
    return result.reasoning_text


def _apply_markdown_patch(state: _PerplexityIntakeState, path: str, value: Any) -> str:
    """Apply one ``diff_block.patches[]`` entry; return the answer delta string.

    Handles all four documented patch modes. Mutates ``state.answer_seen``
    in place. Returns ``""`` when nothing new was extracted.
    """
    # Mode A/B — root patch carrying full markdown_block state (chunks
    # array with offset=0, and/or cumulative ``answer`` string).
    if path == "" and isinstance(value, dict):
        delta = ""
        chunks = value.get("chunks")
        if isinstance(chunks, list):
            offset = value.get("chunk_starting_offset")
            new_text = "".join(c for c in chunks if isinstance(c, str))
            if offset in (None, 0):
                if new_text != state.answer_seen:
                    d = (
                        new_text[len(state.answer_seen) :]
                        if new_text.startswith(state.answer_seen)
                        else new_text
                    )
                    if d:
                        delta += d
                    state.answer_seen = new_text
            elif new_text:
                delta += new_text
                state.answer_seen += new_text
        answer_str = value.get("answer")
        if isinstance(answer_str, str) and answer_str and answer_str.startswith(state.answer_seen):
            d = answer_str[len(state.answer_seen) :]
            if d:
                delta += d
            state.answer_seen = answer_str
        return delta

    # Mode C — incremental chunk append at ``/chunks/N``.
    if path.startswith("/chunks/") and isinstance(value, str):
        state.answer_seen += value
        return value

    # Mode D — cumulative answer at ``/markdown_block`` or
    # ``/markdown_block/answer``.
    if path == "/markdown_block" and isinstance(value, dict):
        answer_str = value.get("answer")
        if isinstance(answer_str, str) and answer_str:
            if answer_str.startswith(state.answer_seen):
                d = answer_str[len(state.answer_seen) :]
                state.answer_seen = answer_str
                return d
            if answer_str != state.answer_seen:
                state.answer_seen = answer_str
                return answer_str
        return ""

    if path == "/markdown_block/answer" and isinstance(value, str):
        if value.startswith(state.answer_seen):
            d = value[len(state.answer_seen) :]
            state.answer_seen = value
            return d
        if value != state.answer_seen:
            state.answer_seen = value
            return value
        return ""

    return ""


# ── Per-event dispatch subgraph ─────────────────────────────────────────────


_eg: GraphBuilder[
    _PerplexityIntakeState, None, _PerplexityEventEnvelope, None
] = GraphBuilder(
    name="pplx_event_dispatch",
    state_type=_PerplexityIntakeState,
    input_type=_PerplexityEventEnvelope,
)


@_eg.step
async def absorb_event(
    ctx: StepContext[_PerplexityIntakeState, None, _PerplexityEventEnvelope],
) -> None:
    """Capture IDs + final flag, compute ``has_plan_block``, enqueue blocks.

    Mirrors the front matter of the original ``_dispatch_one_event``: walk
    ``_PPLX_ID_FIELDS`` into ``state.ids``, set ``state.final`` if the
    event carries ``final_sse_message: true``, filter blocks to dicts, and
    compute the cross-block ``has_plan_block`` precondition that gates the
    ``event.text`` mirror.
    """
    state = ctx.state
    event = ctx.inputs.event
    state.current_event = event

    for key in _PPLX_ID_FIELDS:
        val = event.get(key)
        if isinstance(val, str) and val:
            state.ids[key] = val

    if event.get("final_sse_message"):
        state.final = True

    blocks_raw = event.get("blocks") or []
    blocks: list[dict[str, Any]] = (
        [b for b in blocks_raw if isinstance(b, dict)] if isinstance(blocks_raw, list) else []
    )
    state.has_plan_block = any(isinstance(b.get("plan_block"), dict) for b in blocks)
    state.blocks_queue.extend(blocks)


@_eg.step
async def apply_text_mirror(
    ctx: StepContext[_PerplexityIntakeState, None, None],
) -> None:
    """Walk ``event.text`` JSON-as-step-list when no ``plan_block`` is present.

    Clarifying-questions steps are silently suppressed here — the standalone
    Perplexity request surface owns the 400 escalation. When a structured
    ``plan_block`` exists in any block of the event, we skip the text mirror
    entirely to avoid double-emission against the structured channel.
    """
    state = ctx.state
    event = state.current_event
    if event is None or state.has_plan_block:
        return
    text = event.get("text")
    if not isinstance(text, str):
        return
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(parsed, list):
        return
    for step in parsed:
        if not isinstance(step, dict):
            continue
        if step.get("step_type") == "RESEARCH_CLARIFYING_QUESTIONS":
            continue
        rendered = _consume_step(state, step)
        if rendered:
            state.pending_reasoning_delta += rendered


@_eg.step
async def pop_next_block(
    ctx: StepContext[_PerplexityIntakeState, None, None],
) -> Any:
    """Pop one block dict from the queue, or signal end-of-event via :class:`_EventDone`."""
    state = ctx.state
    if not state.blocks_queue:
        return _EventDone()
    return _BlockDispatch(block=state.blocks_queue.popleft())


@_eg.step
async def apply_plan_arm(
    ctx: StepContext[_PerplexityIntakeState, None, _BlockDispatch],
) -> _BlockDispatch:
    """Plan-block arm: ``pro_search_steps`` / ``plan`` / ``reasoning_plan_block``.

    Walks ``plan_block.goals[].description`` (prefix-diffed against
    ``state.reasoning_seen``) and ``plan_block.steps[]`` (deduped via
    ``state.seen_step_uuids``). Passes the :class:`_BlockDispatch` through
    so the bare-markdown arm sees the same block next.
    """
    state = ctx.state
    block = ctx.inputs.block
    intended_usage = block.get("intended_usage")
    if intended_usage not in ("pro_search_steps", "plan", "reasoning_plan_block"):
        return ctx.inputs
    plan_block = block.get("plan_block") or {}
    if not isinstance(plan_block, dict):
        return ctx.inputs

    goals = plan_block.get("goals") or []
    if isinstance(goals, list):
        for goal in goals:
            if not isinstance(goal, dict):
                continue
            desc = goal.get("description")
            if isinstance(desc, str) and desc.startswith(state.reasoning_seen):
                new = desc[len(state.reasoning_seen) :]
                if new:
                    state.pending_reasoning_delta += new
                    state.reasoning_seen = desc

    for step in plan_block.get("steps") or []:
        if not isinstance(step, dict):
            continue
        rendered = _consume_step(state, step)
        if rendered:
            state.pending_reasoning_delta += rendered

    return ctx.inputs


@_eg.step
async def apply_bare_markdown_arm(
    ctx: StepContext[_PerplexityIntakeState, None, _BlockDispatch],
) -> _BlockDispatch:
    """Bare ``markdown_block`` (no ``diff_block`` wrapper) — terminal full-answer mirror.

    Prefix-diffs ``markdown_block.answer`` against ``state.answer_seen`` and
    appends the new tail to ``state.pending_answer_delta``. Skipped when
    the block carries a ``diff_block`` (the diff-arm wins) or when
    ``intended_usage == "ask_text"`` (it duplicates ``ask_text_0_markdown``).
    """
    state = ctx.state
    block = ctx.inputs.block
    intended_usage = block.get("intended_usage")
    mb = block.get("markdown_block")
    if not isinstance(mb, dict) or block.get("diff_block") or intended_usage == "ask_text":
        return ctx.inputs
    answer_str = mb.get("answer")
    if isinstance(answer_str, str) and answer_str and answer_str.startswith(state.answer_seen):
        bare_delta = answer_str[len(state.answer_seen) :]
        if bare_delta:
            state.pending_answer_delta += bare_delta
        state.answer_seen = answer_str
    return ctx.inputs


@_eg.step
async def apply_diff_block_arm(
    ctx: StepContext[_PerplexityIntakeState, None, _BlockDispatch],
) -> None:
    """Diff-block arm: per-patch dispatch on path.

    For each patch:

    - ``/goals*`` — prefix-diffed reasoning text into ``pending_reasoning_delta``.
    - ``/progress`` — ignored.
    - ``/markdown_block*`` (when ``field == "markdown_block"``) — delegated
      to :func:`_apply_markdown_patch`.

    When the block has no ``diff_block`` at all, log the unknown
    ``intended_usage`` once per stream (via
    ``state.logged_unknown_intended_usages``) and return. ``ask_text``
    blocks are skipped to avoid doubling ``ask_text_0_markdown`` patches.
    """
    state = ctx.state
    block = ctx.inputs.block
    intended_usage = block.get("intended_usage")
    diff_block = block.get("diff_block")

    if not isinstance(diff_block, dict):
        if (
            intended_usage
            and intended_usage not in _KNOWN_INTENDED_USAGES
            and intended_usage not in state.logged_unknown_intended_usages
        ):
            state.logged_unknown_intended_usages.add(intended_usage)
            logger.debug(
                "pplx intake: unhandled intended_usage=%s keys=%s",
                intended_usage,
                list(block.keys()),
            )
        return

    if intended_usage == "ask_text":
        return

    field_name = diff_block.get("field")
    patches = diff_block.get("patches") or []
    if not isinstance(patches, list):
        return

    for patch in patches:
        if not isinstance(patch, dict):
            continue
        path = patch.get("path", "")
        value = patch.get("value")

        if path.startswith("/goals"):
            if isinstance(value, str) and value.startswith(state.reasoning_seen):
                new = value[len(state.reasoning_seen) :]
                if new:
                    state.pending_reasoning_delta += new
                    state.reasoning_seen = value
            continue

        if path == "/progress":
            continue

        if field_name != "markdown_block":
            continue

        delta = _apply_markdown_patch(state, path, value)
        if delta:
            state.pending_answer_delta += delta


@_eg.step
async def flush_event_deltas(
    ctx: StepContext[_PerplexityIntakeState, None, _EventDone],
) -> None:
    """Emit accumulated reasoning + answer deltas via ``parts_manager``; reset per-event scratch.

    Called once per event (after all blocks have been drained). Same SSE
    granularity as the original ``_dispatch_one_event`` — one
    ``handle_thinking_delta`` plus one ``handle_text_delta`` call at most
    per event, whose return events are appended to ``state.out_events``.
    """
    state = ctx.state

    if state.pending_reasoning_delta:
        state.out_events.extend(
            state.parts_manager.handle_thinking_delta(
                vendor_part_id=_REASONING_VENDOR_ID,
                content=state.pending_reasoning_delta,
            )
        )
    if state.pending_answer_delta:
        state.out_events.extend(
            state.parts_manager.handle_text_delta(
                vendor_part_id=_ANSWER_VENDOR_ID,
                content=state.pending_answer_delta,
            )
        )

    # Reset per-event scratch. ``blocks_queue`` is already drained by
    # construction (``pop_next_block`` only returns ``_EventDone`` when
    # empty). Defensive assert guards future refactors.
    assert not state.blocks_queue, "blocks_queue must be empty at flush"
    state.pending_reasoning_delta = ""
    state.pending_answer_delta = ""
    state.has_plan_block = False
    state.current_event = None


_eg.add(
    _eg.edge_from(_eg.start_node).to(absorb_event),
    _eg.edge_from(absorb_event).to(apply_text_mirror),
    _eg.edge_from(apply_text_mirror).to(pop_next_block),
    _eg.edge_from(pop_next_block).to(
        _eg.decision()
        .branch(_eg.match(_EventDone).to(flush_event_deltas))
        .branch(_eg.match(_BlockDispatch).to(apply_plan_arm))
    ),
    _eg.edge_from(apply_plan_arm).to(apply_bare_markdown_arm),
    _eg.edge_from(apply_bare_markdown_arm).to(apply_diff_block_arm),
    _eg.edge_from(apply_diff_block_arm).to(pop_next_block),
    _eg.edge_from(flush_event_deltas).to(_eg.end_node),
)


_event_dispatch_graph = _eg.build()


# ── Outer intake graph (events queue dispatcher) ──────────────────────────


_g: GraphBuilder[
    _PerplexityIntakeState, None, None, list[ModelResponseStreamEvent]
] = GraphBuilder(
    name="pplx_intake",
    state_type=_PerplexityIntakeState,
    output_type=list[ModelResponseStreamEvent],
)


@_g.step
async def frame_next_event(
    ctx: StepContext[_PerplexityIntakeState, None, None],
) -> Any:
    """Router source: pop the next dispatch envelope from the queue, or signal end via :class:`_FeedDone`."""
    state = ctx.state
    if not state.events_queue:
        return _FeedDone()
    return state.events_queue.popleft()


_dispatch_event_step = _g.add_subgraph(_event_dispatch_graph, label="dispatch_event")  # ty: ignore[unresolved-attribute]


@_g.step
async def emit_done(
    ctx: StepContext[_PerplexityIntakeState, None, _FeedDone],
) -> list[ModelResponseStreamEvent]:
    """Terminal step — drain the accumulated IR events and reset for the next feed."""
    out = ctx.state.out_events
    ctx.state.out_events = []
    return out


_g.add(
    _g.edge_from(_g.start_node).to(frame_next_event),
    _g.edge_from(frame_next_event).to(
        _g.decision()
        .branch(_g.match(_FeedDone).to(emit_done))
        .branch(_g.match(_PerplexityEventEnvelope).to(_dispatch_event_step))
    ),
    _g.edge_from(_dispatch_event_step).to(frame_next_event),
    _g.edge_from(emit_done).to(_g.end_node),
)


_intake_graph = _g.build()


# ── Public class ───────────────────────────────────────────────────────────


class PerplexityResponseIntakeFSM:
    """Async pydantic-graph-driven Perplexity Pro SSE intake.

    Behavioral twin of
    :class:`ccproxy.lightllm.response.intake_perplexity.PerplexityResponseIntake`,
    re-expressed as a two-level :class:`GraphBuilder` FSM: an outer graph
    drains the events queue and dispatches each envelope into a nested
    per-event subgraph that pops blocks one at a time and routes them
    through three independent arms. ``parts_manager`` and the stream-level
    state (``answer_seen``, ``reasoning_seen``, ``ids``, etc.) persist
    across calls; per-event scratch is reset at each event's flush.
    """

    name = "perplexity_pro"

    def __init__(self, *, model: str, request_params: ModelRequestParameters) -> None:
        self._model = model
        self._request_params = request_params
        self._sse_buffer = bytearray()
        self.upstream_raw_bytes = bytearray()
        self._state = _PerplexityIntakeState(
            parts_manager=ModelResponsePartsManager(model_request_parameters=request_params),
        )

    @property
    def parts_manager(self) -> ModelResponsePartsManager:
        """Expose the underlying parts manager for tests and downstream renderers."""
        return self._state.parts_manager

    @property
    def state(self) -> _PerplexityIntakeState:
        """Expose the FSM state for tests reaching for identifier capture, seen-uuids, etc."""
        return self._state

    async def feed(self, data: bytes) -> list[ModelResponseStreamEvent]:
        """Buffer bytes, frame SSE events, drive the FSM, return emitted IR events."""
        if not data:
            return []
        self.upstream_raw_bytes.extend(data)
        self._sse_buffer.extend(data)
        for envelope in self._drain_sse_envelopes():
            self._state.events_queue.append(envelope)
        if not self._state.events_queue:
            return []
        result = await _intake_graph.run(state=self._state)
        return result

    async def close(self) -> list[ModelResponseStreamEvent]:
        """Stream end. No trailing events required — parts_manager keeps state."""
        return []

    def _drain_sse_envelopes(self) -> Iterator[_PerplexityEventEnvelope]:
        """Frame SSE events from ``self._sse_buffer``; wrap each into a dispatch envelope.

        Handles both ``\\r\\n\\r\\n`` (industry standard) and ``\\n\\n`` (some servers)
        separators; partial frames remain buffered for the next ``feed`` call.
        Non-JSON payloads and ``[DONE]`` sentinels are skipped silently.
        """
        while True:
            crlf = self._sse_buffer.find(b"\r\n\r\n")
            lf = self._sse_buffer.find(b"\n\n")
            if crlf == -1 and lf == -1:
                return
            if crlf != -1 and (lf == -1 or crlf < lf):
                sep_idx, sep_len = crlf, 4
            else:
                sep_idx, sep_len = lf, 2
            frame = bytes(self._sse_buffer[:sep_idx])
            del self._sse_buffer[: sep_idx + sep_len]
            event_dict = _parse_frame(frame)
            if event_dict is not None:
                yield _PerplexityEventEnvelope(event=event_dict)


def _parse_frame(frame: bytes) -> dict[str, Any] | None:
    """Extract the JSON payload from a single SSE frame.

    Walks lines looking for one starting with ``data:`` (per SSE spec).
    Returns ``None`` for keepalive comments, non-data frames, ``[DONE]``
    sentinels, and JSON parse failures.
    """
    for raw_line in frame.split(b"\n"):
        line = raw_line.rstrip(b"\r")
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].lstrip()
        if not payload or payload == b"[DONE]":
            return None
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None
