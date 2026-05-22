"""Perplexity Pro SSE bytes → pydantic-ai IR events via FSM.

Pydantic-graph FSM port of
:class:`ccproxy.lightllm.response.intake_perplexity.PerplexityResponseIntake`.
One graph run per :meth:`PerplexityResponseIntakeFSM.feed` call: bytes are
appended to the SSE buffer, complete SSE frames are drained, each frame's
``data:`` payload is JSON-decoded into an event dict, wrapped in a
:class:`_PerplexityEventEnvelope`, and pushed onto an in-state queue. The
FSM router drains the queue dispatching each envelope to
:func:`handle_event_chunk`, which performs identifier capture, walks the
``event.text`` JSON mirror (when no ``plan_block`` is present), walks the
``blocks[]`` for reasoning + answer deltas, emits IR events via the
``ModelResponsePartsManager``, and accumulates them into
``state.out_events``.

Unlike Anthropic's string-discriminated SSE union, Perplexity's wire is a
single JSON-event-per-frame shape with optional ``blocks``, ``text``, and
top-level identifier fields. Every event flows through the same handler
step; the four documented patch modes (Mode A root cumulative, Mode B
chunks-array, Mode C ``/chunks/N`` append, Mode D ``/markdown_block``) are
handled inline by :meth:`_apply_markdown_patch`. See ``docs/pplx.md`` for
the full wire-format reference.

The behavioral contract matches
:mod:`ccproxy.lightllm.response.intake_perplexity` byte-for-byte: same SSE
framing rules (``\\r\\n\\r\\n`` and ``\\n\\n`` separators, ``[DONE]``
silently ignored, ``data:``-prefix only), same prefix-diff semantics on
answer and reasoning, same ``ask_text`` skip filter, same step
deduplication via ``seen_step_uuids``, same ``RESEARCH_CLARIFYING_QUESTIONS``
silent suppression (the request-side surfaces it as a 400; intake's role is
emission only), same unknown-``intended_usage`` DEBUG dedup.

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


class _FeedDone:
    """Marker returned by the router when the events queue is exhausted."""


# ── State ──────────────────────────────────────────────────────────────────


@dataclass
class _PerplexityIntakeState:
    """FSM state for one Perplexity intake graph run.

    The ``events_queue`` is the queue of dispatch envelopes drained from the
    SSE buffer *before* the graph run starts; the FSM router pops from it.
    The ``out_events`` list accumulates :class:`ModelResponseStreamEvent`
    instances emitted by the handler step; the terminal step returns it.

    The streaming state fields (``answer_seen``, ``reasoning_seen``, ``ids``,
    etc.) persist across feed calls so prefix-diffing and identifier capture
    work over the whole stream.
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


# ── Helpers (called from the FSM step body) ────────────────────────────────


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


def _dispatch_one_event(state: _PerplexityIntakeState, event: dict[str, Any]) -> None:
    """Apply one Perplexity SSE event to ``state``; emit IR events into ``state.out_events``.

    Mirrors :meth:`PerplexityResponseIntake._dispatch_event` byte-for-byte.
    """
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

    reasoning_delta = ""
    answer_delta = ""

    # event.text mirror: walked only when no plan_block exists (avoids
    # double-emission against the structured channel). Clarifying questions
    # are silently suppressed here — the standalone Perplexity request
    # surface owns the 400 escalation.
    text = event.get("text")
    has_plan_block = any(isinstance(b.get("plan_block"), dict) for b in blocks)
    if isinstance(text, str):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list) and not has_plan_block:
            for step in parsed:
                if not isinstance(step, dict):
                    continue
                if step.get("step_type") == "RESEARCH_CLARIFYING_QUESTIONS":
                    continue
                rendered = _consume_step(state, step)
                if rendered:
                    reasoning_delta += rendered

    for block in blocks:
        intended_usage = block.get("intended_usage")

        if intended_usage in ("pro_search_steps", "plan", "reasoning_plan_block"):
            plan_block = block.get("plan_block") or {}
            if isinstance(plan_block, dict):
                goals = plan_block.get("goals") or []
                if isinstance(goals, list):
                    for goal in goals:
                        if not isinstance(goal, dict):
                            continue
                        desc = goal.get("description")
                        if isinstance(desc, str) and desc.startswith(state.reasoning_seen):
                            new = desc[len(state.reasoning_seen) :]
                            if new:
                                reasoning_delta += new
                                state.reasoning_seen = desc

                for step in plan_block.get("steps") or []:
                    if not isinstance(step, dict):
                        continue
                    rendered = _consume_step(state, step)
                    if rendered:
                        reasoning_delta += rendered

        # Bare ``markdown_block`` (no ``diff_block`` wrapper) — the terminal
        # event re-sends the full answer this way. Prefix-diff against
        # ``answer_seen`` surfaces any tail text not seen in earlier patches.
        mb = block.get("markdown_block")
        if isinstance(mb, dict) and not block.get("diff_block") and intended_usage != "ask_text":
            answer_str = mb.get("answer")
            if isinstance(answer_str, str) and answer_str and answer_str.startswith(state.answer_seen):
                bare_delta = answer_str[len(state.answer_seen) :]
                if bare_delta:
                    answer_delta += bare_delta
                state.answer_seen = answer_str

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
            continue

        # The ``ask_text`` block duplicates ``ask_text_0_markdown``'s
        # patches; processing both would double every chunk. Markdown wins.
        if intended_usage == "ask_text":
            continue

        field_name = diff_block.get("field")
        patches = diff_block.get("patches") or []
        if not isinstance(patches, list):
            continue

        for patch in patches:
            if not isinstance(patch, dict):
                continue
            path = patch.get("path", "")
            value = patch.get("value")

            if path.startswith("/goals"):
                if isinstance(value, str) and value.startswith(state.reasoning_seen):
                    new = value[len(state.reasoning_seen) :]
                    if new:
                        reasoning_delta += new
                        state.reasoning_seen = value
                continue

            if path == "/progress":
                continue

            if field_name != "markdown_block":
                continue

            delta = _apply_markdown_patch(state, path, value)
            if delta:
                answer_delta += delta

    if reasoning_delta:
        state.out_events.extend(
            state.parts_manager.handle_thinking_delta(
                vendor_part_id=_REASONING_VENDOR_ID,
                content=reasoning_delta,
            )
        )

    if answer_delta:
        state.out_events.extend(
            state.parts_manager.handle_text_delta(
                vendor_part_id=_ANSWER_VENDOR_ID,
                content=answer_delta,
            )
        )


# ── Graph ──────────────────────────────────────────────────────────────────


_g: GraphBuilder[
    _PerplexityIntakeState, None, None, list[ModelResponseStreamEvent]
] = GraphBuilder(
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


@_g.step
async def handle_event_chunk(
    ctx: StepContext[_PerplexityIntakeState, None, _PerplexityEventEnvelope],
) -> None:
    """Dispatch one Perplexity SSE event to the parts manager."""
    _dispatch_one_event(ctx.state, ctx.inputs.event)


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
        .branch(_g.match(_PerplexityEventEnvelope).to(handle_event_chunk))
    ),
    _g.edge_from(handle_event_chunk).to(frame_next_event),
    _g.edge_from(emit_done).to(_g.end_node),
)


_intake_graph = _g.build()


# ── Public class ───────────────────────────────────────────────────────────


class PerplexityResponseIntakeFSM:
    """Async pydantic-graph-driven Perplexity Pro SSE intake.

    Behavioral twin of
    :class:`ccproxy.lightllm.response.intake_perplexity.PerplexityResponseIntake`,
    re-expressed as a :mod:`pydantic_graph.beta` ``GraphBuilder`` FSM. One
    graph run per :meth:`feed` call drains all complete SSE frames buffered
    by that call into typed dispatch envelopes, dispatches each to the
    handler step, and returns the accumulated IR events. Partial frames
    remain in the SSE buffer for the next call. ``parts_manager`` and the
    stream-level state (``answer_seen``, ``reasoning_seen``, ``ids``, etc.)
    persist across calls.
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
