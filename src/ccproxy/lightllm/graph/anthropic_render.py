"""IR events → Anthropic Messages SSE wire bytes via pydantic-graph FSM.

Pydantic-graph FSM port of
:class:`ccproxy.lightllm.response.render_anthropic.AnthropicResponseRender`.
One graph run per :meth:`AnthropicResponseRenderFSM.render` call: the single
:class:`ModelResponseStreamEvent` is pushed onto an in-state queue, the FSM
router drains the queue dispatching the event to a per-variant handler step,
and a terminal step pulls the accumulated SSE bytes out of state.

The behavioral contract matches
:mod:`ccproxy.lightllm.response.render_anthropic` byte-for-byte: same
``message_start`` synthesis, same ``content_block_*`` lifecycle (closing a
prior open block when a new ``PartStartEvent`` arrives without an intervening
``PartEndEvent``), same initial-content delta replay for parts that arrive
already populated, same delta-variant dispatch (``text_delta`` /
``thinking_delta`` / ``signature_delta`` / ``input_json_delta``).

:meth:`close` is intentionally imperative — the terminator sequence (flush
open block, ensure ``message_start`` for empty streams, emit ``message_delta``
+ ``message_stop``) is fixed and doesn't benefit from FSM dispatch.

The persistent-loop bridge between sync mitmproxy callables and this async
FSM lives in :class:`SSEPipeline` (Phase Q). For tests, the parametrize
fixture in ``tests/test_lightllm_response_render_anthropic.py`` wraps the
async FSM in a one-loop-per-call sync adapter.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic_ai.messages import (
    FinalResultEvent,
    ModelResponsePart,
    ModelResponsePartDelta,
    ModelResponseStreamEvent,
    NativeToolCallPart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_graph import GraphBuilder, StepContext, TypeExpression

import ccproxy.lightllm.graph._subgraph_patch  # noqa: F401  -- installs GraphBuilder.add_subgraph

logger = logging.getLogger(__name__)


# ── Wire emission helpers (module-level — pure byte emitters) ──────────────


def _emit(event_name: str, body: Mapping[str, object]) -> bytes:
    return f"event: {event_name}\ndata: {json.dumps(body, separators=(',', ':'))}\n\n".encode()


def _emit_message_start(message_id: str, model: str) -> bytes:
    return _emit(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )


def _emit_content_block_start(idx: int, part: ModelResponsePart) -> bytes:
    block: dict[str, object]
    if isinstance(part, TextPart):
        block = {"type": "text", "text": ""}
    elif isinstance(part, ThinkingPart):
        if part.id == "redacted_thinking":
            # Anthropic redacted_thinking carries the opaque payload in `data`;
            # pydantic-ai stashes that on the part's `signature` field.
            block = {"type": "redacted_thinking", "data": part.signature or ""}
        else:
            block = {"type": "thinking", "thinking": "", "signature": ""}
    elif isinstance(part, ToolCallPart | NativeToolCallPart):
        block = {
            "type": "tool_use",
            "id": part.tool_call_id,
            "name": part.tool_name,
            "input": {},
        }
    else:
        # CompactionPart, FilePart, builtin-tool-return variants: no clean
        # Anthropic-streaming wire mapping; emit an empty text block so the
        # envelope stays well-formed.
        logger.debug(
            "anthropic render: no wire mapping for part %s; emitting empty text block",
            type(part).__name__,
        )
        block = {"type": "text", "text": ""}
    return _emit(
        "content_block_start",
        {"type": "content_block_start", "index": idx, "content_block": block},
    )


def _tool_args_to_json_string(args_delta: str | Mapping[str, object] | None) -> str | None:
    """Serialize a ``ToolCallPartDelta.args_delta`` to the wire ``partial_json`` shape.

    On the Anthropic wire ``input_json_delta.partial_json`` is always a string —
    the partially-arrived JSON. If the IR carries a dict (because the upstream
    intake already merged accumulated deltas), JSON-encode it.
    """
    if args_delta is None:
        return None
    if isinstance(args_delta, str):
        return args_delta
    return json.dumps(args_delta, separators=(",", ":"))


def _emit_initial_content_deltas(idx: int, part: ModelResponsePart) -> bytes:
    """Emit deltas for any non-empty content carried by a starting part.

    The intake collapses an Anthropic ``content_block_start`` whose initial
    content is non-empty (text/thinking) directly into a ``PartStartEvent``
    with that content already populated. On the wire, the equivalent
    Anthropic events are ``content_block_start`` (empty) + a single
    ``content_block_delta`` (with the initial value). Replay the deltas so
    the rendered stream preserves the full content.
    """
    out = bytearray()
    if isinstance(part, TextPart) and part.content:
        out += _emit(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "text_delta", "text": part.content},
            },
        )
    elif isinstance(part, ThinkingPart) and part.id != "redacted_thinking":
        if part.content:
            out += _emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "thinking_delta", "thinking": part.content},
                },
            )
        if part.signature:
            out += _emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "signature_delta", "signature": part.signature},
                },
            )
    elif isinstance(part, ToolCallPart | NativeToolCallPart):
        partial_json = _tool_args_to_json_string(part.args)
        if partial_json:
            out += _emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                },
            )
    return bytes(out)


def _emit_content_block_stop(idx: int) -> bytes:
    return _emit("content_block_stop", {"type": "content_block_stop", "index": idx})


def _emit_message_delta() -> bytes:
    return _emit(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 0},
        },
    )


def _emit_message_stop() -> bytes:
    return _emit("message_stop", {"type": "message_stop"})


# ── State ──────────────────────────────────────────────────────────────────


@dataclass
class _AnthropicRenderState:
    """FSM state for one Anthropic render graph run.

    The ``pending_events`` queue holds the single :class:`ModelResponseStreamEvent`
    pushed by :meth:`AnthropicResponseRenderFSM.render` before each graph run; the
    FSM router pops from it. ``out`` accumulates the SSE wire bytes emitted by
    handler steps; the terminal step returns ``bytes(out)`` and resets the buffer
    so the same state can drive the next render call. ``message_id``, ``model``,
    ``started``, and ``open_block_index`` persist across render calls so the
    stream-level lifecycle stays consistent. ``current_ir_index`` is a transient
    scratch field written by the inner-subgraph open steps.
    """

    message_id: str
    model: str
    started: bool = False
    open_block_index: int | None = None
    current_ir_index: int = 0
    """Transient: the IR part/delta index, stashed by the inner-subgraph open step."""
    pending_events: deque[ModelResponseStreamEvent] = field(default_factory=deque)
    out: bytearray = field(default_factory=bytearray)

    # ── Telemetry (never-silently-drop diagnostics) ───────────────────────────
    events_received: int = 0
    """Total IR events dispatched through the render FSM (across all render calls)."""

    bytes_emitted: int = 0
    """Total SSE bytes emitted by render steps (excludes the close terminator)."""


class _RenderDone:
    """Marker returned by the router when the events queue is exhausted."""


class _NoDelta:
    """Marker returned by ``open_delta`` when there is no open block to target."""


# ── Graph ──────────────────────────────────────────────────────────────────


_g: GraphBuilder[_AnthropicRenderState, None, None, bytes] = GraphBuilder(
    name="anthropic_render",
    state_type=_AnthropicRenderState,
    output_type=bytes,
)


@_g.step
async def take_next_event(
    ctx: StepContext[_AnthropicRenderState, None, None],
) -> ModelResponseStreamEvent | _RenderDone:
    """Router source: pop the next event from the queue, or signal end via :class:`_RenderDone`."""
    if not ctx.state.pending_events:
        return _RenderDone()
    ctx.state.events_received += 1
    return ctx.state.pending_events.popleft()


# ── Inner subgraph: part_start dispatch ───────────────────────────────────
#
# ``handle_part_start`` previously used an isinstance ladder on ``event.part``.
# This subgraph replaces that: ``open_part`` stashes ``event.index`` on state
# (needed by all leaf handlers), ensures ``message_start`` is emitted, closes
# any prior open block, then returns ``event.part`` to the decision fan-out.

_psg: GraphBuilder[_AnthropicRenderState, None, PartStartEvent, None] = GraphBuilder(
    name="anthropic_render_part_start",
    state_type=_AnthropicRenderState,
    input_type=PartStartEvent,
)


@_psg.step
async def open_part(
    ctx: StepContext[_AnthropicRenderState, None, PartStartEvent],
) -> ModelResponsePart:
    """Stash the IR index, emit ``message_start`` if needed, close any prior block, return the part."""
    event = ctx.inputs
    state = ctx.state
    state.current_ir_index = event.index
    if not state.started:
        state.out += _emit_message_start(state.message_id, state.model)
        state.started = True
    if state.open_block_index is not None:
        # New part start without an explicit PartEndEvent — close the previous
        # block before opening the new one.
        state.out += _emit_content_block_stop(state.open_block_index)
    state.out += _emit_content_block_start(event.index, event.part)
    state.open_block_index = event.index
    return event.part


@_psg.step
async def handle_text_part_start(ctx: StepContext[_AnthropicRenderState, None, TextPart]) -> None:
    """``TextPart`` — emit initial content delta if the part arrived pre-populated."""
    state = ctx.state
    state.out += _emit_initial_content_deltas(state.current_ir_index, ctx.inputs)


@_psg.step
async def handle_thinking_part_start(ctx: StepContext[_AnthropicRenderState, None, ThinkingPart]) -> None:
    """``ThinkingPart`` — emit initial thinking/signature deltas if pre-populated."""
    state = ctx.state
    state.out += _emit_initial_content_deltas(state.current_ir_index, ctx.inputs)


@_psg.step
async def handle_tool_call_part_start(ctx: StepContext[_AnthropicRenderState, None, ToolCallPart]) -> None:
    """``ToolCallPart`` — emit initial args delta if pre-populated."""
    state = ctx.state
    state.out += _emit_initial_content_deltas(state.current_ir_index, ctx.inputs)


@_psg.step
async def handle_native_tool_call_part_start(
    ctx: StepContext[_AnthropicRenderState, None, NativeToolCallPart],
) -> None:
    """``NativeToolCallPart`` — emit initial args delta if pre-populated."""
    state = ctx.state
    state.out += _emit_initial_content_deltas(state.current_ir_index, ctx.inputs)


@_psg.step
async def handle_unknown_part_start(ctx: StepContext[_AnthropicRenderState, None, object]) -> None:
    """Catch-all for part types with no handler — log instead of silently dropping."""
    logger.debug("anthropic render: unhandled part type %s in part_start; skipping", type(ctx.inputs).__name__)


_psg.add(
    _psg.edge_from(_psg.start_node).to(open_part),
    _psg.edge_from(open_part).to(
        _psg.decision()
        .branch(_psg.match(TextPart).to(handle_text_part_start))
        .branch(_psg.match(ThinkingPart).to(handle_thinking_part_start))
        .branch(_psg.match(ToolCallPart).to(handle_tool_call_part_start))
        .branch(_psg.match(NativeToolCallPart).to(handle_native_tool_call_part_start))
        .branch(_psg.match(TypeExpression[object]).to(handle_unknown_part_start))
    ),
    _psg.edge_from(
        handle_text_part_start,
        handle_thinking_part_start,
        handle_tool_call_part_start,
        handle_native_tool_call_part_start,
        handle_unknown_part_start,
    ).to(_psg.end_node),
)

_part_start_graph = _psg.build()
_dispatch_part_start = _g.add_subgraph(_part_start_graph, label="part_start")  # ty: ignore[unresolved-attribute]


# ── Inner subgraph: part_delta dispatch ───────────────────────────────────
#
# ``handle_part_delta`` previously used an isinstance ladder on ``event.delta``.
# This subgraph replaces that: ``open_delta`` stashes ``event.index`` on state
# (needed by the leaf handlers to address the open block), validates the open
# block is present, then returns ``event.delta`` to the decision fan-out.

_pdg: GraphBuilder[_AnthropicRenderState, None, PartDeltaEvent, None] = GraphBuilder(
    name="anthropic_render_part_delta",
    state_type=_AnthropicRenderState,
    input_type=PartDeltaEvent,
)


@_pdg.step
async def open_delta(
    ctx: StepContext[_AnthropicRenderState, None, PartDeltaEvent],
) -> ModelResponsePartDelta | _NoDelta:
    """Stash the IR index; return the delta for dispatch, or :class:`_NoDelta` if no open block."""
    state = ctx.state
    state.current_ir_index = ctx.inputs.index
    if state.open_block_index is None:
        logger.debug("anthropic render: PartDeltaEvent with no open block; dropping")
        return _NoDelta()
    return ctx.inputs.delta


@_pdg.step
async def handle_text_part_delta(ctx: StepContext[_AnthropicRenderState, None, TextPartDelta]) -> None:
    """``text_delta`` — emit a ``content_block_delta`` with ``text_delta`` payload."""
    state = ctx.state
    wire_delta: dict[str, object] = {"type": "text_delta", "text": ctx.inputs.content_delta}
    state.out += _emit(
        "content_block_delta",
        {"type": "content_block_delta", "index": state.current_ir_index, "delta": wire_delta},
    )


@_pdg.step
async def handle_thinking_part_delta(ctx: StepContext[_AnthropicRenderState, None, ThinkingPartDelta]) -> None:
    """``thinking_delta`` / ``signature_delta`` — emit the appropriate ``content_block_delta``."""
    state = ctx.state
    delta = ctx.inputs
    wire_delta: dict[str, object]
    if delta.signature_delta is not None:
        wire_delta = {"type": "signature_delta", "signature": delta.signature_delta}
    elif delta.content_delta is not None:
        wire_delta = {"type": "thinking_delta", "thinking": delta.content_delta}
    else:
        logger.debug("anthropic render: empty ThinkingPartDelta; dropping")
        return
    state.out += _emit(
        "content_block_delta",
        {"type": "content_block_delta", "index": state.current_ir_index, "delta": wire_delta},
    )


@_pdg.step
async def handle_tool_call_part_delta(ctx: StepContext[_AnthropicRenderState, None, ToolCallPartDelta]) -> None:
    """``input_json_delta`` — emit partial tool-call JSON args."""
    state = ctx.state
    partial_json = _tool_args_to_json_string(ctx.inputs.args_delta)
    if partial_json is None:
        logger.debug("anthropic render: ToolCallPartDelta with no args_delta; dropping")
        return
    wire_delta: dict[str, object] = {"type": "input_json_delta", "partial_json": partial_json}
    state.out += _emit(
        "content_block_delta",
        {"type": "content_block_delta", "index": state.current_ir_index, "delta": wire_delta},
    )


@_pdg.step
async def handle_no_open_block(ctx: StepContext[_AnthropicRenderState, None, _NoDelta]) -> None:
    """No-op terminal for the guard path when there is no open block."""
    del ctx  # protocol-required parameter; intentionally unused


@_pdg.step
async def handle_unknown_part_delta(ctx: StepContext[_AnthropicRenderState, None, object]) -> None:
    """Catch-all for delta types with no handler — log instead of silently dropping."""
    logger.debug("anthropic render: unknown delta type %s; dropping", type(ctx.inputs).__name__)


_pdg.add(
    _pdg.edge_from(_pdg.start_node).to(open_delta),
    _pdg.edge_from(open_delta).to(
        _pdg.decision()
        .branch(_pdg.match(_NoDelta).to(handle_no_open_block))
        .branch(_pdg.match(TextPartDelta).to(handle_text_part_delta))
        .branch(_pdg.match(ThinkingPartDelta).to(handle_thinking_part_delta))
        .branch(_pdg.match(ToolCallPartDelta).to(handle_tool_call_part_delta))
        .branch(_pdg.match(TypeExpression[object]).to(handle_unknown_part_delta))
    ),
    _pdg.edge_from(
        handle_no_open_block,
        handle_text_part_delta,
        handle_thinking_part_delta,
        handle_tool_call_part_delta,
        handle_unknown_part_delta,
    ).to(_pdg.end_node),
)

_part_delta_graph = _pdg.build()
_dispatch_part_delta = _g.add_subgraph(_part_delta_graph, label="part_delta")  # ty: ignore[unresolved-attribute]


@_g.step
async def handle_part_end(
    ctx: StepContext[_AnthropicRenderState, None, PartEndEvent],
) -> None:
    """Close the open block."""
    event = ctx.inputs
    state = ctx.state
    if state.open_block_index is None:
        return
    state.out += _emit_content_block_stop(event.index)
    state.open_block_index = None


@_g.step
async def handle_final_result(
    ctx: StepContext[_AnthropicRenderState, None, FinalResultEvent],
) -> None:
    """No-op: ``FinalResultEvent`` is an internal agent-loop signal with no Anthropic wire equivalent."""
    del ctx  # protocol-required parameter; intentionally unused


@_g.step
async def emit_done(
    ctx: StepContext[_AnthropicRenderState, None, _RenderDone],
) -> bytes:
    """Terminal step — drain the accumulated wire bytes and reset for the next render call."""
    out = bytes(ctx.state.out)
    ctx.state.bytes_emitted += len(out)
    ctx.state.out = bytearray()
    return out


_g.add(
    _g.edge_from(_g.start_node).to(take_next_event),
    _g.edge_from(take_next_event).to(
        _g.decision()
        .branch(_g.match(_RenderDone).to(emit_done))
        .branch(_g.match(PartStartEvent).to(_dispatch_part_start))
        .branch(_g.match(PartDeltaEvent).to(_dispatch_part_delta))
        .branch(_g.match(PartEndEvent).to(handle_part_end))
        .branch(_g.match(FinalResultEvent).to(handle_final_result))
    ),
    _g.edge_from(
        _dispatch_part_start,
        _dispatch_part_delta,
        handle_part_end,
        handle_final_result,
    ).to(take_next_event),
    _g.edge_from(emit_done).to(_g.end_node),
)


_render_graph = _g.build()


# ── Public class ───────────────────────────────────────────────────────────


class AnthropicResponseRenderFSM:
    """Async pydantic-graph-driven Anthropic Messages SSE renderer.

    Behavioral twin of
    :class:`ccproxy.lightllm.response.render_anthropic.AnthropicResponseRender`,
    re-expressed as a :mod:`pydantic_graph.beta` ``GraphBuilder`` FSM. One graph
    run per :meth:`render` call drives a single
    :class:`ModelResponseStreamEvent` through the per-variant dispatch ladder
    and returns the emitted SSE bytes. :meth:`close` is imperative — the
    terminator sequence (flush open block, ensure ``message_start`` for empty
    streams, emit ``message_delta`` + ``message_stop``) is fixed.

    State machine tracking one open content block at a time, mirroring the
    Anthropic streaming protocol's ``content_block_start`` /
    ``content_block_delta`` / ``content_block_stop`` envelope.
    """

    name = "anthropic_messages"

    def __init__(self, *, model: str = "unknown") -> None:
        self._state = _AnthropicRenderState(
            message_id=f"msg_{uuid.uuid4().hex[:24]}",
            model=model,
        )

    @property
    def state(self) -> _AnthropicRenderState:
        """Expose FSM state for tests and telemetry inspection."""
        return self._state

    async def render(self, event: ModelResponseStreamEvent) -> bytes:
        """One IR event → zero-or-more bytes of Anthropic SSE wire output."""
        self._state.pending_events.append(event)
        result: bytes = await _render_graph.run(state=self._state)
        return result

    async def close(self) -> bytes:
        """Flush any open block, then emit ``message_delta`` + ``message_stop``.

        Imperative (no FSM): the terminator sequence is a fixed three-step
        emission with no per-event dispatch. Emits a telemetry warning when IR
        events arrived but no content bytes were rendered — a silent empty
        Anthropic response must be explainable from logs.
        """
        state = self._state
        if state.events_received and not state.bytes_emitted:
            logger.warning(
                "anthropic render received %d IR event(s) but emitted NO content bytes "
                "before close — every event mapped to a no-op wire surface",
                state.events_received,
            )
        out = bytearray()
        if state.open_block_index is not None:
            out += _emit_content_block_stop(state.open_block_index)
            state.open_block_index = None
        if not state.started:
            # Empty stream — still emit a valid envelope so the client sees a
            # parseable response.
            out += _emit_message_start(state.message_id, state.model)
            state.started = True
        out += _emit_message_delta()
        out += _emit_message_stop()
        return bytes(out)
