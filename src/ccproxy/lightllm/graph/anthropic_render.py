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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import (
    FinalResultEvent,
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
from pydantic_graph import GraphBuilder, StepContext

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelResponseStreamEvent

logger = logging.getLogger(__name__)


# ── Wire emission helpers (module-level — pure byte emitters) ──────────────


def _emit(event_name: str, body: dict[str, Any]) -> bytes:
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


def _emit_content_block_start(idx: int, part: Any) -> bytes:
    block: dict[str, Any]
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


def _tool_args_to_json_string(args_delta: str | dict[str, Any] | None) -> str | None:
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


def _emit_initial_content_deltas(idx: int, part: Any) -> bytes:
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


def _emit_content_block_delta(idx: int, delta: Any) -> bytes:
    wire_delta: dict[str, Any]
    if isinstance(delta, TextPartDelta):
        wire_delta = {"type": "text_delta", "text": delta.content_delta}
    elif isinstance(delta, ThinkingPartDelta):
        if delta.signature_delta is not None:
            wire_delta = {"type": "signature_delta", "signature": delta.signature_delta}
        elif delta.content_delta is not None:
            wire_delta = {"type": "thinking_delta", "thinking": delta.content_delta}
        else:
            logger.debug("anthropic render: empty ThinkingPartDelta; dropping")
            return b""
    elif isinstance(delta, ToolCallPartDelta):
        partial_json = _tool_args_to_json_string(delta.args_delta)
        if partial_json is None:
            logger.debug("anthropic render: ToolCallPartDelta with no args_delta; dropping")
            return b""
        wire_delta = {"type": "input_json_delta", "partial_json": partial_json}
    else:
        logger.debug("anthropic render: unknown delta type %s; dropping", type(delta).__name__)
        return b""
    return _emit(
        "content_block_delta",
        {"type": "content_block_delta", "index": idx, "delta": wire_delta},
    )


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
    stream-level lifecycle stays consistent.
    """

    message_id: str
    model: str
    started: bool = False
    open_block_index: int | None = None
    pending_events: deque[Any] = field(default_factory=deque)
    out: bytearray = field(default_factory=bytearray)


class _RenderDone:
    """Marker returned by the router when the events queue is exhausted."""


# ── Graph ──────────────────────────────────────────────────────────────────


_g: GraphBuilder[_AnthropicRenderState, None, None, bytes] = GraphBuilder(
    state_type=_AnthropicRenderState,
    output_type=bytes,
)


@_g.step
async def take_next_event(
    ctx: StepContext[_AnthropicRenderState, None, None],
) -> Any:
    """Router source: pop the next event from the queue, or signal end via :class:`_RenderDone`."""
    if not ctx.state.pending_events:
        return _RenderDone()
    return ctx.state.pending_events.popleft()


@_g.step
async def handle_part_start(
    ctx: StepContext[_AnthropicRenderState, None, PartStartEvent],
) -> None:
    """Open a new content block, closing any prior open block first."""
    event = ctx.inputs
    state = ctx.state
    if not state.started:
        state.out += _emit_message_start(state.message_id, state.model)
        state.started = True
    if state.open_block_index is not None:
        # New part start without an explicit PartEndEvent — close the previous
        # block before opening the new one. PartStartEvent.index is the IR
        # part index; we mirror it as the Anthropic block index.
        state.out += _emit_content_block_stop(state.open_block_index)
    state.out += _emit_content_block_start(event.index, event.part)
    state.open_block_index = event.index
    # If the start event already carries content (e.g. the intake collapsed an
    # empty content_block_start + the first delta into a single PartStartEvent
    # with a non-empty TextPart), emit that content as an initial delta so the
    # downstream client sees the same accumulated text.
    state.out += _emit_initial_content_deltas(event.index, event.part)


@_g.step
async def handle_part_delta(
    ctx: StepContext[_AnthropicRenderState, None, PartDeltaEvent],
) -> None:
    """Emit a ``content_block_delta`` for the open block."""
    event = ctx.inputs
    state = ctx.state
    if state.open_block_index is None:
        # Defensive: a delta without an open block can't be expressed in
        # Anthropic's wire format.
        logger.debug("anthropic render: PartDeltaEvent with no open block; dropping")
        return
    state.out += _emit_content_block_delta(event.index, event.delta)


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
    ctx.state.out = bytearray()
    return out


_g.add(
    _g.edge_from(_g.start_node).to(take_next_event),
    _g.edge_from(take_next_event).to(
        _g.decision()
        .branch(_g.match(_RenderDone).to(emit_done))
        .branch(_g.match(PartStartEvent).to(handle_part_start))
        .branch(_g.match(PartDeltaEvent).to(handle_part_delta))
        .branch(_g.match(PartEndEvent).to(handle_part_end))
        .branch(_g.match(FinalResultEvent).to(handle_final_result))
    ),
    _g.edge_from(
        handle_part_start,
        handle_part_delta,
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

    async def render(self, event: ModelResponseStreamEvent) -> bytes:
        """One IR event → zero-or-more bytes of Anthropic SSE wire output."""
        self._state.pending_events.append(event)
        result: bytes = await _render_graph.run(state=self._state)
        return result

    async def close(self) -> bytes:
        """Flush any open block, then emit ``message_delta`` + ``message_stop``.

        Imperative (no FSM): the terminator sequence is a fixed three-step
        emission with no per-event dispatch.
        """
        state = self._state
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
