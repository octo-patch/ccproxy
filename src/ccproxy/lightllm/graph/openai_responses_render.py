"""IR events → OpenAI Responses API SSE wire bytes via pydantic-graph FSM.

Listener-side render FSM for ``InboundFormat.OPENAI_RESPONSES``.
Consumes pydantic-ai :class:`ModelResponseStreamEvent` instances and
emits the OpenAI Responses streaming wire format — the per-item +
per-content-part lifecycle the Codex CLI expects.

The Responses streaming protocol is structurally richer than Chat
Completions. Each item in ``output[]`` brackets with
``response.output_item.added`` / ``response.output_item.done``;
message items further bracket their content parts with
``response.content_part.added`` / ``response.content_part.done``. Text
chunks stream via ``response.output_text.delta`` and conclude with
``response.output_text.done`` carrying the accumulated text. Function
calls stream their JSON arguments via
``response.function_call_arguments.delta``; reasoning items stream via
``response.reasoning_text.delta``. The stream prelude is a single
``response.created`` event with a Response envelope snapshot; the
postlude is ``response.completed`` with final usage.

Mirrors :mod:`ccproxy.lightllm.graph.openai_render` in shape: state is
held across :meth:`render` calls, the graph dispatches one IR event
per run, and :meth:`close` emits the imperative terminator (no FSM
dispatch — the postlude is a fixed two-event sequence).

The 56-event upstream intake FSM lives separately in
``openai_responses_intake.py`` — this module is render-only.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import (
    FinalResultEvent,
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


# ── Wire emission helpers ──────────────────────────────────────────────────


def _args_to_str(args: str | dict[str, Any] | None) -> str:
    """Coerce IR tool-call args (string fragment | dict | None) to a JSON string."""
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    return json.dumps(args, separators=(",", ":"))


def _emit_event(event_name: str, payload: dict[str, Any]) -> bytes:
    """Encode one event as a Responses SSE frame.

    Responses uses the named-event SSE form
    (``event: <name>\\ndata: <json>\\n\\n``) — same convention as
    Anthropic, distinct from OpenAI Chat Completion's data-only form.
    """
    data = json.dumps(payload, separators=(",", ":"))
    return f"event: {event_name}\ndata: {data}\n\n".encode()


# ── State ──────────────────────────────────────────────────────────────────


@dataclass
class _OpenItemState:
    """Per-item state for an open output item (message / function_call / reasoning).

    ``output_index`` is the position in ``output[]``. ``content_index`` is
    the current content part within a message item (always 0 in this
    implementation — we don't open multiple content parts per message).
    ``text_buffer`` accumulates the streamed text for the ``.done``
    event payload; ``args_buffer`` does the same for function_call
    arguments.
    """

    item_type: str
    """``"message"`` / ``"function_call"`` / ``"reasoning"``."""

    item_id: str
    """The item id emitted on ``output_item.added``."""

    output_index: int
    """Position in the response's ``output[]`` array."""

    text_buffer: str = ""
    """Accumulated text (message: output_text; reasoning: reasoning_text)."""

    args_buffer: str = ""
    """Accumulated JSON argument string for function_call items."""

    tool_name: str = ""
    """Function tool name for function_call ``.done`` events."""

    tool_call_id: str = ""
    """Function call id for function_call ``output_item.done`` events."""

    content_part_opened: bool = False
    """True after ``response.content_part.added`` was emitted (message items only)."""


@dataclass
class _OpenAIResponsesRenderState:
    """FSM state for one Responses render graph run.

    Persists across :meth:`render` calls so the stream-level lifecycle
    (sequence_number monotonicity, item open/close state, response_id)
    stays consistent. ``pending_events`` holds the single
    :class:`ModelResponseStreamEvent` pushed by :meth:`render` before
    each graph run; the FSM router pops from it. ``out`` accumulates
    SSE bytes emitted by handler steps.
    """

    response_id: str
    """``resp_<24-hex>`` — stamped on every event's response envelope (and prelude)."""

    created_at: int
    """Unix seconds — stamped in the prelude snapshot."""

    model: str
    """Model slug — stamped in the prelude snapshot."""

    sequence_number: int = 0
    """Monotonic per-event counter, reset to 0 on construction."""

    response_created_emitted: bool = False
    """Lazily emitted on the first :meth:`render` call so we know the model."""

    next_output_index: int = 0
    """Allocator for ``output_index`` on each new item."""

    part_to_output_index: dict[int, int] = field(default_factory=dict)
    """Map IR part index → output_index so deltas can address the right open item."""

    open_items: dict[int, _OpenItemState] = field(default_factory=dict)
    """Indexed by ``output_index`` so each delta/end can find its open item."""

    finish_status: str = "completed"
    """``"completed"`` / ``"incomplete"`` / ``"failed"`` — stamped in postlude."""

    pending_events: deque[Any] = field(default_factory=deque)
    """Single-event queue popped by the FSM router."""

    out: bytearray = field(default_factory=bytearray)
    """Accumulated SSE wire bytes; drained by the terminal step."""


class _RenderDone:
    """Marker returned by the router when the events queue is exhausted."""


# ── Prelude helper ─────────────────────────────────────────────────────────


def _ensure_response_created(state: _OpenAIResponsesRenderState) -> None:
    """Emit ``response.created`` lazily before the first item event.

    The Responses prelude is a single ``response.created`` event
    carrying an in-progress envelope snapshot (id, object, model,
    status, empty output[], usage:None). Codex CLI expects this to
    arrive before any per-item events.
    """
    if state.response_created_emitted:
        return
    state.response_created_emitted = True

    snapshot = _response_envelope_snapshot(state, status="in_progress")
    state.out += _emit_event(
        "response.created",
        {
            "type": "response.created",
            "response": snapshot,
            "sequence_number": state.sequence_number,
        },
    )
    state.sequence_number += 1


def _response_envelope_snapshot(
    state: _OpenAIResponsesRenderState,
    *,
    status: str,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the Response envelope snapshot stamped in prelude/postlude."""
    return {
        "id": state.response_id,
        "object": "response",
        "created_at": state.created_at,
        "status": status,
        "model": state.model,
        "output": [],
        "usage": usage,
    }


def _bump_seq(state: _OpenAIResponsesRenderState) -> int:
    """Allocate the next sequence_number and advance the counter."""
    seq = state.sequence_number
    state.sequence_number += 1
    return seq


# ── Item lifecycle helpers ─────────────────────────────────────────────────


def _open_message_item(
    state: _OpenAIResponsesRenderState,
    *,
    ir_index: int,
) -> _OpenItemState:
    """Emit ``response.output_item.added`` + ``response.content_part.added`` for a new message item.

    Codex's Codex-mode responses always carry assistant role for
    streamed text — we hardcode it here. If we ever need to render
    cross-format streams where the assistant emits as a different
    role, parametrize from the IR.
    """
    output_index = state.next_output_index
    state.next_output_index += 1
    item_id = f"msg_{uuid.uuid4().hex[:24]}"

    item = _OpenItemState(
        item_type="message",
        item_id=item_id,
        output_index=output_index,
    )
    state.open_items[output_index] = item
    state.part_to_output_index[ir_index] = output_index

    state.out += _emit_event(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "message",
                "status": "in_progress",
                "content": [],
                "role": "assistant",
            },
            "sequence_number": _bump_seq(state),
        },
    )
    state.out += _emit_event(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": "",
            },
            "sequence_number": _bump_seq(state),
        },
    )
    item.content_part_opened = True
    return item


def _open_function_call_item(
    state: _OpenAIResponsesRenderState,
    *,
    ir_index: int,
    part: ToolCallPart,
) -> _OpenItemState:
    """Emit ``response.output_item.added`` for a new function_call item."""
    output_index = state.next_output_index
    state.next_output_index += 1
    item_id = f"fc_{uuid.uuid4().hex[:24]}"

    item = _OpenItemState(
        item_type="function_call",
        item_id=item_id,
        output_index=output_index,
        tool_name=part.tool_name,
        tool_call_id=part.tool_call_id or "",
    )
    state.open_items[output_index] = item
    state.part_to_output_index[ir_index] = output_index

    state.out += _emit_event(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "function_call",
                "status": "in_progress",
                "call_id": part.tool_call_id,
                "name": part.tool_name,
                "arguments": "",
            },
            "sequence_number": _bump_seq(state),
        },
    )
    return item


def _open_reasoning_item(
    state: _OpenAIResponsesRenderState,
    *,
    ir_index: int,
) -> _OpenItemState:
    """Emit ``response.output_item.added`` for a new reasoning item."""
    output_index = state.next_output_index
    state.next_output_index += 1
    item_id = f"rs_{uuid.uuid4().hex[:24]}"

    item = _OpenItemState(
        item_type="reasoning",
        item_id=item_id,
        output_index=output_index,
    )
    state.open_items[output_index] = item
    state.part_to_output_index[ir_index] = output_index

    state.out += _emit_event(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "reasoning",
                "status": "in_progress",
                "summary": [],
                "content": [],
            },
            "sequence_number": _bump_seq(state),
        },
    )
    return item


def _close_item(
    state: _OpenAIResponsesRenderState,
    item: _OpenItemState,
) -> None:
    """Emit the per-type ``.done`` events plus ``output_item.done`` for an open item."""
    if item.item_type == "message":
        if item.content_part_opened:
            state.out += _emit_event(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "content_index": 0,
                    "text": item.text_buffer,
                    "logprobs": [],
                    "sequence_number": _bump_seq(state),
                },
            )
            state.out += _emit_event(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": item.text_buffer,
                    },
                    "sequence_number": _bump_seq(state),
                },
            )
        state.out += _emit_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": item.output_index,
                "item": {
                    "id": item.item_id,
                    "type": "message",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "annotations": [],
                            "logprobs": [],
                            "text": item.text_buffer,
                        }
                    ],
                    "role": "assistant",
                },
                "sequence_number": _bump_seq(state),
            },
        )
    elif item.item_type == "function_call":
        state.out += _emit_event(
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "item_id": item.item_id,
                "output_index": item.output_index,
                "name": item.tool_name,
                "arguments": item.args_buffer,
                "sequence_number": _bump_seq(state),
            },
        )
        state.out += _emit_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": item.output_index,
                "item": {
                    "id": item.item_id,
                    "type": "function_call",
                    "status": "completed",
                    "call_id": item.tool_call_id,
                    "name": item.tool_name,
                    "arguments": item.args_buffer,
                },
                "sequence_number": _bump_seq(state),
            },
        )
    elif item.item_type == "reasoning":
        state.out += _emit_event(
            "response.reasoning_text.done",
            {
                "type": "response.reasoning_text.done",
                "item_id": item.item_id,
                "output_index": item.output_index,
                "content_index": 0,
                "text": item.text_buffer,
                "sequence_number": _bump_seq(state),
            },
        )
        state.out += _emit_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": item.output_index,
                "item": {
                    "id": item.item_id,
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [],
                    "content": [
                        {
                            "type": "reasoning_text",
                            "text": item.text_buffer,
                        }
                    ],
                },
                "sequence_number": _bump_seq(state),
            },
        )


# ── Graph ──────────────────────────────────────────────────────────────────


_g: GraphBuilder[_OpenAIResponsesRenderState, None, None, bytes] = GraphBuilder(
    state_type=_OpenAIResponsesRenderState,
    output_type=bytes,
)


@_g.step
async def take_next_event(
    ctx: StepContext[_OpenAIResponsesRenderState, None, None],
) -> Any:
    """Router source: pop the next event from the queue, or signal end via :class:`_RenderDone`."""
    if not ctx.state.pending_events:
        return _RenderDone()
    return ctx.state.pending_events.popleft()


@_g.step
async def handle_part_start(
    ctx: StepContext[_OpenAIResponsesRenderState, None, PartStartEvent],
) -> None:
    """Open a new output item for the incoming IR part."""
    event = ctx.inputs
    state = ctx.state
    _ensure_response_created(state)

    part = event.part
    if isinstance(part, TextPart):
        item = _open_message_item(state, ir_index=event.index)
        if part.content:
            item.text_buffer += part.content
            state.out += _emit_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "content_index": 0,
                    "delta": part.content,
                    "logprobs": [],
                    "sequence_number": _bump_seq(state),
                },
            )
        return

    if isinstance(part, ToolCallPart):
        item = _open_function_call_item(state, ir_index=event.index, part=part)
        args_str = _args_to_str(part.args)
        if args_str:
            item.args_buffer += args_str
            state.out += _emit_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "delta": args_str,
                    "sequence_number": _bump_seq(state),
                },
            )
        return

    if isinstance(part, ThinkingPart):
        item = _open_reasoning_item(state, ir_index=event.index)
        if part.content:
            item.text_buffer += part.content
            state.out += _emit_event(
                "response.reasoning_text.delta",
                {
                    "type": "response.reasoning_text.delta",
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "content_index": 0,
                    "delta": part.content,
                    "sequence_number": _bump_seq(state),
                },
            )
        return

    # Other part kinds (NativeToolCall*, CompactionPart, FilePart) have no
    # current Responses wire surface — silently no-op.


@_g.step
async def handle_part_delta(
    ctx: StepContext[_OpenAIResponsesRenderState, None, PartDeltaEvent],
) -> None:
    """Emit a delta event for the matching open item."""
    event = ctx.inputs
    state = ctx.state
    delta = event.delta

    output_index = state.part_to_output_index.get(event.index)
    if output_index is None:
        # PartDelta arrived before PartStart — likely an upstream FSM that
        # streams deltas without a prior start event. Open a message item
        # lazily for text deltas; tool_call deltas open a function_call.
        _ensure_response_created(state)
        if isinstance(delta, TextPartDelta):
            item = _open_message_item(state, ir_index=event.index)
        elif isinstance(delta, ToolCallPartDelta):
            synthetic = ToolCallPart(
                tool_name=delta.tool_name_delta or "",
                args=delta.args_delta if isinstance(delta.args_delta, str | dict) else None,
                tool_call_id=delta.tool_call_id or "",
            )
            item = _open_function_call_item(state, ir_index=event.index, part=synthetic)
        elif isinstance(delta, ThinkingPartDelta):
            item = _open_reasoning_item(state, ir_index=event.index)
        else:
            return
        output_index = item.output_index

    item = state.open_items[output_index]

    if isinstance(delta, TextPartDelta):
        if delta.content_delta:
            item.text_buffer += delta.content_delta
            state.out += _emit_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "content_index": 0,
                    "delta": delta.content_delta,
                    "logprobs": [],
                    "sequence_number": _bump_seq(state),
                },
            )
        return

    if isinstance(delta, ToolCallPartDelta):
        args_str = _args_to_str(delta.args_delta)
        if args_str:
            item.args_buffer += args_str
            state.out += _emit_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "delta": args_str,
                    "sequence_number": _bump_seq(state),
                },
            )
        return

    if isinstance(delta, ThinkingPartDelta):
        text_delta = delta.content_delta
        if text_delta:
            item.text_buffer += text_delta
            state.out += _emit_event(
                "response.reasoning_text.delta",
                {
                    "type": "response.reasoning_text.delta",
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "content_index": 0,
                    "delta": text_delta,
                    "sequence_number": _bump_seq(state),
                },
            )
        return


@_g.step
async def handle_part_end(
    ctx: StepContext[_OpenAIResponsesRenderState, None, PartEndEvent],
) -> None:
    """Close the matching open item — emit its per-type ``.done`` plus ``output_item.done``."""
    event = ctx.inputs
    state = ctx.state
    output_index = state.part_to_output_index.get(event.index)
    if output_index is None:
        return
    item = state.open_items.pop(output_index, None)
    if item is None:
        return
    _close_item(state, item)


@_g.step
async def handle_final_result(
    ctx: StepContext[_OpenAIResponsesRenderState, None, FinalResultEvent],
) -> None:
    """No-op: ``FinalResultEvent`` is an internal agent-loop signal with no Responses wire equivalent."""
    del ctx


@_g.step
async def emit_done(
    ctx: StepContext[_OpenAIResponsesRenderState, None, _RenderDone],
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


class OpenAIResponsesRenderFSM:
    """Async pydantic-graph-driven OpenAI Responses SSE renderer.

    One :meth:`render` call dispatches one
    :class:`ModelResponseStreamEvent` through the FSM and returns the
    emitted SSE bytes. :meth:`close` is imperative — it closes any
    still-open items, then emits the fixed ``response.completed``
    terminator.
    """

    name = "openai_responses"

    def __init__(self, *, model: str = "unknown") -> None:
        self._state = _OpenAIResponsesRenderState(
            response_id=f"resp_{uuid.uuid4().hex[:24]}",
            created_at=int(time.time()),
            model=model,
        )

    async def render(self, event: ModelResponseStreamEvent) -> bytes:
        """One IR event → zero-or-more bytes of OpenAI Responses SSE wire output."""
        self._state.pending_events.append(event)
        result: bytes = await _render_graph.run(state=self._state)
        return result

    async def close(self) -> bytes:
        """Close any still-open items, then emit ``response.completed``."""
        state = self._state
        out = bytearray()

        # Drain any items left open (the upstream FSM may not have emitted
        # PartEndEvent for every open part if the stream cut short).
        for output_index in sorted(state.open_items.keys()):
            item = state.open_items.pop(output_index)
            saved_out = state.out
            state.out = out
            _close_item(state, item)
            state.out = saved_out

        # Postlude — response.completed with the final envelope snapshot.
        snapshot = _response_envelope_snapshot(
            state,
            status=state.finish_status,
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        out += _emit_event(
            "response.completed",
            {
                "type": "response.completed",
                "response": snapshot,
                "sequence_number": _bump_seq(state),
            },
        )
        return bytes(out)
