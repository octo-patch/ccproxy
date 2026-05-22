"""IR events → OpenAI Chat Completion SSE wire bytes via pydantic-graph FSM.

Pydantic-graph FSM port of
:class:`ccproxy.lightllm.response.render_openai.OpenAIResponseRender`. One
graph run per :meth:`OpenAIResponseRenderFSM.render` call: the single
:class:`ModelResponseStreamEvent` is pushed onto an in-state queue, the FSM
router drains the queue dispatching the event to a per-variant handler step,
and a terminal step pulls the accumulated SSE bytes out of state.

The behavioral contract matches
:mod:`ccproxy.lightllm.response.render_openai` byte-for-byte: same chunk id
envelope (``chatcmpl-<24-hex>``), same lazy role chunk, same content / tool_call
delta dispatch, same IR-part-index → OpenAI-tool-call-index allocator, same
finish reason tracking, same ``[DONE]`` terminator.

OpenAI Chat Completion SSE is structurally simpler than Anthropic's: no per-
block lifecycle, no ``content_block_start``/``stop`` envelope. Each chunk is
a partial update to a single linear assistant message. :meth:`render` emits
one or two ``chat.completion.chunk`` frames per IR event (the role chunk
is emitted lazily, exactly once, before the first content chunk).

:meth:`close` is intentionally imperative — the terminator sequence (final
``finish_reason`` chunk + ``data: [DONE]\\n\\n``) is fixed and doesn't benefit
from FSM dispatch.

The persistent-loop bridge between sync mitmproxy callables and this async
FSM lives in :class:`SSEPipeline` (Phase Q). For tests, the parametrize
fixture in ``tests/test_lightllm_response_render_openai.py`` wraps the
async FSM in a one-loop-per-call sync adapter.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.messages import (
    FinalResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_graph import GraphBuilder, StepContext

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelResponseStreamEvent

logger = logging.getLogger(__name__)


_FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


# ── Wire emission helpers (module-level — pure byte emitters) ──────────────


def _args_to_str(args: str | dict[str, Any] | None) -> str:
    """OpenAI Chat Completion wires tool-call arguments as a JSON string.

    pydantic-ai's IR holds either a string fragment (already-serialized
    JSON), a fully-formed dict, or ``None``. Normalize to the on-wire shape.
    """
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    return json.dumps(args, separators=(",", ":"))


def _emit_chunk(
    *,
    chunk_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> bytes:
    chunk: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
    return f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode()


# ── State ──────────────────────────────────────────────────────────────────


@dataclass
class _OpenAIRenderState:
    """FSM state for one OpenAI render graph run.

    The ``pending_events`` queue holds the single :class:`ModelResponseStreamEvent`
    pushed by :meth:`OpenAIResponseRenderFSM.render` before each graph run; the
    FSM router pops from it. ``out`` accumulates the SSE wire bytes emitted by
    handler steps; the terminal step returns ``bytes(out)`` and resets the buffer.
    The remaining fields (``chunk_id``, ``created``, ``model``, ``role_emitted``,
    ``part_to_tool_call_index``, ``next_tool_call_index``, ``finish_reason``)
    persist across render calls so the stream-level lifecycle stays consistent.
    """

    chunk_id: str
    created: int
    model: str
    role_emitted: bool = False
    part_to_tool_call_index: dict[int, int] = field(default_factory=dict)
    next_tool_call_index: int = 0
    finish_reason: _FinishReason = "stop"
    pending_events: deque[Any] = field(default_factory=deque)
    out: bytearray = field(default_factory=bytearray)


class _RenderDone:
    """Marker returned by the router when the events queue is exhausted."""


# ── Render helpers (operate on state) ──────────────────────────────────────


def _ensure_role(state: _OpenAIRenderState) -> None:
    """Emit the role chunk once, lazily, before any content chunk."""
    if state.role_emitted:
        return
    state.role_emitted = True
    state.out += _emit_chunk(
        chunk_id=state.chunk_id,
        created=state.created,
        model=state.model,
        delta={"role": "assistant"},
    )


# ── Graph ──────────────────────────────────────────────────────────────────


_g: GraphBuilder[_OpenAIRenderState, None, None, bytes] = GraphBuilder(
    state_type=_OpenAIRenderState,
    output_type=bytes,
)


@_g.step
async def take_next_event(
    ctx: StepContext[_OpenAIRenderState, None, None],
) -> Any:
    """Router source: pop the next event from the queue, or signal end via :class:`_RenderDone`."""
    if not ctx.state.pending_events:
        return _RenderDone()
    return ctx.state.pending_events.popleft()


@_g.step
async def handle_part_start(
    ctx: StepContext[_OpenAIRenderState, None, PartStartEvent],
) -> None:
    """Open a new content surface (text or tool_call)."""
    event = ctx.inputs
    state = ctx.state
    _ensure_role(state)

    part = event.part
    if isinstance(part, TextPart):
        if part.content:
            state.out += _emit_chunk(
                chunk_id=state.chunk_id,
                created=state.created,
                model=state.model,
                delta={"content": part.content},
            )
        return
    if isinstance(part, ToolCallPart):
        tc_index = state.next_tool_call_index
        state.next_tool_call_index += 1
        state.part_to_tool_call_index[event.index] = tc_index
        state.out += _emit_chunk(
            chunk_id=state.chunk_id,
            created=state.created,
            model=state.model,
            delta={
                "tool_calls": [
                    {
                        "index": tc_index,
                        "id": part.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": part.tool_name,
                            "arguments": _args_to_str(part.args),
                        },
                    }
                ]
            },
        )
        state.finish_reason = "tool_calls"
        return
    # ThinkingPart, CompactionPart, FilePart, NativeToolCall* etc. have no
    # OpenAI Chat Completion wire surface — the role chunk above is the only
    # output. They fall through to a no-op.


@_g.step
async def handle_part_delta(
    ctx: StepContext[_OpenAIRenderState, None, PartDeltaEvent],
) -> None:
    """Emit a delta chunk for the open content surface."""
    event = ctx.inputs
    state = ctx.state
    delta = event.delta

    if isinstance(delta, TextPartDelta):
        _ensure_role(state)
        state.out += _emit_chunk(
            chunk_id=state.chunk_id,
            created=state.created,
            model=state.model,
            delta={"content": delta.content_delta},
        )
        return

    if isinstance(delta, ToolCallPartDelta):
        _ensure_role(state)
        tc_index = state.part_to_tool_call_index.get(event.index)
        if tc_index is None:
            # First sighting of this IR part via a delta — allocate an
            # OpenAI tool-call slot and emit the envelope (id + name + type).
            tc_index = state.next_tool_call_index
            state.next_tool_call_index += 1
            state.part_to_tool_call_index[event.index] = tc_index
            envelope: dict[str, Any] = {"index": tc_index, "type": "function"}
            if delta.tool_call_id is not None:
                envelope["id"] = delta.tool_call_id
            fn: dict[str, Any] = {}
            if delta.tool_name_delta is not None:
                fn["name"] = delta.tool_name_delta
            fn["arguments"] = _args_to_str(delta.args_delta)
            envelope["function"] = fn
            state.finish_reason = "tool_calls"
            state.out += _emit_chunk(
                chunk_id=state.chunk_id,
                created=state.created,
                model=state.model,
                delta={"tool_calls": [envelope]},
            )
            return

        state.finish_reason = "tool_calls"
        args_str = _args_to_str(delta.args_delta)
        state.out += _emit_chunk(
            chunk_id=state.chunk_id,
            created=state.created,
            model=state.model,
            delta={
                "tool_calls": [
                    {
                        "index": tc_index,
                        "function": {"arguments": args_str},
                    }
                ]
            },
        )
        return

    if isinstance(delta, ThinkingPartDelta):
        # OpenAI Chat Completion SSE has no on-wire surface for thinking
        # content (the ``reasoning`` field is OpenAI Responses only).
        return


@_g.step
async def handle_part_end(
    ctx: StepContext[_OpenAIRenderState, None, PartEndEvent],
) -> None:
    """No-op: OpenAI Chat Completion has no per-block stop marker."""
    del ctx  # protocol-required parameter; intentionally unused


@_g.step
async def handle_final_result(
    ctx: StepContext[_OpenAIRenderState, None, FinalResultEvent],
) -> None:
    """No-op: ``FinalResultEvent`` is an internal agent-loop signal with no OpenAI wire equivalent."""
    del ctx  # protocol-required parameter; intentionally unused


@_g.step
async def emit_done(
    ctx: StepContext[_OpenAIRenderState, None, _RenderDone],
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


class OpenAIResponseRenderFSM:
    """Async pydantic-graph-driven OpenAI Chat Completion SSE renderer.

    Behavioral twin of
    :class:`ccproxy.lightllm.response.render_openai.OpenAIResponseRender`,
    re-expressed as a :mod:`pydantic_graph.beta` ``GraphBuilder`` FSM. One
    graph run per :meth:`render` call drives a single
    :class:`ModelResponseStreamEvent` through the per-variant dispatch ladder
    and returns the emitted SSE bytes. :meth:`close` is imperative — the
    terminator sequence is fixed.
    """

    name = "openai_chat"

    def __init__(self, *, model: str = "unknown") -> None:
        self._state = _OpenAIRenderState(
            chunk_id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=model,
        )

    async def render(self, event: ModelResponseStreamEvent) -> bytes:
        """One IR event → zero-or-more bytes of OpenAI Chat Completion SSE wire output."""
        self._state.pending_events.append(event)
        result: bytes = await _render_graph.run(state=self._state)
        return result

    async def close(self) -> bytes:
        """Emit the final ``finish_reason`` chunk plus the ``[DONE]`` terminator.

        Imperative (no FSM): the terminator sequence is a fixed two-step
        emission with no per-event dispatch.
        """
        state = self._state
        out = bytearray()
        out += _emit_chunk(
            chunk_id=state.chunk_id,
            created=state.created,
            model=state.model,
            delta={},
            finish_reason=state.finish_reason,
        )
        out += b"data: [DONE]\n\n"
        return bytes(out)
