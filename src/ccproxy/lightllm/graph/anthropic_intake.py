"""Anthropic Messages SSE bytes → pydantic-ai IR events via FSM.

Pydantic-graph FSM port of
:class:`ccproxy.lightllm.response.intake_anthropic.AnthropicResponseIntake`.
One graph run per :meth:`AnthropicResponseIntakeFSM.feed` call: bytes are
appended to the SSE buffer, complete SSE frames are drained and validated into
typed :class:`BetaRawMessageStreamEvent` instances, those events are pushed
onto an in-state queue, and the FSM router drains the queue dispatching each
event to a per-variant handler step. Handler steps mutate
``state.parts_manager`` and append emitted
:class:`ModelResponseStreamEvent` objects to ``state.out_events``.

The behavioral contract matches
:mod:`ccproxy.lightllm.response.intake_anthropic` byte-for-byte: same SSE
framing rules (``\\r\\n\\r\\n`` and ``\\n\\n`` separators, ``data:`` payload
concatenation), same dispatch ladder, same parts-manager calls, same
hard-coded ``provider_name = "anthropic"``.

The persistent-loop bridge between sync mitmproxy callables and this async
FSM lives in :class:`SSEPipeline` (Phase Q). For tests, the parametrize
fixture in ``tests/test_lightllm_response_intake_anthropic.py`` wraps the
async FSM in a one-loop-per-call sync adapter.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from anthropic.types.beta import (
    BetaCitationsDelta,
    BetaCodeExecutionToolResultBlock,
    BetaCompactionBlock,
    BetaCompactionContentBlockDelta,
    BetaContentBlock,
    BetaInputJSONDelta,
    BetaMCPToolResultBlock,
    BetaMCPToolUseBlock,
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
    BetaRawContentBlockStopEvent,
    BetaRawMessageDeltaEvent,
    BetaRawMessageStartEvent,
    BetaRawMessageStopEvent,
    BetaRawMessageStreamEvent,
    BetaRedactedThinkingBlock,
    BetaServerToolUseBlock,
    BetaSignatureDelta,
    BetaTextBlock,
    BetaTextDelta,
    BetaThinkingBlock,
    BetaThinkingDelta,
    BetaToolUseBlock,
    BetaWebFetchToolResultBlock,
    BetaWebSearchToolResultBlock,
)
from pydantic import TypeAdapter, ValidationError

# Private pydantic-ai imports — see the matching note in
# ``response/intake_anthropic.py``. We need byte-identical dispatch behavior
# and there is no public replacement.
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import (
    CompactionPart,
    ModelResponseStreamEvent,
    NativeToolCallPart,
)
from pydantic_ai.models.anthropic import (
    _map_code_execution_tool_result_block,
    _map_mcp_server_result_block,
    _map_mcp_server_use_block,
    _map_server_tool_use_block,
    _map_web_fetch_tool_result_block,
    _map_web_search_tool_result_block,
)
from pydantic_graph import GraphBuilder, StepContext, TypeExpression

import ccproxy.lightllm.graph._subgraph_patch  # noqa: F401  -- installs GraphBuilder.add_subgraph

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestParameters

logger = logging.getLogger(__name__)


_EVENT_ADAPTER: TypeAdapter[BetaRawMessageStreamEvent] = TypeAdapter(BetaRawMessageStreamEvent)
"""``BetaRawMessageStreamEvent`` is ``Annotated[Union[...], Field(discriminator='type')]``;
the canonical way to validate one instance from a JSON payload is via a ``TypeAdapter``.
"""


# ── State ──────────────────────────────────────────────────────────────────


@dataclass
class _AnthropicIntakeState:
    """FSM state for one Anthropic intake graph run.

    The ``events_queue`` is the queue of typed
    :class:`BetaRawMessageStreamEvent` instances drained from the SSE buffer
    *before* the graph run starts; the FSM router pops from it. The
    ``out_events`` list accumulates :class:`ModelResponseStreamEvent` instances
    emitted by handler steps; the terminal step returns it.
    ``parts_manager``, ``current_block``, ``builtin_tool_calls`` persist across
    feed calls so multi-feed reassembly works.
    """

    parts_manager: ModelResponsePartsManager
    provider_name: str
    current_block: BetaContentBlock | None = None
    current_index: int | None = None
    """SSE ``index`` of the block currently being dispatched; set by the inner-subgraph open step."""
    builtin_tool_calls: dict[str, NativeToolCallPart] = field(default_factory=dict)
    events_queue: deque[BetaRawMessageStreamEvent] = field(default_factory=deque)
    out_events: list[ModelResponseStreamEvent] = field(default_factory=list)

    # ── Telemetry (never-silently-drop diagnostics) ───────────────────────────
    frames_seen: int = 0
    """Total typed SSE events drained from the wire (across all feed calls)."""

    frames_unparseable: int = 0
    """SSE frames whose ``data:`` payload failed ``BetaRawMessageStreamEvent`` validation."""

    emitted_events: int = 0
    """Total IR ``ModelResponseStreamEvent`` instances emitted (across all feed calls)."""


class _FeedDone:
    """Marker returned by the router when the events queue is exhausted."""


class _IgnoredEvent:
    """Marker for events that produce no IR output (message_start, message_delta,
    message_stop). They still need to flow through the FSM so the router stays
    decision-driven, but they have no per-event handler beyond clearing
    ``current_block`` (which is handled inline in the router for clarity).
    """


type _RoutedEvent = (
    BetaRawContentBlockStartEvent
    | BetaRawContentBlockDeltaEvent
    | BetaRawContentBlockStopEvent
    | _IgnoredEvent
    | _FeedDone
)


# ── Graph ──────────────────────────────────────────────────────────────────


_g: GraphBuilder[_AnthropicIntakeState, None, None, list[ModelResponseStreamEvent]] = GraphBuilder(
    name="anthropic_intake",
    state_type=_AnthropicIntakeState,
    output_type=list[ModelResponseStreamEvent],
)


@_g.step
async def frame_next_event(
    ctx: StepContext[_AnthropicIntakeState, None, None],
) -> _RoutedEvent:
    """Router source: pop the next typed event from the queue, or signal end via :class:`_FeedDone`."""
    state = ctx.state
    while state.events_queue:
        event = state.events_queue.popleft()
        # ``message_start`` and ``message_delta`` carry usage / metadata that
        # pydantic-ai stashes on ``StreamedResponse``; they have no IR-event
        # equivalent. Surface them as :class:`_IgnoredEvent` so the FSM stays
        # decision-driven.
        if isinstance(event, (BetaRawMessageStartEvent, BetaRawMessageDeltaEvent)):
            return _IgnoredEvent()
        if isinstance(event, BetaRawMessageStopEvent):
            state.current_block = None
            return _IgnoredEvent()
        return event
    return _FeedDone()


# ── Inner subgraph: content_block_start dispatch ────────────────────────────
#
# Each ``content_block_start`` event carries one ``content_block`` (a single
# object, not a list), so this subgraph is a flat type-switch with no loop:
# ``open_block`` stashes the block + its SSE index on state and hands the block
# to a decision that routes on the concrete block class.

_bsg: GraphBuilder[_AnthropicIntakeState, None, BetaRawContentBlockStartEvent, None] = GraphBuilder(
    name="anthropic_block_start_dispatch",
    state_type=_AnthropicIntakeState,
    input_type=BetaRawContentBlockStartEvent,
)


@_bsg.step
async def open_block(
    ctx: StepContext[_AnthropicIntakeState, None, BetaRawContentBlockStartEvent],
) -> BetaContentBlock:
    """Open a new content block: stash it + its SSE index on state, hand the block to the decision."""
    block = ctx.inputs.content_block
    ctx.state.current_block = block
    ctx.state.current_index = ctx.inputs.index
    return block


@_bsg.step
async def handle_text_block(ctx: StepContext[_AnthropicIntakeState, None, BetaTextBlock]) -> None:
    """``text`` block — emit a text delta for the initial body (skipped when empty)."""
    state = ctx.state
    if ctx.inputs.text:
        state.out_events.extend(
            state.parts_manager.handle_text_delta(vendor_part_id=state.current_index, content=ctx.inputs.text)
        )


@_bsg.step
async def handle_thinking_block(ctx: StepContext[_AnthropicIntakeState, None, BetaThinkingBlock]) -> None:
    """``thinking`` block — emit a thinking delta with the initial content + signature."""
    state = ctx.state
    block = ctx.inputs
    state.out_events.extend(
        state.parts_manager.handle_thinking_delta(
            vendor_part_id=state.current_index,
            content=block.thinking,
            signature=block.signature,
            provider_name=state.provider_name,
        )
    )


@_bsg.step
async def handle_redacted_thinking_block(
    ctx: StepContext[_AnthropicIntakeState, None, BetaRedactedThinkingBlock],
) -> None:
    """``redacted_thinking`` block — emit a redacted thinking part."""
    state = ctx.state
    state.out_events.extend(
        state.parts_manager.handle_thinking_delta(
            vendor_part_id=state.current_index,
            id="redacted_thinking",
            signature=ctx.inputs.data,
            provider_name=state.provider_name,
        )
    )


@_bsg.step
async def handle_tool_use_block(ctx: StepContext[_AnthropicIntakeState, None, BetaToolUseBlock]) -> None:
    """``tool_use`` block — open a tool-call part; args arrive via later input_json deltas."""
    state = ctx.state
    block = ctx.inputs
    maybe_event = state.parts_manager.handle_tool_call_delta(
        vendor_part_id=state.current_index,
        tool_name=block.name,
        args=block.input or None,
        tool_call_id=block.id,
    )
    if maybe_event is not None:
        state.out_events.append(maybe_event)


@_bsg.step
async def handle_server_tool_use_block(
    ctx: StepContext[_AnthropicIntakeState, None, BetaServerToolUseBlock],
) -> None:
    """``server_tool_use`` block — record the builtin call and emit it with deferred args."""
    state = ctx.state
    call_part = _map_server_tool_use_block(ctx.inputs, state.provider_name)
    state.builtin_tool_calls[call_part.tool_call_id] = call_part
    state.out_events.append(state.parts_manager.handle_part(vendor_part_id=state.current_index, part=call_part))


@_bsg.step
async def handle_web_search_tool_result_block(
    ctx: StepContext[_AnthropicIntakeState, None, BetaWebSearchToolResultBlock],
) -> None:
    """``web_search_tool_result`` block — emit the mapped result part."""
    state = ctx.state
    state.out_events.append(
        state.parts_manager.handle_part(
            vendor_part_id=state.current_index,
            part=_map_web_search_tool_result_block(ctx.inputs, state.provider_name),
        )
    )


@_bsg.step
async def handle_code_execution_tool_result_block(
    ctx: StepContext[_AnthropicIntakeState, None, BetaCodeExecutionToolResultBlock],
) -> None:
    """``code_execution_tool_result`` block — emit the mapped result part."""
    state = ctx.state
    state.out_events.append(
        state.parts_manager.handle_part(
            vendor_part_id=state.current_index,
            part=_map_code_execution_tool_result_block(ctx.inputs, state.provider_name),
        )
    )


@_bsg.step
async def handle_web_fetch_tool_result_block(
    ctx: StepContext[_AnthropicIntakeState, None, BetaWebFetchToolResultBlock],
) -> None:
    """``web_fetch_tool_result`` block — emit the mapped result part."""
    state = ctx.state
    state.out_events.append(
        state.parts_manager.handle_part(
            vendor_part_id=state.current_index,
            part=_map_web_fetch_tool_result_block(ctx.inputs, state.provider_name),
        )
    )


@_bsg.step
async def handle_mcp_tool_use_block(ctx: StepContext[_AnthropicIntakeState, None, BetaMCPToolUseBlock]) -> None:
    """``mcp_tool_use`` block — emit the call with deferred args + an opening args delta."""
    state = ctx.state
    call_part = _map_mcp_server_use_block(ctx.inputs, state.provider_name)
    state.builtin_tool_calls[call_part.tool_call_id] = call_part

    args_json = call_part.args_as_json_str()
    # Drop the final ``{}}`` so we can add tool args deltas
    args_json_delta = args_json[:-3]
    assert args_json_delta.endswith('"tool_args":'), f'Expected {args_json_delta!r} to end in `"tool_args":`'

    state.out_events.append(
        state.parts_manager.handle_part(vendor_part_id=state.current_index, part=replace(call_part, args=None))
    )
    maybe_event = state.parts_manager.handle_tool_call_delta(
        vendor_part_id=state.current_index,
        args=args_json_delta,
    )
    if maybe_event is not None:
        state.out_events.append(maybe_event)


@_bsg.step
async def handle_mcp_tool_result_block(
    ctx: StepContext[_AnthropicIntakeState, None, BetaMCPToolResultBlock],
) -> None:
    """``mcp_tool_result`` block — emit the mapped result, correlated to the prior call."""
    state = ctx.state
    mcp_call_part = state.builtin_tool_calls.get(ctx.inputs.tool_use_id)
    state.out_events.append(
        state.parts_manager.handle_part(
            vendor_part_id=state.current_index,
            part=_map_mcp_server_result_block(ctx.inputs, mcp_call_part, state.provider_name),
        )
    )


@_bsg.step
async def handle_compaction_block(ctx: StepContext[_AnthropicIntakeState, None, BetaCompactionBlock]) -> None:
    """``compaction`` block — emit a CompactionPart with the initial content."""
    state = ctx.state
    state.out_events.append(
        state.parts_manager.handle_part(
            vendor_part_id=state.current_index,
            part=CompactionPart(content=ctx.inputs.content, provider_name=state.provider_name),
        )
    )


@_bsg.step
async def handle_unknown_block(ctx: StepContext[_AnthropicIntakeState, None, object]) -> None:
    """Catch-all for content_block variants with no IR handler — log instead of silently dropping."""
    logger.debug("anthropic intake: unhandled content_block type %s; skipping", type(ctx.inputs).__name__)


_bsg.add(
    _bsg.edge_from(_bsg.start_node).to(open_block),
    _bsg.edge_from(open_block).to(
        _bsg.decision()
        .branch(_bsg.match(BetaTextBlock).to(handle_text_block))
        .branch(_bsg.match(BetaThinkingBlock).to(handle_thinking_block))
        .branch(_bsg.match(BetaRedactedThinkingBlock).to(handle_redacted_thinking_block))
        .branch(_bsg.match(BetaToolUseBlock).to(handle_tool_use_block))
        .branch(_bsg.match(BetaServerToolUseBlock).to(handle_server_tool_use_block))
        .branch(_bsg.match(BetaWebSearchToolResultBlock).to(handle_web_search_tool_result_block))
        .branch(_bsg.match(BetaCodeExecutionToolResultBlock).to(handle_code_execution_tool_result_block))
        .branch(_bsg.match(BetaWebFetchToolResultBlock).to(handle_web_fetch_tool_result_block))
        .branch(_bsg.match(BetaMCPToolUseBlock).to(handle_mcp_tool_use_block))
        .branch(_bsg.match(BetaMCPToolResultBlock).to(handle_mcp_tool_result_block))
        .branch(_bsg.match(BetaCompactionBlock).to(handle_compaction_block))
        .branch(_bsg.match(TypeExpression[object]).to(handle_unknown_block))
    ),
    _bsg.edge_from(
        handle_text_block,
        handle_thinking_block,
        handle_redacted_thinking_block,
        handle_tool_use_block,
        handle_server_tool_use_block,
        handle_web_search_tool_result_block,
        handle_code_execution_tool_result_block,
        handle_web_fetch_tool_result_block,
        handle_mcp_tool_use_block,
        handle_mcp_tool_result_block,
        handle_compaction_block,
        handle_unknown_block,
    ).to(_bsg.end_node),
)

_block_start_graph = _bsg.build()
_dispatch_block_start = _g.add_subgraph(_block_start_graph, label="block_start")  # ty: ignore[unresolved-attribute]


# ── Inner subgraph: content_block_delta dispatch ────────────────────────────
#
# Symmetric to the block-start subgraph: ``open_delta`` stashes the SSE index
# and hands ``event.delta`` (a single object) to a decision that routes on the
# concrete delta class. No loop.

type _BlockDelta = (
    BetaTextDelta
    | BetaThinkingDelta
    | BetaSignatureDelta
    | BetaInputJSONDelta
    | BetaCompactionContentBlockDelta
    | BetaCitationsDelta
)

_bdg: GraphBuilder[_AnthropicIntakeState, None, BetaRawContentBlockDeltaEvent, None] = GraphBuilder(
    name="anthropic_block_delta_dispatch",
    state_type=_AnthropicIntakeState,
    input_type=BetaRawContentBlockDeltaEvent,
)


@_bdg.step
async def open_delta(
    ctx: StepContext[_AnthropicIntakeState, None, BetaRawContentBlockDeltaEvent],
) -> _BlockDelta:
    """Stash the SSE index and hand the delta to the decision."""
    ctx.state.current_index = ctx.inputs.index
    return ctx.inputs.delta


@_bdg.step
async def handle_text_delta(ctx: StepContext[_AnthropicIntakeState, None, BetaTextDelta]) -> None:
    """``text_delta`` — append incremental text to the open part."""
    state = ctx.state
    state.out_events.extend(
        state.parts_manager.handle_text_delta(vendor_part_id=state.current_index, content=ctx.inputs.text)
    )


@_bdg.step
async def handle_thinking_delta(ctx: StepContext[_AnthropicIntakeState, None, BetaThinkingDelta]) -> None:
    """``thinking_delta`` — append incremental thinking content."""
    state = ctx.state
    state.out_events.extend(
        state.parts_manager.handle_thinking_delta(
            vendor_part_id=state.current_index,
            content=ctx.inputs.thinking,
            provider_name=state.provider_name,
        )
    )


@_bdg.step
async def handle_signature_delta(ctx: StepContext[_AnthropicIntakeState, None, BetaSignatureDelta]) -> None:
    """``signature_delta`` — attach the thinking signature."""
    state = ctx.state
    state.out_events.extend(
        state.parts_manager.handle_thinking_delta(
            vendor_part_id=state.current_index,
            signature=ctx.inputs.signature,
            provider_name=state.provider_name,
        )
    )


@_bdg.step
async def handle_input_json_delta(ctx: StepContext[_AnthropicIntakeState, None, BetaInputJSONDelta]) -> None:
    """``input_json_delta`` — append partial tool-call args JSON."""
    state = ctx.state
    maybe_event = state.parts_manager.handle_tool_call_delta(
        vendor_part_id=state.current_index,
        args=ctx.inputs.partial_json,
    )
    if maybe_event is not None:
        state.out_events.append(maybe_event)


@_bdg.step
async def handle_compaction_delta(
    ctx: StepContext[_AnthropicIntakeState, None, BetaCompactionContentBlockDelta],
) -> None:
    """``compaction`` delta — emit a CompactionPart for any new content."""
    state = ctx.state
    if ctx.inputs.content:
        state.out_events.append(
            state.parts_manager.handle_part(
                vendor_part_id=state.current_index,
                part=CompactionPart(content=ctx.inputs.content, provider_name=state.provider_name),
            )
        )


@_bdg.step
async def handle_citations_delta(ctx: StepContext[_AnthropicIntakeState, None, BetaCitationsDelta]) -> None:
    """``citations_delta`` — no-op."""
    # TODO(upstream pydantic-ai): citations not yet wired through to IR events.
    del ctx  # protocol-required parameter; intentionally unused


@_bdg.step
async def handle_unknown_delta(ctx: StepContext[_AnthropicIntakeState, None, object]) -> None:
    """Catch-all for delta variants with no IR handler — log instead of silently dropping."""
    logger.debug("anthropic intake: unhandled content_block delta %s; skipping", type(ctx.inputs).__name__)


_bdg.add(
    _bdg.edge_from(_bdg.start_node).to(open_delta),
    _bdg.edge_from(open_delta).to(
        _bdg.decision()
        .branch(_bdg.match(BetaTextDelta).to(handle_text_delta))
        .branch(_bdg.match(BetaThinkingDelta).to(handle_thinking_delta))
        .branch(_bdg.match(BetaSignatureDelta).to(handle_signature_delta))
        .branch(_bdg.match(BetaInputJSONDelta).to(handle_input_json_delta))
        .branch(_bdg.match(BetaCompactionContentBlockDelta).to(handle_compaction_delta))
        .branch(_bdg.match(BetaCitationsDelta).to(handle_citations_delta))
        .branch(_bdg.match(TypeExpression[object]).to(handle_unknown_delta))
    ),
    _bdg.edge_from(
        handle_text_delta,
        handle_thinking_delta,
        handle_signature_delta,
        handle_input_json_delta,
        handle_compaction_delta,
        handle_citations_delta,
        handle_unknown_delta,
    ).to(_bdg.end_node),
)

_block_delta_graph = _bdg.build()
_dispatch_block_delta = _g.add_subgraph(_block_delta_graph, label="block_delta")  # ty: ignore[unresolved-attribute]


@_g.step
async def handle_content_block_stop(
    ctx: StepContext[_AnthropicIntakeState, None, BetaRawContentBlockStopEvent],
) -> None:
    """Handle ``content_block_stop`` — close the block. MCP tool-use needs a final ``}`` for its args."""
    event = ctx.inputs
    state = ctx.state
    if isinstance(state.current_block, BetaMCPToolUseBlock):
        maybe_event = state.parts_manager.handle_tool_call_delta(
            vendor_part_id=event.index,
            args="}",
        )
        if maybe_event is not None:
            state.out_events.append(maybe_event)
    state.current_block = None


@_g.step
async def skip_ignored_event(
    ctx: StepContext[_AnthropicIntakeState, None, _IgnoredEvent],
) -> None:
    """No-op for events with no IR equivalent (message_start, message_delta, message_stop)."""
    del ctx  # protocol-required parameter; intentionally unused


@_g.step
async def emit_done(
    ctx: StepContext[_AnthropicIntakeState, None, _FeedDone],
) -> list[ModelResponseStreamEvent]:
    """Terminal step — drain the accumulated IR events and reset for the next feed."""
    out = ctx.state.out_events
    ctx.state.emitted_events += len(out)
    ctx.state.out_events = []
    return out


_g.add(
    _g.edge_from(_g.start_node).to(frame_next_event),
    _g.edge_from(frame_next_event).to(
        _g.decision()
        .branch(_g.match(_FeedDone).to(emit_done))
        .branch(_g.match(_IgnoredEvent).to(skip_ignored_event))
        .branch(_g.match(BetaRawContentBlockStartEvent).to(_dispatch_block_start))
        .branch(_g.match(BetaRawContentBlockDeltaEvent).to(_dispatch_block_delta))
        .branch(_g.match(BetaRawContentBlockStopEvent).to(handle_content_block_stop))
    ),
    _g.edge_from(
        _dispatch_block_start,
        _dispatch_block_delta,
        handle_content_block_stop,
        skip_ignored_event,
    ).to(frame_next_event),
    _g.edge_from(emit_done).to(_g.end_node),
)


_intake_graph = _g.build()


# ── Public class ───────────────────────────────────────────────────────────


class AnthropicResponseIntakeFSM:
    """Async pydantic-graph-driven Anthropic Messages SSE intake.

    Behavioral twin of
    :class:`ccproxy.lightllm.response.intake_anthropic.AnthropicResponseIntake`,
    re-expressed as a :mod:`pydantic_graph.beta` ``GraphBuilder`` FSM. One graph
    run per :meth:`feed` call drains all complete SSE frames buffered by that
    call into typed Anthropic events, dispatches each one to a handler step,
    and returns the accumulated IR events. Partial frames remain in the SSE
    buffer for the next call. ``parts_manager`` and ``current_block`` persist
    across calls.
    """

    name = "anthropic"

    def __init__(self, *, model: str, request_params: ModelRequestParameters) -> None:
        self._model = model
        self._request_params = request_params
        self._sse_buffer = bytearray()
        self.upstream_raw_bytes = bytearray()
        # ``provider_name`` matches what pydantic-ai's ``AnthropicStreamedResponse``
        # uses; hard-coded to "anthropic" because this intake is selected for
        # anthropic-family upstreams (anthropic, deepseek-anthropic-compat,
        # zai-anthropic-compat).
        self._state = _AnthropicIntakeState(
            parts_manager=ModelResponsePartsManager(model_request_parameters=request_params),
            provider_name="anthropic",
        )

    @property
    def parts_manager(self) -> ModelResponsePartsManager:
        """Expose the underlying parts manager for tests and downstream renderers."""
        return self._state.parts_manager

    @property
    def state(self) -> _AnthropicIntakeState:
        """Expose FSM state for tests and telemetry inspection."""
        return self._state

    async def feed(self, data: bytes) -> list[ModelResponseStreamEvent]:
        """Buffer bytes, frame SSE events, drive the FSM, return emitted IR events."""
        self.upstream_raw_bytes.extend(data)
        if not data:
            return []
        self._sse_buffer.extend(data)
        # Drain complete SSE frames into typed Anthropic events.
        for raw_event in self._drain_sse_events():
            self._state.events_queue.append(raw_event)
            self._state.frames_seen += 1
        # If there were no complete frames, short-circuit — the graph run would
        # produce no events.
        if not self._state.events_queue:
            return []
        result = await _intake_graph.run(state=self._state)
        return result

    async def close(self) -> list[ModelResponseStreamEvent]:
        """Stream end. ``message_stop`` already closes everything; nothing to flush.

        Emits a telemetry warning when the stream carried events but produced no
        IR output — a silent empty Anthropic turn must be explainable from logs.
        """
        s = self._state
        if s.frames_seen and not s.emitted_events:
            logger.warning(
                "anthropic intake produced NO IR events after %d frame(s) "
                "(unparseable=%d) — the upstream stream carried no renderable content",
                s.frames_seen,
                s.frames_unparseable,
            )
        return []

    def _drain_sse_events(self) -> Iterator[BetaRawMessageStreamEvent]:
        """Frame SSE events from ``self._sse_buffer``; validate each into a typed event.

        Handles both ``\\r\\n\\r\\n`` (industry standard) and ``\\n\\n`` (some servers)
        separators; partial frames remain buffered for the next ``feed`` call.
        """
        while True:
            # SSE separator is \r\n\r\n on the wire; some servers emit \n\n.
            # Pick whichever boundary appears first in the buffer.
            crlf = self._sse_buffer.find(b"\r\n\r\n")
            lf = self._sse_buffer.find(b"\n\n")
            if crlf == -1 and lf == -1:
                return
            if crlf != -1 and (lf == -1 or crlf < lf):
                frame_bytes = bytes(self._sse_buffer[:crlf])
                del self._sse_buffer[: crlf + 4]
            else:
                frame_bytes = bytes(self._sse_buffer[:lf])
                del self._sse_buffer[: lf + 2]

            payload = self._extract_data_payload(frame_bytes)
            if not payload:
                continue
            try:
                yield _EVENT_ADAPTER.validate_json(payload)
            except ValidationError:
                self._state.frames_unparseable += 1
                logger.debug("anthropic intake: skipping unparseable frame", exc_info=True)

    @staticmethod
    def _extract_data_payload(frame: bytes) -> bytes | None:
        """Return the concatenated ``data:`` line payload from one SSE frame, or ``None``."""
        payloads: list[bytes] = []
        for line in frame.split(b"\n"):
            stripped = line.strip()
            if not stripped.startswith(b"data:"):
                continue
            value = stripped[5:].strip()
            if value:
                payloads.append(value)
        if not payloads:
            return None
        return b"\n".join(payloads)
