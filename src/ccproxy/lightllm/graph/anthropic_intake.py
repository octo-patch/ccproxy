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
from pydantic_graph import GraphBuilder, StepContext

if TYPE_CHECKING:
    from anthropic.types.beta import BetaContentBlock
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
    builtin_tool_calls: dict[str, NativeToolCallPart] = field(default_factory=dict)
    events_queue: deque[BetaRawMessageStreamEvent] = field(default_factory=deque)
    out_events: list[ModelResponseStreamEvent] = field(default_factory=list)


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


@_g.step
async def handle_content_block_start(
    ctx: StepContext[_AnthropicIntakeState, None, BetaRawContentBlockStartEvent],
) -> None:
    """Handle ``content_block_start`` — open a new content block of the matched variant."""
    event = ctx.inputs
    state = ctx.state
    current_block: BetaContentBlock = event.content_block
    state.current_block = current_block
    provider_name = state.provider_name
    pm = state.parts_manager

    if isinstance(current_block, BetaTextBlock) and current_block.text:
        state.out_events.extend(pm.handle_text_delta(vendor_part_id=event.index, content=current_block.text))
        return
    if isinstance(current_block, BetaThinkingBlock):
        state.out_events.extend(
            pm.handle_thinking_delta(
                vendor_part_id=event.index,
                content=current_block.thinking,
                signature=current_block.signature,
                provider_name=provider_name,
            )
        )
        return
    if isinstance(current_block, BetaRedactedThinkingBlock):
        state.out_events.extend(
            pm.handle_thinking_delta(
                vendor_part_id=event.index,
                id="redacted_thinking",
                signature=current_block.data,
                provider_name=provider_name,
            )
        )
        return
    if isinstance(current_block, BetaToolUseBlock):
        maybe_event = pm.handle_tool_call_delta(
            vendor_part_id=event.index,
            tool_name=current_block.name,
            args=current_block.input or None,
            tool_call_id=current_block.id,
        )
        if maybe_event is not None:
            state.out_events.append(maybe_event)
        return
    if isinstance(current_block, BetaServerToolUseBlock):
        call_part = _map_server_tool_use_block(current_block, provider_name)
        state.builtin_tool_calls[call_part.tool_call_id] = call_part
        state.out_events.append(pm.handle_part(vendor_part_id=event.index, part=call_part))
        return
    if isinstance(current_block, BetaWebSearchToolResultBlock):
        state.out_events.append(
            pm.handle_part(
                vendor_part_id=event.index,
                part=_map_web_search_tool_result_block(current_block, provider_name),
            )
        )
        return
    if isinstance(current_block, BetaCodeExecutionToolResultBlock):
        state.out_events.append(
            pm.handle_part(
                vendor_part_id=event.index,
                part=_map_code_execution_tool_result_block(current_block, provider_name),
            )
        )
        return
    if isinstance(current_block, BetaWebFetchToolResultBlock):
        state.out_events.append(
            pm.handle_part(
                vendor_part_id=event.index,
                part=_map_web_fetch_tool_result_block(current_block, provider_name),
            )
        )
        return
    if isinstance(current_block, BetaMCPToolUseBlock):
        call_part = _map_mcp_server_use_block(current_block, provider_name)
        state.builtin_tool_calls[call_part.tool_call_id] = call_part

        args_json = call_part.args_as_json_str()
        # Drop the final ``{}}`` so we can add tool args deltas
        args_json_delta = args_json[:-3]
        assert args_json_delta.endswith('"tool_args":'), f'Expected {args_json_delta!r} to end in `"tool_args":`'

        state.out_events.append(pm.handle_part(vendor_part_id=event.index, part=replace(call_part, args=None)))
        maybe_event = pm.handle_tool_call_delta(
            vendor_part_id=event.index,
            args=args_json_delta,
        )
        if maybe_event is not None:
            state.out_events.append(maybe_event)
        return
    if isinstance(current_block, BetaMCPToolResultBlock):
        mcp_call_part = state.builtin_tool_calls.get(current_block.tool_use_id)
        state.out_events.append(
            pm.handle_part(
                vendor_part_id=event.index,
                part=_map_mcp_server_result_block(current_block, mcp_call_part, provider_name),
            )
        )
        return
    if isinstance(current_block, BetaCompactionBlock):
        state.out_events.append(
            pm.handle_part(
                vendor_part_id=event.index,
                part=CompactionPart(content=current_block.content, provider_name=provider_name),
            )
        )
        return


@_g.step
async def handle_content_block_delta(
    ctx: StepContext[_AnthropicIntakeState, None, BetaRawContentBlockDeltaEvent],
) -> None:
    """Handle ``content_block_delta`` — incremental update to the open block."""
    event = ctx.inputs
    state = ctx.state
    provider_name = state.provider_name
    pm = state.parts_manager
    delta = event.delta

    if isinstance(delta, BetaTextDelta):
        state.out_events.extend(pm.handle_text_delta(vendor_part_id=event.index, content=delta.text))
        return
    if isinstance(delta, BetaThinkingDelta):
        state.out_events.extend(
            pm.handle_thinking_delta(
                vendor_part_id=event.index,
                content=delta.thinking,
                provider_name=provider_name,
            )
        )
        return
    if isinstance(delta, BetaSignatureDelta):
        state.out_events.extend(
            pm.handle_thinking_delta(
                vendor_part_id=event.index,
                signature=delta.signature,
                provider_name=provider_name,
            )
        )
        return
    if isinstance(delta, BetaInputJSONDelta):
        maybe_event = pm.handle_tool_call_delta(
            vendor_part_id=event.index,
            args=delta.partial_json,
        )
        if maybe_event is not None:
            state.out_events.append(maybe_event)
        return
    if isinstance(delta, BetaCompactionContentBlockDelta):
        if delta.content:
            state.out_events.append(
                pm.handle_part(
                    vendor_part_id=event.index,
                    part=CompactionPart(content=delta.content, provider_name=provider_name),
                )
            )
        return
    if isinstance(delta, BetaCitationsDelta):
        # TODO(upstream pydantic-ai): citations not yet wired through to IR events.
        return


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
    ctx.state.out_events = []
    return out


_g.add(
    _g.edge_from(_g.start_node).to(frame_next_event),
    _g.edge_from(frame_next_event).to(
        _g.decision()
        .branch(_g.match(_FeedDone).to(emit_done))
        .branch(_g.match(_IgnoredEvent).to(skip_ignored_event))
        .branch(_g.match(BetaRawContentBlockStartEvent).to(handle_content_block_start))
        .branch(_g.match(BetaRawContentBlockDeltaEvent).to(handle_content_block_delta))
        .branch(_g.match(BetaRawContentBlockStopEvent).to(handle_content_block_stop))
    ),
    _g.edge_from(
        handle_content_block_start,
        handle_content_block_delta,
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

    async def feed(self, data: bytes) -> list[ModelResponseStreamEvent]:
        """Buffer bytes, frame SSE events, drive the FSM, return emitted IR events."""
        self.upstream_raw_bytes.extend(data)
        if not data:
            return []
        self._sse_buffer.extend(data)
        # Drain complete SSE frames into typed Anthropic events.
        for raw_event in self._drain_sse_events():
            self._state.events_queue.append(raw_event)
        # If there were no complete frames, short-circuit — the graph run would
        # produce no events.
        if not self._state.events_queue:
            return []
        result = await _intake_graph.run(state=self._state)
        return result

    async def close(self) -> list[ModelResponseStreamEvent]:
        """Stream end. ``message_stop`` already closes everything; nothing to flush."""
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
