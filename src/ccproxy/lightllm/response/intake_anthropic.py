"""Anthropic Messages SSE bytes → pydantic-ai IR events (sync).

Sync transliteration of ``AnthropicStreamedResponse._get_event_iterator``
from ``pydantic_ai.models.anthropic`` (1.85.1: ``models/anthropic.py:1673-1829``).
The async ``async for event in self._response`` outer loop is replaced
with our own sync SSE-bytes-to-event-objects parser; every internal
``self._parts_manager.handle_*_delta(...)`` call is identical because
those methods are sync in pydantic-ai.

Source-tracking: keep the dispatch in :meth:`_dispatch_event` in lock-step
with pydantic-ai's iterator. If pydantic-ai adds a new ``BetaContentBlock``
variant upstream, mirror it here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

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

# ``pydantic_ai._parts_manager.ModelResponsePartsManager`` and the ``_map_*`` helpers in
# ``pydantic_ai.models.anthropic`` are flagged as private by their leading underscore but
# are imported directly here because (a) we are explicitly transliterating pydantic-ai's
# per-vendor dispatch and need byte-identical behavior, and (b) there is no public
# replacement. See the "Risks and mitigations" section of
# ``plans/reshape-wire-py-as-lexical-graham.md``.
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import CompactionPart
from pydantic_ai.models.anthropic import (
    _map_code_execution_tool_result_block,
    _map_mcp_server_result_block,
    _map_mcp_server_use_block,
    _map_server_tool_use_block,
    _map_web_fetch_tool_result_block,
    _map_web_search_tool_result_block,
)

if TYPE_CHECKING:
    from anthropic.types.beta import BetaContentBlock
    from pydantic_ai.messages import BuiltinToolCallPart, ModelResponseStreamEvent
    from pydantic_ai.models import ModelRequestParameters

logger = logging.getLogger(__name__)

_EVENT_ADAPTER: TypeAdapter[BetaRawMessageStreamEvent] = TypeAdapter(BetaRawMessageStreamEvent)
"""``BetaRawMessageStreamEvent`` is ``Annotated[Union[...], Field(discriminator='type')]``;
the canonical way to validate one instance from a JSON payload is via a ``TypeAdapter``.
"""


class AnthropicResponseIntake:
    """Per-stream sync intake for Anthropic Messages SSE.

    Buffers partial frames, validates each complete frame into the discriminated
    ``BetaRawMessageStreamEvent`` union via ``_EVENT_ADAPTER``, and dispatches
    each event to drive ``ModelResponsePartsManager`` (sync).
    """

    name = "anthropic"

    def __init__(self, *, model: str, request_params: ModelRequestParameters) -> None:
        # ``request_params`` is accepted to honor the ``ResponseIntake`` Protocol; pydantic-ai
        # 1.85.1's ``ModelResponsePartsManager`` is a no-arg dataclass. Newer pydantic-ai versions
        # accept ``model_request_parameters=`` — switch when we upgrade the pin.
        self._parts_manager = ModelResponsePartsManager()
        self._model = model
        self._request_params = request_params
        self._sse_buffer = bytearray()
        self.upstream_raw_bytes = bytearray()
        self._current_block: BetaContentBlock | None = None
        self._builtin_tool_calls: dict[str, BuiltinToolCallPart] = {}
        # ``provider_name`` matches what pydantic-ai's ``AnthropicStreamedResponse`` uses;
        # we hard-code "anthropic" because this intake is selected for anthropic-family
        # upstreams (anthropic, deepseek-anthropic-compat, zai-anthropic-compat).
        self._provider_name = "anthropic"

    @property
    def parts_manager(self) -> ModelResponsePartsManager:
        """Expose the underlying parts manager for tests and downstream renderers."""
        return self._parts_manager

    def feed(self, data: bytes) -> Iterator[ModelResponseStreamEvent]:
        """Buffer bytes, frame SSE events, dispatch each parsed event to the parts manager."""
        self.upstream_raw_bytes.extend(data)
        if not data:
            return
        self._sse_buffer.extend(data)
        for raw_event in self._drain_sse_events():
            yield from self._dispatch_event(raw_event)

    def close(self) -> Iterator[ModelResponseStreamEvent]:
        """Stream end. Typically a no-op for Anthropic — ``BetaRawMessageStopEvent`` already closes everything."""
        yield from ()

    def _drain_sse_events(self) -> Iterator[BetaRawMessageStreamEvent]:
        """Frame SSE events from ``self._sse_buffer``; validate each into a typed event.

        Handles both ``\\r\\n\\r\\n`` (industry standard) and ``\\n\\n`` (some servers)
        separators; partial frames remain buffered for the next ``feed`` call. The
        ``event:`` line names the event type but Anthropic also encodes the type inside
        the JSON ``type`` field, so the ``TypeAdapter`` discriminator drives parsing.
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

    def _dispatch_event(self, event: BetaRawMessageStreamEvent) -> Iterator[ModelResponseStreamEvent]:
        """Sync transliteration of ``AnthropicStreamedResponse._get_event_iterator``.

        Mirrors ``pydantic_ai/models/anthropic.py:1673-1829`` (1.85.1).
        """
        if isinstance(event, BetaRawMessageStartEvent):
            # Usage / metadata bookkeeping is stored upstream on ``StreamedResponse``;
            # we don't surface it through the IR event stream (handled separately if needed).
            return

        if isinstance(event, BetaRawContentBlockStartEvent):
            yield from self._handle_content_block_start(event)
            return

        if isinstance(event, BetaRawContentBlockDeltaEvent):
            yield from self._handle_content_block_delta(event)
            return

        if isinstance(event, BetaRawMessageDeltaEvent):
            # Usage and finish_reason are pydantic-ai StreamedResponse state, not IR events.
            return

        if isinstance(event, BetaRawContentBlockStopEvent):
            yield from self._handle_content_block_stop(event)
            return

        if isinstance(event, BetaRawMessageStopEvent):
            self._current_block = None
            return

    def _handle_content_block_start(self, event: BetaRawContentBlockStartEvent) -> Iterator[ModelResponseStreamEvent]:
        current_block: BetaContentBlock = event.content_block
        self._current_block = current_block

        if isinstance(current_block, BetaTextBlock) and current_block.text:
            yield from self._parts_manager.handle_text_delta(vendor_part_id=event.index, content=current_block.text)
            return
        if isinstance(current_block, BetaThinkingBlock):
            yield from self._parts_manager.handle_thinking_delta(
                vendor_part_id=event.index,
                content=current_block.thinking,
                signature=current_block.signature,
                provider_name=self._provider_name,
            )
            return
        if isinstance(current_block, BetaRedactedThinkingBlock):
            yield from self._parts_manager.handle_thinking_delta(
                vendor_part_id=event.index,
                id="redacted_thinking",
                signature=current_block.data,
                provider_name=self._provider_name,
            )
            return
        if isinstance(current_block, BetaToolUseBlock):
            maybe_event = self._parts_manager.handle_tool_call_delta(
                vendor_part_id=event.index,
                tool_name=current_block.name,
                args=cast("dict[str, Any]", current_block.input) or None,
                tool_call_id=current_block.id,
            )
            if maybe_event is not None:
                yield maybe_event
            return
        if isinstance(current_block, BetaServerToolUseBlock):
            call_part = _map_server_tool_use_block(current_block, self._provider_name)
            self._builtin_tool_calls[call_part.tool_call_id] = call_part
            yield self._parts_manager.handle_part(
                vendor_part_id=event.index,
                part=call_part,
            )
            return
        if isinstance(current_block, BetaWebSearchToolResultBlock):
            yield self._parts_manager.handle_part(
                vendor_part_id=event.index,
                part=_map_web_search_tool_result_block(current_block, self._provider_name),
            )
            return
        if isinstance(current_block, BetaCodeExecutionToolResultBlock):
            yield self._parts_manager.handle_part(
                vendor_part_id=event.index,
                part=_map_code_execution_tool_result_block(current_block, self._provider_name),
            )
            return
        if isinstance(current_block, BetaWebFetchToolResultBlock):
            yield self._parts_manager.handle_part(
                vendor_part_id=event.index,
                part=_map_web_fetch_tool_result_block(current_block, self._provider_name),
            )
            return
        if isinstance(current_block, BetaMCPToolUseBlock):
            call_part = _map_mcp_server_use_block(current_block, self._provider_name)
            self._builtin_tool_calls[call_part.tool_call_id] = call_part

            args_json = call_part.args_as_json_str()
            # Drop the final ``{}}`` so we can add tool args deltas
            args_json_delta = args_json[:-3]
            assert args_json_delta.endswith('"tool_args":'), f'Expected {args_json_delta!r} to end in `"tool_args":`'

            yield self._parts_manager.handle_part(
                vendor_part_id=event.index,
                part=replace(call_part, args=None),
            )
            maybe_event = self._parts_manager.handle_tool_call_delta(
                vendor_part_id=event.index,
                args=args_json_delta,
            )
            if maybe_event is not None:
                yield maybe_event
            return
        if isinstance(current_block, BetaMCPToolResultBlock):
            mcp_call_part = self._builtin_tool_calls.get(current_block.tool_use_id)
            yield self._parts_manager.handle_part(
                vendor_part_id=event.index,
                part=_map_mcp_server_result_block(current_block, mcp_call_part, self._provider_name),
            )
            return
        if isinstance(current_block, BetaCompactionBlock):
            yield self._parts_manager.handle_part(
                vendor_part_id=event.index,
                part=CompactionPart(content=current_block.content, provider_name=self._provider_name),
            )
            return

    def _handle_content_block_delta(self, event: BetaRawContentBlockDeltaEvent) -> Iterator[ModelResponseStreamEvent]:
        delta = event.delta
        if isinstance(delta, BetaTextDelta):
            yield from self._parts_manager.handle_text_delta(vendor_part_id=event.index, content=delta.text)
            return
        if isinstance(delta, BetaThinkingDelta):
            yield from self._parts_manager.handle_thinking_delta(
                vendor_part_id=event.index,
                content=delta.thinking,
                provider_name=self._provider_name,
            )
            return
        if isinstance(delta, BetaSignatureDelta):
            yield from self._parts_manager.handle_thinking_delta(
                vendor_part_id=event.index,
                signature=delta.signature,
                provider_name=self._provider_name,
            )
            return
        if isinstance(delta, BetaInputJSONDelta):
            maybe_event = self._parts_manager.handle_tool_call_delta(
                vendor_part_id=event.index,
                args=delta.partial_json,
            )
            if maybe_event is not None:
                yield maybe_event
            return
        if isinstance(delta, BetaCompactionContentBlockDelta):
            if delta.content:
                # Re-emit part with updated content; replaces the initial block start part.
                yield self._parts_manager.handle_part(
                    vendor_part_id=event.index,
                    part=CompactionPart(content=delta.content, provider_name=self._provider_name),
                )
            return
        if isinstance(delta, BetaCitationsDelta):
            # TODO(upstream pydantic-ai): citations not yet wired through to IR events.
            return

    def _handle_content_block_stop(self, event: BetaRawContentBlockStopEvent) -> Iterator[ModelResponseStreamEvent]:
        if isinstance(self._current_block, BetaMCPToolUseBlock):
            maybe_event = self._parts_manager.handle_tool_call_delta(
                vendor_part_id=event.index,
                args="}",
            )
            if maybe_event is not None:
                yield maybe_event
        self._current_block = None
