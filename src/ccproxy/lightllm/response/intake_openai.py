"""OpenAI Chat Completion SSE → pydantic-ai IR events (sync).

Synchronous transliteration of pydantic-ai's
``OpenAIStreamedResponse._get_event_iterator``
(``pydantic_ai/models/openai.py:3183-3234``) plus the per-choice
mapping hooks (``_map_text_delta``, ``_map_tool_call_delta``).
Drives ``ModelResponsePartsManager`` directly without any async
machinery so it can be invoked from mitmproxy's synchronous
``flow.response.stream`` callable.

Wire shape:
- SSE frames separated by ``\\r\\n\\r\\n`` or ``\\n\\n``.
- Each frame is a ``data: <ChatCompletionChunk JSON>`` line.
- A ``data: [DONE]`` frame terminates the stream — it is NOT JSON
  and must be filtered before validation.
- ``chunk.choices`` is conventionally length-1; we handle only
  ``choices[0]`` and log a warning on multi-choice chunks.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

from openai.types.chat import ChatCompletionChunk
from pydantic import TypeAdapter, ValidationError
from pydantic_ai._parts_manager import ModelResponsePartsManager

if TYPE_CHECKING:
    from openai.types.chat import chat_completion_chunk
    from pydantic_ai.messages import FinishReason, ModelResponseStreamEvent
    from pydantic_ai.models import ModelRequestParameters


logger = logging.getLogger(__name__)


_CHUNK_ADAPTER: TypeAdapter[ChatCompletionChunk] = TypeAdapter(ChatCompletionChunk)


_CHAT_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_call",
    "content_filter": "content_filter",
    "function_call": "tool_call",
}


class OpenAIResponseIntake:
    """SSE bytes → pydantic-ai IR events for an OpenAI Chat Completions stream."""

    name = "openai"

    def __init__(self, *, model: str, request_params: ModelRequestParameters) -> None:
        self._parts_manager = ModelResponsePartsManager()
        self._request_params = request_params
        self._model = model
        self._sse_buffer = bytearray()
        self.upstream_raw_bytes = bytearray()
        self._terminated = False
        self._has_refusal = False
        self._refusal_text = ""
        self.provider_response_id: str | None = None
        self.finish_reason: FinishReason | None = None
        self.provider_details: dict[str, object] | None = None

    def feed(self, data: bytes) -> Iterator[ModelResponseStreamEvent]:
        """Buffer incoming bytes, frame SSE events, yield IR events."""
        self.upstream_raw_bytes.extend(data)
        if self._terminated:
            return
        self._sse_buffer.extend(data)
        for chunk in self._drain_sse_events():
            yield from self._dispatch_chunk(chunk)

    def close(self) -> Iterator[ModelResponseStreamEvent]:
        """Stream end. Refusal text is stashed on ``provider_details`` per pydantic-ai."""
        if self._refusal_text:
            self.provider_details = {**(self.provider_details or {}), "refusal": self._refusal_text}
        yield from ()

    def _drain_sse_events(self) -> Iterator[ChatCompletionChunk]:
        """Frame the SSE buffer; handle ``[DONE]`` terminator; validate each chunk."""
        while True:
            if self._terminated:
                return
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
            payload = _extract_data_payload(frame)
            if payload is None:
                continue
            if payload == b"[DONE]":
                self._terminated = True
                return
            try:
                yield _CHUNK_ADAPTER.validate_json(payload)
            except ValidationError:
                logger.debug("openai intake: skipping unparseable chunk: %r", payload)

    def _dispatch_chunk(self, chunk: ChatCompletionChunk) -> Iterator[ModelResponseStreamEvent]:
        """Per-chunk dispatch — mirrors ``OpenAIStreamedResponse._get_event_iterator``."""
        if chunk.id:
            self.provider_response_id = chunk.id
        if chunk.model:
            self._model = chunk.model

        if not chunk.choices:
            return
        if len(chunk.choices) > 1:
            logger.warning(
                "openai intake: chunk has %d choices; only choices[0] is processed",
                len(chunk.choices),
            )
        choice = chunk.choices[0]
        # Azure OpenAI + async content filter has been observed to emit None deltas;
        # pydantic validates `delta` as non-None on Choice but the openai SDK's loose
        # constructor lets it through. Defend at runtime; type-system sees this as
        # unreachable so suppress the diagnostic.
        if choice.delta is None:  # type: ignore[unreachable]
            return  # type: ignore[unreachable]

        if choice.delta.refusal:
            self._has_refusal = True
            self.finish_reason = "content_filter"
            self._refusal_text += choice.delta.refusal
            return

        if (raw_finish_reason := choice.finish_reason) and not self._has_refusal:
            self.finish_reason = _CHAT_FINISH_REASON_MAP.get(raw_finish_reason)

        if provider_details := _map_provider_details(choice):
            if self._has_refusal:
                provider_details.pop("finish_reason", None)
            self.provider_details = {**(self.provider_details or {}), **provider_details}

        yield from self._map_text_delta(choice)
        yield from self._map_tool_call_delta(choice)

    def _map_text_delta(self, choice: chat_completion_chunk.Choice) -> Iterator[ModelResponseStreamEvent]:
        content = choice.delta.content
        if content:
            yield from self._parts_manager.handle_text_delta(
                vendor_part_id="content",
                content=content,
            )

    def _map_tool_call_delta(self, choice: chat_completion_chunk.Choice) -> Iterator[ModelResponseStreamEvent]:
        for dtc in choice.delta.tool_calls or []:
            fn = dtc.function
            tool_name = fn.name if fn is not None else None
            args = fn.arguments if fn is not None else None
            maybe_event = self._parts_manager.handle_tool_call_delta(
                vendor_part_id=dtc.index,
                tool_name=tool_name,
                args=args,
                tool_call_id=dtc.id,
            )
            if maybe_event is not None:
                yield maybe_event


def _extract_data_payload(frame: bytes) -> bytes | None:
    """Return the payload of the first ``data:`` line in a frame, or ``None``."""
    for line in frame.split(b"\n"):
        stripped = line.strip()
        if stripped.startswith(b"data:"):
            return stripped[5:].strip() or None
    return None


def _map_provider_details(choice: chat_completion_chunk.Choice) -> dict[str, object] | None:
    """Mirror of pydantic-ai's ``_map_provider_details`` for a single chunk choice.

    We don't carry logprobs across the wire boundary (they ride the
    chunks unmodified), so this only surfaces the raw ``finish_reason``.
    """
    details: dict[str, object] = {}
    if raw := choice.finish_reason:
        details["finish_reason"] = raw
    return details or None
