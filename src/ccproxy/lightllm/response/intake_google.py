"""Google ``streamGenerateContent`` SSE bytes → pydantic-ai IR events (sync).

Transliterates ``pydantic_ai.models.google.GeminiStreamedResponse._get_event_iterator``
into a synchronous, bytes-driven dispatcher that drives
``ModelResponsePartsManager`` and emits ``ModelResponseStreamEvent`` objects as
each SSE event arrives.

Operates on bytes that have ALREADY been unwrapped by ccproxy's
``EnvelopeUnwrapStream`` — i.e. payloads of the shape::

    data: {"candidates": [...], "usageMetadata": {...}, "modelVersion": "..."}

(NOT the cloudcode-pa ``{response: {...}}`` envelope).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING
from uuid import uuid4

from google.genai.types import GenerateContentResponse
from pydantic import TypeAdapter
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import BinaryContent, FilePart

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelResponseStreamEvent
    from pydantic_ai.models import ModelRequestParameters


logger = logging.getLogger(__name__)

_RESPONSE_ADAPTER: TypeAdapter[GenerateContentResponse] = TypeAdapter(GenerateContentResponse)


class GoogleResponseIntake:
    """Sync dispatcher: Google ``streamGenerateContent`` SSE → IR events."""

    name = "google"

    def __init__(self, *, model: str, request_params: ModelRequestParameters) -> None:
        self._model = model
        self._request_params = request_params
        self._parts_manager = ModelResponsePartsManager()
        self._sse_buffer = bytearray()
        self.upstream_raw_bytes = bytearray()

    def feed(self, data: bytes) -> Iterator[ModelResponseStreamEvent]:
        """Process incoming bytes; yield zero-or-more IR events."""
        if not data:
            return
        self.upstream_raw_bytes.extend(data)
        self._sse_buffer.extend(data)
        for chunk in self._drain_sse_events():
            yield from self._dispatch_chunk(chunk)

    def close(self) -> Iterator[ModelResponseStreamEvent]:
        """Stream end. Drain any complete remaining event in the buffer."""
        if self._sse_buffer:
            # Some servers omit the trailing blank line on the last event.
            tail = bytes(self._sse_buffer)
            self._sse_buffer.clear()
            chunk = self._parse_event(tail)
            if chunk is not None:
                yield from self._dispatch_chunk(chunk)

    def _drain_sse_events(self) -> Iterator[GenerateContentResponse]:
        """Frame the buffer into complete SSE events, yielding parsed chunks.

        Accepts both ``\r\n\r\n`` and ``\n\n`` event terminators; whichever
        boundary appears first wins. Partial frames remain in the buffer for
        the next ``feed`` call.
        """
        while True:
            crlf = self._sse_buffer.find(b"\r\n\r\n")
            lf = self._sse_buffer.find(b"\n\n")
            if crlf == -1 and lf == -1:
                return
            if crlf != -1 and (lf == -1 or crlf < lf):
                event = bytes(self._sse_buffer[:crlf])
                del self._sse_buffer[: crlf + 4]
            else:
                event = bytes(self._sse_buffer[:lf])
                del self._sse_buffer[: lf + 2]
            chunk = self._parse_event(event)
            if chunk is not None:
                yield chunk

    def _parse_event(self, event: bytes) -> GenerateContentResponse | None:
        """Parse a single SSE event into a ``GenerateContentResponse``."""
        payloads: list[bytes] = []
        for raw_line in event.split(b"\n"):
            line = raw_line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            payloads.append(payload)
        if not payloads:
            return None
        raw = b"\n".join(payloads)
        try:
            return _RESPONSE_ADAPTER.validate_json(raw)
        except Exception:
            logger.debug("google intake: skipping unparseable SSE event", exc_info=True)
            return None

    def _dispatch_chunk(self, chunk: GenerateContentResponse) -> Iterator[ModelResponseStreamEvent]:
        """Sync transliteration of ``GeminiStreamedResponse._get_event_iterator``."""
        if not chunk.candidates:
            return
        candidate = chunk.candidates[0]
        if candidate.content is None or candidate.content.parts is None:
            return
        for part in candidate.content.parts:
            if part.text is not None:
                if not part.text:
                    continue
                yield from self._parts_manager.handle_text_delta(
                    vendor_part_id=None,
                    content=part.text,
                )
            elif part.function_call is not None:
                event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=uuid4(),
                    tool_name=part.function_call.name,
                    args=part.function_call.args,
                    tool_call_id=part.function_call.id,
                )
                if event is not None:
                    yield event
            elif part.inline_data is not None:
                data = part.inline_data.data
                mime_type = part.inline_data.mime_type
                if not data or not mime_type:
                    logger.debug("google intake: skipping inlineData part with missing data/mime_type")
                    continue
                binary = BinaryContent(data=data, media_type=mime_type)
                yield self._parts_manager.handle_part(
                    vendor_part_id=uuid4(),
                    part=FilePart(content=BinaryContent.narrow_type(binary)),
                )
            elif part.function_response is not None:
                logger.warning("google intake: unexpected functionResponse part in upstream response; skipping")
                continue
