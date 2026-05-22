"""Google ``streamGenerateContent`` SSE bytes → pydantic-ai IR events via FSM.

Pydantic-graph FSM port of
:class:`ccproxy.lightllm.response.intake_google.GoogleResponseIntake`. One
graph run per :meth:`GoogleResponseIntakeFSM.feed` call: bytes are appended
to the SSE buffer, complete SSE frames are drained, each frame's ``data:``
payload JSON is checked for the cloudcode-pa ``{response: {...}}`` envelope
and unwrapped if present, then validated into a typed
:class:`GenerateContentResponse`. Each chunk is wrapped in a dispatch
envelope, those envelopes are pushed onto an in-state queue, and the FSM
router drains the queue dispatching each envelope to a per-variant handler
step. Handler steps mutate ``state.parts_manager`` and append emitted
:class:`ModelResponseStreamEvent` objects to ``state.out_events``.

The behavioral contract matches
:mod:`ccproxy.lightllm.response.intake_google` byte-for-byte for unwrapped
input: same SSE framing rules (``\\r\\n\\r\\n`` and ``\\n\\n`` separators),
same dispatch ladder (text → function_call → inline_data → function_response
warning), same multi-part-per-chunk handling, same close-tail-buffer drain.

The cloudcode-pa envelope unwrap (previously done by
:class:`ccproxy.hooks.gemini_envelope.EnvelopeUnwrapStream` on streaming
flows and :func:`ccproxy.hooks.gemini_envelope.unwrap_buffered` on buffered
flows) is folded into :meth:`_parse_event`: if the parsed JSON is a dict
with exactly one key ``"response"`` whose value is a dict, the inner dict
is taken as the chunk payload. Otherwise the JSON is treated as the chunk
payload directly. This makes the FSM-driven path the single source of
truth for Gemini response handling.

The persistent-loop bridge between sync mitmproxy callables and this async
FSM lives in :class:`SSEPipeline` (Phase Q). For tests, the parametrize
fixture in ``tests/test_lightllm_response_intake_google.py`` wraps the
async FSM in a one-loop-per-call sync adapter.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from google.genai.types import GenerateContentResponse
from pydantic import TypeAdapter, ValidationError

# Private pydantic-ai imports — same justification as the matching note in
# ``response/intake_google.py``. We need byte-identical dispatch behavior
# and there is no public replacement.
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import BinaryContent, FilePart, ModelResponseStreamEvent
from pydantic_graph.beta import GraphBuilder, StepContext

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestParameters

logger = logging.getLogger(__name__)


_RESPONSE_ADAPTER: TypeAdapter[GenerateContentResponse] = TypeAdapter(GenerateContentResponse)


# ── Dispatch envelopes ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class _GenerateChunk:
    """Chunk carrying one ``GenerateContentResponse`` to dispatch through the parts loop."""

    chunk: GenerateContentResponse


class _FeedDone:
    """Marker returned by the router when the events queue is exhausted."""


# ── State ──────────────────────────────────────────────────────────────────


@dataclass
class _GoogleIntakeState:
    """FSM state for one Google intake graph run.

    The ``events_queue`` is the queue of dispatch envelopes drained from the
    SSE buffer *before* the graph run starts; the FSM router pops from it.
    The ``out_events`` list accumulates :class:`ModelResponseStreamEvent`
    instances emitted by handler steps; the terminal step returns it.
    ``parts_manager`` persists across feed calls so multi-feed reassembly
    works.
    """

    parts_manager: ModelResponsePartsManager
    events_queue: deque[Any] = field(default_factory=deque)
    out_events: list[ModelResponseStreamEvent] = field(default_factory=list)


# ── Graph ──────────────────────────────────────────────────────────────────


_g: GraphBuilder[
    _GoogleIntakeState, None, None, list[ModelResponseStreamEvent]
] = GraphBuilder(
    state_type=_GoogleIntakeState,
    output_type=list[ModelResponseStreamEvent],
)


@_g.step
async def frame_next_event(
    ctx: StepContext[_GoogleIntakeState, None, None],
) -> Any:
    """Router source: pop the next dispatch envelope from the queue, or signal end via :class:`_FeedDone`."""
    state = ctx.state
    if not state.events_queue:
        return _FeedDone()
    return state.events_queue.popleft()


@_g.step
async def handle_generate_chunk(
    ctx: StepContext[_GoogleIntakeState, None, _GenerateChunk],
) -> None:
    """Dispatch a ``GenerateContentResponse`` chunk to the parts manager.

    Sync transliteration of ``GeminiStreamedResponse._get_event_iterator``.
    """
    state = ctx.state
    chunk = ctx.inputs.chunk
    pm = state.parts_manager

    if not chunk.candidates:
        return
    candidate = chunk.candidates[0]
    if candidate.content is None or candidate.content.parts is None:
        return
    for part in candidate.content.parts:
        if part.text is not None:
            if not part.text:
                continue
            state.out_events.extend(
                pm.handle_text_delta(
                    vendor_part_id=None,
                    content=part.text,
                )
            )
        elif part.function_call is not None:
            event = pm.handle_tool_call_delta(
                vendor_part_id=uuid4(),
                tool_name=part.function_call.name,
                args=part.function_call.args,
                tool_call_id=part.function_call.id,
            )
            if event is not None:
                state.out_events.append(event)
        elif part.inline_data is not None:
            data = part.inline_data.data
            mime_type = part.inline_data.mime_type
            if not data or not mime_type:
                logger.debug(
                    "google intake: skipping inlineData part with missing data/mime_type"
                )
                continue
            binary = BinaryContent(data=data, media_type=mime_type)
            state.out_events.append(
                pm.handle_part(
                    vendor_part_id=uuid4(),
                    part=FilePart(content=BinaryContent.narrow_type(binary)),
                )
            )
        elif part.function_response is not None:
            logger.warning(
                "google intake: unexpected functionResponse part in upstream response; skipping"
            )
            continue


@_g.step
async def emit_done(
    ctx: StepContext[_GoogleIntakeState, None, _FeedDone],
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
        .branch(_g.match(_GenerateChunk).to(handle_generate_chunk))
    ),
    _g.edge_from(handle_generate_chunk).to(frame_next_event),
    _g.edge_from(emit_done).to(_g.end_node),
)


_intake_graph = _g.build()


# ── Public class ───────────────────────────────────────────────────────────


class GoogleResponseIntakeFSM:
    """Async pydantic-graph-driven Google ``streamGenerateContent`` SSE intake.

    Behavioral twin of
    :class:`ccproxy.lightllm.response.intake_google.GoogleResponseIntake`,
    re-expressed as a :mod:`pydantic_graph.beta` ``GraphBuilder`` FSM. One
    graph run per :meth:`feed` call drains all complete SSE frames buffered
    by that call into typed ``GenerateContentResponse`` chunks (transparently
    peeling off the cloudcode-pa ``{response: {...}}`` envelope when present),
    wraps each in a dispatch envelope, dispatches each to a handler step,
    and returns the accumulated IR events. Partial frames remain in the SSE
    buffer for the next call. ``parts_manager`` persists across calls.
    """

    name = "google"

    def __init__(self, *, model: str, request_params: ModelRequestParameters) -> None:
        self._model = model
        self._request_params = request_params
        self._sse_buffer = bytearray()
        self.upstream_raw_bytes = bytearray()
        self._state = _GoogleIntakeState(
            parts_manager=ModelResponsePartsManager(),
        )

    @property
    def parts_manager(self) -> ModelResponsePartsManager:
        """Expose the underlying parts manager for tests and downstream renderers."""
        return self._state.parts_manager

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
        """Stream end. Drain any complete remaining event in the buffer.

        Some servers omit the trailing blank line on the last event; this
        catches them by treating the tail as a complete frame.
        """
        if not self._sse_buffer:
            return []
        tail = bytes(self._sse_buffer)
        self._sse_buffer.clear()
        envelope = self._parse_event(tail)
        if envelope is None:
            return []
        self._state.events_queue.append(envelope)
        return await _intake_graph.run(state=self._state)

    def _drain_sse_envelopes(self) -> Iterator[_GenerateChunk]:
        """Frame SSE events from ``self._sse_buffer``; validate surviving frames into a dispatch envelope.

        Handles both ``\\r\\n\\r\\n`` (industry standard) and ``\\n\\n`` (some servers)
        separators; partial frames remain buffered for the next ``feed`` call.
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
            envelope = self._parse_event(event)
            if envelope is not None:
                yield envelope

    @staticmethod
    def _parse_event(event: bytes) -> _GenerateChunk | None:
        """Parse a single SSE event into a ``_GenerateChunk``.

        Concatenates all ``data:`` lines into one JSON payload, peels off
        the cloudcode-pa ``{response: {...}}`` envelope if present, and
        validates the result into a typed ``GenerateContentResponse``.
        """
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
            parsed: Any = json.loads(raw)
        except (ValueError, TypeError):
            logger.debug("google intake: skipping unparseable SSE event", exc_info=True)
            return None
        # cloudcode-pa wraps each chunk in {response: {...}}; standard Gemini
        # generateContent emits the chunk directly. Detect by checking for a
        # single ``response`` key wrapping a dict — anything else falls
        # through as the chunk itself.
        if (
            isinstance(parsed, dict)
            and len(parsed) == 1
            and "response" in parsed
            and isinstance(parsed["response"], dict)
        ):
            parsed = parsed["response"]
        try:
            chunk = _RESPONSE_ADAPTER.validate_python(parsed)
        except ValidationError:
            logger.debug("google intake: skipping unparseable SSE event", exc_info=True)
            return None
        return _GenerateChunk(chunk=chunk)
