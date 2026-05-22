"""OpenAI Chat Completion SSE bytes → pydantic-ai IR events via FSM.

Pydantic-graph FSM port of
:class:`ccproxy.lightllm.response.intake_openai.OpenAIResponseIntake`. One
graph run per :meth:`OpenAIResponseIntakeFSM.feed` call: bytes are appended
to the SSE buffer, complete SSE frames are drained, the ``[DONE]`` sentinel
flips a terminator flag, surviving frames are validated into typed
:class:`ChatCompletionChunk` instances and wrapped in dispatch envelopes,
those envelopes are pushed onto an in-state queue, and the FSM router drains
the queue dispatching each envelope to a per-variant handler step. Handler
steps mutate ``state.parts_manager`` and append emitted
:class:`ModelResponseStreamEvent` objects to ``state.out_events``.

Unlike Anthropic's string-discriminated SSE union, OpenAI's wire is a single
``chat.completion.chunk`` envelope with optional fields on ``choices[0].delta``.
The intake wraps each post-validation chunk in one of three frozen
dispatch envelopes — ``_RefusalChunk`` (refusal short-circuits text), the
generic ``_StandardChunk`` (text + tool_calls), and ``_EmptyChoicesChunk``
(usage-only final chunks). The router routes by Python type, mirroring the
Anthropic FSM topology.

The behavioral contract matches
:mod:`ccproxy.lightllm.response.intake_openai` byte-for-byte: same SSE
framing rules, same ``[DONE]`` terminator, same dispatch ladder, same
``finish_reason`` mapping, same refusal handling, same multi-choice warning,
same provider-details collection.

The persistent-loop bridge between sync mitmproxy callables and this async
FSM lives in :class:`SSEPipeline` (Phase Q). For tests, the parametrize
fixture in ``tests/test_lightllm_response_intake_openai.py`` wraps the
async FSM in a one-loop-per-call sync adapter.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from openai.types.chat import ChatCompletionChunk
from pydantic import TypeAdapter, ValidationError

# Private pydantic-ai imports — see the matching note in
# ``response/intake_openai.py``. We need byte-identical dispatch behavior
# and there is no public replacement.
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_graph import GraphBuilder, StepContext

if TYPE_CHECKING:
    from openai.types.chat import chat_completion_chunk
    from pydantic_ai.messages import FinishReason
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


# ── Dispatch envelopes ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class _RefusalChunk:
    """Chunk where ``choices[0].delta.refusal`` is set — short-circuit text emission."""

    chunk: ChatCompletionChunk


@dataclass(frozen=True)
class _StandardChunk:
    """Chunk carrying a normal delta (text content or tool_calls or empty)."""

    chunk: ChatCompletionChunk


@dataclass(frozen=True)
class _EmptyChoicesChunk:
    """Usage-only chunk with ``choices == []`` — no IR emission, but provider id/model still update."""

    chunk: ChatCompletionChunk


class _FeedDone:
    """Marker returned by the router when the events queue is exhausted."""


# ── State ──────────────────────────────────────────────────────────────────


@dataclass
class _OpenAIIntakeState:
    """FSM state for one OpenAI intake graph run.

    The ``events_queue`` is the queue of dispatch envelopes drained from the
    SSE buffer *before* the graph run starts; the FSM router pops from it.
    The ``out_events`` list accumulates :class:`ModelResponseStreamEvent`
    instances emitted by handler steps; the terminal step returns it.
    ``parts_manager`` and the stream-level metadata fields persist across
    feed calls so multi-feed reassembly works.
    """

    parts_manager: ModelResponsePartsManager
    model: str
    has_refusal: bool = False
    refusal_text: str = ""
    finish_reason: FinishReason | None = None
    provider_response_id: str | None = None
    provider_details: dict[str, object] | None = None
    events_queue: deque[Any] = field(default_factory=deque)
    out_events: list[ModelResponseStreamEvent] = field(default_factory=list)


# ── Graph ──────────────────────────────────────────────────────────────────


_g: GraphBuilder[
    _OpenAIIntakeState, None, None, list[ModelResponseStreamEvent]
] = GraphBuilder(
    state_type=_OpenAIIntakeState,
    output_type=list[ModelResponseStreamEvent],
)


@_g.step
async def frame_next_event(
    ctx: StepContext[_OpenAIIntakeState, None, None],
) -> Any:
    """Router source: pop the next dispatch envelope from the queue, or signal end."""
    state = ctx.state
    if not state.events_queue:
        return _FeedDone()
    return state.events_queue.popleft()


def _absorb_chunk_metadata(state: _OpenAIIntakeState, chunk: ChatCompletionChunk) -> None:
    """Update stream-level metadata (id, model) from any chunk."""
    if chunk.id:
        state.provider_response_id = chunk.id
    if chunk.model:
        state.model = chunk.model


def _map_provider_details(choice: chat_completion_chunk.Choice) -> dict[str, object] | None:
    """Mirror of pydantic-ai's ``_map_provider_details`` for a single chunk choice.

    We don't carry logprobs across the wire boundary (they ride the
    chunks unmodified), so this only surfaces the raw ``finish_reason``.
    """
    details: dict[str, object] = {}
    if raw := choice.finish_reason:
        details["finish_reason"] = raw
    return details or None


@_g.step
async def handle_empty_choices(
    ctx: StepContext[_OpenAIIntakeState, None, _EmptyChoicesChunk],
) -> None:
    """Usage-only chunks: absorb id/model, no IR event."""
    _absorb_chunk_metadata(ctx.state, ctx.inputs.chunk)


@_g.step
async def handle_refusal(
    ctx: StepContext[_OpenAIIntakeState, None, _RefusalChunk],
) -> None:
    """Refusal short-circuits text emission and stashes refusal text on state."""
    state = ctx.state
    chunk = ctx.inputs.chunk
    _absorb_chunk_metadata(state, chunk)
    choice = chunk.choices[0]
    # The dispatch wrapped this in ``_RefusalChunk`` only if delta.refusal was truthy.
    state.has_refusal = True
    state.finish_reason = "content_filter"
    state.refusal_text += choice.delta.refusal or ""


@_g.step
async def handle_standard_chunk(
    ctx: StepContext[_OpenAIIntakeState, None, _StandardChunk],
) -> None:
    """Standard chunk: dispatch text deltas + tool_call deltas to the parts manager."""
    state = ctx.state
    chunk = ctx.inputs.chunk
    _absorb_chunk_metadata(state, chunk)
    choice = chunk.choices[0]

    if (raw_finish_reason := choice.finish_reason) and not state.has_refusal:
        state.finish_reason = _CHAT_FINISH_REASON_MAP.get(raw_finish_reason)

    if provider_details := _map_provider_details(choice):
        if state.has_refusal:
            provider_details.pop("finish_reason", None)
        state.provider_details = {**(state.provider_details or {}), **provider_details}

    content = choice.delta.content
    if content:
        state.out_events.extend(
            state.parts_manager.handle_text_delta(
                vendor_part_id="content",
                content=content,
            )
        )

    for dtc in choice.delta.tool_calls or []:
        fn = dtc.function
        tool_name = fn.name if fn is not None else None
        args = fn.arguments if fn is not None else None
        maybe_event = state.parts_manager.handle_tool_call_delta(
            vendor_part_id=dtc.index,
            tool_name=tool_name,
            args=args,
            tool_call_id=dtc.id,
        )
        if maybe_event is not None:
            state.out_events.append(maybe_event)


@_g.step
async def emit_done(
    ctx: StepContext[_OpenAIIntakeState, None, _FeedDone],
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
        .branch(_g.match(_EmptyChoicesChunk).to(handle_empty_choices))
        .branch(_g.match(_RefusalChunk).to(handle_refusal))
        .branch(_g.match(_StandardChunk).to(handle_standard_chunk))
    ),
    _g.edge_from(
        handle_empty_choices,
        handle_refusal,
        handle_standard_chunk,
    ).to(frame_next_event),
    _g.edge_from(emit_done).to(_g.end_node),
)


_intake_graph = _g.build()


# ── Public class ───────────────────────────────────────────────────────────


class OpenAIResponseIntakeFSM:
    """Async pydantic-graph-driven OpenAI Chat Completion SSE intake.

    Behavioral twin of
    :class:`ccproxy.lightllm.response.intake_openai.OpenAIResponseIntake`,
    re-expressed as a :mod:`pydantic_graph.beta` ``GraphBuilder`` FSM. One
    graph run per :meth:`feed` call drains all complete SSE frames buffered
    by that call into typed OpenAI chunks, wraps each in a dispatch envelope,
    dispatches each to a handler step, and returns the accumulated IR events.
    Partial frames remain in the SSE buffer for the next call. ``parts_manager``
    and the stream-level metadata persist across calls.
    """

    name = "openai"

    def __init__(self, *, model: str, request_params: ModelRequestParameters) -> None:
        self._request_params = request_params
        self._sse_buffer = bytearray()
        self.upstream_raw_bytes = bytearray()
        self._terminated = False
        # Stream-level fields live on the FSM state but are surfaced under the
        # same private names the legacy intake exposes so tests reaching for
        # them work unchanged.
        self._state = _OpenAIIntakeState(
            parts_manager=ModelResponsePartsManager(model_request_parameters=request_params),
            model=model,
        )

    @property
    def parts_manager(self) -> ModelResponsePartsManager:
        """Expose the underlying parts manager for tests and downstream renderers."""
        return self._state.parts_manager

    @property
    def _model(self) -> str:
        """Legacy attribute name — tests inspect this directly."""
        return self._state.model

    @property
    def _has_refusal(self) -> bool:
        return self._state.has_refusal

    @property
    def _refusal_text(self) -> str:
        return self._state.refusal_text

    @property
    def finish_reason(self) -> FinishReason | None:
        return self._state.finish_reason

    @property
    def provider_response_id(self) -> str | None:
        return self._state.provider_response_id

    @property
    def provider_details(self) -> dict[str, object] | None:
        return self._state.provider_details

    async def feed(self, data: bytes) -> list[ModelResponseStreamEvent]:
        """Buffer bytes, frame SSE events, drive the FSM, return emitted IR events."""
        self.upstream_raw_bytes.extend(data)
        if self._terminated:
            return []
        self._sse_buffer.extend(data)
        # Drain complete SSE frames into typed dispatch envelopes.
        for envelope in self._drain_sse_envelopes():
            self._state.events_queue.append(envelope)
        if not self._state.events_queue:
            return []
        result = await _intake_graph.run(state=self._state)
        return result

    async def close(self) -> list[ModelResponseStreamEvent]:
        """Stream end. Refusal text is stashed on ``provider_details`` per pydantic-ai."""
        if self._state.refusal_text:
            self._state.provider_details = {
                **(self._state.provider_details or {}),
                "refusal": self._state.refusal_text,
            }
        return []

    def _drain_sse_envelopes(self) -> Iterator[Any]:
        """Frame SSE events from ``self._sse_buffer``; flip ``_terminated`` on ``[DONE]``;
        validate surviving frames into a dispatch envelope.

        Handles both ``\\r\\n\\r\\n`` (industry standard) and ``\\n\\n`` (some servers)
        separators; partial frames remain buffered for the next ``feed`` call.
        """
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
                chunk = _CHUNK_ADAPTER.validate_json(payload)
            except ValidationError:
                logger.debug("openai intake: skipping unparseable chunk: %r", payload)
                continue
            envelope = self._classify_chunk(chunk)
            if envelope is not None:
                yield envelope

    def _classify_chunk(self, chunk: ChatCompletionChunk) -> Any:
        """Wrap a validated chunk in the matching dispatch envelope.

        Returns ``None`` to skip the chunk entirely (Azure-style ``delta=None`` defense).
        """
        if not chunk.choices:
            return _EmptyChoicesChunk(chunk=chunk)
        if len(chunk.choices) > 1:
            logger.warning(
                "openai intake: chunk has %d choices; only choices[0] is processed",
                len(chunk.choices),
            )
        choice = chunk.choices[0]
        if choice.delta.refusal:
            return _RefusalChunk(chunk=chunk)
        return _StandardChunk(chunk=chunk)


def _extract_data_payload(frame: bytes) -> bytes | None:
    """Return the payload of the first ``data:`` line in a frame, or ``None``."""
    for line in frame.split(b"\n"):
        stripped = line.strip()
        if stripped.startswith(b"data:"):
            return stripped[5:].strip() or None
    return None
