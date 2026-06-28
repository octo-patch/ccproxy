"""Tests for the OpenAI Responses SSE -> IR intake FSM."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import pytest
from openai.types import responses as resp
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import (
    FinishReason,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.models import ModelRequestParameters

from ccproxy.lightllm.graph.openai_responses_intake import OpenAIResponsesIntakeFSM


class _IntakeLike(Protocol):
    @property
    def upstream_raw_bytes(self) -> bytearray: ...

    @property
    def _terminated(self) -> bool: ...

    @property
    def _model(self) -> str: ...

    @property
    def _has_refusal(self) -> bool: ...

    @property
    def _refusal_text(self) -> str: ...

    @property
    def provider_response_id(self) -> str | None: ...

    @property
    def provider_details(self) -> dict[str, object] | None: ...

    @property
    def finish_reason(self) -> FinishReason | None: ...

    @property
    def parts_manager(self) -> ModelResponsePartsManager: ...

    def feed(self, data: bytes) -> Iterable[ModelResponseStreamEvent]: ...

    def close(self) -> Iterable[ModelResponseStreamEvent]: ...


class _ResponsesFSMAdapter:
    def __init__(self, *, model: str, request_params: ModelRequestParameters) -> None:
        self._fsm = OpenAIResponsesIntakeFSM(model=model, request_params=request_params)

    @property
    def parts_manager(self) -> ModelResponsePartsManager:
        return self._fsm.parts_manager

    @property
    def upstream_raw_bytes(self) -> bytearray:
        return self._fsm.upstream_raw_bytes

    @property
    def _terminated(self) -> bool:
        return self._fsm._terminated

    @property
    def _model(self) -> str:
        return self._fsm._model

    @property
    def _has_refusal(self) -> bool:
        return self._fsm._has_refusal

    @property
    def _refusal_text(self) -> str:
        return self._fsm._refusal_text

    @property
    def provider_response_id(self) -> str | None:
        return self._fsm.provider_response_id

    @property
    def provider_details(self) -> dict[str, object] | None:
        return self._fsm.provider_details

    @property
    def finish_reason(self) -> FinishReason | None:
        return self._fsm.finish_reason

    def feed(self, data: bytes) -> list[ModelResponseStreamEvent]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._fsm.feed(data))
        finally:
            loop.close()

    def close(self) -> list[ModelResponseStreamEvent]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._fsm.close())
        finally:
            loop.close()


_IntakeFactory = Callable[..., _IntakeLike]


@pytest.fixture
def intake_factory() -> _IntakeFactory:
    def _make(*, model: str = "gpt-5") -> _IntakeLike:
        return _ResponsesFSMAdapter(model=model, request_params=ModelRequestParameters())

    return _make


def _base_response(*, response_id: str = "resp_001", model: str = "gpt-5") -> resp.Response:
    return resp.Response(
        id=response_id,
        model=model,
        object="response",
        created_at=1704067200,
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _sse(event: Any) -> bytes:
    payload = event.model_dump_json(exclude_none=True)
    event_name = event.type
    return f"event: {event_name}\ndata: {payload}\n\n".encode()


def _build_stream(events: list[Any], *, done: bool = False) -> bytes:
    out = b"".join(_sse(event) for event in events)
    if done:
        out += b"data: [DONE]\n\n"
    return out


def _feed_all(intake: _IntakeLike, data: bytes) -> list[ModelResponseStreamEvent]:
    events = list(intake.feed(data))
    events.extend(intake.close())
    return events


def _chunked(data: bytes, size: int) -> Iterator[bytes]:
    for offset in range(0, len(data), size):
        yield data[offset : offset + size]


def _text_starts(events: list[ModelResponseStreamEvent]) -> list[tuple[PartStartEvent, TextPart]]:
    out: list[tuple[PartStartEvent, TextPart]] = []
    for event in events:
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            out.append((event, event.part))
    return out


def _text_deltas(
    events: list[ModelResponseStreamEvent],
) -> list[tuple[PartDeltaEvent, TextPartDelta]]:
    out: list[tuple[PartDeltaEvent, TextPartDelta]] = []
    for event in events:
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            out.append((event, event.delta))
    return out


def _tool_starts(events: list[ModelResponseStreamEvent]) -> list[tuple[PartStartEvent, ToolCallPart]]:
    out: list[tuple[PartStartEvent, ToolCallPart]] = []
    for event in events:
        if isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart):
            out.append((event, event.part))
    return out


def _tool_deltas(
    events: list[ModelResponseStreamEvent],
) -> list[tuple[PartDeltaEvent, ToolCallPartDelta]]:
    out: list[tuple[PartDeltaEvent, ToolCallPartDelta]] = []
    for event in events:
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, ToolCallPartDelta):
            out.append((event, event.delta))
    return out


def _thinking_starts(events: list[ModelResponseStreamEvent]) -> list[tuple[PartStartEvent, ThinkingPart]]:
    out: list[tuple[PartStartEvent, ThinkingPart]] = []
    for event in events:
        if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
            out.append((event, event.part))
    return out


def _thinking_deltas(
    events: list[ModelResponseStreamEvent],
) -> list[tuple[PartDeltaEvent, ThinkingPartDelta]]:
    out: list[tuple[PartDeltaEvent, ThinkingPartDelta]] = []
    for event in events:
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
            out.append((event, event.delta))
    return out


class TestTextLifecycle:
    def test_text_stream_maps_to_text_part(self, intake_factory: _IntakeFactory) -> None:
        base = _base_response()
        stream = _build_stream(
            [
                resp.ResponseCreatedEvent(response=base, type="response.created", sequence_number=0),
                resp.ResponseInProgressEvent(response=base, type="response.in_progress", sequence_number=1),
                resp.ResponseOutputItemAddedEvent(
                    item=resp.ResponseOutputMessage.model_construct(
                        id="msg_001",
                        content=[],
                        role="assistant",
                        status="in_progress",
                        type="message",
                        phase="final_answer",
                    ),
                    output_index=0,
                    type="response.output_item.added",
                    sequence_number=2,
                ),
                resp.ResponseContentPartAddedEvent(
                    content_index=0,
                    item_id="msg_001",
                    output_index=0,
                    part=resp.ResponseOutputText(text="", type="output_text", annotations=[]),
                    type="response.content_part.added",
                    sequence_number=3,
                ),
                resp.ResponseTextDeltaEvent(
                    content_index=0,
                    delta="Hello",
                    item_id="msg_001",
                    output_index=0,
                    type="response.output_text.delta",
                    sequence_number=4,
                    logprobs=[],
                ),
                resp.ResponseTextDeltaEvent(
                    content_index=0,
                    delta=", world",
                    item_id="msg_001",
                    output_index=0,
                    type="response.output_text.delta",
                    sequence_number=5,
                    logprobs=[],
                ),
                resp.ResponseTextDoneEvent(
                    content_index=0,
                    item_id="msg_001",
                    output_index=0,
                    text="Hello, world",
                    type="response.output_text.done",
                    sequence_number=6,
                    logprobs=[],
                ),
                resp.ResponseOutputItemDoneEvent(
                    item=resp.ResponseOutputMessage.model_construct(
                        id="msg_001",
                        content=[resp.ResponseOutputText(text="Hello, world", type="output_text", annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                        phase="final_answer",
                    ),
                    output_index=0,
                    type="response.output_item.done",
                    sequence_number=7,
                ),
                resp.ResponseCompletedEvent(
                    response=base.model_copy(update={"status": "completed"}),
                    type="response.completed",
                    sequence_number=8,
                ),
            ],
            done=True,
        )
        intake = intake_factory()
        events = _feed_all(intake, stream)

        starts = _text_starts(events)
        deltas = _text_deltas(events)
        assert len(starts) == 1
        assert starts[0][1].content == "Hello"
        assert [delta.content_delta for _, delta in deltas] == [", world", ""]
        assert intake.provider_response_id == "resp_001"
        assert intake.finish_reason == "stop"
        assert intake.provider_details == {"finish_reason": "completed"}
        [part] = intake.parts_manager.get_parts()
        assert isinstance(part, TextPart)
        assert part.provider_details == {"phase": "final_answer"}


class TestFunctionCalls:
    def test_function_call_arguments_stream(self, intake_factory: _IntakeFactory) -> None:
        base = _base_response()
        stream = _build_stream(
            [
                resp.ResponseCreatedEvent(response=base, type="response.created", sequence_number=0),
                resp.ResponseOutputItemAddedEvent(
                    item=resp.ResponseFunctionToolCall(
                        id="fc_001",
                        call_id="call_1",
                        name="lookup",
                        arguments="",
                        type="function_call",
                        status="in_progress",
                    ),
                    output_index=0,
                    type="response.output_item.added",
                    sequence_number=1,
                ),
                resp.ResponseFunctionCallArgumentsDeltaEvent(
                    delta='{"q":',
                    item_id="fc_001",
                    output_index=0,
                    type="response.function_call_arguments.delta",
                    sequence_number=2,
                ),
                resp.ResponseFunctionCallArgumentsDeltaEvent(
                    delta='"hi"}',
                    item_id="fc_001",
                    output_index=0,
                    type="response.function_call_arguments.delta",
                    sequence_number=3,
                ),
                resp.ResponseFunctionCallArgumentsDoneEvent(
                    arguments='{"q":"hi"}',
                    item_id="fc_001",
                    name="lookup",
                    output_index=0,
                    type="response.function_call_arguments.done",
                    sequence_number=4,
                ),
                resp.ResponseOutputItemDoneEvent(
                    item=resp.ResponseFunctionToolCall(
                        id="fc_001",
                        call_id="call_1",
                        name="lookup",
                        arguments='{"q":"hi"}',
                        type="function_call",
                        status="completed",
                    ),
                    output_index=0,
                    type="response.output_item.done",
                    sequence_number=5,
                ),
                resp.ResponseCompletedEvent(
                    response=base.model_copy(update={"status": "completed"}),
                    type="response.completed",
                    sequence_number=6,
                ),
            ],
        )
        intake = intake_factory()
        events = _feed_all(intake, stream)

        starts = _tool_starts(events)
        deltas = _tool_deltas(events)
        assert len(starts) == 1
        assert starts[0][1].tool_name == "lookup"
        assert starts[0][1].tool_call_id == "call_1"
        assert [delta.args_delta for _, delta in deltas] == ['{"q":', '"hi"}']
        [part] = intake.parts_manager.get_parts()
        assert isinstance(part, ToolCallPart)
        assert part.args == '{"q":"hi"}'


class TestReasoning:
    def test_reasoning_summary_and_signature(self, intake_factory: _IntakeFactory) -> None:
        base = _base_response()
        stream = _build_stream(
            [
                resp.ResponseCreatedEvent(response=base, type="response.created", sequence_number=0),
                resp.ResponseOutputItemAddedEvent(
                    item=resp.ResponseReasoningItem(
                        id="rs_001",
                        summary=[],
                        type="reasoning",
                        content=[],
                        encrypted_content=None,
                        status="in_progress",
                    ),
                    output_index=0,
                    type="response.output_item.added",
                    sequence_number=1,
                ),
                resp.ResponseReasoningSummaryTextDeltaEvent(
                    delta="First ",
                    item_id="rs_001",
                    output_index=0,
                    sequence_number=2,
                    summary_index=0,
                    type="response.reasoning_summary_text.delta",
                ),
                resp.ResponseReasoningSummaryTextDeltaEvent(
                    delta="step.",
                    item_id="rs_001",
                    output_index=0,
                    sequence_number=3,
                    summary_index=0,
                    type="response.reasoning_summary_text.delta",
                ),
                resp.ResponseOutputItemDoneEvent(
                    item=resp.ResponseReasoningItem(
                        id="rs_001",
                        summary=[],
                        type="reasoning",
                        content=[],
                        encrypted_content="sealed",
                        status="completed",
                    ),
                    output_index=0,
                    type="response.output_item.done",
                    sequence_number=4,
                ),
            ],
        )
        intake = intake_factory()
        events = _feed_all(intake, stream)

        starts = _thinking_starts(events)
        deltas = _thinking_deltas(events)
        assert len(starts) == 1
        assert starts[0][1].content == "First "
        assert [delta.content_delta for _, delta in deltas] == ["step.", None]
        assert deltas[-1][1].signature_delta == "sealed"
        [part] = intake.parts_manager.get_parts()
        assert isinstance(part, ThinkingPart)
        assert part.content == "First step."
        assert part.signature == "sealed"


class TestRefusal:
    def test_refusal_records_content_filter_metadata(self, intake_factory: _IntakeFactory) -> None:
        base = _base_response()
        stream = _build_stream(
            [
                resp.ResponseCreatedEvent(response=base, type="response.created", sequence_number=0),
                resp.ResponseRefusalDeltaEvent(
                    content_index=0,
                    delta="No ",
                    item_id="msg_001",
                    output_index=0,
                    type="response.refusal.delta",
                    sequence_number=1,
                ),
                resp.ResponseRefusalDoneEvent(
                    content_index=0,
                    item_id="msg_001",
                    output_index=0,
                    refusal="No thanks.",
                    type="response.refusal.done",
                    sequence_number=2,
                ),
                resp.ResponseCompletedEvent(
                    response=base.model_copy(update={"status": "completed"}),
                    type="response.completed",
                    sequence_number=3,
                ),
            ],
        )
        intake = intake_factory()
        events = _feed_all(intake, stream)

        assert events == []
        assert intake._has_refusal is True
        assert intake.finish_reason == "content_filter"
        assert intake.provider_details == {"refusal": "No thanks."}


@dataclass(frozen=True)
class BoundaryCase:
    name: str
    chunk_size: int | None


BOUNDARY_CASES = [
    BoundaryCase(name="single_chunk", chunk_size=None),
    BoundaryCase(name="byte_at_a_time", chunk_size=1),
    BoundaryCase(name="seventeen_byte_blocks", chunk_size=17),
]


class TestChunkBoundaries:
    @pytest.mark.parametrize("case", [pytest.param(c, id=c.name) for c in BOUNDARY_CASES])
    def test_text_stream_invariant(self, case: BoundaryCase, intake_factory: _IntakeFactory) -> None:
        base = _base_response()
        stream = _build_stream(
            [
                resp.ResponseCreatedEvent(response=base, type="response.created", sequence_number=0),
                resp.ResponseTextDeltaEvent(
                    content_index=0,
                    delta="a",
                    item_id="msg_001",
                    output_index=0,
                    type="response.output_text.delta",
                    sequence_number=1,
                    logprobs=[],
                ),
                resp.ResponseTextDeltaEvent(
                    content_index=0,
                    delta="b",
                    item_id="msg_001",
                    output_index=0,
                    type="response.output_text.delta",
                    sequence_number=2,
                    logprobs=[],
                ),
                resp.ResponseCompletedEvent(
                    response=base.model_copy(update={"status": "completed"}),
                    type="response.completed",
                    sequence_number=3,
                ),
            ],
            done=True,
        )
        intake = intake_factory()
        events: list[ModelResponseStreamEvent] = []
        if case.chunk_size is None:
            events.extend(intake.feed(stream))
        else:
            for slice_ in _chunked(stream, case.chunk_size):
                events.extend(intake.feed(slice_))
        events.extend(intake.close())

        assert [part.content for _, part in _text_starts(events)] == ["a"]
        assert [delta.content_delta for _, delta in _text_deltas(events)] == ["b"]
        assert intake._terminated is True


class TestSilentDropTelemetry:
    """Never-silently-drop diagnostics for the OpenAI Responses intake.

    Drives the real async FSM directly so the state-level telemetry counters
    are observable.
    """

    @staticmethod
    def _run(fsm: OpenAIResponsesIntakeFSM, data: bytes) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(fsm.feed(data))
            loop.run_until_complete(fsm.close())
        finally:
            loop.close()

    def test_envelope_only_stream_emits_no_ir_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """A stream of only response-envelope events (created/in_progress/completed)
        with no output items parses frames but emits ZERO IR events — the intake
        must WARN, never go silent."""
        base = _base_response()
        fsm = OpenAIResponsesIntakeFSM(model="gpt-5", request_params=ModelRequestParameters())
        stream = _build_stream(
            [
                resp.ResponseCreatedEvent(response=base, type="response.created", sequence_number=0),
                resp.ResponseInProgressEvent(response=base, type="response.in_progress", sequence_number=1),
                resp.ResponseCompletedEvent(response=base, type="response.completed", sequence_number=2),
            ],
            done=True,
        )
        with caplog.at_level("WARNING", logger="ccproxy.lightllm.graph.openai_responses_intake"):
            self._run(fsm, stream)
        assert fsm.state.frames_seen >= 1
        assert fsm.state.emitted_events == 0
        assert "produced NO IR events" in caplog.text

    def test_clean_text_stream_emits_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        base = _base_response()
        fsm = OpenAIResponsesIntakeFSM(model="gpt-5", request_params=ModelRequestParameters())
        stream = _build_stream(
            [
                resp.ResponseCreatedEvent(response=base, type="response.created", sequence_number=0),
                resp.ResponseTextDeltaEvent(
                    content_index=0,
                    delta="Hello",
                    item_id="msg_001",
                    output_index=0,
                    type="response.output_text.delta",
                    sequence_number=1,
                    logprobs=[],
                ),
                resp.ResponseCompletedEvent(response=base, type="response.completed", sequence_number=2),
            ],
            done=True,
        )
        with caplog.at_level("WARNING", logger="ccproxy.lightllm.graph.openai_responses_intake"):
            self._run(fsm, stream)
        assert fsm.state.emitted_events >= 1
        assert "produced NO IR events" not in caplog.text

    def test_unparseable_frame_is_counted(self, caplog: pytest.LogCaptureFixture) -> None:
        fsm = OpenAIResponsesIntakeFSM(model="gpt-5", request_params=ModelRequestParameters())
        stream = b"event: garbage\ndata: {not json\n\n"
        with caplog.at_level("DEBUG", logger="ccproxy.lightllm.graph.openai_responses_intake"):
            self._run(fsm, stream)
        assert fsm.state.frames_unparseable >= 1
        assert "unparseable frame" in caplog.text
