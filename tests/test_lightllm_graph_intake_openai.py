"""Tests for the OpenAI Chat Completion SSE → IR intake FSM.

The production FSM is async; ``_OpenAIFSMAdapter`` wraps it with a
one-fresh-loop-per-call sync surface for tests (the persistent-loop bridge
lives in :class:`SSEPipeline` for production).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

import pytest
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import (
    FinishReason,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.models import ModelRequestParameters

from ccproxy.lightllm.graph.openai_intake import OpenAIResponseIntakeFSM

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class _IntakeLike(Protocol):
    """Sync-callable surface around the async FSM intake."""

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


class _OpenAIFSMAdapter:
    """Sync-facing adapter around the async :class:`OpenAIResponseIntakeFSM`.

    The production FSM is async (the persistent-loop bridge lives in
    :class:`SSEPipeline`). For tests, one fresh asyncio loop per
    ``feed`` / ``close`` call is fine — tests aren't on a hot path.
    """

    def __init__(self, *, model: str, request_params: ModelRequestParameters) -> None:
        self._fsm = OpenAIResponseIntakeFSM(model=model, request_params=request_params)

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
    """Factory for the FSM intake wrapped in a sync adapter."""

    def _make(*, model: str = "gpt-4o") -> _IntakeLike:
        return _OpenAIFSMAdapter(model=model, request_params=ModelRequestParameters())

    return _make


# ---------------------------------------------------------------------------
# Helpers — build synthetic SSE byte streams that match the OpenAI wire shape
# ---------------------------------------------------------------------------


def _chunk(
    *,
    chunk_id: str = "chatcmpl-abc",
    model: str = "gpt-4o",
    created: int = 1700000000,
    delta: dict[str, object] | None = None,
    finish_reason: str | None = None,
    index: int = 0,
    no_choices: bool = False,
) -> dict[str, object]:
    """Build a single ChatCompletionChunk-shape dict for SSE serialization."""
    choice: dict[str, object] = {"index": index, "delta": delta or {}, "finish_reason": finish_reason}
    choices: list[dict[str, object]] = [] if no_choices else [choice]
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }


def _sse(payload: object) -> bytes:
    """Serialize one chunk dict (or sentinel ``[DONE]``) as an SSE frame."""
    if payload == "[DONE]":
        return b"data: [DONE]\n\n"
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _build_stream(payloads: list[object]) -> bytes:
    """Concatenate a list of chunk dicts (and an optional ``[DONE]``) into SSE bytes."""
    return b"".join(_sse(p) for p in payloads)


def _feed_all(intake: _IntakeLike, data: bytes) -> list[ModelResponseStreamEvent]:
    events = list(intake.feed(data))
    events.extend(intake.close())
    return events


def _chunked(data: bytes, size: int) -> Iterator[bytes]:
    for offset in range(0, len(data), size):
        yield data[offset : offset + size]


def _text_starts(events: list[ModelResponseStreamEvent]) -> list[tuple[PartStartEvent, TextPart]]:
    """Return (event, part) tuples for every PartStartEvent carrying a TextPart."""
    out: list[tuple[PartStartEvent, TextPart]] = []
    for e in events:
        if isinstance(e, PartStartEvent) and isinstance(e.part, TextPart):
            out.append((e, e.part))
    return out


def _text_deltas(
    events: list[ModelResponseStreamEvent],
) -> list[tuple[PartDeltaEvent, TextPartDelta]]:
    """Return (event, delta) tuples for every PartDeltaEvent carrying a TextPartDelta."""
    out: list[tuple[PartDeltaEvent, TextPartDelta]] = []
    for e in events:
        if isinstance(e, PartDeltaEvent) and isinstance(e.delta, TextPartDelta):
            out.append((e, e.delta))
    return out


def _tool_starts(events: list[ModelResponseStreamEvent]) -> list[tuple[PartStartEvent, ToolCallPart]]:
    """Return (event, part) tuples for every PartStartEvent carrying a ToolCallPart."""
    out: list[tuple[PartStartEvent, ToolCallPart]] = []
    for e in events:
        if isinstance(e, PartStartEvent) and isinstance(e.part, ToolCallPart):
            out.append((e, e.part))
    return out


def _tool_deltas(
    events: list[ModelResponseStreamEvent],
) -> list[tuple[PartDeltaEvent, ToolCallPartDelta]]:
    """Return (event, delta) tuples for every PartDeltaEvent carrying a ToolCallPartDelta."""
    out: list[tuple[PartDeltaEvent, ToolCallPartDelta]] = []
    for e in events:
        if isinstance(e, PartDeltaEvent) and isinstance(e.delta, ToolCallPartDelta):
            out.append((e, e.delta))
    return out


# ---------------------------------------------------------------------------
# 1) Synthetic SSE roundtrip — single chunk
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_role_then_text_then_finish_then_done(self, intake_factory: _IntakeFactory) -> None:
        stream = _build_stream(
            [
                _chunk(delta={"role": "assistant"}),
                _chunk(delta={"content": "Hello"}),
                _chunk(delta={"content": ", world"}),
                _chunk(delta={}, finish_reason="stop"),
                "[DONE]",
            ]
        )
        intake = intake_factory()
        events = _feed_all(intake, stream)

        # Exactly one TextPart start and one delta event
        starts = _text_starts(events)
        deltas = _text_deltas(events)
        assert len(starts) == 1
        assert starts[0][1].content == "Hello"
        assert len(deltas) == 1
        assert deltas[0][1].content_delta == ", world"

        # Provider metadata captured on intake state
        assert intake.provider_response_id == "chatcmpl-abc"
        assert intake.finish_reason == "stop"
        assert intake.provider_details == {"finish_reason": "stop"}

    def test_model_reassignment_from_chunk(self, intake_factory: _IntakeFactory) -> None:
        """Chunk's ``model`` field overrides the constructor value."""
        stream = _build_stream([_chunk(model="gpt-4o-2024-08-06", delta={"content": "x"}), "[DONE]"])
        intake = intake_factory(model="gpt-4o")
        list(intake.feed(stream))
        assert intake._model == "gpt-4o-2024-08-06"

    def test_empty_choices_chunk_skipped(self, intake_factory: _IntakeFactory) -> None:
        """Usage-only final chunks (no choices) don't produce IR events."""
        stream = _build_stream([_chunk(delta={"content": "hi"}), _chunk(no_choices=True), "[DONE]"])
        intake = intake_factory()
        events = _feed_all(intake, stream)
        assert len(_text_starts(events)) == 1


# ---------------------------------------------------------------------------
# 2) Chunk-boundary robustness — same IR events regardless of byte slicing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryCase:
    name: str
    chunk_size: int | None  # None = single-feed


BOUNDARY_CASES: list[BoundaryCase] = [
    BoundaryCase(name="single_chunk", chunk_size=None),
    BoundaryCase(name="byte_at_a_time", chunk_size=1),
    BoundaryCase(name="sixteen_byte_blocks", chunk_size=16),
]


class TestChunkBoundaryRobustness:
    @pytest.mark.parametrize("case", [pytest.param(c, id=c.name) for c in BOUNDARY_CASES])
    def test_text_stream_invariant(self, case: BoundaryCase, intake_factory: _IntakeFactory) -> None:
        stream = _build_stream(
            [
                _chunk(delta={"role": "assistant"}),
                _chunk(delta={"content": "abc"}),
                _chunk(delta={"content": "def"}),
                _chunk(delta={"content": "ghi"}),
                _chunk(delta={}, finish_reason="stop"),
                "[DONE]",
            ]
        )
        intake = intake_factory()
        events: list[ModelResponseStreamEvent] = []
        if case.chunk_size is None:
            events.extend(intake.feed(stream))
        else:
            for slice_ in _chunked(stream, case.chunk_size):
                events.extend(intake.feed(slice_))
        events.extend(intake.close())

        text_starts = _text_starts(events)
        text_deltas = _text_deltas(events)
        assert len(text_starts) == 1
        assert text_starts[0][1].content == "abc"
        # Two subsequent content deltas merge into TextPartDelta events
        assert [delta.content_delta for _, delta in text_deltas] == ["def", "ghi"]
        assert intake.finish_reason == "stop"


# ---------------------------------------------------------------------------
# 3) [DONE] terminator handling
# ---------------------------------------------------------------------------


class TestDoneTerminator:
    def test_done_sets_terminated_flag(self, intake_factory: _IntakeFactory) -> None:
        stream = _build_stream([_chunk(delta={"content": "x"}), "[DONE]"])
        intake = intake_factory()
        list(intake.feed(stream))
        assert intake._terminated is True

    def test_bytes_after_done_are_ignored(self, intake_factory: _IntakeFactory) -> None:
        """Any frame arriving after ``[DONE]`` must not be processed."""
        before = _build_stream([_chunk(delta={"content": "x"}), "[DONE]"])
        after = _sse(_chunk(delta={"content": "should_be_dropped"}))
        intake = intake_factory()
        first_events = list(intake.feed(before))
        # Feed garbage post-DONE; intake should swallow it
        second_events = list(intake.feed(after))
        assert second_events == []
        # The "should_be_dropped" content must not appear in any event
        for _, part in _text_starts(first_events):
            assert "should_be_dropped" not in part.content
        for _, delta in _text_deltas(first_events):
            assert delta.content_delta is None or "should_be_dropped" not in delta.content_delta

    def test_done_split_across_feed_calls(self, intake_factory: _IntakeFactory) -> None:
        """``data: [DONE]\\n\\n`` arriving across feed() boundaries still terminates."""
        stream = _build_stream([_chunk(delta={"content": "x"}), "[DONE]"])
        intake = intake_factory()
        # Split mid-[DONE] frame
        split_at = stream.index(b"[DONE]") + 2
        list(intake.feed(stream[:split_at]))
        list(intake.feed(stream[split_at:]))
        assert intake._terminated is True

    def test_upstream_raw_bytes_includes_done_frame(self, intake_factory: _IntakeFactory) -> None:
        stream = _build_stream([_chunk(delta={"content": "x"}), "[DONE]"])
        intake = intake_factory()
        list(intake.feed(stream))
        assert bytes(intake.upstream_raw_bytes) == stream


# ---------------------------------------------------------------------------
# 4) Tool call sequence — chunked function arguments
# ---------------------------------------------------------------------------


class TestToolCallStream:
    def test_chunked_tool_call_arguments(self, intake_factory: _IntakeFactory) -> None:
        """First chunk carries id+name; subsequent chunks deliver partial JSON args."""
        tool_call_chunks: list[object] = [
            _chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ],
                }
            ),
            _chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"loca'},
                        }
                    ],
                }
            ),
            _chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": 'tion": "SF"}'},
                        }
                    ],
                }
            ),
            _chunk(delta={}, finish_reason="tool_calls"),
            "[DONE]",
        ]
        stream = _build_stream(tool_call_chunks)
        intake = intake_factory()
        events = _feed_all(intake, stream)

        tool_starts = _tool_starts(events)
        tool_deltas = _tool_deltas(events)
        # Exactly one tool-call PartStartEvent when name + id appear
        assert len(tool_starts) == 1
        start_part = tool_starts[0][1]
        assert start_part.tool_name == "get_weather"
        assert start_part.tool_call_id == "call_abc"

        # Subsequent argument deltas land as PartDeltaEvents
        deltas_concat = "".join(
            delta.args_delta if isinstance(delta.args_delta, str) else "" for _, delta in tool_deltas
        )
        # All argument pieces accumulated in the deltas
        assert "loca" in deltas_concat or "loca" in start_part.args_as_json_str()
        assert intake.finish_reason == "tool_call"

    def test_multiple_concurrent_tool_calls_differ_by_index(self, intake_factory: _IntakeFactory) -> None:
        """Two tool calls in the same stream are routed by ``index``."""
        chunks: list[object] = [
            _chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "fn_a", "arguments": ""},
                        }
                    ],
                }
            ),
            _chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "fn_b", "arguments": ""},
                        }
                    ],
                }
            ),
            "[DONE]",
        ]
        stream = _build_stream(chunks)
        intake = intake_factory()
        events = _feed_all(intake, stream)

        tool_starts = _tool_starts(events)
        assert len(tool_starts) == 2
        names = {part.tool_name for _, part in tool_starts}
        assert names == {"fn_a", "fn_b"}


# ---------------------------------------------------------------------------
# 5) Refusal handling
# ---------------------------------------------------------------------------


class TestRefusal:
    def test_refusal_text_stashed_and_terminates_content(self, intake_factory: _IntakeFactory) -> None:
        """Refusal blocks text emission and stashes the refusal string in provider_details."""
        stream = _build_stream(
            [
                _chunk(delta={"role": "assistant"}),
                _chunk(delta={"refusal": "I cannot "}),
                _chunk(delta={"refusal": "comply."}),
                _chunk(delta={}, finish_reason="content_filter"),
                "[DONE]",
            ]
        )
        intake = intake_factory()
        events = _feed_all(intake, stream)

        # No TextPart emitted because refusal short-circuits the delta dispatch
        assert _text_starts(events) == []
        assert intake._has_refusal is True
        assert intake._refusal_text == "I cannot comply."
        assert intake.finish_reason == "content_filter"
        assert intake.provider_details is not None
        assert intake.provider_details["refusal"] == "I cannot comply."
        # When refusal is set, raw finish_reason from chunks is dropped from provider_details
        assert "finish_reason" not in intake.provider_details


# ---------------------------------------------------------------------------
# 6) upstream_raw_bytes tee
# ---------------------------------------------------------------------------


class TestRawBytesTee:
    def test_tee_accumulates_every_fed_byte(self, intake_factory: _IntakeFactory) -> None:
        stream = _build_stream(
            [
                _chunk(delta={"content": "alpha"}),
                _chunk(delta={"content": "beta"}),
                "[DONE]",
            ]
        )
        intake = intake_factory()
        for slice_ in _chunked(stream, 7):
            list(intake.feed(slice_))
        assert bytes(intake.upstream_raw_bytes) == stream

    def test_tee_accumulates_bytes_after_done(self, intake_factory: _IntakeFactory) -> None:
        """Raw tee includes bytes received after the terminator — they're recorded but unprocessed."""
        before = _build_stream([_chunk(delta={"content": "x"}), "[DONE]"])
        trailing = b"garbage trailing bytes"
        intake = intake_factory()
        list(intake.feed(before))
        list(intake.feed(trailing))
        assert bytes(intake.upstream_raw_bytes) == before + trailing


# ---------------------------------------------------------------------------
# Unparseable frame resilience
# ---------------------------------------------------------------------------


class TestParseErrors:
    def test_invalid_json_frame_skipped(self, intake_factory: _IntakeFactory) -> None:
        bad = b"data: {not valid json\n\n"
        good = _sse(_chunk(delta={"content": "hi"}))
        intake = intake_factory()
        events = list(intake.feed(bad + good))
        starts = _text_starts(events)
        assert len(starts) == 1
        assert starts[0][1].content == "hi"

    def test_frame_without_data_line_skipped(self, intake_factory: _IntakeFactory) -> None:
        """SSE comments / event lines without data are ignored."""
        stream = b": heartbeat\n\n" + _sse(_chunk(delta={"content": "hi"}))
        intake = intake_factory()
        events = list(intake.feed(stream))
        assert len(_text_starts(events)) == 1


# ---------------------------------------------------------------------------
# Wire-format edge cases — CRLF separators, multi-choice
# ---------------------------------------------------------------------------


class TestWireFormat:
    def test_crlf_separator(self, intake_factory: _IntakeFactory) -> None:
        """Some servers emit ``\\r\\n\\r\\n`` between SSE frames."""
        chunk = _chunk(delta={"content": "crlf"})
        frame = b"data: " + json.dumps(chunk).encode() + b"\r\n\r\n"
        intake = intake_factory()
        events = list(intake.feed(frame))
        starts = _text_starts(events)
        assert len(starts) == 1
        assert starts[0][1].content == "crlf"

    def test_multi_choice_chunk_emits_warning_and_uses_first(
        self, intake_factory: _IntakeFactory, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Multi-choice chunks process only ``choices[0]`` with a warning."""
        chunk_dict = {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "delta": {"content": "first"}, "finish_reason": None},
                {"index": 1, "delta": {"content": "second"}, "finish_reason": None},
            ],
        }
        stream = _sse(chunk_dict)
        intake = intake_factory()
        # Both implementations emit the warning under their own logger name;
        # capture root-level WARNING to stay implementation-agnostic.
        with caplog.at_level("WARNING"):
            events = list(intake.feed(stream))
        starts = _text_starts(events)
        assert len(starts) == 1
        assert starts[0][1].content == "first"
        assert any("2 choices" in r.message for r in caplog.records)
