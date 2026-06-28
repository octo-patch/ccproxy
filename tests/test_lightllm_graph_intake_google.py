"""Tests for the Google ``streamGenerateContent`` SSE → IR intake FSM.

Validates SSE framing, multi-part chunk dispatch, function-call deltas,
inline binary data, the ``upstream_raw_bytes`` tee for downstream
inspectors, and the cloudcode-pa ``{response: {...}}`` envelope unwrap.

The production FSM is async; ``_GoogleFSMAdapter`` wraps it with a
one-fresh-loop-per-call sync surface for tests (the persistent-loop bridge
lives in :class:`SSEPipeline` for production).
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

import pytest
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import (
    BinaryContent,
    FilePart,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestParameters

from ccproxy.lightllm.graph.google_intake import GoogleResponseIntakeFSM

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class _IntakeLike(Protocol):
    """Sync-callable surface around the async FSM intake."""

    @property
    def upstream_raw_bytes(self) -> bytearray: ...

    @property
    def parts_manager(self) -> ModelResponsePartsManager: ...

    def feed(self, data: bytes) -> Iterable[ModelResponseStreamEvent]: ...

    def close(self) -> Iterable[ModelResponseStreamEvent]: ...


class _GoogleFSMAdapter:
    """Sync-facing adapter around the async :class:`GoogleResponseIntakeFSM`.

    The production FSM is async (the persistent-loop bridge lives in
    :class:`SSEPipeline`). For tests, one fresh asyncio loop per
    ``feed`` / ``close`` call is fine — tests aren't on a hot path.
    """

    def __init__(self, *, model: str, request_params: ModelRequestParameters) -> None:
        self._fsm = GoogleResponseIntakeFSM(model=model, request_params=request_params)

    @property
    def parts_manager(self) -> ModelResponsePartsManager:
        return self._fsm.parts_manager

    @property
    def upstream_raw_bytes(self) -> bytearray:
        return self._fsm.upstream_raw_bytes

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

    def _make(*, model: str = "gemini-2.5-flash") -> _IntakeLike:
        return _GoogleFSMAdapter(model=model, request_params=ModelRequestParameters())

    return _make


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    *,
    parts: list[dict[str, object]] | None = None,
    finish_reason: str | None = "STOP",
    no_candidates: bool = False,
    role: str = "model",
    model_version: str = "gemini-2.5-flash",
    usage: dict[str, int] | None = None,
) -> dict[str, object]:
    """Build a single ``GenerateContentResponse``-shape dict."""
    body: dict[str, object] = {"modelVersion": model_version}
    if usage is not None:
        body["usageMetadata"] = usage
    if no_candidates:
        return body
    candidate: dict[str, object] = {
        "content": {"role": role, "parts": parts or []},
    }
    if finish_reason is not None:
        candidate["finishReason"] = finish_reason
    body["candidates"] = [candidate]
    return body


def _sse(payload: dict[str, object]) -> bytes:
    """Serialize one chunk dict as an SSE frame."""
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _build_stream(payloads: list[dict[str, object]]) -> bytes:
    return b"".join(_sse(p) for p in payloads)


def _feed_all(intake: _IntakeLike, data: bytes) -> list[ModelResponseStreamEvent]:
    events = list(intake.feed(data))
    events.extend(intake.close())
    return events


def _chunked(data: bytes, size: int) -> Iterator[bytes]:
    for offset in range(0, len(data), size):
        yield data[offset : offset + size]


# ---------------------------------------------------------------------------
# 1) Synthetic SSE roundtrip — text-only response
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_single_text_chunk(self, intake_factory: _IntakeFactory) -> None:
        stream = _build_stream([_chunk(parts=[{"text": "Hello"}], finish_reason="STOP")])
        intake = intake_factory()
        events = _feed_all(intake, stream)

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        deltas = [e for e in events if isinstance(e, PartDeltaEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].part, TextPart)
        assert starts[0].part.content == "Hello"
        assert deltas == []

    def test_multi_chunk_text_concatenation(self, intake_factory: _IntakeFactory) -> None:
        stream = _build_stream(
            [
                _chunk(parts=[{"text": "Hello"}], finish_reason=None),
                _chunk(parts=[{"text": ", "}], finish_reason=None),
                _chunk(parts=[{"text": "world"}], finish_reason="STOP"),
            ]
        )
        intake = intake_factory()
        events = _feed_all(intake, stream)

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        deltas = [e for e in events if isinstance(e, PartDeltaEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].part, TextPart)
        assert starts[0].part.content == "Hello"
        assert [d.delta.content_delta for d in deltas if isinstance(d.delta, TextPartDelta)] == [", ", "world"]

    def test_empty_text_part_is_skipped(self, intake_factory: _IntakeFactory) -> None:
        """Per ``GeminiStreamedResponse``, empty text deltas are ignored."""
        stream = _build_stream(
            [
                _chunk(parts=[{"text": ""}], finish_reason=None),
                _chunk(parts=[{"text": "ok"}], finish_reason="STOP"),
            ]
        )
        intake = intake_factory()
        events = _feed_all(intake, stream)

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].part, TextPart)
        assert starts[0].part.content == "ok"

    def test_chunk_without_candidates_is_skipped(self, intake_factory: _IntakeFactory) -> None:
        """Usage-only final chunks (no candidates) don't produce IR events."""
        stream = _build_stream(
            [
                _chunk(parts=[{"text": "hi"}], finish_reason=None),
                _chunk(
                    no_candidates=True,
                    usage={"promptTokenCount": 3, "candidatesTokenCount": 1},
                ),
            ]
        )
        intake = intake_factory()
        events = _feed_all(intake, stream)
        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1


# ---------------------------------------------------------------------------
# 2) Chunk-boundary robustness — same IR events regardless of byte slicing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryCase:
    name: str
    chunk_size: int | None  # None = single-feed


BOUNDARY_CASES: list[BoundaryCase] = [
    BoundaryCase(name="single_feed", chunk_size=None),
    BoundaryCase(name="byte_at_a_time", chunk_size=1),
    BoundaryCase(name="sixteen_byte_blocks", chunk_size=16),
    BoundaryCase(name="hundred_byte_blocks", chunk_size=100),
]


class TestChunkBoundaryRobustness:
    @pytest.mark.parametrize("case", [pytest.param(c, id=c.name) for c in BOUNDARY_CASES])
    def test_text_stream_invariant(self, case: BoundaryCase, intake_factory: _IntakeFactory) -> None:
        stream = _build_stream(
            [
                _chunk(parts=[{"text": "abc"}], finish_reason=None),
                _chunk(parts=[{"text": "def"}], finish_reason=None),
                _chunk(parts=[{"text": "ghi"}], finish_reason="STOP"),
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

        text_starts = [e for e in events if isinstance(e, PartStartEvent) and isinstance(e.part, TextPart)]
        text_deltas = [e for e in events if isinstance(e, PartDeltaEvent) and isinstance(e.delta, TextPartDelta)]
        assert len(text_starts) == 1
        first_part = text_starts[0].part
        assert isinstance(first_part, TextPart)
        assert first_part.content == "abc"
        delta_contents = [d.delta.content_delta for d in text_deltas if isinstance(d.delta, TextPartDelta)]
        assert delta_contents == ["def", "ghi"]

    def test_lf_only_event_terminator(self, intake_factory: _IntakeFactory) -> None:
        """SSE servers that emit ``\\n\\n`` (not ``\\r\\n\\r\\n``) still frame correctly."""
        payload = _chunk(parts=[{"text": "Hi"}], finish_reason="STOP")
        stream = b"data: " + json.dumps(payload).encode() + b"\n\n"
        intake = intake_factory()
        events = _feed_all(intake, stream)
        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].part, TextPart)
        assert starts[0].part.content == "Hi"

    def test_crlf_event_terminator(self, intake_factory: _IntakeFactory) -> None:
        """SSE wire-standard ``\\r\\n\\r\\n`` terminator is also accepted."""
        payload = _chunk(parts=[{"text": "Hi"}], finish_reason="STOP")
        stream = b"data: " + json.dumps(payload).encode() + b"\r\n\r\n"
        intake = intake_factory()
        events = _feed_all(intake, stream)
        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].part, TextPart)
        assert starts[0].part.content == "Hi"


# ---------------------------------------------------------------------------
# 3) Function call response
# ---------------------------------------------------------------------------


class TestFunctionCall:
    def test_single_function_call(self, intake_factory: _IntakeFactory) -> None:
        stream = _build_stream(
            [
                _chunk(
                    parts=[
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"city": "Tokyo"},
                                "id": "call_abc",
                            }
                        }
                    ],
                    finish_reason="STOP",
                )
            ]
        )
        intake = intake_factory()
        events = _feed_all(intake, stream)

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        part = starts[0].part
        assert isinstance(part, ToolCallPart)
        assert part.tool_name == "get_weather"
        assert part.args == {"city": "Tokyo"}
        assert part.tool_call_id == "call_abc"

    def test_text_then_function_call_emits_both_parts(self, intake_factory: _IntakeFactory) -> None:
        """A chunk with both text and functionCall parts yields both events in order."""
        stream = _build_stream(
            [
                _chunk(
                    parts=[
                        {"text": "Looking that up..."},
                        {
                            "functionCall": {
                                "name": "search",
                                "args": {"q": "weather"},
                                "id": "c1",
                            }
                        },
                    ],
                    finish_reason="STOP",
                )
            ]
        )
        intake = intake_factory()
        events = _feed_all(intake, stream)

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 2
        assert isinstance(starts[0].part, TextPart)
        assert starts[0].part.content == "Looking that up..."
        assert isinstance(starts[1].part, ToolCallPart)
        assert starts[1].part.tool_name == "search"
        assert starts[1].part.args == {"q": "weather"}
        assert starts[1].part.tool_call_id == "c1"

    def test_function_call_without_id(self, intake_factory: _IntakeFactory) -> None:
        """``id`` is optional in Gemini's functionCall shape."""
        stream = _build_stream(
            [
                _chunk(
                    parts=[
                        {
                            "functionCall": {
                                "name": "no_id_tool",
                                "args": {"x": 1},
                            }
                        }
                    ],
                    finish_reason="STOP",
                )
            ]
        )
        intake = intake_factory()
        events = _feed_all(intake, stream)

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        part = starts[0].part
        assert isinstance(part, ToolCallPart)
        assert part.tool_name == "no_id_tool"
        assert part.args == {"x": 1}


# ---------------------------------------------------------------------------
# 4) Inline data (image) response
# ---------------------------------------------------------------------------


class TestInlineData:
    def test_inline_image_emits_file_part(self, intake_factory: _IntakeFactory) -> None:
        png_bytes = b"\x89PNG\r\n\x1a\nfake-image-data"
        b64 = base64.b64encode(png_bytes).decode()
        stream = _build_stream(
            [
                _chunk(
                    parts=[
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": b64,
                            }
                        }
                    ],
                    finish_reason="STOP",
                )
            ]
        )
        intake = intake_factory()
        events = _feed_all(intake, stream)

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        part = starts[0].part
        assert isinstance(part, FilePart)
        assert isinstance(part.content, BinaryContent)
        assert part.content.data == png_bytes
        assert part.content.media_type == "image/png"

    def test_inline_data_skipped_when_missing_mime(self, intake_factory: _IntakeFactory) -> None:
        """Defensive: an inlineData without mimeType is skipped rather than emitting a malformed FilePart."""
        # The google.genai validator rejects mimeType=None, so we use ``b64`` data
        # with an empty string mimeType (validator accepts) — intake should skip.
        b64 = base64.b64encode(b"x").decode()
        stream = _build_stream(
            [
                _chunk(
                    parts=[
                        {"inlineData": {"data": b64, "mimeType": ""}},
                        {"text": "fallback"},
                    ],
                    finish_reason="STOP",
                )
            ]
        )
        intake = intake_factory()
        events = _feed_all(intake, stream)

        # FilePart skipped; only the fallback text part emitted.
        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].part, TextPart)
        assert starts[0].part.content == "fallback"


# ---------------------------------------------------------------------------
# 5) upstream_raw_bytes tee
# ---------------------------------------------------------------------------


class TestUpstreamRawBytes:
    def test_tee_captures_every_byte(self, intake_factory: _IntakeFactory) -> None:
        stream = _build_stream(
            [
                _chunk(parts=[{"text": "abc"}], finish_reason=None),
                _chunk(parts=[{"text": "def"}], finish_reason="STOP"),
            ]
        )
        intake = intake_factory()
        _feed_all(intake, stream)
        assert bytes(intake.upstream_raw_bytes) == stream

    def test_tee_under_byte_at_a_time_feeding(self, intake_factory: _IntakeFactory) -> None:
        stream = _build_stream([_chunk(parts=[{"text": "hello"}], finish_reason="STOP")])
        intake = intake_factory()
        for slice_ in _chunked(stream, 1):
            list(intake.feed(slice_))
        list(intake.close())
        assert bytes(intake.upstream_raw_bytes) == stream

    def test_empty_feed_no_side_effects(self, intake_factory: _IntakeFactory) -> None:
        intake = intake_factory()
        events = list(intake.feed(b""))
        assert events == []
        assert bytes(intake.upstream_raw_bytes) == b""


# ---------------------------------------------------------------------------
# 6) Defensive paths
# ---------------------------------------------------------------------------


class TestDefensive:
    def test_function_response_is_skipped_with_warning(
        self, intake_factory: _IntakeFactory, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``functionResponse`` parts are client-side; if seen upstream we skip + log."""
        stream = _build_stream(
            [
                _chunk(
                    parts=[
                        {
                            "functionResponse": {
                                "name": "client_tool",
                                "response": {"value": 1},
                            }
                        },
                        {"text": "ok"},
                    ],
                    finish_reason="STOP",
                )
            ]
        )
        intake = intake_factory()
        with caplog.at_level("WARNING"):
            events = _feed_all(intake, stream)

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].part, TextPart)
        assert any("functionResponse" in r.message for r in caplog.records)

    def test_unparseable_json_payload_is_skipped(self, intake_factory: _IntakeFactory) -> None:
        bad = b"data: not-json\n\n"
        good = _sse(_chunk(parts=[{"text": "ok"}], finish_reason="STOP"))
        intake = intake_factory()
        events = _feed_all(intake, bad + good)
        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].part, TextPart)
        assert starts[0].part.content == "ok"


# ---------------------------------------------------------------------------
# 7) cloudcode-pa envelope unwrap
# ---------------------------------------------------------------------------


def _envelope(chunk: dict[str, object]) -> dict[str, object]:
    """Wrap a standard ``GenerateContentResponse`` dict in the cloudcode-pa envelope."""
    return {"response": chunk}


class TestEnvelopeUnwrap:
    """Cloudcode-pa wraps each chunk in ``{response: {...}}``; the FSM peels it transparently.

    The legacy intake operates on already-unwrapped bytes (envelope unwrap
    used to live in ``EnvelopeUnwrapStream`` / ``unwrap_buffered``). Folding
    that unwrap into the intake is the Phase N motivation, so the test here
    is FSM-only.
    """

    def test_envelope_wrapped_text_chunk_equivalent_to_bare(self) -> None:
        """A wrapped chunk produces the same IR events as the same chunk fed bare."""
        bare = _chunk(parts=[{"text": "Hello"}], finish_reason="STOP")
        wrapped = _envelope(bare)

        bare_intake = _GoogleFSMAdapter(model="gemini-2.5-flash", request_params=ModelRequestParameters())
        wrapped_intake = _GoogleFSMAdapter(model="gemini-2.5-flash", request_params=ModelRequestParameters())

        bare_events = _feed_all(bare_intake, _sse(bare))
        wrapped_events = _feed_all(wrapped_intake, _sse(wrapped))

        # Same number of events, same parts.
        assert len(bare_events) == len(wrapped_events)
        for be, we in zip(bare_events, wrapped_events, strict=True):
            assert type(be) is type(we)
            if isinstance(be, PartStartEvent) and isinstance(we, PartStartEvent):
                assert isinstance(be.part, TextPart)
                assert isinstance(we.part, TextPart)
                assert be.part.content == we.part.content
            elif isinstance(be, PartDeltaEvent) and isinstance(we, PartDeltaEvent):
                assert isinstance(be.delta, TextPartDelta)
                assert isinstance(we.delta, TextPartDelta)
                assert be.delta.content_delta == we.delta.content_delta

    def test_envelope_wrapped_function_call(self) -> None:
        """Function call chunks survive the unwrap intact."""
        bare = _chunk(
            parts=[
                {
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"city": "Tokyo"},
                        "id": "call_abc",
                    }
                }
            ],
            finish_reason="STOP",
        )
        wrapped = _envelope(bare)

        intake = _GoogleFSMAdapter(model="gemini-2.5-flash", request_params=ModelRequestParameters())
        events = _feed_all(intake, _sse(wrapped))

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        part = starts[0].part
        assert isinstance(part, ToolCallPart)
        assert part.tool_name == "get_weather"
        assert part.args == {"city": "Tokyo"}
        assert part.tool_call_id == "call_abc"

    def test_envelope_mixed_with_bare_in_same_stream(self) -> None:
        """Streams containing both wrapped and bare chunks (defensive) parse correctly."""
        bare_a = _chunk(parts=[{"text": "abc"}], finish_reason=None)
        bare_b = _chunk(parts=[{"text": "def"}], finish_reason="STOP")
        wrapped_a = _envelope(bare_a)
        stream = _sse(wrapped_a) + _sse(bare_b)

        intake = _GoogleFSMAdapter(model="gemini-2.5-flash", request_params=ModelRequestParameters())
        events = _feed_all(intake, stream)

        text_starts = [e for e in events if isinstance(e, PartStartEvent) and isinstance(e.part, TextPart)]
        text_deltas = [e for e in events if isinstance(e, PartDeltaEvent) and isinstance(e.delta, TextPartDelta)]
        assert len(text_starts) == 1
        first = text_starts[0].part
        assert isinstance(first, TextPart)
        assert first.content == "abc"
        delta_contents = [d.delta.content_delta for d in text_deltas if isinstance(d.delta, TextPartDelta)]
        assert delta_contents == ["def"]


class TestSilentDropTelemetry:
    """Never-silently-drop diagnostics for the Google intake.

    Drives the real async FSM directly so the state-level telemetry counters
    are observable.
    """

    @staticmethod
    def _run(fsm: GoogleResponseIntakeFSM, data: bytes) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(fsm.feed(data))
            loop.run_until_complete(fsm.close())
        finally:
            loop.close()

    def test_candidateless_stream_emits_no_ir_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Chunks with no candidates parse fine but emit ZERO IR events — the
        intake must WARN, never go silent."""
        fsm = GoogleResponseIntakeFSM(model="gemini-2.5-flash", request_params=ModelRequestParameters())
        stream = _sse(_chunk(no_candidates=True)) + _sse(_chunk(no_candidates=True))
        with caplog.at_level("WARNING", logger="ccproxy.lightllm.graph.google_intake"):
            self._run(fsm, stream)
        assert fsm.state.frames_seen >= 1
        assert fsm.state.emitted_events == 0
        assert "produced NO IR events" in caplog.text

    def test_clean_text_stream_emits_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        fsm = GoogleResponseIntakeFSM(model="gemini-2.5-flash", request_params=ModelRequestParameters())
        stream = _sse(_chunk(parts=[{"text": "hello"}]))
        with caplog.at_level("WARNING", logger="ccproxy.lightllm.graph.google_intake"):
            self._run(fsm, stream)
        assert fsm.state.emitted_events >= 1
        assert "produced NO IR events" not in caplog.text

    def test_unparseable_frame_is_counted(self, caplog: pytest.LogCaptureFixture) -> None:
        fsm = GoogleResponseIntakeFSM(model="gemini-2.5-flash", request_params=ModelRequestParameters())
        stream = b"data: not-json\n\n"
        with caplog.at_level("DEBUG", logger="ccproxy.lightllm.graph.google_intake"):
            self._run(fsm, stream)
        assert fsm.state.frames_unparseable >= 1
        assert "unparseable SSE event" in caplog.text
