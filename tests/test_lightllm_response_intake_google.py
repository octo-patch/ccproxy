"""Tests for the Google ``streamGenerateContent`` SSE → IR intake.

Validates the synchronous transliteration of
``GeminiStreamedResponse._get_event_iterator``: SSE framing, multi-part
chunk dispatch, function-call deltas, inline binary data, and the
``upstream_raw_bytes`` tee for downstream inspectors.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
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

from ccproxy.lightllm.response.intake_google import GoogleResponseIntake

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


def _make_intake(*, model: str = "gemini-2.5-flash") -> GoogleResponseIntake:
    return GoogleResponseIntake(model=model, request_params=ModelRequestParameters())


def _feed_all(intake: GoogleResponseIntake, data: bytes) -> list[ModelResponseStreamEvent]:
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
    def test_single_text_chunk(self) -> None:
        stream = _build_stream([_chunk(parts=[{"text": "Hello"}], finish_reason="STOP")])
        intake = _make_intake()
        events = _feed_all(intake, stream)

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        deltas = [e for e in events if isinstance(e, PartDeltaEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].part, TextPart)
        assert starts[0].part.content == "Hello"
        assert deltas == []

    def test_multi_chunk_text_concatenation(self) -> None:
        stream = _build_stream(
            [
                _chunk(parts=[{"text": "Hello"}], finish_reason=None),
                _chunk(parts=[{"text": ", "}], finish_reason=None),
                _chunk(parts=[{"text": "world"}], finish_reason="STOP"),
            ]
        )
        intake = _make_intake()
        events = _feed_all(intake, stream)

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        deltas = [e for e in events if isinstance(e, PartDeltaEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].part, TextPart)
        assert starts[0].part.content == "Hello"
        assert [d.delta.content_delta for d in deltas if isinstance(d.delta, TextPartDelta)] == [", ", "world"]

    def test_empty_text_part_is_skipped(self) -> None:
        """Per ``GeminiStreamedResponse``, empty text deltas are ignored."""
        stream = _build_stream(
            [
                _chunk(parts=[{"text": ""}], finish_reason=None),
                _chunk(parts=[{"text": "ok"}], finish_reason="STOP"),
            ]
        )
        intake = _make_intake()
        events = _feed_all(intake, stream)

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].part, TextPart)
        assert starts[0].part.content == "ok"

    def test_chunk_without_candidates_is_skipped(self) -> None:
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
        intake = _make_intake()
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
    def test_text_stream_invariant(self, case: BoundaryCase) -> None:
        stream = _build_stream(
            [
                _chunk(parts=[{"text": "abc"}], finish_reason=None),
                _chunk(parts=[{"text": "def"}], finish_reason=None),
                _chunk(parts=[{"text": "ghi"}], finish_reason="STOP"),
            ]
        )
        intake = _make_intake()
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

    def test_lf_only_event_terminator(self) -> None:
        """SSE servers that emit ``\\n\\n`` (not ``\\r\\n\\r\\n``) still frame correctly."""
        payload = _chunk(parts=[{"text": "Hi"}], finish_reason="STOP")
        stream = b"data: " + json.dumps(payload).encode() + b"\n\n"
        intake = _make_intake()
        events = _feed_all(intake, stream)
        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].part, TextPart)
        assert starts[0].part.content == "Hi"

    def test_crlf_event_terminator(self) -> None:
        """SSE wire-standard ``\\r\\n\\r\\n`` terminator is also accepted."""
        payload = _chunk(parts=[{"text": "Hi"}], finish_reason="STOP")
        stream = b"data: " + json.dumps(payload).encode() + b"\r\n\r\n"
        intake = _make_intake()
        events = _feed_all(intake, stream)
        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].part, TextPart)
        assert starts[0].part.content == "Hi"


# ---------------------------------------------------------------------------
# 3) Function call response
# ---------------------------------------------------------------------------


class TestFunctionCall:
    def test_single_function_call(self) -> None:
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
        intake = _make_intake()
        events = _feed_all(intake, stream)

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        part = starts[0].part
        assert isinstance(part, ToolCallPart)
        assert part.tool_name == "get_weather"
        assert part.args == {"city": "Tokyo"}
        assert part.tool_call_id == "call_abc"

    def test_text_then_function_call_emits_both_parts(self) -> None:
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
        intake = _make_intake()
        events = _feed_all(intake, stream)

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 2
        assert isinstance(starts[0].part, TextPart)
        assert starts[0].part.content == "Looking that up..."
        assert isinstance(starts[1].part, ToolCallPart)
        assert starts[1].part.tool_name == "search"
        assert starts[1].part.args == {"q": "weather"}
        assert starts[1].part.tool_call_id == "c1"

    def test_function_call_without_id(self) -> None:
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
        intake = _make_intake()
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
    def test_inline_image_emits_file_part(self) -> None:
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
        intake = _make_intake()
        events = _feed_all(intake, stream)

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        part = starts[0].part
        assert isinstance(part, FilePart)
        assert isinstance(part.content, BinaryContent)
        assert part.content.data == png_bytes
        assert part.content.media_type == "image/png"

    def test_inline_data_skipped_when_missing_mime(self) -> None:
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
        intake = _make_intake()
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
    def test_tee_captures_every_byte(self) -> None:
        stream = _build_stream(
            [
                _chunk(parts=[{"text": "abc"}], finish_reason=None),
                _chunk(parts=[{"text": "def"}], finish_reason="STOP"),
            ]
        )
        intake = _make_intake()
        _feed_all(intake, stream)
        assert bytes(intake.upstream_raw_bytes) == stream

    def test_tee_under_byte_at_a_time_feeding(self) -> None:
        stream = _build_stream([_chunk(parts=[{"text": "hello"}], finish_reason="STOP")])
        intake = _make_intake()
        for slice_ in _chunked(stream, 1):
            list(intake.feed(slice_))
        list(intake.close())
        assert bytes(intake.upstream_raw_bytes) == stream

    def test_empty_feed_no_side_effects(self) -> None:
        intake = _make_intake()
        events = list(intake.feed(b""))
        assert events == []
        assert bytes(intake.upstream_raw_bytes) == b""


# ---------------------------------------------------------------------------
# 6) Defensive paths
# ---------------------------------------------------------------------------


class TestDefensive:
    def test_function_response_is_skipped_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
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
        intake = _make_intake()
        with caplog.at_level("WARNING"):
            events = _feed_all(intake, stream)

        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].part, TextPart)
        assert any("functionResponse" in r.message for r in caplog.records)

    def test_unparseable_json_payload_is_skipped(self) -> None:
        bad = b"data: not-json\n\n"
        good = _sse(_chunk(parts=[{"text": "ok"}], finish_reason="STOP"))
        intake = _make_intake()
        events = _feed_all(intake, bad + good)
        starts = [e for e in events if isinstance(e, PartStartEvent)]
        assert len(starts) == 1
        assert isinstance(starts[0].part, TextPart)
        assert starts[0].part.content == "ok"
