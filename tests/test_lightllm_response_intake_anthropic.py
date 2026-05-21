"""Tests for ``ccproxy.lightllm.response.intake_anthropic.AnthropicResponseIntake``.

Covers:
- Synthetic SSE roundtrip with a representative event mix.
- Chunk-boundary robustness (1-byte, 16-byte, single-large-chunk all
  produce the same IR event list).
- Partial frame buffering across multiple ``feed`` calls.
- Text delta accumulation across multiple ``BetaRawContentBlockDeltaEvent``s.
- Tool call sequence: ``tool_use`` start + ``input_json_delta`` + stop
  produces a ``ToolCallPart``.
- Thinking block sequence: ``thinking`` start + ``thinking_delta`` + stop
  produces a ``ThinkingPart``.
- ``upstream_raw_bytes`` is a byte-for-byte tee of all fed data.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestParameters

from ccproxy.lightllm.response.intake_anthropic import AnthropicResponseIntake

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame(event: dict[str, Any]) -> bytes:
    """Render one event dict as an Anthropic-style SSE frame."""
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _frames(events: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(_frame(e) for e in events)


def _new_intake() -> AnthropicResponseIntake:
    return AnthropicResponseIntake(
        model="claude-3-haiku-20240307",
        request_params=ModelRequestParameters(),
    )


def _drive(intake: AnthropicResponseIntake, data: bytes, chunk_size: int) -> list[ModelResponseStreamEvent]:
    """Feed ``data`` to ``intake`` in chunks of ``chunk_size`` bytes."""
    events: list[ModelResponseStreamEvent] = []
    for start in range(0, len(data), chunk_size):
        events.extend(intake.feed(data[start : start + chunk_size]))
    events.extend(intake.close())
    return events


def _summarize(events: list[ModelResponseStreamEvent]) -> list[tuple[str, int, str]]:
    """Reduce IR events to ``(event_kind, index, content_summary)`` tuples for equality checks."""
    summary: list[tuple[str, int, str]] = []
    for ev in events:
        if isinstance(ev, PartStartEvent):
            part = ev.part
            if isinstance(part, TextPart):
                content = f"TextPart:{part.content}"
            elif isinstance(part, ThinkingPart):
                content = f"ThinkingPart:{part.content}|sig={part.signature}"
            elif isinstance(part, ToolCallPart):
                content = f"ToolCallPart:{part.tool_name}|args={part.args}|id={part.tool_call_id}"
            else:
                content = f"{type(part).__name__}"
            summary.append(("part_start", ev.index, content))
        elif isinstance(ev, PartDeltaEvent):
            delta = ev.delta
            if isinstance(delta, TextPartDelta):
                content = f"TextPartDelta:{delta.content_delta}"
            else:
                content = f"{type(delta).__name__}"
            summary.append(("part_delta", ev.index, content))
    return summary


# ---------------------------------------------------------------------------
# Canonical event fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamFixture:
    name: str
    """Descriptive name for the test scenario."""

    events: list[dict[str, Any]]
    """Anthropic raw stream event dicts in emission order."""


TEXT_STREAM = StreamFixture(
    name="single_text_block",
    events=[
        {
            "type": "message_start",
            "message": {
                "id": "msg_01abc",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-3-haiku-20240307",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "world"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ],
)


TOOL_USE_STREAM = StreamFixture(
    name="tool_use_block_with_json_deltas",
    events=[
        {
            "type": "message_start",
            "message": {
                "id": "msg_tool",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-3-haiku-20240307",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_01XYZ",
                "name": "get_weather",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": ' "Paris"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    ],
)


THINKING_STREAM = StreamFixture(
    name="thinking_block_with_signature",
    events=[
        {
            "type": "message_start",
            "message": {
                "id": "msg_think",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-3-haiku-20240307",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 15, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Let me think."},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "abc123"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 3},
        },
        {"type": "message_stop"},
    ],
)


# ---------------------------------------------------------------------------
# 1. Synthetic SSE roundtrip
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_text_stream_roundtrips_to_concatenated_text(self) -> None:
        intake = _new_intake()
        sse = _frames(TEXT_STREAM.events)

        events = list(intake.feed(sse))
        events.extend(intake.close())

        parts = intake.parts_manager.get_parts()
        assert len(parts) == 1
        text_part = parts[0]
        assert isinstance(text_part, TextPart)
        assert text_part.content == "Hello world"

        # First emission for a non-empty text block is a PartStartEvent;
        # subsequent deltas are PartDeltaEvents. The block-start event also
        # has an empty text body which yields no IR event.
        assert any(isinstance(e, PartStartEvent) for e in events)
        assert any(isinstance(e, PartDeltaEvent) for e in events)

    def test_tool_use_stream_assembles_tool_call_part(self) -> None:
        intake = _new_intake()
        sse = _frames(TOOL_USE_STREAM.events)

        list(intake.feed(sse))
        list(intake.close())

        parts = intake.parts_manager.get_parts()
        assert len(parts) == 1
        tool_part = parts[0]
        assert isinstance(tool_part, ToolCallPart)
        assert tool_part.tool_name == "get_weather"
        assert tool_part.tool_call_id == "toolu_01XYZ"
        # Args accumulate as the concatenated JSON string of all input_json_delta payloads.
        assert tool_part.args == '{"city": "Paris"}'

    def test_thinking_stream_assembles_thinking_part(self) -> None:
        intake = _new_intake()
        sse = _frames(THINKING_STREAM.events)

        list(intake.feed(sse))
        list(intake.close())

        parts = intake.parts_manager.get_parts()
        assert len(parts) == 1
        thinking_part = parts[0]
        assert isinstance(thinking_part, ThinkingPart)
        assert thinking_part.content == "Let me think."
        assert thinking_part.signature == "abc123"


# ---------------------------------------------------------------------------
# 2. Chunk-boundary robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture",
    [
        pytest.param(TEXT_STREAM, id=TEXT_STREAM.name),
        pytest.param(TOOL_USE_STREAM, id=TOOL_USE_STREAM.name),
        pytest.param(THINKING_STREAM, id=THINKING_STREAM.name),
    ],
)
def test_chunk_boundaries_do_not_affect_ir_events(fixture: StreamFixture) -> None:
    """Feeding the same byte stream in different chunk sizes yields identical IR events."""
    sse = _frames(fixture.events)

    summaries: list[list[tuple[str, int, str]]] = []
    for chunk_size in (1, 16, len(sse)):
        intake = _new_intake()
        events = _drive(intake, sse, chunk_size)
        summaries.append(_summarize(events))

    one_byte, sixteen_byte, single_chunk = summaries
    assert one_byte == sixteen_byte == single_chunk


# ---------------------------------------------------------------------------
# 3. Partial frame handling
# ---------------------------------------------------------------------------


class TestPartialFrameHandling:
    def test_half_frame_buffered_until_completion(self) -> None:
        intake = _new_intake()
        # message_start has no SSE-level IR emission, but content_block_delta does — use that.
        block_start = _frame(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        )
        delta = _frame(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "partial"},
            }
        )
        full = block_start + delta
        midpoint = len(block_start) + (len(delta) // 2)

        first_half = full[:midpoint]
        second_half = full[midpoint:]

        first_events = list(intake.feed(first_half))
        # block_start has empty text body, so nothing IR-visible on its own.
        # The delta is split — its frame is not yet closed by ``\n\n``.
        # ``block_start`` alone produces no IR event, so the first call yields nothing.
        assert first_events == []

        second_events = list(intake.feed(second_half))
        assert any(isinstance(e, PartStartEvent) for e in second_events)


# ---------------------------------------------------------------------------
# 4. upstream_raw_bytes tee
# ---------------------------------------------------------------------------


def test_upstream_raw_bytes_is_byte_for_byte_tee() -> None:
    intake = _new_intake()
    sse = _frames(TEXT_STREAM.events)

    # Feed in irregular chunks
    cursor = 0
    for chunk_size in (5, 17, 41, len(sse)):
        end = min(cursor + chunk_size, len(sse))
        list(intake.feed(sse[cursor:end]))
        cursor = end
        if cursor >= len(sse):
            break

    assert bytes(intake.upstream_raw_bytes) == sse


# ---------------------------------------------------------------------------
# 5. Both SSE separator styles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("separator", "label"),
    [
        pytest.param(b"\n\n", "lf_lf", id="lf_lf_separator"),
        pytest.param(b"\r\n\r\n", "crlf_crlf", id="crlf_crlf_separator"),
    ],
)
def test_both_sse_separators_are_recognized(separator: bytes, label: str) -> None:
    intake = _new_intake()
    payload = json.dumps(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "ready"},
        }
    ).encode()
    sse = b"event: content_block_start\ndata: " + payload + separator

    events = list(intake.feed(sse))
    events.extend(intake.close())
    assert any(isinstance(e, PartStartEvent) for e in events), label


# ---------------------------------------------------------------------------
# 6. Empty feed and close
# ---------------------------------------------------------------------------


def test_empty_feed_yields_nothing() -> None:
    intake = _new_intake()
    assert list(intake.feed(b"")) == []
    assert list(intake.close()) == []
    assert bytes(intake.upstream_raw_bytes) == b""


def test_unparseable_frame_is_skipped_without_crashing(caplog: pytest.LogCaptureFixture) -> None:
    intake = _new_intake()
    bad = b"event: broken\ndata: {not valid json}\n\n"

    events = list(intake.feed(bad))
    assert events == []
    # The intake debug-logs the failure rather than crashing.
