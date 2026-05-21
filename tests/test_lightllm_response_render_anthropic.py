"""Tests for ``ccproxy.lightllm.response.render_anthropic.AnthropicResponseRender``.

Covers:
- Empty stream — just ``close()`` — emits ``message_start`` + ``message_delta``
  + ``message_stop``.
- Single text part — start/delta/end + close — verifies the full event
  sequence on the wire.
- Multi-block (text then tool_use) — verifies proper open/close transitions
  when a new ``PartStartEvent`` arrives without an explicit ``PartEndEvent``.
- Thinking block — start/content delta/signature delta/end + close — verifies
  the three Anthropic delta event names emitted for a thinking block.
- Redacted thinking — verifies the ``redacted_thinking`` block descriptor.
- Tool call with JSON args — verifies ``tool_use`` block start and
  ``input_json_delta`` deltas.
- Roundtrip property — render IR events from
  ``AnthropicResponseIntake.feed`` of a captured SSE byte stream, feed the
  rendered bytes back into a fresh intake, assert the resulting
  ``ModelResponse`` is structurally equal.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic_ai.messages import (
    FinalResultEvent,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.models import ModelRequestParameters

from ccproxy.lightllm.response.intake_anthropic import AnthropicResponseIntake
from ccproxy.lightllm.response.render_anthropic import AnthropicResponseRender

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse(data: bytes) -> list[tuple[str, dict[str, Any]]]:
    """Parse raw SSE bytes into ``(event_name, payload_dict)`` tuples."""
    frames: list[tuple[str, dict[str, Any]]] = []
    for frame in data.split(b"\n\n"):
        if not frame.strip():
            continue
        event_name = ""
        data_payload = ""
        for line in frame.split(b"\n"):
            text = line.decode()
            if text.startswith("event:"):
                event_name = text[len("event:") :].strip()
            elif text.startswith("data:"):
                data_payload = text[len("data:") :].strip()
        assert event_name, f"frame missing event: line: {frame!r}"
        assert data_payload, f"frame missing data: line: {frame!r}"
        frames.append((event_name, json.loads(data_payload)))
    return frames


def _render_all(events: Iterable[ModelResponseStreamEvent]) -> bytes:
    render = AnthropicResponseRender(model="claude-3-haiku-20240307")
    out = bytearray()
    for ev in events:
        out += render.render(ev)
    out += render.close()
    return bytes(out)


def _frame_anthropic_sse(events: list[dict[str, Any]]) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


# ---------------------------------------------------------------------------
# 1. Empty stream
# ---------------------------------------------------------------------------


def test_empty_stream_emits_message_start_delta_stop() -> None:
    render = AnthropicResponseRender(model="claude-3-haiku-20240307")
    out = render.close()
    frames = _parse_sse(out)
    names = [name for name, _ in frames]
    assert names == ["message_start", "message_delta", "message_stop"]

    _, message_start_payload = frames[0]
    assert message_start_payload["type"] == "message_start"
    assert message_start_payload["message"]["model"] == "claude-3-haiku-20240307"
    assert message_start_payload["message"]["role"] == "assistant"
    assert message_start_payload["message"]["content"] == []

    _, message_delta_payload = frames[1]
    assert message_delta_payload == {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 0},
    }

    _, message_stop_payload = frames[2]
    assert message_stop_payload == {"type": "message_stop"}


# ---------------------------------------------------------------------------
# 2. Single text part
# ---------------------------------------------------------------------------


def test_single_text_part_emits_full_block_lifecycle() -> None:
    events: list[ModelResponseStreamEvent] = [
        PartStartEvent(index=0, part=TextPart(content="")),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="hello")),
        PartEndEvent(index=0, part=TextPart(content="hello")),
    ]
    out = _render_all(events)
    frames = _parse_sse(out)
    names = [name for name, _ in frames]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    _, start_payload = frames[1]
    assert start_payload == {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }

    _, delta_payload = frames[2]
    assert delta_payload == {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "hello"},
    }

    _, stop_payload = frames[3]
    assert stop_payload == {"type": "content_block_stop", "index": 0}


# ---------------------------------------------------------------------------
# 3. Multi-block (text then tool_use)
# ---------------------------------------------------------------------------


def test_multi_block_closes_previous_when_new_part_starts_without_end() -> None:
    """A ``PartStartEvent`` arriving while a block is open closes the previous block first."""
    events: list[ModelResponseStreamEvent] = [
        PartStartEvent(index=0, part=TextPart(content="")),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="Looking up weather")),
        PartStartEvent(
            index=1,
            part=ToolCallPart(tool_name="get_weather", args="", tool_call_id="toolu_01XYZ"),
        ),
        PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta='{"city":"Paris"}')),
        PartEndEvent(
            index=1,
            part=ToolCallPart(tool_name="get_weather", args='{"city":"Paris"}', tool_call_id="toolu_01XYZ"),
        ),
    ]
    out = _render_all(events)
    frames = _parse_sse(out)
    names = [name for name, _ in frames]
    assert names == [
        "message_start",
        "content_block_start",  # text block start (index 0)
        "content_block_delta",  # text delta
        "content_block_stop",  # text block closed because tool_use starts
        "content_block_start",  # tool_use block start (index 1)
        "content_block_delta",  # input_json_delta
        "content_block_stop",  # tool_use block stop from PartEndEvent
        "message_delta",
        "message_stop",
    ]

    _, tool_start_payload = frames[4]
    assert tool_start_payload == {
        "type": "content_block_start",
        "index": 1,
        "content_block": {
            "type": "tool_use",
            "id": "toolu_01XYZ",
            "name": "get_weather",
            "input": {},
        },
    }

    _, tool_delta_payload = frames[5]
    assert tool_delta_payload == {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": '{"city":"Paris"}'},
    }


# ---------------------------------------------------------------------------
# 4. Thinking block
# ---------------------------------------------------------------------------


def test_thinking_block_emits_thinking_then_signature_deltas() -> None:
    events: list[ModelResponseStreamEvent] = [
        PartStartEvent(index=0, part=ThinkingPart(content="")),
        PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="reasoning")),
        PartDeltaEvent(index=0, delta=ThinkingPartDelta(signature_delta="abc123")),
        PartEndEvent(index=0, part=ThinkingPart(content="reasoning", signature="abc123")),
    ]
    out = _render_all(events)
    frames = _parse_sse(out)
    names = [name for name, _ in frames]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",  # thinking_delta
        "content_block_delta",  # signature_delta
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    _, start_payload = frames[1]
    assert start_payload == {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    }

    _, thinking_delta = frames[2]
    assert thinking_delta == {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "reasoning"},
    }

    _, signature_delta = frames[3]
    assert signature_delta == {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "signature_delta", "signature": "abc123"},
    }


# ---------------------------------------------------------------------------
# 5. Redacted thinking
# ---------------------------------------------------------------------------


def test_redacted_thinking_block_uses_redacted_thinking_type() -> None:
    events: list[ModelResponseStreamEvent] = [
        PartStartEvent(
            index=0,
            part=ThinkingPart(content="", id="redacted_thinking", signature="opaque_blob"),
        ),
        PartEndEvent(
            index=0,
            part=ThinkingPart(content="", id="redacted_thinking", signature="opaque_blob"),
        ),
    ]
    out = _render_all(events)
    frames = _parse_sse(out)
    names = [name for name, _ in frames]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    _, start_payload = frames[1]
    assert start_payload == {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "redacted_thinking", "data": "opaque_blob"},
    }


# ---------------------------------------------------------------------------
# 6. Tool call with JSON args (dict input gets JSON-encoded to partial_json)
# ---------------------------------------------------------------------------


def test_tool_call_with_dict_args_delta_json_encodes_partial_json() -> None:
    events: list[ModelResponseStreamEvent] = [
        PartStartEvent(
            index=0,
            part=ToolCallPart(tool_name="get_weather", args=None, tool_call_id="toolu_002"),
        ),
        PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={"city": "Paris"})),
        PartEndEvent(
            index=0,
            part=ToolCallPart(tool_name="get_weather", args={"city": "Paris"}, tool_call_id="toolu_002"),
        ),
    ]
    out = _render_all(events)
    frames = _parse_sse(out)
    names = [name for name, _ in frames]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    _, delta_payload = frames[2]
    # dict args_delta gets JSON-string-encoded for the wire.
    assert delta_payload == {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": '{"city":"Paris"}'},
    }


# ---------------------------------------------------------------------------
# 7. Roundtrip property test against AnthropicResponseIntake
# ---------------------------------------------------------------------------


def _new_intake() -> AnthropicResponseIntake:
    return AnthropicResponseIntake(
        model="claude-3-haiku-20240307",
        request_params=ModelRequestParameters(),
    )


CAPTURED_TEXT_STREAM: list[dict[str, Any]] = [
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
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " "}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "world"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 5},
    },
    {"type": "message_stop"},
]


CAPTURED_TOOL_STREAM: list[dict[str, Any]] = [
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
]


def _ir_events_from_sse(sse: bytes) -> list[ModelResponseStreamEvent]:
    intake = _new_intake()
    events = list(intake.feed(sse))
    events.extend(intake.close())
    return events


def _summary_from_intake(sse: bytes) -> list[tuple[str, str]]:
    """Reduce an intake-parsed stream into the concrete ``(part_type, content)``
    summary used for equality checks (independent of the IR event-stream shape).
    """
    intake = _new_intake()
    list(intake.feed(sse))
    list(intake.close())
    summary: list[tuple[str, str]] = []
    for part in intake.parts_manager.get_parts():
        if isinstance(part, TextPart):
            summary.append(("text", part.content))
        elif isinstance(part, ThinkingPart):
            summary.append(("thinking", f"{part.content}|sig={part.signature}|id={part.id}"))
        elif isinstance(part, ToolCallPart):
            summary.append(("tool_call", f"{part.tool_name}|args={part.args}|id={part.tool_call_id}"))
        else:
            summary.append((type(part).__name__, str(part)))
    return summary


def test_roundtrip_text_stream_preserves_semantics() -> None:
    sse = _frame_anthropic_sse(CAPTURED_TEXT_STREAM)
    original_summary = _summary_from_intake(sse)

    # Parse → render → parse again and confirm equivalence.
    ir_events = _ir_events_from_sse(sse)
    rendered = _render_all(ir_events)
    roundtrip_summary = _summary_from_intake(rendered)

    assert original_summary == roundtrip_summary
    assert original_summary == [("text", "Hello world")]


def test_roundtrip_tool_stream_preserves_semantics() -> None:
    sse = _frame_anthropic_sse(CAPTURED_TOOL_STREAM)
    original_summary = _summary_from_intake(sse)

    ir_events = _ir_events_from_sse(sse)
    rendered = _render_all(ir_events)
    roundtrip_summary = _summary_from_intake(rendered)

    assert original_summary == roundtrip_summary
    assert original_summary == [("tool_call", 'get_weather|args={"city": "Paris"}|id=toolu_01XYZ')]


# ---------------------------------------------------------------------------
# 8. Internal agent-loop events are dropped
# ---------------------------------------------------------------------------


def test_final_result_event_emits_no_bytes() -> None:
    render = AnthropicResponseRender(model="claude-3-haiku-20240307")
    out = render.render(FinalResultEvent(tool_name=None, tool_call_id=None))
    assert out == b""
