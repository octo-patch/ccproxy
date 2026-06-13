"""Tests for the IR → OpenAI Responses SSE renderer FSM.

The production FSM is async; ``_RenderFSMAdapter`` wraps it with a
one-fresh-loop-per-call sync surface for the tests.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import pytest
from pydantic_ai.messages import (
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

from ccproxy.lightllm.graph.openai_responses_render import OpenAIResponsesRenderFSM


class _RenderLike(Protocol):
    name: str

    def render(self, event: ModelResponseStreamEvent) -> bytes: ...

    def close(self) -> bytes: ...


class _RenderFSMAdapter:
    """Sync-facing adapter around the async :class:`OpenAIResponsesRenderFSM`."""

    name = "openai_responses"

    def __init__(self, *, model: str = "gpt-5") -> None:
        self._fsm = OpenAIResponsesRenderFSM(model=model)

    def render(self, event: ModelResponseStreamEvent) -> bytes:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._fsm.render(event))
        finally:
            loop.close()

    def close(self) -> bytes:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._fsm.close())
        finally:
            loop.close()


_RenderFactory = Callable[..., _RenderLike]


@pytest.fixture
def render_factory() -> _RenderFactory:
    def _make(*, model: str = "gpt-5") -> _RenderLike:
        return _RenderFSMAdapter(model=model)

    return _make


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_all(render: _RenderLike, events: Sequence[ModelResponseStreamEvent]) -> bytes:
    out = bytearray()
    for event in events:
        out += render.render(event)
    out += render.close()
    return bytes(out)


def _parse_events(data: bytes) -> list[dict[str, Any]]:
    """Decode a Responses SSE stream into a list of ``{event, data}`` dicts."""
    events: list[dict[str, Any]] = []
    for frame in data.split(b"\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        event_name: str | None = None
        data_payload: bytes | None = None
        for line in frame.split(b"\n"):
            line = line.strip()
            if line.startswith(b"event:"):
                event_name = line[6:].strip().decode()
            elif line.startswith(b"data:"):
                data_payload = line[5:].strip()
        if event_name and data_payload is not None:
            events.append(
                {
                    "event": event_name,
                    "data": json.loads(data_payload),
                }
            )
    return events


def _event_sequence(events: list[dict[str, Any]]) -> list[str]:
    return [e["event"] for e in events]


def _seq_numbers(events: list[dict[str, Any]]) -> list[int]:
    return [e["data"]["sequence_number"] for e in events]


# ---------------------------------------------------------------------------
# 1) Empty stream / minimal lifecycle
# ---------------------------------------------------------------------------


class TestEmptyStream:
    def test_close_alone_emits_only_completed(self, render_factory: _RenderFactory) -> None:
        """No events before close — emit response.completed only.

        ``response.created`` is lazy on the first ``render()`` call, so a
        stream with zero events never emits it. Codex would interpret this
        as an empty completed response.
        """
        render = render_factory()
        out = render.close()
        events = _parse_events(out)
        assert _event_sequence(events) == ["response.completed"]
        assert events[0]["data"]["response"]["status"] == "completed"

    def test_response_completed_carries_response_id(self, render_factory: _RenderFactory) -> None:
        render = render_factory()
        out = render.close()
        events = _parse_events(out)
        rid = events[0]["data"]["response"]["id"]
        assert rid.startswith("resp_")
        assert len(rid) == len("resp_") + 24  # uuid4.hex[:24]


# ---------------------------------------------------------------------------
# 2) Single text part — full lifecycle
# ---------------------------------------------------------------------------


class TestTextPart:
    def test_part_start_emits_created_item_and_content_part(self, render_factory: _RenderFactory) -> None:
        events = [PartStartEvent(index=0, part=TextPart(content="Hello"))]
        render = render_factory()
        out = _render_all(render, events)
        decoded = _parse_events(out)
        seq = _event_sequence(decoded)

        assert seq == [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        assert _seq_numbers(decoded) == list(range(8))

    def test_text_delta_accumulates_into_done_text(self, render_factory: _RenderFactory) -> None:
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(index=0, part=TextPart(content="")),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="Hello, ")),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="world!")),
            PartEndEvent(index=0, part=TextPart(content="")),
        ]
        render = render_factory()
        out = _render_all(render, events)
        decoded = _parse_events(out)

        deltas = [e for e in decoded if e["event"] == "response.output_text.delta"]
        assert [d["data"]["delta"] for d in deltas] == ["Hello, ", "world!"]

        done_text = next(e for e in decoded if e["event"] == "response.output_text.done")["data"]["text"]
        assert done_text == "Hello, world!"

    def test_message_item_done_carries_full_content(self, render_factory: _RenderFactory) -> None:
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(index=0, part=TextPart(content="Greetings.")),
            PartEndEvent(index=0, part=TextPart(content="")),
        ]
        render = render_factory()
        out = _render_all(render, events)
        decoded = _parse_events(out)
        item_done = next(e for e in decoded if e["event"] == "response.output_item.done")
        item = item_done["data"]["item"]
        assert item["type"] == "message"
        assert item["status"] == "completed"
        assert item["content"][0]["text"] == "Greetings."
        assert item["role"] == "assistant"


# ---------------------------------------------------------------------------
# 3) Function call part
# ---------------------------------------------------------------------------


class TestFunctionCallPart:
    def test_function_call_emits_args_delta_and_done(self, render_factory: _RenderFactory) -> None:
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(
                index=0,
                part=ToolCallPart(
                    tool_name="get_weather",
                    args={"city": "SF"},
                    tool_call_id="call_1",
                ),
            ),
            PartEndEvent(index=0, part=TextPart(content="")),
        ]
        render = render_factory()
        out = _render_all(render, events)
        decoded = _parse_events(out)
        seq = _event_sequence(decoded)
        assert "response.output_item.added" in seq
        assert "response.function_call_arguments.delta" in seq
        assert "response.function_call_arguments.done" in seq
        assert "response.output_item.done" in seq

        added = next(e for e in decoded if e["event"] == "response.output_item.added")
        item = added["data"]["item"]
        assert item["type"] == "function_call"
        assert item["call_id"] == "call_1"
        assert item["name"] == "get_weather"

        done = next(e for e in decoded if e["event"] == "response.function_call_arguments.done")
        assert json.loads(done["data"]["arguments"]) == {"city": "SF"}
        assert done["data"]["name"] == "get_weather"

        item_done = next(e for e in decoded if e["event"] == "response.output_item.done")
        assert item_done["data"]["item"]["call_id"] == "call_1"
        assert item_done["data"]["item"]["name"] == "get_weather"

    def test_function_call_streamed_args_via_deltas(self, render_factory: _RenderFactory) -> None:
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(
                index=0,
                part=ToolCallPart(
                    tool_name="echo",
                    args=None,
                    tool_call_id="call_2",
                ),
            ),
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"msg":')),
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='"hi"}')),
            PartEndEvent(index=0, part=TextPart(content="")),
        ]
        render = render_factory()
        out = _render_all(render, events)
        decoded = _parse_events(out)
        args_deltas = [e for e in decoded if e["event"] == "response.function_call_arguments.delta"]
        assert [d["data"]["delta"] for d in args_deltas] == ['{"msg":', '"hi"}']
        done = next(e for e in decoded if e["event"] == "response.function_call_arguments.done")
        assert done["data"]["arguments"] == '{"msg":"hi"}'


# ---------------------------------------------------------------------------
# 4) Reasoning part
# ---------------------------------------------------------------------------


class TestReasoningPart:
    def test_reasoning_emits_text_delta_and_done(self, render_factory: _RenderFactory) -> None:
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(
                index=0,
                part=ThinkingPart(content="Reasoning step.", provider_name="openai"),
            ),
            PartEndEvent(index=0, part=TextPart(content="")),
        ]
        render = render_factory()
        out = _render_all(render, events)
        decoded = _parse_events(out)
        seq = _event_sequence(decoded)
        assert "response.output_item.added" in seq
        assert "response.reasoning_text.delta" in seq
        assert "response.reasoning_text.done" in seq

        added = next(e for e in decoded if e["event"] == "response.output_item.added")
        assert added["data"]["item"]["type"] == "reasoning"

        done = next(e for e in decoded if e["event"] == "response.reasoning_text.done")
        assert done["data"]["text"] == "Reasoning step."

    def test_reasoning_text_accumulates_across_deltas(self, render_factory: _RenderFactory) -> None:
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(
                index=0,
                part=ThinkingPart(content="", provider_name="openai"),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta="Step 1: "),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta="examine input."),
            ),
            PartEndEvent(index=0, part=TextPart(content="")),
        ]
        render = render_factory()
        out = _render_all(render, events)
        decoded = _parse_events(out)
        done = next(e for e in decoded if e["event"] == "response.reasoning_text.done")
        assert done["data"]["text"] == "Step 1: examine input."


# ---------------------------------------------------------------------------
# 5) Multi-part stream — output_index allocation
# ---------------------------------------------------------------------------


class TestMultiPart:
    def test_multiple_parts_get_distinct_output_indices(self, render_factory: _RenderFactory) -> None:
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(index=0, part=TextPart(content="Hello.")),
            PartEndEvent(index=0, part=TextPart(content="")),
            PartStartEvent(
                index=1,
                part=ToolCallPart(tool_name="ping", args={}, tool_call_id="c1"),
            ),
            PartEndEvent(index=1, part=TextPart(content="")),
        ]
        render = render_factory()
        out = _render_all(render, events)
        decoded = _parse_events(out)
        item_added = [e for e in decoded if e["event"] == "response.output_item.added"]
        indices = [e["data"]["output_index"] for e in item_added]
        assert indices == [0, 1]

    def test_sequence_numbers_remain_monotonic_across_parts(self, render_factory: _RenderFactory) -> None:
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(index=0, part=TextPart(content="A")),
            PartEndEvent(index=0, part=TextPart(content="")),
            PartStartEvent(
                index=1,
                part=ToolCallPart(tool_name="t", args={}, tool_call_id="c"),
            ),
            PartEndEvent(index=1, part=TextPart(content="")),
        ]
        render = render_factory()
        out = _render_all(render, events)
        decoded = _parse_events(out)
        seqs = _seq_numbers(decoded)
        assert seqs == sorted(seqs)
        assert seqs == list(range(len(seqs)))


# ---------------------------------------------------------------------------
# 6) Lazy part open — PartDelta arriving before PartStart
# ---------------------------------------------------------------------------


class TestLazyOpen:
    def test_text_delta_without_prior_start_opens_message(self, render_factory: _RenderFactory) -> None:
        """Some upstream FSMs stream deltas without a prior start event."""
        events: list[ModelResponseStreamEvent] = [
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="hi")),
            PartEndEvent(index=0, part=TextPart(content="")),
        ]
        render = render_factory()
        out = _render_all(render, events)
        decoded = _parse_events(out)
        seq = _event_sequence(decoded)
        # lazy open of message item + content part still produces full sequence
        assert "response.output_item.added" in seq
        assert "response.content_part.added" in seq
        assert "response.output_text.delta" in seq
        assert "response.output_text.done" in seq


# ---------------------------------------------------------------------------
# 7) Unclosed items get auto-closed at close()
# ---------------------------------------------------------------------------


class TestAutoClose:
    def test_close_drains_open_items(self, render_factory: _RenderFactory) -> None:
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(index=0, part=TextPart(content="Stream cut short")),
            # Note: no PartEndEvent
        ]
        render = render_factory()
        out = _render_all(render, events)
        decoded = _parse_events(out)
        seq = _event_sequence(decoded)
        # close() emits the missing output_text.done + content_part.done + output_item.done
        assert "response.output_text.done" in seq
        assert "response.output_item.done" in seq
        assert seq[-1] == "response.completed"
