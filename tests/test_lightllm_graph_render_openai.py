"""Tests for the IR -> OpenAI Chat Completion SSE renderer FSM.

The production FSMs are async; ``_OpenAIRenderFSMAdapter`` /
``_OpenAIIntakeFSMAdapter`` wrap them with one-fresh-loop-per-call sync
surfaces (the persistent-loop bridge lives in :class:`SSEPipeline` for
production).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import pytest
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

from ccproxy.lightllm.graph.openai_intake import OpenAIResponseIntakeFSM
from ccproxy.lightllm.graph.openai_render import OpenAIResponseRenderFSM

# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class _RenderLike(Protocol):
    """Sync-callable surface around the async FSM render."""

    name: str

    def render(self, event: ModelResponseStreamEvent) -> bytes: ...

    def close(self) -> bytes: ...


class _OpenAIRenderFSMAdapter:
    """Sync-facing adapter around the async :class:`OpenAIResponseRenderFSM`.

    The production FSM is async (the persistent-loop bridge lives in
    :class:`SSEPipeline`). For tests, one fresh asyncio loop per
    ``render`` / ``close`` call is fine — tests aren't on a hot path.
    """

    name = "openai_chat"

    def __init__(self, *, model: str = "gpt-4o") -> None:
        self._fsm = OpenAIResponseRenderFSM(model=model)

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
    """Factory for the FSM render wrapped in a sync adapter."""

    def _make(*, model: str = "gpt-4o") -> _RenderLike:
        return _OpenAIRenderFSMAdapter(model=model)

    return _make


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intake(*, model: str = "gpt-4o") -> Any:
    return _OpenAIIntakeFSMAdapter(model=model)


class _OpenAIIntakeFSMAdapter:
    """Sync-facing adapter around the async :class:`OpenAIResponseIntakeFSM`."""

    def __init__(self, *, model: str = "gpt-4o") -> None:
        self._fsm = OpenAIResponseIntakeFSM(model=model, request_params=ModelRequestParameters())

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


def _render_all(render: _RenderLike, events: list[ModelResponseStreamEvent]) -> bytes:
    out = bytearray()
    for event in events:
        out += render.render(event)
    out += render.close()
    return bytes(out)


def _parse_frames(data: bytes) -> list[dict[str, Any]]:
    """Decode an OpenAI SSE stream into a list of chunk dicts, dropping ``[DONE]``."""
    frames: list[dict[str, Any]] = []
    for frame in data.split(b"\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        for line in frame.split(b"\n"):
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            frames.append(json.loads(payload))
    return frames


def _deltas(data: bytes) -> list[dict[str, Any]]:
    """Convenience: extract every ``choices[0].delta`` from a rendered stream."""
    return [chunk["choices"][0]["delta"] for chunk in _parse_frames(data)]


def _finish_reasons(data: bytes) -> list[Any]:
    """Convenience: extract every ``choices[0].finish_reason`` from a rendered stream."""
    return [chunk["choices"][0]["finish_reason"] for chunk in _parse_frames(data)]


def _ends_with_done(data: bytes) -> bool:
    return data.endswith(b"data: [DONE]\n\n")


# ---------------------------------------------------------------------------
# 1) Empty stream
# ---------------------------------------------------------------------------


class TestEmptyStream:
    def test_close_alone_emits_finish_and_done(self, render_factory: _RenderFactory) -> None:
        render = render_factory()
        out = render.close()
        assert _ends_with_done(out)
        frames = _parse_frames(out)
        assert len(frames) == 1
        choices = frames[0]["choices"]
        assert isinstance(choices, list)
        assert choices[0]["finish_reason"] == "stop"
        assert choices[0]["delta"] == {}

    def test_close_chunk_shape_matches_openai_schema(self, render_factory: _RenderFactory) -> None:
        """The final chunk must carry id/object/created/model/choices."""
        render = render_factory(model="gpt-4o")
        frames = _parse_frames(render.close())
        chunk = frames[0]
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "gpt-4o"
        assert isinstance(chunk["id"], str)
        assert chunk["id"].startswith("chatcmpl-")
        assert isinstance(chunk["created"], int)


# ---------------------------------------------------------------------------
# 2) Single text reply
# ---------------------------------------------------------------------------


class TestSingleTextReply:
    def test_role_then_content_then_finish_then_done(self, render_factory: _RenderFactory) -> None:
        render = render_factory()
        text_part = TextPart(content="Hello, world")
        events: list[ModelResponseStreamEvent] = [PartStartEvent(index=0, part=text_part)]
        out = _render_all(render, events)
        assert _ends_with_done(out)
        deltas = _deltas(out)
        # Role chunk + content chunk + final-finish chunk
        assert len(deltas) == 3
        assert deltas[0] == {"role": "assistant"}
        assert deltas[1] == {"content": "Hello, world"}
        assert deltas[2] == {}
        # Default finish_reason is stop
        assert _finish_reasons(out) == [None, None, "stop"]

    def test_empty_textpart_skips_content_chunk(self, render_factory: _RenderFactory) -> None:
        """A ``TextPart('')`` only emits the role chunk; the wire skips empty content."""
        render = render_factory()
        events: list[ModelResponseStreamEvent] = [PartStartEvent(index=0, part=TextPart(content=""))]
        out = _render_all(render, events)
        deltas = _deltas(out)
        # role, final-finish — no empty content chunk
        assert deltas == [{"role": "assistant"}, {}]


# ---------------------------------------------------------------------------
# 3) Multi-chunk text
# ---------------------------------------------------------------------------


class TestMultiChunkText:
    def test_each_delta_emits_its_own_chunk(self, render_factory: _RenderFactory) -> None:
        """Three text deltas produce three content chunks plus the role+finish."""
        render = render_factory()
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(index=0, part=TextPart(content="abc")),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="def")),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="ghi")),
        ]
        out = _render_all(render, events)
        deltas = _deltas(out)
        assert deltas == [
            {"role": "assistant"},
            {"content": "abc"},
            {"content": "def"},
            {"content": "ghi"},
            {},
        ]

    def test_delta_before_start_still_emits_role(self, render_factory: _RenderFactory) -> None:
        """A misbehaving intake that yields a delta with no prior start still gets a well-formed assistant."""
        render = render_factory()
        events: list[ModelResponseStreamEvent] = [
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="naked")),
        ]
        out = _render_all(render, events)
        deltas = _deltas(out)
        assert deltas[0] == {"role": "assistant"}
        assert {"content": "naked"} in deltas


# ---------------------------------------------------------------------------
# 4) Tool call
# ---------------------------------------------------------------------------


class TestSingleToolCall:
    def test_part_start_emits_tool_call_envelope(self, render_factory: _RenderFactory) -> None:
        render = render_factory()
        tool_part = ToolCallPart(
            tool_name="get_weather",
            args={"location": "SF"},
            tool_call_id="call_abc",
        )
        events: list[ModelResponseStreamEvent] = [PartStartEvent(index=0, part=tool_part)]
        out = _render_all(render, events)
        deltas = _deltas(out)
        # role, tool_call envelope, final-finish
        assert deltas[0] == {"role": "assistant"}
        # First tool_call chunk has id+type+function.name+function.arguments
        tc_envelope = deltas[1]
        assert isinstance(tc_envelope, dict)
        tool_calls = tc_envelope["tool_calls"]
        assert isinstance(tool_calls, list)
        assert tool_calls == [
            {
                "index": 0,
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location":"SF"}'},
            }
        ]
        # Finish reason is tool_calls
        assert _finish_reasons(out)[-1] == "tool_calls"

    def test_part_start_then_delta_appends_arguments(self, render_factory: _RenderFactory) -> None:
        """First chunk carries id+name, second chunk delivers partial arguments."""
        render = render_factory()
        tool_part = ToolCallPart(tool_name="get_weather", args="", tool_call_id="call_abc")
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(index=0, part=tool_part),
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"loca')),
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='tion":"SF"}')),
        ]
        out = _render_all(render, events)
        deltas = _deltas(out)
        # role, envelope, arg-delta-1, arg-delta-2, final-finish
        assert len(deltas) == 5
        assert deltas[2]["tool_calls"] == [{"index": 0, "function": {"arguments": '{"loca'}}]
        assert deltas[3]["tool_calls"] == [{"index": 0, "function": {"arguments": 'tion":"SF"}'}}]

    def test_args_dict_serialized_to_json_string(self, render_factory: _RenderFactory) -> None:
        """A ``ToolCallPart.args`` dict must be JSON-encoded on the wire."""
        render = render_factory()
        tool_part = ToolCallPart(
            tool_name="add",
            args={"x": 1, "y": 2},
            tool_call_id="call_d",
        )
        events: list[ModelResponseStreamEvent] = [PartStartEvent(index=0, part=tool_part)]
        out = _render_all(render, events)
        deltas = _deltas(out)
        tool_calls = deltas[1]["tool_calls"]
        assert isinstance(tool_calls, list)
        args_str = tool_calls[0]["function"]["arguments"]
        # Round-trip the JSON to ignore key ordering
        assert json.loads(args_str) == {"x": 1, "y": 2}

    def test_tool_call_delta_dict_args_serialized(self, render_factory: _RenderFactory) -> None:
        """A delta whose ``args_delta`` is a dict gets serialized to JSON."""
        render = render_factory()
        tool_part = ToolCallPart(tool_name="get", tool_call_id="call_x")
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(index=0, part=tool_part),
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={"k": "v"})),
        ]
        out = _render_all(render, events)
        deltas = _deltas(out)
        # Delta arrives in deltas[2] (after role + envelope)
        assert deltas[2]["tool_calls"] == [{"index": 0, "function": {"arguments": '{"k":"v"}'}}]


# ---------------------------------------------------------------------------
# 5) Two tool calls — unique indices
# ---------------------------------------------------------------------------


class TestMultipleToolCalls:
    def test_two_distinct_part_indices_get_unique_tool_call_indices(self, render_factory: _RenderFactory) -> None:
        render = render_factory()
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(index=0, part=ToolCallPart(tool_name="fn_a", tool_call_id="call_0")),
            PartStartEvent(index=1, part=ToolCallPart(tool_name="fn_b", tool_call_id="call_1")),
        ]
        out = _render_all(render, events)
        deltas = _deltas(out)
        # role, envelope_a, envelope_b, finish
        tc_a = deltas[1]["tool_calls"]
        tc_b = deltas[2]["tool_calls"]
        assert isinstance(tc_a, list)
        assert isinstance(tc_b, list)
        assert tc_a[0]["index"] == 0
        assert tc_b[0]["index"] == 1
        assert tc_a[0]["id"] == "call_0"
        assert tc_b[0]["id"] == "call_1"

    def test_interleaved_deltas_route_to_correct_index(self, render_factory: _RenderFactory) -> None:
        """Deltas on IR part 0 and IR part 1 must land in OpenAI tool_calls 0 and 1 respectively."""
        render = render_factory()
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(index=0, part=ToolCallPart(tool_name="fn_a", tool_call_id="call_0")),
            PartStartEvent(index=1, part=ToolCallPart(tool_name="fn_b", tool_call_id="call_1")),
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"a":')),
            PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta='{"b":')),
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='1}')),
            PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta='2}')),
        ]
        out = _render_all(render, events)
        deltas = _deltas(out)
        # role, env_a, env_b, d0, d1, d0, d1, finish
        assert deltas[3]["tool_calls"] == [{"index": 0, "function": {"arguments": '{"a":'}}]
        assert deltas[4]["tool_calls"] == [{"index": 1, "function": {"arguments": '{"b":'}}]
        assert deltas[5]["tool_calls"] == [{"index": 0, "function": {"arguments": "1}"}}]
        assert deltas[6]["tool_calls"] == [{"index": 1, "function": {"arguments": "2}"}}]

    def test_tool_call_delta_without_prior_start_allocates_slot(self, render_factory: _RenderFactory) -> None:
        """An intake emitting a delta before its start still gets a usable envelope."""
        render = render_factory()
        events: list[ModelResponseStreamEvent] = [
            PartDeltaEvent(
                index=0,
                delta=ToolCallPartDelta(
                    tool_name_delta="get_weather",
                    args_delta='{"city":"NYC"}',
                    tool_call_id="call_99",
                ),
            )
        ]
        out = _render_all(render, events)
        deltas = _deltas(out)
        # role, envelope, finish
        assert deltas[0] == {"role": "assistant"}
        env = deltas[1]["tool_calls"]
        assert env == [
            {
                "index": 0,
                "id": "call_99",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
            }
        ]


# ---------------------------------------------------------------------------
# 6) Thinking parts — OpenAI Chat has no on-wire surface
# ---------------------------------------------------------------------------


class TestThinkingDropped:
    def test_thinking_part_start_does_not_emit_content(self, render_factory: _RenderFactory) -> None:
        """``PartStartEvent(ThinkingPart)`` only triggers the role chunk; no content."""
        render = render_factory()
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(index=0, part=ThinkingPart(content="reasoning...")),
        ]
        out = _render_all(render, events)
        deltas = _deltas(out)
        # role + final-finish; no thinking content
        assert deltas == [{"role": "assistant"}, {}]

    def test_thinking_delta_emits_nothing(self, render_factory: _RenderFactory) -> None:
        """``ThinkingPartDelta`` produces no on-wire output."""
        render = render_factory()
        events: list[ModelResponseStreamEvent] = [
            PartStartEvent(index=0, part=ThinkingPart(content="initial")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="more")),
        ]
        out = _render_all(render, events)
        deltas = _deltas(out)
        # No content chunks at all
        assert deltas == [{"role": "assistant"}, {}]


# ---------------------------------------------------------------------------
# 7) Informational events are no-ops
# ---------------------------------------------------------------------------


class TestInformationalEvents:
    def test_part_end_emits_nothing(self, render_factory: _RenderFactory) -> None:
        render = render_factory()
        event = PartEndEvent(index=0, part=TextPart(content="x"))
        assert render.render(event) == b""

    def test_final_result_event_emits_nothing(self, render_factory: _RenderFactory) -> None:
        render = render_factory()
        event = FinalResultEvent(tool_name=None, tool_call_id=None)
        assert render.render(event) == b""


# ---------------------------------------------------------------------------
# 8) DONE terminator semantics
# ---------------------------------------------------------------------------


class TestDoneTerminator:
    def test_close_always_emits_done(self, render_factory: _RenderFactory) -> None:
        render = render_factory()
        out = render.close()
        assert _ends_with_done(out)

    def test_done_appears_after_final_chunk(self, render_factory: _RenderFactory) -> None:
        render = render_factory()
        out = _render_all(render, [PartStartEvent(index=0, part=TextPart(content="hi"))])
        # The [DONE] frame is the very last frame
        idx = out.rfind(b"data: ")
        assert out[idx:] == b"data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# 9) Roundtrip property test — cross-implementation matrix
# ---------------------------------------------------------------------------


class _IntakeLike(Protocol):
    """Common sync-callable surface for both intake implementations."""

    def feed(self, data: bytes) -> Iterable[ModelResponseStreamEvent]: ...

    def close(self) -> Iterable[ModelResponseStreamEvent]: ...


def _new_intake(*, model: str = "gpt-4o") -> _IntakeLike:
    return _OpenAIIntakeFSMAdapter(model=model)


def _new_render(*, model: str = "gpt-4o") -> _RenderLike:
    return _OpenAIRenderFSMAdapter(model=model)


@dataclass(frozen=True)
class RoundtripCase:
    name: str
    """Descriptive name for the test scenario."""

    events: list[ModelResponseStreamEvent]
    """IR events to seed the renderer."""


def _events_text_only() -> list[ModelResponseStreamEvent]:
    return [
        PartStartEvent(index=0, part=TextPart(content="Hello")),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=", ")),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="world")),
    ]


def _events_tool_call() -> list[ModelResponseStreamEvent]:
    return [
        PartStartEvent(
            index=0,
            part=ToolCallPart(tool_name="get_weather", args="", tool_call_id="call_xyz"),
        ),
        PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"city":')),
        PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='"NYC"}')),
    ]


def _events_two_tool_calls() -> list[ModelResponseStreamEvent]:
    return [
        PartStartEvent(
            index=0,
            part=ToolCallPart(tool_name="fn_a", args="", tool_call_id="call_a"),
        ),
        PartStartEvent(
            index=1,
            part=ToolCallPart(tool_name="fn_b", args="", tool_call_id="call_b"),
        ),
        PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"x":1}')),
        PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta='{"y":2}')),
    ]


ROUNDTRIP_CASES: list[RoundtripCase] = [
    RoundtripCase(name="text_only", events=_events_text_only()),
    RoundtripCase(name="tool_call", events=_events_tool_call()),
    RoundtripCase(name="two_tool_calls", events=_events_two_tool_calls()),
]


def _collect_text(events: list[ModelResponseStreamEvent]) -> str:
    """Reconstruct the assistant text from a stream of IR events."""
    text = ""
    for e in events:
        if isinstance(e, PartStartEvent) and isinstance(e.part, TextPart):
            text += e.part.content
        elif isinstance(e, PartDeltaEvent) and isinstance(e.delta, TextPartDelta):
            text += e.delta.content_delta
    return text


def _collect_tool_calls(events: list[ModelResponseStreamEvent]) -> list[tuple[str, str | None, str]]:
    """Reconstruct (tool_name, tool_call_id, args_json_str) tuples from IR events.

    Concatenates the start-args (if any) with all subsequent string ``args_delta``s.
    """
    per_index: dict[int, dict[str, object]] = {}
    for e in events:
        if isinstance(e, PartStartEvent) and isinstance(e.part, ToolCallPart):
            args0 = e.part.args
            if args0 is None:
                args_str = ""
            elif isinstance(args0, str):
                args_str = args0
            else:
                args_str = json.dumps(args0, separators=(",", ":"))
            per_index[e.index] = {
                "tool_name": e.part.tool_name,
                "tool_call_id": e.part.tool_call_id,
                "args": args_str,
            }
        elif isinstance(e, PartDeltaEvent) and isinstance(e.delta, ToolCallPartDelta):
            slot = per_index.setdefault(
                e.index, {"tool_name": e.delta.tool_name_delta or "", "tool_call_id": e.delta.tool_call_id, "args": ""}
            )
            d = e.delta.args_delta
            if d is None:
                pass
            elif isinstance(d, str):
                slot["args"] = str(slot["args"]) + d
            else:
                slot["args"] = str(slot["args"]) + json.dumps(d, separators=(",", ":"))
    out: list[tuple[str, str | None, str]] = []
    for _idx, slot in sorted(per_index.items()):
        tcid = slot["tool_call_id"] if isinstance(slot["tool_call_id"], str) else None
        out.append((str(slot["tool_name"]), tcid, str(slot["args"])))
    return out


class TestRoundtrip:
    """Render IR -> wire bytes -> feed back through intake -> compare semantics."""

    @pytest.mark.parametrize(
        "case",
        [pytest.param(c, id=c.name) for c in ROUNDTRIP_CASES],
    )
    def test_render_then_intake_reconstructs_same_assistant_message(
        self, case: RoundtripCase
    ) -> None:
        # 1. Render
        render = _new_render()
        wire_bytes = _render_all(render, case.events)
        assert _ends_with_done(wire_bytes)

        # 2. Feed back through a fresh intake
        intake = _new_intake()
        intake_events: list[ModelResponseStreamEvent] = []
        intake_events.extend(intake.feed(wire_bytes))
        intake_events.extend(intake.close())

        # 3. Semantic equality: text content and tool calls match
        original_text = _collect_text(case.events)
        roundtripped_text = _collect_text(intake_events)
        assert roundtripped_text == original_text

        original_tools = _collect_tool_calls(case.events)
        roundtripped_tools = _collect_tool_calls(intake_events)
        # Args may be re-encoded but JSON-equivalent
        assert len(original_tools) == len(roundtripped_tools)
        for orig, rt in zip(original_tools, roundtripped_tools, strict=True):
            assert orig[0] == rt[0]  # tool_name
            assert orig[1] == rt[1]  # tool_call_id
            # JSON-equality on args
            if orig[2] and rt[2]:
                assert json.loads(orig[2]) == json.loads(rt[2])
            else:
                assert orig[2] == rt[2]


# ---------------------------------------------------------------------------
# 10) Type-coverage smoke — make sure render() accepts every variant
# ---------------------------------------------------------------------------


class TestEventCoverage:
    @pytest.mark.parametrize(
        "event",
        [
            PartStartEvent(index=0, part=TextPart(content="x")),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="x")),
            PartEndEvent(index=0, part=TextPart(content="x")),
            FinalResultEvent(tool_name=None, tool_call_id=None),
        ],
        ids=["part_start", "part_delta", "part_end", "final_result"],
    )
    def test_every_event_variant_does_not_raise(
        self, event: ModelResponseStreamEvent, render_factory: _RenderFactory
    ) -> None:
        render = render_factory()
        # Just exercise the dispatch — return value verified in other tests
        result = render.render(event)
        assert isinstance(result, bytes)
