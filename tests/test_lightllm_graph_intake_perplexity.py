"""Tests for the Perplexity Pro response intake FSM (SSE → pydantic-ai IR).

The production FSM is async; ``_PerplexityFSMAdapter`` wraps it with a
one-fresh-loop-per-call sync surface for tests (the persistent-loop bridge
lives in :class:`SSEPipeline` for production).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from typing import Any, Protocol

import pytest
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import (
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.models import ModelRequestParameters

from ccproxy.lightllm.graph.perplexity_intake import (
    _ANSWER_VENDOR_ID,
    _REASONING_VENDOR_ID,
    PerplexityResponseIntakeFSM,
)

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class _StateView(Protocol):
    """Subset of stream-level state the intake exposes for assertions."""

    ids: dict[str, str]
    final: bool
    seen_step_uuids: set[str]
    logged_unknown_intended_usages: set[str]


class _IntakeLike(Protocol):
    """Sync-callable surface around the async FSM intake."""

    @property
    def upstream_raw_bytes(self) -> bytearray: ...

    @property
    def parts_manager(self) -> ModelResponsePartsManager: ...

    @property
    def _state(self) -> _StateView: ...

    def feed(self, data: bytes) -> Iterable[ModelResponseStreamEvent]: ...

    def close(self) -> Iterable[ModelResponseStreamEvent]: ...


class _PerplexityFSMAdapter:
    """Sync-facing adapter around the async :class:`PerplexityResponseIntakeFSM`.

    The production FSM is async (the persistent-loop bridge lives in
    :class:`SSEPipeline`). For tests, one fresh asyncio loop per
    ``feed`` / ``close`` call is fine — tests aren't on a hot path.
    """

    def __init__(self, *, model: str, request_params: ModelRequestParameters) -> None:
        self._fsm = PerplexityResponseIntakeFSM(model=model, request_params=request_params)

    @property
    def parts_manager(self) -> ModelResponsePartsManager:
        return self._fsm.parts_manager

    @property
    def upstream_raw_bytes(self) -> bytearray:
        return self._fsm.upstream_raw_bytes

    @property
    def _state(self) -> _StateView:
        return self._fsm.state  # type: ignore[return-value]

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

    def _make(*, model: str = "perplexity/best") -> _IntakeLike:
        return _PerplexityFSMAdapter(model=model, request_params=ModelRequestParameters())

    return _make


@pytest.fixture
def intake_logger_name() -> str:
    """Name of the logger emitting unknown-intended_usage DEBUG records."""
    return "ccproxy.lightllm.graph.perplexity_intake"


# ----------------------- helpers -----------------------


def _sse_payload(payload: dict[str, Any]) -> bytes:
    """Encode one ``data: <json>\\n\\n`` SSE frame."""
    return f"data: {json.dumps(payload)}\n\n".encode()


def _collect_feed(intake: _IntakeLike, data: bytes) -> list[ModelResponseStreamEvent]:
    return list(intake.feed(data))


def _final_text(events: list[ModelResponseStreamEvent]) -> str:
    """Reconstruct the accumulated TextPart content from a stream of IR events."""
    text = ""
    for event in events:
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            text = event.part.content
        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            text += event.delta.content_delta or ""
    return text


def _final_thinking(events: list[ModelResponseStreamEvent]) -> str:
    text = ""
    for event in events:
        if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
            text = event.part.content
        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
            text += event.delta.content_delta or ""
    return text


# ----------------------- synthetic roundtrip -----------------------


def test_synthetic_full_answer_roundtrip_via_mode_a(intake_factory: _IntakeFactory) -> None:
    """One Mode-A event with a cumulative ``answer`` string yields one TextPart."""
    intake = intake_factory()
    event = {
        "blocks": [
            {
                "intended_usage": "ask_text_0_markdown",
                "diff_block": {
                    "field": "markdown_block",
                    "patches": [{"path": "/markdown_block", "value": {"answer": "Hello world."}}],
                },
            }
        ]
    }
    events = _collect_feed(intake, _sse_payload(event))
    assert _final_text(events) == "Hello world."


def test_synthetic_mode_b_then_mode_c_chunked_answer(intake_factory: _IntakeFactory) -> None:
    """Mode B sets chunks[0]; Mode C appends /chunks/1, /chunks/2."""
    intake = intake_factory()
    e1 = {
        "blocks": [
            {
                "intended_usage": "ask_text_0_markdown",
                "diff_block": {
                    "field": "markdown_block",
                    "patches": [
                        {
                            "path": "",
                            "value": {
                                "chunks": ["2 + 2 eq"],
                                "chunk_starting_offset": 0,
                                "answer": None,
                            },
                        }
                    ],
                },
            }
        ]
    }
    e2 = {
        "blocks": [
            {
                "intended_usage": "ask_text_0_markdown",
                "diff_block": {
                    "field": "markdown_block",
                    "patches": [{"path": "/chunks/1", "value": "ual"}],
                },
            }
        ]
    }
    e3 = {
        "blocks": [
            {
                "intended_usage": "ask_text_0_markdown",
                "diff_block": {
                    "field": "markdown_block",
                    "patches": [{"path": "/chunks/2", "value": "s 4."}],
                },
            }
        ]
    }
    e_final = {"final_sse_message": True, "thread_url_slug": "slug-1"}

    events: list[ModelResponseStreamEvent] = []
    events.extend(intake.feed(_sse_payload(e1)))
    events.extend(intake.feed(_sse_payload(e2)))
    events.extend(intake.feed(_sse_payload(e3)))
    events.extend(intake.feed(_sse_payload(e_final)))

    assert _final_text(events) == "2 + 2 equals 4."


def test_ask_text_block_is_skipped_no_double_emission(intake_factory: _IntakeFactory) -> None:
    """Both ``ask_text_0_markdown`` and ``ask_text`` ship identical patches; we only emit markdown."""
    intake = intake_factory()
    payload = {
        "blocks": [
            {
                "intended_usage": "ask_text_0_markdown",
                "diff_block": {
                    "field": "markdown_block",
                    "patches": [{"path": "/markdown_block", "value": {"answer": "hi"}}],
                },
            },
            {
                "intended_usage": "ask_text",
                "diff_block": {
                    "field": "markdown_block",
                    "patches": [{"path": "/markdown_block", "value": {"answer": "hi"}}],
                },
            },
        ]
    }
    events = _collect_feed(intake, _sse_payload(payload))
    assert _final_text(events) == "hi"  # NOT "hihi"


def test_reasoning_goals_prefix_diff(intake_factory: _IntakeFactory) -> None:
    """plan_block.goals[].description is cumulative; emit only the tail."""
    intake = intake_factory()
    e1 = {
        "blocks": [
            {
                "intended_usage": "pro_search_steps",
                "plan_block": {"goals": [{"description": "Looking up"}]},
            }
        ]
    }
    e2 = {
        "blocks": [
            {
                "intended_usage": "pro_search_steps",
                "plan_block": {"goals": [{"description": "Looking up X"}]},
            }
        ]
    }
    events = list(intake.feed(_sse_payload(e1)))
    events.extend(intake.feed(_sse_payload(e2)))

    assert _final_thinking(events) == "Looking up X"


def test_identifier_capture_preserved_in_state(intake_factory: _IntakeFactory) -> None:
    """Top-level event fields populate ``self._state.ids``."""
    intake = intake_factory()
    e = {
        "backend_uuid": "B-1",
        "context_uuid": "C-1",
        "read_write_token": "RW-1",
        "thread_url_slug": "slug-1",
        "thread_title": "Quantum?",
        "display_model": "claude46sonnet",
        "blocks": [],
    }
    _collect_feed(intake, _sse_payload(e))
    assert intake._state.ids == {
        "backend_uuid": "B-1",
        "context_uuid": "C-1",
        "read_write_token": "RW-1",
        "thread_url_slug": "slug-1",
        "thread_title": "Quantum?",
        "display_model": "claude46sonnet",
    }


def test_final_sse_message_sets_final_flag(intake_factory: _IntakeFactory) -> None:
    intake = intake_factory()
    _collect_feed(intake, _sse_payload({"blocks": [], "final_sse_message": True}))
    assert intake._state.final is True


def test_close_yields_no_events(intake_factory: _IntakeFactory) -> None:
    intake = intake_factory()
    _collect_feed(intake, _sse_payload({"blocks": []}))
    assert list(intake.close()) == []


# ----------------------- chunk-boundary robustness -----------------------


def test_chunk_boundary_byte_by_byte_feed(intake_factory: _IntakeFactory) -> None:
    """Fed one byte at a time, the intake produces the same final text."""
    intake = intake_factory()
    payload = {
        "blocks": [
            {
                "intended_usage": "ask_text_0_markdown",
                "diff_block": {
                    "field": "markdown_block",
                    "patches": [{"path": "/markdown_block", "value": {"answer": "Hello"}}],
                },
            }
        ]
    }
    blob = _sse_payload(payload)
    events: list[ModelResponseStreamEvent] = []
    for i in range(len(blob)):
        events.extend(intake.feed(blob[i : i + 1]))

    assert _final_text(events) == "Hello"


def test_chunk_boundary_split_inside_separator(intake_factory: _IntakeFactory) -> None:
    """Separator ``\\n\\n`` arriving across two calls is still framed correctly."""
    intake = intake_factory()
    payload = _sse_payload(
        {
            "blocks": [
                {
                    "intended_usage": "ask_text_0_markdown",
                    "diff_block": {
                        "field": "markdown_block",
                        "patches": [{"path": "/markdown_block", "value": {"answer": "AB"}}],
                    },
                }
            ]
        }
    )
    cut = payload.find(b"\n\n") + 1  # split right between the two \n
    events = list(intake.feed(payload[:cut]))
    events.extend(intake.feed(payload[cut:]))
    assert _final_text(events) == "AB"


def test_crlf_separator_recognized(intake_factory: _IntakeFactory) -> None:
    """``\\r\\n\\r\\n`` is a valid SSE separator."""
    intake = intake_factory()
    payload_body = json.dumps(
        {
            "blocks": [
                {
                    "intended_usage": "ask_text_0_markdown",
                    "diff_block": {
                        "field": "markdown_block",
                        "patches": [{"path": "/markdown_block", "value": {"answer": "X"}}],
                    },
                }
            ]
        }
    )
    blob = f"data: {payload_body}\r\n\r\n".encode()
    events = _collect_feed(intake, blob)
    assert _final_text(events) == "X"


def test_multiple_events_one_feed_call(intake_factory: _IntakeFactory) -> None:
    """Two SSE events arriving in a single bytes blob both get processed."""
    intake = intake_factory()
    e1 = _sse_payload(
        {
            "blocks": [
                {
                    "intended_usage": "ask_text_0_markdown",
                    "diff_block": {
                        "field": "markdown_block",
                        "patches": [
                            {
                                "path": "",
                                "value": {
                                    "chunks": ["foo"],
                                    "chunk_starting_offset": 0,
                                },
                            }
                        ],
                    },
                }
            ]
        }
    )
    e2 = _sse_payload(
        {
            "blocks": [
                {
                    "intended_usage": "ask_text_0_markdown",
                    "diff_block": {
                        "field": "markdown_block",
                        "patches": [{"path": "/chunks/1", "value": "bar"}],
                    },
                }
            ]
        }
    )
    events = _collect_feed(intake, e1 + e2)
    assert _final_text(events) == "foobar"


# ----------------------- step events (don't crash) -----------------------


def test_step_event_with_mcp_tool_input_renders_into_thinking(intake_factory: _IntakeFactory) -> None:
    """plan_block.steps[] with an MCP tool call routes rendered text into ThinkingPart."""
    intake = intake_factory()
    event = {
        "blocks": [
            {
                "intended_usage": "pro_search_steps",
                "plan_block": {
                    "goals": [],
                    "steps": [
                        {
                            "uuid": "step-1",
                            "step_type": "MCP_TOOL_INPUT",
                            "mcp_tool_input_content": {
                                "goal_id": "0",
                                "tool_name": "get_me",
                                "tool_args": {},
                                "app": "GitHub",
                                "tool_input_summary": "Getting user info",
                                "request_user_approval": {"request_user_approval": False},
                                "mcp_server_type": "MCP_SERVER_TYPE_REMOTE",
                                "source_type": "github_mcp_direct",
                            },
                        }
                    ],
                },
            }
        ]
    }
    events = _collect_feed(intake, _sse_payload(event))
    thinking = _final_thinking(events)
    assert "[GitHub]" in thinking
    assert "get_me" in thinking


def test_step_dedup_via_uuid_across_cumulative_events(intake_factory: _IntakeFactory) -> None:
    """Two events carrying the same step uuid emit reasoning text only once."""
    intake = intake_factory()
    step_event = {
        "blocks": [
            {
                "intended_usage": "pro_search_steps",
                "plan_block": {
                    "goals": [],
                    "steps": [
                        {
                            "uuid": "dedup-1",
                            "step_type": "MCP_TOOL_INPUT",
                            "mcp_tool_input_content": {
                                "tool_name": "x",
                                "tool_args": {},
                                "app": "GitHub",
                            },
                        }
                    ],
                },
            }
        ]
    }
    events = list(intake.feed(_sse_payload(step_event)))
    first_pass = _final_thinking(events)

    more_events = list(intake.feed(_sse_payload(step_event)))
    second_pass = _final_thinking(events + more_events)

    assert first_pass == second_pass  # repeated step doesn't accumulate further
    assert "dedup-1" in intake._state.seen_step_uuids


def test_clarifying_questions_step_does_not_crash_intake(intake_factory: _IntakeFactory) -> None:
    """RESEARCH_CLARIFYING_QUESTIONS is silently suppressed in the intake."""
    intake = intake_factory()
    event = {
        "text": json.dumps(
            [
                {
                    "step_type": "RESEARCH_CLARIFYING_QUESTIONS",
                    "content": {"questions": ["What aspect?"]},
                }
            ]
        ),
        "blocks": [],
    }
    events = _collect_feed(intake, _sse_payload(event))
    # No exception, no events emitted (clarifying questions are not emitted as
    # reasoning text on the intake path).
    assert events == []


def test_plan_event_doesnt_crash_with_bare_metadata(intake_factory: _IntakeFactory) -> None:
    """A 'plan' event with only goals (no steps) yields reasoning + no crash."""
    intake = intake_factory()
    event = {
        "blocks": [
            {
                "intended_usage": "plan",
                "plan_block": {
                    "progress": "DONE",
                    "goals": [
                        {"id": "0", "description": "Opening GitHub"},
                    ],
                    "steps": [],
                    "final": True,
                },
            }
        ]
    }
    events = _collect_feed(intake, _sse_payload(event))
    assert "Opening GitHub" in _final_thinking(events)


def test_unknown_intended_usage_logs_at_debug(
    intake_factory: _IntakeFactory,
    intake_logger_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown intended_usage values get DEBUG-logged once per stream."""
    import logging

    intake = intake_factory()
    event = {"blocks": [{"intended_usage": "totally_new_block_type", "totally_new_block": {}}]}
    with caplog.at_level(logging.DEBUG, logger=intake_logger_name):
        _collect_feed(intake, _sse_payload(event))
    assert "totally_new_block_type" in intake._state.logged_unknown_intended_usages
    assert any("totally_new_block_type" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=intake_logger_name):
        _collect_feed(intake, _sse_payload(event))
    assert not any("totally_new_block_type" in r.message for r in caplog.records)


# ----------------------- upstream_raw_bytes tee -----------------------


def test_upstream_raw_bytes_byte_for_byte_tee(intake_factory: _IntakeFactory) -> None:
    """``upstream_raw_bytes`` accumulates every byte passed to ``feed``."""
    intake = intake_factory()
    blob1 = b'data: {"final_sse_message": false, "blocks": []}\n\n'
    blob2 = b'data: {"final_sse_message": true, "blocks": []}\n\n'
    list(intake.feed(blob1))
    list(intake.feed(blob2))
    assert bytes(intake.upstream_raw_bytes) == blob1 + blob2


def test_upstream_raw_bytes_includes_unparseable_input(intake_factory: _IntakeFactory) -> None:
    """Even non-JSON / partial frames are kept in the tee."""
    intake = intake_factory()
    blob = b"data: not-json\n\ndata: also-bad\n\n"
    list(intake.feed(blob))
    assert bytes(intake.upstream_raw_bytes) == blob


def test_upstream_raw_bytes_empty_after_construction(intake_factory: _IntakeFactory) -> None:
    intake = intake_factory()
    assert intake.upstream_raw_bytes == bytearray()


def test_empty_feed_is_noop(intake_factory: _IntakeFactory) -> None:
    intake = intake_factory()
    assert list(intake.feed(b"")) == []
    assert intake.upstream_raw_bytes == bytearray()


def test_done_sentinel_doesnt_crash(intake_factory: _IntakeFactory) -> None:
    """``data: [DONE]`` (OpenAI sentinel; not standard for pplx) is gracefully ignored."""
    intake = intake_factory()
    blob = b"data: [DONE]\n\n"
    events = _collect_feed(intake, blob)
    assert events == []


def test_keepalive_comments_are_skipped(intake_factory: _IntakeFactory) -> None:
    """Lines not starting with ``data:`` (e.g. SSE comments) are dropped."""
    intake = intake_factory()
    blob = b": keepalive\n\n"
    events = _collect_feed(intake, blob)
    assert events == []


# ----------------------- finishing semantics -----------------------


def test_vendor_part_ids_use_stable_constants() -> None:
    """Sanity check on the published constants used by render-side coupling."""
    assert _ANSWER_VENDOR_ID == "pplx-answer"
    assert _REASONING_VENDOR_ID == "pplx-reasoning"


def test_separate_text_and_thinking_parts_emitted(intake_factory: _IntakeFactory) -> None:
    """An event carrying both an answer delta and a goal description produces
    two distinct parts."""
    intake = intake_factory()
    event = {
        "blocks": [
            {
                "intended_usage": "ask_text_0_markdown",
                "diff_block": {
                    "field": "markdown_block",
                    "patches": [{"path": "/markdown_block", "value": {"answer": "OK"}}],
                },
            },
            {
                "intended_usage": "pro_search_steps",
                "plan_block": {"goals": [{"description": "searching"}]},
            },
        ]
    }
    events = _collect_feed(intake, _sse_payload(event))
    assert _final_text(events) == "OK"
    assert _final_thinking(events) == "searching"
