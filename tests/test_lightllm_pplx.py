"""Tests for the Perplexity Pro lightllm adapter and supporting helpers."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from ccproxy.config import PplxConfig, PplxThreadConfig
from ccproxy.lightllm.pplx import (
    PERPLEXITY_BLOCK_USE_CASES,
    PERPLEXITY_MODELS,
    PerplexityClarifyingQuestionsError,
    StreamState,
    _build_pplx_payload,
    _extract_deltas,
    _flatten_last_user_turn,
    _flatten_messages,
    _parse_sse_line,
    _thread_to_openai_messages,
)
from ccproxy.lightllm.pplx_threads import (
    PerplexityThreadStore,
    clear_pplx_threads,
    get_pplx_thread_store,
)


def test_models_catalog_has_known_ids() -> None:
    assert "perplexity/best" in PERPLEXITY_MODELS
    assert "perplexity/deep-research" in PERPLEXITY_MODELS
    assert "openai/gpt-5.4" in PERPLEXITY_MODELS
    assert PERPLEXITY_MODELS["perplexity/best"]["identifier"] == "default"


def test_build_payload_first_turn_full_production_shape() -> None:
    payload = _build_pplx_payload(query="what is quantum?", model_id="perplexity/best", extras={})
    params = payload["params"]
    assert payload["query_str"] == "what is quantum?"
    assert params["query_source"] == "home"
    assert params["time_from_first_type"] == 18361
    assert params["use_schematized_api"] is True
    assert params["send_back_text_in_streaming_api"] is False
    assert params["prompt_source"] == "user"
    assert params["dsl_query"] == "what is quantum?"
    assert params["version"] == "2.18"
    assert params["model_preference"] == "default"
    assert isinstance(params["frontend_uuid"], str) and params["frontend_uuid"]
    assert isinstance(params["frontend_context_uuid"], str) and params["frontend_context_uuid"]
    assert params["supported_block_use_cases"] == PERPLEXITY_BLOCK_USE_CASES
    assert params["supported_features"] == ["browser_agent_permission_banner_v1.1"]


def test_build_payload_followup_injects_identifiers() -> None:
    payload = _build_pplx_payload(
        query="and superposition?",
        model_id="perplexity/best",
        extras={
            "last_backend_uuid": "backend-1",
            "read_write_token": "rw-1",
            "frontend_context_uuid": "ctx-stable",
        },
    )
    params = payload["params"]
    assert params["query_source"] == "followup"
    assert params["followup_source"] == "link"
    assert params["last_backend_uuid"] == "backend-1"
    assert params["read_write_token"] == "rw-1"  # noqa: S105
    assert params["frontend_context_uuid"] == "ctx-stable"
    assert params["time_from_first_type"] == 8758


def test_build_payload_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown Perplexity model"):
        _build_pplx_payload(query="hi", model_id="not-a-real-model", extras={})


def test_build_payload_space_uuid_forces_collection_query_source() -> None:
    payload = _build_pplx_payload(
        query="ask",
        model_id="perplexity/best",
        extras={"space_uuid": "space-1", "is_incognito": True},
    )
    params = payload["params"]
    assert params["query_source"] == "collection"
    assert params["target_collection_uuid"] == "space-1"
    assert params["target_thread_access_level"] == 1
    assert params["is_incognito"] is False


def test_build_payload_honors_perplexity_wire_field_overrides() -> None:
    payload = _build_pplx_payload(
        query="ask",
        model_id="perplexity/best",
        extras={
            "source": "sidebar",
            "sources": ["scholar", "edgar"],
            "search_focus": "writing",
            "search_recency_filter": "DAY",
            "is_incognito": "true",
            "skip_search_enabled": False,
            "is_nav_suggestions_disabled": False,
            "always_search_override": True,
            "override_no_search": True,
        },
    )
    params = payload["params"]
    assert params["source"] == "sidebar"
    assert params["sources"] == ["scholar", "edgar"]
    assert params["search_focus"] == "writing"
    assert params["search_recency_filter"] == "DAY"
    assert params["is_incognito"] is True
    assert params["skip_search_enabled"] is False
    assert params["is_nav_suggestions_disabled"] is False
    assert params["always_search_override"] is True
    assert params["override_no_search"] is True


def test_flatten_messages_drops_image_url_parts() -> None:
    messages = [
        {"role": "system", "content": "you are helpful"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this image?"},
                {"type": "image_url", "image_url": {"url": "http://x/img.png"}},
            ],
        },
    ]
    out = _flatten_messages(messages)
    assert out.startswith("[System]: you are helpful")
    assert "what is in this image?" in out
    assert "image_url" not in out


def test_flatten_last_user_turn_extracts_only_new_turn() -> None:
    assert (
        _flatten_last_user_turn(
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ]
        )
        == "c"
    )

    assert (
        _flatten_last_user_turn(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {"type": "image_url", "image_url": {"url": "http://x/img.png"}},
                    ],
                }
            ]
        )
        == "hi"
    )

    assert (
        _flatten_last_user_turn(
            [
                {"role": "user", "content": "a"},
                {"role": "tool", "content": "result"},
                {"role": "user", "content": "b"},
            ]
        )
        == "b"
    )

    assert _flatten_last_user_turn([]) == ""
    assert _flatten_last_user_turn([{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}]) == ""


def test_parse_sse_line_basic() -> None:
    assert _parse_sse_line('data: {"a": 1}') == {"a": 1}
    assert _parse_sse_line(b'data: {"b": 2}') == {"b": 2}
    assert _parse_sse_line("event: ping") is None
    assert _parse_sse_line("data: [DONE]") is None
    assert _parse_sse_line("not data") is None


def test_extract_deltas_prefix_diffs_answer_and_reasoning() -> None:
    state = StreamState()
    e1 = {
        "blocks": [
            {
                "intended_usage": "ask_text_0_markdown",
                "diff_block": {
                    "field": "markdown_block",
                    "patches": [
                        {"path": "/markdown_block", "value": {"answer": "Hello"}},
                    ],
                },
            }
        ],
        "backend_uuid": "B-1",
        "context_uuid": "C-1",
    }
    ans, reason = _extract_deltas(e1, state)
    assert ans == "Hello"
    assert reason is None
    assert state.ids["backend_uuid"] == "B-1"

    e2 = {
        "blocks": [
            {
                "intended_usage": "ask_text_0_markdown",
                "diff_block": {
                    "field": "markdown_block",
                    "patches": [
                        {"path": "/markdown_block", "value": {"answer": "Hello, world"}},
                    ],
                },
            },
            {
                "intended_usage": "pro_search_steps",
                "plan_block": {"goals": [{"description": "Searching"}]},
            },
        ]
    }
    ans, reason = _extract_deltas(e2, state)
    assert ans == ", world"
    assert reason == "Searching"

    e3 = {"final_sse_message": True, "thread_url_slug": "slug-1", "read_write_token": "rw-1"}
    ans, reason = _extract_deltas(e3, state)
    assert ans is None
    assert reason is None
    assert state.final is True
    assert state.ids["thread_url_slug"] == "slug-1"
    assert state.ids["read_write_token"] == "rw-1"  # noqa: S105


def test_extract_deltas_raises_on_clarifying_questions() -> None:
    state = StreamState()
    event = {
        "text": json.dumps([{"step_type": "RESEARCH_CLARIFYING_QUESTIONS", "content": {"questions": ["a?", "b?"]}}])
    }
    with pytest.raises(PerplexityClarifyingQuestionsError) as exc_info:
        _extract_deltas(event, state)
    assert exc_info.value.questions == ["a?", "b?"]


def test_thread_to_openai_messages_round_trip() -> None:
    """Convert a thread (real ``GET /rest/thread/<slug>`` shape) to OpenAI messages.

    Each entry has ``blocks[]`` keyed by ``intended_usage``; the
    ``ask_text_0_markdown`` block carries the answer markdown, the
    ``web_results`` block carries citation sources.
    """
    thread = {
        "entries": [
            {
                "query_str": "what is quantum computing?",
                "structured_answer_block_usages": ["ask_text_0_markdown"],
                "blocks": [
                    {
                        "intended_usage": "ask_text_0_markdown",
                        "markdown_block": {"answer": "Quantum [1] computing [2]."},
                    },
                    {
                        "intended_usage": "web_results",
                        "web_result_block": {
                            "web_results": [
                                {"url": "http://a"},
                                {"url": "http://b"},
                            ]
                        },
                    },
                ],
            },
            {
                "query_str": "follow up",
                "structured_answer_block_usages": ["ask_text_0_markdown"],
                "blocks": [
                    {
                        "intended_usage": "ask_text_0_markdown",
                        "markdown_block": {"answer": "Plain answer."},
                    },
                ],
            },
        ]
    }
    msgs = _thread_to_openai_messages(thread, citation_mode="markdown")
    assert len(msgs) == 4
    assert msgs[0] == {"role": "user", "content": "what is quantum computing?"}
    assert msgs[1]["role"] == "assistant"
    assert "[1](http://a)" in msgs[1]["content"]
    assert "[2](http://b)" in msgs[1]["content"]
    assert msgs[2] == {"role": "user", "content": "follow up"}
    assert msgs[3] == {"role": "assistant", "content": "Plain answer."}


def test_thread_to_openai_messages_include_reasoning() -> None:
    """When ``include_reasoning=True``, plan_block.goals descriptions are appended."""
    thread = {
        "entries": [
            {
                "query_str": "q",
                "structured_answer_block_usages": ["ask_text_0_markdown"],
                "blocks": [
                    {
                        "intended_usage": "ask_text_0_markdown",
                        "markdown_block": {"answer": "answer text"},
                    },
                    {
                        "intended_usage": "pro_search_steps",
                        "plan_block": {
                            "goals": [
                                {"description": "Looking up X"},
                                {"description": "Comparing Y"},
                            ]
                        },
                    },
                ],
            }
        ]
    }
    msgs = _thread_to_openai_messages(thread, include_reasoning=True)
    assert msgs[1]["role"] == "assistant"
    content = msgs[1]["content"]
    assert "answer text" in content
    assert "**Reasoning:**" in content
    assert "- Looking up X" in content
    assert "- Comparing Y" in content


def test_thread_to_openai_messages_uses_structured_answer_block_usages_hint() -> None:
    """When the hint names a non-default block, the helper follows it."""
    thread = {
        "entries": [
            {
                "query_str": "q",
                "structured_answer_block_usages": ["alternate_answer_iu"],
                "blocks": [
                    {
                        "intended_usage": "ask_text_0_markdown",
                        "markdown_block": {"answer": "WRONG"},
                    },
                    {
                        "intended_usage": "alternate_answer_iu",
                        "markdown_block": {"answer": "RIGHT"},
                    },
                ],
            }
        ]
    }
    msgs = _thread_to_openai_messages(thread)
    assert msgs[1]["content"] == "RIGHT"


def test_thread_to_openai_messages_real_fixture_news_claude() -> None:
    """Regression: real Perplexity thread shape from a 2026-05-18 capture.

    Fixture: ``research/pplx/response-content/threads/raw/upstream-news-claude-*.json``
    A query about the latest Claude model; verifies the parser produces a
    user/assistant pair with markdown-formatted citations.
    """
    from pathlib import Path

    fixture_dir = Path(__file__).parent / "fixtures" / "pplx_threads"
    fixture = fixture_dir / "upstream-news-claude.json"
    if not fixture.exists():
        pytest.skip(f"missing fixture {fixture}")
    thread = json.loads(fixture.read_text(encoding="utf-8"))
    msgs = _thread_to_openai_messages(thread, citation_mode="markdown")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert "latest Anthropic Claude model" in msgs[0]["content"]
    assert msgs[1]["role"] == "assistant"
    answer = msgs[1]["content"]
    # The answer talks about Claude Opus 4.7 and has markdown citations.
    assert "Claude Opus 4.7" in answer
    assert "[1](http" in answer  # citation reformatted as markdown link


def test_thread_store_save_get_lifecycle() -> None:
    clear_pplx_threads()
    store = get_pplx_thread_store()
    store.save(
        conversation_id="conv-1",
        backend_uuid="B-1",
        read_write_token="RW-1",  # noqa: S106
        context_uuid="C-1",
        thread_url_slug="slug-1",
    )
    state = store.get("conv-1")
    assert state is not None
    assert state.backend_uuid == "B-1"
    assert state.thread_url_slug == "slug-1"
    assert store.get("nonexistent") is None


def test_thread_store_ttl_eviction() -> None:
    store = PerplexityThreadStore(ttl_seconds=0.05)
    store.save(
        conversation_id="conv-1",
        backend_uuid="B-1",
        read_write_token="RW-1",  # noqa: S106
        context_uuid="C-1",
        thread_url_slug="slug-1",
    )
    assert store.size() == 1
    time.sleep(0.1)
    store.save(
        conversation_id="conv-2",
        backend_uuid="B-2",
        read_write_token="RW-2",  # noqa: S106
        context_uuid="C-2",
        thread_url_slug="slug-2",
    )
    assert store.get("conv-1") is None
    assert store.get("conv-2") is not None


def test_pplx_thread_config_defaults() -> None:
    cfg = PplxConfig()
    assert cfg.thread.consistency_mode == "warn"
    assert cfg.thread.citation_mode == "markdown"
    assert cfg.thread.ttl_seconds == 1800.0
    assert cfg.thread.fetch_page_size == 100


def test_pplx_thread_config_rejects_invalid_literal() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PplxThreadConfig(consistency_mode="bogus")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValidationError):
        PplxThreadConfig(citation_mode="bogus")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValidationError):
        PplxThreadConfig(ttl_seconds=-1)


def test_extract_pplx_files_data_uri_path() -> None:
    from ccproxy.hooks.extract_pplx_files import _decode_data_uri

    info = _decode_data_uri(
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    assert info is not None
    assert info.mimetype == "image/png"
    assert info.is_image is True


def test_count_client_user_turns_with_system_messages() -> None:
    from ccproxy.hooks.pplx_thread_inject import _count_client_user_turns

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3-new"},
    ]
    assert _count_client_user_turns(messages) == 2


def test_pplx_addon_scan_for_ids() -> None:
    from ccproxy.inspector.pplx_addon import PerplexityAddon

    raw = (
        b'data: {"backend_uuid":"B-1","context_uuid":"C-1","thread_url_slug":"slug-X","blocks":[]}\n'
        b'data: {"final":true,"read_write_token":"RW-1","blocks":[]}'
    )
    ids = PerplexityAddon._scan_for_ids(raw)
    assert ids == {
        "backend_uuid": "B-1",
        "context_uuid": "C-1",
        "thread_url_slug": "slug-X",
        "read_write_token": "RW-1",
    }


def _make_payload_bytes(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


# --- Step rendering integration tests (plan_block.steps[] + non-spec fields) ---


def _mcp_event(step_type: str, *, uuid: str, content: dict[str, Any]) -> dict[str, Any]:
    """Synthesize a pro_search_steps event carrying one plan_block step."""
    return {
        "blocks": [
            {
                "intended_usage": "pro_search_steps",
                "plan_block": {
                    "progress": "IN_PROGRESS",
                    "goals": [],
                    "steps": [
                        {
                            "uuid": uuid,
                            "step_type": step_type,
                            f"{step_type.lower()}_content": content,
                        }
                    ],
                    "final": False,
                },
            }
        ],
        "display_model": "claude46sonnet",
    }


def test_extract_deltas_walks_plan_block_steps_for_mcp() -> None:
    from ccproxy.lightllm.pplx import StreamState, _extract_deltas

    state = StreamState()
    event = _mcp_event(
        "MCP_TOOL_INPUT",
        uuid="step-1",
        content={
            "goal_id": "0",
            "tool_name": "get_me",
            "tool_args": {},
            "app": "GitHub",
            "tool_input_summary": "Getting user info",
            "request_user_approval": {"request_user_approval": False},
            "mcp_server_type": "MCP_SERVER_TYPE_REMOTE",
            "source_type": "github_mcp_direct",
        },
    )
    _, reasoning = _extract_deltas(event, state)
    assert reasoning is not None
    assert "[GitHub] get_me" in reasoning
    assert len(state.mcp_steps) == 1
    assert state.mcp_steps[0]["tool_name"] == "get_me"
    assert state.mcp_steps[0]["app"] == "GitHub"
    assert len(state.all_steps) == 1
    assert state.all_steps[0]["step_type"] == "MCP_TOOL_INPUT"
    assert "step-1" in state.seen_step_uuids


def test_extract_deltas_dedups_step_uuid_across_events() -> None:
    from ccproxy.lightllm.pplx import StreamState, _extract_deltas

    state = StreamState()
    event = _mcp_event(
        "MCP_TOOL_INPUT",
        uuid="dup-1",
        content={"tool_name": "x", "tool_args": {}, "app": "GitHub"},
    )
    _extract_deltas(event, state)
    _extract_deltas(event, state)
    _extract_deltas(event, state)
    assert len(state.mcp_steps) == 1  # only once across 3 cumulative events
    assert len(state.all_steps) == 1


def test_extract_deltas_captures_goals_snapshot() -> None:
    from ccproxy.lightllm.pplx import StreamState, _extract_deltas

    state = StreamState()
    event = {
        "blocks": [
            {
                "intended_usage": "plan",
                "plan_block": {
                    "progress": "DONE",
                    "goals": [
                        {"id": "0", "description": "Opening GitHub", "final": True},
                        {"id": "1", "description": "Searching PRs", "final": True},
                    ],
                    "steps": [],
                    "final": True,
                },
            }
        ]
    }
    _extract_deltas(event, state)
    assert len(state.goals) == 2
    assert state.goals[0]["description"] == "Opening GitHub"


def test_extract_deltas_handles_bare_markdown_block() -> None:
    """Terminal event ships markdown_block directly under the block (no diff_block)."""
    from ccproxy.lightllm.pplx import StreamState, _extract_deltas

    state = StreamState()
    state.answer_seen = "Hello"  # simulate diff_block already accumulated this
    event = {
        "blocks": [
            {
                "intended_usage": "ask_text_0_markdown",
                "markdown_block": {
                    "progress": "DONE",
                    "answer": "Hello, world!",
                    "chunks": [],
                },
            }
        ]
    }
    answer_delta, _ = _extract_deltas(event, state)
    assert answer_delta == ", world!"
    assert state.answer_seen == "Hello, world!"


def test_extract_deltas_logs_unknown_intended_usage(caplog) -> None:
    import logging

    from ccproxy.lightllm.pplx import StreamState, _extract_deltas

    state = StreamState()
    event = {"blocks": [{"intended_usage": "totally_new_block_type", "totally_new_block": {}}]}
    with caplog.at_level(logging.DEBUG, logger="ccproxy.lightllm.pplx"):
        _extract_deltas(event, state)
    assert "totally_new_block_type" in state.logged_unknown_intended_usages
    assert any("totally_new_block_type" in r.message for r in caplog.records)
    # Re-fire — should NOT log again (dedup).
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="ccproxy.lightllm.pplx"):
        _extract_deltas(event, state)
    assert not any("totally_new_block_type" in r.message for r in caplog.records)


def test_text_field_steps_skipped_when_plan_block_present() -> None:
    """Avoid double-emit: the structured channel wins when both exist in one event."""
    from ccproxy.lightllm.pplx import StreamState, _extract_deltas

    state = StreamState()
    event = {
        "text": json.dumps(
            [{"step_type": "MCP_TOOL_INPUT", "uuid": "from-text", "content": {"tool_name": "x", "app": "A"}}]
        ),
        "blocks": [
            {
                "intended_usage": "pro_search_steps",
                "plan_block": {
                    "steps": [
                        {
                            "step_type": "MCP_TOOL_INPUT",
                            "uuid": "from-structured",
                            "mcp_tool_input_content": {"tool_name": "y", "app": "B"},
                        }
                    ],
                    "goals": [],
                },
            }
        ],
    }
    _extract_deltas(event, state)
    # Only the structured channel step was consumed
    assert len(state.mcp_steps) == 1
    assert state.mcp_steps[0]["tool_name"] == "y"


def test_text_field_steps_processed_when_no_plan_block() -> None:
    from ccproxy.lightllm.pplx import StreamState, _extract_deltas

    state = StreamState()
    event = {
        "text": json.dumps(
            [{"step_type": "MCP_TOOL_INPUT", "uuid": "text-only", "content": {"tool_name": "z", "app": "C"}}]
        ),
        "blocks": [],
    }
    _, reasoning = _extract_deltas(event, state)
    assert reasoning is not None
    assert "[C] z" in reasoning
    assert len(state.mcp_steps) == 1
