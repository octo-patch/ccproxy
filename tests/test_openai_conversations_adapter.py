"""Tests for :mod:`ccproxy.lightllm.adapters.openai_conversations`.

Covers:
- :func:`build_conversation_body` — new-conversation flatten vs continuation
  single-turn, field invariants, effort / hint mapping, threading fields.
- :func:`build_conversation_prepare_body` — none / sent / success states,
  stripped fields, ``partial_query`` presence, slim contextual info.
- :class:`OpenAIConversationsAdapter.render` — model default fallback,
  ``raw_extras`` wiring, output is valid JSON bytes with expected shape.
- Dispatch seam — ``dispatch_dump_sync("openai_conversations", ...)`` no longer
  raises :class:`~ccproxy.lightllm.graph.UnsupportedUpstreamError`.

Fixtures build :class:`~ccproxy.lightllm.parsed.ParsedRequest` objects directly
(no mitmproxy flow needed), following the pattern established in
``tests/test_lightllm_graph_google_dump.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from ccproxy.lightllm.adapters.openai_conversations import (
    OpenAIConversationsAdapter,
    build_conversation_body,
    build_conversation_prepare_body,
)
from ccproxy.lightllm.graph import UnsupportedUpstreamError, dispatch_dump_sync
from ccproxy.lightllm.parsed import ParsedRequest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_parsed(
    *,
    messages: list[ModelMessage],
    model: str = "gpt-5-5-pro",
    raw_extras: dict[str, Any] | None = None,
    request_parameters: ModelRequestParameters | None = None,
    settings: ModelSettings | None = None,
    stream: bool = False,
) -> ParsedRequest:
    return ParsedRequest(
        model=model,
        messages=messages,
        request_parameters=request_parameters or ModelRequestParameters(),
        settings=settings or ModelSettings(),
        stream=stream,
        raw_extras=raw_extras or {},
    )


def _single_user(text: str = "Hello") -> list[ModelMessage]:
    return [ModelRequest(parts=[UserPromptPart(content=text)])]


def _multi_turn(
    system: str = "Be brief.",
    user1: str = "Hi",
    assistant1: str = "Hello",
    user2: str = "Say bye",
) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[SystemPromptPart(content=system), UserPromptPart(content=user1)]),
        ModelResponse(parts=[TextPart(content=assistant1)]),
        ModelRequest(parts=[UserPromptPart(content=user2)]),
    ]


def _render(parsed: ParsedRequest) -> dict[str, Any]:
    raw = OpenAIConversationsAdapter.render(parsed)
    decoded: dict[str, Any] = json.loads(raw)
    return decoded


# ---------------------------------------------------------------------------
# build_conversation_body — new conversation
# ---------------------------------------------------------------------------


class TestBuildConversationBodyNew:
    def test_action_is_next(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user("hi"),
            model="gpt-5-5-pro",
        )
        assert body["action"] == "next"

    def test_exactly_one_message(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user("hi"),
            model="gpt-5-5-pro",
        )
        assert len(body["messages"]) == 1

    def test_message_shape(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user("hi"),
            model="gpt-5-5-pro",
        )
        msg = body["messages"][0]
        assert msg["author"]["role"] == "user"
        assert msg["content"]["content_type"] == "text"
        assert isinstance(msg["content"]["parts"], list)
        assert len(msg["content"]["parts"]) == 1
        assert "id" in msg
        assert "create_time" in msg

    def test_single_user_message_text(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user("what is 2+2"),
            model="gpt-5-5-pro",
        )
        text = body["messages"][0]["content"]["parts"][0]
        assert text == "what is 2+2"

    def test_multi_turn_flattens_to_one_message(self) -> None:
        body = build_conversation_body(
            messages_ir=_multi_turn(),
            model="gpt-5-5-pro",
        )
        assert len(body["messages"]) == 1

    def test_multi_turn_flatten_includes_prior_turns(self) -> None:
        body = build_conversation_body(
            messages_ir=_multi_turn(
                system="Be brief.",
                user1="Hi",
                assistant1="Hello",
                user2="Say bye",
            ),
            model="gpt-5-5-pro",
        )
        text = body["messages"][0]["content"]["parts"][0]
        assert "system: Be brief." in text
        assert "assistant: Hello" in text
        assert text.endswith("user: Say bye")

    def test_new_conversation_parent_message_id_is_root(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
            is_continuation=False,
        )
        assert body["parent_message_id"] == "client-created-root"

    def test_new_conversation_omits_conversation_id(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
            conversation_id="some-id",
            is_continuation=False,
        )
        assert "conversation_id" not in body

    def test_client_prepare_state_sent_when_prepared(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
            prepared=True,
        )
        assert body["client_prepare_state"] == "sent"

    def test_client_prepare_state_none_when_not_prepared(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
            prepared=False,
        )
        assert body["client_prepare_state"] == "none"

    def test_history_and_training_disabled_false_by_default(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
        )
        assert body["history_and_training_disabled"] is False

    def test_history_and_training_disabled_true_when_temporary(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
            temporary_chat=True,
        )
        assert body["history_and_training_disabled"] is True

    def test_supported_encodings(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
        )
        assert body["supported_encodings"] == ["v1"]

    def test_timezone_and_offset_present(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
        )
        assert "timezone" in body
        assert "timezone_offset_min" in body
        assert isinstance(body["timezone_offset_min"], int)

    def test_client_contextual_info_has_app_name(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
        )
        assert body["client_contextual_info"]["app_name"] == "chatgpt.com"

    def test_conversation_mode(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
        )
        assert body["conversation_mode"] == {"kind": "primary_assistant"}

    def test_supports_buffering(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
        )
        assert body["supports_buffering"] is True

    def test_model_is_propagated(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-thinking",
        )
        assert body["model"] == "gpt-5-thinking"

    def test_system_hints_empty_by_default(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
        )
        assert body["system_hints"] == []

    def test_system_hints_forwarded(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
            system_hints=["picture_v2", "search"],
        )
        assert body["system_hints"] == ["picture_v2", "search"]

    def test_thinking_effort_omitted_when_none(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
            thinking_effort=None,
        )
        assert "thinking_effort" not in body

    def test_thinking_effort_forwarded(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
            thinking_effort="max",
        )
        assert body["thinking_effort"] == "max"


# ---------------------------------------------------------------------------
# build_conversation_body — continuation
# ---------------------------------------------------------------------------


class TestBuildConversationBodyContinuation:
    def test_exactly_one_message_on_continuation(self) -> None:
        body = build_conversation_body(
            messages_ir=_multi_turn(),
            model="gpt-5-5-pro",
            conversation_id="conv-abc",
            parent_message_id="msg-xyz",
            is_continuation=True,
        )
        assert len(body["messages"]) == 1

    def test_continuation_emits_only_last_user_turn(self) -> None:
        body = build_conversation_body(
            messages_ir=_multi_turn(
                system="Be brief.",
                user1="Hi",
                assistant1="Hello",
                user2="Say bye",
            ),
            model="gpt-5-5-pro",
            conversation_id="conv-abc",
            parent_message_id="msg-xyz",
            is_continuation=True,
        )
        text = body["messages"][0]["content"]["parts"][0]
        assert text == "Say bye"
        assert "Hi" not in text
        assert "Hello" not in text
        assert "Be brief" not in text

    def test_continuation_sets_parent_message_id(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
            conversation_id="conv-abc",
            parent_message_id="msg-xyz",
            is_continuation=True,
        )
        assert body["parent_message_id"] == "msg-xyz"

    def test_continuation_includes_conversation_id(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="gpt-5-5-pro",
            conversation_id="conv-abc",
            parent_message_id="msg-xyz",
            is_continuation=True,
        )
        assert body["conversation_id"] == "conv-abc"


# ---------------------------------------------------------------------------
# build_conversation_prepare_body
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrepareBodyTestCase:
    name: str
    """Descriptive test scenario name (used as pytest id)."""

    state: Literal["none", "sent", "success"]
    """Prepare state: ``"none"``, ``"sent"``, or ``"success"``."""

    expect_partial_query: bool
    """Whether ``partial_query`` should be present in the prepare body."""


_PREPARE_BODY_TEST_CASES: list[PrepareBodyTestCase] = [
    PrepareBodyTestCase(
        name="none_omits_partial_query",
        state="none",
        expect_partial_query=False,
    ),
    PrepareBodyTestCase(
        name="sent_includes_partial_query",
        state="sent",
        expect_partial_query=True,
    ),
    PrepareBodyTestCase(
        name="success_includes_partial_query",
        state="success",
        expect_partial_query=True,
    ),
]


@pytest.fixture
def _final_body() -> dict[str, Any]:
    return build_conversation_body(
        messages_ir=_single_user("Hello world"),
        model="gpt-5-5-pro",
    )


class TestBuildConversationPrepareBody:
    @pytest.mark.parametrize(
        "test_case",
        [pytest.param(tc, id=tc.name) for tc in _PREPARE_BODY_TEST_CASES],
    )
    def test_partial_query_presence(self, test_case: PrepareBodyTestCase, _final_body: dict[str, Any]) -> None:
        prepare = build_conversation_prepare_body(final_body=_final_body, state=test_case.state)
        if test_case.expect_partial_query:
            assert "partial_query" in prepare
        else:
            assert "partial_query" not in prepare

    @pytest.mark.parametrize(
        "test_case",
        [pytest.param(tc, id=tc.name) for tc in _PREPARE_BODY_TEST_CASES],
    )
    def test_client_prepare_state(self, test_case: PrepareBodyTestCase, _final_body: dict[str, Any]) -> None:
        prepare = build_conversation_prepare_body(final_body=_final_body, state=test_case.state)
        assert prepare["client_prepare_state"] == test_case.state

    def test_messages_stripped(self, _final_body: dict[str, Any]) -> None:
        prepare = build_conversation_prepare_body(final_body=_final_body, state="sent")
        assert "messages" not in prepare

    def test_enable_message_followups_stripped(self, _final_body: dict[str, Any]) -> None:
        prepare = build_conversation_prepare_body(final_body=_final_body, state="sent")
        assert "enable_message_followups" not in prepare

    def test_paragen_cot_stripped(self, _final_body: dict[str, Any]) -> None:
        prepare = build_conversation_prepare_body(final_body=_final_body, state="sent")
        assert "paragen_cot_summary_display_override" not in prepare

    def test_force_parallel_switch_stripped(self, _final_body: dict[str, Any]) -> None:
        prepare = build_conversation_prepare_body(final_body=_final_body, state="sent")
        assert "force_parallel_switch" not in prepare

    def test_fork_from_shared_post_false(self, _final_body: dict[str, Any]) -> None:
        prepare = build_conversation_prepare_body(final_body=_final_body, state="none")
        assert prepare["fork_from_shared_post"] is False

    def test_slim_client_contextual_info(self, _final_body: dict[str, Any]) -> None:
        prepare = build_conversation_prepare_body(final_body=_final_body, state="none")
        assert prepare["client_contextual_info"] == {"app_name": "chatgpt.com"}

    def test_partial_query_shape(self, _final_body: dict[str, Any]) -> None:
        prepare = build_conversation_prepare_body(final_body=_final_body, state="sent")
        pq = prepare["partial_query"]
        assert pq["author"]["role"] == "user"
        assert pq["content"]["content_type"] == "text"
        assert isinstance(pq["content"]["parts"], list)
        assert len(pq["content"]["parts"]) == 1
        assert isinstance(pq["id"], str)

    def test_partial_query_text_is_first_5_runes(self, _final_body: dict[str, Any]) -> None:
        prepare = build_conversation_prepare_body(final_body=_final_body, state="sent")
        pq_text = prepare["partial_query"]["content"]["parts"][0]
        assert pq_text == "Hello"

    def test_model_defaults_to_auto_when_empty(self) -> None:
        body = build_conversation_body(
            messages_ir=_single_user(),
            model="",
        )
        prepare = build_conversation_prepare_body(final_body=body, state="none")
        assert prepare["model"] == "auto"

    def test_model_preserved_when_non_empty(self, _final_body: dict[str, Any]) -> None:
        prepare = build_conversation_prepare_body(final_body=_final_body, state="none")
        assert prepare["model"] == "gpt-5-5-pro"


# ---------------------------------------------------------------------------
# OpenAIConversationsAdapter.render — model default and raw_extras wiring
# ---------------------------------------------------------------------------


class TestRenderDefaultModel:
    def test_default_model_when_empty(self) -> None:
        parsed = _build_parsed(messages=_single_user(), model="")
        body = _render(parsed)
        assert body["model"] == "gpt-5-5-pro"

    def test_explicit_model_passes_through(self) -> None:
        parsed = _build_parsed(messages=_single_user(), model="gpt-5-thinking")
        body = _render(parsed)
        assert body["model"] == "gpt-5-thinking"


class TestRenderThreading:
    def test_new_conversation_no_thread_extras(self) -> None:
        parsed = _build_parsed(messages=_single_user("hello"))
        body = _render(parsed)
        assert body["parent_message_id"] == "client-created-root"
        assert "conversation_id" not in body
        assert len(body["messages"]) == 1

    def test_continuation_via_raw_extras(self) -> None:
        parsed = _build_parsed(
            messages=_multi_turn(user2="follow-up"),
            raw_extras={
                "openai_conversations": {
                    "conversation_id": "conv-111",
                    "parent_message_id": "msg-222",
                    "is_continuation": True,
                }
            },
        )
        body = _render(parsed)
        assert body["parent_message_id"] == "msg-222"
        assert body["conversation_id"] == "conv-111"
        text = body["messages"][0]["content"]["parts"][0]
        assert text == "follow-up"
        assert len(body["messages"]) == 1

    def test_is_continuation_false_still_flattens(self) -> None:
        parsed = _build_parsed(
            messages=_multi_turn(),
            raw_extras={
                "openai_conversations": {
                    "conversation_id": "conv-111",
                    "parent_message_id": "msg-222",
                    "is_continuation": False,
                }
            },
        )
        body = _render(parsed)
        assert body["parent_message_id"] == "client-created-root"
        assert "conversation_id" not in body


class TestRenderEffortMapping:
    @pytest.mark.parametrize(
        ("raw_effort", "expected"),
        [
            ("low", "standard"),
            ("medium", "extended"),
            ("high", "max"),
            ("max", "max"),
            ("standard", "standard"),
        ],
    )
    def test_effort_mapped(self, raw_effort: str, expected: str) -> None:
        parsed = _build_parsed(
            messages=_single_user(),
            raw_extras={"thinking_effort": raw_effort},
        )
        body = _render(parsed)
        assert body["thinking_effort"] == expected

    def test_effort_from_extra_body_reasoning(self) -> None:
        parsed = _build_parsed(
            messages=_single_user(),
            raw_extras={"extra_body": {"reasoning": {"effort": "high"}}},
        )
        body = _render(parsed)
        assert body["thinking_effort"] == "max"

    def test_effort_absent_when_not_set(self) -> None:
        parsed = _build_parsed(messages=_single_user())
        body = _render(parsed)
        assert "thinking_effort" not in body


class TestRenderSystemHints:
    def test_hints_from_raw_extras(self) -> None:
        parsed = _build_parsed(
            messages=_single_user(),
            raw_extras={"system_hints": ["picture_v2"]},
        )
        body = _render(parsed)
        assert "picture_v2" in body["system_hints"]

    def test_hints_from_extra_body(self) -> None:
        parsed = _build_parsed(
            messages=_single_user(),
            raw_extras={"extra_body": {"system_hints": ["search"]}},
        )
        body = _render(parsed)
        assert "search" in body["system_hints"]

    def test_hints_from_tools(self) -> None:
        tools = [
            ToolDefinition(name="image_generation", description="", parameters_json_schema={}),
            ToolDefinition(name="web_search_preview", description="", parameters_json_schema={}),
        ]
        params = ModelRequestParameters(function_tools=tools)
        parsed = _build_parsed(
            messages=_single_user(),
            request_parameters=params,
        )
        body = _render(parsed)
        assert "picture_v2" in body["system_hints"]
        assert "search" in body["system_hints"]

    def test_deep_research_hint(self) -> None:
        tools = [
            ToolDefinition(name="deep_research", description="", parameters_json_schema={}),
        ]
        params = ModelRequestParameters(function_tools=tools)
        parsed = _build_parsed(
            messages=_single_user(),
            request_parameters=params,
        )
        body = _render(parsed)
        assert "connector:connector_openai_deep_research" in body["system_hints"]

    def test_unknown_tool_produces_no_hint(self) -> None:
        tools = [
            ToolDefinition(name="function", description="", parameters_json_schema={}),
        ]
        params = ModelRequestParameters(function_tools=tools)
        parsed = _build_parsed(
            messages=_single_user(),
            request_parameters=params,
        )
        body = _render(parsed)
        assert body["system_hints"] == []


class TestRenderInvariantsFromDifferentInputFormats:
    """Assert body shape invariants hold regardless of the listener format.

    These simulate Anthropic, Gemini, and OpenAI Responses IR — all parsed
    into pydantic-ai IR before reaching the adapter.
    """

    def test_anthropic_style_ir(self) -> None:
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content="You are a helpful assistant."),
                    UserPromptPart(content="What is the capital of France?"),
                ]
            )
        ]
        parsed = _build_parsed(messages=messages)
        body = _render(parsed)
        assert body["action"] == "next"
        assert len(body["messages"]) == 1
        text = body["messages"][0]["content"]["parts"][0]
        assert "capital of France" in text
        assert body["client_prepare_state"] == "sent"
        assert "history_and_training_disabled" in body
        assert body["supported_encodings"] == ["v1"]
        assert body["client_contextual_info"]["app_name"] == "chatgpt.com"

    def test_gemini_style_multi_turn_ir(self) -> None:
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there!")]),
            ModelRequest(parts=[UserPromptPart(content="Tell me a joke")]),
        ]
        parsed = _build_parsed(messages=messages)
        body = _render(parsed)
        assert len(body["messages"]) == 1
        text = body["messages"][0]["content"]["parts"][0]
        assert "Tell me a joke" in text

    def test_responses_style_with_system(self) -> None:
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content="You are a coder."),
                    UserPromptPart(content="Write hello world in Python"),
                ]
            )
        ]
        parsed = _build_parsed(messages=messages)
        body = _render(parsed)
        assert len(body["messages"]) == 1
        text = body["messages"][0]["content"]["parts"][0]
        assert "Write hello world in Python" in text


class TestDispatchSeam:
    def test_dispatch_dump_sync_openai_conversations(self) -> None:
        parsed = _build_parsed(messages=_single_user("dispatch test"))
        result = dispatch_dump_sync(parsed, provider_type="openai_conversations")
        body = json.loads(result)
        assert body["action"] == "next"
        assert len(body["messages"]) == 1

    def test_unknown_provider_still_raises(self) -> None:
        parsed = _build_parsed(messages=_single_user())
        with pytest.raises(UnsupportedUpstreamError):
            dispatch_dump_sync(parsed, provider_type="not-a-real-provider")
