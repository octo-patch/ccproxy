"""Tests for the OpenAI Chat Completions inbound parser."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic_ai.messages import (
    INVALID_JSON_KEY,
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ccproxy.lightllm.openai_inbound import parse_openai_chat

# ---------------------------------------------------------------------------
# Simple roles: system / developer / user / assistant / tool
# ---------------------------------------------------------------------------


class TestRoles:
    async def test_system_string(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": "Be helpful."}],
        }
        result = await parse_openai_chat(body)
        assert len(result.messages) == 1
        msg = result.messages[0]
        assert isinstance(msg, ModelRequest)
        assert isinstance(msg.parts[0], SystemPromptPart)
        assert msg.parts[0].content == "Be helpful."

    async def test_developer_role_maps_to_system(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "developer", "content": "Stay focused."}],
        }
        result = await parse_openai_chat(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelRequest)
        assert isinstance(msg.parts[0], SystemPromptPart)
        assert msg.parts[0].content == "Stay focused."

    async def test_user_string(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi."}],
        }
        result = await parse_openai_chat(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelRequest)
        assert isinstance(msg.parts[0], UserPromptPart)
        assert msg.parts[0].content == "Hi."

    async def test_user_content_blocks(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "one"},
                        {"type": "text", "text": "two"},
                    ],
                }
            ],
        }
        result = await parse_openai_chat(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelRequest)
        part = msg.parts[0]
        assert isinstance(part, UserPromptPart)
        assert isinstance(part.content, list)
        assert part.content == ["one", "two"]

    async def test_assistant_text(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "assistant", "content": "Hello back."}],
        }
        result = await parse_openai_chat(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelResponse)
        assert isinstance(msg.parts[0], TextPart)
        assert msg.parts[0].content == "Hello back."

    async def test_assistant_content_blocks(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": "second"},
                    ],
                }
            ],
        }
        result = await parse_openai_chat(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelResponse)
        assert [getattr(p, "content", None) for p in msg.parts] == ["first", "second"]


# ---------------------------------------------------------------------------
# Tool calls + tool results
# ---------------------------------------------------------------------------


class TestToolCalls:
    async def test_assistant_tool_calls_with_string_arguments(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "foo.txt", "limit": 10}',
                            },
                        }
                    ],
                }
            ],
        }
        result = await parse_openai_chat(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelResponse)
        assert len(msg.parts) == 1
        part = msg.parts[0]
        assert isinstance(part, ToolCallPart)
        assert part.tool_name == "read_file"
        assert part.tool_call_id == "call_1"
        assert part.args == {"path": "foo.txt", "limit": 10}

    async def test_assistant_tool_calls_then_text(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Here goes.",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                }
            ],
        }
        result = await parse_openai_chat(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelResponse)
        kinds = [type(p).__name__ for p in msg.parts]
        assert kinds == ["TextPart", "ToolCallPart"]
        text_part = msg.parts[0]
        assert isinstance(text_part, TextPart)
        assert text_part.content == "Here goes."

    async def test_tool_message_resolves_tool_name(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_x",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_x",
                    "content": "search results here",
                },
            ],
        }
        result = await parse_openai_chat(body)
        assert isinstance(result.messages[0], ModelResponse)
        tool_return_msg = result.messages[1]
        assert isinstance(tool_return_msg, ModelRequest)
        assert len(tool_return_msg.parts) == 1
        part = tool_return_msg.parts[0]
        assert isinstance(part, ToolReturnPart)
        assert part.tool_call_id == "call_x"
        assert part.tool_name == "search"
        assert part.content == "search results here"

    async def test_tool_message_with_list_content_flattens_text(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_z",
                            "type": "function",
                            "function": {"name": "fetch", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_z",
                    "content": [
                        {"type": "text", "text": "alpha"},
                        {"type": "text", "text": "beta"},
                    ],
                },
            ],
        }
        result = await parse_openai_chat(body)
        tool_return_msg = result.messages[1]
        assert isinstance(tool_return_msg, ModelRequest)
        part = tool_return_msg.parts[0]
        assert isinstance(part, ToolReturnPart)
        assert part.content == "alphabeta"


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

_PNG_PIXEL_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8A"
    "AAAASUVORK5CYII="
)


class TestImages:
    async def test_image_url_data_uri_becomes_binary_content(self) -> None:
        data_uri = f"data:image/png;base64,{_PNG_PIXEL_B64}"
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        }
        result = await parse_openai_chat(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelRequest)
        part = msg.parts[0]
        assert isinstance(part, UserPromptPart)
        assert isinstance(part.content, list)
        item = part.content[0]
        assert isinstance(item, BinaryContent)
        assert item.media_type == "image/png"
        assert item.data == base64.b64decode(_PNG_PIXEL_B64)

    async def test_image_url_https_becomes_image_url(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/cat.png",
                                "detail": "high",
                            },
                        }
                    ],
                }
            ],
        }
        result = await parse_openai_chat(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelRequest)
        part = msg.parts[0]
        assert isinstance(part, UserPromptPart)
        assert isinstance(part.content, list)
        item = part.content[0]
        assert isinstance(item, ImageUrl)
        assert item.url == "https://example.com/cat.png"
        assert result.raw_extras.get("image_detail:msg:0:block:0") == "high"


# ---------------------------------------------------------------------------
# Tools list + tool_choice + response_format
# ---------------------------------------------------------------------------


class TestRequestParameters:
    async def test_tools_become_function_tools(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
        }
        result = await parse_openai_chat(body)
        tools = result.request_parameters.function_tools
        assert len(tools) == 1
        assert tools[0].name == "read_file"
        assert tools[0].description == "Read a file"
        assert tools[0].parameters_json_schema == {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    async def test_tool_choice_stashed_in_raw_extras(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [],
            "tool_choice": "required",
        }
        result = await parse_openai_chat(body)
        assert result.raw_extras["tool_choice"] == "required"

    async def test_response_format_stashed_in_raw_extras(self) -> None:
        rf = {
            "type": "json_schema",
            "json_schema": {"name": "x", "schema": {"type": "object"}},
        }
        body = {"model": "gpt-4o", "messages": [], "response_format": rf}
        result = await parse_openai_chat(body)
        assert result.raw_extras["response_format"] == rf


# ---------------------------------------------------------------------------
# ModelSettings mapping
# ---------------------------------------------------------------------------


class TestSettings:
    async def test_common_sampling_fields(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [],
            "temperature": 0.5,
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "logit_bias": {"50256": -100},
            "seed": 42,
            "parallel_tool_calls": False,
        }
        result = await parse_openai_chat(body)
        s = result.settings
        assert s.get("temperature") == 0.5
        assert s.get("top_p") == 0.9
        assert s.get("presence_penalty") == 0.1
        assert s.get("frequency_penalty") == 0.2
        assert s.get("logit_bias") == {"50256": -100}
        assert s.get("seed") == 42
        assert s.get("parallel_tool_calls") is False

    async def test_max_completion_tokens_wins_over_max_tokens(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [],
            "max_tokens": 100,
            "max_completion_tokens": 200,
        }
        result = await parse_openai_chat(body)
        assert result.settings.get("max_tokens") == 200

    async def test_max_tokens_only(self) -> None:
        body = {"model": "gpt-4o", "messages": [], "max_tokens": 50}
        result = await parse_openai_chat(body)
        assert result.settings.get("max_tokens") == 50

    async def test_stop_string_becomes_stop_sequences_list(self) -> None:
        body = {"model": "gpt-4o", "messages": [], "stop": "\n"}
        result = await parse_openai_chat(body)
        assert result.settings.get("stop_sequences") == ["\n"]

    async def test_stop_list_passes_through(self) -> None:
        body = {"model": "gpt-4o", "messages": [], "stop": ["END", "STOP"]}
        result = await parse_openai_chat(body)
        assert result.settings.get("stop_sequences") == ["END", "STOP"]

    async def test_logprobs_and_top_logprobs(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [],
            "logprobs": True,
            "top_logprobs": 5,
        }
        result = await parse_openai_chat(body)
        assert result.settings.get("openai_logprobs") is True
        assert result.settings.get("openai_top_logprobs") == 5

    async def test_user_field(self) -> None:
        body = {"model": "gpt-4o", "messages": [], "user": "***"}
        result = await parse_openai_chat(body)
        assert result.settings.get("openai_user") == "***"
        assert "user" not in result.raw_extras

    async def test_unknown_fields_land_in_raw_extras(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [],
            "custom_field": {"foo": "bar"},
            "some_other_thing": 7,
        }
        result = await parse_openai_chat(body)
        assert result.raw_extras["custom_field"] == {"foo": "bar"}
        assert result.raw_extras["some_other_thing"] == 7


# ---------------------------------------------------------------------------
# Streaming flag
# ---------------------------------------------------------------------------


class TestStream:
    async def test_stream_true(self) -> None:
        body = {"model": "gpt-4o", "messages": [], "stream": True}
        result = await parse_openai_chat(body)
        assert result.stream is True

    async def test_stream_false(self) -> None:
        body = {"model": "gpt-4o", "messages": [], "stream": False}
        result = await parse_openai_chat(body)
        assert result.stream is False

    async def test_stream_default(self) -> None:
        body = {"model": "gpt-4o", "messages": []}
        result = await parse_openai_chat(body)
        assert result.stream is False


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    async def test_refusal_top_level_field(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "refusal": "I can't help with that.",
                }
            ],
        }
        result = await parse_openai_chat(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelResponse)
        assert len(msg.parts) == 1
        assert isinstance(msg.parts[0], TextPart)
        assert msg.parts[0].content == "I can't help with that."
        assert result.raw_extras["refusal:msg:0"] == "I can't help with that."

    async def test_refusal_block_in_content(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "refusal", "refusal": "Nope."},
                    ],
                }
            ],
        }
        result = await parse_openai_chat(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelResponse)
        assert isinstance(msg.parts[0], TextPart)
        assert msg.parts[0].content == "Nope."
        assert result.raw_extras["refusal:msg:0"] == "Nope."


# ---------------------------------------------------------------------------
# Lossiness regressions (per the brief)
# ---------------------------------------------------------------------------


class TestLossinessRegressions:
    """Four regression cases analogous to the Anthropic parser:

    1. tool_name populated from neighboring tool_calls.
    2. Image media_type preserved.
    3. Invalid JSON args wrapped via INVALID_JSON_KEY.
    4. Unknown blocks preserved in raw_extras.
    """

    async def test_regression_tool_name_populated_from_neighbor(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_42",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_42",
                    "content": "found",
                },
            ],
        }
        result = await parse_openai_chat(body)
        tr = result.messages[1]
        assert isinstance(tr, ModelRequest)
        part = tr.parts[0]
        assert isinstance(part, ToolReturnPart)
        # Regression: tool_name is recovered from the assistant's tool_calls
        assert part.tool_name == "lookup"

    async def test_regression_tool_name_empty_when_no_match(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "orphan",
                    "content": "no matching call",
                }
            ],
        }
        result = await parse_openai_chat(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelRequest)
        part = msg.parts[0]
        assert isinstance(part, ToolReturnPart)
        # Regression: missing match yields empty string (with warning), not crash
        assert part.tool_name == ""
        assert part.tool_call_id == "orphan"

    async def test_regression_image_media_type_preserved(self) -> None:
        # GIF data URI — distinct media_type to prove we don't hardcode png/jpeg
        gif_uri = f"data:image/gif;base64,{_PNG_PIXEL_B64}"
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": gif_uri}},
                    ],
                }
            ],
        }
        result = await parse_openai_chat(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelRequest)
        part = msg.parts[0]
        assert isinstance(part, UserPromptPart)
        assert isinstance(part.content, list)
        item = part.content[0]
        assert isinstance(item, BinaryContent)
        # Regression: media_type preserved
        assert item.media_type == "image/gif"

    async def test_regression_invalid_json_args_wrapped(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "bad_call",
                            "type": "function",
                            "function": {
                                "name": "edit",
                                "arguments": "{not valid json",
                            },
                        }
                    ],
                }
            ],
        }
        result = await parse_openai_chat(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelResponse)
        tcp = msg.parts[0]
        assert isinstance(tcp, ToolCallPart)
        # Regression: malformed JSON wrapped via INVALID_JSON_KEY
        assert tcp.args == {INVALID_JSON_KEY: "{not valid json"}

    async def test_regression_unknown_block_preserved_in_raw_extras(self) -> None:
        unknown = {"type": "video_url", "video_url": {"url": "https://x.com/v.mp4"}}
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "before"}, unknown],
                }
            ],
        }
        result = await parse_openai_chat(body)
        # Regression: unknown blocks preserved
        assert result.raw_extras["unknown_block:msg:0:block:1"] == unknown
        msg = result.messages[0]
        assert isinstance(msg, ModelRequest)
        part = msg.parts[0]
        assert isinstance(part, UserPromptPart)
        assert isinstance(part.content, list)
        # Placeholder emitted so the conversation isn't visibly broken
        assert part.content[0] == "before"
        assert part.content[1] == json.dumps(unknown)


# ---------------------------------------------------------------------------
# Parametrized dataclass-driven cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContentCase:
    name: str
    """Descriptive name for the test scenario."""

    body: dict[str, Any]
    """The full OpenAI Chat Completions body."""

    expected_message_kinds: list[str]
    """Expected sequence of pydantic-ai message class names."""

    expected_first_part_kind: str
    """Expected class name of the first message's first part."""


CONTENT_CASES: list[ContentCase] = [
    ContentCase(
        name="single_user_string",
        body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        },
        expected_message_kinds=["ModelRequest"],
        expected_first_part_kind="UserPromptPart",
    ),
    ContentCase(
        name="single_system_string",
        body={
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": "sys"}],
        },
        expected_message_kinds=["ModelRequest"],
        expected_first_part_kind="SystemPromptPart",
    ),
    ContentCase(
        name="assistant_then_user",
        body={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ack"},
            ],
        },
        expected_message_kinds=["ModelRequest", "ModelResponse"],
        expected_first_part_kind="UserPromptPart",
    ),
    ContentCase(
        name="developer_role",
        body={
            "model": "gpt-4o",
            "messages": [{"role": "developer", "content": "rules"}],
        },
        expected_message_kinds=["ModelRequest"],
        expected_first_part_kind="SystemPromptPart",
    ),
]


@pytest.mark.parametrize(
    "case", [pytest.param(c, id=c.name) for c in CONTENT_CASES]
)
async def test_content_cases(case: ContentCase) -> None:
    """Smoke-table over basic role/content shapes."""
    result = await parse_openai_chat(case.body)
    actual_message_kinds = [type(m).__name__ for m in result.messages]
    assert actual_message_kinds == case.expected_message_kinds
    first_msg = result.messages[0]
    assert type(first_msg.parts[0]).__name__ == case.expected_first_part_kind


# ---------------------------------------------------------------------------
# Combined fidelity case
# ---------------------------------------------------------------------------


class TestCombined:
    async def test_full_round_trip_request_shape(self) -> None:
        """A realistic OpenAI body exercises most of the parser at once."""
        body = {
            "model": "gpt-4o-2024-08-06",
            "messages": [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "How big is 2+2?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_arith",
                            "type": "function",
                            "function": {
                                "name": "calc",
                                "arguments": '{"expression": "2+2"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_arith",
                    "content": "4",
                },
                {"role": "assistant", "content": "4."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calc",
                        "description": "Evaluate an arithmetic expression",
                        "parameters": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_completion_tokens": 256,
            "stream": False,
        }
        result = await parse_openai_chat(body)

        assert result.model == "gpt-4o-2024-08-06"
        assert result.stream is False
        assert result.settings.get("temperature") == 0.0
        assert result.settings.get("max_tokens") == 256
        assert result.raw_extras["tool_choice"] == "auto"

        kinds = [type(m).__name__ for m in result.messages]
        assert kinds == [
            "ModelRequest",
            "ModelRequest",
            "ModelResponse",
            "ModelRequest",
            "ModelResponse",
        ]

        sys_msg = result.messages[0]
        assert isinstance(sys_msg, ModelRequest)
        assert isinstance(sys_msg.parts[0], SystemPromptPart)
        assert sys_msg.parts[0].content == "Be precise."

        tool_call_msg = result.messages[2]
        assert isinstance(tool_call_msg, ModelResponse)
        assert isinstance(tool_call_msg.parts[0], ToolCallPart)
        assert tool_call_msg.parts[0].args == {"expression": "2+2"}

        tool_return_msg = result.messages[3]
        assert isinstance(tool_return_msg, ModelRequest)
        assert isinstance(tool_return_msg.parts[0], ToolReturnPart)
        assert tool_return_msg.parts[0].tool_name == "calc"
        assert tool_return_msg.parts[0].content == "4"

        assert len(result.request_parameters.function_tools) == 1
        assert result.request_parameters.function_tools[0].name == "calc"
