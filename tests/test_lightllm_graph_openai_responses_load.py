"""Tests for the OpenAI Responses inbound parser."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai.messages import (
    ImageUrl,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ccproxy.lightllm.adapters._envelope import parse_request, render_request
from ccproxy.lightllm.parsed import InboundFormat, ParsedRequest

Parse = Callable[[dict[str, Any]], ParsedRequest]


@pytest.fixture
def parse() -> Parse:
    def _parse(body: dict[str, Any]) -> ParsedRequest:
        return parse_request(body, inbound_format=InboundFormat.OPENAI_RESPONSES)

    return _parse


# ---------------------------------------------------------------------------
# input: shorthand forms
# ---------------------------------------------------------------------------


class TestInputShorthand:
    def test_bare_string_input(self, parse: Parse) -> None:
        body = {"model": "gpt-5", "input": "Say hello in one word."}
        result = parse(body)
        assert len(result.messages) == 1
        msg = result.messages[0]
        assert isinstance(msg, ModelRequest)
        assert isinstance(msg.parts[0], UserPromptPart)
        assert msg.parts[0].content == ["Say hello in one word."]

    def test_empty_string_input_drops(self, parse: Parse) -> None:
        body = {"model": "gpt-5", "input": ""}
        result = parse(body)
        assert result.messages == []

    def test_missing_input_drops(self, parse: Parse) -> None:
        body = {"model": "gpt-5"}
        result = parse(body)
        assert result.messages == []

    def test_instructions_field(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "instructions": "Be concise.",
            "input": "Hi",
        }
        result = parse(body)
        # Instructions become a leading SystemPromptPart in the same
        # ModelRequest as the user message (MessagesBuilder folds
        # consecutive request parts together).
        parts = [p for m in result.messages if isinstance(m, ModelRequest) for p in m.parts]
        system_parts = [p for p in parts if isinstance(p, SystemPromptPart)]
        assert len(system_parts) == 1
        assert system_parts[0].content == "Be concise."


# ---------------------------------------------------------------------------
# input[] message items: roles + content parts
# ---------------------------------------------------------------------------


class TestMessageItems:
    def test_message_user_input_text(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ],
        }
        result = parse(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelRequest)
        assert isinstance(msg.parts[0], UserPromptPart)
        assert msg.parts[0].content == ["Hello"]

    def test_message_user_input_image(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What's this?"},
                        {
                            "type": "input_image",
                            "image_url": {"url": "https://example.com/img.png"},
                        },
                    ],
                }
            ],
        }
        result = parse(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelRequest)
        part = msg.parts[0]
        assert isinstance(part, UserPromptPart)
        assert isinstance(part.content, list)
        assert part.content[0] == "What's this?"
        assert isinstance(part.content[1], ImageUrl)
        assert part.content[1].url == "https://example.com/img.png"

    def test_message_system_role(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "system", "content": "Be terse."},
                {"type": "message", "role": "user", "content": "Hi"},
            ],
        }
        result = parse(body)
        parts = [p for m in result.messages if isinstance(m, ModelRequest) for p in m.parts]
        system_parts = [p for p in parts if isinstance(p, SystemPromptPart)]
        assert len(system_parts) == 1
        assert system_parts[0].content == "Be terse."

    def test_message_developer_role_maps_to_system(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "developer", "content": "Stay focused."},
            ],
        }
        result = parse(body)
        parts = [p for m in result.messages if isinstance(m, ModelRequest) for p in m.parts]
        assert isinstance(parts[0], SystemPromptPart)
        assert parts[0].content == "Stay focused."

    def test_message_assistant_output_text(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Sure!"}],
                }
            ],
        }
        result = parse(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelResponse)
        assert isinstance(msg.parts[0], TextPart)
        assert msg.parts[0].content == "Sure!"


# ---------------------------------------------------------------------------
# Function calls and tool returns
# ---------------------------------------------------------------------------


class TestToolItems:
    def test_function_call(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"city":"SF"}',
                }
            ],
        }
        result = parse(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelResponse)
        part = msg.parts[0]
        assert isinstance(part, ToolCallPart)
        assert part.tool_name == "get_weather"
        assert part.tool_call_id == "call_abc"
        assert part.args == '{"city":"SF"}'

    def test_function_call_with_dict_args_serialized(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_xyz",
                    "name": "ping",
                    "arguments": {"host": "localhost"},
                }
            ],
        }
        result = parse(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelResponse)
        part = msg.parts[0]
        assert isinstance(part, ToolCallPart)
        # dict args serialize to JSON string in the IR
        assert isinstance(part.args, str)
        assert json.loads(part.args) == {"host": "localhost"}

    def test_function_call_output_resolves_tool_name(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_999",
                    "name": "search",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_999",
                    "output": "result",
                },
            ],
        }
        result = parse(body)
        # First is ModelResponse with ToolCallPart; second is
        # ModelRequest with ToolReturnPart.
        tr_msg = next(
            (
                m
                for m in result.messages
                if isinstance(m, ModelRequest)
                and any(isinstance(p, ToolReturnPart) for p in m.parts)
            ),
            None,
        )
        assert tr_msg is not None
        tr_part = next(p for p in tr_msg.parts if isinstance(p, ToolReturnPart))
        assert tr_part.tool_name == "search"  # resolved via call_id index
        assert tr_part.tool_call_id == "call_999"
        assert tr_part.content == "result"

    def test_function_call_output_unknown_call_id_blank_name(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "orphan",
                    "output": "data",
                },
            ],
        }
        result = parse(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelRequest)
        part = msg.parts[0]
        assert isinstance(part, ToolReturnPart)
        assert part.tool_name == ""  # no matching function_call
        assert part.tool_call_id == "orphan"


# ---------------------------------------------------------------------------
# Reasoning items
# ---------------------------------------------------------------------------


class TestReasoning:
    def test_reasoning_emits_thinking_part_and_stashes_full_dict(
        self, parse: Parse
    ) -> None:
        reasoning_item = {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Thinking about it."}],
            "content": [{"type": "reasoning_text", "text": "Step 1: ..."}],
            "encrypted_content": "OPAQUE_BLOB",
        }
        body = {
            "model": "gpt-5",
            "input": [reasoning_item],
        }
        result = parse(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelResponse)
        part = msg.parts[0]
        assert isinstance(part, ThinkingPart)
        assert "Thinking about it." in part.content
        assert "Step 1: ..." in part.content

        stash = result.raw_extras.get("openai_responses:reasoning:0")
        assert stash is not None
        assert stash["encrypted_content"] == "OPAQUE_BLOB"
        # Full structured dict preserved for round-trip
        assert stash["summary"] == reasoning_item["summary"]
        assert stash["content"] == reasoning_item["content"]


# ---------------------------------------------------------------------------
# Server-side tool kinds + unknown kinds + item IDs
# ---------------------------------------------------------------------------


class TestRawExtrasStash:
    def test_web_search_call_stashes_under_server_tool(self, parse: Parse) -> None:
        item = {
            "type": "web_search_call",
            "id": "ws_1",
            "query": "what's the weather",
            "status": "completed",
        }
        body = {"model": "gpt-5", "input": [item]}
        result = parse(body)
        stash = result.raw_extras.get("openai_responses:server_tool:0")
        assert stash is not None
        assert stash["type"] == "web_search_call"
        # Item ID also recorded for previous_response_id chaining
        assert result.raw_extras.get("openai_responses:item_id:0") == "ws_1"

    def test_unknown_item_type_stashes_under_unknown_item(self, parse: Parse) -> None:
        item = {"type": "speculative_future_kind", "value": 42}
        body = {"model": "gpt-5", "input": [item]}
        result = parse(body)
        stash = result.raw_extras.get("openai_responses:unknown_item:0")
        assert stash is not None
        assert stash["type"] == "speculative_future_kind"

    def test_mcp_call_stashes_under_server_tool(self, parse: Parse) -> None:
        item = {
            "type": "mcp_call",
            "id": "mcp_call_1",
            "name": "list_files",
            "server_label": "fs",
        }
        body = {"model": "gpt-5", "input": [item]}
        result = parse(body)
        assert "openai_responses:server_tool:0" in result.raw_extras


# ---------------------------------------------------------------------------
# Settings + tools
# ---------------------------------------------------------------------------


class TestSettingsAndTools:
    def test_max_output_tokens_maps_to_max_tokens(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": "hi",
            "max_output_tokens": 128,
            "temperature": 0.4,
            "top_p": 0.9,
        }
        result = parse(body)
        settings = dict(result.settings)
        assert settings["max_tokens"] == 128
        assert settings["temperature"] == 0.4
        assert settings["top_p"] == 0.9

    def test_function_tools_share_chat_shape(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Look up weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        }
        result = parse(body)
        tools = list(result.request_parameters.function_tools)
        assert len(tools) == 1
        assert tools[0].name == "get_weather"
        assert tools[0].description == "Look up weather"

    def test_unknown_top_level_keys_preserved_in_raw_extras(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": "hi",
            "previous_response_id": "resp_prev",
            "prompt_cache_key": "key1",
            "prompt_cache_retention": "in-memory",
            "reasoning": {"effort": "high"},
        }
        result = parse(body)
        assert result.raw_extras.get("previous_response_id") == "resp_prev"
        assert result.raw_extras.get("prompt_cache_key") == "key1"
        assert result.raw_extras.get("prompt_cache_retention") == "in-memory"
        assert result.raw_extras.get("reasoning") == {"effort": "high"}


# ---------------------------------------------------------------------------
# Render round-trip (the critical Phase 4A pipeline path)
# ---------------------------------------------------------------------------


class TestRenderRoundTrip:
    def test_bare_string_renders_to_verbose_message(self, parse: Parse) -> None:
        body = {"model": "gpt-5", "input": "Hello.", "max_output_tokens": 50}
        result = parse(body)
        rendered = render_request(result, inbound_format=InboundFormat.OPENAI_RESPONSES)
        out = json.loads(rendered)
        assert out["model"] == "gpt-5"
        assert out["max_output_tokens"] == 50
        assert out["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello."}],
            }
        ]

    def test_instructions_round_trips(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "instructions": "Be concise.",
            "input": "Hi",
        }
        result = parse(body)
        rendered = render_request(result, inbound_format=InboundFormat.OPENAI_RESPONSES)
        out = json.loads(rendered)
        assert out["instructions"] == "Be concise."

    def test_unknown_top_level_keys_pass_through(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": "hi",
            "previous_response_id": "resp_prev",
            "prompt_cache_key": "key1",
        }
        result = parse(body)
        rendered = render_request(result, inbound_format=InboundFormat.OPENAI_RESPONSES)
        out = json.loads(rendered)
        assert out["previous_response_id"] == "resp_prev"
        assert out["prompt_cache_key"] == "key1"

    def test_server_tool_item_round_trips(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "What's new?"},
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "query": "news",
                },
            ],
        }
        result = parse(body)
        rendered = render_request(result, inbound_format=InboundFormat.OPENAI_RESPONSES)
        out = json.loads(rendered)
        # The web_search_call survives at its original index (1).
        kinds = [item.get("type") for item in out["input"]]
        assert "web_search_call" in kinds

    def test_function_call_round_trips(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "lookup",
                    "arguments": '{"q":"hi"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_abc",
                    "output": "done",
                },
            ],
        }
        result = parse(body)
        rendered = render_request(result, inbound_format=InboundFormat.OPENAI_RESPONSES)
        out = json.loads(rendered)
        kinds = [item.get("type") for item in out["input"]]
        assert "function_call" in kinds
        assert "function_call_output" in kinds

    def test_stream_flag_round_trips(self, parse: Parse) -> None:
        body = {"model": "gpt-5", "input": "hi", "stream": True}
        result = parse(body)
        rendered = render_request(result, inbound_format=InboundFormat.OPENAI_RESPONSES)
        out = json.loads(rendered)
        assert out["stream"] is True

    def test_tools_round_trip(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ping",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
        result = parse(body)
        rendered = render_request(result, inbound_format=InboundFormat.OPENAI_RESPONSES)
        out = json.loads(rendered)
        assert out["tools"][0]["function"]["name"] == "ping"

    def test_image_in_user_message_round_trips(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Look:"},
                        {
                            "type": "input_image",
                            "image_url": {"url": "https://x/a.png"},
                        },
                    ],
                }
            ],
        }
        result = parse(body)
        rendered = render_request(result, inbound_format=InboundFormat.OPENAI_RESPONSES)
        out = json.loads(rendered)
        content = out["input"][0]["content"]
        kinds = [p["type"] for p in content]
        assert "input_text" in kinds
        assert "input_image" in kinds
        img = next(p for p in content if p["type"] == "input_image")
        assert img["image_url"]["url"] == "https://x/a.png"


# ---------------------------------------------------------------------------
# Edge cases: non-dict items, image URL string form, refusal stashing
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_non_dict_input_item_stashes_unknown(self, parse: Parse) -> None:
        body = {"model": "gpt-5", "input": ["not-a-dict"]}
        result = parse(body)
        assert "openai_responses:unknown_item:0" in result.raw_extras

    def test_image_url_string_form_accepted(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "https://x/y.png"},
                    ],
                }
            ],
        }
        result = parse(body)
        msg = result.messages[0]
        assert isinstance(msg, ModelRequest)
        part = msg.parts[0]
        assert isinstance(part, UserPromptPart)
        assert isinstance(part.content[0], ImageUrl)
        assert part.content[0].url == "https://x/y.png"

    def test_assistant_refusal_content_stashed(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "Can't help."}],
                }
            ],
        }
        result = parse(body)
        keys = [k for k in result.raw_extras if k.startswith("openai_responses:refusal:")]
        assert keys, f"expected refusal stash, got: {list(result.raw_extras)}"

    def test_unknown_role_stashes(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [{"type": "message", "role": "tool", "content": "x"}],
        }
        result = parse(body)
        assert "openai_responses:unknown_item:0" in result.raw_extras

    def test_input_file_content_part_stashed(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_file", "file_id": "f_1"}],
                }
            ],
        }
        result = parse(body)
        keys = [k for k in result.raw_extras if k.startswith("unknown_block:msg:")]
        assert keys


# ---------------------------------------------------------------------------
# Render with response-side parts (ModelResponse) — exercises _dump_response_parts
# ---------------------------------------------------------------------------


class TestRenderResponseParts:
    def test_assistant_text_round_trips(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                },
            ],
        }
        result = parse(body)
        rendered = render_request(result, inbound_format=InboundFormat.OPENAI_RESPONSES)
        out = json.loads(rendered)
        kinds = [it["type"] for it in out["input"]]
        # User message + assistant message
        assert kinds.count("message") == 2

    def test_assistant_with_function_call(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "weather?"},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "get_weather",
                    "arguments": '{"city":"SF"}',
                },
            ],
        }
        result = parse(body)
        rendered = render_request(result, inbound_format=InboundFormat.OPENAI_RESPONSES)
        out = json.loads(rendered)
        fc_items = [it for it in out["input"] if it.get("type") == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["name"] == "get_weather"

    def test_reasoning_round_trips_via_stash(self, parse: Parse) -> None:
        body = {
            "model": "gpt-5",
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "Thought."}],
                    "content": [],
                    "encrypted_content": "BLOB",
                }
            ],
        }
        result = parse(body)
        rendered = render_request(result, inbound_format=InboundFormat.OPENAI_RESPONSES)
        out = json.loads(rendered)
        rs_items = [it for it in out["input"] if it.get("type") == "reasoning"]
        assert len(rs_items) == 1
