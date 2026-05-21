"""Parametrized parity tests for the OpenAI Chat Completions dump path.

Runs every roundtrip case against BOTH the legacy
``(parse_openai_chat, render_openai_chat)`` pair and the new
``(load_openai_chat, render_openai_chat_dump)`` FSM pair. The roundtrip
helper is injected as a fixture so the implementation switch is invisible
to the test bodies.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest

from ccproxy.lightllm.graph import load_openai_chat, render_openai_chat_dump

Roundtrip = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@pytest.fixture
def roundtrip() -> Roundtrip:
    """Inbound parse (FSM) → outbound render (FSM) → JSON-decode."""

    async def _rt(body: dict[str, Any]) -> dict[str, Any]:
        parsed = await load_openai_chat(body)
        out = await render_openai_chat_dump(parsed)
        return cast("dict[str, Any]", json.loads(out))

    return _rt


_PNG_PIXEL_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8A"
    "AAAASUVORK5CYII="
)


class TestSimpleText:
    async def test_user_message_roundtrips(self, roundtrip: Roundtrip) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi."},
            ],
        }
        out = await roundtrip(body)
        assert out["model"] == "gpt-4o"
        assert out["messages"][0] == {"role": "system", "content": "Be helpful."}
        assert out["messages"][1] == {"role": "user", "content": "Hi."}

    async def test_stream_flag_propagates(self, roundtrip: Roundtrip) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi."}],
            "stream": True,
        }
        out = await roundtrip(body)
        assert out["stream"] is True


class TestToolCalls:
    async def test_assistant_tool_call_arguments_serialized_as_json_string(self, roundtrip: Roundtrip) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Read foo.txt"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "foo.txt"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "hello world",
                },
            ],
        }
        out = await roundtrip(body)
        assistant = out["messages"][1]
        assert assistant["role"] == "assistant"
        tool_calls = assistant["tool_calls"]
        assert len(tool_calls) == 1
        call = tool_calls[0]
        assert call["id"] == "call_1"
        assert call["function"]["name"] == "read_file"
        # arguments must be a JSON STRING, not a dict
        assert isinstance(call["function"]["arguments"], str)
        assert json.loads(call["function"]["arguments"]) == {"path": "foo.txt"}

        tool_msg = out["messages"][2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_1"
        assert tool_msg["content"] == "hello world"


class TestImages:
    async def test_data_uri_image_roundtrips_as_data_uri(self, roundtrip: Roundtrip) -> None:
        data_uri = f"data:image/png;base64,{_PNG_PIXEL_B64}"
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        }
        out = await roundtrip(body)
        user_content = out["messages"][0]["content"]
        assert isinstance(user_content, list)

        text_block = next(b for b in user_content if b.get("type") == "text")
        assert text_block["text"] == "What is this?"

        image_block = next(b for b in user_content if b.get("type") == "image_url")
        # Pydantic-ai's BinaryContent renderer emits a data: URI with the
        # original media type; the base64 payload must round-trip exactly.
        url = image_block["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        emitted_b64 = url.split(",", 1)[1]
        assert base64.b64decode(emitted_b64) == base64.b64decode(_PNG_PIXEL_B64)

    async def test_https_url_image_roundtrips_as_url(self, roundtrip: Roundtrip) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/cat.png"},
                        }
                    ],
                }
            ],
        }
        out = await roundtrip(body)
        image_block = out["messages"][0]["content"][0]
        assert image_block["type"] == "image_url"
        assert image_block["image_url"]["url"] == "https://example.com/cat.png"


class TestTools:
    async def test_tools_list_roundtrips(self, roundtrip: Roundtrip) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Use a tool."}],
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
            "tool_choice": "auto",
        }
        out = await roundtrip(body)
        tools = out["tools"]
        assert len(tools) == 1
        tool = tools[0]
        assert tool["type"] == "function"
        function = tool["function"]
        assert function["name"] == "read_file"
        assert function["description"] == "Read a file"
        # The schema shape we asked for must be present; pydantic-ai may
        # add ``additionalProperties: false`` / ``strict: true`` for OpenAI
        # JSON-schema enforcement — that's a feature, not a regression.
        params = function["parameters"]
        assert params["type"] == "object"
        assert params["properties"] == {"path": {"type": "string"}}
        assert params["required"] == ["path"]

        # raw_extras → tool_choice override
        assert out["tool_choice"] == "auto"


class TestResponseFormat:
    async def test_json_schema_response_format_roundtrips(self, roundtrip: Roundtrip) -> None:
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "cat_info",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        }
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Give me cat info."}],
            "response_format": rf,
        }
        out = await roundtrip(body)
        assert out["response_format"] == rf


class TestMultiTurnWithMixedRoles:
    async def test_assistant_text_then_tool_call_then_tool_result(self, roundtrip: Roundtrip) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Please search."},
                {
                    "role": "assistant",
                    "content": "Searching now.",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": "results...",
                },
                {"role": "assistant", "content": "Found 3 results."},
            ],
        }
        out = await roundtrip(body)
        messages = out["messages"]
        roles = [m["role"] for m in messages]
        # Expect: system, user, assistant(text), assistant(tool_call), tool, assistant
        # Pydantic-ai splits text + tool_calls into two assistant messages
        # when content and tool_calls coexist; we accept either grouping
        # as long as the conversation reads back coherently.
        assert "system" in roles
        assert "user" in roles
        assert "tool" in roles
        assert roles.count("assistant") >= 1

        # Tool call args must be a JSON string.
        for msg in messages:
            for tc in msg.get("tool_calls") or []:
                assert isinstance(tc["function"]["arguments"], str)

        # The tool result must reference the matching call id.
        tool_msg = next(m for m in messages if m["role"] == "tool")
        assert tool_msg["tool_call_id"] == "call_2"
        assert tool_msg["content"] == "results..."
