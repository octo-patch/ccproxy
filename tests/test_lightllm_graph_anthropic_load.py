"""Parametrized parity tests for the Anthropic load (wire → IR) path.

Runs every semantic case + the four lossiness regressions (tool_name
resolution, image media_type preservation, non-standard TTL preservation,
unknown-block preservation) against BOTH
``ccproxy.lightllm.anthropic_inbound.parse_anthropic_messages`` (legacy) and
``ccproxy.lightllm.graph.anthropic_load.load_anthropic`` (FSM). At Phase H the
``implementation`` parametrize collapses to a single ``"fsm"`` param and the
legacy branch is removed.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    CachePoint,
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

from ccproxy.lightllm.graph import load_anthropic
from ccproxy.lightllm.parsed import ParsedRequest

Parse = Callable[[dict[str, Any]], Awaitable[ParsedRequest]]


@pytest.fixture
def parse() -> Parse:
    return load_anthropic

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap(messages: list[dict[str, Any]], **extras: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"model": "claude-3-5-haiku-20241022", "messages": messages}
    body.update(extras)
    return body


# ---------------------------------------------------------------------------
# System prompt parsing
# ---------------------------------------------------------------------------


class TestParseSystem:
    async def test_string(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(messages=[{"role": "user", "content": "hi"}], system="Be helpful.")
        )
        first = parsed.messages[0]
        assert isinstance(first, ModelRequest)
        assert isinstance(first.parts[0], SystemPromptPart)
        assert first.parts[0].content == "Be helpful."

    async def test_list_blocks(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                messages=[{"role": "user", "content": "x"}],
                system=[
                    {"type": "text", "text": "First"},
                    {"type": "text", "text": "Second"},
                ],
            )
        )
        first = parsed.messages[0]
        assert isinstance(first, ModelRequest)
        # System parts are prepended before UserPromptPart.
        system_parts = [p for p in first.parts if isinstance(p, SystemPromptPart)]
        assert len(system_parts) == 2
        assert system_parts[0].content == "First"
        assert system_parts[1].content == "Second"

    async def test_uniform_cache_control_lifts_to_settings(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                messages=[{"role": "user", "content": "x"}],
                system=[
                    {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
                ],
            )
        )
        settings_dict: dict[str, Any] = {**parsed.settings}
        assert settings_dict.get("anthropic_cache_instructions") == "5m"
        # No raw_extras override since the cache_control was uniform.
        assert "system" not in parsed.raw_extras

    async def test_mixed_cache_control_preserves_raw_blocks(self, parse: Parse) -> None:
        raw_system = [
            {"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "uncached"},
        ]
        parsed = await parse(_wrap(messages=[{"role": "user", "content": "x"}], system=raw_system))
        assert parsed.raw_extras["system"] == raw_system

    async def test_empty_string_no_system_part(self, parse: Parse) -> None:
        parsed = await parse(_wrap(messages=[{"role": "user", "content": "x"}], system=""))
        first = parsed.messages[0]
        assert isinstance(first, ModelRequest)
        assert not any(isinstance(p, SystemPromptPart) for p in first.parts)

    async def test_no_system_field(self, parse: Parse) -> None:
        parsed = await parse(_wrap(messages=[{"role": "user", "content": "x"}]))
        first = parsed.messages[0]
        assert isinstance(first, ModelRequest)
        assert not any(isinstance(p, SystemPromptPart) for p in first.parts)


# ---------------------------------------------------------------------------
# Tool parsing
# ---------------------------------------------------------------------------


class TestParseTools:
    async def test_basic(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                messages=[{"role": "user", "content": "x"}],
                tools=[
                    {"name": "read", "description": "Read file", "input_schema": {"type": "object"}},
                ],
            )
        )
        tools = parsed.request_parameters.function_tools
        assert len(tools) == 1
        assert tools[0].name == "read"
        assert tools[0].description == "Read file"
        assert tools[0].parameters_json_schema == {"type": "object"}

    async def test_uniform_cache_lifts_to_settings(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                messages=[{"role": "user", "content": "x"}],
                tools=[
                    {"name": "a", "input_schema": {}, "cache_control": {"type": "ephemeral"}},
                    {"name": "b", "input_schema": {}, "cache_control": {"type": "ephemeral"}},
                ],
            )
        )
        settings_dict: dict[str, Any] = {**parsed.settings}
        assert settings_dict.get("anthropic_cache_tool_definitions") == "5m"
        assert "tools" not in parsed.raw_extras

    async def test_mixed_cache_preserves_raw_tools(self, parse: Parse) -> None:
        raw_tools = [
            {"name": "a", "input_schema": {}, "cache_control": {"type": "ephemeral"}},
            {"name": "b", "input_schema": {}},
        ]
        parsed = await parse(_wrap(messages=[{"role": "user", "content": "x"}], tools=raw_tools))
        assert parsed.raw_extras["tools"] == raw_tools

    async def test_unsupported_ttl_preserves_raw_tools(self, parse: Parse) -> None:
        raw_tools = [
            {"name": "a", "input_schema": {}, "cache_control": {"type": "ephemeral", "ttl": "24h"}},
            {"name": "b", "input_schema": {}, "cache_control": {"type": "ephemeral", "ttl": "24h"}},
        ]
        parsed = await parse(_wrap(messages=[{"role": "user", "content": "x"}], tools=raw_tools))
        assert parsed.raw_extras["tools"] == raw_tools
        settings_dict: dict[str, Any] = {**parsed.settings}
        assert "anthropic_cache_tool_definitions" not in settings_dict


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class TestParseMessages:
    async def test_simple_user_string(self, parse: Parse) -> None:
        parsed = await parse(_wrap([{"role": "user", "content": "hello"}]))
        first = parsed.messages[0]
        assert isinstance(first, ModelRequest)
        assert isinstance(first.parts[0], UserPromptPart)
        assert first.parts[0].content == "hello"

    async def test_user_content_blocks(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "one"},
                            {"type": "text", "text": "two"},
                        ],
                    }
                ]
            )
        )
        up = parsed.messages[0].parts[0]
        assert isinstance(up, UserPromptPart)
        assert isinstance(up.content, list)
        assert up.content[0] == "one"
        assert up.content[1] == "two"

    async def test_cache_control_on_text_block(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}},
                            {"type": "text", "text": "plain"},
                        ],
                    }
                ]
            )
        )
        up = parsed.messages[0].parts[0]
        assert isinstance(up, UserPromptPart)
        assert isinstance(up.content, list)
        assert up.content[0] == "cached"
        assert isinstance(up.content[1], CachePoint)
        assert up.content[1].ttl == "5m"
        assert up.content[2] == "plain"

    async def test_cache_control_1h_ttl(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "x", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                        ],
                    }
                ]
            )
        )
        up = parsed.messages[0].parts[0]
        assert isinstance(up, UserPromptPart)
        assert isinstance(up.content, list)
        cp = up.content[1]
        assert isinstance(cp, CachePoint)
        assert cp.ttl == "1h"

    async def test_assistant_text(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap([{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}])
        )
        first = parsed.messages[0]
        assert isinstance(first, ModelResponse)
        assert isinstance(first.parts[0], TextPart)
        assert first.parts[0].content == "hi"

    async def test_assistant_string_content(self, parse: Parse) -> None:
        parsed = await parse(_wrap([{"role": "assistant", "content": "hi"}]))
        first = parsed.messages[0]
        assert isinstance(first, ModelResponse)
        assert isinstance(first.parts[0], TextPart)
        assert first.parts[0].content == "hi"

    async def test_tool_use(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call_1",
                                "name": "read_file",
                                "input": {"path": "/etc/example"},
                            },
                        ],
                    }
                ]
            )
        )
        tc = parsed.messages[0].parts[0]
        assert isinstance(tc, ToolCallPart)
        assert tc.tool_name == "read_file"
        assert tc.args == {"path": "/etc/example"}
        assert tc.tool_call_id == "call_1"

    async def test_thinking(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Let me think...", "signature": "sig"},
                        ],
                    }
                ]
            )
        )
        tp = parsed.messages[0].parts[0]
        assert isinstance(tp, ThinkingPart)
        assert tp.content == "Let me think..."
        assert tp.signature == "sig"

    async def test_redacted_thinking(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "redacted_thinking", "data": "encrypted"}],
                    }
                ]
            )
        )
        tp = parsed.messages[0].parts[0]
        assert isinstance(tp, ThinkingPart)
        assert tp.id == "redacted_thinking"
        assert tp.content == ""
        assert tp.signature == "encrypted"

    async def test_tool_result(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "call_1", "content": "file contents"},
                        ],
                    },
                ]
            )
        )
        tr = parsed.messages[1].parts[0]
        assert isinstance(tr, ToolReturnPart)
        assert tr.tool_call_id == "call_1"
        assert tr.content == "file contents"
        # Two-pass tool_name resolution succeeded.
        assert tr.tool_name == "read_file"

    async def test_system_role_message(self, parse: Parse) -> None:
        parsed = await parse(_wrap([{"role": "system", "content": "You are helpful"}]))
        first = parsed.messages[0]
        assert isinstance(first, ModelRequest)
        assert isinstance(first.parts[0], SystemPromptPart)
        assert first.parts[0].content == "You are helpful"

    async def test_full_conversation(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [
                    {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "hmm", "signature": "s"},
                            {"type": "text", "text": "hi"},
                            {"type": "tool_use", "id": "c1", "name": "read", "input": {}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "c1", "content": "data"},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
                ]
            )
        )
        assert len(parsed.messages) == 4
        assert isinstance(parsed.messages[0], ModelRequest)
        assert isinstance(parsed.messages[1], ModelResponse)
        assert isinstance(parsed.messages[2], ModelRequest)
        assert isinstance(parsed.messages[3], ModelResponse)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_non_list_non_string_content_returns_empty_request(self, parse: Parse) -> None:
        parsed = await parse(_wrap([{"role": "user", "content": 42}]))
        first = parsed.messages[0]
        assert isinstance(first, ModelRequest)
        assert first.parts == []

    async def test_image_block_base64(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": "aGVsbG8=",
                                },
                            }
                        ],
                    }
                ]
            )
        )
        up = parsed.messages[0].parts[0]
        assert isinstance(up, UserPromptPart)
        assert isinstance(up.content, list)
        binary = up.content[0]
        assert isinstance(binary, BinaryContent)
        assert binary.media_type == "image/jpeg"
        assert binary.data == b"hello"

    async def test_image_block_url(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": "https://example.com/x.png",
                                },
                            }
                        ],
                    }
                ]
            )
        )
        up = parsed.messages[0].parts[0]
        assert isinstance(up, UserPromptPart)
        assert isinstance(up.content, list)
        item = up.content[0]
        assert isinstance(item, ImageUrl)
        assert item.url == "https://example.com/x.png"

    async def test_image_block_with_cache_control(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "AAA=",
                                },
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ]
            )
        )
        up = parsed.messages[0].parts[0]
        assert isinstance(up, UserPromptPart)
        assert isinstance(up.content, list)
        assert isinstance(up.content[0], BinaryContent)
        assert isinstance(up.content[1], CachePoint)

    async def test_unknown_user_block_text_includes_json(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [
                    {
                        "role": "user",
                        "content": [{"type": "custom_block", "data": "something"}],
                    }
                ]
            )
        )
        up = parsed.messages[0].parts[0]
        assert isinstance(up, UserPromptPart)
        assert isinstance(up.content, list)
        # The IR carries the JSON representation so downstream sees content.
        first_item = up.content[0]
        assert isinstance(first_item, str)
        assert "custom_block" in first_item

    async def test_tool_result_with_list_content(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "c1", "name": "read", "input": {}}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "c1",
                                "content": [
                                    {"type": "text", "text": "line 1"},
                                    {"type": "text", "text": "line 2"},
                                ],
                            }
                        ],
                    },
                ]
            )
        )
        tr = parsed.messages[1].parts[0]
        assert isinstance(tr, ToolReturnPart)
        assert tr.content == "line 1\nline 2"
        assert tr.tool_name == "read"

    async def test_tool_result_flushed_after_text(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "c1", "name": "read", "input": {}}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "before"},
                            {"type": "tool_result", "tool_use_id": "c1", "content": "result"},
                        ],
                    },
                ]
            )
        )
        req = parsed.messages[1]
        assert isinstance(req, ModelRequest)
        assert len(req.parts) == 2
        assert isinstance(req.parts[0], UserPromptPart)
        assert isinstance(req.parts[1], ToolReturnPart)

    async def test_unknown_assistant_block_text_includes_json(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap([{"role": "assistant", "content": [{"type": "custom", "data": "x"}]}])
        )
        resp = parsed.messages[0]
        assert isinstance(resp, ModelResponse)
        text_part = resp.parts[0]
        assert isinstance(text_part, TextPart)
        assert "custom" in text_part.content

    async def test_empty_assistant_content(self, parse: Parse) -> None:
        parsed = await parse(_wrap([{"role": "assistant", "content": []}]))
        resp = parsed.messages[0]
        assert isinstance(resp, ModelResponse)
        first_part = resp.parts[0]
        assert isinstance(first_part, TextPart)
        assert first_part.content == ""

    async def test_tool_result_orphan_tool_use_id_warns(self, caplog: pytest.LogCaptureFixture, parse: Parse) -> None:
        # Capture from both parsers' loggers; each emits to a different namespace
        # but the message text contains the orphan id so the assertion stays single.
        with caplog.at_level("DEBUG"):
            parsed = await parse(
                _wrap(
                    [
                        {
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": "orphan", "content": "data"}],
                        }
                    ]
                )
            )
        tr = parsed.messages[0].parts[0]
        assert isinstance(tr, ToolReturnPart)
        assert tr.tool_name == ""
        assert any("orphan" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Settings + raw_extras
# ---------------------------------------------------------------------------


class TestSettings:
    async def test_basic_sampling_fields(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [{"role": "user", "content": "x"}],
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                stop_sequences=["STOP"],
            )
        )
        settings_dict: dict[str, Any] = {**parsed.settings}
        assert settings_dict["max_tokens"] == 512
        assert settings_dict["temperature"] == 0.7
        assert settings_dict["top_p"] == 0.9
        assert settings_dict["top_k"] == 40
        assert settings_dict["stop_sequences"] == ["STOP"]

    async def test_metadata_preserved_in_raw_extras(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [{"role": "user", "content": "x"}],
                metadata={"user_id": "alice"},
            )
        )
        assert parsed.raw_extras["metadata"] == {"user_id": "alice"}

    async def test_stream_flag(self, parse: Parse) -> None:
        parsed = await parse(_wrap([{"role": "user", "content": "x"}], stream=True))
        assert parsed.stream is True

    async def test_stream_default_false(self, parse: Parse) -> None:
        parsed = await parse(_wrap([{"role": "user", "content": "x"}]))
        assert parsed.stream is False

    async def test_unknown_top_level_field_preserved(self, parse: Parse) -> None:
        parsed = await parse(
            _wrap(
                [{"role": "user", "content": "x"}],
                service_tier="standard_only",
            )
        )
        assert parsed.raw_extras["service_tier"] == "standard_only"

    async def test_model_name(self, parse: Parse) -> None:
        parsed = await parse(
            {"model": "claude-3-5-haiku-20241022", "messages": [{"role": "user", "content": "x"}]}
        )
        assert parsed.model == "claude-3-5-haiku-20241022"


# ---------------------------------------------------------------------------
# Lossiness regressions — these specifically test the four fixes called
# out in the refactor plan.
# ---------------------------------------------------------------------------


class TestLossinessRegressions:
    async def test_tool_name_populated_from_neighboring_tool_use(self, parse: Parse) -> None:
        body: dict[str, Any] = {
            "model": "claude-3-5-haiku-20241022",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_a",
                            "name": "read_file",
                            "input": {"path": "foo.txt"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_a", "content": "file contents"}],
                },
            ],
        }
        parsed = await parse(body)
        tr = parsed.messages[1].parts[0]
        assert isinstance(tr, ToolReturnPart)
        assert tr.tool_name == "read_file"
        assert tr.tool_call_id == "toolu_a"

    async def test_image_preserves_media_type(self, parse: Parse) -> None:
        body: dict[str, Any] = {
            "model": "claude-3-5-haiku-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KG",
                            },
                        }
                    ],
                }
            ],
        }
        parsed = await parse(body)
        up = parsed.messages[0].parts[0]
        assert isinstance(up, UserPromptPart)
        assert isinstance(up.content, list)
        item = up.content[0]
        assert isinstance(item, BinaryContent)
        assert item.media_type == "image/png"

    async def test_nonstandard_ttl_preserved_in_raw_extras(self, parse: Parse) -> None:
        body: dict[str, Any] = {
            "model": "claude-3-5-haiku-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral", "ttl": "24h"}}],
                }
            ],
        }
        parsed = await parse(body)
        assert "cc:msg:0:block:0" in parsed.raw_extras
        assert parsed.raw_extras["cc:msg:0:block:0"]["ttl"] == "24h"
        # No CachePoint was emitted because pydantic-ai can't represent the TTL.
        up = parsed.messages[0].parts[0]
        assert isinstance(up, UserPromptPart)
        assert isinstance(up.content, list)
        assert not any(isinstance(item, CachePoint) for item in up.content)

    async def test_unknown_block_preserved_in_raw_extras(self, parse: Parse) -> None:
        body: dict[str, Any] = {
            "model": "claude-3-5-haiku-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "future_block_type_2027", "data": "..."}],
                }
            ],
        }
        parsed = await parse(body)
        assert "unknown_block:msg:0:idx:0" in parsed.raw_extras
        stash = parsed.raw_extras["unknown_block:msg:0:idx:0"]
        assert stash["type"] == "future_block_type_2027"
        assert stash["data"] == "..."
