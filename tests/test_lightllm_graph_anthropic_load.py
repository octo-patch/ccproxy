"""Parametrized parity tests for the Anthropic load (wire → IR) path.

Tests the new adapter-based wire → IR parsing against every semantic case
plus lossiness regressions (tool_name resolution, image media_type preservation,
non-standard TTL preservation, unknown-block preservation).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

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

from ccproxy.lightllm.adapters._envelope import parse_request
from ccproxy.lightllm.parsed import InboundFormat, ParsedRequest

Parse = Callable[[dict[str, Any]], ParsedRequest]


@pytest.fixture
def parse() -> Parse:
    def _parse(body: dict[str, Any]) -> ParsedRequest:
        return parse_request(body, inbound_format=InboundFormat.ANTHROPIC_MESSAGES)

    return _parse


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
    def test_string(self, parse: Parse) -> None:
        parsed = parse(_wrap(messages=[{"role": "user", "content": "hi"}], system="Be helpful."))
        first = parsed.messages[0]
        assert isinstance(first, ModelRequest)
        assert isinstance(first.parts[0], SystemPromptPart)
        assert first.parts[0].content == "Be helpful."

    def test_list_blocks(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_uniform_cache_control_lifts_to_settings(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_mixed_cache_control_preserves_raw_blocks(self, parse: Parse) -> None:
        raw_system = [
            {"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "uncached"},
        ]
        parsed = parse(_wrap(messages=[{"role": "user", "content": "x"}], system=raw_system))
        assert parsed.raw_extras["system"] == raw_system

    def test_empty_string_no_system_part(self, parse: Parse) -> None:
        parsed = parse(_wrap(messages=[{"role": "user", "content": "x"}], system=""))
        first = parsed.messages[0]
        assert isinstance(first, ModelRequest)
        assert not any(isinstance(p, SystemPromptPart) for p in first.parts)

    def test_no_system_field(self, parse: Parse) -> None:
        parsed = parse(_wrap(messages=[{"role": "user", "content": "x"}]))
        first = parsed.messages[0]
        assert isinstance(first, ModelRequest)
        assert not any(isinstance(p, SystemPromptPart) for p in first.parts)


# ---------------------------------------------------------------------------
# Tool parsing
# ---------------------------------------------------------------------------


class TestParseTools:
    def test_basic(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_last_tool_marker_lifts_to_settings(self, parse: Parse) -> None:
        parsed = parse(
            _wrap(
                messages=[{"role": "user", "content": "x"}],
                tools=[
                    {"name": "a", "input_schema": {}},
                    {"name": "b", "input_schema": {}, "cache_control": {"type": "ephemeral"}},
                ],
            )
        )
        settings_dict: dict[str, Any] = {**parsed.settings}
        assert settings_dict.get("anthropic_cache_tool_definitions") == "5m"
        assert "tools" not in parsed.raw_extras

    def test_last_nondeferred_marker_1h_lifts_to_settings(self, parse: Parse) -> None:
        parsed = parse(
            _wrap(
                messages=[{"role": "user", "content": "x"}],
                tools=[
                    {"name": "a", "input_schema": {}},
                    {"name": "b", "input_schema": {}, "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                    {"name": "c", "input_schema": {}, "defer_loading": True},
                ],
            )
        )
        settings_dict: dict[str, Any] = {**parsed.settings}
        assert settings_dict.get("anthropic_cache_tool_definitions") == "1h"
        assert "tools" not in parsed.raw_extras
        tools = parsed.request_parameters.function_tools
        assert [t.defer_loading for t in tools] == [False, False, True]

    def test_all_stamped_preserves_raw_tools(self, parse: Parse) -> None:
        raw_tools = [
            {"name": "a", "input_schema": {}, "cache_control": {"type": "ephemeral"}},
            {"name": "b", "input_schema": {}, "cache_control": {"type": "ephemeral"}},
        ]
        parsed = parse(_wrap(messages=[{"role": "user", "content": "x"}], tools=raw_tools))
        assert parsed.raw_extras["tools"] == raw_tools
        settings_dict: dict[str, Any] = {**parsed.settings}
        assert "anthropic_cache_tool_definitions" not in settings_dict

    def test_marker_not_on_boundary_preserves_raw_tools(self, parse: Parse) -> None:
        raw_tools = [
            {"name": "a", "input_schema": {}, "cache_control": {"type": "ephemeral"}},
            {"name": "b", "input_schema": {}},
        ]
        parsed = parse(_wrap(messages=[{"role": "user", "content": "x"}], tools=raw_tools))
        assert parsed.raw_extras["tools"] == raw_tools

    def test_marker_on_deferred_tool_preserves_raw_tools(self, parse: Parse) -> None:
        raw_tools = [
            {"name": "a", "input_schema": {}},
            {
                "name": "b",
                "input_schema": {},
                "defer_loading": True,
                "cache_control": {"type": "ephemeral"},
            },
        ]
        parsed = parse(_wrap(messages=[{"role": "user", "content": "x"}], tools=raw_tools))
        assert parsed.raw_extras["tools"] == raw_tools
        settings_dict: dict[str, Any] = {**parsed.settings}
        assert "anthropic_cache_tool_definitions" not in settings_dict

    def test_deferred_no_markers_roundtrips_defer_loading(self, parse: Parse) -> None:
        parsed = parse(
            _wrap(
                messages=[{"role": "user", "content": "x"}],
                tools=[
                    {"name": "a", "input_schema": {}},
                    {"name": "b", "input_schema": {}, "defer_loading": True},
                ],
            )
        )
        assert "tools" not in parsed.raw_extras
        tools = parsed.request_parameters.function_tools
        assert [t.defer_loading for t in tools] == [False, True]

    def test_unsupported_ttl_preserves_raw_tools(self, parse: Parse) -> None:
        raw_tools = [
            {"name": "a", "input_schema": {}, "cache_control": {"type": "ephemeral", "ttl": "24h"}},
            {"name": "b", "input_schema": {}, "cache_control": {"type": "ephemeral", "ttl": "24h"}},
        ]
        parsed = parse(_wrap(messages=[{"role": "user", "content": "x"}], tools=raw_tools))
        assert parsed.raw_extras["tools"] == raw_tools
        settings_dict: dict[str, Any] = {**parsed.settings}
        assert "anthropic_cache_tool_definitions" not in settings_dict


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class TestParseMessages:
    def test_simple_user_string(self, parse: Parse) -> None:
        parsed = parse(_wrap([{"role": "user", "content": "hello"}]))
        first = parsed.messages[0]
        assert isinstance(first, ModelRequest)
        assert isinstance(first.parts[0], UserPromptPart)
        assert first.parts[0].content == "hello"

    def test_user_content_blocks(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_cache_control_on_text_block(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_cache_control_1h_ttl(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_assistant_text(self, parse: Parse) -> None:
        parsed = parse(_wrap([{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]))
        first = parsed.messages[0]
        assert isinstance(first, ModelResponse)
        assert isinstance(first.parts[0], TextPart)
        assert first.parts[0].content == "hi"

    def test_assistant_string_content(self, parse: Parse) -> None:
        parsed = parse(_wrap([{"role": "assistant", "content": "hi"}]))
        first = parsed.messages[0]
        assert isinstance(first, ModelResponse)
        assert isinstance(first.parts[0], TextPart)
        assert first.parts[0].content == "hi"

    def test_tool_use(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_thinking(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_redacted_thinking(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_tool_result(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_system_role_message(self, parse: Parse) -> None:
        parsed = parse(_wrap([{"role": "system", "content": "You are helpful"}]))
        first = parsed.messages[0]
        assert isinstance(first, ModelRequest)
        assert isinstance(first.parts[0], SystemPromptPart)
        assert first.parts[0].content == "You are helpful"

    def test_full_conversation(self, parse: Parse) -> None:
        parsed = parse(
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
    def test_non_list_non_string_content_returns_empty_request(self, parse: Parse) -> None:
        # MessagesBuilder doesn't emit empty messages, so a non-list / non-string
        # ``content`` (here: an integer) produces zero IR messages rather than
        # an empty ModelRequest.
        parsed = parse(_wrap([{"role": "user", "content": 42}]))
        assert parsed.messages == []

    def test_image_block_base64(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_image_block_url(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_image_block_with_cache_control(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_unknown_user_block_text_includes_json(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_tool_result_with_list_content(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_tool_result_flushed_after_text(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_unknown_assistant_block_text_includes_json(self, parse: Parse) -> None:
        parsed = parse(_wrap([{"role": "assistant", "content": [{"type": "custom", "data": "x"}]}]))
        resp = parsed.messages[0]
        assert isinstance(resp, ModelResponse)
        text_part = resp.parts[0]
        assert isinstance(text_part, TextPart)
        assert "custom" in text_part.content

    def test_empty_assistant_content(self, parse: Parse) -> None:
        parsed = parse(_wrap([{"role": "assistant", "content": []}]))
        resp = parsed.messages[0]
        assert isinstance(resp, ModelResponse)
        first_part = resp.parts[0]
        assert isinstance(first_part, TextPart)
        assert first_part.content == ""

    def test_tool_result_orphan_tool_use_id_warns(self, caplog: pytest.LogCaptureFixture, parse: Parse) -> None:
        # Capture from both parsers' loggers; each emits to a different namespace
        # but the message text contains the orphan id so the assertion stays single.
        with caplog.at_level("DEBUG"):
            parsed = parse(
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
    def test_basic_sampling_fields(self, parse: Parse) -> None:
        parsed = parse(
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

    def test_metadata_preserved_in_raw_extras(self, parse: Parse) -> None:
        parsed = parse(
            _wrap(
                [{"role": "user", "content": "x"}],
                metadata={"user_id": "alice"},
            )
        )
        assert parsed.raw_extras["metadata"] == {"user_id": "alice"}

    def test_stream_flag(self, parse: Parse) -> None:
        parsed = parse(_wrap([{"role": "user", "content": "x"}], stream=True))
        assert parsed.stream is True

    def test_stream_default_false(self, parse: Parse) -> None:
        parsed = parse(_wrap([{"role": "user", "content": "x"}]))
        assert parsed.stream is False

    def test_unknown_top_level_field_preserved(self, parse: Parse) -> None:
        parsed = parse(
            _wrap(
                [{"role": "user", "content": "x"}],
                service_tier="standard_only",
            )
        )
        assert parsed.raw_extras["service_tier"] == "standard_only"

    def test_model_name(self, parse: Parse) -> None:
        parsed = parse({"model": "claude-3-5-haiku-20241022", "messages": [{"role": "user", "content": "x"}]})
        assert parsed.model == "claude-3-5-haiku-20241022"


# ---------------------------------------------------------------------------
# Lossiness regressions — these specifically test the four fixes called
# out in the refactor plan.
# ---------------------------------------------------------------------------


class TestLossinessRegressions:
    def test_tool_name_populated_from_neighboring_tool_use(self, parse: Parse) -> None:
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
        parsed = parse(body)
        tr = parsed.messages[1].parts[0]
        assert isinstance(tr, ToolReturnPart)
        assert tr.tool_name == "read_file"
        assert tr.tool_call_id == "toolu_a"

    def test_image_preserves_media_type(self, parse: Parse) -> None:
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
        parsed = parse(body)
        up = parsed.messages[0].parts[0]
        assert isinstance(up, UserPromptPart)
        assert isinstance(up.content, list)
        item = up.content[0]
        assert isinstance(item, BinaryContent)
        assert item.media_type == "image/png"

    def test_nonstandard_ttl_preserved_in_raw_extras(self, parse: Parse) -> None:
        body: dict[str, Any] = {
            "model": "claude-3-5-haiku-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral", "ttl": "24h"}}],
                }
            ],
        }
        parsed = parse(body)
        assert "cc:msg:0:block:0" in parsed.raw_extras
        cache_extra = cast(dict[str, Any], parsed.raw_extras["cc:msg:0:block:0"])
        assert cache_extra["ttl"] == "24h"
        # No CachePoint was emitted because pydantic-ai can't represent the TTL.
        up = parsed.messages[0].parts[0]
        assert isinstance(up, UserPromptPart)
        assert isinstance(up.content, list)
        assert not any(isinstance(item, CachePoint) for item in up.content)

    def test_unknown_block_preserved_in_raw_extras(self, parse: Parse) -> None:
        body: dict[str, Any] = {
            "model": "claude-3-5-haiku-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "future_block_type_2027", "data": "..."}],
                }
            ],
        }
        parsed = parse(body)
        assert "unknown_block:msg:0:idx:0" in parsed.raw_extras
        stash = cast(dict[str, Any], parsed.raw_extras["unknown_block:msg:0:idx:0"])
        assert stash["type"] == "future_block_type_2027"
        assert stash["data"] == "..."
