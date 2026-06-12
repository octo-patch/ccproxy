"""Tests for the OpenAI Responses buffered-output renderer.

Validates :func:`ccproxy.lightllm.graph.buffered._parts_to_openai_responses`
which serializes pydantic-ai IR parts into the Responses ``Response``
envelope JSON returned to listener clients.
"""

from __future__ import annotations

import json

from pydantic_ai.messages import TextPart, ThinkingPart, ToolCallPart

from ccproxy.lightllm.graph.buffered import _parts_to_openai_responses


class TestTextOutput:
    def test_single_text_part(self) -> None:
        out = _parts_to_openai_responses(
            parts=[TextPart(content="Hello.")],
            model="claude-sonnet-4-5",
        )
        assert out["object"] == "response"
        assert out["status"] == "completed"
        assert out["model"] == "claude-sonnet-4-5"
        assert out["output"] == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello."}],
            }
        ]

    def test_multi_text_parts_coalesce(self) -> None:
        out = _parts_to_openai_responses(
            parts=[
                TextPart(content="Hello, "),
                TextPart(content="world."),
            ],
            model="claude-sonnet-4-5",
        )
        assert len(out["output"]) == 1
        assert out["output"][0]["content"][0]["text"] == "Hello, world."

    def test_empty_text_part_drops(self) -> None:
        out = _parts_to_openai_responses(
            parts=[TextPart(content="")],
            model="claude-sonnet-4-5",
        )
        # Empty text never produces an output item
        assert out["output"] == []


class TestToolCallOutput:
    def test_tool_call_with_dict_args(self) -> None:
        out = _parts_to_openai_responses(
            parts=[
                ToolCallPart(
                    tool_name="get_weather",
                    args={"city": "SF"},
                    tool_call_id="call_1",
                )
            ],
            model="claude-sonnet-4-5",
        )
        assert len(out["output"]) == 1
        item = out["output"][0]
        assert item["type"] == "function_call"
        assert item["call_id"] == "call_1"
        assert item["name"] == "get_weather"
        assert json.loads(item["arguments"]) == {"city": "SF"}

    def test_tool_call_with_string_args(self) -> None:
        out = _parts_to_openai_responses(
            parts=[
                ToolCallPart(
                    tool_name="echo",
                    args='{"msg":"hi"}',
                    tool_call_id="call_2",
                )
            ],
            model="claude-sonnet-4-5",
        )
        item = out["output"][0]
        assert item["arguments"] == '{"msg":"hi"}'

    def test_text_then_tool_call_emits_two_items(self) -> None:
        out = _parts_to_openai_responses(
            parts=[
                TextPart(content="Calling..."),
                ToolCallPart(tool_name="ping", args={}, tool_call_id="c1"),
            ],
            model="claude-sonnet-4-5",
        )
        kinds = [item["type"] for item in out["output"]]
        assert kinds == ["message", "function_call"]


class TestReasoningOutput:
    def test_thinking_part_emits_reasoning_item(self) -> None:
        out = _parts_to_openai_responses(
            parts=[ThinkingPart(content="Thinking step.", provider_name="anthropic")],
            model="claude-sonnet-4-5",
        )
        assert len(out["output"]) == 1
        item = out["output"][0]
        assert item["type"] == "reasoning"
        assert item["content"] == [{"type": "reasoning_text", "text": "Thinking step."}]


class TestEnvelopeMetadata:
    def test_provider_response_id_used_when_set(self) -> None:
        out = _parts_to_openai_responses(
            parts=[TextPart(content="x")],
            model="m",
            provider_response_id="resp_provided_id",
        )
        assert out["id"] == "resp_provided_id"

    def test_provider_response_id_synthesized_when_missing(self) -> None:
        out = _parts_to_openai_responses(
            parts=[TextPart(content="x")],
            model="m",
        )
        assert out["id"].startswith("resp_")
        assert len(out["id"]) > len("resp_")

    def test_finish_reason_length_yields_incomplete(self) -> None:
        out = _parts_to_openai_responses(
            parts=[TextPart(content="x")],
            model="m",
            finish_reason="length",
        )
        assert out["status"] == "incomplete"

    def test_finish_reason_stop_yields_completed(self) -> None:
        out = _parts_to_openai_responses(
            parts=[TextPart(content="x")],
            model="m",
            finish_reason="stop",
        )
        assert out["status"] == "completed"
