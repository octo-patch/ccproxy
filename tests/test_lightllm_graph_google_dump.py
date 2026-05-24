"""Tests for ``ccproxy.lightllm.outbound_google.render_google``.

Validates that the capture-driven outbound renderer produces correct Google
Gemini ``generateContent`` wire bodies for the four canonical IR shapes:
single user text, multi-part system prompts, tool-call history, and image
content.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from ccproxy.lightllm.adapters.google import GoogleAdapter
from ccproxy.lightllm.parsed import ParsedRequest

Render = Callable[[ParsedRequest], bytes]


@pytest.fixture
def render() -> Render:
    return GoogleAdapter.render


def _build_parsed(
    *,
    messages: list[ModelMessage],
    request_parameters: ModelRequestParameters | None = None,
    settings: ModelSettings | None = None,
    model: str = "gemini-2.5-flash",
) -> ParsedRequest:
    return ParsedRequest(
        model=model,
        messages=messages,
        request_parameters=request_parameters or ModelRequestParameters(),
        settings=settings or ModelSettings(),
    )


class TestSingleUserMessage:
    def test_text_only(self, render: Render) -> None:
        parsed = _build_parsed(
            messages=[ModelRequest(parts=[UserPromptPart(content="Hello")])],
            settings=ModelSettings(temperature=0.7, max_tokens=128),
        )
        body = json.loads(render(parsed))
        assert body["contents"] == [
            {"role": "user", "parts": [{"text": "Hello"}]},
        ]
        # System hoisting absent (no SystemPromptPart).
        assert "systemInstruction" not in body
        # generationConfig carries camelCased generation params.
        gen = body["generationConfig"]
        assert gen["temperature"] == 0.7
        assert gen["maxOutputTokens"] == 128


class TestSystemInstruction:
    def test_single_system_prompt(self, render: Render) -> None:
        parsed = _build_parsed(
            messages=[
                ModelRequest(
                    parts=[
                        SystemPromptPart(content="Be brief."),
                        UserPromptPart(content="Hi"),
                    ]
                )
            ],
        )
        body = json.loads(render(parsed))
        assert body["systemInstruction"] == {
            "role": "user",
            "parts": [{"text": "Be brief."}],
        }
        assert body["contents"] == [
            {"role": "user", "parts": [{"text": "Hi"}]},
        ]

    def test_multi_part_system(self, render: Render) -> None:
        parsed = _build_parsed(
            messages=[
                ModelRequest(
                    parts=[
                        SystemPromptPart(content="You are an assistant."),
                        SystemPromptPart(content="Be concise."),
                        UserPromptPart(content="Q?"),
                    ]
                )
            ],
        )
        body = json.loads(render(parsed))
        # Multiple SystemPromptParts collapse into one systemInstruction
        # block carrying multiple text parts.
        assert body["systemInstruction"] == {
            "role": "user",
            "parts": [
                {"text": "You are an assistant."},
                {"text": "Be concise."},
            ],
        }


class TestToolCallHistory:
    def test_assistant_function_call_and_user_function_response(self, render: Render) -> None:
        parsed = _build_parsed(
            messages=[
                ModelRequest(parts=[UserPromptPart(content="What is 2+2?")]),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="calc",
                            args={"expr": "2+2"},
                            tool_call_id="c1",
                        )
                    ]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="calc",
                            content="4",
                            tool_call_id="c1",
                        )
                    ]
                ),
            ],
            request_parameters=ModelRequestParameters(
                function_tools=[
                    ToolDefinition(
                        name="calc",
                        description="Calculate an expression.",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {"expr": {"type": "string"}},
                        },
                    )
                ],
            ),
        )
        body = json.loads(render(parsed))

        # Assistant turn becomes role='model' with a functionCall part.
        model_turn = body["contents"][1]
        assert model_turn["role"] == "model"
        function_call_part = next(p for p in model_turn["parts"] if "functionCall" in p)
        assert function_call_part["functionCall"] == {
            "name": "calc",
            "args": {"expr": "2+2"},
            "id": "c1",
        }

        # ToolReturnPart maps to role='user' with a functionResponse part.
        user_response_turn = body["contents"][2]
        assert user_response_turn["role"] == "user"
        assert user_response_turn["parts"][0]["functionResponse"] == {
            "name": "calc",
            "response": {"return_value": "4"},
            "id": "c1",
        }

        # Tools surface at the top level with functionDeclarations.
        assert body["tools"] == [
            {
                "functionDeclarations": [
                    {
                        "name": "calc",
                        "description": "Calculate an expression.",
                        "parametersJsonSchema": {
                            "type": "object",
                            "properties": {"expr": {"type": "string"}},
                        },
                    }
                ]
            }
        ]
        # The installed pydantic-ai omits toolConfig when allow_text_output
        # is true and tool_choice is unset (default AUTO is implicit upstream).
        assert "toolConfig" not in body

    def test_required_tool_choice_emits_tool_config(self, render: Render) -> None:
        parsed = _build_parsed(
            messages=[ModelRequest(parts=[UserPromptPart(content="Use the tool.")])],
            request_parameters=ModelRequestParameters(
                function_tools=[
                    ToolDefinition(
                        name="calc",
                        description="Calc",
                        parameters_json_schema={
                            "type": "object",
                            "properties": {"x": {"type": "number"}},
                        },
                    )
                ],
                allow_text_output=False,
            ),
        )
        body = json.loads(render(parsed))
        # When allow_text_output is false, the installed pydantic-ai forces
        # ANY mode with allowed_function_names so the model must invoke a tool.
        assert body["toolConfig"] == {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": ["calc"],
            }
        }


class TestImageContent:
    def test_binary_image_maps_to_inline_data(self, render: Render) -> None:
        raw_bytes = b"\x89PNG\r\n\x1a\nfake-png-payload"
        parsed = _build_parsed(
            messages=[
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[
                                "Describe this:",
                                BinaryContent(
                                    data=raw_bytes,
                                    media_type="image/png",
                                ),
                            ]
                        )
                    ]
                )
            ],
        )
        body = json.loads(render(parsed))

        parts = body["contents"][0]["parts"]
        text_part = next(p for p in parts if "text" in p)
        inline_part = next(p for p in parts if "inlineData" in p)

        assert text_part["text"] == "Describe this:"
        # bytes get base64-encoded in the wire body; camelCased keys.
        assert inline_part["inlineData"]["mimeType"] == "image/png"
        assert inline_part["inlineData"]["data"] == base64.b64encode(raw_bytes).decode("ascii")
