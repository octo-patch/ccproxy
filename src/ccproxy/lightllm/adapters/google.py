"""Google Gemini generateContent UIAdapter (outbound-only).

Converts pydantic-ai's ``list[ModelMessage]`` IR to Google Gemini
``generateContent`` wire bytes. This is an OUTBOUND-ONLY adapter — ccproxy
doesn't accept Gemini-format inbound requests, so :meth:`load_messages`
raises :class:`NotImplementedError`.

Replaces the CaptureSentinel-based ``ccproxy.lightllm.graph.google_dump`` with
direct construction of the Google API wire body (camelCase keys, base64-encoded
inline data, generationConfig hoist for sampling parameters).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, cast

from pydantic.alias_generators import to_camel
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.ui import UIAdapter, UIEventStream


@dataclass
class GoogleAdapter(UIAdapter[Any, dict[str, Any], Any, AgentDepsT, OutputDataT]):
    """Outbound-only UIAdapter for Google Gemini ``generateContent``.

    :meth:`load_messages` raises :class:`NotImplementedError` because ccproxy
    does not host a Google-format listener. :meth:`render` builds the
    full Gemini wire body (system instruction, contents, tools,
    generationConfig, raw_extras) from any
    :class:`~ccproxy.lightllm.adapters.LLMRenderInput`-shaped input
    (typically a :class:`~ccproxy.pipeline.context.Context`).

    :meth:`build_event_stream` raises :class:`NotImplementedError`;
    streaming intake/render lives in :mod:`ccproxy.lightllm.graph.google_*`.
    """

    @classmethod
    def load_messages(cls, *_args: Any, **_kwargs: Any) -> list[ModelMessage]:
        raise NotImplementedError(
            "ccproxy does not host a Google-format listener; "
            "GoogleAdapter is outbound-only."
        )

    def build_event_stream(
        self,
    ) -> UIEventStream[Any, Any, AgentDepsT, OutputDataT]:
        raise NotImplementedError(
            "Google streaming intake/render lives in ccproxy.lightllm.graph.google_*."
        )

    @classmethod
    def render(cls, req: Any) -> bytes:
        """Render an :class:`LLMRenderInput` (typically a Context) to Google ``generateContent`` wire bytes."""
        body: dict[str, Any] = {}

        # Extract system instruction from messages
        system_parts: list[dict[str, Any]] = []
        content_messages: list[ModelMessage] = []

        for msg in req.messages:
            if isinstance(msg, ModelRequest):
                has_system = any(isinstance(p, SystemPromptPart) for p in msg.parts)
                if has_system:
                    user_parts = []
                    for part in msg.parts:
                        if isinstance(part, SystemPromptPart):
                            system_parts.append({"text": part.content})
                        else:
                            user_parts.append(part)
                    if user_parts:
                        content_messages.append(ModelRequest(parts=user_parts))
                else:
                    content_messages.append(msg)
            else:
                content_messages.append(msg)

        if system_parts:
            body["systemInstruction"] = {"role": "user", "parts": system_parts}

        # Build contents array
        contents: list[dict[str, Any]] = []
        for msg in content_messages:
            if isinstance(msg, ModelRequest):
                parts: list[dict[str, Any]] = []
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        if isinstance(part.content, str):
                            parts.append({"text": part.content})
                        elif isinstance(part.content, list):
                            for item in part.content:
                                if isinstance(item, str):
                                    parts.append({"text": item})
                                elif isinstance(item, BinaryContent):
                                    parts.append(
                                        {
                                            "inlineData": {
                                                "mimeType": item.media_type,
                                                "data": base64.b64encode(item.data).decode("ascii"),
                                            }
                                        }
                                    )
                                elif isinstance(item, ImageUrl):
                                    parts.append(
                                        {
                                            "fileData": {
                                                "fileUri": str(item.url),
                                                "mimeType": item.media_type or "image/jpeg",
                                            }
                                        }
                                    )
                                elif isinstance(item, DocumentUrl):
                                    parts.append(
                                        {
                                            "fileData": {
                                                "fileUri": str(item.url),
                                                "mimeType": item.media_type or "application/pdf",
                                            }
                                        }
                                    )
                                elif isinstance(item, VideoUrl):
                                    parts.append(
                                        {
                                            "fileData": {
                                                "fileUri": str(item.url),
                                                "mimeType": item.media_type or "video/mp4",
                                            }
                                        }
                                    )
                                elif isinstance(item, AudioUrl):
                                    parts.append(
                                        {
                                            "fileData": {
                                                "fileUri": str(item.url),
                                                "mimeType": item.media_type or "audio/mpeg",
                                            }
                                        }
                                    )
                                elif isinstance(item, UploadedFile):
                                    parts.append(
                                        {
                                            "fileData": {
                                                "fileUri": item.file_id,
                                                "mimeType": item.media_type or "application/octet-stream",
                                            }
                                        }
                                    )
                    elif isinstance(part, ToolReturnPart):
                        parts.append(
                            {
                                "functionResponse": {
                                    "name": part.tool_name,
                                    "response": {"return_value": part.content},
                                    "id": part.tool_call_id,
                                }
                            }
                        )
                if parts:
                    contents.append({"role": "user", "parts": parts})

            elif isinstance(msg, ModelResponse):
                parts = []
                for resp_part in msg.parts:
                    # Response parts: TextPart, ThinkingPart, ToolCallPart, etc.
                    if isinstance(resp_part, (TextPart, ThinkingPart)):
                        parts.append({"text": resp_part.content})
                    elif isinstance(resp_part, ToolCallPart):
                        parts.append(
                            {
                                "functionCall": {
                                    "name": resp_part.tool_name,
                                    "args": resp_part.args,
                                    "id": resp_part.tool_call_id,
                                }
                            }
                        )
                if parts:
                    contents.append({"role": "model", "parts": parts})

        if contents:
            body["contents"] = contents

        # Build tools section
        if req.request_parameters.function_tools:
            function_declarations: list[dict[str, Any]] = []
            for tool in req.request_parameters.function_tools:
                decl: dict[str, Any] = {
                    "name": tool.name,
                    "description": tool.description or "",
                }
                if tool.parameters_json_schema:
                    decl["parametersJsonSchema"] = tool.parameters_json_schema
                function_declarations.append(decl)

            body["tools"] = [{"functionDeclarations": function_declarations}]

            if not req.request_parameters.allow_text_output:
                body["toolConfig"] = {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [t.name for t in req.request_parameters.function_tools],
                    }
                }

        # Build generationConfig from settings
        settings_dict = cast(dict[str, Any], req.settings)
        generation_config: dict[str, Any] = {}

        if "temperature" in settings_dict:
            generation_config["temperature"] = settings_dict["temperature"]
        if "top_p" in settings_dict:
            generation_config["topP"] = settings_dict["top_p"]
        if "top_k" in settings_dict:
            generation_config["topK"] = settings_dict["top_k"]
        if "max_tokens" in settings_dict:
            generation_config["maxOutputTokens"] = settings_dict["max_tokens"]
        if "stop_sequences" in settings_dict:
            generation_config["stopSequences"] = settings_dict["stop_sequences"]

        if "google_thinking_config" in settings_dict:
            thinking_cfg = settings_dict["google_thinking_config"]
            if thinking_cfg:
                generation_config["thinkingConfig"] = _camelize(thinking_cfg)

        if generation_config:
            body["generationConfig"] = generation_config

        if "google_cached_content" in settings_dict:
            cached = settings_dict["google_cached_content"]
            if cached:
                body["cachedContent"] = cached

        if "google_safety_settings" in settings_dict:
            safety = settings_dict["google_safety_settings"]
            if safety:
                body["safetySettings"] = _camelize(safety)

        for key, value in req.raw_extras.items():
            if key not in body and value is not None:
                camel_key = to_camel(key)
                body[camel_key] = _camelize(value)

        return json.dumps(body, separators=(",", ":")).encode()


def _camelize(value: Any) -> Any:
    """Recursively convert dict keys to camelCase and encode ``bytes`` as base64."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for k, v in value.items():
            result[to_camel(k)] = _camelize(v)
        return result
    if isinstance(value, list):
        return [_camelize(item) for item in value]
    if isinstance(value, tuple):
        return [_camelize(item) for item in value]
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return value
