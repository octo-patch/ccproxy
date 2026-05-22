"""OpenAI Chat Completions UIAdapter.

Converts OpenAI Chat Completions request JSON to / from pydantic-ai's
``list[ModelMessage]`` IR. Reuses the SDK's `TypedDict`s
(``openai.types.chat.*``) for typed dispatch — the wire types are dicts
at runtime, so we read via dict syntax and use ``cast(...)`` for IDE /
type-checker support without paying a Pydantic validation tax.

Replaces the four-FSM stack in ``ccproxy.lightllm.graph.openai_load`` +
``openai_dump`` with a single procedural adapter modeled on the
pydantic-ai UI adapters in ``pydantic_ai.ui.{ag_ui,vercel_ai}``.

``build_event_stream`` raises ``NotImplementedError``; streaming
intake/render still lives in ``ccproxy.lightllm.graph.openai_*``.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_param import (
    File as ChatCompletionContentPartFileParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from pydantic_ai.messages import (
    INVALID_JSON_KEY,
    BinaryContent,
    CachePoint,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.ui import MessagesBuilder, UIAdapter, UIEventStream


@dataclass
class OpenAIChatAdapter(
    UIAdapter[CompletionCreateParamsBase, ChatCompletionMessageParam, Any, AgentDepsT, OutputDataT]
):
    """UIAdapter for the OpenAI Chat Completions wire format.

    Maps:

    * ``system`` / ``developer`` role → :class:`SystemPromptPart`
    * ``user`` role: text / ``image_url`` / ``input_audio`` / ``file``
      content parts → :mod:`pydantic_ai.messages` multimodal types
    * ``assistant`` role: ``content`` → :class:`TextPart`,
      ``tool_calls`` → :class:`ToolCallPart`
    * ``tool`` role → :class:`ToolReturnPart` (``tool_name`` recovered
      by pre-scanning assistant turns)
    """

    @classmethod
    def build_run_input(cls, body: bytes) -> CompletionCreateParamsBase:
        return cast(CompletionCreateParamsBase, json.loads(body))

    @cached_property
    def messages(self) -> list[ModelMessage]:
        return self.load_messages(self.run_input["messages"])

    # ── load (wire → IR) ─────────────────────────────────────────────────────

    @classmethod
    def load_messages(
        cls,
        messages: Iterable[ChatCompletionMessageParam],
        *,
        raw_extras: dict[str, Any] | None = None,
    ) -> list[ModelMessage]:
        """Convert an OpenAI ``messages`` array into pydantic-ai IR.

        ``tool`` role messages don't carry the tool name — we scan all
        assistant turns first to build a ``{tool_call_id: tool_name}``
        index before iterating in order.

        When ``raw_extras`` is provided, wire fields the IR doesn't model
        natively are stashed there for lossless round-trip:

        * ``image_detail:msg:N:block:M`` — ``image_url.detail`` field
        * ``file:msg:N:block:M`` — full ``file`` content block
        * ``unknown_block:msg:N:block:M`` — unrecognized user content block
        * ``refusal:msg:N`` — assistant refusal text
        * ``function_call:msg:N`` — legacy assistant ``function_call`` field
        """
        messages = list(messages)
        tool_name_by_id: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            assistant = cast(ChatCompletionAssistantMessageParam, msg)
            for tc in assistant.get("tool_calls") or []:
                if tc.get("type") == "function":
                    fn = cast(ChatCompletionMessageFunctionToolCallParam, tc)
                    tool_name_by_id[fn["id"]] = fn["function"]["name"]

        builder = MessagesBuilder()

        for msg_index, msg in enumerate(messages):
            role = msg["role"]

            if role in ("system", "developer"):
                system = cast(ChatCompletionSystemMessageParam, msg)
                s_content = system["content"]
                if isinstance(s_content, str):
                    builder.add(SystemPromptPart(content=s_content))
                else:
                    for s_part in s_content:
                        builder.add(SystemPromptPart(content=s_part["text"]))

            elif role == "user":
                user = cast(ChatCompletionUserMessageParam, msg)
                builder.add(
                    UserPromptPart(
                        content=cls._load_user_content(
                            user["content"], msg_index=msg_index, raw_extras=raw_extras
                        )
                    )
                )

            elif role == "assistant":
                assistant = cast(ChatCompletionAssistantMessageParam, msg)
                a_content = assistant.get("content")
                if isinstance(a_content, str):
                    if a_content:
                        builder.add(TextPart(content=a_content))
                elif a_content is not None:
                    for a_part in a_content:
                        a_type = a_part.get("type")
                        if a_type == "text":
                            text_part = cast(ChatCompletionContentPartTextParam, a_part)
                            builder.add(TextPart(content=text_part["text"]))
                        elif a_type == "refusal":
                            refusal_text = cast(str, a_part.get("refusal", ""))
                            builder.add(TextPart(content=refusal_text))
                            if raw_extras is not None:
                                raw_extras[f"refusal:msg:{msg_index}"] = refusal_text

                refusal = msg.get("refusal")
                if isinstance(refusal, str) and refusal:
                    builder.add(TextPart(content=refusal))
                    if raw_extras is not None:
                        raw_extras.setdefault(f"refusal:msg:{msg_index}", refusal)

                for tc in assistant.get("tool_calls") or []:
                    if tc.get("type") != "function":
                        continue
                    fn = cast(ChatCompletionMessageFunctionToolCallParam, tc)
                    builder.add(
                        ToolCallPart(
                            tool_name=fn["function"]["name"],
                            args=cls._parse_args(fn["function"]["arguments"]),
                            tool_call_id=fn["id"],
                        )
                    )

                if raw_extras is not None:
                    legacy_fn_call = cast(dict[str, Any], msg).get("function_call")
                    if legacy_fn_call is not None:
                        raw_extras[f"function_call:msg:{msg_index}"] = legacy_fn_call

            elif role == "tool":
                tool = cast(ChatCompletionToolMessageParam, msg)
                t_content = tool["content"]
                if not isinstance(t_content, str):
                    t_content = "".join(
                        p["text"] for p in t_content if p.get("type") == "text"
                    )
                builder.add(
                    ToolReturnPart(
                        tool_name=tool_name_by_id.get(tool["tool_call_id"], ""),
                        content=t_content,
                        tool_call_id=tool["tool_call_id"],
                    )
                )

        return builder.messages

    # ── dump (IR → wire) ─────────────────────────────────────────────────────

    @classmethod
    def dump_messages(cls, messages: Sequence[ModelMessage]) -> list[ChatCompletionMessageParam]:
        """Convert pydantic-ai IR back to an OpenAI ``messages`` array."""
        result: list[ChatCompletionMessageParam] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                result.extend(cls._dump_request(message))
            elif isinstance(message, ModelResponse) and (msg := cls._dump_response(message)) is not None:
                result.append(msg)
        return result

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_args(arguments: str) -> str | dict[str, Any]:
        """Parse a JSON-string tool-call ``arguments``.

        Wraps malformed JSON in ``{INVALID_JSON_KEY: raw_string}`` so pydantic-ai's
        downstream tool-call machinery surfaces it as a retryable error rather
        than silently passing a stringified blob to a tool expecting a dict.
        """
        if not arguments:
            return {}
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return {INVALID_JSON_KEY: arguments}
        if isinstance(parsed, dict):
            return parsed
        return {INVALID_JSON_KEY: arguments}

    @classmethod
    def _load_user_content(
        cls,
        content: str | Iterable[ChatCompletionContentPartParam],
        *,
        msg_index: int = 0,
        raw_extras: dict[str, Any] | None = None,
    ) -> str | list[UserContent]:
        if isinstance(content, str):
            return content

        parts: list[UserContent] = []
        for block_index, item in enumerate(content):
            part_type = item.get("type")

            if part_type == "text":
                text_item = cast(ChatCompletionContentPartTextParam, item)
                parts.append(text_item["text"])

            elif part_type == "image_url":
                img_item = cast(ChatCompletionContentPartImageParam, item)
                image_url = img_item["image_url"]
                url = image_url["url"]
                detail = image_url.get("detail")
                if raw_extras is not None and isinstance(detail, str):
                    raw_extras[f"image_detail:msg:{msg_index}:block:{block_index}"] = detail
                if url.startswith("data:"):
                    parts.append(BinaryContent.from_data_uri(url))
                else:
                    parts.append(ImageUrl(url=url))

            elif part_type == "input_audio":
                audio_item = cast(ChatCompletionContentPartInputAudioParam, item)
                audio = audio_item["input_audio"]
                raw = audio["data"]
                fmt = audio["format"]
                if raw.startswith("data:"):
                    parts.append(BinaryContent.from_data_uri(raw))
                else:
                    parts.append(BinaryContent(data=base64.b64decode(raw), media_type=f"audio/{fmt}"))

            elif part_type == "file":
                file_item = cast(ChatCompletionContentPartFileParam, item)
                if raw_extras is not None:
                    raw_extras[f"file:msg:{msg_index}:block:{block_index}"] = dict(item)
                f = file_item["file"]
                file_id = f.get("file_id")
                file_data = f.get("file_data")
                if file_id:
                    parts.append(UploadedFile(file_id=file_id, provider_name="openai"))
                elif file_data:
                    if file_data.startswith("data:"):
                        parts.append(BinaryContent.from_data_uri(file_data))
                    else:
                        media = "application/octet-stream"
                        parts.append(BinaryContent(data=base64.b64decode(file_data), media_type=media))
                else:
                    parts.append(json.dumps(dict(item)))

            else:  # type: ignore[unreachable]
                # Unknown block — preserve in raw_extras and emit a JSON-string
                # placeholder. The SDK TypedDict claims exhaustive variants;
                # runtime allows arbitrary unknown types.
                if raw_extras is not None:  # type: ignore[unreachable]
                    raw_extras[f"unknown_block:msg:{msg_index}:block:{block_index}"] = dict(item)
                parts.append(json.dumps(dict(item)))

        if len(parts) == 1 and isinstance(parts[0], str):
            return parts[0]
        return parts

    @staticmethod
    def _dump_request(
        message: ModelRequest,
    ) -> list[ChatCompletionMessageParam]:
        result: list[ChatCompletionMessageParam] = []
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                result.append({"role": "system", "content": part.content})

            elif isinstance(part, UserPromptPart):
                content = part.content
                if isinstance(content, str):
                    result.append({"role": "user", "content": content})
                else:
                    oai_parts: list[ChatCompletionContentPartParam] = []
                    for item in content:
                        if isinstance(item, str):
                            oai_parts.append({"type": "text", "text": item})
                        elif isinstance(item, BinaryContent):
                            if item.is_image:
                                oai_parts.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": item.data_uri},
                                    }
                                )
                            elif item.is_audio:
                                fmt = item.format if item.format in ("wav", "mp3") else "wav"
                                oai_parts.append(
                                    {
                                        "type": "input_audio",
                                        "input_audio": {
                                            "data": item.base64,
                                            "format": cast(Literal["wav", "mp3"], fmt),
                                        },
                                    }
                                )
                        elif isinstance(item, ImageUrl):
                            vendor = item.vendor_metadata or {}
                            image_url: dict[str, Any] = {"url": item.url}
                            if detail := vendor.get("detail"):
                                image_url["detail"] = detail
                            oai_parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": cast(Any, image_url),
                                }
                            )
                        elif isinstance(item, UploadedFile) and item.provider_name == "openai":
                            oai_parts.append(
                                {
                                    "type": "file",
                                    "file": {"file_id": item.file_id},
                                }
                            )
                        elif isinstance(item, CachePoint):
                            # OpenAI has no cache-point concept.
                            pass
                    if oai_parts:
                        result.append({"role": "user", "content": oai_parts})

            elif isinstance(part, ToolReturnPart):
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": part.tool_call_id,
                        "content": part.model_response_str(),
                    }
                )

            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    result.append({"role": "user", "content": part.model_response()})
                else:
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": part.tool_call_id,
                            "content": part.model_response(),
                        }
                    )

        return result

    @staticmethod
    def _dump_response(
        message: ModelResponse,
    ) -> ChatCompletionAssistantMessageParam | None:
        text = ""
        tool_calls: list[ChatCompletionMessageFunctionToolCallParam] = []

        for part in message.parts:
            if isinstance(part, TextPart):
                text += part.content
            elif isinstance(part, ToolCallPart):
                args = part.args
                arguments = args if isinstance(args, str) else json.dumps(args or {})
                tool_calls.append(
                    {
                        "id": part.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": part.tool_name,
                            "arguments": arguments,
                        },
                    }
                )

        if not text and not tool_calls:
            return None
        msg: ChatCompletionAssistantMessageParam = {"role": "assistant"}
        if text:
            msg["content"] = text
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    def build_event_stream(
        self,
    ) -> UIEventStream[CompletionCreateParamsBase, Any, AgentDepsT, OutputDataT]:
        raise NotImplementedError("Implement a UIEventStream subclass to produce OpenAI SSE chunks.")
