"""Render a :class:`ParsedRequest` to OpenAI Chat Completions wire bytes via FSM.

Replaces the ``_CaptureOpenAIClient`` + ``OpenAIChatModel`` instantiation hack
in :mod:`ccproxy.lightllm.outbound_openai`. One :class:`_UserContentState`
graph run per :class:`UserPromptPart` with a list content (the only place a
polymorphic-walk FSM is genuinely useful on the OpenAI side); the imperative
wrapper :func:`render_openai_chat_dump` walks the IR conversation, assembles
typed ``ChatCompletionMessageParam`` dicts via the per-part / per-message
helpers, and stitches the static envelope (model, settings, tools,
tool_choice, response_format, ``raw_extras``).

Wire dicts use the SDK TypedDicts from ``openai.types.chat`` as the typed
boundary — no hand-rolled mirror models.
"""

from __future__ import annotations

import base64
import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    DocumentUrl,
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
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from ccproxy.lightllm.parsed import ParsedRequest

# ── User-content FSM ───────────────────────────────────────────────────────


@dataclass
class _UserContentState:
    """State for walking one :class:`UserPromptPart`'s content list."""

    queue: deque[Any] = field(default_factory=deque)
    parts: list[ChatCompletionContentPartParam] = field(default_factory=list)


@dataclass
class FetchNextUserContentNode(
    BaseNode[_UserContentState, None, list[ChatCompletionContentPartParam]]
):
    """Router for one user-content-list item — dispatches by IR type via ``match``."""

    async def run(
        self, ctx: GraphRunContext[_UserContentState, None]
    ) -> (
        BaseNode[_UserContentState, None, Any]
        | End[list[ChatCompletionContentPartParam]]
    ):
        if not ctx.state.queue:
            return End(ctx.state.parts)

        item = ctx.state.queue.popleft()

        match item:
            case str():
                return ParseUserTextItemNode(text=item)
            case BinaryContent():
                return ParseUserBinaryItemNode(item=item)
            case ImageUrl():
                return ParseUserImageUrlItemNode(item=item)
            case UploadedFile():
                return ParseUserUploadedFileItemNode(item=item)
            case CachePoint() | AudioUrl() | DocumentUrl():
                # OpenAI has no cache concept; no top-level audio URL / doc URL
                # content parts on the Chat Completions wire.
                return FetchNextUserContentNode()
            case _:
                return FetchNextUserContentNode()


@dataclass
class ParseUserTextItemNode(BaseNode[_UserContentState, None]):
    """Emit a text content part."""

    text: str

    async def run(
        self, ctx: GraphRunContext[_UserContentState, None]
    ) -> BaseNode[_UserContentState, None, Any]:
        ctx.state.parts.append(cast(ChatCompletionContentPartTextParam, {"type": "text", "text": self.text}))
        return FetchNextUserContentNode()


@dataclass
class ParseUserBinaryItemNode(BaseNode[_UserContentState, None]):
    """Emit an image_url (image bytes → data URI) or input_audio content part.

    Documents / other media have no OpenAI Chat Completions equivalent and
    are dropped.
    """

    item: BinaryContent

    async def run(
        self, ctx: GraphRunContext[_UserContentState, None]
    ) -> BaseNode[_UserContentState, None, Any]:
        media_type = self.item.media_type
        if media_type.startswith("image/"):
            data_uri = f"data:{media_type};base64,{base64.b64encode(self.item.data).decode('ascii')}"
            ctx.state.parts.append(
                cast(
                    ChatCompletionContentPartImageParam,
                    {"type": "image_url", "image_url": {"url": data_uri}},
                )
            )
        elif media_type.startswith("audio/"):
            audio_format = media_type.split("/", 1)[1]
            if audio_format not in ("wav", "mp3"):
                audio_format = "wav"
            ctx.state.parts.append(
                cast(
                    ChatCompletionContentPartInputAudioParam,
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(self.item.data).decode("ascii"),
                            "format": cast(Literal["wav", "mp3"], audio_format),
                        },
                    },
                )
            )
        return FetchNextUserContentNode()


@dataclass
class ParseUserImageUrlItemNode(BaseNode[_UserContentState, None]):
    """Emit an image_url content part from an :class:`ImageUrl` (with optional detail)."""

    item: ImageUrl

    async def run(
        self, ctx: GraphRunContext[_UserContentState, None]
    ) -> BaseNode[_UserContentState, None, Any]:
        vendor = self.item.vendor_metadata or {}
        image_url: dict[str, Any] = {"url": self.item.url}
        if detail := vendor.get("detail"):
            image_url["detail"] = detail
        ctx.state.parts.append(
            cast(
                ChatCompletionContentPartImageParam,
                {"type": "image_url", "image_url": cast(Any, image_url)},
            )
        )
        return FetchNextUserContentNode()


@dataclass
class ParseUserUploadedFileItemNode(BaseNode[_UserContentState, None]):
    """Emit a ``file`` content part from an OpenAI-provider :class:`UploadedFile`."""

    item: UploadedFile

    async def run(
        self, ctx: GraphRunContext[_UserContentState, None]
    ) -> BaseNode[_UserContentState, None, Any]:
        if self.item.provider_name != "openai":
            return FetchNextUserContentNode()
        ctx.state.parts.append(
            cast(
                ChatCompletionContentPartParam,
                {"type": "file", "file": {"file_id": self.item.file_id}},
            )
        )
        return FetchNextUserContentNode()


_user_content_graph = Graph[_UserContentState, None, list[ChatCompletionContentPartParam]](
    nodes=(
        FetchNextUserContentNode,
        ParseUserTextItemNode,
        ParseUserBinaryItemNode,
        ParseUserImageUrlItemNode,
        ParseUserUploadedFileItemNode,
    ),
)


async def _render_user_content(
    content: Any,
) -> str | list[ChatCompletionContentPartParam]:
    """Convert a :class:`UserPromptPart` content list to OpenAI content parts.

    A bare string passes through. A single-item string list collapses back to
    a bare string (matches pydantic-ai's emission convention).
    """
    if isinstance(content, str):
        return content
    state = _UserContentState(queue=deque(content))
    result = await _user_content_graph.run(FetchNextUserContentNode(), state=state)
    parts = result.output
    if len(parts) == 1 and parts[0].get("type") == "text":
        text_part = cast(ChatCompletionContentPartTextParam, parts[0])
        return text_part["text"]
    return parts


# ── Per-message imperative renderers ───────────────────────────────────────


def _format_tool_call(part: ToolCallPart) -> ChatCompletionMessageFunctionToolCallParam:
    """Emit one ``tool_calls[]`` entry — ``arguments`` is a JSON string per OpenAI."""
    args = part.args
    arguments = args if isinstance(args, str) else json.dumps(args or {})
    return {
        "id": part.tool_call_id,
        "type": "function",
        "function": {"name": part.tool_name, "arguments": arguments},
    }


async def _render_request_messages(msg: ModelRequest) -> list[ChatCompletionMessageParam]:
    """Walk a :class:`ModelRequest`'s parts → list of OpenAI message dicts."""
    out: list[ChatCompletionMessageParam] = []
    for part in msg.parts:
        if isinstance(part, SystemPromptPart):
            out.append({"role": "system", "content": part.content})
        elif isinstance(part, UserPromptPart):
            content = await _render_user_content(part.content)
            out.append(cast(ChatCompletionUserMessageParam, {"role": "user", "content": content}))
        elif isinstance(part, ToolReturnPart):
            out.append(
                cast(
                    ChatCompletionToolMessageParam,
                    {
                        "role": "tool",
                        "tool_call_id": part.tool_call_id,
                        "content": part.model_response_str(),
                    },
                )
            )
        elif isinstance(part, RetryPromptPart):
            if part.tool_name is None:
                out.append({"role": "user", "content": part.model_response()})
            else:
                out.append(
                    cast(
                        ChatCompletionToolMessageParam,
                        {
                            "role": "tool",
                            "tool_call_id": part.tool_call_id,
                            "content": part.model_response(),
                        },
                    )
                )
    return out


def _render_response_message(msg: ModelResponse) -> ChatCompletionAssistantMessageParam | None:
    """Aggregate a :class:`ModelResponse`'s parts into one assistant message dict.

    Multiple :class:`TextPart` are concatenated. :class:`ToolCallPart` entries
    are collected into ``tool_calls[]``. Returns ``None`` if the response has
    neither text nor tool calls (skip emitting an empty message).
    """
    text = ""
    tool_calls: list[ChatCompletionMessageFunctionToolCallParam] = []
    for part in msg.parts:
        if isinstance(part, TextPart):
            text += part.content
        elif isinstance(part, ToolCallPart):
            tool_calls.append(_format_tool_call(part))
        # ThinkingPart, NativeToolCallPart/ReturnPart — no OpenAI Chat equivalent.

    if not text and not tool_calls:
        return None
    out: ChatCompletionAssistantMessageParam = {"role": "assistant"}
    if text:
        out["content"] = text
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


# ── Envelope helpers ───────────────────────────────────────────────────────


def _format_tools(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    """Format :class:`ToolDefinition` entries into OpenAI ``tools[]`` dicts."""
    out: list[dict[str, Any]] = []
    for tool in tools:
        function: dict[str, Any] = {
            "name": tool.name,
            "parameters": tool.parameters_json_schema or {"type": "object", "properties": {}},
        }
        if tool.description:
            function["description"] = tool.description
        out.append({"type": "function", "function": function})
    return out


# OpenAI wire field name → ``ModelSettings`` key (when they differ).
_SETTINGS_TO_WIRE: tuple[tuple[str, str], ...] = (
    ("max_tokens", "max_tokens"),
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("presence_penalty", "presence_penalty"),
    ("frequency_penalty", "frequency_penalty"),
    ("logit_bias", "logit_bias"),
    ("seed", "seed"),
    ("parallel_tool_calls", "parallel_tool_calls"),
    ("openai_logprobs", "logprobs"),
    ("openai_top_logprobs", "top_logprobs"),
    ("openai_user", "user"),
)


def _apply_settings(body: dict[str, Any], settings: dict[str, Any]) -> None:
    """Copy IR settings onto the wire body, mapping renamed keys back."""
    for ir_key, wire_key in _SETTINGS_TO_WIRE:
        if ir_key in settings:
            body[wire_key] = settings[ir_key]
    stop = settings.get("stop_sequences")
    if isinstance(stop, list):
        body["stop"] = list(stop) if len(stop) > 1 else stop[0]


# Wire fields the FSM + envelope wrapper own.
_IR_OWNED_TOP_LEVEL: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "response_format",
        "stream",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "seed",
        "parallel_tool_calls",
        "logprobs",
        "top_logprobs",
        "stop",
        "user",
    }
)

# Keys our inbound parser stashes as IR-internal markers — do NOT re-inject
# these as top-level wire fields.
_INTERNAL_RAW_EXTRA_PREFIXES = (
    "cc:",
    "unknown_block:",
    "refusal:",
    "file:",
    "image_detail:",
    "function_call:",
)


def _stitch_raw_extras(body: dict[str, Any], parsed: ParsedRequest) -> None:
    """Re-inject non-IR-internal ``raw_extras`` onto the rendered body.

    * ``tool_choice`` / ``response_format`` overrides win (the inbound parser
      preserves them as raw_extras when the IR couldn't fold them).
    * IR-internal markers are skipped.
    * Other keys are copied verbatim if not already on the body.
    """
    for key in ("tool_choice", "response_format"):
        if key in parsed.raw_extras:
            body[key] = parsed.raw_extras[key]

    for key, value in parsed.raw_extras.items():
        if key in ("tool_choice", "response_format"):
            continue
        if key.startswith(_INTERNAL_RAW_EXTRA_PREFIXES):
            continue
        body.setdefault(key, value)


# ── Public entrypoint ──────────────────────────────────────────────────────


async def render_openai_chat_dump(parsed: ParsedRequest) -> bytes:
    """Render a :class:`ParsedRequest` to OpenAI Chat Completions wire bytes.

    Walks the IR conversation imperatively (per-part dispatch); drives the
    per-:class:`UserPromptPart` content-walk FSM for polymorphic user content;
    assembles the static envelope (model, settings, tools, ``raw_extras``).
    """
    messages: list[ChatCompletionMessageParam] = []
    for msg in parsed.messages:
        if isinstance(msg, ModelRequest):
            messages.extend(await _render_request_messages(msg))
        elif isinstance(msg, ModelResponse):
            if (assistant := _render_response_message(msg)) is not None:
                messages.append(assistant)

    settings_dict = cast(dict[str, Any], parsed.settings)
    body: dict[str, Any] = {
        "model": parsed.model,
        "messages": messages,
    }
    _apply_settings(body, settings_dict)

    tools = _format_tools(parsed.request_parameters.function_tools)
    if tools:
        body["tools"] = tools

    _stitch_raw_extras(body, parsed)

    if parsed.stream:
        body["stream"] = True

    return json.dumps(body, separators=(",", ":")).encode()
