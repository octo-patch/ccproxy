"""Render a :class:`ParsedRequest` to OpenAI Chat Completions wire bytes via FSM.

Replaces the ``_CaptureOpenAIClient`` + ``OpenAIChatModel`` instantiation hack
in :mod:`ccproxy.lightllm.outbound_openai`. One :class:`_UserContentState`
graph run per :class:`UserPromptPart` with a list content (the only place a
polymorphic-walk FSM is genuinely useful on the OpenAI side); the imperative
wrapper :func:`render_openai_chat_dump` walks the IR conversation, assembles
typed ``ChatCompletionMessageParam`` dicts via the per-part / per-message
helpers, and stitches the static envelope (model, settings, tools,
tool_choice, response_format, ``raw_extras``).

The FSM is built atop :mod:`pydantic_graph.beta`'s ``GraphBuilder``. Wire
dicts use the SDK TypedDicts from ``openai.types.chat`` as the typed boundary
— no hand-rolled mirror models.
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
from pydantic_graph.beta import GraphBuilder, StepContext

from ccproxy.lightllm.parsed import ParsedRequest

# ── User-content FSM ───────────────────────────────────────────────────────


@dataclass
class _UserContentState:
    """State for walking one :class:`UserPromptPart`'s content list."""

    queue: deque[Any] = field(default_factory=deque)
    parts: list[ChatCompletionContentPartParam] = field(default_factory=list)


class _OpenAIDone:
    """Marker returned when the user-content queue is exhausted."""


class _OpenAISkip:
    """Marker for queue items with no OpenAI Chat Completions content equivalent."""


_g: GraphBuilder[
    _UserContentState, None, None, list[ChatCompletionContentPartParam]
] = GraphBuilder(
    state_type=_UserContentState,
    output_type=list[ChatCompletionContentPartParam],
)


@_g.step
async def take_next(ctx: StepContext[_UserContentState, None, None]) -> Any:
    """Router source: pop the next user-content item or signal end via :class:`_OpenAIDone`."""
    if not ctx.state.queue:
        return _OpenAIDone()
    item = ctx.state.queue.popleft()
    if isinstance(item, (str, BinaryContent, ImageUrl, UploadedFile)):
        return item
    # CachePoint, AudioUrl, DocumentUrl — no OpenAI content equivalent.
    if isinstance(item, (CachePoint, AudioUrl, DocumentUrl)):
        return _OpenAISkip()
    return _OpenAISkip()


@_g.step
async def parse_text_item(ctx: StepContext[_UserContentState, None, str]) -> None:
    """Emit a text content part."""
    ctx.state.parts.append(
        cast(ChatCompletionContentPartTextParam, {"type": "text", "text": ctx.inputs})
    )


@_g.step
async def parse_binary_item(ctx: StepContext[_UserContentState, None, BinaryContent]) -> None:
    """Emit an image_url (image bytes → data URI) or input_audio content part."""
    item = ctx.inputs
    media_type = item.media_type
    if media_type.startswith("image/"):
        data_uri = f"data:{media_type};base64,{base64.b64encode(item.data).decode('ascii')}"
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
                        "data": base64.b64encode(item.data).decode("ascii"),
                        "format": cast(Literal["wav", "mp3"], audio_format),
                    },
                },
            )
        )


@_g.step
async def parse_image_url_item(ctx: StepContext[_UserContentState, None, ImageUrl]) -> None:
    """Emit an image_url content part from an :class:`ImageUrl` (with optional detail)."""
    item = ctx.inputs
    vendor = item.vendor_metadata or {}
    image_url: dict[str, Any] = {"url": item.url}
    if detail := vendor.get("detail"):
        image_url["detail"] = detail
    ctx.state.parts.append(
        cast(
            ChatCompletionContentPartImageParam,
            {"type": "image_url", "image_url": cast(Any, image_url)},
        )
    )


@_g.step
async def parse_uploaded_file_item(
    ctx: StepContext[_UserContentState, None, UploadedFile],
) -> None:
    """Emit a ``file`` content part from an OpenAI-provider :class:`UploadedFile`."""
    item = ctx.inputs
    if item.provider_name != "openai":
        return
    ctx.state.parts.append(
        cast(
            ChatCompletionContentPartParam,
            {"type": "file", "file": {"file_id": item.file_id}},
        )
    )


@_g.step
async def skip_item(ctx: StepContext[_UserContentState, None, _OpenAISkip]) -> None:
    """No-op for queue items with no OpenAI Chat Completions equivalent."""
    del ctx  # protocol-required parameter; intentionally unused


@_g.step
async def emit_parts(
    ctx: StepContext[_UserContentState, None, _OpenAIDone],
) -> list[ChatCompletionContentPartParam]:
    """Terminal step — hand the accumulated content parts to the end node."""
    return ctx.state.parts


_g.add(
    _g.edge_from(_g.start_node).to(take_next),
    _g.edge_from(take_next).to(
        _g.decision()
        .branch(_g.match(_OpenAIDone).to(emit_parts))
        .branch(_g.match(_OpenAISkip).to(skip_item))
        .branch(_g.match(str).to(parse_text_item))
        .branch(_g.match(BinaryContent).to(parse_binary_item))
        .branch(_g.match(ImageUrl).to(parse_image_url_item))
        .branch(_g.match(UploadedFile).to(parse_uploaded_file_item))
    ),
    _g.edge_from(
        parse_text_item,
        parse_binary_item,
        parse_image_url_item,
        parse_uploaded_file_item,
        skip_item,
    ).to(take_next),
    _g.edge_from(emit_parts).to(_g.end_node),
)


_user_content_graph = _g.build()


async def _render_user_content(
    content: Any,
) -> str | list[ChatCompletionContentPartParam]:
    """Convert a :class:`UserPromptPart` content list to OpenAI content parts."""
    if isinstance(content, str):
        return content
    state = _UserContentState(queue=deque(content))
    parts = await _user_content_graph.run(state=state)
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
    """Aggregate a :class:`ModelResponse`'s parts into one assistant message dict."""
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
    """Re-inject non-IR-internal ``raw_extras`` onto the rendered body."""
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
    """Render a :class:`ParsedRequest` to OpenAI Chat Completions wire bytes."""
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
