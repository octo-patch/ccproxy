"""Parse an OpenAI Chat Completions request body to :class:`ParsedRequest` via FSM.

Inverse of :mod:`ccproxy.lightllm.graph.openai_dump`. Replaces the imperative
:mod:`ccproxy.lightllm.openai_inbound` parser with one polymorphic-walk FSM
(built atop :mod:`pydantic_graph.beta`'s ``GraphBuilder``) for user-role
content lists; everything else (system / developer / assistant / tool message
dispatch, two-pass ``tool_name`` resolution, settings + tools extraction,
``raw_extras`` accumulation) is imperative envelope handling.

The FSM mirrors the Anthropic-load shape: one graph run per
``UserPromptPart`` content list, decision-routed dispatch over block types,
per-block-type steps emitting :class:`UserContent` items.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic_ai.messages import (
    INVALID_JSON_KEY,
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_graph.beta import GraphBuilder, StepContext

from ccproxy.lightllm.parsed import ParsedRequest

logger = logging.getLogger(__name__)


# Wire fields absorbed into ModelSettings. Everything else lands in raw_extras.
_COMMON_SETTINGS_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "seed",
        "parallel_tool_calls",
    }
)
_OPENAI_SETTINGS_KEYS = frozenset({"logprobs", "top_logprobs"})

_ABSORBED_BODY_KEYS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "response_format",
        "stream",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "user",
        *_COMMON_SETTINGS_KEYS,
        *_OPENAI_SETTINGS_KEYS,
    }
)


# ── User-content FSM ───────────────────────────────────────────────────────


@dataclass
class _UserContentState:
    """State for one user-message content list's load FSM."""

    queue: deque[tuple[int, Any]] = field(default_factory=deque)
    items: list[UserContent] = field(default_factory=list)
    msg_index: int = 0
    raw_extras: dict[str, Any] = field(default_factory=dict)


class _UserDone:
    """Marker returned when the user-content queue is exhausted."""


@dataclass
class _UserBlock:
    """Base typed envelope for user-side block dispatch."""

    block_index: int
    block: dict[str, Any]


@dataclass
class _UserTextBlock(_UserBlock):
    pass


@dataclass
class _UserImageUrlBlock(_UserBlock):
    pass


@dataclass
class _UserInputAudioBlock(_UserBlock):
    pass


@dataclass
class _UserFileBlock(_UserBlock):
    pass


@dataclass
class _UserUnknownBlock(_UserBlock):
    pass


@dataclass
class _UserNonDictBlock:
    """A non-dict queue item (coerced to its ``str`` form)."""

    block_index: int
    raw: Any


_g: GraphBuilder[_UserContentState, None, None, list[UserContent]] = GraphBuilder(
    state_type=_UserContentState,
    output_type=list[UserContent],
)


@_g.step
async def take_next(ctx: StepContext[_UserContentState, None, None]) -> Any:
    """Router source: pop the next block and dispatch by ``type``."""
    if not ctx.state.queue:
        return _UserDone()
    block_index, raw_block = ctx.state.queue.popleft()
    if not isinstance(raw_block, dict):
        return _UserNonDictBlock(block_index=block_index, raw=raw_block)
    block: dict[str, Any] = raw_block
    block_type = block.get("type", "")
    if block_type == "text":
        return _UserTextBlock(block_index=block_index, block=block)
    if block_type == "image_url":
        return _UserImageUrlBlock(block_index=block_index, block=block)
    if block_type == "input_audio":
        return _UserInputAudioBlock(block_index=block_index, block=block)
    if block_type == "file":
        return _UserFileBlock(block_index=block_index, block=block)
    return _UserUnknownBlock(block_index=block_index, block=block)


@_g.step
async def parse_text(ctx: StepContext[_UserContentState, None, _UserTextBlock]) -> None:
    """Append a text item to the accumulator."""
    ctx.state.items.append(cast(str, ctx.inputs.block.get("text", "")))


@_g.step
async def parse_image_url(ctx: StepContext[_UserContentState, None, _UserImageUrlBlock]) -> None:
    """Append an image item — ``data:`` URIs become :class:`BinaryContent`, HTTP(S) becomes :class:`ImageUrl`."""
    payload = ctx.inputs
    image_block = payload.block.get("image_url") or {}
    url = ""
    detail: str | None = None
    if isinstance(image_block, dict):
        url = cast(str, image_block.get("url", ""))
        raw_detail = image_block.get("detail")
        if isinstance(raw_detail, str):
            detail = raw_detail
    if detail is None:
        outer_detail = payload.block.get("detail")
        if isinstance(outer_detail, str):
            detail = outer_detail
    if detail is not None:
        ctx.state.raw_extras[
            f"image_detail:msg:{ctx.state.msg_index}:block:{payload.block_index}"
        ] = detail

    if url.startswith("data:"):
        try:
            ctx.state.items.append(cast(UserContent, BinaryContent.from_data_uri(url)))
            return
        except (ValueError, binascii.Error):
            logger.warning("OpenAI load: malformed data URI; falling back to ImageUrl")
    ctx.state.items.append(ImageUrl(url=url))


@_g.step
async def parse_input_audio(
    ctx: StepContext[_UserContentState, None, _UserInputAudioBlock],
) -> None:
    """Append an :class:`BinaryContent` audio item from an ``input_audio`` block."""
    audio = ctx.inputs.block.get("input_audio") or {}
    data = ""
    audio_format = "wav"
    if isinstance(audio, dict):
        data = cast(str, audio.get("data", ""))
        audio_format = cast(str, audio.get("format", "wav"))
    try:
        data_bytes = base64.b64decode(data) if data else b""
    except (ValueError, binascii.Error):
        logger.warning("OpenAI load: malformed base64 audio payload; emitting empty bytes")
        data_bytes = b""
    ctx.state.items.append(BinaryContent(data=data_bytes, media_type=f"audio/{audio_format}"))


@_g.step
async def parse_file(ctx: StepContext[_UserContentState, None, _UserFileBlock]) -> None:
    """Stash a ``file`` block in raw_extras and emit a JSON-string placeholder."""
    payload = ctx.inputs
    ctx.state.raw_extras[
        f"file:msg:{ctx.state.msg_index}:block:{payload.block_index}"
    ] = payload.block
    ctx.state.items.append(json.dumps(payload.block))


@_g.step
async def parse_unknown(ctx: StepContext[_UserContentState, None, _UserUnknownBlock]) -> None:
    """Stash an unknown block in raw_extras and emit a JSON-string placeholder."""
    payload = ctx.inputs
    ctx.state.raw_extras[
        f"unknown_block:msg:{ctx.state.msg_index}:block:{payload.block_index}"
    ] = payload.block
    ctx.state.items.append(json.dumps(payload.block))


@_g.step
async def parse_non_dict(ctx: StepContext[_UserContentState, None, _UserNonDictBlock]) -> None:
    """Append a string-coerced form of a non-dict block to the accumulator."""
    ctx.state.items.append(str(ctx.inputs.raw))


@_g.step
async def emit_items(
    ctx: StepContext[_UserContentState, None, _UserDone],
) -> list[UserContent]:
    """Terminal step — hand the accumulated content items to the end node."""
    return ctx.state.items


_g.add(
    _g.edge_from(_g.start_node).to(take_next),
    _g.edge_from(take_next).to(
        _g.decision()
        .branch(_g.match(_UserDone).to(emit_items))
        .branch(_g.match(_UserTextBlock).to(parse_text))
        .branch(_g.match(_UserImageUrlBlock).to(parse_image_url))
        .branch(_g.match(_UserInputAudioBlock).to(parse_input_audio))
        .branch(_g.match(_UserFileBlock).to(parse_file))
        .branch(_g.match(_UserUnknownBlock).to(parse_unknown))
        .branch(_g.match(_UserNonDictBlock).to(parse_non_dict))
    ),
    _g.edge_from(
        parse_text,
        parse_image_url,
        parse_input_audio,
        parse_file,
        parse_unknown,
        parse_non_dict,
    ).to(take_next),
    _g.edge_from(emit_items).to(_g.end_node),
)


_user_content_graph = _g.build()


async def _load_user_content(
    content: Any, *, msg_index: int, raw_extras: dict[str, Any]
) -> str | list[UserContent] | None:
    """Convert a user-role wire ``content`` into IR-friendly content (drives the FSM)."""
    if isinstance(content, str):
        return content if content else None
    if not isinstance(content, list):
        return None

    state = _UserContentState(
        queue=deque(enumerate(content)),
        msg_index=msg_index,
        raw_extras=raw_extras,
    )
    items = await _user_content_graph.run(state=state)
    if not items:
        return None
    return items


# ── Per-role imperative loaders ────────────────────────────────────────────


def _build_tool_name_map(raw_messages: Sequence[Any]) -> dict[str, str]:
    """Pre-pass: build ``tool_call_id → tool_name`` from assistant ``tool_calls[]``."""
    mapping: dict[str, str] = {}
    for msg in raw_messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            function = call.get("function") or {}
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(call_id, str) and isinstance(name, str):
                mapping[call_id] = name
    return mapping


def _flatten_text_blocks(blocks: Sequence[Any]) -> str:
    """Concatenate ``text`` fields from a list of ``{type, text}`` dicts."""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _coerce_tool_content(content: Any) -> str:
    """OpenAI ``tool`` role accepts string or list of text blocks; flatten to string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _flatten_text_blocks(content)
    if content is None:
        return ""
    return str(content)


def _parse_tool_args(args_str: Any) -> dict[str, Any] | str:
    """Parse a tool-call ``arguments`` JSON string; wrap invalid JSON via ``INVALID_JSON_KEY``."""
    if isinstance(args_str, dict):
        return cast(dict[str, Any], args_str)
    if not args_str:
        return {}
    if not isinstance(args_str, str):
        return {INVALID_JSON_KEY: str(args_str)}
    try:
        parsed = json.loads(args_str)
    except (json.JSONDecodeError, ValueError):
        return {INVALID_JSON_KEY: args_str}
    if isinstance(parsed, dict):
        return cast(dict[str, Any], parsed)
    return {INVALID_JSON_KEY: args_str}


async def _load_request_message(
    msg: dict[str, Any],
    *,
    msg_index: int,
    tool_name_map: dict[str, str],
    raw_extras: dict[str, Any],
) -> ModelRequest:
    """Parse a non-assistant role message (system / developer / user / tool)."""
    role = msg.get("role", "")
    content = msg.get("content", "")
    parts: list[ModelRequestPart] = []

    if role == "tool":
        tool_call_id = cast(str, msg.get("tool_call_id", ""))
        tool_name = tool_name_map.get(tool_call_id, "")
        if tool_call_id and not tool_name:
            logger.warning(
                "OpenAI load: tool message tool_call_id=%r has no matching "
                "assistant tool_calls entry; emitting empty tool_name",
                tool_call_id,
            )
        parts.append(
            ToolReturnPart(
                tool_name=tool_name,
                content=_coerce_tool_content(content),
                tool_call_id=tool_call_id,
            )
        )
        return ModelRequest(parts=parts)

    if role in ("system", "developer"):
        if isinstance(content, str):
            if content:
                parts.append(SystemPromptPart(content=content))
        elif isinstance(content, list):
            text = _flatten_text_blocks(content)
            if text:
                parts.append(SystemPromptPart(content=text))
        return ModelRequest(parts=parts)

    # role == "user" or anything else we treat as user
    user_content = await _load_user_content(content, msg_index=msg_index, raw_extras=raw_extras)
    if user_content is not None:
        parts.append(UserPromptPart(content=user_content))
    return ModelRequest(parts=parts)


def _load_assistant_message(
    msg: dict[str, Any], *, msg_index: int, raw_extras: dict[str, Any]
) -> ModelResponse:
    """Parse an assistant-role message into a :class:`ModelResponse`."""
    parts: list[ModelResponsePart] = []
    content = msg.get("content")
    refusal = msg.get("refusal")

    if isinstance(content, str) and content:
        parts.append(TextPart(content=content))
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                parts.append(TextPart(content=str(block)))
                continue
            block_type = block.get("type", "")
            if block_type == "text":
                parts.append(TextPart(content=cast(str, block.get("text", ""))))
            elif block_type == "refusal":
                refusal_text = cast(str, block.get("refusal", ""))
                parts.append(TextPart(content=refusal_text))
                raw_extras[f"refusal:msg:{msg_index}"] = refusal_text
            else:
                parts.append(TextPart(content=json.dumps(block)))

    if isinstance(refusal, str) and refusal:
        parts.append(TextPart(content=refusal))
        raw_extras.setdefault(f"refusal:msg:{msg_index}", refusal)

    tool_calls = msg.get("tool_calls") or []
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            if not isinstance(function, dict):
                continue
            parts.append(
                ToolCallPart(
                    tool_name=cast(str, function.get("name", "")),
                    args=_parse_tool_args(function.get("arguments", "")),
                    tool_call_id=cast(str, call.get("id", "")),
                )
            )

    if "function_call" in msg:
        raw_extras[f"function_call:msg:{msg_index}"] = msg["function_call"]

    return ModelResponse(parts=parts) if parts else ModelResponse(parts=[TextPart(content="")])


# ── Tools + settings (imperative) ──────────────────────────────────────────


def _parse_tools(raw_tools: Sequence[Any]) -> list[ToolDefinition]:
    """Parse OpenAI ``tools[].function`` entries into :class:`ToolDefinition`."""
    result: list[ToolDefinition] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") or {}
        if not isinstance(function, dict):
            continue
        result.append(
            ToolDefinition(
                name=cast(str, function.get("name", "")),
                parameters_json_schema=cast(
                    dict[str, Any],
                    function.get("parameters") or {"type": "object", "properties": {}},
                ),
                description=cast("str | None", function.get("description")),
            )
        )
    return result


def _parse_settings(body: dict[str, Any]) -> ModelSettings:
    """Extract :class:`ModelSettings` from the OpenAI wire body."""
    settings: dict[str, Any] = {}

    max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, int):
        settings["max_tokens"] = max_tokens

    for key in _COMMON_SETTINGS_KEYS:
        if key in body:
            settings[key] = body[key]

    stop = body.get("stop")
    if isinstance(stop, str):
        settings["stop_sequences"] = [stop]
    elif isinstance(stop, list):
        settings["stop_sequences"] = list(stop)

    if "logprobs" in body:
        settings["openai_logprobs"] = body["logprobs"]
    if "top_logprobs" in body:
        settings["openai_top_logprobs"] = body["top_logprobs"]
    if "user" in body:
        settings["openai_user"] = body["user"]

    return cast(ModelSettings, settings)


# ── Public entrypoint ──────────────────────────────────────────────────────


async def load_openai_chat(body: dict[str, Any]) -> ParsedRequest:
    """Parse an OpenAI Chat Completions request body into the IR via the FSM."""
    model = cast(str, body.get("model", ""))
    raw_messages: list[dict[str, Any]] = cast(
        list[dict[str, Any]], body.get("messages", []) or []
    )

    tool_name_map = _build_tool_name_map(raw_messages)

    raw_extras: dict[str, Any] = {}
    messages: list[ModelMessage] = []
    for index, msg in enumerate(raw_messages):
        role = msg.get("role", "")
        if role == "assistant":
            messages.append(_load_assistant_message(msg, msg_index=index, raw_extras=raw_extras))
        else:
            messages.append(
                await _load_request_message(
                    msg,
                    msg_index=index,
                    tool_name_map=tool_name_map,
                    raw_extras=raw_extras,
                )
            )

    raw_tools = cast(list[Any], body.get("tools", []) or [])
    function_tools = _parse_tools(raw_tools)
    settings = _parse_settings(body)
    request_parameters = ModelRequestParameters(function_tools=function_tools)

    if "tool_choice" in body:
        raw_extras["tool_choice"] = body["tool_choice"]
    if "response_format" in body:
        raw_extras["response_format"] = body["response_format"]

    for key, value in body.items():
        if key in _ABSORBED_BODY_KEYS:
            continue
        if key in raw_extras:
            continue
        raw_extras[key] = value

    stream = bool(body.get("stream", False))

    return ParsedRequest(
        model=model,
        messages=messages,
        request_parameters=request_parameters,
        settings=settings,
        stream=stream,
        raw_extras=raw_extras,
    )
