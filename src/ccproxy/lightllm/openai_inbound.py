"""OpenAI Chat Completions request body → pydantic-ai IR.

Parses an OpenAI Chat Completions API request body (the wire shape that
hits ``/v1/chat/completions``) into a :class:`ParsedRequest` carrying
pydantic-ai's ``ModelMessage`` IR, ``ModelRequestParameters``, and
``ModelSettings``. Anything the IR doesn't absorb lands in
``raw_extras`` so passthrough rendering can stitch it back into the
outbound wire body.

This module is the inverse of pydantic-ai's
``OpenAIChatModel._map_messages``
(``pydantic_ai/models/openai.py:1432``) — use that as the fidelity
reference for which fields exist on the OpenAI wire.

Lossiness fixes vs the old ``pipeline/wire.py``:

* ``tool_name`` on ``ToolReturnPart`` — OpenAI's ``tool`` role messages
  carry only ``tool_call_id``, not the tool name. We do a two-pass walk:
  pass 1 builds ``tool_call_id → tool_name`` from every assistant
  ``tool_calls[].function.name``; pass 2 populates
  ``ToolReturnPart.tool_name`` so the outbound mapper can round-trip to
  Anthropic.
* Image media type — preserved via ``BinaryContent(data, media_type)``
  for ``data:image/...;base64,...`` URIs (the wire spelling Claude Code
  and other clients use). HTTP URLs become ``ImageUrl`` so pydantic-ai's
  ``_infer_media_type`` can resolve from the URL.
* Invalid tool-call JSON — wrapped as
  ``{INVALID_JSON_KEY: original_string}`` via pydantic-ai's
  ``messages.INVALID_JSON_KEY`` constant so the model can still see what
  the previous call argued, even if it wasn't valid JSON.
* Unknown content block types — preserved in
  ``raw_extras['unknown_block:msg:{i}:block:{j}']`` so the outbound
  assembler can re-emit them; we emit a ``TextPart`` placeholder so the
  conversation isn't visibly broken.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
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

from ccproxy.lightllm.parsed import ParsedRequest

logger = logging.getLogger(__name__)


# Wire fields absorbed into ModelSettings (the common base). Everything
# else from the wire body that isn't a known role/tool/message field
# lands in ``raw_extras``.
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

# OpenAI-specific settings keys we rename onto OpenAIChatModelSettings.
_OPENAI_SETTINGS_KEYS = frozenset({"logprobs", "top_logprobs"})

# Wire fields that have IR-carried meaning — skipped during raw_extras
# capture because they're already absorbed.
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
        # Everything that maps onto ModelSettings (see above).
        *_COMMON_SETTINGS_KEYS,
        *_OPENAI_SETTINGS_KEYS,
    }
)


async def parse_openai_chat(body: dict[str, Any]) -> ParsedRequest:
    """Parse an OpenAI Chat Completions request body into the IR."""
    model = cast(str, body.get("model", ""))
    raw_messages: list[dict[str, Any]] = cast(
        list[dict[str, Any]], body.get("messages", []) or []
    )

    tool_name_map = _build_tool_name_map(raw_messages=raw_messages)

    raw_extras: dict[str, Any] = {}
    messages = _parse_messages(
        raw_messages=raw_messages,
        tool_name_map=tool_name_map,
        raw_extras=raw_extras,
    )

    raw_tools = cast(list[Any], body.get("tools", []) or [])
    function_tools = _parse_tools(raw_tools=raw_tools)

    settings = _parse_settings(body=body)

    request_parameters = ModelRequestParameters(function_tools=function_tools)

    # tool_choice and response_format don't fit cleanly into IR fields
    # (output_mode / output_object require an OutputObjectDefinition
    # built upstream); preserve verbatim for the outbound renderer.
    if "tool_choice" in body:
        raw_extras["tool_choice"] = body["tool_choice"]
    if "response_format" in body:
        raw_extras["response_format"] = body["response_format"]

    # Stash every other top-level wire field that we didn't absorb so
    # passthrough rendering can stitch them back in.
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


# ---------------------------------------------------------------------------
# Tool-name resolution
# ---------------------------------------------------------------------------


def _build_tool_name_map(*, raw_messages: list[dict[str, Any]]) -> dict[str, str]:
    """Pass 1: build a ``tool_call_id → tool_name`` map.

    OpenAI's ``tool`` role messages don't carry the tool name on the
    wire, only the ``tool_call_id``. To round-trip to Anthropic via the
    IR, we need ``ToolReturnPart.tool_name`` — recover it from the
    matching assistant ``tool_calls[].function.name``.
    """
    mapping: dict[str, str] = {}
    for msg in raw_messages:
        if msg.get("role") != "assistant":
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


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def _parse_messages(
    *,
    raw_messages: list[dict[str, Any]],
    tool_name_map: dict[str, str],
    raw_extras: dict[str, Any],
) -> list[ModelMessage]:
    """Pass 2: convert each wire message into a ``ModelMessage``."""
    result: list[ModelMessage] = []
    for index, msg in enumerate(raw_messages):
        role = msg.get("role", "")
        if role == "assistant":
            result.append(
                _parse_assistant(
                    msg=msg,
                    msg_index=index,
                    raw_extras=raw_extras,
                )
            )
        else:
            result.append(
                _parse_request_role(
                    msg=msg,
                    msg_index=index,
                    tool_name_map=tool_name_map,
                    raw_extras=raw_extras,
                )
            )
    return result


def _parse_request_role(
    *,
    msg: dict[str, Any],
    msg_index: int,
    tool_name_map: dict[str, str],
    raw_extras: dict[str, Any],
) -> ModelRequest:
    """Parse a non-assistant role (system/developer/user/tool)."""
    role = msg.get("role", "")
    content = msg.get("content", "")
    parts: list[ModelRequestPart] = []

    if role == "tool":
        tool_call_id = cast(str, msg.get("tool_call_id", ""))
        tool_name = tool_name_map.get(tool_call_id, "")
        if tool_call_id and not tool_name:
            logger.warning(
                "OpenAI inbound: tool message tool_call_id=%r has no matching "
                "assistant tool_calls entry; emitting empty tool_name",
                tool_call_id,
            )
        tool_content = _coerce_tool_content(content)
        parts.append(
            ToolReturnPart(
                tool_name=tool_name,
                content=tool_content,
                tool_call_id=tool_call_id,
            )
        )
        return ModelRequest(parts=parts)

    if role in ("system", "developer"):
        if isinstance(content, str):
            if content:
                parts.append(SystemPromptPart(content=content))
        elif isinstance(content, list):
            text = _flatten_text_blocks(blocks=content)
            if text:
                parts.append(SystemPromptPart(content=text))
        return ModelRequest(parts=parts)

    # role == "user" (or any other non-tool/non-assistant role we treat
    # as user)
    user_content = _parse_user_content(
        content=content,
        msg_index=msg_index,
        raw_extras=raw_extras,
    )
    if user_content is not None:
        parts.append(UserPromptPart(content=user_content))
    return ModelRequest(parts=parts)


def _coerce_tool_content(content: Any) -> str:
    """OpenAI's ``tool`` role accepts ``str`` or ``list[block]``.

    We collapse the list form to its concatenated text for the
    ``ToolReturnPart.content`` field, which is permissive enough but
    keeps the IR simple.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _flatten_text_blocks(blocks=cast(list[Any], content))
    if content is None:
        return ""
    return str(content)


def _flatten_text_blocks(*, blocks: list[Any]) -> str:
    """Concatenate ``text`` fields from a list of ``{type, text}`` dicts."""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _parse_user_content(
    *,
    content: Any,
    msg_index: int,
    raw_extras: dict[str, Any],
) -> str | list[UserContent] | None:
    """Convert a user-role ``content`` field to IR-friendly content.

    Returns ``None`` if there's nothing to emit (e.g., empty list).
    """
    if isinstance(content, str):
        return content if content else None

    if not isinstance(content, list):
        return None

    items: list[UserContent] = []
    for block_index, block in enumerate(content):
        if not isinstance(block, dict):
            items.append(str(block))
            continue
        block_type = block.get("type", "")

        if block_type == "text":
            items.append(cast(str, block.get("text", "")))
            continue

        if block_type == "image_url":
            image_block = block.get("image_url") or {}
            url = ""
            detail: str | None = None
            if isinstance(image_block, dict):
                url = cast(str, image_block.get("url", ""))
                raw_detail = image_block.get("detail")
                if isinstance(raw_detail, str):
                    detail = raw_detail
            if not isinstance(detail, str):
                outer_detail = block.get("detail")
                if isinstance(outer_detail, str):
                    detail = outer_detail
            if detail is not None:
                raw_extras[f"image_detail:msg:{msg_index}:block:{block_index}"] = detail
            items.append(_image_url_to_user_content(url=url))
            continue

        if block_type == "input_audio":
            audio = block.get("input_audio") or {}
            data = ""
            audio_format = "wav"
            if isinstance(audio, dict):
                data = cast(str, audio.get("data", ""))
                audio_format = cast(str, audio.get("format", "wav"))
            items.append(
                BinaryContent(
                    data=_safe_b64decode(data=data),
                    media_type=f"audio/{audio_format}",
                )
            )
            continue

        if block_type == "file":
            raw_extras[f"file:msg:{msg_index}:block:{block_index}"] = block
            items.append(json.dumps(block))
            continue

        # Unknown block type — preserve verbatim, emit a stringified
        # placeholder so the IR shape stays sane.
        raw_extras[f"unknown_block:msg:{msg_index}:block:{block_index}"] = block
        items.append(json.dumps(block))

    if not items:
        return None
    return items


def _image_url_to_user_content(*, url: str) -> UserContent:
    """Turn an OpenAI ``image_url`` into a pydantic-ai ``UserContent``.

    ``data:image/...;base64,...`` becomes ``BinaryContent`` so we keep
    the media type and the bytes; plain HTTP(S) URLs become ``ImageUrl``
    so pydantic-ai's downstream mappers can resolve them.
    """
    if url.startswith("data:"):
        try:
            return cast(UserContent, BinaryContent.from_data_uri(url))
        except (ValueError, binascii.Error):
            logger.warning("OpenAI inbound: malformed data URI; falling back to ImageUrl")
            return ImageUrl(url=url)
    return ImageUrl(url=url)


def _safe_b64decode(*, data: str) -> bytes:
    """Decode a base64 string, returning empty bytes on failure."""
    try:
        return base64.b64decode(data)
    except (ValueError, binascii.Error):
        logger.warning("OpenAI inbound: malformed base64 audio payload; emitting empty bytes")
        return b""


def _parse_assistant(
    *,
    msg: dict[str, Any],
    msg_index: int,
    raw_extras: dict[str, Any],
) -> ModelResponse:
    """Parse an assistant-role message into a ``ModelResponse``."""
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
            name = cast(str, function.get("name", ""))
            args_str = function.get("arguments", "")
            args = _parse_tool_args(args_str=args_str)
            parts.append(
                ToolCallPart(
                    tool_name=name,
                    args=args,
                    tool_call_id=cast(str, call.get("id", "")),
                )
            )

    # Legacy ``function_call`` (pre-tool_calls). Preserve verbatim so the
    # outbound renderer can re-emit it.
    if "function_call" in msg:
        raw_extras[f"function_call:msg:{msg_index}"] = msg["function_call"]

    return ModelResponse(parts=parts) if parts else ModelResponse(parts=[TextPart(content="")])


def _parse_tool_args(*, args_str: Any) -> dict[str, Any] | str:
    """Parse a JSON-string ``arguments`` value into a dict.

    On parse failure, wrap the raw string via pydantic-ai's
    ``INVALID_JSON_KEY`` so the model still sees what it argued the
    previous time.
    """
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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _parse_tools(*, raw_tools: list[Any]) -> list[ToolDefinition]:
    """Parse OpenAI ``tools[].function`` entries into ``ToolDefinition``."""
    result: list[ToolDefinition] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        tool_dict = cast(dict[str, Any], tool)
        function = tool_dict.get("function") or {}
        if not isinstance(function, dict):
            continue
        function_dict = cast(dict[str, Any], function)
        name = cast(str, function_dict.get("name", ""))
        description = function_dict.get("description")
        parameters = function_dict.get("parameters") or {"type": "object", "properties": {}}
        result.append(
            ToolDefinition(
                name=name,
                parameters_json_schema=cast(dict[str, Any], parameters),
                description=cast("str | None", description),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _parse_settings(*, body: dict[str, Any]) -> ModelSettings:
    """Extract ``ModelSettings`` from the OpenAI wire body.

    ``max_completion_tokens`` (newer OpenAI) wins over ``max_tokens``
    when both are present. ``stop`` is normalized into ``stop_sequences``
    (the IR's name).
    """
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
