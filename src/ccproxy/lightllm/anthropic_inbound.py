"""Anthropic Messages API request body → pydantic-ai ``ParsedRequest``.

The inverse of ``pydantic_ai.models.anthropic.AnthropicModel._map_message``.
Replaces the lossy ``ccproxy.pipeline.wire`` parser:

* ``ToolReturnPart.tool_name`` is resolved via a two-pass walk over assistant
  ``tool_use`` blocks instead of being hardcoded to ``""``.
* Image blocks become ``BinaryContent(data, media_type)`` (or ``ImageUrl``)
  instead of bare base64 strings, preserving ``media_type``.
* ``cache_control.ttl`` values pydantic-ai cannot represent (anything other
  than ``"5m"`` / ``"1h"``) are stashed in ``raw_extras`` instead of being
  coerced.
* Unknown content blocks are stashed in ``raw_extras`` so the outbound
  renderer can reconstruct them; their text is fed into the IR as JSON so
  downstream consumers still see *something* for those blocks.

Cache-control on system blocks and tool definitions, which pydantic-ai has
no per-block IR carrier for, is compressed to
``AnthropicModelSettings.anthropic_cache_{instructions,tool_definitions}``
when uniform across blocks; otherwise the original wire blocks are stashed
in ``raw_extras`` for the outbound renderer to override.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, cast

from pydantic_ai.messages import (
    BinaryContent,
    CachePoint,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
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

# pydantic-ai's CachePoint only accepts these two TTLs (Literal['5m', '1h']).
_SUPPORTED_TTLS: frozenset[str] = frozenset({"5m", "1h"})

# Top-level Anthropic body fields the IR + ModelSettings absorb. Anything else
# in the body that isn't in this set gets parked in ``raw_extras`` keyed by
# its wire name.
_ABSORBED_TOP_LEVEL: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "system",
        "tools",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "stream",
        "metadata",
    }
)


async def parse_anthropic_messages(body: dict[str, Any]) -> ParsedRequest:
    """Parse an Anthropic Messages API request body into the IR.

    ``body`` is the already-JSON-decoded request body (a dict). Returns a
    :class:`ParsedRequest` carrying pydantic-ai IR messages, the function
    tools as :class:`ModelRequestParameters`, sampling/behavior settings as
    :class:`ModelSettings`, the declared model name, the stream flag, and
    ``raw_extras`` for any wire fields the IR doesn't absorb.
    """
    raw_extras: dict[str, Any] = {}

    model = str(body.get("model", ""))
    stream = bool(body.get("stream", False))

    raw_messages = body.get("messages") or []
    tool_name_lookup = _build_tool_name_lookup(raw_messages)
    messages = _parse_messages(raw_messages, tool_name_lookup, raw_extras=raw_extras)

    settings: ModelSettings = _build_settings(body, raw_extras=raw_extras)
    request_parameters = _build_request_parameters(body, settings=settings, raw_extras=raw_extras)

    system = _parse_system(body.get("system"), settings=settings, raw_extras=raw_extras)
    if system:
        # Prepend system parts to the first ModelRequest, or create one if
        # the conversation begins with an assistant turn.
        messages = _attach_system_prompts(messages, system)

    # Park any top-level wire fields the IR didn't absorb so the outbound
    # renderer can stitch them back in for passthrough.
    for key, value in body.items():
        if key in _ABSORBED_TOP_LEVEL:
            continue
        raw_extras.setdefault(key, value)

    return ParsedRequest(
        model=model,
        messages=messages,
        request_parameters=request_parameters,
        settings=settings,
        stream=stream,
        raw_extras=raw_extras,
    )


# ---------------------------------------------------------------------------
# Tool name resolution: pass 1 over assistant tool_use blocks
# ---------------------------------------------------------------------------


def _build_tool_name_lookup(raw_messages: list[Any]) -> dict[str, str]:
    """Walk assistant messages to build ``tool_use_id -> tool_name``."""
    lookup: dict[str, str] = {}
    for msg in raw_messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_id = block.get("id", "")
                tool_name = block.get("name", "")
                if tool_id:
                    lookup[tool_id] = tool_name
    return lookup


# ---------------------------------------------------------------------------
# Messages: pass 2 with tool_name lookup
# ---------------------------------------------------------------------------


def _parse_messages(
    raw_messages: list[Any],
    tool_name_lookup: dict[str, str],
    *,
    raw_extras: dict[str, Any],
) -> list[ModelMessage]:
    result: list[ModelMessage] = []
    for i, msg in enumerate(raw_messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            result.append(_parse_assistant_message(content, msg_index=i, raw_extras=raw_extras))
        else:
            result.append(
                _parse_request_message(
                    msg,
                    msg_index=i,
                    tool_name_lookup=tool_name_lookup,
                    raw_extras=raw_extras,
                )
            )
    return result


def _parse_request_message(
    msg: dict[str, Any],
    *,
    msg_index: int,
    tool_name_lookup: dict[str, str],
    raw_extras: dict[str, Any],
) -> ModelRequest:
    """Parse a user/system role message into ``ModelRequest``."""
    content = msg.get("content", "")
    parts: list[SystemPromptPart | UserPromptPart | ToolReturnPart] = []

    if isinstance(content, str):
        if msg.get("role") == "system":
            parts.append(SystemPromptPart(content=content))
        else:
            parts.append(UserPromptPart(content=content))
        return ModelRequest(parts=parts)

    if not isinstance(content, list):
        return ModelRequest(parts=[])

    user_content_items: list[UserContent] = []

    for j, raw_block in enumerate(content):
        if not isinstance(raw_block, dict):
            user_content_items.append(json.dumps(raw_block))
            raw_extras[f"unknown_block:msg:{msg_index}:idx:{j}"] = raw_block
            continue
        block = cast("dict[str, Any]", raw_block)

        block_type = block.get("type", "")

        if block_type == "tool_result":
            if user_content_items:
                parts.append(UserPromptPart(content=list(user_content_items)))
                user_content_items = []
            parts.append(
                _parse_tool_result_block(
                    block,
                    tool_name_lookup=tool_name_lookup,
                )
            )

        elif block_type == "text":
            text = block.get("text", "")
            user_content_items.append(text)
            _emit_cache_control(
                block.get("cache_control"),
                items=user_content_items,
                msg_index=msg_index,
                block_index=j,
                raw_extras=raw_extras,
            )

        elif block_type == "image":
            user_content_items.append(_parse_image_block(block))
            _emit_cache_control(
                block.get("cache_control"),
                items=user_content_items,
                msg_index=msg_index,
                block_index=j,
                raw_extras=raw_extras,
            )

        else:
            user_content_items.append(json.dumps(block))
            raw_extras[f"unknown_block:msg:{msg_index}:idx:{j}"] = block

    if user_content_items:
        parts.append(UserPromptPart(content=list(user_content_items)))

    return ModelRequest(parts=parts)


def _parse_tool_result_block(
    block: dict[str, Any],
    *,
    tool_name_lookup: dict[str, str],
) -> ToolReturnPart:
    """Parse an Anthropic ``tool_result`` content block."""
    raw_content = block.get("content", "")
    if isinstance(raw_content, list):
        texts = [b.get("text", "") for b in raw_content if isinstance(b, dict) and b.get("type") == "text"]
        content: Any = "\n".join(texts) if texts else str(raw_content)
    else:
        content = raw_content

    tool_use_id = block.get("tool_use_id", "")
    tool_name = tool_name_lookup.get(tool_use_id, "")
    if not tool_name and tool_use_id:
        logger.debug(
            "anthropic inbound: tool_result references unknown tool_use_id %r — leaving tool_name blank",
            tool_use_id,
        )

    return ToolReturnPart(
        tool_name=tool_name,
        content=content,
        tool_call_id=tool_use_id,
    )


def _parse_image_block(block: dict[str, Any]) -> UserContent:
    """Parse an Anthropic ``image`` block into a ``BinaryContent`` or ``ImageUrl``."""
    source = block.get("source") or {}
    if not isinstance(source, dict):
        return ""

    source_type = source.get("type", "base64")
    media_type = source.get("media_type", "application/octet-stream")

    if source_type == "url":
        url = source.get("url", "")
        return ImageUrl(url=url, media_type=media_type) if url else ""

    data_field = source.get("data", "")
    if isinstance(data_field, bytes):
        data_bytes = data_field
    else:
        try:
            data_bytes = base64.b64decode(data_field) if data_field else b""
        except (ValueError, TypeError):
            # Treat non-base64 payloads as opaque bytes so we don't fail the
            # whole request — preserves whatever the upstream wanted.
            data_bytes = data_field.encode("utf-8") if isinstance(data_field, str) else b""

    return BinaryContent(data=data_bytes, media_type=media_type)


def _parse_assistant_message(
    content: str | list[Any],
    *,
    msg_index: int,
    raw_extras: dict[str, Any],
) -> ModelResponse:
    """Parse an assistant role message into ``ModelResponse``."""
    if isinstance(content, str):
        return ModelResponse(parts=[TextPart(content=content)])

    parts: list[ModelResponsePart] = []
    for j, raw_block in enumerate(content):
        if not isinstance(raw_block, dict):
            parts.append(TextPart(content=json.dumps(raw_block)))
            raw_extras[f"unknown_block:msg:{msg_index}:idx:{j}"] = raw_block
            continue
        block = cast("dict[str, Any]", raw_block)

        block_type = block.get("type", "")
        if block_type == "text":
            parts.append(TextPart(content=block.get("text", "")))
        elif block_type == "tool_use":
            parts.append(
                ToolCallPart(
                    tool_name=block.get("name", ""),
                    args=block.get("input"),
                    tool_call_id=block.get("id", ""),
                )
            )
        elif block_type == "thinking":
            parts.append(
                ThinkingPart(
                    content=block.get("thinking", ""),
                    signature=block.get("signature"),
                )
            )
        elif block_type == "redacted_thinking":
            parts.append(
                ThinkingPart(
                    content="",
                    id="redacted_thinking",
                    signature=block.get("data"),
                )
            )
        else:
            parts.append(TextPart(content=json.dumps(block)))
            raw_extras[f"unknown_block:msg:{msg_index}:idx:{j}"] = block

    if not parts:
        parts.append(TextPart(content=""))
    return ModelResponse(parts=parts)


# ---------------------------------------------------------------------------
# Cache control
# ---------------------------------------------------------------------------


def _emit_cache_control(
    cc: Any,
    *,
    items: list[UserContent],
    msg_index: int,
    block_index: int,
    raw_extras: dict[str, Any],
) -> None:
    """Append a ``CachePoint`` after the just-added content item.

    If the wire ``ttl`` isn't one pydantic-ai supports, stash the original
    cache_control dict in ``raw_extras`` and skip the IR marker — the
    outbound renderer is responsible for re-applying it.
    """
    if not isinstance(cc, dict):
        return
    cc_dict = cast("dict[str, Any]", cc)
    ttl = cc_dict.get("ttl", "5m")
    if ttl == "5m" or ttl == "1h":
        items.append(CachePoint(ttl=ttl))
        return
    raw_extras[f"cc:msg:{msg_index}:block:{block_index}"] = cc_dict


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


def _parse_system(
    raw_system: Any,
    *,
    settings: ModelSettings,
    raw_extras: dict[str, Any],
) -> list[SystemPromptPart]:
    """Parse the wire ``system`` field into ``SystemPromptPart`` entries.

    Cache control on system blocks is compressed to
    ``anthropic_cache_instructions`` when uniform across blocks; non-uniform
    blocks land in ``raw_extras['system']`` for the outbound renderer to
    override.
    """
    if raw_system is None:
        return []

    if isinstance(raw_system, str):
        return [SystemPromptPart(content=raw_system)] if raw_system else []

    if not isinstance(raw_system, list):
        return []

    parts: list[SystemPromptPart] = []
    cache_ttls: list[str | None] = []
    for block in raw_system:
        if not isinstance(block, dict):
            continue
        text = block.get("text", "")
        parts.append(SystemPromptPart(content=text))
        cc = block.get("cache_control")
        if isinstance(cc, dict):
            cache_ttls.append(cc.get("ttl", "5m"))
        else:
            cache_ttls.append(None)

    cached_ttls = {ttl for ttl in cache_ttls if ttl is not None}
    if not cached_ttls:
        return parts

    # Uniform single supported TTL → settings-level cache marker.
    if len(cached_ttls) == 1:
        only_ttl = next(iter(cached_ttls))
        all_blocks_cached = all(t is not None for t in cache_ttls)
        if all_blocks_cached and only_ttl in _SUPPORTED_TTLS:
            anthropic_settings = cast(dict[str, Any], settings)
            anthropic_settings["anthropic_cache_instructions"] = only_ttl
            return parts

    # Anything else (mixed, partial coverage, unsupported TTL) — preserve the
    # original blocks for the outbound renderer.
    raw_extras["system"] = raw_system
    return parts


def _attach_system_prompts(
    messages: list[ModelMessage],
    system_parts: list[SystemPromptPart],
) -> list[ModelMessage]:
    """Prepend ``system_parts`` to the first ``ModelRequest`` in ``messages``."""
    if not system_parts:
        return messages
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest):
            new_parts: list[Any] = [*system_parts, *msg.parts]
            messages[i] = ModelRequest(parts=new_parts)
            return messages
    # No ModelRequest in history — start one to anchor the system parts.
    return [ModelRequest(parts=list(system_parts)), *messages]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _build_request_parameters(
    body: dict[str, Any],
    *,
    settings: ModelSettings,
    raw_extras: dict[str, Any],
) -> ModelRequestParameters:
    raw_tools = body.get("tools") or []
    function_tools, has_mixed_cache = _parse_tools(
        raw_tools,
        settings=settings,
    )
    if has_mixed_cache:
        raw_extras["tools"] = raw_tools

    return ModelRequestParameters(function_tools=function_tools)


def _parse_tools(
    raw_tools: list[Any],
    *,
    settings: ModelSettings,
) -> tuple[list[ToolDefinition], bool]:
    """Parse Anthropic tool definitions.

    Returns the parsed ``ToolDefinition`` list and a flag indicating whether
    cache-control across tools was non-uniform (in which case the caller
    should stash the originals in ``raw_extras['tools']``).
    """
    tools: list[ToolDefinition] = []
    cache_ttls: list[str | None] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name", "")
        description = tool.get("description")
        schema = tool.get("input_schema") or {}
        tools.append(
            ToolDefinition(
                name=name,
                description=description,
                parameters_json_schema=schema,
            )
        )
        cc = tool.get("cache_control")
        if isinstance(cc, dict):
            cache_ttls.append(cc.get("ttl", "5m"))
        else:
            cache_ttls.append(None)

    cached_ttls = {ttl for ttl in cache_ttls if ttl is not None}
    if not cached_ttls:
        return tools, False

    if len(cached_ttls) == 1:
        only_ttl = next(iter(cached_ttls))
        all_cached = all(t is not None for t in cache_ttls)
        if all_cached and only_ttl in _SUPPORTED_TTLS:
            anthropic_settings = cast(dict[str, Any], settings)
            anthropic_settings["anthropic_cache_tool_definitions"] = only_ttl
            return tools, False

    return tools, True


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _build_settings(
    body: dict[str, Any],
    *,
    raw_extras: dict[str, Any],
) -> ModelSettings:
    settings: dict[str, Any] = {}
    if "max_tokens" in body:
        settings["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        settings["temperature"] = body["temperature"]
    if "top_p" in body:
        settings["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        settings["stop_sequences"] = body["stop_sequences"]
    # ``top_k`` lives in AnthropicModelSettings, not the cross-provider
    # ``ModelSettings`` — the TypedDict is total=False so an extra key
    # passes at runtime; static typing tolerates it through the cast.
    if "top_k" in body:
        settings["top_k"] = body["top_k"]
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        # ``ModelSettings`` has no top-level metadata slot; preserve the
        # wire dict for the outbound renderer.
        raw_extras["metadata"] = metadata
    return cast(ModelSettings, settings)
