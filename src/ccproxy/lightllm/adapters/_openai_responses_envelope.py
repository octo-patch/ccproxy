"""OpenAI Responses-specific envelope helpers.

Per-item-kind dispatch for the ``input[]`` discriminated union, plus
settings/raw_extras helpers for the ``/v1/responses`` request body.

The ``input[]`` union has 27 distinct ``type`` values. They split into
four buckets:

* **IR-modellable** — ``message`` (and the ``EasyInputMessageParam``
  shorthand), ``function_call``, ``function_call_output``. Become
  ``pydantic_ai.messages`` parts directly.
* **Reasoning** — ``reasoning`` items have a structured ``summary[]`` +
  ``content[]`` plus optional ``encrypted_content`` that
  :class:`ThinkingPart` cannot fully model. Extract joined text into a
  :class:`ThinkingPart` and stash the FULL raw dict under
  ``openai_responses:reasoning:N`` for lossless round-trip.
* **Server-side tools** — 17 kinds (``web_search_call``,
  ``code_interpreter_call``, ``mcp_call``, etc.) have no IR equivalent.
  Stash under ``openai_responses:server_tool:N``.
* **Unknown** — forward-compat fallback for future SDK additions. Stash
  under ``openai_responses:unknown_item:N``.

Item ``id`` fields (used by ``previous_response_id`` chaining) are
stashed under ``openai_responses:item_id:N`` for every item that
carries one.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic_ai.messages import (
    ImageUrl,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.ui import MessagesBuilder

logger = logging.getLogger(__name__)


# Top-level body fields the IR + ModelSettings absorb. Everything else
# lands in ``raw_extras`` keyed by wire name.
_ABSORBED_TOP_LEVEL: frozenset[str] = frozenset(
    {
        "model",
        "input",
        "instructions",
        "tools",
        "temperature",
        "top_p",
        "max_output_tokens",
        "stream",
        "metadata",
    }
)


# Server-side tool kinds — enumerated so the catch-all branch only
# fires for genuine forward-compat unknown items.
_SERVER_TOOL_KINDS: frozenset[str] = frozenset(
    {
        "web_search_call",
        "code_interpreter_call",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
        "mcp_approval_response",
        "file_search_call",
        "computer_call",
        "computer_call_output",
        "apply_patch_call",
        "apply_patch_call_output",
        "local_shell_call",
        "local_shell_call_output",
        "shell_call",
        "shell_call_output",
        "image_generation_call",
        "custom_tool_call",
        "custom_tool_call_output",
        "tool_search_call",
        "tool_search_call_output",
        "compaction",
        "item_reference",
    }
)


# Roles the IR maps to system prompts (instructions hierarchy).
_SYSTEM_ROLES: frozenset[str] = frozenset({"system", "developer"})


def _parse_responses_settings(body: Mapping[str, Any]) -> ModelSettings:
    """Extract sampling settings from a ``/v1/responses`` request body.

    The Responses API uses ``max_output_tokens`` where Chat uses
    ``max_completion_tokens``/``max_tokens``. Map both into the IR's
    canonical ``max_tokens`` key; the original wire name is preserved
    via raw_extras so render() can restore it.
    """
    settings: dict[str, Any] = {}

    max_tokens = body.get("max_output_tokens")
    if isinstance(max_tokens, int):
        settings["max_tokens"] = max_tokens

    for key in ("temperature", "top_p"):
        if key in body:
            settings[key] = body[key]

    return cast(ModelSettings, settings)


def _apply_responses_settings(body: dict[str, Any], settings: Mapping[str, Any]) -> None:
    """Copy IR settings onto a Responses wire body."""
    if "max_tokens" in settings:
        body["max_output_tokens"] = settings["max_tokens"]
    for key in ("temperature", "top_p"):
        if key in settings:
            body[key] = settings[key]


def _parse_responses_tools(raw_tools: Sequence[Any]) -> list[ToolDefinition]:
    """Parse Responses ``tools[]`` entries into :class:`ToolDefinition`."""
    tools: list[ToolDefinition] = []
    for tool in raw_tools:
        if not isinstance(tool, Mapping):
            continue
        if tool.get("type") != "function":
            continue

        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue

        parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}

        description = tool.get("description")
        strict = tool.get("strict")
        tools.append(
            ToolDefinition(
                name=name,
                parameters_json_schema=cast(dict[str, Any], parameters),
                description=description if isinstance(description, str) else None,
                strict=strict if isinstance(strict, bool) else None,
            )
        )
    return tools


def _format_responses_tools(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    """Format :class:`ToolDefinition` entries into Responses ``tools[]`` dicts."""
    out: list[dict[str, Any]] = []
    for tool in tools:
        item: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "parameters": tool.parameters_json_schema or {"type": "object", "properties": {}},
        }
        if tool.description:
            item["description"] = tool.description
        if tool.strict is not None:
            item["strict"] = tool.strict
        out.append(item)
    return out


def _build_tool_call_id_index(input_items: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Pre-scan ``input[]`` for ``function_call`` items to map call_id → tool name.

    Used so a ``function_call_output`` item can carry the tool name
    forward into its :class:`ToolReturnPart`. Mirrors
    :mod:`_anthropic_envelope`'s ``tool_use_id → tool_name`` index for
    ``tool_result`` blocks.
    """
    index: dict[str, str] = {}
    for item in input_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            if isinstance(call_id, str) and isinstance(name, str) and call_id:
                index[call_id] = name
    return index


def _load_message_content(
    content: Any,
    *,
    msg_index: int,
    raw_extras: dict[str, Any],
) -> list[UserContent]:
    """Parse a ``message`` item's ``content`` (string or content-part list).

    Returns a list of pydantic-ai user-content items. Unknown content
    parts are JSON-serialized and stashed via the
    ``unknown_block:msg:N:idx:M`` convention.
    """
    if isinstance(content, str):
        return [content] if content else []
    if not isinstance(content, list):
        return []

    out: list[UserContent] = []
    for part_index, part in enumerate(content):
        if not isinstance(part, dict):
            raw_extras[f"unknown_block:msg:{msg_index}:idx:{part_index}"] = part
            out.append(json.dumps(part))
            continue
        block = cast(dict[str, Any], part)
        ptype = block.get("type")
        if ptype in ("input_text", "text", "output_text"):
            out.append(block.get("text", ""))
        elif ptype == "input_image":
            url = block.get("image_url")
            if isinstance(url, dict):
                url_str = cast(dict[str, Any], url).get("url", "")
            elif isinstance(url, str):
                url_str = url
            else:
                url_str = ""
            if url_str:
                out.append(ImageUrl(url=url_str))
            else:
                raw_extras[f"unknown_block:msg:{msg_index}:idx:{part_index}"] = block
        elif ptype in ("input_file", "refusal"):
            raw_extras[f"unknown_block:msg:{msg_index}:idx:{part_index}"] = block
        else:
            raw_extras[f"unknown_block:msg:{msg_index}:idx:{part_index}"] = block
            out.append(json.dumps(block))
    return out


def _reasoning_text(item: Mapping[str, Any]) -> str:
    """Join all ``summary[].text`` + ``content[].text`` into one string.

    pydantic-ai's :class:`ThinkingPart` carries a single content string;
    the SDK splits reasoning into two parallel lists. We join them in
    order (summary then content) with newlines.
    """
    pieces: list[str] = []
    summary = item.get("summary")
    if isinstance(summary, list):
        for block in summary:
            if isinstance(block, dict):
                txt = block.get("text")
                if isinstance(txt, str) and txt:
                    pieces.append(txt)
    content = item.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                txt = block.get("text")
                if isinstance(txt, str) and txt:
                    pieces.append(txt)
    return "\n".join(pieces)


def parse_input_item(
    item: Mapping[str, Any],
    builder: MessagesBuilder,
    *,
    item_index: int,
    tool_name_by_id: Mapping[str, str],
    raw_extras: dict[str, Any],
) -> None:
    """Dispatch a single ``input[]`` item to the appropriate IR part.

    Items that don't model into IR are stashed in ``raw_extras`` under
    one of four conventional keys (see module docstring).
    """
    item_id = item.get("id")
    if isinstance(item_id, str) and item_id:
        raw_extras[f"openai_responses:item_id:{item_index}"] = item_id

    item_type = item.get("type")

    if item_type == "message" or (item_type is None and "role" in item):
        role = item.get("role")
        content = item.get("content")

        if role in _SYSTEM_ROLES:
            if isinstance(content, str):
                if content:
                    builder.add(SystemPromptPart(content=content))
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("input_text", "text"):
                        text = part.get("text", "")
                        if text:
                            builder.add(SystemPromptPart(content=text))
            return

        if role == "user":
            parts = _load_message_content(content, msg_index=item_index, raw_extras=raw_extras)
            if parts:
                builder.add(UserPromptPart(content=parts))
            return

        if role == "assistant":
            if isinstance(content, str):
                if content:
                    builder.add(TextPart(content=content))
                return
            if isinstance(content, list):
                for part_index, part in enumerate(content):
                    if not isinstance(part, dict):
                        builder.add(TextPart(content=json.dumps(part)))
                        continue
                    block = cast(dict[str, Any], part)
                    ptype = block.get("type")
                    if ptype in ("output_text", "text"):
                        builder.add(TextPart(content=block.get("text", "")))
                    elif ptype == "refusal":
                        raw_extras[
                            f"openai_responses:refusal:{item_index}:{part_index}"
                        ] = dict(block)
                    else:
                        raw_extras[
                            f"unknown_block:msg:{item_index}:idx:{part_index}"
                        ] = block
                        builder.add(TextPart(content=json.dumps(block)))
            return

        # Unknown role — stash whole item, don't crash.
        raw_extras[f"openai_responses:unknown_item:{item_index}"] = dict(item)
        return

    if item_type == "function_call":
        args = item.get("arguments", "")
        if isinstance(args, dict):
            args = json.dumps(args, separators=(",", ":"))
        builder.add(
            ToolCallPart(
                tool_name=item.get("name", ""),
                args=args,
                tool_call_id=item.get("call_id", ""),
            )
        )
        return

    if item_type == "function_call_output":
        call_id = item.get("call_id", "")
        output = item.get("output", "")
        if not isinstance(output, str):
            output = json.dumps(output, separators=(",", ":"))
        tool_name = tool_name_by_id.get(call_id, "")
        if not tool_name and call_id:
            logger.debug(
                "openai_responses load: function_call_output references unknown call_id %r — leaving tool_name blank",
                call_id,
            )
        builder.add(
            ToolReturnPart(
                tool_name=tool_name,
                content=output,
                tool_call_id=call_id,
            )
        )
        return

    if item_type == "reasoning":
        text = _reasoning_text(item)
        builder.add(
            ThinkingPart(
                content=text,
                signature=None,
                provider_name="openai",
            )
        )
        raw_extras[f"openai_responses:reasoning:{item_index}"] = dict(item)
        return

    if item_type in _SERVER_TOOL_KINDS:
        raw_extras[f"openai_responses:server_tool:{item_index}"] = dict(item)
        return

    # Forward-compat: unknown item type. Don't crash; stash and continue.
    logger.debug(
        "openai_responses load: unknown item type %r at index %d — stashing in raw_extras",
        item_type,
        item_index,
    )
    raw_extras[f"openai_responses:unknown_item:{item_index}"] = dict(item)


# ── render-side helpers ──────────────────────────────────────────────────────


def _format_user_content(parts: Sequence[Any]) -> list[dict[str, Any]]:
    """Render pydantic-ai user-content items into Responses content parts.

    String items become ``{"type": "input_text", "text": ...}``;
    :class:`ImageUrl` items become ``{"type": "input_image", "image_url":
    {"url": ...}}``. Other items are best-effort serialized.
    """
    out: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            out.append({"type": "input_text", "text": part})
        elif isinstance(part, ImageUrl):
            out.append({"type": "input_image", "image_url": {"url": part.url}})
        else:
            # Best effort — wrap in input_text via JSON serialization.
            out.append({"type": "input_text", "text": json.dumps(part, default=str)})
    return out


_RAW_EXTRA_INTERNAL_PREFIXES: tuple[str, ...] = (
    "openai_responses:reasoning:",
    "openai_responses:server_tool:",
    "openai_responses:item_id:",
    "openai_responses:unknown_item:",
    "openai_responses:refusal:",
    "unknown_block:",
    "cc:",
)


def _stitch_raw_extras_top_level(body: dict[str, Any], raw_extras: Mapping[str, Any]) -> None:
    """Re-inject top-level fields preserved in ``raw_extras``.

    Per-item raw_extras (``openai_responses:server_tool:N`` etc.) are
    handled by the adapter's render path which inserts them into
    ``input[]`` at their original positions. Top-level keys like
    ``previous_response_id``, ``prompt_cache_key``,
    ``prompt_cache_retention``, ``reasoning``, ``tool_choice`` etc. are
    copied verbatim onto the wire body here.
    """
    for key, value in raw_extras.items():
        if key.startswith(_RAW_EXTRA_INTERNAL_PREFIXES):
            continue
        if key in _ABSORBED_TOP_LEVEL:
            continue
        body.setdefault(key, value)
