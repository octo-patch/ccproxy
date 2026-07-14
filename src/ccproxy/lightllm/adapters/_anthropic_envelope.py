"""Anthropic-specific envelope helpers.

Handles tool/settings parsing, system prompt extraction, cache control normalization,
and raw_extras stitching for the Anthropic Messages API wire format. Companion to
:class:`AnthropicAdapter`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from ccproxy.lightllm.adapters._tool_kinds import ANTHROPIC_TYPED_TOOLS

# pydantic-ai's CachePoint accepts only these two TTLs (Literal['5m', '1h']).
_SUPPORTED_TTLS: frozenset[str] = frozenset({"5m", "1h"})

# Top-level Anthropic body fields the IR + ModelSettings absorb. Anything else
# in the body gets parked in ``raw_extras`` keyed by its wire name.
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


def _parse_tools(raw_tools: Sequence[Any], *, settings: ModelSettings) -> tuple[list[ToolDefinition], bool]:
    """Parse Anthropic tool definitions.

    Server-side tools carry a versioned ``type`` discriminator (e.g.
    ``web_search_20250305``) that maps to a ``ToolPartKind`` in
    :data:`ANTHROPIC_TYPED_TOOLS`. When matched, ``tool_kind`` is set so the
    parts_manager's ``_typed_call_part`` promotes the response's
    ``ToolCallPart`` to its typed subclass (e.g. ``ToolSearchCallPart``).
    User-defined tools (no ``type`` field) get ``tool_kind=None``.

    Two independent conditions force ``needs_override=True`` so the caller
    preserves the original array verbatim via ``raw_extras['tools']``:

    * **Typed tools** — any entry whose ``type`` is present and not
      ``"custom"`` (Anthropic's explicit discriminator for user-defined
      tools) is a server/builtin tool whose wire shape the IR cannot model
      (versioned ``type``, side fields like ``max_uses``). ``ToolDefinition``
      entries are still built for typed-part promotion, but the wire rides
      the override.
    * **Non-canonical cache markers** — the lift to
      ``settings['anthropic_cache_tool_definitions']`` fires only for the
      shape real Anthropic clients emit (and pydantic-ai's dump contract):
      exactly one marker, sitting on the last non-deferred tool, with a
      supported TTL. Any other stamping pattern (all-stamped, mixed TTLs,
      marker on a deferred or non-boundary tool) overrides instead. When the
      typed override fires the lift is skipped entirely — the marker already
      reaches the wire verbatim, and lifting the knob too would double-count
      it in the cache engine's census.
    """
    tools: list[ToolDefinition] = []
    cache_ttls: list[str | None] = []
    has_typed_tool = False
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        wire_type = tool.get("type")
        tool_kind = ANTHROPIC_TYPED_TOOLS.get(wire_type) if isinstance(wire_type, str) else None
        if isinstance(wire_type, str) and wire_type != "custom":
            has_typed_tool = True
        tools.append(
            ToolDefinition(
                name=tool.get("name", ""),
                description=tool.get("description"),
                parameters_json_schema=tool.get("input_schema") or {},
                tool_kind=tool_kind,
                defer_loading=bool(tool.get("defer_loading")),
            )
        )
        cc = tool.get("cache_control")
        cache_ttls.append(cc.get("ttl", "5m") if isinstance(cc, dict) else None)

    if has_typed_tool:
        return tools, True
    marked = [i for i, ttl in enumerate(cache_ttls) if ttl is not None]
    if not marked:
        return tools, False
    last_stable = next((i for i in range(len(tools) - 1, -1, -1) if not tools[i].defer_loading), None)
    if len(marked) == 1 and marked[0] == last_stable and cache_ttls[marked[0]] in _SUPPORTED_TTLS:
        cast(dict[str, Any], settings)["anthropic_cache_tool_definitions"] = cache_ttls[marked[0]]
        return tools, False
    return tools, True


def _build_settings(body: dict[str, Any], *, raw_extras: dict[str, Any]) -> ModelSettings:
    """Extract sampling + behavior settings from the wire body."""
    settings: dict[str, Any] = {}
    for key in ("max_tokens", "temperature", "top_p", "stop_sequences", "top_k"):
        if key in body:
            settings[key] = body[key]
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        raw_extras["metadata"] = metadata
    return cast(ModelSettings, settings)


def _format_tools(tools: Sequence[ToolDefinition], settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Format :class:`ToolDefinition` entries as Anthropic tool dicts."""
    if not tools:
        return []
    out: list[dict[str, Any]] = []
    for tool in tools:
        entry: dict[str, Any] = {
            "name": tool.name,
            "input_schema": tool.parameters_json_schema or {"type": "object"},
        }
        if tool.description:
            entry["description"] = tool.description
        if tool.defer_loading:
            entry["defer_loading"] = True
        out.append(entry)

    # Contract source of truth — pydantic_ai/models/anthropic.py (pinned venv):
    #   # Add cache_control to the last non-deferred tool if enabled. Anthropic rejects
    #   # `cache_control` on tools with `defer_loading=True` (`Tools with defer_loading
    #   # cannot use prompt caching`); they're hidden from the model until tool search
    #   # discovers them, so they aren't part of the cacheable prompt prefix anyway.
    cache_ttl = settings.get("anthropic_cache_tool_definitions")
    if cache_ttl:
        for entry in reversed(out):
            if entry.get("defer_loading") is not True:
                entry["cache_control"] = {"type": "ephemeral", "ttl": cache_ttl}
                break
    return out


# Top-level wire fields the FSM + envelope wrapper own. ``raw_extras`` keys not
# in this set (and not IR-internal markers) get copied verbatim.
_IR_OWNED_TOP_LEVEL: frozenset[str] = frozenset(
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
    }
)


def _parse_system(raw_system: Any, *, settings: ModelSettings, raw_extras: dict[str, Any]) -> list[SystemPromptPart]:
    """Extract the top-level Anthropic ``system`` field into SystemPromptParts.

    Cache control on system blocks is normalized:

    * All blocks share the same supported TTL (``5m`` / ``1h``) → lift to
      ``settings['anthropic_cache_instructions']`` so the dump side can re-attach
      uniformly. Returns plain SystemPromptParts (no per-block cache markers).
    * Mixed or non-standard TTLs → stash the raw block list in
      ``raw_extras['system']`` so the dump side can passthrough verbatim.
      Returns SystemPromptParts without cache markers (the round-trip rides
      on raw_extras).
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
        parts.append(SystemPromptPart(content=block.get("text", "")))
        cc = block.get("cache_control")
        cache_ttls.append(cc.get("ttl", "5m") if isinstance(cc, dict) else None)

    cached_ttls = {ttl for ttl in cache_ttls if ttl is not None}
    if not cached_ttls:
        return parts

    if len(cached_ttls) == 1:
        only_ttl = next(iter(cached_ttls))
        if all(t is not None for t in cache_ttls) and only_ttl in _SUPPORTED_TTLS:
            cast(dict[str, Any], settings)["anthropic_cache_instructions"] = only_ttl
            return parts

    raw_extras["system"] = raw_system
    return parts


def _attach_system_prompts(messages: list[ModelMessage], system_parts: list[SystemPromptPart]) -> list[ModelMessage]:
    """Prepend ``system_parts`` to the first ``ModelRequest`` in ``messages``.

    If no ``ModelRequest`` exists, a new one is created at position 0.
    """
    if not system_parts:
        return messages
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest):
            new_parts: list[Any] = [*system_parts, *msg.parts]
            messages[i] = ModelRequest(parts=new_parts)
            return messages
    return [ModelRequest(parts=list(system_parts)), *messages]


def _stitch_raw_extras(body: dict[str, Any], raw_extras: dict[str, Any]) -> None:
    """Re-inject ``raw_extras`` entries onto the rendered body."""
    for key in ("system", "tools"):
        if key in raw_extras:
            body[key] = raw_extras[key]

    for key, value in raw_extras.items():
        if key in ("system", "tools"):
            continue
        if key.startswith(("cc:", "unknown_block:")):
            continue
        body.setdefault(key, value)
