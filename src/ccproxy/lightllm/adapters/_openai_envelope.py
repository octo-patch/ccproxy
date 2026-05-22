"""OpenAI-specific envelope helpers.

Extracted from the retired FSM modules (graph/openai_load.py + openai_dump.py).
Handles tool/settings parsing, wire-to-IR key mapping, and raw_extras stitching
for the OpenAI Chat Completions API wire format.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

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


# Wire fields the FSM + envelope wrapper own.
_OPENAI_IR_OWNED_TOP_LEVEL: frozenset[str] = frozenset(
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


def _stitch_raw_extras(body: dict[str, Any], raw_extras: dict[str, Any]) -> None:
    """Re-inject non-IR-internal ``raw_extras`` onto the rendered body."""
    for key in ("tool_choice", "response_format"):
        if key in raw_extras:
            body[key] = raw_extras[key]

    for key, value in raw_extras.items():
        if key in ("tool_choice", "response_format"):
            continue
        if key.startswith(_INTERNAL_RAW_EXTRA_PREFIXES):
            continue
        body.setdefault(key, value)
