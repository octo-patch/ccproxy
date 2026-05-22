"""ParsedRequest bridge for the new UIAdapters.

Phase B scaffolding: ``Context.ensure_parsed`` and ``Context._flush_parsed_to_body``
still operate on :class:`ParsedRequest`. This module builds + renders one
using the new :class:`AnthropicAdapter` / :class:`OpenAIChatAdapter` for
the messages, and uses local envelope helpers for tools, settings, and raw_extras.
"""

from __future__ import annotations

import json
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam
from pydantic_ai.models import ModelRequestParameters

from ccproxy.lightllm.adapters._anthropic_envelope import (
    _ABSORBED_TOP_LEVEL as _ANTHROPIC_ABSORBED,
)
from ccproxy.lightllm.adapters._anthropic_envelope import (
    _attach_system_prompts as _anthropic_attach_system_prompts,
)
from ccproxy.lightllm.adapters._anthropic_envelope import (
    _build_settings as _anthropic_build_settings,
)
from ccproxy.lightllm.adapters._anthropic_envelope import (
    _format_tools as _anthropic_format_tools,
)
from ccproxy.lightllm.adapters._anthropic_envelope import (
    _parse_system as _anthropic_parse_system,
)
from ccproxy.lightllm.adapters._anthropic_envelope import (
    _parse_tools as _anthropic_parse_tools,
)
from ccproxy.lightllm.adapters._anthropic_envelope import (
    _stitch_raw_extras as _anthropic_stitch_raw_extras,
)
from ccproxy.lightllm.adapters._openai_envelope import (
    _ABSORBED_BODY_KEYS as _OPENAI_ABSORBED,
)
from ccproxy.lightllm.adapters._openai_envelope import (
    _apply_settings as _openai_apply_settings,
)
from ccproxy.lightllm.adapters._openai_envelope import (
    _format_tools as _openai_format_tools,
)
from ccproxy.lightllm.adapters._openai_envelope import (
    _parse_settings as _openai_parse_settings,
)
from ccproxy.lightllm.adapters._openai_envelope import (
    _parse_tools as _openai_parse_tools,
)
from ccproxy.lightllm.adapters._openai_envelope import (
    _stitch_raw_extras as _openai_stitch_raw_extras,
)
from ccproxy.lightllm.adapters.anthropic import AnthropicAdapter
from ccproxy.lightllm.adapters.openai_chat import OpenAIChatAdapter
from ccproxy.lightllm.parsed import ListenerFormat, ParsedRequest


def parse_request(body: dict[str, Any], *, listener_format: ListenerFormat) -> ParsedRequest:
    """Build a :class:`ParsedRequest` from a wire body using the new adapters."""
    if listener_format is ListenerFormat.ANTHROPIC_MESSAGES:
        return _parse_anthropic(body)
    if listener_format is ListenerFormat.OPENAI_CHAT:
        return _parse_openai_chat(body)
    raise ValueError(f"no IR parser for listener_format={listener_format}")


def render_request(parsed: ParsedRequest, *, listener_format: ListenerFormat) -> bytes:
    """Render a :class:`ParsedRequest` to wire bytes using the new adapters."""
    if listener_format is ListenerFormat.ANTHROPIC_MESSAGES:
        return _render_anthropic(parsed)
    if listener_format is ListenerFormat.OPENAI_CHAT:
        return _render_openai_chat(parsed)
    raise ValueError(f"no IR renderer for listener_format={listener_format}")


# ── Anthropic ───────────────────────────────────────────────────────────────


def _parse_anthropic(body: dict[str, Any]) -> ParsedRequest:
    raw_extras: dict[str, Any] = {}

    model = str(body.get("model", ""))
    stream = bool(body.get("stream", False))

    raw_messages = body.get("messages") or []
    # System is handled by _anthropic_parse_system below — pass system=None to the
    # adapter so it doesn't double-process and emit sentinel CachePoint markers.
    messages = AnthropicAdapter.load_messages(raw_messages, system=None, raw_extras=raw_extras)

    settings = _anthropic_build_settings(body, raw_extras=raw_extras)

    raw_tools = body.get("tools") or []
    function_tools, has_mixed_cache = _anthropic_parse_tools(raw_tools, settings=settings)
    if has_mixed_cache:
        raw_extras["tools"] = raw_tools
    request_parameters = ModelRequestParameters(function_tools=function_tools)

    system_parts = _anthropic_parse_system(
        body.get("system"), settings=settings, raw_extras=raw_extras
    )
    if system_parts:
        messages = _anthropic_attach_system_prompts(messages, system_parts)

    for key, value in body.items():
        if key in _ANTHROPIC_ABSORBED:
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


def _render_anthropic(parsed: ParsedRequest) -> bytes:
    settings_dict = cast(dict[str, Any], parsed.settings)
    system = AnthropicAdapter.dump_system(parsed.messages)
    messages = AnthropicAdapter.dump_messages(parsed.messages)
    tools = _anthropic_format_tools(parsed.request_parameters.function_tools, settings_dict)

    body: dict[str, Any] = {
        "model": parsed.model,
        "messages": messages,
    }
    for key in ("max_tokens", "temperature", "top_p", "top_k", "stop_sequences"):
        if key in settings_dict:
            body[key] = settings_dict[key]
    if system is not None:
        body["system"] = system
    if tools:
        body["tools"] = tools

    _anthropic_stitch_raw_extras(body, parsed.raw_extras)

    if parsed.stream:
        body["stream"] = True

    return json.dumps(body, separators=(",", ":")).encode()


# ── OpenAI Chat Completions ─────────────────────────────────────────────────


def _parse_openai_chat(body: dict[str, Any]) -> ParsedRequest:
    model = cast(str, body.get("model", ""))
    raw_messages: list[dict[str, Any]] = cast(list[dict[str, Any]], body.get("messages", []) or [])

    raw_extras: dict[str, Any] = {}
    messages = OpenAIChatAdapter.load_messages(
        cast(list[ChatCompletionMessageParam], raw_messages),
        raw_extras=raw_extras,
    )

    raw_tools = cast(list[Any], body.get("tools", []) or [])
    function_tools = _openai_parse_tools(raw_tools)
    settings = _openai_parse_settings(body)
    request_parameters = ModelRequestParameters(function_tools=function_tools)

    if "tool_choice" in body:
        raw_extras["tool_choice"] = body["tool_choice"]
    if "response_format" in body:
        raw_extras["response_format"] = body["response_format"]

    for key, value in body.items():
        if key in _OPENAI_ABSORBED:
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


def _render_openai_chat(parsed: ParsedRequest) -> bytes:
    settings_dict = cast(dict[str, Any], parsed.settings)
    messages = OpenAIChatAdapter.dump_messages(parsed.messages)

    body: dict[str, Any] = {
        "model": parsed.model,
        "messages": messages,
    }
    _openai_apply_settings(body, settings_dict)

    tools = _openai_format_tools(parsed.request_parameters.function_tools)
    if tools:
        body["tools"] = tools

    _openai_stitch_raw_extras(body, parsed.raw_extras)

    if parsed.stream:
        body["stream"] = True

    return json.dumps(body, separators=(",", ":")).encode()
