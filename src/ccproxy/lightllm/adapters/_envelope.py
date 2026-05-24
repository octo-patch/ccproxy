"""Wire-body parsing into typed IR fields.

Companion to the four ``UIAdapter`` subclasses
(:class:`AnthropicAdapter`, :class:`OpenAIChatAdapter`,
:class:`GoogleAdapter`, :class:`PerplexityAdapter`). Each listener-format
parser destructures a wire JSON body into a tuple of the IR fields
(messages, request_parameters, settings, raw_extras) that
:class:`ccproxy.pipeline.context.Context` and :class:`ParsedRequest`
share.

The render side lives on the adapters themselves —
:meth:`AnthropicAdapter.render` and :meth:`OpenAIChatAdapter.render` take
:class:`~ccproxy.lightllm.adapters.LLMRenderInput` (the Protocol Context
satisfies) and return wire bytes directly.

:func:`parse_request` and :func:`render_request` are thin test-fixture
wrappers around :func:`parse_request_into_fields`; production code uses
:meth:`Context.parse_sync` and :func:`dispatch_dump_sync` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from openai.types.chat import ChatCompletionMessageParam
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings

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
    _parse_system as _anthropic_parse_system,
)
from ccproxy.lightllm.adapters._anthropic_envelope import (
    _parse_tools as _anthropic_parse_tools,
)
from ccproxy.lightllm.adapters._openai_envelope import (
    _ABSORBED_BODY_KEYS as _OPENAI_ABSORBED,
)
from ccproxy.lightllm.adapters._openai_envelope import (
    _parse_settings as _openai_parse_settings,
)
from ccproxy.lightllm.adapters._openai_envelope import (
    _parse_tools as _openai_parse_tools,
)
from ccproxy.lightllm.adapters._openai_responses_envelope import (
    _ABSORBED_TOP_LEVEL as _RESPONSES_ABSORBED,
)
from ccproxy.lightllm.adapters._openai_responses_envelope import (
    _parse_responses_settings,
)
from ccproxy.lightllm.adapters.anthropic import AnthropicAdapter
from ccproxy.lightllm.adapters.openai_chat import OpenAIChatAdapter
from ccproxy.lightllm.adapters.openai_responses import OpenAIResponsesAdapter
from ccproxy.lightllm.parsed import InboundFormat, ParsedRequest

if TYPE_CHECKING:
    from ccproxy.pipeline.context import Context


@dataclass(frozen=True)
class _ParsedFields:
    """Bundle of IR fields produced by a listener-format parser."""

    messages: list[ModelMessage]
    request_parameters: ModelRequestParameters
    settings: ModelSettings
    raw_extras: dict[str, Any]


def parse_request_into_fields(
    *,
    body: dict[str, Any],
    inbound_format: InboundFormat,
    ctx: Context,
) -> None:
    """Parse ``body`` and populate ``ctx``'s lazy-parsed slots."""
    fields = _parse_fields(body=body, inbound_format=inbound_format)
    ctx._cached_messages = fields.messages
    ctx._cached_request_parameters = fields.request_parameters
    ctx._cached_settings = fields.settings
    ctx._cached_raw_extras = fields.raw_extras


def parse_request(body: dict[str, Any], *, inbound_format: InboundFormat) -> ParsedRequest:
    """Parse ``body`` into a :class:`ParsedRequest` bundle.

    Test-fixture convenience wrapper. Production code (including the
    inspector) uses :meth:`Context.parse_sync` which routes through
    :func:`parse_request_into_fields` to populate Context's lazy-parse
    slots in place.
    """
    fields = _parse_fields(body=body, inbound_format=inbound_format)
    return ParsedRequest(
        model=str(body.get("model", "")),
        messages=fields.messages,
        request_parameters=fields.request_parameters,
        settings=fields.settings,
        stream=bool(body.get("stream", False)),
        raw_extras=fields.raw_extras,
    )


def render_request(parsed: ParsedRequest, *, inbound_format: InboundFormat) -> bytes:
    """Render a :class:`ParsedRequest` to wire bytes via the matching adapter.

    Test-fixture convenience wrapper. Production
    code routes through :func:`ccproxy.lightllm.graph.dispatch_dump_sync`
    with a :class:`~ccproxy.pipeline.context.Context`.
    """
    if inbound_format is InboundFormat.ANTHROPIC_MESSAGES:
        return AnthropicAdapter.render(parsed)
    if inbound_format is InboundFormat.OPENAI_CHAT:
        return OpenAIChatAdapter.render(parsed)
    if inbound_format is InboundFormat.OPENAI_RESPONSES:
        return OpenAIResponsesAdapter.render(parsed)
    raise ValueError(f"no IR renderer for inbound_format={inbound_format}")


def _parse_fields(*, body: dict[str, Any], inbound_format: InboundFormat) -> _ParsedFields:
    if inbound_format is InboundFormat.ANTHROPIC_MESSAGES:
        return _parse_anthropic(body)
    if inbound_format is InboundFormat.OPENAI_CHAT:
        return _parse_openai_chat(body)
    if inbound_format is InboundFormat.OPENAI_RESPONSES:
        return _parse_openai_responses(body)
    raise ValueError(f"no IR parser for inbound_format={inbound_format}")


# ── Anthropic ───────────────────────────────────────────────────────────────


def _parse_anthropic(body: dict[str, Any]) -> _ParsedFields:
    raw_extras: dict[str, Any] = {}

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

    system_parts = _anthropic_parse_system(body.get("system"), settings=settings, raw_extras=raw_extras)
    if system_parts:
        messages = _anthropic_attach_system_prompts(messages, system_parts)

    for key, value in body.items():
        if key in _ANTHROPIC_ABSORBED:
            continue
        raw_extras.setdefault(key, value)

    return _ParsedFields(
        messages=messages,
        request_parameters=request_parameters,
        settings=settings,
        raw_extras=raw_extras,
    )


# ── OpenAI Chat Completions ─────────────────────────────────────────────────


def _parse_openai_chat(body: dict[str, Any]) -> _ParsedFields:
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

    return _ParsedFields(
        messages=messages,
        request_parameters=request_parameters,
        settings=settings,
        raw_extras=raw_extras,
    )


# ── OpenAI Responses ────────────────────────────────────────────────────────


def _parse_openai_responses(body: dict[str, Any]) -> _ParsedFields:
    """Parse a ``/v1/responses`` request body into typed IR fields.

    Handles the bare-string ``input`` shorthand by wrapping into a
    single user message. Tools share the Chat shape, so we reuse
    :func:`_openai_parse_tools`. Settings use Responses-specific
    naming (``max_output_tokens`` vs Chat's ``max_completion_tokens``)
    so a dedicated :func:`_parse_responses_settings` runs.
    """
    raw_input: Any = body.get("input")
    if isinstance(raw_input, str):
        input_items: list[Any] = (
            [{"type": "message", "role": "user", "content": raw_input}] if raw_input else []
        )
    elif isinstance(raw_input, list):
        input_items = list(raw_input)
    else:
        input_items = []

    raw_extras: dict[str, Any] = {}
    messages = OpenAIResponsesAdapter.load_messages(
        input_items,
        instructions=body.get("instructions"),
        raw_extras=raw_extras,
    )

    raw_tools = cast(list[Any], body.get("tools", []) or [])
    function_tools = _openai_parse_tools(raw_tools)
    settings = _parse_responses_settings(body)
    request_parameters = ModelRequestParameters(function_tools=function_tools)

    if "tool_choice" in body:
        raw_extras["tool_choice"] = body["tool_choice"]

    for key, value in body.items():
        if key in _RESPONSES_ABSORBED:
            continue
        if key in raw_extras:
            continue
        raw_extras[key] = value

    return _ParsedFields(
        messages=messages,
        request_parameters=request_parameters,
        settings=settings,
        raw_extras=raw_extras,
    )
