"""Listener-format enum and the :class:`ParsedRequest` test/test-helper bundle.

``ListenerFormat`` enumerates the listener-side wire formats ccproxy
accepts. Determined by path/headers in ``Context.from_flow``; selects the
matching inbound parser and the matching response renderer.

``ParsedRequest`` is a frozen-dataclass implementation of
:class:`ccproxy.lightllm.adapters.LLMRenderInput`. Production code uses
:class:`ccproxy.pipeline.context.Context` directly (it satisfies the same
Protocol). ``ParsedRequest`` survives because tests construct it as a
simple, no-mitmproxy-flow stub for unit-testing adapters and dispatchers.
The inspector flow-enrichment path also uses it via the
:func:`ccproxy.lightllm.adapters._envelope.parse_request` convenience
wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings


class ListenerFormat(StrEnum):
    UNKNOWN = "unknown"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    OPENAI_CHAT = "openai_chat"


@dataclass(frozen=True)
class ParsedRequest:
    """Frozen-dataclass :class:`LLMRenderInput` implementation.

    Satisfies the same Protocol Context does; useful for unit tests and
    the inspector flow-enrichment path. Production hot path goes through
    Context directly.
    """

    model: str
    """Model name as declared in the listener wire body."""

    messages: list[ModelMessage]
    """Conversation history as pydantic-ai IR messages."""

    request_parameters: ModelRequestParameters
    """Tools, output config, native-tool selection."""

    settings: ModelSettings
    """Sampling + behavior settings (TypedDict at runtime)."""

    stream: bool = False
    """Whether the listener requested SSE streaming."""

    raw_extras: dict[str, Any] = field(default_factory=dict)
    """Wire fields not absorbed into the IR — preserved for passthrough rendering."""
