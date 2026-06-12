"""Inbound-format enum and the :class:`ParsedRequest` test-only bundle.

``InboundFormat`` enumerates the listener-side wire formats ccproxy
accepts. Determined by path/headers in ``Context.from_flow``; selects the
matching inbound parser and the matching response renderer.

``ParsedRequest`` is a frozen-dataclass implementation of
:class:`ccproxy.lightllm.adapters.LLMRenderInput`. All production code
(including the inspector) uses :class:`ccproxy.pipeline.context.Context`
directly via :meth:`Context.parse_sync`, which calls
:func:`ccproxy.lightllm.adapters._envelope.parse_request_into_fields`
to populate the lazy-parse slots in place. ``ParsedRequest`` survives
only as a simple no-mitmproxy-flow stub for unit-testing adapters and
dispatchers — the :func:`parse_request` / :func:`render_request`
convenience wrappers in ``_envelope`` are the test-fixture entry points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type RawExtras = dict[str, JsonValue]


class InboundFormat(StrEnum):
    UNKNOWN = "unknown"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"


@dataclass(frozen=True)
class ParsedRequest:
    """Frozen-dataclass :class:`LLMRenderInput` implementation.

    Satisfies the same Protocol Context does; used by adapter and
    dispatcher unit tests as a simple no-mitmproxy-flow stub. Production
    (including the inspector) goes through Context directly.
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

    raw_extras: RawExtras = field(default_factory=dict)
    """Wire fields not absorbed into the IR — preserved for passthrough rendering."""
