"""Wire-format-neutral view of an incoming request and an outgoing response.

``ParsedRequest`` is what a per-listener inbound parser produces. It carries
pydantic-ai's IR objects (``ModelMessage``, ``ModelRequestParameters``,
``ModelSettings``) plus the model name and the stream flag. ``raw_extras``
preserves any wire fields the IR doesn't absorb, so passthrough rendering
can stitch them back into the outbound wire body.

``ParsedResponse`` is the symmetric envelope on the response side: a
per-upstream-provider response intake produces it from a buffered response
body, and a per-listener-format response renderer consumes it. Streaming
responses don't ride this envelope — they flow as a chunk-fed
``AsyncIterator[ModelResponseStreamEvent]`` between the intake FSM and the
render FSM directly.

``ListenerFormat`` enumerates the listener-side wire formats ccproxy
accepts. Determined by path/headers in ``Context.from_flow``; selects the
matching inbound parser and (later) the matching response renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings


class ListenerFormat(str, Enum):
    UNKNOWN = "unknown"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    OPENAI_CHAT = "openai_chat"


@dataclass(frozen=True)
class ParsedRequest:
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


@dataclass(frozen=True)
class ParsedResponse:
    model: str
    """Model name as reported by the upstream response body."""

    response: ModelResponse
    """Assistant turn as a pydantic-ai IR ``ModelResponse`` (text/tool_call/thinking parts, usage, ...)."""

    stream: bool = False
    """Whether the upstream response was streamed (``True``) or buffered (``False``)."""

    raw_extras: dict[str, Any] = field(default_factory=dict)
    """Provider-side response fields the IR doesn't absorb — preserved for passthrough rendering.

    Mirrors :attr:`ParsedRequest.raw_extras`. Conventional keys on the response side:
    ``usage:msg:N`` (per-message usage delta), ``safety:msg:N:rating:M`` (Gemini safety),
    ``citations:msg:N`` (Perplexity), ``unknown_event:msg:N:event:K`` (unrecognized event).
    """
