"""Wire-format-neutral view of an incoming request.

``ParsedRequest`` is what a per-listener inbound parser produces. It carries
pydantic-ai's IR objects (``ModelMessage``, ``ModelRequestParameters``,
``ModelSettings``) plus the model name and the stream flag. ``raw_extras``
preserves any wire fields the IR doesn't absorb, so passthrough rendering
can stitch them back into the outbound wire body.

``ListenerFormat`` enumerates the listener-side wire formats ccproxy
accepts. Determined by path/headers in ``Context.from_flow``; selects the
matching inbound parser and (later) the matching response renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic_ai.messages import ModelMessage
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
