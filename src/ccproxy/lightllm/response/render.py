"""Per-listener-format IR-event → wire-bytes sync renderer contract.

A ``ResponseRender`` consumes ``ModelResponseStreamEvent`` IR objects
emitted by a ``ResponseIntake`` and produces wire bytes in the
listener-side format. Exhaustive pattern-match on the event union with
``assert_never`` for the default case ensures missing variants surface
at type-check time.

Concrete implementations:

  ``render_anthropic`` — IR → Anthropic Messages SSE wire
  ``render_openai``    — IR → OpenAI Chat Completion SSE wire
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ccproxy.lightllm.parsed import ListenerFormat

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelResponseStreamEvent


@runtime_checkable
class ResponseRender(Protocol):
    """Sync renderer: IR events → listener-format wire bytes."""

    name: str

    def render(self, event: ModelResponseStreamEvent) -> bytes:
        """One IR event → zero-or-more bytes of listener wire output."""
        ...

    def close(self) -> bytes:
        """Stream end. Emit format-specific terminator (e.g. ``message_stop`` / ``data: [DONE]``)."""
        ...


class UnsupportedListenerError(ValueError):
    """Raised when ``select_render`` is asked for a listener format it doesn't know."""


def select_render(listener_format: ListenerFormat) -> ResponseRender:
    """Pick the right renderer by listener wire format."""
    if listener_format is ListenerFormat.ANTHROPIC_MESSAGES:
        from ccproxy.lightllm.response.render_anthropic import AnthropicResponseRender

        return AnthropicResponseRender()
    if listener_format is ListenerFormat.OPENAI_CHAT:
        from ccproxy.lightllm.response.render_openai import OpenAIResponseRender

        return OpenAIResponseRender()
    raise UnsupportedListenerError(f"no response render for listener_format={listener_format}")
