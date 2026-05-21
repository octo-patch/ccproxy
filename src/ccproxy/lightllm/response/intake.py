"""Per-upstream-vendor SSE-bytes → IR event sync dispatcher contract.

A ``ResponseIntake`` is constructed once per response stream. It
buffers incoming bytes, frames SSE events, parses each event payload
into the vendor's pydantic event union (e.g. ``BetaRawMessageStreamEvent``),
and drives pydantic-ai's ``ModelResponsePartsManager`` synchronously
to emit ``ModelResponseStreamEvent`` IR objects.

Concrete implementations live alongside this module:

  ``intake_anthropic`` — Anthropic Messages SSE → IR
  ``intake_openai``    — OpenAI Chat Completion SSE → IR
  ``intake_google``    — Google streamGenerateContent → IR
  ``intake_perplexity``— Perplexity Pro SSE → IR (no pydantic-ai equivalent)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelResponseStreamEvent
    from pydantic_ai.models import ModelRequestParameters


@runtime_checkable
class ResponseIntake(Protocol):
    """Sync dispatcher: raw upstream SSE bytes → pydantic-ai IR events.

    Stateful. ``feed`` is called repeatedly as bytes arrive; ``close``
    is called once when the upstream stream ends.
    """

    name: str
    upstream_raw_bytes: bytearray
    """Cumulative tee of every byte fed in — for inspectors like pplx_addon."""

    def feed(self, data: bytes) -> Iterator[ModelResponseStreamEvent]:
        """Process incoming bytes; yield zero-or-more IR events."""
        ...

    def close(self) -> Iterator[ModelResponseStreamEvent]:
        """Stream end. May yield trailing events (e.g. PartEndEvent for unclosed blocks)."""
        ...


class UnsupportedUpstreamError(ValueError):
    """Raised when ``select_intake`` is asked for an upstream provider it doesn't know."""


def select_intake(
    *, upstream_provider: str, model: str, request_params: ModelRequestParameters
) -> ResponseIntake:
    """Pick the right intake by upstream provider name."""
    if upstream_provider in ("anthropic", "deepseek", "zai"):
        from ccproxy.lightllm.response.intake_anthropic import AnthropicResponseIntake

        return AnthropicResponseIntake(model=model, request_params=request_params)
    if upstream_provider == "openai":
        from ccproxy.lightllm.response.intake_openai import OpenAIResponseIntake

        return OpenAIResponseIntake(model=model, request_params=request_params)
    if upstream_provider in ("google", "gemini", "vertex_ai"):
        from ccproxy.lightllm.response.intake_google import GoogleResponseIntake

        return GoogleResponseIntake(model=model, request_params=request_params)
    if upstream_provider == "perplexity_pro":
        from ccproxy.lightllm.response.intake_perplexity import PerplexityResponseIntake

        return PerplexityResponseIntake(model=model, request_params=request_params)
    raise UnsupportedUpstreamError(f"no response intake for upstream_provider={upstream_provider!r}")
