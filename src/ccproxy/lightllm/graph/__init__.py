"""Pydantic-graph FSM dispatcher for streaming response transformations.

The response-side dispatchers :func:`dispatch_intake` and :func:`dispatch_render`
return per-provider async FSM instances; the persistent-loop bridge in
:class:`ccproxy.lightllm.graph.sse_pipeline.SSEPipeline` drives them from
mitmproxy's sync stream callable.

The request-side :func:`dispatch_dump_sync` routes all providers (Anthropic,
OpenAI, Google, Perplexity) to the new :mod:`ccproxy.lightllm.adapters`
``render`` classmethods. Each accepts an :class:`LLMRenderInput` (Protocol;
:class:`ccproxy.pipeline.context.Context` satisfies it).
"""

from typing import TYPE_CHECKING

from ccproxy.lightllm.graph.anthropic_intake import AnthropicResponseIntakeFSM
from ccproxy.lightllm.graph.anthropic_render import AnthropicResponseRenderFSM
from ccproxy.lightllm.graph.google_intake import GoogleResponseIntakeFSM
from ccproxy.lightllm.graph.openai_intake import OpenAIResponseIntakeFSM
from ccproxy.lightllm.graph.openai_render import OpenAIResponseRenderFSM
from ccproxy.lightllm.graph.perplexity_intake import PerplexityResponseIntakeFSM
from ccproxy.lightllm.parsed import ListenerFormat

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestParameters

    from ccproxy.lightllm.adapters import LLMRenderInput

__all__ = [
    "AnyAsyncIntakeFSM",
    "AnyAsyncRenderFSM",
    "UnsupportedListenerError",
    "UnsupportedUpstreamError",
    "dispatch_dump",
    "dispatch_dump_sync",
    "dispatch_intake",
    "dispatch_render",
]


_ANTHROPIC_COMPATIBLE = frozenset({"anthropic", "deepseek", "zai"})
_GOOGLE_COMPATIBLE = frozenset({"google", "gemini", "vertex_ai", "vertex_ai_beta"})


# Aliases for the union of all response-side FSM types. The Half-B
# :class:`SSEPipeline` types its ``intake`` / ``render`` parameters against
# these so any FSM the dispatchers can produce is acceptable.
AnyAsyncIntakeFSM = (
    AnthropicResponseIntakeFSM | OpenAIResponseIntakeFSM | GoogleResponseIntakeFSM | PerplexityResponseIntakeFSM
)
AnyAsyncRenderFSM = AnthropicResponseRenderFSM | OpenAIResponseRenderFSM


class UnsupportedUpstreamError(ValueError):
    """Raised when :func:`dispatch_dump` is asked to render to an unknown provider."""


class UnsupportedListenerError(ValueError):
    """Raised when :func:`dispatch_render` is asked for a listener format it doesn't know."""


async def dispatch_dump(req: "LLMRenderInput", *, provider: str) -> bytes:
    """Render ``req`` to the wire bytes the named upstream expects.

    All providers route through :func:`dispatch_dump_sync` (kept here for
    test compatibility with code that ``await``s the call).
    """
    return dispatch_dump_sync(req, provider=provider)


def dispatch_intake(
    *,
    upstream_provider: str,
    model: str,
    request_params: "ModelRequestParameters",
) -> AnyAsyncIntakeFSM:
    """Dispatch to the right per-upstream response intake FSM.

    Routes Anthropic-compatible providers (anthropic / deepseek / zai) to the
    Anthropic intake FSM, OpenAI to the OpenAI intake FSM, Google family
    (google / gemini / vertex_ai / vertex_ai_beta) to the Google intake FSM,
    and Perplexity Pro to its own intake FSM. Raises
    :class:`UnsupportedUpstreamError` for anything else — there's no fallback,
    because an unknown upstream means we have no idea how to parse its SSE.
    """
    if upstream_provider in _ANTHROPIC_COMPATIBLE:
        return AnthropicResponseIntakeFSM(model=model, request_params=request_params)
    if upstream_provider == "openai":
        return OpenAIResponseIntakeFSM(model=model, request_params=request_params)
    if upstream_provider in _GOOGLE_COMPATIBLE:
        return GoogleResponseIntakeFSM(model=model, request_params=request_params)
    if upstream_provider == "perplexity_pro":
        return PerplexityResponseIntakeFSM(model=model, request_params=request_params)
    raise UnsupportedUpstreamError(f"no response intake for upstream_provider={upstream_provider!r}")


def dispatch_render(*, listener_format: ListenerFormat, model: str = "unknown") -> AnyAsyncRenderFSM:
    """Dispatch to the right per-listener response render FSM.

    Routes ``ANTHROPIC_MESSAGES`` to the Anthropic render FSM and
    ``OPENAI_CHAT`` to the OpenAI render FSM. Raises
    :class:`UnsupportedListenerError` for ``UNKNOWN`` — there's no fallback,
    because an unknown listener format means we have no idea what wire
    shape to produce.
    """
    if listener_format is ListenerFormat.ANTHROPIC_MESSAGES:
        return AnthropicResponseRenderFSM(model=model)
    if listener_format is ListenerFormat.OPENAI_CHAT:
        return OpenAIResponseRenderFSM(model=model)
    raise UnsupportedListenerError(f"no response render for listener_format={listener_format}")


def dispatch_dump_sync(req: "LLMRenderInput", *, provider: str) -> bytes:
    """Synchronous outbound dispatcher.

    Routes :class:`LLMRenderInput` to the matching adapter's ``render``
    classmethod. Each adapter renders ``req``'s typed fields (messages,
    settings, raw_extras, request_parameters, model, stream) to wire bytes.
    """
    if provider in _ANTHROPIC_COMPATIBLE:
        from ccproxy.lightllm.adapters.anthropic import AnthropicAdapter

        return AnthropicAdapter.render(req)
    if provider == "openai":
        from ccproxy.lightllm.adapters.openai_chat import OpenAIChatAdapter

        return OpenAIChatAdapter.render(req)
    if provider in _GOOGLE_COMPATIBLE:
        from ccproxy.lightllm.adapters.google import GoogleAdapter

        return GoogleAdapter.render(req)
    if provider == "perplexity_pro":
        from ccproxy.lightllm.adapters.perplexity import PerplexityAdapter

        return PerplexityAdapter.render(req)

    raise UnsupportedUpstreamError(f"no outbound renderer for provider={provider!r}")
