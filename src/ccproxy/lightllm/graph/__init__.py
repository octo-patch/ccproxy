"""Pydantic-graph FSM dispatcher for IR ↔ wire transformations.

The FSM-based replacement for the per-provider outbound renderers and
per-listener inbound parsers in :mod:`ccproxy.lightllm`. Each provider has its
own ``*_dump.py`` (IR → wire bytes) and ``*_load.py`` (wire bytes → IR) module
implementing a small `pydantic-graph` state machine; the dispatchers here are
the public entry points the rest of ccproxy calls.

The internal nodes are :class:`pydantic_graph.BaseNode` subclasses with
``async def run(...)`` methods, driven via ``await graph.run(...)``. The
:func:`Context.parse_sync` / :func:`render_outbound_sync` worker-thread bridge
in :mod:`ccproxy.pipeline.context` and :mod:`ccproxy.lightllm.outbound` is the
async-to-sync boundary for mitmproxy addon hooks that must call this layer
synchronously.

The response-side dispatchers :func:`dispatch_intake` and
:func:`dispatch_render` mirror :func:`dispatch_load` and :func:`dispatch_dump`
on the wire-bytes → IR-events → wire-bytes path. They return the per-provider
async FSM instances directly; the persistent-loop bridge in
:class:`ccproxy.lightllm.graph.sse_pipeline.SSEPipeline` drives them from
mitmproxy's sync stream callable.
"""

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, Any

from ccproxy.lightllm.graph.anthropic_dump import render_anthropic_dump
from ccproxy.lightllm.graph.anthropic_intake import AnthropicResponseIntakeFSM
from ccproxy.lightllm.graph.anthropic_load import load_anthropic
from ccproxy.lightllm.graph.anthropic_render import AnthropicResponseRenderFSM
from ccproxy.lightllm.graph.google_dump import render_google_dump
from ccproxy.lightllm.graph.google_intake import GoogleResponseIntakeFSM
from ccproxy.lightllm.graph.openai_dump import render_openai_chat_dump
from ccproxy.lightllm.graph.openai_intake import OpenAIResponseIntakeFSM
from ccproxy.lightllm.graph.openai_load import load_openai_chat
from ccproxy.lightllm.graph.openai_render import OpenAIResponseRenderFSM
from ccproxy.lightllm.graph.perplexity_dump import render_perplexity_pro_dump
from ccproxy.lightllm.graph.perplexity_intake import PerplexityResponseIntakeFSM
from ccproxy.lightllm.parsed import ListenerFormat, ParsedRequest

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestParameters

__all__ = [
    "AnyAsyncIntakeFSM",
    "AnyAsyncRenderFSM",
    "UnsupportedListenerError",
    "UnsupportedUpstreamError",
    "dispatch_dump",
    "dispatch_dump_sync",
    "dispatch_intake",
    "dispatch_load",
    "dispatch_render",
    "load_anthropic",
    "load_openai_chat",
    "render_anthropic_dump",
    "render_google_dump",
    "render_openai_chat_dump",
    "render_perplexity_pro_dump",
]


_ANTHROPIC_COMPATIBLE = frozenset({"anthropic", "deepseek", "zai"})
_GOOGLE_COMPATIBLE = frozenset({"google", "gemini", "vertex_ai", "vertex_ai_beta"})


# Aliases for the union of all response-side FSM types. The Half-B
# :class:`SSEPipeline` types its ``intake`` / ``render`` parameters against
# these so any FSM the dispatchers can produce is acceptable.
AnyAsyncIntakeFSM = (
    AnthropicResponseIntakeFSM
    | OpenAIResponseIntakeFSM
    | GoogleResponseIntakeFSM
    | PerplexityResponseIntakeFSM
)
AnyAsyncRenderFSM = AnthropicResponseRenderFSM | OpenAIResponseRenderFSM


class UnsupportedUpstreamError(ValueError):
    """Raised when :func:`dispatch_dump` is asked to render to an unknown provider."""


class UnsupportedListenerError(ValueError):
    """Raised when :func:`dispatch_render` is asked for a listener format it doesn't know."""


async def dispatch_load(body: dict[str, Any], *, listener_format: ListenerFormat) -> ParsedRequest:
    """Dispatch to the right per-listener load function based on ``listener_format``."""
    if listener_format is ListenerFormat.ANTHROPIC_MESSAGES:
        return await load_anthropic(body)
    if listener_format is ListenerFormat.OPENAI_CHAT:
        return await load_openai_chat(body)
    raise ValueError(f"no IR parser for listener_format={listener_format}")


async def dispatch_dump(parsed: ParsedRequest, *, provider: str) -> bytes:
    """Render ``parsed`` to the wire bytes the named upstream expects.

    Anthropic-compatible providers and OpenAI route to the pydantic-graph
    FSM dumps. Google / Vertex AI / Perplexity Pro still route to the
    legacy renderers until Phase G lands their FSM dumps.
    """
    if provider in _ANTHROPIC_COMPATIBLE:
        return await render_anthropic_dump(parsed)
    if provider == "openai":
        return await render_openai_chat_dump(parsed)
    if provider in _GOOGLE_COMPATIBLE:
        return await render_google_dump(parsed)
    if provider == "perplexity_pro":
        return await render_perplexity_pro_dump(parsed)
    raise UnsupportedUpstreamError(f"no outbound renderer for provider={provider!r}")


def dispatch_intake(
    *,
    upstream_provider: str,
    model: str,
    request_params: "ModelRequestParameters",
) -> AnyAsyncIntakeFSM:
    """Dispatch to the right per-upstream response intake FSM.

    Mirrors :func:`dispatch_dump` on the response side: routes
    Anthropic-compatible providers (anthropic / deepseek / zai) to the
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
    raise UnsupportedUpstreamError(
        f"no response intake for upstream_provider={upstream_provider!r}"
    )


def dispatch_render(
    *, listener_format: ListenerFormat, model: str = "unknown"
) -> AnyAsyncRenderFSM:
    """Dispatch to the right per-listener response render FSM.

    Mirrors :func:`dispatch_load` on the response side: routes
    ``ANTHROPIC_MESSAGES`` to the Anthropic render FSM and ``OPENAI_CHAT`` to
    the OpenAI render FSM. Raises :class:`UnsupportedListenerError` for
    ``UNKNOWN`` — there's no fallback, because an unknown listener format
    means we have no idea what wire shape to produce.
    """
    if listener_format is ListenerFormat.ANTHROPIC_MESSAGES:
        return AnthropicResponseRenderFSM(model=model)
    if listener_format is ListenerFormat.OPENAI_CHAT:
        return OpenAIResponseRenderFSM(model=model)
    raise UnsupportedListenerError(
        f"no response render for listener_format={listener_format}"
    )


def dispatch_dump_sync(parsed: ParsedRequest, *, provider: str) -> bytes:
    """Sync facade over :func:`dispatch_dump` — keeps the worker-thread bridge alive.

    The bridge is required because pydantic-graph's ``Graph.run_sync`` is
    deprecated and uses ``loop.run_until_complete`` under the hood — calling
    that from inside mitmproxy's already-running asyncio loop raises
    ``RuntimeError: This event loop is already running``. Identical pattern to
    :func:`ccproxy.pipeline.context.Context._run_coro_sync` (commit
    ``016d7d1``) and the legacy
    :func:`ccproxy.lightllm.outbound.render_outbound_sync`.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(dispatch_dump(parsed, provider=provider))
        finally:
            loop.close()

    def _worker() -> bytes:
        worker_loop = asyncio.new_event_loop()
        try:
            return worker_loop.run_until_complete(dispatch_dump(parsed, provider=provider))
        finally:
            worker_loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_worker).result()
