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
"""

import asyncio
import concurrent.futures
from typing import Any

from ccproxy.lightllm.graph.anthropic_dump import render_anthropic_dump
from ccproxy.lightllm.graph.anthropic_load import load_anthropic
from ccproxy.lightllm.graph.google_dump import render_google_dump
from ccproxy.lightllm.graph.openai_dump import render_openai_chat_dump
from ccproxy.lightllm.graph.openai_load import load_openai_chat
from ccproxy.lightllm.graph.perplexity_dump import render_perplexity_pro_dump
from ccproxy.lightllm.parsed import ListenerFormat, ParsedRequest

__all__ = [
    "dispatch_dump",
    "dispatch_dump_sync",
    "dispatch_load",
    "load_anthropic",
    "load_openai_chat",
    "render_anthropic_dump",
    "render_google_dump",
    "render_openai_chat_dump",
    "render_perplexity_pro_dump",
]


_ANTHROPIC_COMPATIBLE = frozenset({"anthropic", "deepseek", "zai"})
_GOOGLE_COMPATIBLE = frozenset({"google", "gemini", "vertex_ai"})


class UnsupportedUpstreamError(ValueError):
    """Raised when :func:`dispatch_dump` is asked to render to an unknown provider."""


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
