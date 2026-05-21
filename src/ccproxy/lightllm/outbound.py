"""Outbound dispatcher: route ``ParsedRequest`` to the right upstream renderer.

The four per-provider renderers each take a ``ParsedRequest`` (IR plus
``raw_extras``) and emit upstream wire bytes. This module picks the right
one by provider name — typically the value of ``Provider.provider`` from
the ccproxy config, set by the transform router via sentinel lookup.

Provider names match the existing config strings:

    ``anthropic``        → ``render_anthropic``
    ``openai``           → ``render_openai_chat``
    ``google`` / ``gemini`` → ``render_google``
    ``perplexity_pro``   → ``render_perplexity_pro``

Other provider strings (``deepseek``, ``zai`` — Anthropic-compatible
forks) route to the Anthropic renderer with the same kwargs; the actual
upstream URL is handled separately by the transform router via
``Provider.host``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ccproxy.lightllm.outbound_anthropic import render_anthropic
from ccproxy.lightllm.outbound_google import render_google
from ccproxy.lightllm.outbound_openai import render_openai_chat
from ccproxy.lightllm.outbound_perplexity import render_perplexity_pro

if TYPE_CHECKING:
    from ccproxy.lightllm.parsed import ParsedRequest


_ANTHROPIC_COMPATIBLE = frozenset({"anthropic", "deepseek", "zai"})
_GOOGLE_COMPATIBLE = frozenset({"google", "gemini", "vertex_ai"})


class UnsupportedUpstreamError(ValueError):
    """Raised when ``render_outbound`` is asked to render to a provider it doesn't know."""


async def render_outbound(parsed: ParsedRequest, *, provider: str) -> bytes:
    """Render ``parsed`` to the wire bytes the named upstream expects."""
    if provider in _ANTHROPIC_COMPATIBLE:
        return await render_anthropic(parsed)
    if provider == "openai":
        return await render_openai_chat(parsed)
    if provider in _GOOGLE_COMPATIBLE:
        return await render_google(parsed)
    if provider == "perplexity_pro":
        return await render_perplexity_pro(parsed)
    raise UnsupportedUpstreamError(f"no outbound renderer for provider={provider!r}")


def render_outbound_sync(parsed: ParsedRequest, *, provider: str) -> bytes:
    """Sync facade over :func:`render_outbound`.

    Drives the async renderer to completion. From outside any event loop
    we run on a private loop on the calling thread. From inside a
    running loop (e.g. a sync hook body invoked by mitmproxy's async
    runtime) we dispatch to a worker thread that owns its own loop —
    asyncio forbids nested ``run_until_complete`` calls in the same
    thread. Safe because the renderers raise ``CaptureSentinel`` before
    any real I/O.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(render_outbound(parsed, provider=provider))
        finally:
            loop.close()
    import concurrent.futures

    def _worker() -> bytes:
        worker_loop = asyncio.new_event_loop()
        try:
            return worker_loop.run_until_complete(render_outbound(parsed, provider=provider))
        finally:
            worker_loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_worker).result()
