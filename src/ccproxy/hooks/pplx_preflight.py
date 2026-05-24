"""Pre-flight ``GET /search/new`` before each Perplexity ask request.

Per ``core-query.md:80-141`` the Perplexity backend wants every
``/rest/sse/perplexity_ask`` call preceded by a GET to ``/search/new`` to
initialize a search session — without it the SSE stream may return silently
with no results. This hook runs in the outbound DAG after the transform
router has built the Perplexity wire payload (so ``query_str`` is available
on ``ctx._body``).

Best-effort: any failure is logged as a warning, the main request still
proceeds. The preflight URL is the only place ccproxy needs to send a
``GET`` with the session cookie outside the main SSE call — minimal
headers per the docs (omit Content-Type and ``Accept: text/event-stream``;
those trigger Cloudflare scrutiny).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

from ccproxy.config import get_config
from ccproxy.lightllm.pplx import (
    PERPLEXITY_BROWSER_UA,
    PERPLEXITY_PREFLIGHT_URL,
    PERPLEXITY_PROVIDER_NAME,
    PERPLEXITY_SESSION_COOKIE,
    PERPLEXITY_URL_BASE,
)
from ccproxy.pipeline.hook import hook

if TYPE_CHECKING:
    from ccproxy.pipeline.context import Context

logger = logging.getLogger(__name__)

__all__ = ["pplx_preflight", "pplx_preflight_guard"]


def pplx_preflight_guard(ctx: Context) -> bool:
    """Run only when forward_oauth resolved the Perplexity sentinel."""
    assert ctx.flow is not None
    return ctx.flow.metadata.get("ccproxy.oauth_provider") == PERPLEXITY_PROVIDER_NAME


@hook(reads=["query_str"], writes=[])
def pplx_preflight(ctx: Context, _: dict[str, Any]) -> Context:
    """Fire ``GET /search/new`` with the complete ``query_str`` as a warm-up.

    Failures are warned-and-swallowed: the main ``perplexity_ask`` proceeds
    regardless. The preflight's success state is stamped on
    ``flow.metadata["ccproxy.pplx.preflight"]`` for observability.
    """
    assert ctx.flow is not None
    body = ctx._body if isinstance(ctx._body, dict) else {}
    query = body.get("query_str")
    if not isinstance(query, str) or not query:
        return ctx

    config = get_config()
    token = config.resolve_oauth_token(PERPLEXITY_PROVIDER_NAME)
    if not token:
        logger.debug("pplx_preflight: no session token available; skipping")
        return ctx
    preflight_config = config.pplx.search

    try:
        httpx.get(
            PERPLEXITY_PREFLIGHT_URL,
            params={"q": query},
            headers={
                "Cookie": f"{PERPLEXITY_SESSION_COOKIE}={token}",
                "User-Agent": PERPLEXITY_BROWSER_UA,
                "Referer": f"{PERPLEXITY_URL_BASE}/",
                "Origin": PERPLEXITY_URL_BASE,
                "Accept": "application/json",
            },
            timeout=preflight_config.preflight_timeout_seconds,
            follow_redirects=True,
        )
        ctx.flow.metadata["ccproxy.pplx.preflight"] = True
    except Exception:
        logger.warning("pplx_preflight: side request failed", exc_info=True)
        ctx.flow.metadata["ccproxy.pplx.preflight"] = False
    return ctx
