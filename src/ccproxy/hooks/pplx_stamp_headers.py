"""Stamp Perplexity Pro's required browser-shape headers on the outbound flow.

Perplexity's ``/rest/sse/perplexity_ask`` authenticates via a
``__Secure-next-auth.session-token`` cookie (Pro subscription), not via the
default ``Authorization: Bearer`` header that :mod:`inject_auth` injects.
Pre-refactor, ``PerplexityProConfig.validate_environment`` (a litellm
``BaseConfig`` hook) stamped the cookie and the Chrome-shape sibling
headers (``User-Agent``, ``Origin``, ``Referer``, ``x-perplexity-*``,
``x-app-api*``, ``sec-fetch-*``) on every request. The pydantic-graph FSM
migration removed litellm and with it that step — this hook re-implements
it as an outbound DAG entry.

Runs after :mod:`inject_auth` (which stamps ``ctx.metadata.auth_provider``
and writes the placeholder ``Authorization`` header) and before
:mod:`pplx_preflight`. The ``Authorization`` header is cleared
once the Cookie equivalent is in place — leaking the sentinel-resolved header
to Perplexity would expose the sentinel-resolution surface and risks
Cloudflare scrutiny.

The hook is best-effort with respect to its own work: a missing token logs
DEBUG and returns ``ctx`` unchanged so the request still reaches the
upstream and surfaces the auth failure end-to-end rather than silently
short-circuiting here.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from ccproxy.config import get_config
from ccproxy.lightllm.pplx import (
    PERPLEXITY_API_VERSION,
    PERPLEXITY_BROWSER_UA,
    PERPLEXITY_PROVIDER_NAME,
    PERPLEXITY_SESSION_COOKIE,
    PERPLEXITY_URL_BASE,
)
from ccproxy.pipeline.hook import hook

if TYPE_CHECKING:
    from ccproxy.pipeline.context import Context

logger = logging.getLogger(__name__)

__all__ = ["pplx_stamp_headers", "pplx_stamp_headers_guard"]


def pplx_stamp_headers_guard(ctx: Context) -> bool:
    """Run only when inject_auth resolved the Perplexity sentinel."""
    return ctx.metadata.auth_provider == PERPLEXITY_PROVIDER_NAME


@hook(reads=[], writes=[])
def pplx_stamp_headers(ctx: Context, _: dict[str, Any]) -> Context:
    """Replace ``Authorization: Bearer`` with the Perplexity Pro browser-shape headers.

    Drops the ``Authorization`` header set by :mod:`inject_auth` and
    stamps the Chrome-shape cookie-auth bundle Perplexity's WebUI expects.
    """
    config = get_config()
    token = config.resolve_auth_token(PERPLEXITY_PROVIDER_NAME)
    if not token:
        logger.debug("pplx_stamp_headers: no session token resolved; skipping")
        return ctx

    ctx.set_header("Cookie", f"{PERPLEXITY_SESSION_COOKIE}={token}")
    ctx.set_header("User-Agent", PERPLEXITY_BROWSER_UA)
    ctx.set_header("Origin", PERPLEXITY_URL_BASE)
    ctx.set_header("Referer", f"{PERPLEXITY_URL_BASE}/")
    ctx.set_header("Accept", "text/event-stream, application/json")
    ctx.set_header("Content-Type", "application/json")
    ctx.set_header("x-perplexity-request-reason", "perplexity-query-state-provider")
    ctx.set_header("x-app-apiversion", PERPLEXITY_API_VERSION)
    ctx.set_header("x-app-apiclient", "default")
    ctx.set_header("x-request-id", str(uuid.uuid4()))
    ctx.set_header("sec-fetch-dest", "empty")
    ctx.set_header("sec-fetch-mode", "cors")
    ctx.set_header("sec-fetch-site", "same-origin")
    # Drop the placeholder Authorization header so Perplexity sees a clean
    # browser-shape request — leaking the sentinel-resolution
    # surface risks Cloudflare scrutiny.
    ctx.set_header("Authorization", "")
    return ctx
