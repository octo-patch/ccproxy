"""Synthetic ``GET /pplx/messages/<session_id>`` handler.

Converts a Perplexity thread (fetched via ccproxy's session cookie) into
OpenAI-shaped ``messages[]`` for session resume. Registered as a REQUEST
route at higher priority than ``register_transform_routes`` so the
transform router doesn't try to forward ``/pplx/...`` to a provider.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from ccproxy.lightllm.pplx import (
    PERPLEXITY_BLOCK_USE_CASES,
    PERPLEXITY_BROWSER_UA,
    PERPLEXITY_PROVIDER_NAME,
    PERPLEXITY_SESSION_COOKIE,
    PERPLEXITY_URL_BASE,
    _thread_to_openai_messages,
)

if TYPE_CHECKING:
    from mitmproxy.http import HTTPFlow

    from ccproxy.inspector.router import InspectorRouter

logger = logging.getLogger(__name__)


def register_pplx_routes(router: InspectorRouter) -> None:
    """Register ``GET /pplx/messages/<session_id>`` on ``router``."""
    from mitmproxy.proxy.mode_specs import ReverseMode

    from ccproxy.config import get_config
    from ccproxy.inspector.router import RouteType

    cfg = get_config()
    mcp_auth = cfg.mcp.http.auth
    expected_token: str | None = None
    if mcp_auth is not None:
        if isinstance(mcp_auth, str):
            expected_token = mcp_auth
        else:
            expected_token = mcp_auth.resolve("pplx messages endpoint bearer token")

    @router.route("/pplx/messages/<session_id>", rtype=RouteType.REQUEST, catch_error=False)
    def handle_pplx_messages(flow: HTTPFlow, session_id: str, **_kwargs: object) -> None:  # pyright: ignore[reportUnusedFunction]
        if not isinstance(flow.client_conn.proxy_mode, ReverseMode):
            return
        if flow.request.method != "GET":
            return

        from mitmproxy.http import Response

        # Auth
        if expected_token is not None:
            auth_header = flow.request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer ") or auth_header[7:] != expected_token:
                flow.response = Response.make(
                    401,
                    json.dumps({"error": {"message": "unauthorized", "type": "auth_error", "code": 401}}).encode(),
                    {"Content-Type": "application/json"},
                )
                return

        # Provider check
        session_cfg = get_config()
        if PERPLEXITY_PROVIDER_NAME not in session_cfg.providers:
            flow.response = Response.make(
                503,
                json.dumps(
                    {
                        "error": {
                            "message": f"provider {PERPLEXITY_PROVIDER_NAME!r} not configured",
                            "type": "pplx_unavailable",
                            "code": 503,
                        }
                    }
                ).encode(),
                {"Content-Type": "application/json"},
            )
            return

        token = session_cfg.resolve_oauth_token(PERPLEXITY_PROVIDER_NAME)
        if not token:
            flow.response = Response.make(
                503,
                json.dumps(
                    {
                        "error": {
                            "message": f"no session cookie resolved for {PERPLEXITY_PROVIDER_NAME!r}",
                            "type": "pplx_unavailable",
                            "code": 503,
                        }
                    }
                ).encode(),
                {"Content-Type": "application/json"},
            )
            return

        # Fetch thread from Perplexity
        import httpx

        params: list[tuple[str, str | int | float | None]] = [
            ("version", "2.18"),
            ("source", "default"),
            ("limit", "100"),
            ("offset", "0"),
            ("from_first", "true"),
            ("with_parent_info", "true"),
            ("with_schematized_response", "true"),
        ]
        params.extend(("supported_block_use_cases", uc) for uc in PERPLEXITY_BLOCK_USE_CASES)

        headers = {
            "Cookie": f"{PERPLEXITY_SESSION_COOKIE}={token}",
            "User-Agent": PERPLEXITY_BROWSER_UA,
            "Origin": PERPLEXITY_URL_BASE,
            "Referer": f"{PERPLEXITY_URL_BASE}/",
            "Accept": "application/json",
            "x-app-apiclient": "default",
            "x-app-apiversion": "2.18",
            "x-perplexity-request-reason": "perplexity-query-state-provider",
            "x-perplexity-request-endpoint": f"{PERPLEXITY_URL_BASE}/rest/thread/{session_id}",
        }

        try:
            resp = httpx.get(
                f"{PERPLEXITY_URL_BASE}/rest/thread/{session_id}",
                params=params,
                headers=headers,
                timeout=15.0,
            )
        except httpx.HTTPError as exc:
            logger.warning("pplx messages: fetch failed for %s: %s", session_id, exc)
            flow.response = Response.make(
                502,
                json.dumps(
                    {
                        "error": {
                            "message": f"Perplexity thread fetch failed: {exc}",
                            "type": "pplx_fetch_error",
                            "code": 502,
                        }
                    }
                ).encode(),
                {"Content-Type": "application/json"},
            )
            return

        if resp.status_code == 404:
            flow.response = Response.make(
                404,
                json.dumps(
                    {
                        "error": {
                            "message": (
                                f"Perplexity thread {session_id!r} not found or no longer accessible. "
                                f"Verify the slug or remove metadata.session_id to start a new thread."
                            ),
                            "type": "pplx_thread_not_found",
                            "code": 404,
                        }
                    }
                ).encode(),
                {"Content-Type": "application/json"},
            )
            return

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("pplx messages: upstream error for %s: %s", session_id, exc)
            flow.response = Response.make(
                502,
                json.dumps(
                    {
                        "error": {
                            "message": f"Perplexity returned {exc.response.status_code}",
                            "type": "pplx_upstream_error",
                            "code": 502,
                        }
                    }
                ).encode(),
                {"Content-Type": "application/json"},
            )
            return

        thread = resp.json()

        # Convert
        citation_mode = flow.request.query.get("citation_mode") or session_cfg.pplx.thread.citation_mode
        include_reasoning = flow.request.query.get("include_reasoning") == "true"
        messages = _thread_to_openai_messages(thread, citation_mode=citation_mode, include_reasoning=include_reasoning)

        thread_meta_raw = thread.get("thread")
        thread_meta: dict[str, object] = thread_meta_raw if isinstance(thread_meta_raw, dict) else {}
        entries_raw = thread.get("entries")
        entries: list[object] = entries_raw if isinstance(entries_raw, list) else []

        result = {
            "messages": messages,
            "metadata": {"session_id": session_id},
            "thread_info": {
                "slug": (thread_meta.get("slug") if thread_meta else None) or session_id,
                "context_uuid": thread_meta.get("context_uuid") if thread_meta else None,
                "title": thread_meta.get("title") if thread_meta else None,
                "entry_count": len(entries),
            },
        }

        flow.response = Response.make(
            200,
            json.dumps(result).encode(),
            {"Content-Type": "application/json"},
        )
        logger.debug("pplx messages: served %d messages for session %s", len(messages), session_id)
