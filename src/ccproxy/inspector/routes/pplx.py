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

import httpx

from ccproxy.hooks.pplx_thread_inject import _fetch_thread
from ccproxy.lightllm.pplx import PERPLEXITY_PROVIDER_NAME, _thread_to_openai_messages

if TYPE_CHECKING:
    from mitmproxy.http import HTTPFlow

    from ccproxy.inspector.router import InspectorRouter

logger = logging.getLogger(__name__)


def _upstream_headers(response: httpx.Response) -> dict[str, str]:
    content_type = response.headers.get("content-type", "application/json")
    return {"Content-Type": content_type}


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

    @router.route("/pplx/messages/{session_id}", rtype=RouteType.REQUEST, catch_error=False)
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

        try:
            thread = _fetch_thread(session_id, token)
        except httpx.HTTPStatusError as exc:
            upstream = exc.response
            flow.response = Response.make(upstream.status_code, upstream.content, _upstream_headers(upstream))
            return
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
