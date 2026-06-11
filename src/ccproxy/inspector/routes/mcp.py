"""MCP surface on the proxy listener: ``POST /mcp/notify`` + ``/mcp`` rewrite.

``POST /mcp/notify`` ingests fire-and-forget MCP terminal events (see
``docs/mcp.md``) into the ``NotificationBuffer``. The contract is 200-always:
callers (mcptty) treat delivery as best-effort and never retry, so malformed
payloads are logged and acknowledged with ``{"status": "error"}`` rather than
rejected.

``/mcp`` rewrites reverse-proxy flows to the in-process FastMCP
streamable-HTTP server, so MCP clients can reach the daemon through the proxy
socket without knowing the internal bind. Bearer auth stays enforced by the
FastMCP app itself; SSE responses stream through the existing
``InspectorAddon.responseheaders`` passthrough.

Both routes register before the transform ``/{path}`` catch-all and are gated
to ``ReverseMode`` flows — WireGuard-tunneled traffic to a real upstream's
``/mcp*`` paths continues to forward unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from mitmproxy.http import HTTPFlow

    from ccproxy.inspector.router import InspectorRouter

logger = logging.getLogger(__name__)


class NotifyRequest(BaseModel):
    """Incoming notification from mcptty."""

    task_id: str
    session_id: str
    claude_session_id: str = ""
    event: dict[str, Any]


def _json_response(flow: HTTPFlow, payload: dict[str, str]) -> None:
    from mitmproxy.http import Response

    flow.response = Response.make(
        200,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )


def register_mcp_routes(router: InspectorRouter) -> None:
    """Register ``POST /mcp/notify`` and the ``/mcp`` same-socket rewrite."""
    from ccproxy.inspector.router import RouteType

    @router.route("/mcp/notify", rtype=RouteType.REQUEST, catch_error=False)
    def handle_notify(flow: HTTPFlow, **kwargs: object) -> None:  # pyright: ignore[reportUnusedFunction]
        from mitmproxy.proxy.mode_specs import ReverseMode

        if not isinstance(flow.client_conn.proxy_mode, ReverseMode):
            return
        if flow.request.method != "POST":
            return

        try:
            notify = NotifyRequest.model_validate_json(flow.request.content or b"")
        except (ValidationError, ValueError) as exc:
            logger.warning("Discarding malformed /mcp/notify payload: %s", exc)
            _json_response(flow, {"status": "error"})
            return

        from ccproxy.mcp.buffer import DEFAULT_TTL_SECONDS, get_buffer

        try:
            from ccproxy.config import get_config

            ttl_seconds = get_config().mcp.buffer.ttl_seconds
        except Exception:
            ttl_seconds = DEFAULT_TTL_SECONDS

        buffer = get_buffer()
        # TTL enforcement is lazy: each ingest sweeps entries idle past the TTL.
        buffer.expire(ttl_seconds)
        buffer.append(notify.task_id, notify.session_id, notify.event)
        _json_response(flow, {"status": "ok"})
        logger.debug(
            "Buffered MCP event for task %s session %s",
            notify.task_id,
            notify.session_id,
        )

    @router.route("/mcp", rtype=RouteType.REQUEST, catch_error=False)
    def handle_mcp(flow: HTTPFlow, **kwargs: object) -> None:  # pyright: ignore[reportUnusedFunction]
        from mitmproxy.proxy.mode_specs import ReverseMode

        if not isinstance(flow.client_conn.proxy_mode, ReverseMode):
            return

        from ccproxy.config import get_config

        mcp_cfg = get_config().mcp.http
        if not mcp_cfg.enabled:
            return

        from mitmproxy.connection import Server

        flow.request.host = mcp_cfg.host
        flow.request.port = mcp_cfg.port
        flow.request.scheme = "http"
        flow.server_conn = Server(address=(mcp_cfg.host, mcp_cfg.port))
        logger.debug("Rewrote /mcp flow to %s:%d", mcp_cfg.host, mcp_cfg.port)
