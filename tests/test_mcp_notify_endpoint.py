"""Tests for ccproxy.inspector.routes.mcp — /mcp/notify ingestion + /mcp rewrite."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from mitmproxy.proxy.mode_specs import ProxyMode

from ccproxy.config import CCProxyConfig, McpConfig, McpHttpConfig, set_config_instance
from ccproxy.inspector.router import InspectorRouter
from ccproxy.inspector.routes.mcp import register_mcp_routes
from ccproxy.mcp.buffer import get_buffer


def _make_router() -> InspectorRouter:
    router = InspectorRouter(name="test_mcp_routes", request_passthrough=True, response_passthrough=True)
    register_mcp_routes(router)
    return router


def _handler(router: InspectorRouter, path: str) -> Any:
    return next(h for _, parser, h in router.request_routes if parser._format == path)


def _make_flow(
    method: str = "POST",
    path: str = "/mcp/notify",
    body: dict[str, Any] | None = None,
    raw_body: bytes | None = None,
    reverse: bool = True,
) -> MagicMock:
    flow = MagicMock()
    flow.request.method = method
    flow.request.path = path
    flow.request.host = "localhost"
    flow.request.port = 4001
    flow.request.scheme = "http"
    flow.request.content = raw_body if raw_body is not None else json.dumps(body or {}).encode()
    flow.response = None
    if reverse:
        flow.client_conn.proxy_mode = ProxyMode.parse("reverse:http://localhost:1@4001")
    else:
        flow.client_conn.proxy_mode = ProxyMode.parse("wireguard@51820")
    return flow


def test_register_mcp_routes_registers_notify_and_mcp() -> None:
    router = _make_router()
    paths = {parser._format for _, parser, _ in router.request_routes}
    assert paths == {"/mcp/notify", "/mcp"}


def test_valid_event_returns_200_ok() -> None:
    router = _make_router()
    flow = _make_flow(body={"task_id": "t1", "session_id": "s1", "event": {"type": "output", "text": "hello"}})

    _handler(router, "/mcp/notify")(flow)

    assert flow.response is not None
    assert flow.response.status_code == 200
    assert json.loads(flow.response.content) == {"status": "ok"}


def test_valid_event_stored_in_buffer() -> None:
    router = _make_router()
    event = {"type": "output", "text": "hello"}
    flow = _make_flow(body={"task_id": "t1", "session_id": "s1", "event": event})

    _handler(router, "/mcp/notify")(flow)

    buf = get_buffer()
    assert not buf.is_empty()
    assert buf.drain_session("s1") == {"t1": [event]}


def test_missing_required_fields_returns_200_error() -> None:
    """Contract is 200-always (fire-and-forget); invalid payloads are acknowledged and dropped."""
    router = _make_router()
    for body in (
        {"session_id": "s1", "event": {"type": "output"}},  # no task_id
        {"task_id": "t1", "event": {"type": "output"}},  # no session_id
        {"task_id": "t1", "session_id": "s1"},  # no event
    ):
        flow = _make_flow(body=body)
        _handler(router, "/mcp/notify")(flow)

        assert flow.response is not None
        assert flow.response.status_code == 200
        assert json.loads(flow.response.content) == {"status": "error"}

    assert get_buffer().is_empty()


def test_non_json_body_returns_200_error() -> None:
    router = _make_router()
    flow = _make_flow(raw_body=b"not json at all")

    _handler(router, "/mcp/notify")(flow)

    assert flow.response is not None
    assert flow.response.status_code == 200
    assert json.loads(flow.response.content) == {"status": "error"}
    assert get_buffer().is_empty()


def test_notify_skips_non_post() -> None:
    router = _make_router()
    flow = _make_flow(method="GET", body={"task_id": "t1", "session_id": "s1", "event": {}})

    _handler(router, "/mcp/notify")(flow)

    assert flow.response is None
    assert get_buffer().is_empty()


def test_notify_skips_wireguard_flows() -> None:
    router = _make_router()
    flow = _make_flow(body={"task_id": "t1", "session_id": "s1", "event": {}}, reverse=False)

    _handler(router, "/mcp/notify")(flow)

    assert flow.response is None
    assert get_buffer().is_empty()


def test_multiple_posts_accumulate_in_buffer() -> None:
    router = _make_router()
    events = [
        {"type": "output", "text": "line1"},
        {"type": "output", "text": "line2"},
        {"type": "exit", "code": 0},
    ]
    handler = _handler(router, "/mcp/notify")
    for event in events:
        handler(_make_flow(body={"task_id": "t1", "session_id": "s1", "event": event}))

    assert get_buffer().drain_session("s1") == {"t1": events}


def test_stale_tasks_expire_on_ingest() -> None:
    """TTL enforcement is lazy: each ingest sweeps entries idle past mcp.buffer.ttl_seconds."""
    set_config_instance(CCProxyConfig(mcp=McpConfig()))
    router = _make_router()
    handler = _handler(router, "/mcp/notify")

    handler(_make_flow(body={"task_id": "stale", "session_id": "s-old", "event": {"type": "output"}}))
    buf = get_buffer()
    # Age the stale task past the default 600s TTL.
    buf._buffers["stale"].last_seen -= 601

    handler(_make_flow(body={"task_id": "fresh", "session_id": "s-new", "event": {"type": "output"}}))

    assert not buf.has_events_for_session("s-old")
    assert buf.has_events_for_session("s-new")


def test_different_session_ids_separated_in_buffer() -> None:
    router = _make_router()
    event_a = {"type": "output", "text": "from session A"}
    event_b = {"type": "output", "text": "from session B"}
    handler = _handler(router, "/mcp/notify")

    handler(_make_flow(body={"task_id": "t1", "session_id": "session-a", "event": event_a}))
    handler(_make_flow(body={"task_id": "t2", "session_id": "session-b", "event": event_b}))

    buf = get_buffer()
    assert buf.drain_session("session-a") == {"t1": [event_a]}
    assert buf.drain_session("session-b") == {"t2": [event_b]}


def test_mcp_route_rewrites_to_internal_server() -> None:
    set_config_instance(CCProxyConfig(mcp=McpConfig(http=McpHttpConfig(enabled=True, host="127.0.0.1", port=4030))))
    router = _make_router()
    flow = _make_flow(method="POST", path="/mcp", body={"jsonrpc": "2.0"})

    _handler(router, "/mcp")(flow)

    assert flow.response is None
    assert flow.request.host == "127.0.0.1"
    assert flow.request.port == 4030
    assert flow.request.scheme == "http"
    assert flow.server_conn.address == ("127.0.0.1", 4030)


def test_mcp_route_untouched_when_disabled() -> None:
    set_config_instance(CCProxyConfig(mcp=McpConfig(http=McpHttpConfig(enabled=False))))
    router = _make_router()
    flow = _make_flow(method="POST", path="/mcp", body={"jsonrpc": "2.0"})

    _handler(router, "/mcp")(flow)

    assert flow.response is None
    assert flow.request.host == "localhost"
    assert flow.request.port == 4001


def test_mcp_route_skips_wireguard_flows() -> None:
    set_config_instance(CCProxyConfig(mcp=McpConfig(http=McpHttpConfig(enabled=True))))
    router = _make_router()
    flow = _make_flow(method="POST", path="/mcp", body={"jsonrpc": "2.0"}, reverse=False)

    _handler(router, "/mcp")(flow)

    assert flow.response is None
    assert flow.request.host == "localhost"
    assert flow.request.port == 4001
