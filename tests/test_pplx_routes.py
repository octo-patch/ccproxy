"""Tests for ccproxy.inspector.routes.pplx — GET /pplx/messages/<session_id>."""
# ruff: noqa: S107  # fake session-cookie literals

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import httpx
from mitmproxy.proxy.mode_specs import ProxyMode

from ccproxy.auth.sources import FileAuthSource
from ccproxy.config import CCProxyConfig, McpConfig, McpHttpConfig, Provider, set_config_instance
from ccproxy.inspector.router import InspectorRouter
from ccproxy.inspector.routes.pplx import register_pplx_routes

_THREAD_URL = "https://www.perplexity.ai/rest/thread/slug-1"

_THREAD_PAGE = {
    "thread": {"slug": "slug-1", "context_uuid": "C1", "title": "What is quantum computing?"},
    "entries": [
        {
            "query_str": "what is quantum computing?",
            "backend_uuid": "B1",
            "context_uuid": "C1",
            "structured_answer_block_usages": ["ask_text_0_markdown"],
            "blocks": [
                {
                    "intended_usage": "ask_text_0_markdown",
                    "markdown_block": {"answer": "Quantum computing is neat."},
                },
            ],
        },
    ],
}


def set_pplx_config(tmp_path: Path, *, mcp_auth: str | None = None, token: str = "cookie-token") -> None:
    token_file = tmp_path / "pplx-token"
    token_file.write_text(token)
    set_config_instance(
        CCProxyConfig(
            providers={
                "perplexity_pro": Provider(
                    auth=FileAuthSource(file=str(token_file)),
                    base_url="https://www.perplexity.ai",
                    path="/rest/sse/perplexity_ask",
                    type="perplexity_pro",
                ),
            },
            mcp=McpConfig(http=McpHttpConfig(auth=mcp_auth)),
        )
    )


def _make_router() -> InspectorRouter:
    router = InspectorRouter(name="test_pplx_routes", request_passthrough=True, response_passthrough=True)
    register_pplx_routes(router)
    return router


def _handler(router: InspectorRouter) -> Any:
    return next(h for _, parser, h in router.request_routes if parser._format == "/pplx/messages/{session_id}")


def _make_flow(
    method: str = "GET",
    *,
    reverse: bool = True,
    bearer: str | None = None,
    query: dict[str, str] | None = None,
) -> MagicMock:
    flow = MagicMock()
    flow.request.method = method
    flow.request.path = "/pplx/messages/slug-1"
    flow.request.headers = {"Authorization": f"Bearer {bearer}"} if bearer is not None else {}
    flow.request.query = query or {}
    flow.response = None
    if reverse:
        flow.client_conn.proxy_mode = ProxyMode.parse("reverse:http://localhost:1@4001")
    else:
        flow.client_conn.proxy_mode = ProxyMode.parse("wireguard@51820")
    return flow


def _body(flow: MagicMock) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(flow.response.content))


def test_skips_non_reverse_and_non_get(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    handler = _handler(_make_router())

    wireguard_flow = _make_flow(reverse=False)
    handler(wireguard_flow, session_id="slug-1")
    assert wireguard_flow.response is None

    post_flow = _make_flow(method="POST")
    handler(post_flow, session_id="slug-1")
    assert post_flow.response is None


def test_auth_required_when_mcp_token_configured(tmp_path: Path) -> None:
    set_pplx_config(tmp_path, mcp_auth="secret-token")
    handler = _handler(_make_router())

    missing = _make_flow()
    handler(missing, session_id="slug-1")
    assert missing.response.status_code == 401
    assert _body(missing)["error"] == {"message": "unauthorized", "type": "auth_error", "code": 401}

    wrong = _make_flow(bearer="wrong")
    handler(wrong, session_id="slug-1")
    assert wrong.response.status_code == 401


def test_provider_not_configured_returns_503(tmp_path: Path) -> None:
    set_config_instance(CCProxyConfig(providers={}))
    handler = _handler(_make_router())
    flow = _make_flow()

    handler(flow, session_id="slug-1")

    assert flow.response.status_code == 503
    assert _body(flow)["error"]["type"] == "pplx_unavailable"
    assert "'perplexity_pro' not configured" in _body(flow)["error"]["message"]


def test_empty_token_returns_503(tmp_path: Path) -> None:
    set_pplx_config(tmp_path, token="")
    handler = _handler(_make_router())
    flow = _make_flow()

    handler(flow, session_id="slug-1")

    assert flow.response.status_code == 503
    assert "no session cookie resolved" in _body(flow)["error"]["message"]


def test_happy_path_converts_thread_to_messages(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    handler = _handler(_make_router())
    flow = _make_flow()
    page = httpx.Response(200, json=_THREAD_PAGE, request=httpx.Request("GET", _THREAD_URL))

    with patch("ccproxy.hooks.pplx_thread_inject.httpx.get", return_value=page):
        handler(flow, session_id="slug-1")

    assert flow.response.status_code == 200
    body = _body(flow)
    assert body["messages"] == [
        {"role": "user", "content": "what is quantum computing?"},
        {"role": "assistant", "content": "Quantum computing is neat."},
    ]
    assert body["metadata"] == {"session_id": "slug-1"}
    assert body["thread_info"] == {
        "slug": "slug-1",
        "context_uuid": "C1",
        "title": "What is quantum computing?",
        "entry_count": 1,
    }


def test_upstream_status_error_passes_through(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    handler = _handler(_make_router())
    flow = _make_flow()
    not_found = httpx.Response(404, json={"detail": "unknown thread"}, request=httpx.Request("GET", _THREAD_URL))

    with patch("ccproxy.hooks.pplx_thread_inject.httpx.get", return_value=not_found):
        handler(flow, session_id="slug-1")

    assert flow.response.status_code == 404
    assert _body(flow) == {"detail": "unknown thread"}


def test_network_error_returns_502(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    handler = _handler(_make_router())
    flow = _make_flow()

    with patch(
        "ccproxy.hooks.pplx_thread_inject.httpx.get",
        side_effect=httpx.ConnectError("down", request=httpx.Request("GET", _THREAD_URL)),
    ):
        handler(flow, session_id="slug-1")

    assert flow.response.status_code == 502
    assert _body(flow)["error"]["type"] == "pplx_fetch_error"
