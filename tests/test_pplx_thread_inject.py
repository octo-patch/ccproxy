"""Tests for ccproxy.hooks.pplx_thread_inject — three-mode thread resolution."""
# ruff: noqa: S106, S107  # fake session-cookie / read_write_token literals

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import httpx
import pytest
from mitmproxy.http import HTTPFlow

from ccproxy.auth.sources import FileAuthSource
from ccproxy.config import CCProxyConfig, PplxConfig, PplxThreadConfig, Provider, set_config_instance
from ccproxy.hooks.pplx_thread_inject import (
    _count_client_user_turns,
    _fetch_thread,
    pplx_thread_inject,
    pplx_thread_inject_guard,
)
from ccproxy.lightllm.pplx import PerplexityError
from ccproxy.lightllm.pplx_threads import get_pplx_thread_store
from ccproxy.pipeline.context import Context

_THREAD_URL = "https://www.perplexity.ai/rest/thread/slug-1"


def make_ctx(
    body: dict[str, Any],
    *,
    auth_provider: str = "perplexity_pro",
    conversation_id: str | None = None,
) -> Context:
    flow = MagicMock()
    flow.id = "test-id"
    flow.request.content = json.dumps(body).encode()
    flow.request.headers = {}
    flow.metadata = {"ccproxy.auth_provider": auth_provider}
    if conversation_id is not None:
        flow.metadata["ccproxy.conversation_id"] = conversation_id
    return Context.from_flow(flow)


def _flow(ctx: Context) -> HTTPFlow:
    assert ctx.flow is not None
    return ctx.flow


def set_pplx_config(
    tmp_path: Path,
    *,
    consistency_mode: Literal["warn", "strict", "ignore"] = "warn",
    token: str = "cookie-token",
) -> None:
    token_file = tmp_path / "pplx-token"
    token_file.write_text(token)
    set_config_instance(
        CCProxyConfig(
            providers={
                "perplexity_pro": Provider(
                    auth=FileAuthSource(file=str(token_file)),
                    host="www.perplexity.ai",
                    path="/rest/sse/perplexity_ask",
                    type="perplexity_pro",
                ),
            },
            pplx=PplxConfig(thread=PplxThreadConfig(consistency_mode=consistency_mode)),
        )
    )


def _page(entries: list[dict[str, Any]], **extra: Any) -> httpx.Response:
    return httpx.Response(
        200,
        json={"entries": entries, **extra},
        request=httpx.Request("GET", _THREAD_URL),
    )


_ENTRY = {"backend_uuid": "B1", "context_uuid": "C1", "read_write_token": "T1"}


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


def test_guard_true_for_perplexity_sentinel() -> None:
    ctx = make_ctx({"messages": []})
    assert pplx_thread_inject_guard(ctx) is True


def test_guard_false_for_other_provider() -> None:
    ctx = make_ctx({"messages": []}, auth_provider="anthropic")
    assert pplx_thread_inject_guard(ctx) is False


# ---------------------------------------------------------------------------
# Mode 3 — pass-through
# ---------------------------------------------------------------------------


def test_mode3_passthrough_leaves_body_untouched(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    ctx = make_ctx({"messages": [{"role": "user", "content": "hi"}]}, conversation_id="conv-miss")

    result = pplx_thread_inject(ctx, {})

    assert "pplx" not in result._body
    assert "ccproxy.pplx.resolved_via" not in _flow(result).metadata


# ---------------------------------------------------------------------------
# Mode 2 — organic L1 cache
# ---------------------------------------------------------------------------


def test_mode2_l1_cache_hit_injects_identifiers(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    get_pplx_thread_store().save(
        conversation_id="conv-1",
        backend_uuid="B1",
        read_write_token="T1",
        context_uuid="C1",
        thread_url_slug="S1",
    )
    ctx = make_ctx({"messages": [{"role": "user", "content": "again"}]}, conversation_id="conv-1")

    result = pplx_thread_inject(ctx, {})

    assert result._body["pplx"] == {
        "last_backend_uuid": "B1",
        "frontend_context_uuid": "C1",
        "read_write_token": "T1",
    }
    assert result.metadata.pplx.resolved_via == "l1_cache"


def test_mode2_omits_read_write_token_when_absent(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    get_pplx_thread_store().save(
        conversation_id="conv-2",
        backend_uuid="B2",
        read_write_token=None,
        context_uuid="C2",
        thread_url_slug=None,
    )
    ctx = make_ctx({"messages": [{"role": "user", "content": "again"}]}, conversation_id="conv-2")

    result = pplx_thread_inject(ctx, {})

    assert result._body["pplx"] == {
        "last_backend_uuid": "B2",
        "frontend_context_uuid": "C2",
    }


# ---------------------------------------------------------------------------
# Mode 1 — explicit metadata.session_id
# ---------------------------------------------------------------------------


def test_mode1_metadata_slug_resolves_latest_entry(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    body = {
        "metadata": {"session_id": "slug-1"},
        # Two user turns in history (excluding the new final turn) == two server entries.
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
        ],
    }
    ctx = make_ctx(body)
    pages = [_page([{"backend_uuid": "B0", "context_uuid": "C0"}, _ENTRY])]

    with patch("ccproxy.hooks.pplx_thread_inject.httpx.get", side_effect=pages):
        result = pplx_thread_inject(ctx, {})

    assert result._body["pplx"] == {
        "last_backend_uuid": "B1",
        "frontend_context_uuid": "C1",
        "read_write_token": "T1",
    }
    assert result.metadata.pplx.resolved_via == "metadata"
    assert "ccproxy.pplx.divergence" not in _flow(result).metadata


def test_mode1_divergence_warn_stamps_metadata(tmp_path: Path) -> None:
    set_pplx_config(tmp_path, consistency_mode="warn")
    body = {
        "metadata": {"session_id": "slug-1"},
        "messages": [{"role": "user", "content": "q1"}, {"role": "user", "content": "q2"}],
    }
    ctx = make_ctx(body)

    with patch("ccproxy.hooks.pplx_thread_inject.httpx.get", side_effect=[_page([_ENTRY, _ENTRY, _ENTRY])]):
        result = pplx_thread_inject(ctx, {})

    assert result.metadata.pplx.divergence == "turn_count_mismatch: client=1 server=3"
    assert result._body["pplx"]["last_backend_uuid"] == "B1"


def test_mode1_divergence_strict_raises_409(tmp_path: Path) -> None:
    set_pplx_config(tmp_path, consistency_mode="strict")
    body = {
        "metadata": {"session_id": "slug-1"},
        "messages": [{"role": "user", "content": "q1"}, {"role": "user", "content": "q2"}],
    }
    ctx = make_ctx(body)

    with (
        patch("ccproxy.hooks.pplx_thread_inject.httpx.get", side_effect=[_page([_ENTRY, _ENTRY, _ENTRY])]),
        pytest.raises(PerplexityError, match=r"diverged from incoming history"),
    ):
        pplx_thread_inject(ctx, {})


def test_mode1_no_token_raises_503(tmp_path: Path) -> None:
    set_config_instance(CCProxyConfig(providers={}))
    ctx = make_ctx({"metadata": {"session_id": "slug-1"}, "messages": []})

    with pytest.raises(PerplexityError, match=r"no session token is configured"):
        pplx_thread_inject(ctx, {})


def test_mode1_upstream_404_propagates(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    ctx = make_ctx({"metadata": {"session_id": "slug-1"}, "messages": []})
    not_found = httpx.Response(404, json={"detail": "not found"}, request=httpx.Request("GET", _THREAD_URL))

    with (
        patch("ccproxy.hooks.pplx_thread_inject.httpx.get", return_value=not_found),
        pytest.raises(httpx.HTTPStatusError),
    ):
        pplx_thread_inject(ctx, {})


def test_mode1_network_error_raises_502(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    ctx = make_ctx({"metadata": {"session_id": "slug-1"}, "messages": []})

    with (
        patch(
            "ccproxy.hooks.pplx_thread_inject.httpx.get",
            side_effect=httpx.ConnectError("boom", request=httpx.Request("GET", _THREAD_URL)),
        ),
        pytest.raises(PerplexityError, match=r"thread fetch failed for 'slug-1'"),
    ):
        pplx_thread_inject(ctx, {})


def test_mode1_empty_entries_raises_502(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    ctx = make_ctx({"metadata": {"session_id": "slug-1"}, "messages": []})

    with (
        patch("ccproxy.hooks.pplx_thread_inject.httpx.get", side_effect=[_page([])]),
        pytest.raises(PerplexityError, match=r"no usable continuation identifiers"),
    ):
        pplx_thread_inject(ctx, {})


# ---------------------------------------------------------------------------
# _fetch_thread pagination
# ---------------------------------------------------------------------------


def test_fetch_thread_merges_pages(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    pages = [
        _page([{"backend_uuid": "B1", "context_uuid": "C1"}], has_next=True, end_cursor="cur-1"),
        _page([{"backend_uuid": "B2", "context_uuid": "C2"}]),
    ]

    with patch("ccproxy.hooks.pplx_thread_inject.httpx.get", side_effect=pages) as mock_get:
        thread = _fetch_thread("slug-1", "cookie-token")

    assert [e["backend_uuid"] for e in thread["entries"]] == ["B1", "B2"]
    assert thread["ccproxy_pages_fetched"] == 2
    assert thread["has_next"] is False
    assert mock_get.call_count == 2


def test_fetch_thread_missing_cursor_raises(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)

    with (
        patch("ccproxy.hooks.pplx_thread_inject.httpx.get", side_effect=[_page([_ENTRY], has_next=True)]),
        pytest.raises(PerplexityError, match=r"without a pagination cursor"),
    ):
        _fetch_thread("slug-1", "cookie-token")


def test_fetch_thread_repeated_cursor_raises(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    pages = [
        _page([_ENTRY], has_next=True, end_cursor="cur-1"),
        _page([_ENTRY], has_next=True, end_cursor="cur-1"),
    ]

    with (
        patch("ccproxy.hooks.pplx_thread_inject.httpx.get", side_effect=pages),
        pytest.raises(PerplexityError, match=r"repeated pagination cursor 'cur-1'"),
    ):
        _fetch_thread("slug-1", "cookie-token")


# ---------------------------------------------------------------------------
# _count_client_user_turns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param([], 0, id="empty"),
        pytest.param([{"role": "user", "content": "q"}], 0, id="single_message"),
        pytest.param(
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ],
            1,
            id="one_history_user_turn",
        ),
        pytest.param(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ],
            1,
            id="system_interleaved_not_counted",
        ),
        pytest.param(
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
            ],
            2,
            id="two_history_user_turns",
        ),
        pytest.param(["not-a-dict", {"role": "user", "content": "q"}], 0, id="non_dict_entries_ignored"),
    ],
)
def test_count_client_user_turns(messages: list[Any], expected: int) -> None:
    assert _count_client_user_turns(messages) == expected
