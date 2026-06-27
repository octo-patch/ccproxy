"""Tests for :mod:`ccproxy.inspector.openai_conversations_addon` and
related helpers (:mod:`ccproxy.openai_conversations.profile`,
:mod:`ccproxy.hooks.openai_conversations_thread_inject`).

Uses mocked / fake transport clients — no real network calls.

Coverage:
- Warmup only runs when usable cookies are absent or throttle expired.
- Sentinel refreshed only when expired/within skew.
- Conduit prepare runs through none→sent→success, first call sends empty
  x-conduit-token, subsequent calls use the returned token.
- Prepare and final request share one x-oai-turn-trace-id.
- All calls (warmup, sentinel, prepare) use the SAME ``get_client(host, profile)``
  key; the same client instance is returned from the cache.
- Response retry runs at most once and marks metadata so it cannot loop.
- Non-openai_conversations providers are byte-for-byte unaffected.
- X-OAI-IS sourced from cookie jar, not credential state.
- thread_inject populates raw_extras on store hit, not on miss.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccproxy.inspector.openai_conversations_addon import (
    OpenAIConversationsAddon,
    _scan_sse_for_conversation_ids,
    _warmup_timestamps,
)
from ccproxy.openai_conversations.conversation_store import ConversationStore

# Backing metadata key for the CcproxyMetadata.oaic_retry_done facade field.
_RETRY_DONE_KEY = "ccproxy.oaic_retry_done"

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_flow(
    *,
    provider: str = "openai_conversations",
    provider_type: str = "openai_conversations",
    profile: str = "chrome136",
    host: str = "chatgpt.com",
    path: str = "/backend-api/f/conversation",
    method: str = "POST",
    content: bytes = b'{"model": "gpt-5-5-pro", "messages": [], "action": "next"}',
    status_code: int = 200,
    response_content: bytes = b"",
    response_headers: dict[str, str] | None = None,
    auth_injected: bool = True,
) -> MagicMock:
    flow = MagicMock()
    flow.id = "test-flow-id"
    flow.metadata = {
        "ccproxy.auth_provider": provider,
    }
    if auth_injected:
        flow.metadata["ccproxy.auth_injected"] = True
    flow.request.method = method
    flow.request.path = path
    flow.request.pretty_url = f"https://{host}{path}"
    flow.request.pretty_host = host
    flow.request.content = content
    flow.request.headers = {}
    flow.response = MagicMock()
    flow.response.status_code = status_code
    flow.response.content = response_content
    flow.response.headers = MagicMock()
    flow.response.headers.get = MagicMock(return_value=None)
    flow.response.headers.clear = MagicMock()
    flow.response.headers.add = MagicMock()
    flow.response.headers.multi_items = MagicMock(return_value=[])
    if response_headers:
        flow.response.headers.get = MagicMock(side_effect=lambda k, default=None: response_headers.get(k, default))
    return flow


def _make_config(
    *,
    provider_name: str = "openai_conversations",
    provider_type: str = "openai_conversations",
    profile: str = "chrome136",
    host: str = "chatgpt.com",
    credential_path: str = "/dev/null",
    warmup_throttle_seconds: float = 600.0,
    sentinel_skew_seconds: float = 60.0,
    request_timeout_seconds: float = 10.0,
) -> MagicMock:
    cfg = MagicMock()
    provider = MagicMock()
    provider.type = provider_type
    provider.host = host
    provider.fingerprint_profile = profile
    provider.auth = MagicMock()
    provider.auth.file_path = credential_path
    cfg.providers = {provider_name: provider}
    oaic_cfg = MagicMock()
    oaic_cfg.warmup_throttle_seconds = warmup_throttle_seconds
    oaic_cfg.sentinel_skew_seconds = sentinel_skew_seconds
    oaic_cfg.request_timeout_seconds = request_timeout_seconds
    cfg.lightllm.openai_conversations = oaic_cfg
    return cfg


def _make_credential_state(
    *,
    access_token: str = "jwt.access.token",  # noqa: S107
    device_id: str = "dev-id-uuid",
    chat_req_token: str = "",
    proof_token: str = "",
    chat_req_token_expires_at_ms: int = 0,
) -> MagicMock:
    state = MagicMock()
    state.access_token = access_token
    state.device_id = device_id
    state.chat_req_token = chat_req_token
    state.proof_token = proof_token
    state.chat_req_token_expires_at_ms = chat_req_token_expires_at_ms
    return state


def _make_mock_client(
    *,
    get_side_effect: list[Any] | None = None,
    post_responses: list[MagicMock] | None = None,
    has_cookies: bool = False,
    oai_is_cookie: str = "",
) -> MagicMock:
    """Build a mock httpx.AsyncClient-like object."""
    client = MagicMock()

    # Cookie jar
    cookie_jar = MagicMock()
    if has_cookies:
        mock_cookie = MagicMock()
        mock_cookie.name = "cf_clearance"
        cookie_jar.jar = [mock_cookie]
    else:
        cookie_jar.jar = []
    cookie_jar.get = MagicMock(return_value=oai_is_cookie or None)
    cookie_jar.set = MagicMock()
    client.cookies = cookie_jar

    # GET
    async def _get(url: str, **kwargs: Any) -> Any:
        resp = MagicMock()
        resp.status_code = 200
        resp.aread = AsyncMock()
        return resp

    if get_side_effect is not None:
        _get_iter = iter(get_side_effect)

        async def _get_se(url: str, **kwargs: Any) -> Any:
            try:
                item = next(_get_iter)
            except StopIteration:
                resp = MagicMock()
                resp.status_code = 200
                resp.aread = AsyncMock()
                return resp
            if isinstance(item, Exception):
                raise item
            return item

        client.get = AsyncMock(side_effect=_get_se)
    else:
        client.get = AsyncMock(side_effect=_get)

    # POST
    if post_responses:
        post_iter = iter(post_responses)

        async def _post_se(url: str, **kwargs: Any) -> Any:
            try:
                return next(post_iter)
            except StopIteration:
                r = MagicMock()
                r.status_code = 200
                r.json = MagicMock(return_value={})
                r.raise_for_status = MagicMock()
                return r

        client.post = AsyncMock(side_effect=_post_se)
    else:
        default_post_resp = MagicMock()
        default_post_resp.status_code = 200
        default_post_resp.json = MagicMock(return_value={"conduit_token": "tok-123"})
        default_post_resp.raise_for_status = MagicMock()
        client.post = AsyncMock(return_value=default_post_resp)

    # request() method (for retry)
    retry_resp = MagicMock()
    retry_resp.status_code = 200
    retry_resp.headers = MagicMock()
    retry_resp.headers.multi_items = MagicMock(return_value=[])
    retry_resp.content = b'{"retried": true}'
    client.request = AsyncMock(return_value=retry_resp)

    return client


def _sentinel_prepare_response(*, required: bool = True, persona: str = "chatgpt-paid") -> MagicMock:
    """Mock the /sentinel/chat-requirements/prepare response.

    Difficulty ``"ffffff"`` (the max 6-hex prefix) lets the real ``solve_pow``
    accept attempt 0 immediately, keeping the test fast and deterministic while
    still exercising the genuine PoW path.
    """
    resp = MagicMock()
    resp.status_code = 200
    pow_info: dict[str, Any] = (
        {"required": True, "seed": "seed-0", "difficulty": "ffffff"} if required else {"required": False}
    )
    resp.json = MagicMock(return_value={"prepare_token": "prep-token", "proofofwork": pow_info, "persona": persona})
    resp.raise_for_status = MagicMock()
    return resp


def _sentinel_finalize_response(token: str = "chat-req-tok", persona: str = "chatgpt-paid") -> MagicMock:  # noqa: S107
    """Mock the /sentinel/chat-requirements/finalize response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value={"token": token, "persona": persona})
    resp.raise_for_status = MagicMock()
    return resp


def _prepare_post_response(conduit_token: str = "conduit-tok") -> MagicMock:  # noqa: S107
    resp = MagicMock()
    resp.status_code = 200
    resp.text = ""
    resp.json = MagicMock(return_value={"conduit_token": conduit_token})
    return resp


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------


class TestProfile:
    def test_get_browser_headers_contains_required_keys(self) -> None:
        from ccproxy.openai_conversations.profile import get_browser_headers

        h = get_browser_headers(device_id="dev-1", session_id="sess-1")
        assert "user-agent" in h
        assert "sec-ch-ua" in h
        assert "oai-device-id" in h
        assert h["oai-device-id"] == "dev-1"
        assert h["oai-session-id"] == "sess-1"

    def test_final_only_headers_absent_on_prepare(self) -> None:
        from ccproxy.openai_conversations.profile import get_browser_headers

        h = get_browser_headers(device_id="d", session_id="s", final=False)
        assert "oai-echo-logs" not in h
        assert "oai-telemetry" not in h

    def test_final_only_headers_present_on_final(self) -> None:
        from ccproxy.openai_conversations.profile import get_browser_headers

        h = get_browser_headers(device_id="d", session_id="s", final=True)
        assert "oai-echo-logs" in h
        assert "oai-telemetry" in h

    def test_referer_uses_conversation_id_when_set(self) -> None:
        from ccproxy.openai_conversations.profile import get_browser_headers

        h = get_browser_headers(device_id="d", session_id="s", conversation_id="conv-abc")
        assert h["referer"] == "https://chatgpt.com/c/conv-abc"

    def test_referer_defaults_to_root(self) -> None:
        from ccproxy.openai_conversations.profile import get_browser_headers

        h = get_browser_headers(device_id="d", session_id="s")
        assert h["referer"] == "https://chatgpt.com/"


# ---------------------------------------------------------------------------
# SSE scanner
# ---------------------------------------------------------------------------


class TestScanSSE:
    def test_finds_conversation_id_and_parent_message_id(self) -> None:
        event1 = '{"conversation_id": "chatgpt-conv-1", "message": {"id": "msg-1", "author": {"role": "assistant"}}}'
        event2 = '{"conversation_id": "chatgpt-conv-1", "message": {"id": "msg-2", "author": {"role": "assistant"}}}'
        sse = f"data: {{}}\n\ndata: {event1}\n\ndata: {event2}\n\ndata: [DONE]\n\n"
        cid, pid = _scan_sse_for_conversation_ids(sse.encode())
        assert cid == "chatgpt-conv-1"
        assert pid == "msg-2"

    def test_returns_empty_when_no_events(self) -> None:
        cid, pid = _scan_sse_for_conversation_ids(b"")
        assert cid == ""
        assert pid == ""

    def test_ignores_non_assistant_messages(self) -> None:
        sse = 'data: {"conversation_id": "c1", "message": {"id": "m1", "author": {"role": "user"}}}\n\n'
        cid, pid = _scan_sse_for_conversation_ids(sse.encode())
        assert cid == "c1"
        assert pid == ""

    def test_last_event_wins_for_parent_message_id(self) -> None:
        sse = (
            'data: {"message": {"id": "early-msg", "author": {"role": "assistant"}}}\n\n'
            'data: {"conversation_id": "c1", "message": {"id": "final-msg", "author": {"role": "assistant"}}}\n\n'
        )
        _, pid = _scan_sse_for_conversation_ids(sse.encode())
        assert pid == "final-msg"


# ---------------------------------------------------------------------------
# Thread inject hook
# ---------------------------------------------------------------------------


class TestThreadInjectHook:
    def test_guard_passes_for_openai_conversations(self) -> None:
        from ccproxy.hooks.openai_conversations_thread_inject import (
            openai_conversations_thread_inject_guard,
        )

        ctx = MagicMock()
        ctx.metadata.auth_provider = "openai_conversations"
        assert openai_conversations_thread_inject_guard(ctx) is True

    def test_guard_rejects_other_providers(self) -> None:
        from ccproxy.hooks.openai_conversations_thread_inject import (
            openai_conversations_thread_inject_guard,
        )

        ctx = MagicMock()
        ctx.metadata.auth_provider = "anthropic"
        assert openai_conversations_thread_inject_guard(ctx) is False

    def test_populates_raw_extras_on_store_hit(self) -> None:
        from ccproxy.hooks.openai_conversations_thread_inject import (
            openai_conversations_thread_inject,
        )

        store = ConversationStore()
        store.save(
            key="sha12-key",
            conversation_id="chatgpt-server-id",
            parent_message_id="parent-msg-id",
        )

        ctx = MagicMock()
        ctx._body = {}
        ctx.metadata.conversation_id = "sha12-key"

        with patch(
            "ccproxy.hooks.openai_conversations_thread_inject.get_conversation_store",
            return_value=store,
        ):
            result = openai_conversations_thread_inject(ctx, {})

        extras = result._body["openai_conversations"]
        assert extras["conversation_id"] == "chatgpt-server-id"
        assert extras["parent_message_id"] == "parent-msg-id"
        assert extras["is_continuation"] is True

    def test_writes_not_continuation_on_miss(self) -> None:
        from ccproxy.hooks.openai_conversations_thread_inject import (
            openai_conversations_thread_inject,
        )

        store = ConversationStore()

        ctx = MagicMock()
        ctx._body = {}
        ctx.metadata.conversation_id = "unknown-key"

        with patch(
            "ccproxy.hooks.openai_conversations_thread_inject.get_conversation_store",
            return_value=store,
        ):
            result = openai_conversations_thread_inject(ctx, {})

        extras = result._body.get("openai_conversations", {})
        assert extras.get("is_continuation") is False

    def test_noop_when_no_conversation_id(self) -> None:
        from ccproxy.hooks.openai_conversations_thread_inject import (
            openai_conversations_thread_inject,
        )

        ctx = MagicMock()
        ctx._body = {}
        ctx.metadata.conversation_id = None

        result = openai_conversations_thread_inject(ctx, {})
        assert result is ctx
        assert "openai_conversations" not in result._body


# ---------------------------------------------------------------------------
# Addon — non-OAIC flows are unaffected
# ---------------------------------------------------------------------------


class TestNonOAICFlowsUnaffected:
    @pytest.mark.asyncio
    async def test_request_noop_for_anthropic_provider(self) -> None:
        flow = _make_flow(provider="anthropic")

        mock_cfg = _make_config(provider_type="anthropic")
        provider = MagicMock()
        provider.type = "anthropic"
        mock_cfg.providers = {"anthropic": provider}

        addon = OpenAIConversationsAddon()
        with patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=mock_cfg):
            await addon.request(flow)

        # Headers should be untouched — no browser headers, no sentinel, no conduit.
        assert flow.request.headers == {}

    @pytest.mark.asyncio
    async def test_response_noop_for_anthropic_provider(self) -> None:
        flow = _make_flow(provider="anthropic")

        mock_cfg = _make_config(provider_type="anthropic")
        provider = MagicMock()
        provider.type = "anthropic"
        mock_cfg.providers = {"anthropic": provider}

        addon = OpenAIConversationsAddon()
        with patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=mock_cfg):
            await addon.response(flow)

        flow.response.headers.clear.assert_not_called()

    @pytest.mark.asyncio
    async def test_request_noop_when_no_auth_provider(self) -> None:
        flow = _make_flow(provider="")
        flow.metadata = {}

        addon = OpenAIConversationsAddon()
        await addon.request(flow)
        assert flow.request.headers == {}


# ---------------------------------------------------------------------------
# Addon — warmup throttle
# ---------------------------------------------------------------------------


class TestWarmupThrottle:
    def setup_method(self) -> None:
        _warmup_timestamps.clear()

    @pytest.mark.asyncio
    async def test_warmup_runs_when_no_usable_cookies(self) -> None:
        flow = _make_flow()
        cfg = _make_config(warmup_throttle_seconds=600.0)
        cred_state = _make_credential_state(chat_req_token_expires_at_ms=9999999999000)

        # Far-future expiry → no sentinel refresh; only the 3 conduit prepares run.
        post_resps = [
            _prepare_post_response("t1"),
            _prepare_post_response("t2"),
            _prepare_post_response("t3"),
        ]
        client = _make_mock_client(has_cookies=False, post_responses=post_resps)

        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.load_credential_state",
                return_value=cred_state,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.update_sentinel_fields",
                return_value=True,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.transport.get_client",
                new=AsyncMock(return_value=client),
            ),
        ):
            addon = OpenAIConversationsAddon()
            await addon.request(flow)

        # All three warmup URLs should have been GETted.
        assert client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_warmup_skipped_when_cookies_present_and_throttle_active(self) -> None:
        provider_name = "openai_conversations"
        profile = "chrome136"
        import time

        _warmup_timestamps[(provider_name, profile)] = time.monotonic()

        flow = _make_flow()
        cfg = _make_config(warmup_throttle_seconds=600.0)
        cred_state = _make_credential_state(chat_req_token_expires_at_ms=9999999999000)

        # Far-future expiry → no sentinel refresh; only the 3 conduit prepares run.
        post_resps = [
            _prepare_post_response("t1"),
            _prepare_post_response("t2"),
            _prepare_post_response("t3"),
        ]
        client = _make_mock_client(has_cookies=True, post_responses=post_resps)

        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.load_credential_state",
                return_value=cred_state,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.update_sentinel_fields",
                return_value=True,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.transport.get_client",
                new=AsyncMock(return_value=client),
            ),
        ):
            addon = OpenAIConversationsAddon()
            await addon.request(flow)

        # Warmup should be skipped (cookies present + throttle active).
        client.get.assert_not_called()


# ---------------------------------------------------------------------------
# Addon — sentinel refresh
# ---------------------------------------------------------------------------


class TestSentinelRefresh:
    @pytest.mark.asyncio
    async def test_sentinel_refreshed_when_expired(self) -> None:
        flow = _make_flow()
        cfg = _make_config()
        # chat_req_token_expires_at_ms=0 → always expired.
        cred_state = _make_credential_state(chat_req_token_expires_at_ms=0)

        post_resps = [
            _sentinel_prepare_response(),
            _sentinel_finalize_response(token="new-chat-req-tok"),  # noqa: S106
            _prepare_post_response("t1"),
            _prepare_post_response("t2"),
            _prepare_post_response("t3"),
        ]
        client = _make_mock_client(has_cookies=True, post_responses=post_resps)

        _warmup_timestamps[("openai_conversations", "chrome136")] = 0.0

        update_mock = MagicMock(return_value=True)
        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.load_credential_state",
                return_value=cred_state,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.update_sentinel_fields",
                new=update_mock,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.transport.get_client",
                new=AsyncMock(return_value=client),
            ),
        ):
            addon = OpenAIConversationsAddon()
            await addon.request(flow)

        update_mock.assert_called_once()
        # The finalize token and solved proof go as two separate headers; the
        # combined openai-sentinel-token blob must NOT be sent.
        assert flow.request.headers.get("openai-sentinel-chat-requirements-token") == "new-chat-req-tok"
        proof = flow.request.headers.get("openai-sentinel-proof-token")
        assert proof is not None and proof.startswith("gAAAAAB") and proof.endswith("~S")
        assert flow.request.headers.get("openai-sentinel-token") is None

    @pytest.mark.asyncio
    async def test_sentinel_not_refreshed_when_valid(self) -> None:
        flow = _make_flow()
        cfg = _make_config(sentinel_skew_seconds=60.0)
        # Far-future expiry → not expired; cached chat_req_token + proof reused.
        cred_state = _make_credential_state(
            chat_req_token="existing-tok",  # noqa: S106
            proof_token="existing-proof",  # noqa: S106
            chat_req_token_expires_at_ms=9999999999000,
        )

        prepare_resps = [
            _prepare_post_response("t1"),
            _prepare_post_response("t2"),
            _prepare_post_response("t3"),
        ]
        client = _make_mock_client(has_cookies=True, post_responses=prepare_resps)

        import time

        _warmup_timestamps[("openai_conversations", "chrome136")] = time.monotonic()

        update_mock = MagicMock(return_value=True)
        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.load_credential_state",
                return_value=cred_state,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.update_sentinel_fields",
                new=update_mock,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.transport.get_client",
                new=AsyncMock(return_value=client),
            ),
        ):
            addon = OpenAIConversationsAddon()
            await addon.request(flow)

        # Sentinel NOT refreshed — update_sentinel_fields should not be called.
        update_mock.assert_not_called()
        # The cached chat-requirements + proof tokens are reused on the stamped headers.
        assert flow.request.headers.get("openai-sentinel-chat-requirements-token") == "existing-tok"
        assert flow.request.headers.get("openai-sentinel-proof-token") == "existing-proof"


# ---------------------------------------------------------------------------
# Addon — conduit prepare chain
# ---------------------------------------------------------------------------


class TestConduitPrepare:
    def setup_method(self) -> None:
        _warmup_timestamps.clear()

    @pytest.mark.asyncio
    async def test_prepare_uses_empty_conduit_token_first(self) -> None:
        """First prepare call must send empty x-conduit-token."""
        flow = _make_flow()
        cfg = _make_config()
        cred_state = _make_credential_state(chat_req_token_expires_at_ms=9999999999000)

        prepare_calls: list[dict[str, Any]] = []

        async def _post_capture(
            url: str,
            *,
            content: bytes = b"",
            headers: dict[str, str] | None = None,
            timeout: Any = None,
            **kw: Any,
        ) -> Any:
            h = headers or {}
            prepare_calls.append({"url": url, "x_conduit_token": h.get("x-conduit-token", "__missing__")})
            r = MagicMock()
            r.status_code = 200
            r.text = ""
            r.json = MagicMock(return_value={"conduit_token": f"tok-{len(prepare_calls)}"})
            return r

        client = _make_mock_client(has_cookies=True)
        client.post = AsyncMock(side_effect=_post_capture)

        import time

        _warmup_timestamps[("openai_conversations", "chrome136")] = time.monotonic()

        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.load_credential_state",
                return_value=cred_state,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.update_sentinel_fields",
                return_value=True,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.transport.get_client",
                new=AsyncMock(return_value=client),
            ),
        ):
            addon = OpenAIConversationsAddon()
            await addon.request(flow)

        # Far-future chat-requirements expiry → no sentinel calls; the only POSTs
        # are the three conduit prepare states (none, sent, success).
        prepare_only = [c for c in prepare_calls if "prepare" in c["url"]]
        assert len(prepare_only) == 3
        # First prepare: always empty conduit token.
        assert prepare_only[0]["x_conduit_token"] == ""
        # Second prepare: non-empty (got token from first prepare).
        assert prepare_only[1]["x_conduit_token"] != ""
        # Third prepare: non-empty (got token from second prepare).
        assert prepare_only[2]["x_conduit_token"] != ""
        # Chaining: each call sends the token returned by the previous one.
        tokens = [prepare_only[i]["x_conduit_token"] for i in range(3)]
        assert tokens[1] != tokens[0]
        assert tokens[2] != tokens[1]

    @pytest.mark.asyncio
    async def test_prepare_and_final_share_turn_trace_id(self) -> None:
        """All three prepare calls AND the final flow must share one x-oai-turn-trace-id."""
        flow = _make_flow()
        cfg = _make_config()
        cred_state = _make_credential_state(chat_req_token_expires_at_ms=9999999999000)

        trace_ids_seen: list[str] = []

        async def _post_capture(
            url: str,
            *,
            content: bytes = b"",
            headers: dict[str, str] | None = None,
            timeout: Any = None,
            **kw: Any,
        ) -> Any:
            h = headers or {}
            tid = h.get("x-oai-turn-trace-id", "")
            if tid:
                trace_ids_seen.append(tid)
            r = MagicMock()
            r.status_code = 200
            r.text = ""
            r.json = MagicMock(return_value={"conduit_token": f"tok-{len(trace_ids_seen)}"})
            return r

        client = _make_mock_client(has_cookies=True)
        client.post = AsyncMock(side_effect=_post_capture)

        import time

        _warmup_timestamps[("openai_conversations", "chrome136")] = time.monotonic()

        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.load_credential_state",
                return_value=cred_state,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.update_sentinel_fields",
                return_value=True,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.transport.get_client",
                new=AsyncMock(return_value=client),
            ),
        ):
            addon = OpenAIConversationsAddon()
            await addon.request(flow)

        prepare_trace_ids = [t for t in trace_ids_seen if t]
        assert len(prepare_trace_ids) == 3
        # All prepare calls must use the SAME trace id.
        assert len(set(prepare_trace_ids)) == 1

        # The final flow must also carry the same trace id.
        flow_trace_id = flow.request.headers.get("x-oai-turn-trace-id", "")
        assert flow_trace_id == prepare_trace_ids[0]

    @pytest.mark.asyncio
    async def test_final_conduit_token_stamped_on_flow(self) -> None:
        flow = _make_flow()
        cfg = _make_config()
        cred_state = _make_credential_state(chat_req_token_expires_at_ms=9999999999000)

        prepare_resps = [
            _prepare_post_response("tok-1"),
            _prepare_post_response("tok-2"),
            _prepare_post_response("FINAL-TOKEN"),
        ]
        client = _make_mock_client(has_cookies=True, post_responses=prepare_resps)

        import time

        _warmup_timestamps[("openai_conversations", "chrome136")] = time.monotonic()

        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.load_credential_state",
                return_value=cred_state,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.update_sentinel_fields",
                return_value=True,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.transport.get_client",
                new=AsyncMock(return_value=client),
            ),
        ):
            addon = OpenAIConversationsAddon()
            await addon.request(flow)

        assert flow.request.headers.get("x-conduit-token") == "FINAL-TOKEN"


# ---------------------------------------------------------------------------
# Addon — same client instance for warmup/sentinel/prepare
# ---------------------------------------------------------------------------


class TestSharedClientInstance:
    def setup_method(self) -> None:
        _warmup_timestamps.clear()

    @pytest.mark.asyncio
    async def test_same_client_used_for_all_operations(self) -> None:
        """All warmup, sentinel, and prepare operations must use the same client instance."""
        flow = _make_flow()
        cfg = _make_config()
        cred_state = _make_credential_state(chat_req_token_expires_at_ms=0)

        post_resps = [
            _sentinel_prepare_response(),
            _sentinel_finalize_response(),
            _prepare_post_response("t1"),
            _prepare_post_response("t2"),
            _prepare_post_response("t3"),
        ]
        client = _make_mock_client(has_cookies=False, post_responses=post_resps)

        get_client_calls: list[tuple[str, str]] = []

        async def _mock_get_client(*, host: str, profile: str, **kw: Any) -> MagicMock:
            get_client_calls.append((host, profile))
            return client

        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.load_credential_state",
                return_value=cred_state,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.update_sentinel_fields",
                return_value=True,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.transport.get_client",
                new=AsyncMock(side_effect=_mock_get_client),
            ),
        ):
            addon = OpenAIConversationsAddon()
            await addon.request(flow)

        # get_client must always be called with the SAME (host, profile) key.
        assert len(get_client_calls) >= 1
        unique_keys = set(get_client_calls)
        assert len(unique_keys) == 1, f"Multiple get_client keys used: {unique_keys}"
        assert get_client_calls[0] == ("chatgpt.com", "chrome136")


# ---------------------------------------------------------------------------
# Addon — X-OAI-IS from cookie jar
# ---------------------------------------------------------------------------


class TestXOaiIs:
    def setup_method(self) -> None:
        _warmup_timestamps.clear()

    @pytest.mark.asyncio
    async def test_x_oai_is_set_when_cookie_present(self) -> None:
        flow = _make_flow()
        cfg = _make_config()
        cred_state = _make_credential_state(chat_req_token_expires_at_ms=9999999999000)

        prepare_resps = [
            _prepare_post_response("t1"),
            _prepare_post_response("t2"),
            _prepare_post_response("t3"),
        ]
        client = _make_mock_client(
            has_cookies=True,
            post_responses=prepare_resps,
            oai_is_cookie="oai-is-value-from-jar",
        )

        import time

        _warmup_timestamps[("openai_conversations", "chrome136")] = time.monotonic()

        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.load_credential_state",
                return_value=cred_state,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.update_sentinel_fields",
                return_value=True,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.transport.get_client",
                new=AsyncMock(return_value=client),
            ),
        ):
            addon = OpenAIConversationsAddon()
            await addon.request(flow)

        assert flow.request.headers.get("x-oai-is") == "oai-is-value-from-jar"

    @pytest.mark.asyncio
    async def test_x_oai_is_absent_when_no_cookie(self) -> None:
        flow = _make_flow()
        cfg = _make_config()
        cred_state = _make_credential_state(chat_req_token_expires_at_ms=9999999999000)

        prepare_resps = [
            _prepare_post_response("t1"),
            _prepare_post_response("t2"),
            _prepare_post_response("t3"),
        ]
        client = _make_mock_client(has_cookies=True, post_responses=prepare_resps, oai_is_cookie="")

        import time

        _warmup_timestamps[("openai_conversations", "chrome136")] = time.monotonic()

        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.load_credential_state",
                return_value=cred_state,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.update_sentinel_fields",
                return_value=True,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.transport.get_client",
                new=AsyncMock(return_value=client),
            ),
        ):
            addon = OpenAIConversationsAddon()
            await addon.request(flow)

        assert "x-oai-is" not in flow.request.headers


# ---------------------------------------------------------------------------
# Addon — response retry
# ---------------------------------------------------------------------------


class TestResponseRetry:
    def setup_method(self) -> None:
        _warmup_timestamps.clear()

    @pytest.mark.asyncio
    async def test_retry_runs_once_on_401(self) -> None:
        flow = _make_flow(status_code=401)
        cfg = _make_config()
        cred_state = _make_credential_state(chat_req_token_expires_at_ms=0)

        # _retry_once runs the sentinel prepare→finalize then replays via client.request.
        post_resps = [_sentinel_prepare_response(), _sentinel_finalize_response("new-tok")]
        client = _make_mock_client(has_cookies=True, post_responses=post_resps)

        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.load_credential_state",
                return_value=cred_state,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.update_sentinel_fields",
                return_value=True,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.transport.get_client",
                new=AsyncMock(return_value=client),
            ),
        ):
            addon = OpenAIConversationsAddon()
            await addon.response(flow)

        # Retry was performed.
        client.request.assert_called_once()
        # Loop guard must be set.
        assert flow.metadata[_RETRY_DONE_KEY] is True

    @pytest.mark.asyncio
    async def test_retry_does_not_loop(self) -> None:
        """Second response with 401 must not trigger another retry."""
        flow = _make_flow(status_code=401)
        flow.metadata[_RETRY_DONE_KEY] = True

        cfg = _make_config()
        client = _make_mock_client(has_cookies=True)

        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.transport.get_client",
                new=AsyncMock(return_value=client),
            ),
        ):
            addon = OpenAIConversationsAddon()
            await addon.response(flow)

        client.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_marks_auth_injected_false_to_prevent_auth_addon_loop(self) -> None:
        """After retry, ccproxy.auth_injected must be False so AuthAddon skips."""
        flow = _make_flow(status_code=401)
        cfg = _make_config()
        cred_state = _make_credential_state(chat_req_token_expires_at_ms=0)

        post_resps = [_sentinel_prepare_response(), _sentinel_finalize_response()]
        client = _make_mock_client(has_cookies=True, post_responses=post_resps)

        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.load_credential_state",
                return_value=cred_state,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.update_sentinel_fields",
                return_value=True,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.transport.get_client",
                new=AsyncMock(return_value=client),
            ),
        ):
            addon = OpenAIConversationsAddon()
            await addon.response(flow)

        assert flow.metadata.get("ccproxy.auth_injected") is False

    @pytest.mark.asyncio
    async def test_retry_fires_on_cf_challenge(self) -> None:
        """CF-mitigated responses also trigger the one-shot retry."""
        flow = _make_flow(
            status_code=403,
            response_headers={"cf-mitigated": "challenge"},
        )
        cfg = _make_config()
        cred_state = _make_credential_state(chat_req_token_expires_at_ms=0)

        client = _make_mock_client(
            has_cookies=True,
            post_responses=[_sentinel_prepare_response(), _sentinel_finalize_response()],
        )

        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.load_credential_state",
                return_value=cred_state,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.update_sentinel_fields",
                return_value=True,
            ),
            patch(
                "ccproxy.inspector.openai_conversations_addon.transport.get_client",
                new=AsyncMock(return_value=client),
            ),
        ):
            addon = OpenAIConversationsAddon()
            await addon.response(flow)

        assert flow.metadata[_RETRY_DONE_KEY] is True


# ---------------------------------------------------------------------------
# Addon — ConversationStore write-back
# ---------------------------------------------------------------------------


class TestConversationStoreWriteBack:
    @pytest.mark.asyncio
    async def test_writes_back_ids_from_sse(self) -> None:
        sse = (
            'data: {"conversation_id": "server-conv-1", '
            '"message": {"id": "parent-1", "author": {"role": "assistant"}}}\n\n'
            "data: [DONE]\n\n"
        )
        flow = _make_flow(status_code=200, response_content=sse.encode())
        flow.metadata["ccproxy.conversation_id"] = "sha12-ccproxy-key"

        cfg = _make_config()
        store = ConversationStore()

        addon = OpenAIConversationsAddon()
        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.get_conversation_store",
                return_value=store,
            ),
        ):
            await addon._write_back_conversation(flow)

        cached = store.get("sha12-ccproxy-key")
        assert cached is not None
        assert cached.conversation_id == "server-conv-1"
        assert cached.parent_message_id == "parent-1"

    @pytest.mark.asyncio
    async def test_no_write_when_status_is_error(self) -> None:
        flow = _make_flow(status_code=500, response_content=b"error")
        flow.metadata["ccproxy.conversation_id"] = "sha12-key"

        cfg = _make_config()
        store = ConversationStore()

        addon = OpenAIConversationsAddon()
        with (
            patch("ccproxy.inspector.openai_conversations_addon.get_config", return_value=cfg),
            patch(
                "ccproxy.inspector.openai_conversations_addon.get_conversation_store",
                return_value=store,
            ),
        ):
            await addon._write_back_conversation(flow)

        assert store.get("sha12-key") is None
