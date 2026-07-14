"""Tests for the inject_auth hook."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from mitmproxy.http import HTTPFlow

from ccproxy.auth.sources import CommandAuthSource
from ccproxy.config import CCProxyConfig, Provider, set_config_instance
from ccproxy.constants import AUTH_SENTINEL_PREFIX, AuthConfigError
from ccproxy.hooks.inject_auth import (
    inject_auth,
    inject_auth_guard,
    inject_provider_auth,
)
from ccproxy.pipeline.context import Context


class _ExtraHeaderAuthSource(CommandAuthSource):
    def extra_headers(self, label: str = "Auth") -> dict[str, str]:
        return {"ChatGPT-Account-ID": "acct_test"}


def _make_ctx(headers: dict[str, str] | None = None) -> Context:
    """Context with a plain dict for headers so mutations are observable."""
    flow = MagicMock()
    flow.id = "test-flow"
    flow.request.content = json.dumps({"model": "test-model", "messages": []}).encode()
    flow.request.headers = dict(headers or {})
    flow.request.query = {}
    flow.metadata = {}
    return Context.from_flow(flow)


def _flow(ctx: Context) -> HTTPFlow:
    assert ctx.flow is not None
    return ctx.flow


def _literal(value: str) -> str:
    return value


def _make_provider(*, value: str = "tok", header: str | None = None) -> Provider:
    """Build a Provider whose auth.resolve() returns ``value`` via shell echo."""
    return Provider(
        auth=CommandAuthSource(command=f"printf '%s' {value}", header=header),
        base_url="https://api.example.com",
        path="/v1/messages",
        type="anthropic",
    )


@pytest.fixture
def clean_config() -> CCProxyConfig:
    config = CCProxyConfig()
    set_config_instance(config)
    return config


class TestInjectAuthGuard:
    def test_true_when_x_api_key_set(self, clean_config: CCProxyConfig) -> None:
        ctx = _make_ctx({"x-api-key": "some-key"})
        assert inject_auth_guard(ctx) is True

    def test_true_when_authorization_set(self, clean_config: CCProxyConfig) -> None:
        ctx = _make_ctx({"authorization": "Bearer token"})
        assert inject_auth_guard(ctx) is True

    def test_true_when_x_goog_api_key_set(self, clean_config: CCProxyConfig) -> None:
        ctx = _make_ctx({"x-goog-api-key": "google-key"})
        assert inject_auth_guard(ctx) is True

    def test_false_when_all_empty(self, clean_config: CCProxyConfig) -> None:
        ctx = _make_ctx()
        assert inject_auth_guard(ctx) is False

    def test_true_when_multiple_headers_set(self, clean_config: CCProxyConfig) -> None:
        ctx = _make_ctx({"x-api-key": "key", "authorization": "Bearer tok"})
        assert inject_auth_guard(ctx) is True


class TestInjectAuthSentinelPath:
    def test_sentinel_injects_bearer_and_sets_metadata(self, clean_config: CCProxyConfig) -> None:
        clean_config.providers = {"anthropic": _make_provider(value="real-token-xyz")}
        ctx = _make_ctx({"x-api-key": f"{AUTH_SENTINEL_PREFIX}anthropic"})

        result = inject_auth(ctx, {})

        assert result is ctx
        assert ctx.get_header("authorization") == "Bearer real-token-xyz"
        assert _flow(ctx).metadata["ccproxy.auth_injected"] is True
        assert _flow(ctx).metadata["ccproxy.auth_provider"] == "anthropic"

    def test_sentinel_clears_x_api_key(self, clean_config: CCProxyConfig) -> None:
        clean_config.providers = {"anthropic": _make_provider(value="real-token")}
        ctx = _make_ctx({"x-api-key": f"{AUTH_SENTINEL_PREFIX}anthropic"})

        inject_auth(ctx, {})

        # x-api-key must be cleared since default target is authorization
        assert ctx.get_header("x-api-key") == ""

    def test_sentinel_via_goog_api_key_header(self, clean_config: CCProxyConfig) -> None:
        clean_config.providers = {"google": _make_provider(value="goog-token")}
        ctx = _make_ctx({"x-goog-api-key": f"{AUTH_SENTINEL_PREFIX}google"})

        result = inject_auth(ctx, {})

        assert result is ctx
        assert ctx.get_header("authorization") == "Bearer goog-token"
        assert _flow(ctx).metadata["ccproxy.auth_provider"] == "google"

    def test_sentinel_via_authorization_bearer(self, clean_config: CCProxyConfig) -> None:
        """OpenAI clients send the sentinel as ``Authorization: Bearer <key>``."""
        clean_config.providers = {"anthropic": _make_provider(value="real-bearer-token")}
        ctx = _make_ctx({"authorization": f"Bearer {AUTH_SENTINEL_PREFIX}anthropic"})

        result = inject_auth(ctx, {})

        assert result is ctx
        # The Bearer-token sentinel was peeled, the real token re-injected with Bearer
        assert ctx.get_header("authorization") == "Bearer real-bearer-token"
        assert _flow(ctx).metadata["ccproxy.auth_provider"] == "anthropic"

    def test_sentinel_via_authorization_bearer_with_custom_target(
        self,
        clean_config: CCProxyConfig,
    ) -> None:
        """Inbound Authorization can route to a different outbound header."""
        clean_config.providers = {"deepseek": _make_provider(value="ds-token", header="x-api-key")}
        ctx = _make_ctx({"authorization": f"Bearer {AUTH_SENTINEL_PREFIX}deepseek"})

        inject_auth(ctx, {})

        assert ctx.get_header("x-api-key") == "ds-token"
        # Source authorization header cleared so the sentinel doesn't leak.
        assert ctx.get_header("authorization") == ""
        assert _flow(ctx).metadata["ccproxy.auth_provider"] == "deepseek"

    def test_sentinel_stamps_companion_auth_headers(self, clean_config: CCProxyConfig) -> None:
        clean_config.providers = {
            "codex": Provider(
                auth=_ExtraHeaderAuthSource(command="printf '%s' codex-token"),
                base_url="https://chatgpt.com",
                path="/backend-api/codex/responses",
                type="openai_responses",
            )
        }
        ctx = _make_ctx({"authorization": f"Bearer {AUTH_SENTINEL_PREFIX}codex"})

        inject_auth(ctx, {})

        assert ctx.get_header("authorization") == "Bearer codex-token"
        assert ctx.get_header("ChatGPT-Account-ID") == "acct_test"
        assert _flow(ctx).metadata["ccproxy.auth_provider"] == "codex"

    def test_sentinel_no_token_raises_auth_config_error(self, clean_config: CCProxyConfig) -> None:
        ctx = _make_ctx({"x-api-key": f"{AUTH_SENTINEL_PREFIX}missing-provider"})

        with pytest.raises(AuthConfigError, match="missing-provider"):
            inject_auth(ctx, {})

    def test_sentinel_provider_without_auth_is_rejected(self, clean_config: CCProxyConfig) -> None:
        clean_config.providers = {
            "public": Provider(
                base_url="https://api.example.com",
                path="/v1/chat/completions",
                type="openai",
            )
        }
        ctx = _make_ctx({"authorization": f"Bearer {AUTH_SENTINEL_PREFIX}public"})

        with pytest.raises(AuthConfigError, match="no auth source"):
            inject_auth(ctx, {})

    def test_sentinel_get_config_exception_raises_auth_config_error(self) -> None:
        ctx = _make_ctx({"x-api-key": f"{AUTH_SENTINEL_PREFIX}err-provider"})

        with (
            patch("ccproxy.hooks.inject_auth.get_config", side_effect=RuntimeError("config exploded")),
            pytest.raises(AuthConfigError, match="err-provider"),
        ):
            inject_auth(ctx, {})


class TestInjectAuthPassthrough:
    def test_non_sentinel_api_key_no_injection(self, clean_config: CCProxyConfig) -> None:
        ctx = _make_ctx({"x-api-key": "sk-real-key-not-a-sentinel"})

        result = inject_auth(ctx, {})

        assert result is ctx
        assert "ccproxy.auth_injected" not in _flow(ctx).metadata
        assert "ccproxy.auth_provider" not in _flow(ctx).metadata

    def test_real_auth_header_passes_through(self, clean_config: CCProxyConfig) -> None:
        clean_config.providers = {"anthropic": _make_provider(value="some-tok")}
        ctx = _make_ctx({"authorization": "Bearer real-existing-token"})

        result = inject_auth(ctx, {})

        assert result is ctx
        assert ctx.get_header("authorization") == "Bearer real-existing-token"
        assert "ccproxy.auth_injected" not in _flow(ctx).metadata


class TestInjectProviderAuth:
    def test_default_header_sets_authorization_bearer(self, clean_config: CCProxyConfig) -> None:
        clean_config.providers = {"anthropic": _make_provider()}
        ctx = _make_ctx()

        inject_provider_auth(ctx, "anthropic", token=_literal("my-token"))

        assert ctx.get_header("authorization") == "Bearer my-token"
        assert _flow(ctx).metadata["ccproxy.auth_injected"] is True
        assert ctx.get_header("x-api-key") == ""
        assert ctx.get_header("x-goog-api-key") == ""

    def test_custom_goog_api_key_header(self, clean_config: CCProxyConfig) -> None:
        clean_config.providers = {"google": _make_provider(header="x-goog-api-key")}
        ctx = _make_ctx()

        inject_provider_auth(ctx, "google", token=_literal("goog-token"))

        assert ctx.get_header("x-goog-api-key") == "goog-token"
        assert _flow(ctx).metadata["ccproxy.auth_injected"] is True
        # x-api-key cleared (not the target)
        assert ctx.get_header("x-api-key") == ""
        # authorization not touched
        assert ctx.get_header("authorization") == ""

    def test_custom_x_api_key_header(self, clean_config: CCProxyConfig) -> None:
        clean_config.providers = {"prov": _make_provider(header="x-api-key")}
        ctx = _make_ctx()

        inject_provider_auth(ctx, "prov", token=_literal("my-secret"))

        assert ctx.get_header("x-api-key") == "my-secret"
        assert ctx.get_header("x-goog-api-key") == ""
        assert _flow(ctx).metadata["ccproxy.auth_injected"] is True

    def test_always_sets_injected_flag(self, clean_config: CCProxyConfig) -> None:
        clean_config.providers = {"any": _make_provider()}
        ctx = _make_ctx()
        inject_provider_auth(ctx, "any", token=_literal("any-token"))
        assert _flow(ctx).metadata["ccproxy.auth_injected"] is True

    def test_inject_preserves_other_headers(self, clean_config: CCProxyConfig) -> None:
        clean_config.providers = {"prov": _make_provider()}
        ctx = _make_ctx({"content-type": "application/json", "anthropic-version": "2023-06-01"})

        inject_provider_auth(ctx, "prov", token=_literal("tok"))

        assert ctx.get_header("content-type") == "application/json"
        assert ctx.get_header("anthropic-version") == "2023-06-01"

    def test_model_selected_provider_replaces_prior_sentinel_auth(self, clean_config: CCProxyConfig) -> None:
        clean_config.providers = {
            "first": _make_provider(),
            "second": _make_provider(header="x-api-key"),
        }
        ctx = _make_ctx()
        inject_provider_auth(ctx, "first", token=_literal("first-token"))

        inject_provider_auth(ctx, "second", token=_literal("second-token"))

        assert ctx.get_header("authorization") == ""
        assert ctx.get_header("x-api-key") == "second-token"
        assert _flow(ctx).metadata["ccproxy.auth_provider"] == "second"

    def test_provider_switch_removes_prior_query_credential(self, clean_config: CCProxyConfig) -> None:
        first = Provider(
            auth=CommandAuthSource(command="printf first", query_param="key"),
            base_url="https://first.example",
            type="gemini",
        )
        second = _make_provider(header="x-api-key")
        clean_config.providers = {"first": first, "second": second}
        ctx = _make_ctx()

        inject_provider_auth(ctx, "first", token=_literal("first-token"))
        inject_provider_auth(ctx, "second", token=_literal("second-token"))

        assert "key" not in _flow(ctx).request.query
        assert ctx.get_header("x-api-key") == "second-token"
        assert _flow(ctx).metadata["ccproxy.auth_query_param"] == ""
