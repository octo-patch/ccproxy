"""Tests for the cache_breakpoints pipeline hook."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import CachePoint, ModelRequest, UserPromptPart

from ccproxy.auth.sources import CommandAuthSource
from ccproxy.config import CCProxyConfig, Provider, set_config_instance
from ccproxy.hooks.cache_breakpoints import (
    POLICY_HEADER,
    cache_breakpoints,
    cache_breakpoints_guard,
    parse_policy_header,
)
from ccproxy.lightllm.cache_policy import CachePolicy
from ccproxy.pipeline.context import Context

_BODY = {
    "model": "claude-sonnet-5",
    "max_tokens": 128,
    "system": "You are helpful.",
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "one"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": [{"type": "text", "text": "two"}]},
    ],
}


def _make_ctx(headers: dict[str, str] | None = None, *, auth_provider: str = "anthropic") -> Context:
    flow = MagicMock()
    flow.id = "test-flow"
    flow.request.path = "/v1/messages"
    flow.request.content = json.dumps(_BODY).encode()
    flow.request.headers = dict(headers or {})
    flow.metadata = {}
    ctx = Context.from_flow(flow)
    ctx.metadata.auth_provider = auth_provider
    return ctx


def _provider(type_: str) -> Provider:
    return Provider(
        auth=CommandAuthSource(command="printf tok"),
        base_url="https://api.example.com",
        path="/v1/messages",
        type=type_,
    )


@pytest.fixture
def config() -> CCProxyConfig:
    cfg = CCProxyConfig()
    cfg.providers = {"anthropic": _provider("anthropic"), "openai": _provider("openai")}
    set_config_instance(cfg)
    return cfg


def _count_user_markers(ctx: Context) -> int:
    count = 0
    for msg in ctx.messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, list):
                    count += sum(1 for item in part.content if isinstance(item, CachePoint))
    return count


class TestHeaderDsl:
    def test_full_example_parses(self) -> None:
        policy = parse_policy_header("tools=1h,system=1h,user_tail=2:5m")
        assert policy == CachePolicy(tools="1h", system="1h", user_tail=2, user_tail_ttl="5m")

    def test_bare_keys_default_5m(self) -> None:
        policy = parse_policy_header("tools,system")
        assert policy == CachePolicy(tools="5m", system="5m")

    def test_user_tail_without_ttl_defaults_5m(self) -> None:
        policy = parse_policy_header("user_tail=3")
        assert policy == CachePolicy(user_tail=3, user_tail_ttl="5m")

    def test_bad_user_tail_count_raises_naming_token(self) -> None:
        with pytest.raises(ValueError, match=r"user_tail=x"):
            parse_policy_header("tools,user_tail=x")

    def test_bad_ttl_raises_naming_token(self) -> None:
        with pytest.raises(ValueError, match=r"tools=24h"):
            parse_policy_header("tools=24h")

    def test_unknown_key_raises_naming_token(self) -> None:
        with pytest.raises(ValueError, match=r"assistant_tail=2"):
            parse_policy_header("assistant_tail=2")

    def test_bad_header_via_hook_raises(self, config: CCProxyConfig) -> None:
        ctx = _make_ctx({POLICY_HEADER: "user_tail=x"})
        with pytest.raises(ValueError, match=r"user_tail=x"):
            cache_breakpoints(ctx, {})


class TestPrecedence:
    def test_header_beats_config_params(self, config: CCProxyConfig) -> None:
        ctx = _make_ctx({POLICY_HEADER: "user_tail=1"})
        cache_breakpoints(ctx, {"user_tail": 2, "user_tail_ttl": "5m", "tools": None, "system": None})
        assert _count_user_markers(ctx) == 1

    def test_config_params_apply_without_header(self, config: CCProxyConfig) -> None:
        ctx = _make_ctx()
        cache_breakpoints(ctx, {"user_tail": 2, "user_tail_ttl": "5m", "tools": None, "system": None})
        assert _count_user_markers(ctx) == 2

    def test_guard_false_when_no_policy_source(self, config: CCProxyConfig) -> None:
        ctx = _make_ctx()
        assert cache_breakpoints_guard(ctx) is False

    def test_guard_true_with_header(self, config: CCProxyConfig) -> None:
        ctx = _make_ctx({POLICY_HEADER: "tools"})
        assert cache_breakpoints_guard(ctx) is True

    def test_guard_true_with_config_params(self, config: CCProxyConfig) -> None:
        hooks: dict[str, list[Any]] = dict(config.hooks)
        hooks["outbound"] = [
            *hooks.get("outbound", []),
            {"hook": "ccproxy.hooks.cache_breakpoints", "params": {"user_tail": 2}},
        ]
        config.hooks = hooks
        ctx = _make_ctx()
        assert cache_breakpoints_guard(ctx) is True


class TestProviderScope:
    def test_guard_false_for_non_anthropic_upstream_with_header(self, config: CCProxyConfig) -> None:
        ctx = _make_ctx({POLICY_HEADER: "tools=1h"}, auth_provider="openai")
        assert cache_breakpoints_guard(ctx) is False

    def test_guard_false_for_unknown_provider(self, config: CCProxyConfig) -> None:
        ctx = _make_ctx({POLICY_HEADER: "tools=1h"}, auth_provider="")
        assert cache_breakpoints_guard(ctx) is False

    def test_guard_true_for_anthropic_compatible_fork(self, config: CCProxyConfig) -> None:
        config.providers["zai"] = _provider("anthropic")
        ctx = _make_ctx({POLICY_HEADER: "tools=1h"}, auth_provider="zai")
        assert cache_breakpoints_guard(ctx) is True


class TestHeaderHygiene:
    def test_control_header_stripped_after_consumption(self, config: CCProxyConfig) -> None:
        ctx = _make_ctx({POLICY_HEADER: "tools=1h,system=1h,user_tail=2:5m"})
        cache_breakpoints(ctx, {})
        assert ctx.get_header(POLICY_HEADER) == ""
