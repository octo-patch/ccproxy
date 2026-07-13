"""Tests for the Anthropic cache-breakpoint policy engine."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, cast
from unittest.mock import MagicMock

from pydantic_ai.messages import (
    CachePoint,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings

from ccproxy.lightllm import CachePolicy, apply_cache_policy
from ccproxy.lightllm.adapters.anthropic import AnthropicAdapter
from ccproxy.lightllm.cache_policy import _is_sentinel
from ccproxy.pipeline.context import Context

FULL_POLICY = CachePolicy(tools="1h", system="1h", user_tail=2, user_tail_ttl="5m")


def _convo() -> list[ModelMessage]:
    """System + three user turns interleaved with assistant replies."""
    return [
        ModelRequest(parts=[SystemPromptPart(content="sys"), UserPromptPart(content="one")]),
        ModelResponse(parts=[TextPart(content="a")]),
        ModelRequest(parts=[UserPromptPart(content="two")]),
        ModelResponse(parts=[TextPart(content="b")]),
        ModelRequest(parts=[UserPromptPart(content="three")]),
    ]


def _count_markers(messages: list[ModelMessage], settings: ModelSettings) -> int:
    count = 0
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, list):
                    count += sum(1 for item in part.content if isinstance(item, CachePoint))
    settings_dict = cast(dict[str, Any], settings)
    count += sum(
        1
        for key in (
            "anthropic_cache_tool_definitions",
            "anthropic_cache_instructions",
            "anthropic_cache_messages",
            "anthropic_cache",
        )
        if settings_dict.get(key)
    )
    return count


class TestPlacementInIsolation:
    def test_tools_sets_settings_knob_and_leaves_messages_untouched(self) -> None:
        messages = _convo()
        new_messages, new_settings, report = apply_cache_policy(messages, ModelSettings(), CachePolicy(tools="1h"))
        assert cast(dict[str, Any], new_settings)["anthropic_cache_tool_definitions"] == "1h"
        assert new_messages == messages
        assert report.placed == ["tools"]
        assert report.dropped == []
        assert report.skipped == []

    def test_system_appends_sentinel_after_last_system_part(self) -> None:
        new_messages, _, report = apply_cache_policy(_convo(), ModelSettings(), CachePolicy(system="1h"))
        first = cast(ModelRequest, new_messages[0])
        assert isinstance(first.parts[0], SystemPromptPart)
        sentinel = first.parts[1]
        assert _is_sentinel(sentinel)
        assert cast(list[Any], cast(UserPromptPart, sentinel).content)[0].ttl == "1h"
        assert report.placed == ["system"]

    def test_user_tail_one_promotes_str_content(self) -> None:
        new_messages, _, report = apply_cache_policy(
            _convo(), ModelSettings(), CachePolicy(user_tail=1, user_tail_ttl="5m")
        )
        last = cast(ModelRequest, new_messages[-1])
        part = cast(UserPromptPart, last.parts[0])
        assert isinstance(part.content, list)
        assert part.content[0] == "three"
        assert isinstance(part.content[1], CachePoint)
        assert part.content[1].ttl == "5m"
        assert report.placed == ["user[-1]"]

    def test_user_tail_two_marks_newest_two_user_messages(self) -> None:
        new_messages, _, report = apply_cache_policy(_convo(), ModelSettings(), CachePolicy(user_tail=2))
        assert report.placed == ["user[-1]", "user[-2]"]
        for idx, text in ((-1, "three"), (2, "two")):
            part = cast(UserPromptPart, cast(ModelRequest, new_messages[idx]).parts[0])
            assert isinstance(part.content, list)
            assert part.content == [text, part.content[1]]
            assert isinstance(part.content[1], CachePoint)
        # First user message untouched.
        first_user = cast(UserPromptPart, cast(ModelRequest, new_messages[0]).parts[1])
        assert first_user.content == "one"


class TestBudgetArbitration:
    def test_full_canonical_policy_places_exactly_four(self) -> None:
        new_messages, new_settings, report = apply_cache_policy(_convo(), ModelSettings(), FULL_POLICY)
        assert report.placed == ["tools", "system", "user[-1]", "user[-2]"]
        assert report.dropped == []
        assert _count_markers(new_messages, new_settings) == 4

    def test_authored_markers_win_and_policy_yields(self) -> None:
        messages = _convo()
        messages[2] = ModelRequest(parts=[UserPromptPart(content=["two", CachePoint(ttl="5m")])])
        messages[4] = ModelRequest(parts=[UserPromptPart(content=["three", CachePoint(ttl="1h")])])
        authored = deepcopy(messages)

        new_messages, new_settings, report = apply_cache_policy(messages, ModelSettings(), FULL_POLICY)
        assert report.existing == 2
        assert report.placed == ["tools", "system"]
        # Marked user messages skip as redundant; nothing is removed or re-TTLed.
        assert report.skipped == ["user[-1]", "user[-2]"]
        assert new_messages[2] == authored[2]
        assert new_messages[4] == authored[4]
        assert _count_markers(new_messages, new_settings) == 4

    def test_authored_markers_elsewhere_drop_policy_tail(self) -> None:
        messages = _convo()
        # Two authored markers on the OLDEST user message (mid-content) — policy
        # user_tail targets are unmarked, so overflow drops them.
        messages[0] = ModelRequest(
            parts=[
                SystemPromptPart(content="sys"),
                UserPromptPart(content=["one", CachePoint(ttl="5m"), "more", CachePoint(ttl="5m")]),
            ]
        )
        _, _, report = apply_cache_policy(messages, ModelSettings(), FULL_POLICY)
        assert report.existing == 2
        assert report.placed == ["tools", "system"]
        assert report.dropped == ["user[-1]", "user[-2]"]

    def test_census_counts_settings_shorthand(self) -> None:
        settings = cast(ModelSettings, {"anthropic_cache": True})
        _, _, report = apply_cache_policy(_convo(), settings, FULL_POLICY)
        assert report.existing == 1
        assert report.placed == ["tools", "system", "user[-1]"]
        assert report.dropped == ["user[-2]"]

    def test_census_counts_raw_extras_cc_entries(self) -> None:
        raw_extras = {"cc:msg:0:block:0": {"type": "ephemeral", "ttl": "30m"}}
        _, _, report = apply_cache_policy(_convo(), ModelSettings(), FULL_POLICY, raw_extras=raw_extras)
        assert report.existing == 1
        assert report.placed == ["tools", "system", "user[-1]"]
        assert report.dropped == ["user[-2]"]

    def test_census_counts_tools_override_markers_and_skips_tools_placement(self) -> None:
        raw_extras = {
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"name": "read", "input_schema": {}, "cache_control": {"type": "ephemeral", "ttl": "5m"}},
            ]
        }
        _, new_settings, report = apply_cache_policy(_convo(), ModelSettings(), FULL_POLICY, raw_extras=raw_extras)
        assert report.existing == 1
        assert report.skipped == ["tools"]
        assert report.placed == ["system", "user[-1]", "user[-2]"]
        assert "anthropic_cache_tool_definitions" not in cast(dict[str, Any], new_settings)

    def test_tools_override_without_markers_still_skips_tools_placement(self) -> None:
        """The skip rides on the stitch-time overwrite (wire no-op), not on marker presence."""
        raw_extras = {"tools": [{"type": "web_search_20250305", "name": "web_search"}]}
        _, new_settings, report = apply_cache_policy(_convo(), ModelSettings(), FULL_POLICY, raw_extras=raw_extras)
        assert report.existing == 0
        assert report.skipped == ["tools"]
        assert report.placed == ["system", "user[-1]", "user[-2]"]
        assert "anthropic_cache_tool_definitions" not in cast(dict[str, Any], new_settings)

    def test_idempotence(self) -> None:
        m1, s1, r1 = apply_cache_policy(_convo(), ModelSettings(), FULL_POLICY)
        assert len(r1.placed) == 4
        m2, s2, r2 = apply_cache_policy(m1, s1, FULL_POLICY)
        assert r2.placed == []
        assert r2.skipped == ["tools", "system", "user[-1]", "user[-2]"]
        assert r2.existing == 4
        assert _count_markers(m2, s2) == _count_markers(m1, s1) == 4
        assert m2 == m1

    def test_no_system_parts_skips_system(self) -> None:
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="hi")])]
        _, _, report = apply_cache_policy(messages, ModelSettings(), CachePolicy(system="5m"))
        assert report.placed == []
        assert report.skipped == ["system"]

    def test_user_tail_shortfall_reported(self) -> None:
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="hi")])]
        _, _, report = apply_cache_policy(messages, ModelSettings(), CachePolicy(user_tail=3))
        assert report.placed == ["user[-1]"]
        assert report.skipped == ["user[-2]", "user[-3]"]


class TestPurity:
    def test_inputs_unmodified(self) -> None:
        messages = _convo()
        snapshot = deepcopy(messages)
        settings = ModelSettings()
        settings_snapshot = dict(cast(dict[str, Any], settings))

        new_messages, new_settings, _ = apply_cache_policy(messages, settings, FULL_POLICY)
        assert messages == snapshot
        assert dict(cast(dict[str, Any], settings)) == settings_snapshot
        assert new_messages is not messages
        assert new_settings is not settings


class TestIntegration:
    def test_canonical_policy_renders_wire_markers(self) -> None:
        body = {
            "model": "claude-sonnet-5",
            "max_tokens": 128,
            "system": "You are helpful.",
            "tools": [{"name": "read", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "one"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
                {"role": "user", "content": [{"type": "text", "text": "two"}]},
            ],
        }
        flow = MagicMock()
        flow.id = "test-flow"
        flow.request.path = "/v1/messages"
        flow.request.content = json.dumps(body).encode()
        flow.request.headers = {}
        flow.metadata = {}
        ctx = Context.from_flow(flow)

        new_messages, new_settings, report = apply_cache_policy(
            ctx.messages, ctx.settings, FULL_POLICY, raw_extras=ctx.raw_extras
        )
        assert report.placed == ["tools", "system", "user[-1]", "user[-2]"]
        ctx.messages = new_messages
        ctx.settings = new_settings

        rendered = json.loads(AnthropicAdapter.render(ctx))
        assert rendered["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        system = rendered["system"]
        assert isinstance(system, list)
        assert system[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        user_turns = [m for m in rendered["messages"] if m["role"] == "user"]
        for turn in user_turns:
            assert turn["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}

        wire_markers = sum(
            1
            for container in (rendered["tools"], system, *(m["content"] for m in rendered["messages"]))
            for block in container
            if isinstance(block, dict) and "cache_control" in block
        )
        assert wire_markers <= 4

    def test_typed_tool_override_stays_within_breakpoint_budget(self) -> None:
        """A verbatim tools override carrying a marker is censused; policy places around it.

        Regression for the census blind spot: the typed override routes the client's
        tool marker into raw_extras['tools'], which the engine must count as existing
        or system + user_tail placements could push the wire past 4 breakpoints.
        """
        body = {
            "model": "claude-sonnet-5",
            "max_tokens": 128,
            "system": "You are helpful.",
            "tools": [
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
                {
                    "name": "read",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "one"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
                {"role": "user", "content": [{"type": "text", "text": "two"}]},
            ],
        }
        flow = MagicMock()
        flow.id = "test-flow"
        flow.request.path = "/v1/messages"
        flow.request.content = json.dumps(body).encode()
        flow.request.headers = {}
        flow.metadata = {}
        ctx = Context.from_flow(flow)

        policy = CachePolicy(tools="1h", system="1h", user_tail=2)
        new_messages, new_settings, report = apply_cache_policy(
            ctx.messages, ctx.settings, policy, raw_extras=ctx.raw_extras
        )
        assert report.existing == 1
        assert report.skipped == ["tools"]
        assert report.placed == ["system", "user[-1]", "user[-2]"]
        ctx.messages = new_messages
        ctx.settings = new_settings

        rendered = json.loads(AnthropicAdapter.render(ctx))
        assert rendered["tools"] == body["tools"]

        wire_markers = sum(
            1
            for container in (rendered["tools"], rendered["system"], *(m["content"] for m in rendered["messages"]))
            for block in container
            if isinstance(block, dict) and "cache_control" in block
        )
        assert wire_markers == 4

    def test_deferred_tools_stay_within_breakpoint_budget(self) -> None:
        """4 stable + 2 deferred tools: exactly 4 wire markers, tool marker on stable tool 4."""
        body = {
            "model": "claude-sonnet-5",
            "max_tokens": 128,
            "system": "You are helpful.",
            "tools": [
                *({"name": f"stable_{i}", "input_schema": {"type": "object"}} for i in range(4)),
                *(
                    {"name": f"deferred_{i}", "input_schema": {"type": "object"}, "defer_loading": True}
                    for i in range(2)
                ),
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "one"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
                {"role": "user", "content": [{"type": "text", "text": "two"}]},
            ],
        }
        flow = MagicMock()
        flow.id = "test-flow"
        flow.request.path = "/v1/messages"
        flow.request.content = json.dumps(body).encode()
        flow.request.headers = {}
        flow.metadata = {}
        ctx = Context.from_flow(flow)

        policy = CachePolicy(tools="1h", system="1h", user_tail=2)
        new_messages, new_settings, report = apply_cache_policy(
            ctx.messages, ctx.settings, policy, raw_extras=ctx.raw_extras
        )
        assert report.placed == ["tools", "system", "user[-1]", "user[-2]"]
        ctx.messages = new_messages
        ctx.settings = new_settings

        rendered = json.loads(AnthropicAdapter.render(ctx))
        tools = rendered["tools"]
        assert [("cache_control" in t) for t in tools] == [False, False, False, True, False, False]
        assert tools[3]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert [t.get("defer_loading") for t in tools] == [None, None, None, None, True, True]

        system = rendered["system"]
        wire_markers = sum(
            1
            for container in (tools, system, *(m["content"] for m in rendered["messages"]))
            for block in container
            if isinstance(block, dict) and "cache_control" in block
        )
        assert wire_markers == 4
