"""Tests for pipeline/overrides.py hook override header parsing."""

from __future__ import annotations

import logging

import pytest

from ccproxy.pipeline.overrides import (
    HookOverride,
    OverrideSet,
    extract_overrides_from_context,
    parse_overrides,
)


class TestParseOverrides:
    def test_none_returns_empty(self) -> None:
        result = parse_overrides(None)
        assert result.overrides == {}
        assert result.raw_header == ""

    def test_empty_string_returns_empty(self) -> None:
        result = parse_overrides("")
        assert result.overrides == {}

    def test_force_run(self) -> None:
        result = parse_overrides("+inject_auth")
        assert result.overrides["inject_auth"] == HookOverride.FORCE_RUN

    def test_force_skip(self) -> None:
        result = parse_overrides("-rule_evaluator")
        assert result.overrides["rule_evaluator"] == HookOverride.FORCE_SKIP

    def test_normal_explicit(self) -> None:
        result = parse_overrides("some_hook")
        assert result.overrides["some_hook"] == HookOverride.NORMAL

    def test_multiple_overrides(self) -> None:
        result = parse_overrides("+inject_auth,-rule_evaluator,normal_hook")
        assert result.overrides["inject_auth"] == HookOverride.FORCE_RUN
        assert result.overrides["rule_evaluator"] == HookOverride.FORCE_SKIP
        assert result.overrides["normal_hook"] == HookOverride.NORMAL

    def test_whitespace_stripped(self) -> None:
        result = parse_overrides(" +inject_auth , -rule_evaluator ")
        assert result.overrides["inject_auth"] == HookOverride.FORCE_RUN
        assert result.overrides["rule_evaluator"] == HookOverride.FORCE_SKIP

    def test_empty_parts_ignored(self) -> None:
        result = parse_overrides("+hook,,,-other_hook")
        assert "hook" in result.overrides
        assert "-other_hook" not in result.overrides  # bare '-' would strip to ''

    def test_raw_header_preserved(self) -> None:
        result = parse_overrides("+inject_auth")
        assert result.raw_header == "+inject_auth"

    def test_plus_with_empty_name_ignored(self) -> None:
        result = parse_overrides("+")
        assert result.overrides == {}

    def test_minus_with_empty_name_ignored(self) -> None:
        result = parse_overrides("-")
        assert result.overrides == {}

    def test_debug_log_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger="ccproxy.pipeline.overrides"):
            parse_overrides("+inject_auth")
        assert any("override" in rec.message.lower() for rec in caplog.records)


class TestOverrideSetGetOverride:
    def test_default_is_normal(self) -> None:
        os = OverrideSet(overrides={}, raw_header="")
        assert os.get_override("any_hook") == HookOverride.NORMAL

    def test_returns_configured_override(self) -> None:
        os = OverrideSet(overrides={"my_hook": HookOverride.FORCE_RUN}, raw_header="")
        assert os.get_override("my_hook") == HookOverride.FORCE_RUN


class TestOverrideSetShouldRun:
    def test_force_run_ignores_guard(self) -> None:
        os = OverrideSet(overrides={"h": HookOverride.FORCE_RUN}, raw_header="")
        assert os.should_run("h", False) is True

    def test_force_skip_ignores_guard(self) -> None:
        os = OverrideSet(overrides={"h": HookOverride.FORCE_SKIP}, raw_header="")
        assert os.should_run("h", True) is False

    def test_normal_defers_to_guard_true(self) -> None:
        os = OverrideSet(overrides={}, raw_header="")
        assert os.should_run("h", True) is True

    def test_normal_defers_to_guard_false(self) -> None:
        os = OverrideSet(overrides={}, raw_header="")
        assert os.should_run("h", False) is False


class TestExtractOverridesFromContext:
    def test_lowercase_key(self) -> None:
        headers = {"x-ccproxy-hooks": "+inject_auth"}
        result = extract_overrides_from_context(headers)
        assert result.overrides["inject_auth"] == HookOverride.FORCE_RUN

    def test_mixed_case_key(self) -> None:
        headers = {"X-CCProxy-Hooks": "-rule_evaluator"}
        result = extract_overrides_from_context(headers)
        assert result.overrides["rule_evaluator"] == HookOverride.FORCE_SKIP

    def test_uppercase_key(self) -> None:
        headers = {"X-CCPROXY-HOOKS": "+h"}
        result = extract_overrides_from_context(headers)
        assert "h" in result.overrides

    def test_case_insensitive_fallback(self) -> None:
        headers = {"X-Ccproxy-Hooks": "+model_router"}
        result = extract_overrides_from_context(headers)
        assert "model_router" in result.overrides

    def test_no_header_returns_empty(self) -> None:
        result = extract_overrides_from_context({})
        assert result.overrides == {}
