"""Regression: ``commitbee_compat`` guard must not crash on list-shaped bodies.

Background — Anthropic's ``/api/v2/logs`` event-logging endpoint posts a
JSON-array body (a batch of telemetry events). ``commitbee_compat_guard``
previously called ``ctx._body.get("system")`` unconditionally; on
list-shaped bodies that raised ``AttributeError: 'list' object has no
attribute 'get'`` and the executor logged a hook ERROR per request.

The fix: the guard short-circuits when ``ctx._body`` is not a dict and
returns ``False`` before touching ``.get(...)``. The hook body has the
same short-circuit so an explicit ``FORCE_RUN`` override on an
array-bodied flow doesn't crash either.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from ccproxy.hooks.commitbee_compat import commitbee_compat, commitbee_compat_guard
from ccproxy.pipeline.context import Context


def _make_context(body: Any) -> Context:
    """Build a minimal :class:`Context` with a body of arbitrary shape."""
    return Context(
        flow=cast(Any, MagicMock()),
        _body=body,
        _request=None,
    )


def test_guard_returns_false_for_list_body() -> None:
    """List-shaped body must short-circuit the guard cleanly."""
    ctx = _make_context([{"event": "foo"}, {"event": "bar"}])
    assert commitbee_compat_guard(ctx) is False


def test_guard_returns_false_for_string_body() -> None:
    """String-shaped body (unexpected but possible) must short-circuit too."""
    ctx = _make_context("raw string body")
    assert commitbee_compat_guard(ctx) is False


def test_guard_returns_false_for_none_body() -> None:
    """None-shaped body must short-circuit; no AttributeError."""
    ctx = _make_context(None)
    assert commitbee_compat_guard(ctx) is False


def test_guard_still_matches_dict_with_commitbee_signature() -> None:
    """Existing match path: dict-shaped body with commitbee signature still triggers."""
    sig = "You generate Conventional Commit messages from git diffs from your codebase"
    ctx = _make_context({"system": sig})
    assert commitbee_compat_guard(ctx) is True


def test_hook_body_no_op_on_list_body() -> None:
    """Even if FORCE_RUN bypasses the guard, the hook body must not crash on list bodies."""
    body = [{"event": "foo"}]
    ctx = _make_context(body)
    result = commitbee_compat(ctx, {})
    assert result is ctx
    # Body is untouched (still the same list).
    assert cast(object, ctx._body) is body
