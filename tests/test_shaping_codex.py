"""Tests for Codex/OpenAI Responses shape hooks."""

from __future__ import annotations

import json
from typing import Any, cast

from mitmproxy import http

from ccproxy.pipeline.context import Context
from ccproxy.shaping.codex import enforce_codex_store_false, normalize_responses_input


def _make_ctx(body: dict[str, Any]) -> Context:
    req = http.Request.make(
        "POST",
        "https://chatgpt.com/backend-api/codex/responses",
        json.dumps(body).encode(),
        headers=cast(dict[str | bytes, str | bytes], {"content-type": "application/json"}),
    )
    return Context.from_request(req)


def test_normalize_responses_input_converts_string_shorthand() -> None:
    ctx = _make_ctx({"input": "Reply with ok"})

    normalize_responses_input(ctx, {})

    assert ctx._body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Reply with ok"}],
        }
    ]


def test_normalize_responses_input_leaves_list_input_unchanged() -> None:
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Already normalized"}],
        }
    ]
    ctx = _make_ctx({"input": input_items})

    normalize_responses_input(ctx, {})

    assert ctx._body["input"] == input_items


def test_enforce_codex_store_false_overrides_public_default() -> None:
    ctx = _make_ctx({"store": True})

    enforce_codex_store_false(ctx, {})

    assert ctx._body["store"] is False
