"""Sync facade over the async dispatch_dump (replacement for outbound_sync).

Verifies ``dispatch_dump_sync`` produces bytes byte-equal to
``asyncio.run(dispatch_dump(...))`` across every supported provider, and
that the unsupported-provider path still raises ``UnsupportedUpstreamError``.

This is the FSM-side replacement for ``test_lightllm_outbound_sync.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from ccproxy.lightllm.graph import (
    UnsupportedUpstreamError,
    dispatch_dump,
    dispatch_dump_sync,
)
from ccproxy.lightllm.parsed import ParsedRequest


def _make_parsed(
    *,
    model: str = "test-model",
    raw_extras: dict[str, Any] | None = None,
) -> ParsedRequest:
    return ParsedRequest(
        model=model,
        messages=[ModelRequest(parts=[UserPromptPart(content="hello")])],
        request_parameters=ModelRequestParameters(),
        settings={},
        stream=False,
        raw_extras=raw_extras or {},
    )


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("anthropic", "claude-3"),
        ("deepseek", "deepseek-chat"),
        ("zai", "glm-4"),
        ("openai", "gpt-4o"),
        ("google", "gemini-1.5-pro"),
        ("gemini", "gemini-1.5-pro"),
        ("vertex_ai", "gemini-1.5-pro"),
    ],
)
def test_dispatch_dump_sync_matches_async(provider: str, model: str) -> None:
    parsed = _make_parsed(model=model)
    expected = asyncio.run(dispatch_dump(parsed, provider=provider))
    actual = dispatch_dump_sync(parsed, provider=provider)
    assert actual == expected


def test_dispatch_dump_sync_matches_async_perplexity_pro() -> None:
    """Perplexity Pro mints a ``frontend_uuid`` per request. Lock it via
    patch so both async and sync paths emit identical bytes."""
    parsed = _make_parsed(
        model="perplexity/best",
        raw_extras={
            "pplx": {
                "last_backend_uuid": "11111111-1111-1111-1111-111111111111",
                "frontend_context_uuid": "22222222-2222-2222-2222-222222222222",
                "read_write_token": "tok",
            }
        },
    )

    with patch(
        "ccproxy.lightllm.pplx.uuid.uuid4",
        return_value="33333333-3333-3333-3333-333333333333",
    ):
        expected = asyncio.run(dispatch_dump(parsed, provider="perplexity_pro"))
    with patch(
        "ccproxy.lightllm.pplx.uuid.uuid4",
        return_value="33333333-3333-3333-3333-333333333333",
    ):
        actual = dispatch_dump_sync(parsed, provider="perplexity_pro")

    assert actual == expected


def test_dispatch_dump_sync_raises_for_unknown_provider() -> None:
    parsed = _make_parsed()
    with pytest.raises(UnsupportedUpstreamError, match="no outbound renderer"):
        dispatch_dump_sync(parsed, provider="not-a-real-provider")
