"""E2E quality gate for packaged .mflow fallback shapes."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

CCPROXY_BASE = os.environ.get("CCPROXY_E2E_URL", "http://127.0.0.1:4001")
SHAPES_DIR = Path(__file__).resolve().parents[2] / "src" / "ccproxy" / "templates" / "shapes"

ANTHROPIC_MODEL = os.environ.get("CCPROXY_E2E_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
GEMINI_MODEL = os.environ.get("CCPROXY_E2E_GEMINI_MODEL", "gemini-3.1-pro-preview")


def _proxy_reachable() -> bool:
    try:
        response = httpx.get(f"{CCPROXY_BASE}/health", timeout=2)
    except httpx.HTTPError:
        return False
    return response.status_code < 500


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("CCPROXY_E2E_PACKAGED_SHAPES") != "1",
        reason="run through `just e2e-packaged-mflows` to force packaged shape fallback",
    ),
    pytest.mark.skipif(not _proxy_reachable(), reason=f"ccproxy not reachable at {CCPROXY_BASE}"),
]


def _require_shape(name: str) -> None:
    path = SHAPES_DIR / f"{name}.mflow"
    if not path.exists():
        pytest.fail(f"packaged shape missing: {path}")


def _call_with_retry(fn: Callable[[], Any], *, retries: int = 2, backoff: float = 3.0) -> Any:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if status in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            raise
    raise AssertionError(f"unreachable after retry loop: {last_exc!r}")


@pytest.mark.skipif(not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"), reason="CLAUDE_CODE_OAUTH_TOKEN not set")
def test_anthropic_sdk_uses_packaged_shape() -> None:
    _require_shape("anthropic")
    import anthropic

    client = anthropic.Anthropic(
        api_key="sk-ant-oat-ccproxy-anthropic",
        base_url=CCPROXY_BASE,
    )

    response = _call_with_retry(
        lambda: client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=32,
            messages=[{"role": "user", "content": "Reply with exactly: packaged e2e ok"}],
        )
    )

    assert response.content
    text = response.content[0].text
    assert "packaged e2e ok" in text.lower()


@pytest.mark.skipif(not (Path.home() / ".gemini" / "oauth_creds.json").exists(), reason="Gemini OAuth creds absent")
def test_google_genai_sdk_uses_packaged_shape() -> None:
    _require_shape("gemini")
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key="sk-ant-oat-ccproxy-gemini",
        http_options=types.HttpOptions(base_url=f"{CCPROXY_BASE}/gemini"),
    )

    response = _call_with_retry(
        lambda: client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Reply with exactly: packaged e2e ok",
            config=types.GenerateContentConfig(
                max_output_tokens=128,
                thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=0),
            ),
        )
    )

    assert response.text is not None
    assert "packaged e2e ok" in response.text.lower()
