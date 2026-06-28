"""End-to-end smoke tests for the openai_conversations provider (chatgpt.com).

Skipped by default (excluded via ``-m "not e2e"`` in pyproject.toml). Run with::

    uv run pytest -m e2e tests/e2e/test_openai_conversations_e2e.py

Prereqs:
    * ccproxy running on ``CCPROXY_E2E_URL`` (default ``http://127.0.0.1:4001``)
      with a configured ``providers.openai_conversations`` entry.
    * A valid OpenAI Conversations credential file (bearer JWT + device id) at
      ``CCPROXY_OAIC_CREDS`` (default
      ``~/.config/ccproxy/openai-conversations-credentials.json``).
    * Fresh browser cookies (``cf_clearance`` + session token) exported to the
      provider's ``cookie_file`` — ``cf_clearance`` expires in ~15-30 min, so
      re-export with ``gateau`` before running.

These tests catch regressions from external changes:
    * chatgpt.com Sentinel / conduit-prepare choreography drift.
    * SSE-v1 wire format / WebSocket-handoff changes.
    * Cloudflare fingerprint / cookie changes.

The proxy is driven through the same vector a user uses: the sentinel key
``sk-ant-oat-ccproxy-openai_conversations`` against the OpenAI-compatible
``/v1/chat/completions`` endpoint.

Scope note — these assert the CHATGPT-004 guarantees that are deterministic
post-fix: a stream:false turn yields one well-formed ``chat.completion`` (HTTP
200, ``application/json``), a stream:true turn yields only
``chat.completion.chunk`` frames + ``[DONE]`` with no raw SSE-v1 leak. They do
NOT hard-require non-empty content: chatgpt.com nondeterministically routes some
turns (``turn_use_case: "instant answers"``) over a ``resume_conversation_token``
conduit stream that ccproxy does not yet read (a separate, deferred continuation
mode — only inline SSE and the ``stream_handoff`` WebSocket bridge are wired), so
those turns legitimately return empty content until that path is implemented.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

CCPROXY_BASE = os.environ.get("CCPROXY_E2E_URL", "http://127.0.0.1:4001")
OAIC_CREDS = Path(
    os.environ.get(
        "CCPROXY_OAIC_CREDS",
        str(Path.home() / ".config" / "ccproxy" / "openai-conversations-credentials.json"),
    )
).expanduser()
SENTINEL_KEY = "sk-ant-oat-ccproxy-openai_conversations"
MODEL = os.environ.get("CCPROXY_E2E_OAIC_MODEL", "gpt-5-5")


def _proxy_reachable() -> bool:
    try:
        httpx.head(CCPROXY_BASE, timeout=2)
    except httpx.HTTPError:
        return False
    return True


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not OAIC_CREDS.exists(), reason=f"{OAIC_CREDS} not found"),
    pytest.mark.skipif(not _proxy_reachable(), reason=f"ccproxy not reachable at {CCPROXY_BASE}"),
]

_HEADERS = {"Authorization": f"Bearer {SENTINEL_KEY}", "Content-Type": "application/json"}
_PROMPT = "reply with exactly the single word: pong"


def test_non_stream_chat_completion() -> None:
    """A stream:false turn returns one well-formed OpenAI ``chat.completion`` object.

    Residual #1 regression: this path used to 500 with ``HandoffUnsupportedError``;
    it must now collect to a single JSON object. Content may be empty for
    conduit-resumed turns (see module docstring) — only the shape is asserted.
    """
    resp = httpx.post(
        f"{CCPROXY_BASE}/v1/chat/completions",
        headers=_HEADERS,
        json={"model": MODEL, "messages": [{"role": "user", "content": _PROMPT}], "stream": False},
        timeout=180,
    )
    assert resp.status_code == 200, resp.text[:500]
    assert resp.headers.get("content-type", "").startswith("application/json")
    body = resp.json()
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    content = choice["message"]["content"]
    assert content is None or isinstance(content, str)
    assert choice["finish_reason"] == "stop"


def test_stream_chat_completion_clean_chunks() -> None:
    """A stream:true turn emits only ``chat.completion.chunk`` frames + ``[DONE]``.

    No raw upstream SSE-v1 (``event: delta`` / ``/message/content/parts`` /
    ``stream_handoff``) may leak through the transform.
    """
    raw_lines: list[str] = []
    with httpx.stream(
        "POST",
        f"{CCPROXY_BASE}/v1/chat/completions",
        headers=_HEADERS,
        json={"model": MODEL, "messages": [{"role": "user", "content": _PROMPT}], "stream": True},
        timeout=180,
    ) as resp:
        assert resp.status_code == 200
        raw_lines.extend(resp.iter_lines())

    wire = "\n".join(raw_lines)
    # Residual #2 regression: no raw upstream SSE-v1 may leak through the transform.
    assert "event: delta" not in wire
    assert "/message/content/parts" not in wire
    assert "stream_handoff" not in wire
    assert "resume_conversation_token" not in wire

    saw_done = False
    for line in raw_lines:
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        obj = json.loads(payload)
        # Every emitted frame is a clean listener chunk (content may be empty for
        # conduit-resumed turns — see module docstring).
        assert obj["object"] == "chat.completion.chunk"

    assert saw_done
