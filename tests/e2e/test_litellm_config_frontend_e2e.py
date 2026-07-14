"""Live reverse-listener gate for the LiteLLM configuration frontend."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import httpx
import pytest

CCPROXY_BASE = os.environ.get("CCPROXY_E2E_URL", "http://127.0.0.1:4011")
UPSTREAM_ADDRESS = ("127.0.0.1", 18081)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("CCPROXY_E2E_LITELLM_FRONTEND") != "1",
        reason="run through `just e2e-litellm-config-frontend`",
    ),
]


class _UpstreamHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        content_length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(content_length))
        self.requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("authorization"),
                "x-api-key": self.headers.get("x-api-key"),
                "body": body,
            }
        )
        if self.path == "/v1/messages":
            response_body = {
                "id": "msg_local",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": "anthropic local ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 3},
            }
        else:
            response_body = {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "local ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
        payload = json.dumps(response_body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_live_reverse_listener_compiles_and_routes_litellm_model() -> None:
    _UpstreamHandler.requests.clear()
    upstream = ThreadingHTTPServer(UPSTREAM_ADDRESS, _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        response = httpx.post(
            f"{CCPROXY_BASE}/v1/chat/completions",
            headers={"authorization": "Bearer client-key"},
            json={
                "model": "local/qwen3",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.8,
            },
            timeout=10,
        )
        response.raise_for_status()
        assert response.json()["model"] == "qwen3"

        assert _UpstreamHandler.requests == [
            {
                "path": "/v1/chat/completions",
                "authorization": "Bearer local-e2e-secret",
                "x-api-key": None,
                "body": {
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.8,
                    "top_p": 0.9,
                },
            }
        ]

        transformed = httpx.post(
            f"{CCPROXY_BASE}/v1/chat/completions",
            headers={"authorization": "Bearer client-key"},
            json={
                "model": "local-claude",
                "messages": [{"role": "user", "content": "hello Claude"}],
            },
            timeout=10,
        )
        transformed.raise_for_status()
        assert transformed.json()["choices"][0]["message"]["content"] == "anthropic local ok"
        assert _UpstreamHandler.requests[1] == {
            "path": "/v1/messages",
            "authorization": None,
            "x-api-key": "local-e2e-secret",
            "body": {
                "model": "claude-local",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello Claude"}],
                    }
                ],
                "max_tokens": 64,
            },
        }

        catalog = httpx.get(f"{CCPROXY_BASE}/v1/models", timeout=10)
        catalog.raise_for_status()
        assert [entry["id"] for entry in catalog.json()["data"]] == ["local-fixed", "local-claude"]
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)
