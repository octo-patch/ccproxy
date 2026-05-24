"""Tests for bundled default request-shape assets."""

from __future__ import annotations

import json
import re
from pathlib import Path

from mitmproxy import http
from mitmproxy.io import FlowReader

TEMPLATES_SHAPES_DIR = Path(__file__).parents[1] / "src" / "ccproxy" / "templates" / "shapes"
DUMMY_UUID = "00000000-0000-0000-0000-000000000000"
UUID_RE = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
SECRET_MARKERS = [
    b"authorization",
    b"proxy-authorization",
    b"x-api-key",
    b"x-goog-api-key",
    b"cookie",
    b"set-cookie",
    b"sk-ant-oat",
    b"ya29.",
    b"ccproxy-flow-id",
    b"claude-code-session-id",
    b"client-request-id",
]
BODY_LEAK_MARKERS = [
    "interactive agent",
    "software engineering tasks",
    "available tools",
    "***",
    "starbased",
    "***",
    "***",
]


def _read_flows(path: Path) -> list[http.HTTPFlow]:
    flows: list[http.HTTPFlow] = []
    with path.open("rb") as fo:
        for flow in FlowReader(fo).stream():  # type: ignore[no-untyped-call]
            if isinstance(flow, http.HTTPFlow):
                flows.append(flow)
    return flows


def test_bundled_shape_files_exist() -> None:
    assert (TEMPLATES_SHAPES_DIR / "anthropic.mflow").is_file()
    assert (TEMPLATES_SHAPES_DIR / "gemini.mflow").is_file()


def test_bundled_shapes_are_sanitized() -> None:
    for path in TEMPLATES_SHAPES_DIR.glob("*.mflow"):
        raw = path.read_bytes().lower()
        assert path.stat().st_size < 16_384
        for marker in SECRET_MARKERS:
            assert marker not in raw

        flows = _read_flows(path)
        assert len(flows) == 1
        flow = flows[0]
        assert flow.response is None
        assert dict(flow.metadata) == {}
        assert len(flow.request.content or b"") < 4096
        assert "authorization" not in flow.request.headers
        assert "cookie" not in flow.request.headers

        body = json.loads(flow.request.content or b"{}")
        body_text = json.dumps(body, sort_keys=True).lower()
        for marker in BODY_LEAK_MARKERS:
            assert marker not in body_text
        for match in UUID_RE.findall(flow.request.content or b""):
            assert match.decode().lower() == DUMMY_UUID


def test_anthropic_default_shape_is_minimal() -> None:
    flow = _read_flows(TEMPLATES_SHAPES_DIR / "anthropic.mflow")[0]
    body = json.loads(flow.request.content or b"{}")

    assert flow.request.pretty_host == "api.anthropic.com"
    assert body["messages"] == [{"role": "user", "content": "seed"}]
    assert body["tools"] == []
    assert body["max_tokens"] == 1024
    assert body["stream"] is True

    system = body["system"]
    assert len(system) == 2
    assert system[0]["text"].startswith("x-anthropic-billing-header")
    assert system[1]["text"] == "You are a Claude agent, built on Anthropic's Claude Agent SDK."

    identity = json.loads(body["metadata"]["user_id"])
    assert identity["account_uuid"] == DUMMY_UUID
    assert identity["device_id"] == DUMMY_UUID
    assert identity["session_id"] == DUMMY_UUID


def test_gemini_default_shape_is_minimal() -> None:
    flow = _read_flows(TEMPLATES_SHAPES_DIR / "gemini.mflow")[0]
    body = json.loads(flow.request.content or b"{}")
    request = body["request"]

    assert flow.request.pretty_host == "cloudcode-pa.googleapis.com"
    assert body["user_prompt_id"] == "0000000000000"
    assert "project" not in body
    assert request["session_id"] == DUMMY_UUID
    assert request["contents"] == [{"role": "user", "parts": [{"text": "seed"}]}]
    assert "systemInstruction" not in request
    assert "tools" not in request
