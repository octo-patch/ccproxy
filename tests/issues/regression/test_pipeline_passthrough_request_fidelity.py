"""Regression: pass-through pipeline execution must not synthesize request bytes."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

import pytest
from mitmproxy import http
from mitmproxy.test import tflow

from ccproxy.pipeline.context import Context
from ccproxy.pipeline.executor import PipelineExecutor
from ccproxy.pipeline.hook import GuardFn, HandlerFn, HookParams, HookSpec, always_true


@dataclass(frozen=True)
class RequestCase:
    name: str
    method: str
    url: str
    content: bytes
    headers: http.Headers
    http_version: str = "HTTP/1.1"
    trailers: http.Headers | None = None


@dataclass(frozen=True)
class RequestSnapshot:
    method: str
    scheme: str
    host: str
    port: int
    path: str
    http_version: str
    headers: tuple[tuple[bytes, bytes], ...]
    content: bytes
    trailers: tuple[tuple[bytes, bytes], ...] | None


def _headers(*items: tuple[str, str]) -> http.Headers:
    return http.Headers(tuple((name.encode(), value.encode()) for name, value in items))


REQUEST_CASES = [
    RequestCase(
        name="websocket_upgrade_empty_get",
        method="GET",
        url=(
            "https://api.anthropic.com/api/ws/speech_to_text/voice_stream"
            "?encoding=linear16&sample_rate=16000&channels=1&endpointing_ms=300"
            "&utterance_end_ms=1000&language=en&use_conversation_engine=true"
            "&stt_provider=deepgram-nova3"
        ),
        content=b"",
        headers=_headers(
            ("Connection", "Upgrade"),
            ("Upgrade", "websocket"),
            ("Sec-WebSocket-Version", "13"),
            ("Sec-WebSocket-Extensions", "permessage-deflate; client_max_window_bits"),
            ("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ=="),
            ("User-Agent", "claude-cli/2.1.177 (external, cli)"),
            ("anthropic-client-platform", "claude_code_cli"),
            ("x-app", "cli"),
        ),
    ),
    RequestCase(
        name="head_empty",
        method="HEAD",
        url="https://api.anthropic.com/",
        content=b"",
        headers=_headers(("Accept", "*/*"), ("User-Agent", "Bun/1.3.14")),
    ),
    RequestCase(
        name="options_empty_cors_probe",
        method="OPTIONS",
        url="https://api.anthropic.com/v1/messages?beta=true",
        content=b"",
        headers=_headers(
            ("Origin", "https://console.anthropic.com"),
            ("Access-Control-Request-Method", "POST"),
            ("Access-Control-Request-Headers", "authorization,content-type"),
        ),
    ),
    RequestCase(
        name="formatted_json_object",
        method="POST",
        url="https://api.anthropic.com/v1/messages?beta=true&debug=1",
        content=b'{\n  "messages" : [ ],\n  "model": "claude-test",\n  "metadata": {"user_id":"u"}\n}',
        headers=_headers(
            ("Content-Type", "application/json"),
            ("Accept", "application/json"),
            ("X-Duplicate", "one"),
            ("X-Duplicate", "two"),
        ),
        http_version="HTTP/2.0",
    ),
    RequestCase(
        name="json_array_batch",
        method="POST",
        url="https://api.anthropic.com/api/event_logging/v2/batch",
        content=b'[{"event":"one"}, {"event":"two"}]',
        headers=_headers(("Content-Type", "application/json"), ("x-service-name", "claude-code")),
    ),
    RequestCase(
        name="invalid_json_declared_json",
        method="POST",
        url="https://api.anthropic.com/api/event_logging/v2/batch",
        content=b'{"unfinished": ',
        headers=_headers(("Content-Type", "application/json")),
    ),
    RequestCase(
        name="ndjson_logs",
        method="POST",
        url="https://http-intake.logs.us5.datadoghq.com/api/v2/logs",
        content=b'{"a":1}\n{"b":2}\n',
        headers=_headers(("Content-Type", "application/x-ndjson")),
    ),
    RequestCase(
        name="form_urlencoded",
        method="POST",
        url="https://oauth2.googleapis.com/token",
        content=b"grant_type=refresh_token&scope=a%20b&empty=",
        headers=_headers(("Content-Type", "application/x-www-form-urlencoded")),
    ),
    RequestCase(
        name="multipart_boundary",
        method="POST",
        url="https://api.anthropic.com/v1/files",
        content=(
            b"--ccproxy-boundary\r\n"
            b'Content-Disposition: form-data; name="purpose"\r\n\r\n'
            b"batch\r\n"
            b"--ccproxy-boundary\r\n"
            b'Content-Disposition: form-data; name="file"; filename="a.txt"\r\n'
            b"Content-Type: text/plain\r\n\r\n"
            b"hello\r\n"
            b"--ccproxy-boundary--\r\n"
        ),
        headers=_headers(("Content-Type", "multipart/form-data; boundary=ccproxy-boundary")),
    ),
    RequestCase(
        name="binary_octet_stream",
        method="PUT",
        url="https://api.anthropic.com/api/audio/upload?chunk=1&chunk=1",
        content=bytes([0, 1, 2, 3, 10, 13, 127, 128, 255]) + b"\x00raw-audio\xff",
        headers=_headers(("Content-Type", "application/octet-stream")),
    ),
    RequestCase(
        name="delete_with_body",
        method="DELETE",
        url="https://api.anthropic.com/api/cache/session-1",
        content=b"raw-delete-body",
        headers=_headers(("Content-Type", "text/plain")),
    ),
    RequestCase(
        name="request_trailers",
        method="POST",
        url="https://api.anthropic.com/v1/messages",
        content=b'{"model":"m","messages":[]}',
        headers=_headers(("Content-Type", "application/json"), ("te", "trailers")),
        trailers=_headers(("x-request-trailer", "final"), ("x-request-trailer", "again")),
    ),
]


def _make_spec(
    name: str,
    *,
    handler: HandlerFn,
    reads: Iterable[str] = (),
    writes: Iterable[str] = (),
    guard: GuardFn | None = None,
) -> HookSpec:
    return HookSpec(
        name=name,
        handler=handler,
        guard=guard or always_true,
        reads=frozenset(reads),
        writes=frozenset(writes),
    )


def _touch_metadata(ctx: Context, params: HookParams) -> Context:
    ctx.metadata.conversation_id = "pipeline-saw-this-flow"
    return ctx


def _touch_header(ctx: Context, params: HookParams) -> Context:
    ctx.set_header("x-ccproxy-test", "header-only")
    return ctx


def _mutate_body(ctx: Context, params: HookParams) -> Context:
    ctx.extras.set("ccproxy_test.body_written", True)
    return ctx


def _make_flow(case: RequestCase) -> http.HTTPFlow:
    flow = tflow.tflow()
    flow.id = f"flow-{case.name}"
    flow.metadata = {}
    flow.request = http.Request.make(case.method, case.url, case.content, case.headers)
    flow.request.http_version = case.http_version
    flow.request.trailers = case.trailers
    return flow


def _snapshot_request(request: http.Request) -> RequestSnapshot:
    return RequestSnapshot(
        method=request.method,
        scheme=request.scheme,
        host=request.host,
        port=request.port,
        path=request.path,
        http_version=request.http_version,
        headers=tuple(request.headers.fields),
        content=bytes(request.content or b""),
        trailers=tuple(request.trailers.fields) if request.trailers is not None else None,
    )


@pytest.mark.parametrize("case", REQUEST_CASES, ids=lambda case: case.name)
def test_pipeline_metadata_only_hooks_preserve_request_exactly(case: RequestCase) -> None:
    flow = _make_flow(case)
    before = _snapshot_request(flow.request)

    PipelineExecutor(hooks=[_make_spec("touch_metadata", handler=_touch_metadata)]).execute(flow)

    assert _snapshot_request(flow.request) == before
    assert flow.metadata["ccproxy.conversation_id"] == "pipeline-saw-this-flow"


@pytest.mark.parametrize("case", REQUEST_CASES, ids=lambda case: case.name)
def test_pipeline_header_only_hooks_preserve_request_body_bytes(case: RequestCase) -> None:
    flow = _make_flow(case)
    before = _snapshot_request(flow.request)

    PipelineExecutor(hooks=[_make_spec("touch_header", handler=_touch_header)]).execute(flow)

    after = _snapshot_request(flow.request)
    assert after.content == before.content
    assert after.method == before.method
    assert after.scheme == before.scheme
    assert after.host == before.host
    assert after.port == before.port
    assert after.path == before.path
    assert after.http_version == before.http_version
    assert after.trailers == before.trailers
    assert flow.request.headers["x-ccproxy-test"] == "header-only"


def test_pipeline_body_mutation_still_serializes_body() -> None:
    flow = _make_flow(
        RequestCase(
            name="empty_post",
            method="POST",
            url="https://api.anthropic.com/v1/messages",
            content=b"",
            headers=_headers(("Content-Type", "application/json")),
        )
    )

    PipelineExecutor(hooks=[_make_spec("mutate_body", handler=_mutate_body)]).execute(flow)

    content = flow.request.content
    assert content is not None
    assert json.loads(content) == {"ccproxy_test": {"body_written": True}}
