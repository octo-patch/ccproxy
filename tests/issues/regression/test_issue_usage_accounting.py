"""Regression: usage accounting dropped on the OpenAI-compatible shims.

Reported by a golem supervising session (2026-07-11): the cross-format response
transform dropped token usage on both OpenAI-compatible listeners while the
Anthropic ``/v1/messages`` passthrough stayed correct.

* Bug 1 — ``/v1/chat/completions`` non-stream omitted the ``usage`` block entirely.
* Bug 2 — ``/v1/chat/completions`` streaming never emitted a usage chunk.
* Bug 3 — ``/v1/responses`` reported ``usage: {input_tokens: 0, output_tokens: 0}``.

Downstream, golem's ``pkg/bamlgraph.BAMLModel`` prefers provider-billed usage and
silently fell back to tokenizer estimation (Bug 1/2) or read the zeroed block as
"free" (Bug 3), under-billing an entire API family.

Ground truth: an Anthropic response reporting ``input_tokens=28, output_tokens=4``
must surface as ``prompt_tokens=28 / completion_tokens=4`` on the chat shim and
``input_tokens=28 / output_tokens=4`` on the responses shim.
"""

from __future__ import annotations

import json

from pydantic_ai.models import ModelRequestParameters

from ccproxy.lightllm.graph import dispatch_intake, dispatch_render
from ccproxy.lightllm.graph.buffered import transform_buffered_response_sync
from ccproxy.lightllm.graph.sse_pipeline import SSEPipeline
from ccproxy.lightllm.parsed import InboundFormat

_ANTHROPIC_BODY = json.dumps(
    {
        "id": "msg_regression",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "model": "claude-haiku-4-5",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 28, "output_tokens": 4},
    }
).encode()

_ANTHROPIC_SSE = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_regression",'
    b'"type":"message","role":"assistant","model":"claude-haiku-4-5","content":[],'
    b'"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":28,"output_tokens":1}}}\n\n'
    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"text","text":""}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"ok"}}\n\n'
    b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
    b'"stop_sequence":null},"usage":{"output_tokens":4}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


def test_bug1_chat_completions_nonstream_has_usage() -> None:
    """Bug 1: the non-stream chat.completion carries a populated usage block."""
    out = json.loads(
        transform_buffered_response_sync(
            raw_bytes=_ANTHROPIC_BODY,
            provider_type="anthropic",
            inbound_format=InboundFormat.OPENAI_CHAT,
            model="claude-haiku-4-5",
            request_params=ModelRequestParameters(),
        )
    )
    assert out["usage"] == {"prompt_tokens": 28, "completion_tokens": 4, "total_tokens": 32}


def test_bug2_chat_completions_stream_emits_usage_chunk() -> None:
    """Bug 2: the stream emits a terminal usage chunk (empty choices) before [DONE]."""
    intake = dispatch_intake(
        provider_type="anthropic", model="claude-haiku-4-5", request_params=ModelRequestParameters()
    )
    render = dispatch_render(inbound_format=InboundFormat.OPENAI_CHAT, model="claude-haiku-4-5")
    pipe = SSEPipeline(intake=intake, render=render)
    out = bytearray()
    try:
        chunk = pipe(_ANTHROPIC_SSE)
        if isinstance(chunk, bytes):
            out += chunk
        tail = pipe(b"")
        if isinstance(tail, bytes):
            out += tail
    finally:
        pipe.close()

    stream = out.decode()
    assert stream.rstrip().endswith("data: [DONE]")
    usage_chunks = [
        json.loads(line[len("data:") :].strip())
        for line in stream.splitlines()
        if line.startswith("data:") and line[len("data:") :].strip() not in ("", "[DONE]")
    ]
    usage_chunks = [c for c in usage_chunks if c.get("choices") == [] and "usage" in c]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["usage"] == {"prompt_tokens": 28, "completion_tokens": 4, "total_tokens": 32}


def test_bug3_responses_usage_is_not_zeroed() -> None:
    """Bug 3: /v1/responses reports the real (non-zero) usage, not a defaulted zero block."""
    out = json.loads(
        transform_buffered_response_sync(
            raw_bytes=_ANTHROPIC_BODY,
            provider_type="anthropic",
            inbound_format=InboundFormat.OPENAI_RESPONSES,
            model="claude-haiku-4-5",
            request_params=ModelRequestParameters(),
        )
    )
    assert out["usage"] == {"input_tokens": 28, "output_tokens": 4, "total_tokens": 32}
    assert out["usage"]["input_tokens"] != 0
    assert out["usage"]["output_tokens"] != 0
