"""Funnel: response token-usage capture, projection, and emission.

The cross-format response transform used to drop token usage entirely (the IR
that carries text/tool parts has no usage slot, so ``message_start`` /
``message_delta`` / the OpenAI terminal usage chunk / Gemini ``usageMetadata``
were parsed and discarded). These tests lock in the funnel that restores it:

* ``_usage`` mapping helpers project canonical ``RequestUsage`` into each
  listener wire shape, and capture raw provider usage back out.
* Each intake FSM accumulates usage off the already-parsed wire events and
  exposes it via ``.usage`` (plus carried-through metadata via ``.raw_extras``).
* Buffered transforms project the captured usage into the listener's usage
  block for every listener format.
* Streaming transforms emit the usage at the right seam (a terminal
  ``chat.completion.chunk`` with empty ``choices`` for OpenAI Chat; the
  ``response.completed`` envelope for OpenAI Responses).

Ground truth: an Anthropic body reporting ``input_tokens=28, output_tokens=4,
cache_read=900, cache_creation=100`` must surface, in OpenAI Chat shape, as
``prompt_tokens=1028`` (input + both cache classes), ``completion_tokens=4``,
``total_tokens=1032``, ``prompt_tokens_details.cached_tokens=900``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage

from ccproxy.lightllm.graph import _usage, dispatch_intake, dispatch_render
from ccproxy.lightllm.graph._base import IntakeState, ResponseIntakeFSM
from ccproxy.lightllm.graph.anthropic_intake import AnthropicResponseIntakeFSM
from ccproxy.lightllm.graph.buffered import transform_buffered_response_sync
from ccproxy.lightllm.graph.google_intake import GoogleResponseIntakeFSM
from ccproxy.lightllm.graph.openai_intake import OpenAIResponseIntakeFSM
from ccproxy.lightllm.graph.sse_pipeline import SSEPipeline
from ccproxy.lightllm.parsed import InboundFormat

# ── SSE frame builders ─────────────────────────────────────────────────────


def _anthropic_frame(event: dict[str, Any]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _anthropic_stream(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
    text: str = "ok",
    message_id: str = "msg_abc",
    model: str = "claude-haiku-4-5",
) -> list[bytes]:
    """A minimal but complete Anthropic Messages SSE stream carrying usage."""
    # Anthropic seeds message_start.usage.output_tokens with a small non-zero
    # value in practice; keep that unless the whole stream is deliberately empty.
    start_usage: dict[str, Any] = {"input_tokens": input_tokens, "output_tokens": 1 if output_tokens else 0}
    if cache_read:
        start_usage["cache_read_input_tokens"] = cache_read
    if cache_creation:
        start_usage["cache_creation_input_tokens"] = cache_creation
    return [
        _anthropic_frame(
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": start_usage,
                },
            }
        ),
        _anthropic_frame({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        _anthropic_frame({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        _anthropic_frame({"type": "content_block_stop", "index": 0}),
        _anthropic_frame(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            }
        ),
        _anthropic_frame({"type": "message_stop"}),
    ]


def _anthropic_body(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
    text: str = "ok",
    model: str = "claude-haiku-4-5",
) -> bytes:
    usage: dict[str, Any] = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if cache_read:
        usage["cache_read_input_tokens"] = cache_read
    if cache_creation:
        usage["cache_creation_input_tokens"] = cache_creation
    return json.dumps(
        {
            "id": "msg_body",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": usage,
        }
    ).encode()


# ── _usage mapping helpers ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ProjectionCase:
    name: str
    usage: RequestUsage
    openai_chat: dict[str, Any]
    openai_responses: dict[str, Any]
    anthropic: dict[str, Any]


PROJECTION_CASES: list[ProjectionCase] = [
    ProjectionCase(
        name="with_cache",
        usage=RequestUsage(input_tokens=28, output_tokens=4, cache_read_tokens=900, cache_write_tokens=100),
        openai_chat={
            "prompt_tokens": 1028,
            "completion_tokens": 4,
            "total_tokens": 1032,
            "prompt_tokens_details": {"cached_tokens": 900},
        },
        openai_responses={
            "input_tokens": 1028,
            "output_tokens": 4,
            "total_tokens": 1032,
            "input_tokens_details": {"cached_tokens": 900},
        },
        anthropic={
            "input_tokens": 28,
            "output_tokens": 4,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 100,
        },
    ),
    ProjectionCase(
        name="no_cache",
        usage=RequestUsage(input_tokens=10, output_tokens=5),
        openai_chat={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        openai_responses={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        anthropic={"input_tokens": 10, "output_tokens": 5},
    ),
]


@pytest.mark.parametrize("case", [pytest.param(c, id=c.name) for c in PROJECTION_CASES])
def test_usage_projection(case: ProjectionCase) -> None:
    assert _usage.to_openai_chat(case.usage) == case.openai_chat
    assert _usage.to_openai_responses(case.usage) == case.openai_responses
    assert _usage.to_anthropic(case.usage) == case.anthropic


def test_usage_is_empty() -> None:
    assert _usage.usage_is_empty(None) is True
    assert _usage.usage_is_empty(RequestUsage()) is True
    assert _usage.usage_is_empty(RequestUsage(input_tokens=1)) is False
    assert _usage.usage_is_empty(RequestUsage(cache_read_tokens=5)) is False


def test_capture_anthropic_merges_cumulative_events() -> None:
    """message_start carries input+cache; message_delta carries output — merge, not clobber."""
    start = _usage.usage_from_anthropic(
        {"input_tokens": 28, "output_tokens": 1, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 100}
    )
    merged = _usage.usage_from_anthropic({"output_tokens": 4}, existing=start)
    assert merged.input_tokens == 28
    assert merged.output_tokens == 4
    assert merged.cache_read_tokens == 900
    assert merged.cache_write_tokens == 100


def test_capture_openai_chat_subtracts_cached_from_prompt() -> None:
    """OpenAI prompt_tokens INCLUDES cached; canonical input_tokens excludes it."""
    usage = _usage.usage_from_openai_chat(
        {"prompt_tokens": 1000, "completion_tokens": 40, "prompt_tokens_details": {"cached_tokens": 900}}
    )
    assert usage.input_tokens == 100
    assert usage.cache_read_tokens == 900
    assert usage.output_tokens == 40


def test_capture_google_folds_thoughts_into_output() -> None:
    usage = _usage.usage_from_google(
        {
            "promptTokenCount": 50,
            "candidatesTokenCount": 10,
            "cachedContentTokenCount": 30,
            "thoughtsTokenCount": 5,
        }
    )
    assert usage.input_tokens == 20  # 50 - 30 cached
    assert usage.output_tokens == 15  # 10 candidates + 5 thoughts
    assert usage.cache_read_tokens == 30


# ── Funnel interface is ubiquitous across all intakes ──────────────────────


@pytest.mark.parametrize(
    "provider_type",
    ["anthropic", "openai", "openai_responses", "google", "perplexity_pro", "openai_conversations"],
)
def test_every_intake_exposes_the_funnel_interface(provider_type: str) -> None:
    """The funnel is a uniform mechanism guaranteed by the shared base class.

    Every intake carries `.usage` / `.raw_extras` / `.finish_reason` /
    `.provider_response_id`. Providers whose wire reports no token usage
    (Perplexity, conversations) still expose the slots — they just stay
    empty — so downstream code never special-cases.
    """
    fsm = dispatch_intake(provider_type=provider_type, model="m", request_params=ModelRequestParameters())
    assert isinstance(fsm, ResponseIntakeFSM)
    assert isinstance(fsm.state, IntakeState)
    assert isinstance(fsm.usage, RequestUsage)
    assert _usage.usage_is_empty(fsm.usage)  # nothing fed yet
    assert isinstance(fsm.raw_extras, dict)
    assert fsm.finish_reason is None
    assert fsm.provider_response_id is None


# ── Intake capture ─────────────────────────────────────────────────────────


async def test_anthropic_intake_captures_usage_and_response_id() -> None:
    fsm = AnthropicResponseIntakeFSM(model="claude-haiku-4-5", request_params=ModelRequestParameters())
    for frame in _anthropic_stream(input_tokens=28, output_tokens=4, cache_read=900, cache_creation=100):
        await fsm.feed(frame)
    await fsm.close()
    assert fsm.usage.input_tokens == 28
    assert fsm.usage.output_tokens == 4
    assert fsm.usage.cache_read_tokens == 900
    assert fsm.usage.cache_write_tokens == 100
    # Funnel: the upstream response id is carried through rather than dropped.
    assert fsm.raw_extras["response_id"] == "msg_abc"


async def test_openai_intake_captures_terminal_usage_chunk() -> None:
    fsm = OpenAIResponseIntakeFSM(model="gpt-4o", request_params=ModelRequestParameters())
    content = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
    }
    terminal = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o",
        "choices": [],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
    }
    await fsm.feed(f"data: {json.dumps(content)}\n\n".encode())
    await fsm.feed(f"data: {json.dumps(terminal)}\n\n".encode())
    await fsm.feed(b"data: [DONE]\n\n")
    await fsm.close()
    assert fsm.usage.input_tokens == 60  # 100 prompt - 40 cached
    assert fsm.usage.output_tokens == 20
    assert fsm.usage.cache_read_tokens == 40


async def test_google_intake_captures_usage_metadata() -> None:
    fsm = GoogleResponseIntakeFSM(model="gemini-2.0-flash", request_params=ModelRequestParameters())
    chunk = {
        "candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 10,
            "cachedContentTokenCount": 30,
            "thoughtsTokenCount": 5,
        },
    }
    await fsm.feed(f"data: {json.dumps(chunk)}\n\n".encode())
    await fsm.close()
    assert fsm.usage.input_tokens == 20
    assert fsm.usage.output_tokens == 15
    assert fsm.usage.cache_read_tokens == 30


# ── Buffered projection ────────────────────────────────────────────────────


@dataclass(frozen=True)
class BufferedCase:
    name: str
    inbound_format: InboundFormat
    expected_usage: dict[str, Any]


BUFFERED_CASES: list[BufferedCase] = [
    BufferedCase(
        name="openai_chat",
        inbound_format=InboundFormat.OPENAI_CHAT,
        expected_usage={
            "prompt_tokens": 1028,
            "completion_tokens": 4,
            "total_tokens": 1032,
            "prompt_tokens_details": {"cached_tokens": 900},
        },
    ),
    BufferedCase(
        name="openai_responses",
        inbound_format=InboundFormat.OPENAI_RESPONSES,
        expected_usage={
            "input_tokens": 1028,
            "output_tokens": 4,
            "total_tokens": 1032,
            "input_tokens_details": {"cached_tokens": 900},
        },
    ),
    BufferedCase(
        name="anthropic_messages",
        inbound_format=InboundFormat.ANTHROPIC_MESSAGES,
        expected_usage={
            "input_tokens": 28,
            "output_tokens": 4,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 100,
        },
    ),
]


@pytest.mark.parametrize("case", [pytest.param(c, id=c.name) for c in BUFFERED_CASES])
def test_buffered_projects_usage(case: BufferedCase) -> None:
    out = json.loads(
        transform_buffered_response_sync(
            raw_bytes=_anthropic_body(input_tokens=28, output_tokens=4, cache_read=900, cache_creation=100),
            provider_type="anthropic",
            inbound_format=case.inbound_format,
            model="claude-haiku-4-5",
            request_params=ModelRequestParameters(),
        )
    )
    assert out["usage"] == case.expected_usage


def test_buffered_omits_usage_when_upstream_reports_none() -> None:
    """No usage on the wire → omit the key (chat) rather than fabricate zeros."""
    body = json.dumps(
        {
            "id": "msg_nousage",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-haiku-4-5",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    ).encode()
    out = json.loads(
        transform_buffered_response_sync(
            raw_bytes=body,
            provider_type="anthropic",
            inbound_format=InboundFormat.OPENAI_CHAT,
            model="claude-haiku-4-5",
            request_params=ModelRequestParameters(),
        )
    )
    assert "usage" not in out


def test_buffered_openai_upstream_carries_usage_to_anthropic() -> None:
    """OpenAI ChatCompletion upstream → Anthropic listener also projects usage."""
    body = json.dumps(
        {
            "id": "chatcmpl-buf",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 40},
            },
        }
    ).encode()
    out = json.loads(
        transform_buffered_response_sync(
            raw_bytes=body,
            provider_type="openai",
            inbound_format=InboundFormat.ANTHROPIC_MESSAGES,
            model="gpt-4o",
            request_params=ModelRequestParameters(),
        )
    )
    assert out["usage"] == {
        "input_tokens": 60,
        "output_tokens": 20,
        "cache_read_input_tokens": 40,
    }


# ── Streaming emission ─────────────────────────────────────────────────────


def _run_pipeline(*, inbound_format: InboundFormat, frames: list[bytes]) -> str:
    intake = dispatch_intake(
        provider_type="anthropic", model="claude-haiku-4-5", request_params=ModelRequestParameters()
    )
    render = dispatch_render(inbound_format=inbound_format, model="claude-haiku-4-5")
    pipe = SSEPipeline(intake=intake, render=render)
    out = bytearray()
    try:
        for frame in frames:
            chunk = pipe(frame)
            if isinstance(chunk, bytes):
                out += chunk
        tail = pipe(b"")
        if isinstance(tail, bytes):
            out += tail
    finally:
        pipe.close()
    return out.decode()


def _sse_data_objects(stream: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for line in stream.splitlines():
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            if payload and payload != "[DONE]":
                objects.append(json.loads(payload))
    return objects


def test_streaming_openai_chat_emits_trailing_usage_chunk() -> None:
    stream = _run_pipeline(
        inbound_format=InboundFormat.OPENAI_CHAT,
        frames=_anthropic_stream(input_tokens=28, output_tokens=4, cache_read=900, cache_creation=100),
    )
    assert stream.rstrip().endswith("data: [DONE]")
    usage_chunks = [obj for obj in _sse_data_objects(stream) if obj.get("choices") == [] and "usage" in obj]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["usage"] == {
        "prompt_tokens": 1028,
        "completion_tokens": 4,
        "total_tokens": 1032,
        "prompt_tokens_details": {"cached_tokens": 900},
    }


def test_streaming_openai_responses_completed_carries_usage() -> None:
    stream = _run_pipeline(
        inbound_format=InboundFormat.OPENAI_RESPONSES,
        frames=_anthropic_stream(input_tokens=28, output_tokens=4, cache_read=900, cache_creation=100),
    )
    completed = [obj for obj in _sse_data_objects(stream) if obj.get("type") == "response.completed"]
    assert len(completed) == 1
    assert completed[0]["response"]["usage"] == {
        "input_tokens": 1028,
        "output_tokens": 4,
        "total_tokens": 1032,
        "input_tokens_details": {"cached_tokens": 900},
    }


def test_streaming_chat_without_usage_emits_no_usage_chunk() -> None:
    """Empty/zero upstream usage → no usage chunk (never a zeroed one)."""
    stream = _run_pipeline(
        inbound_format=InboundFormat.OPENAI_CHAT,
        frames=_anthropic_stream(input_tokens=0, output_tokens=0),
    )
    usage_chunks = [obj for obj in _sse_data_objects(stream) if obj.get("choices") == [] and "usage" in obj]
    assert usage_chunks == []
