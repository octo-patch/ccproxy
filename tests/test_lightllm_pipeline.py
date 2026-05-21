"""Integration tests for the SSEPipeline + buffered.py modules.

Tests the wiring between vendor-side intakes and listener-side renderers
via the SSEPipeline sync callable. Exercises both same-format and
cross-format paths.
"""

from __future__ import annotations

import json

import pytest
from pydantic_ai.models import ModelRequestParameters

from ccproxy.lightllm.parsed import ListenerFormat
from ccproxy.lightllm.response.buffered import transform_buffered_response
from ccproxy.lightllm.response.intake import select_intake
from ccproxy.lightllm.response.pipeline import SSEPipeline
from ccproxy.lightllm.response.render import select_render

pytestmark = pytest.mark.asyncio


def _build_anthropic_text_sse(text: str) -> bytes:
    """Build a synthetic Anthropic Messages SSE stream emitting a single text turn."""
    events: list[dict[str, object]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-3-5-haiku-20241022",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


class TestSSEPipelineSameFormat:
    async def test_anthropic_to_anthropic_text_passthrough_semantics(self) -> None:
        """SSEPipeline with Anthropic intake + Anthropic render should be semantically lossless."""
        from ccproxy.lightllm.response.intake_anthropic import AnthropicResponseIntake

        intake = AnthropicResponseIntake(
            model="claude-3-5-haiku-20241022",
            request_params=ModelRequestParameters(),
        )
        render = select_render(ListenerFormat.ANTHROPIC_MESSAGES)
        pipeline = SSEPipeline(intake=intake, render=render)

        upstream_bytes = _build_anthropic_text_sse("hello world")
        out = bytearray()
        rendered = pipeline(upstream_bytes)
        if isinstance(rendered, bytes):
            out.extend(rendered)
        flushed = pipeline(b"")
        if isinstance(flushed, bytes):
            out.extend(flushed)

        # Rendered output re-parses through a fresh Anthropic intake into a
        # ModelResponse with the same text content.
        verify_intake = AnthropicResponseIntake(
            model="claude-3-5-haiku-20241022",
            request_params=ModelRequestParameters(),
        )
        for _ in verify_intake.feed(bytes(out)):
            pass
        for _ in verify_intake.close():
            pass

        parts = verify_intake.parts_manager.get_parts()
        text_parts = [p for p in parts if hasattr(p, "content") and getattr(p, "content", None)]
        assert any("hello world" in str(getattr(p, "content", "")) for p in text_parts)

    async def test_raw_body_tee(self) -> None:
        intake = select_intake(
            upstream_provider="anthropic",
            model="claude-3-5-haiku-20241022",
            request_params=ModelRequestParameters(),
        )
        render = select_render(ListenerFormat.ANTHROPIC_MESSAGES)
        pipeline = SSEPipeline(intake=intake, render=render)

        upstream_bytes = _build_anthropic_text_sse("xyz")
        pipeline(upstream_bytes)
        assert pipeline.upstream_raw_bytes == upstream_bytes
        # raw_body alias works for backward-compat callsites.
        assert pipeline.raw_body == upstream_bytes


class TestSSEPipelineCrossFormat:
    async def test_anthropic_upstream_to_openai_listener(self) -> None:
        """Anthropic SSE → IR events → OpenAI Chat Completion SSE."""
        intake = select_intake(
            upstream_provider="anthropic",
            model="claude-3-5-haiku-20241022",
            request_params=ModelRequestParameters(),
        )
        render = select_render(ListenerFormat.OPENAI_CHAT)
        pipeline = SSEPipeline(intake=intake, render=render)

        upstream_bytes = _build_anthropic_text_sse("response text")
        out = bytearray()
        rendered = pipeline(upstream_bytes)
        if isinstance(rendered, bytes):
            out.extend(rendered)
        flushed = pipeline(b"")
        if isinstance(flushed, bytes):
            out.extend(flushed)

        # Output should be parseable as OpenAI Chat Completion SSE — contains
        # data: chat.completion.chunk JSON, and ends with [DONE].
        text = bytes(out).decode()
        assert "chat.completion.chunk" in text
        assert "response text" in text
        assert "[DONE]" in text


class TestSSEPipelineErrorHandling:
    async def test_malformed_chunk_passes_through(self) -> None:
        intake = select_intake(
            upstream_provider="anthropic",
            model="claude-3-5-haiku-20241022",
            request_params=ModelRequestParameters(),
        )
        render = select_render(ListenerFormat.ANTHROPIC_MESSAGES)
        pipeline = SSEPipeline(intake=intake, render=render)

        # An unparseable frame doesn't crash — the malformed payload is
        # silently dropped by the intake and processing continues.
        malformed = b"event: unknown\ndata: {not valid json\n\n"
        result = pipeline(malformed)
        # No IR events emitted from malformed bytes — render produces nothing.
        assert result == [] or result == b""


class TestBufferedResponse:
    async def test_anthropic_upstream_to_openai_listener_buffered(self) -> None:
        """Buffered upstream response → IR → buffered listener-format response."""
        # Anthropic streaming response body wrapped as one SSE frame.
        chunk = json.dumps(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_buffered",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-3-5-haiku-20241022",
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            }
        ).encode()
        out = transform_buffered_response(
            upstream_provider="anthropic",
            model="claude-3-5-haiku-20241022",
            listener_format=ListenerFormat.OPENAI_CHAT,
            request_params=ModelRequestParameters(),
            upstream_body=chunk,
        )
        assert b"chat.completion.chunk" in out
        assert b"[DONE]" in out
