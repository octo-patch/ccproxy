"""Tests for the persistent-loop graph-side ``SSEPipeline``.

Covers:

- Chunk-boundary robustness (1-byte, 16-byte, all-at-once chunks all
  produce identical wire output for a given upstream).
- The EOS path (``b""`` triggers ``intake.close()`` drain, render terminator
  emission, daemon-thread teardown).
- Explicit :meth:`close` idempotency and post-close behavior.
- Concurrent pipeline instances (two pipelines do NOT share state — each
  owns its own asyncio loop + daemon thread).
- ``upstream_raw_bytes`` / ``raw_body`` tee for inspectors like
  :class:`PerplexityAddon` that read the raw upstream bytes mid-stream.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic_ai.models import ModelRequestParameters

from ccproxy.lightllm.graph import dispatch_intake, dispatch_render
from ccproxy.lightllm.graph.sse_pipeline import SSEPipeline
from ccproxy.lightllm.parsed import InboundFormat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame(event: dict[str, Any]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _build_anthropic_text_sse(text: str) -> bytes:
    """Synthetic Anthropic Messages SSE stream emitting one text block."""
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test_pipeline",
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
    return b"".join(_frame(e) for e in events)


def _make_fsm_pipeline(
    *, provider_type: str = "anthropic", inbound_format: InboundFormat
) -> SSEPipeline:
    intake = dispatch_intake(
        provider_type=provider_type,
        model="claude-3-5-haiku-20241022",
        request_params=ModelRequestParameters(),
    )
    render = dispatch_render(
        inbound_format=inbound_format,
        model="claude-3-5-haiku-20241022",
    )
    return SSEPipeline(intake=intake, render=render)


def _drive_pipeline(pipeline: SSEPipeline, data: bytes, chunk_size: int) -> bytes:
    """Feed ``data`` to ``pipeline`` in chunks of ``chunk_size`` bytes; flush via EOS."""
    out = bytearray()
    if chunk_size <= 0 or chunk_size >= len(data):
        chunks = [data]
    else:
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    for chunk in chunks:
        result = pipeline(chunk)
        if isinstance(result, (bytes, bytearray)):
            out.extend(result)
    flushed = pipeline(b"")
    if isinstance(flushed, (bytes, bytearray)):
        out.extend(flushed)
    return bytes(out)


def _normalize_for_compare(wire: bytes) -> bytes:
    """Normalize random ids + timestamps so two pipeline runs compare equal."""
    import re

    text = wire.decode()
    text = re.sub(r'"id"\s*:\s*"msg_[0-9a-f]+"', '"id":"msg_X"', text)
    text = re.sub(r'"id"\s*:\s*"chatcmpl-[0-9a-f]+"', '"id":"chatcmpl-X"', text)
    text = re.sub(r'"created"\s*:\s*\d+', '"created":0', text)
    text = re.sub(r'"model"\s*:\s*"[^"]+"', '"model":"M"', text)
    return text.encode()


# ---------------------------------------------------------------------------
# Chunk-boundary robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 16, 64, 0], ids=["1-byte", "16-byte", "64-byte", "all-at-once"])
class TestChunkBoundaryRobustness:
    """Wire output must be invariant under chunking — same bytes regardless of slice size."""

    def test_anthropic_to_anthropic(self, chunk_size: int) -> None:
        upstream_bytes = _build_anthropic_text_sse("chunked content")

        reference = _make_fsm_pipeline(inbound_format=InboundFormat.ANTHROPIC_MESSAGES)
        try:
            reference_out = _drive_pipeline(reference, upstream_bytes, chunk_size=0)
        finally:
            reference.close()

        candidate = _make_fsm_pipeline(inbound_format=InboundFormat.ANTHROPIC_MESSAGES)
        try:
            candidate_out = _drive_pipeline(candidate, upstream_bytes, chunk_size=chunk_size)
        finally:
            candidate.close()

        assert _normalize_for_compare(candidate_out) == _normalize_for_compare(reference_out)

    def test_anthropic_to_openai(self, chunk_size: int) -> None:
        upstream_bytes = _build_anthropic_text_sse("chunked cross-format")

        reference = _make_fsm_pipeline(inbound_format=InboundFormat.OPENAI_CHAT)
        try:
            reference_out = _drive_pipeline(reference, upstream_bytes, chunk_size=0)
        finally:
            reference.close()

        candidate = _make_fsm_pipeline(inbound_format=InboundFormat.OPENAI_CHAT)
        try:
            candidate_out = _drive_pipeline(candidate, upstream_bytes, chunk_size=chunk_size)
        finally:
            candidate.close()

        assert _normalize_for_compare(candidate_out) == _normalize_for_compare(reference_out)


# ---------------------------------------------------------------------------
# EOS path
# ---------------------------------------------------------------------------


class TestEndOfStream:
    """``b""`` triggers ``intake.close()`` drain + render terminator emission."""

    def test_anthropic_eos_emits_message_stop(self) -> None:
        upstream_bytes = _build_anthropic_text_sse("eos test")
        pipeline = _make_fsm_pipeline(inbound_format=InboundFormat.ANTHROPIC_MESSAGES)
        try:
            out = _drive_pipeline(pipeline, upstream_bytes, chunk_size=0)
        finally:
            pipeline.close()

        # Anthropic terminator: ``message_delta`` + ``message_stop`` SSE events.
        assert b"event: message_delta" in out
        assert b"event: message_stop" in out

    def test_openai_eos_emits_done_terminator(self) -> None:
        upstream_bytes = _build_anthropic_text_sse("openai eos test")
        pipeline = _make_fsm_pipeline(inbound_format=InboundFormat.OPENAI_CHAT)
        try:
            out = _drive_pipeline(pipeline, upstream_bytes, chunk_size=0)
        finally:
            pipeline.close()

        # OpenAI terminator: ``data: [DONE]\n\n``.
        assert b"data: [DONE]\n\n" in out

    def test_empty_data_without_content_emits_terminator(self) -> None:
        """A pipeline that sees only ``b""`` still emits the render terminator
        so the client gets a well-formed (empty) end-of-stream."""
        pipeline = _make_fsm_pipeline(inbound_format=InboundFormat.ANTHROPIC_MESSAGES)
        try:
            result = pipeline(b"")
        finally:
            pipeline.close()
        assert isinstance(result, bytes)
        # Empty stream still produces a synthesized ``message_start`` +
        # ``message_delta`` + ``message_stop`` sequence (see
        # ``AnthropicResponseRenderFSM.close``).
        assert b"event: message_start" in result
        assert b"event: message_stop" in result


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Explicit close, idempotency, post-close behavior."""

    def test_explicit_close_is_idempotent(self) -> None:
        pipeline = _make_fsm_pipeline(inbound_format=InboundFormat.ANTHROPIC_MESSAGES)
        pipeline.close()
        # Second close must not raise.
        pipeline.close()

    def test_close_then_feed_passes_through(self) -> None:
        """After explicit close, the loop is gone; further chunks pass through."""
        pipeline = _make_fsm_pipeline(inbound_format=InboundFormat.ANTHROPIC_MESSAGES)
        pipeline.close()
        result = pipeline(b"junk bytes after close")
        # The pipeline can't process anything, so it returns the input bytes.
        assert result == b"junk bytes after close"

    def test_close_after_eos_is_noop(self) -> None:
        """EOS path tears down the loop; ``close()`` afterward must not crash."""
        upstream_bytes = _build_anthropic_text_sse("close after eos")
        pipeline = _make_fsm_pipeline(inbound_format=InboundFormat.ANTHROPIC_MESSAGES)
        _drive_pipeline(pipeline, upstream_bytes, chunk_size=0)
        pipeline.close()
        pipeline.close()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrentPipelines:
    """Two pipelines on the same thread must not share state — each owns its own loop."""

    def test_two_pipelines_independent(self) -> None:
        a_bytes = _build_anthropic_text_sse("pipeline A content")
        b_bytes = _build_anthropic_text_sse("pipeline B content")

        pa = _make_fsm_pipeline(inbound_format=InboundFormat.ANTHROPIC_MESSAGES)
        pb = _make_fsm_pipeline(inbound_format=InboundFormat.ANTHROPIC_MESSAGES)
        try:
            a_out = _drive_pipeline(pa, a_bytes, chunk_size=16)
            b_out = _drive_pipeline(pb, b_bytes, chunk_size=16)
        finally:
            pa.close()
            pb.close()

        assert b"pipeline A content" in a_out
        assert b"pipeline B content" in b_out
        # No cross-contamination.
        assert b"pipeline B content" not in a_out
        assert b"pipeline A content" not in b_out


# ---------------------------------------------------------------------------
# Raw-bytes tee
# ---------------------------------------------------------------------------


class TestRawBytesTeeing:
    """``upstream_raw_bytes`` and ``raw_body`` must be byte-for-byte tees of fed data."""

    def test_upstream_raw_bytes_tee(self) -> None:
        upstream_bytes = _build_anthropic_text_sse("teed bytes")
        pipeline = _make_fsm_pipeline(inbound_format=InboundFormat.ANTHROPIC_MESSAGES)
        try:
            for start in range(0, len(upstream_bytes), 16):
                pipeline(upstream_bytes[start : start + 16])
            assert pipeline.upstream_raw_bytes == upstream_bytes
            assert pipeline.raw_body == upstream_bytes
        finally:
            pipeline.close()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Failures during feed don't stall mitmproxy — the chunk passes through."""

    def test_malformed_chunk_does_not_crash(self) -> None:
        pipeline = _make_fsm_pipeline(inbound_format=InboundFormat.ANTHROPIC_MESSAGES)
        try:
            result = pipeline(b"event: unknown\ndata: {not valid json\n\n")
        finally:
            pipeline.close()
        # Intake silently drops unparseable frames; result is empty.
        assert result == [] or result == b""
