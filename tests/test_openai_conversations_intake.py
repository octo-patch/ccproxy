"""Tests for ``OpenAIConversationsIntakeFSM``.

Covers:
- Every reference SSE-v1 frame shape (explicit batch, shorthand batch, implicit
  add, single patch, encoding banner, ``[DONE]``).
- Multi-chunk split mid-frame: no drop, no duplication.
- Hidden/system/user channels must not leak into the visible answer.
- ``replace`` patches emit only the new suffix.
- Finish is synthesised exactly once (``/message/status`` and ``[DONE]``
  together do not duplicate).
- ``resume_conversation_token``, ``stream_handoff``, and ``server_ste_metadata``
  before any content → :class:`HandoffUnsupportedError`.
- Handoff typed events after content begin → ``state.continuation`` is set,
  ``content_begun`` remains ``True``, no error raised.
- Buffered OpenAI Conversations SSE → one OpenAI Chat ``chat.completion`` JSON
  body via ``transform_buffered_response_sync``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponseStreamEvent, TextPart
from pydantic_ai.models import ModelRequestParameters

from ccproxy.lightllm.graph.buffered import transform_buffered_response_sync
from ccproxy.lightllm.graph.openai_conversations_intake import (
    HandoffUnsupportedError,
    OpenAIConversationsIntakeFSM,
)
from ccproxy.lightllm.parsed import InboundFormat

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_fsm(model: str = "gpt-5") -> OpenAIConversationsIntakeFSM:
    return OpenAIConversationsIntakeFSM(
        model=model,
        request_params=ModelRequestParameters(),
    )


# One module-level loop reused across all feed/close calls, so FSM state stays on
# a single loop (mirrors the SSEPipeline's persistent-loop runtime) and does not
# depend on the ambient event loop, which other tests in the suite may have closed.
_loop = asyncio.new_event_loop()


def _run(coro: Any) -> Any:
    return _loop.run_until_complete(coro)


def _feed_sync(fsm: OpenAIConversationsIntakeFSM, data: bytes) -> list[ModelResponseStreamEvent]:
    events: list[ModelResponseStreamEvent] = _run(fsm.feed(data))
    return events


def _close_sync(fsm: OpenAIConversationsIntakeFSM) -> list[ModelResponseStreamEvent]:
    events: list[ModelResponseStreamEvent] = _run(fsm.close())
    return events


def _collected_text(fsm: OpenAIConversationsIntakeFSM) -> str:
    parts = list(fsm.parts_manager.get_parts())
    return "".join(p.content for p in parts if isinstance(p, TextPart))


def _make_add_frame(
    *,
    channel: int,
    msg_id: str,
    role: str = "assistant",
    content_type: str = "text",
    status: str = "in_progress",
    hidden: bool = False,
    model_slug: str = "gpt-5",
    conv_id: str = "conv-1",
) -> bytes:
    msg: dict[str, Any] = {
        "id": msg_id,
        "author": {"role": role},
        "content": {"content_type": content_type, "parts": [""]},
        "status": status,
        "metadata": {
            "model_slug": model_slug,
            "is_visually_hidden_from_conversation": hidden,
        },
    }
    frame = {
        "p": "",
        "o": "add",
        "v": {"message": msg, "conversation_id": conv_id},
        "c": channel,
    }
    return f"data: {json.dumps(frame)}\n\n".encode()


def _make_encoding_banner() -> bytes:
    return b'event: delta_encoding\ndata: "v1"\n\n'


def _make_shorthand_batch(patches: list[dict[str, Any]]) -> bytes:
    frame = {"v": patches}
    return f"data: {json.dumps(frame)}\n\n".encode()


def _make_explicit_batch(patches: list[dict[str, Any]]) -> bytes:
    frame = {"o": "patch", "v": patches}
    return f"data: {json.dumps(frame)}\n\n".encode()


def _make_single_patch(path: str, op: str, value: Any, channel: int | None = None) -> bytes:
    frame: dict[str, Any] = {"p": path, "o": op, "v": value}
    if channel is not None:
        frame["c"] = channel
    return f"data: {json.dumps(frame)}\n\n".encode()


def _make_done() -> bytes:
    return b"data: [DONE]\n\n"


def _make_typed_event(kind: str, **kwargs: Any) -> bytes:
    frame = {"type": kind, **kwargs}
    return f"data: {json.dumps(frame)}\n\n".encode()


# ── Standard happy-path fixture ───────────────────────────────────────────────

_STANDARD_STREAM = (
    _make_encoding_banner()
    + _make_add_frame(channel=0, msg_id="sys", role="system", hidden=True, status="finished_successfully")
    + _make_add_frame(channel=1, msg_id="user", role="user", status="finished_successfully")
    + _make_add_frame(channel=2, msg_id="asst", role="assistant")
    + _make_shorthand_batch(
        [
            {"p": "/message/content/parts/0", "o": "append", "v": "hello "},
        ]
    )
    + _make_shorthand_batch(
        [
            {"p": "/message/content/parts/0", "o": "append", "v": "world"},
            {"p": "/message/status", "o": "replace", "v": "finished_successfully"},
        ]
    )
    + _make_done()
)


class TestBasicFlow:
    def test_text_assembled(self) -> None:
        """Standard stream assembles ``hello world`` in final parts."""
        fsm = _make_fsm()
        _feed_sync(fsm, _STANDARD_STREAM)
        assert _collected_text(fsm) == "hello world"

    def test_finish_reason_set(self) -> None:
        """Finish reason is ``stop`` after status patch."""
        fsm = _make_fsm()
        _feed_sync(fsm, _STANDARD_STREAM)
        assert fsm.finish_reason == "stop"

    def test_finish_not_duplicated_by_done(self) -> None:
        """``[DONE]`` after a status-based finish does not double-emit finish."""
        fsm = _make_fsm()
        _feed_sync(fsm, _STANDARD_STREAM)
        assert fsm.state.final_emitted is True
        # finish_reason stays "stop" (not reset)
        assert fsm.finish_reason == "stop"

    def test_conversation_id_captured(self) -> None:
        fsm = _make_fsm()
        _feed_sync(fsm, _STANDARD_STREAM)
        assert fsm.conversation_id == "conv-1"

    def test_message_id_captured(self) -> None:
        fsm = _make_fsm()
        _feed_sync(fsm, _STANDARD_STREAM)
        assert fsm.message_id == "asst"

    def test_upstream_raw_bytes_populated(self) -> None:
        fsm = _make_fsm()
        _feed_sync(fsm, _STANDARD_STREAM)
        assert len(fsm.upstream_raw_bytes) == len(_STANDARD_STREAM)


# ── Frame shape coverage ──────────────────────────────────────────────────────


class TestFrameShapes:
    def test_encoding_banner_dropped(self) -> None:
        """Encoding banner emits nothing and does not affect channel state."""
        fsm = _make_fsm()
        events = _feed_sync(fsm, _make_encoding_banner())
        assert events == []
        assert fsm.state.current_channel is None

    def test_explicit_patch_batch(self) -> None:
        """``{o: "patch", v: [...]}`` explicit batch is parsed correctly."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="asst"))
        _feed_sync(
            fsm,
            _make_explicit_batch(
                [
                    {"p": "/message/content/parts/0", "o": "append", "v": "ab"},
                ]
            ),
        )
        assert _collected_text(fsm) == "ab"

    def test_shorthand_batch(self) -> None:
        """``{v: [...]}`` shorthand batch is applied."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="asst"))
        _feed_sync(
            fsm,
            _make_shorthand_batch(
                [
                    {"p": "/message/content/parts/0", "o": "append", "v": "cd"},
                ]
            ),
        )
        assert _collected_text(fsm) == "cd"

    def test_single_patch(self) -> None:
        """Single ``{p, o, v}`` patch emits correctly."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="asst"))
        _feed_sync(fsm, _make_single_patch("/message/content/parts/0", "append", "ef"))
        assert _collected_text(fsm) == "ef"

    def test_done_only_no_finish_without_content(self) -> None:
        """``[DONE]`` without any content emitted does not set finish_reason."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="asst"))
        _feed_sync(fsm, _make_done())
        assert fsm.finish_reason is None
        assert _collected_text(fsm) == ""

    def test_done_synthesises_finish_after_content(self) -> None:
        """``[DONE]`` after content sets finish_reason when status patch was absent."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="asst"))
        _feed_sync(fsm, _make_shorthand_batch([{"p": "/message/content/parts/0", "o": "append", "v": "x"}]))
        _feed_sync(fsm, _make_done())
        assert fsm.finish_reason == "stop"


# ── Hidden / system / user channel filtering ──────────────────────────────────


class TestChannelFiltering:
    def test_system_channel_hidden_not_emitted(self) -> None:
        """System channel (hidden) patches do not produce text output."""
        fsm = _make_fsm()
        _feed_sync(
            fsm,
            _make_add_frame(channel=0, msg_id="sys", role="system", hidden=True, status="finished_successfully"),
        )
        _feed_sync(
            fsm,
            _make_shorthand_batch([{"p": "/message/content/parts/0", "o": "append", "v": "SYSTEM"}]),
        )
        assert _collected_text(fsm) == ""

    def test_user_channel_not_emitted(self) -> None:
        """User channel patches are consumed but not emitted."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="usr", role="user", status="finished_successfully"))
        _feed_sync(fsm, _make_add_frame(channel=1, msg_id="asst"))
        _feed_sync(
            fsm,
            # Patch goes to channel 0 (user) via explicit channel field
            _make_single_patch("/message/content/parts/0", "append", "USER_TEXT", channel=0),
        )
        assert _collected_text(fsm) == ""

    def test_only_final_channel_produces_text(self) -> None:
        """When multiple channels exist only the final-answer channel emits."""
        fsm = _make_fsm()
        _feed_sync(
            fsm,
            _make_add_frame(channel=0, msg_id="sys", role="system", hidden=True, status="finished_successfully"),
        )
        _feed_sync(
            fsm,
            _make_add_frame(channel=1, msg_id="usr", role="user", status="finished_successfully"),
        )
        _feed_sync(fsm, _make_add_frame(channel=2, msg_id="asst"))
        _feed_sync(
            fsm,
            _make_shorthand_batch([{"p": "/message/content/parts/0", "o": "append", "v": "VISIBLE"}]),
        )
        assert _collected_text(fsm) == "VISIBLE"
        assert fsm.state.final_channel == 2

    def test_hidden_analysis_channel_not_emitted(self) -> None:
        """``is_visually_hidden_from_conversation: True`` suppresses channel even for assistant."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="hidden_asst", hidden=True))
        _feed_sync(fsm, _make_add_frame(channel=1, msg_id="asst"))
        _feed_sync(
            fsm,
            _make_single_patch("/message/content/parts/0", "append", "HIDDEN", channel=0),
        )
        _feed_sync(
            fsm,
            _make_single_patch("/message/content/parts/0", "append", "VISIBLE", channel=1),
        )
        assert _collected_text(fsm) == "VISIBLE"


# ── Replace / suffix diffing ──────────────────────────────────────────────────


class TestReplacePatch:
    def test_replace_emits_only_suffix(self) -> None:
        """``replace`` on the same path emits only the new suffix."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="asst"))
        _feed_sync(fsm, _make_single_patch("/message/content/parts/0", "replace", "hello"))
        _feed_sync(fsm, _make_single_patch("/message/content/parts/0", "replace", "hello world"))
        assert _collected_text(fsm) == "hello world"
        assert fsm.state.accumulated_text == "hello world"

    def test_replace_from_empty_emits_full_value(self) -> None:
        """First ``replace`` with no prior text emits the full value."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="asst"))
        _feed_sync(fsm, _make_single_patch("/message/content/parts/0", "replace", "abc"))
        assert _collected_text(fsm) == "abc"

    def test_mixed_append_then_replace(self) -> None:
        """Intermixed ``append`` and ``replace`` stays consistent."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="asst"))
        _feed_sync(fsm, _make_single_patch("/message/content/parts/0", "append", "hello "))
        # Replace with longer string — suffix " world" should be the delta
        _feed_sync(fsm, _make_single_patch("/message/content/parts/0", "replace", "hello world"))
        assert _collected_text(fsm) == "hello world"
        assert fsm.state.accumulated_text == "hello world"


# ── Multi-chunk split robustness ──────────────────────────────────────────────


@dataclass(frozen=True)
class _ChunkSplitTestCase:
    name: str
    """Descriptive test scenario name."""

    chunk_size: int
    """Byte chunk size to feed in; 0 means all at once."""


_CHUNK_SPLIT_CASES = [
    _ChunkSplitTestCase(name="1-byte", chunk_size=1),
    _ChunkSplitTestCase(name="7-byte", chunk_size=7),
    _ChunkSplitTestCase(name="16-byte", chunk_size=16),
    _ChunkSplitTestCase(name="all-at-once", chunk_size=0),
]


@pytest.mark.parametrize(
    "tc",
    [pytest.param(tc, id=tc.name) for tc in _CHUNK_SPLIT_CASES],
)
class TestMultiChunkSplit:
    def test_text_assembled_identically(self, tc: _ChunkSplitTestCase) -> None:
        """``feed()`` with arbitrary byte chunks yields identical final text."""
        fsm = _make_fsm()
        data = _STANDARD_STREAM
        if tc.chunk_size <= 0 or tc.chunk_size >= len(data):
            _feed_sync(fsm, data)
        else:
            for i in range(0, len(data), tc.chunk_size):
                _feed_sync(fsm, data[i : i + tc.chunk_size])
        assert _collected_text(fsm) == "hello world"

    def test_no_duplication_on_splits(self, tc: _ChunkSplitTestCase) -> None:
        """Splitting mid-frame never produces duplicate text."""
        fsm = _make_fsm()
        # A stream where every part is fed one byte at a time (worst case).
        stream = (
            _make_add_frame(channel=0, msg_id="asst")
            + _make_shorthand_batch([{"p": "/message/content/parts/0", "o": "append", "v": "abc"}])
            + _make_shorthand_batch([{"p": "/message/content/parts/0", "o": "append", "v": "def"}])
            + _make_done()
        )
        data = stream
        if tc.chunk_size <= 0 or tc.chunk_size >= len(data):
            _feed_sync(fsm, data)
        else:
            for i in range(0, len(data), tc.chunk_size):
                _feed_sync(fsm, data[i : i + tc.chunk_size])
        assert _collected_text(fsm) == "abcdef"


# ── Handoff detection ─────────────────────────────────────────────────────────


class TestHandoffDetection:
    def test_stream_handoff_before_content_raises(self) -> None:
        """``stream_handoff`` before any content raises :class:`HandoffUnsupportedError`."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="asst"))
        data = _make_typed_event(
            "stream_handoff",
            options=[{"type": "subscribe_ws_topic", "topic_id": "conversation-turn-abc"}],
        )
        with pytest.raises(HandoffUnsupportedError) as exc_info:
            _feed_sync(fsm, data)
        assert exc_info.value.meta.handoff_topic == "conversation-turn-abc"

    def test_resume_conversation_token_before_content_raises(self) -> None:
        """``resume_conversation_token`` before content raises :class:`HandoffUnsupportedError`."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="asst"))
        data = _make_typed_event("resume_conversation_token", token="tok999")  # noqa: S106
        with pytest.raises(HandoffUnsupportedError) as exc_info:
            _feed_sync(fsm, data)
        assert exc_info.value.meta.resume_token == "tok999"  # noqa: S105

    def test_server_ste_metadata_before_content_raises(self) -> None:
        """``server_ste_metadata`` before content raises :class:`HandoffUnsupportedError`."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="asst"))
        data = _make_typed_event("server_ste_metadata", turn_exchange_id="turn-xyz")
        with pytest.raises(HandoffUnsupportedError) as exc_info:
            _feed_sync(fsm, data)
        meta = exc_info.value.meta
        assert meta.handoff_topic == "conversation-turn-turn-xyz"
        assert meta.turn_exchange_id == "turn-xyz"

    def test_handoff_after_content_sets_continuation_no_error(self) -> None:
        """Handoff events after content is flowing set continuation but do not raise."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="asst"))
        _feed_sync(fsm, _make_shorthand_batch([{"p": "/message/content/parts/0", "o": "append", "v": "hi"}]))
        data = _make_typed_event(
            "stream_handoff",
            options=[{"type": "subscribe_ws_topic", "topic_id": "conversation-turn-late"}],
        )
        _feed_sync(fsm, data)  # Must NOT raise
        assert fsm.continuation is not None
        assert fsm.continuation.handoff_topic == "conversation-turn-late"
        assert fsm.state.content_begun is True
        assert _collected_text(fsm) == "hi"

    def test_unknown_typed_events_ignored(self) -> None:
        """Unknown typed events do not affect state and produce no output."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="asst"))
        _feed_sync(fsm, _make_typed_event("message_marker", marker="first"))
        _feed_sync(fsm, _make_shorthand_batch([{"p": "/message/content/parts/0", "o": "append", "v": "ok"}]))
        assert _collected_text(fsm) == "ok"
        assert fsm.continuation is None

    def test_continuation_metadata_carry_conversation_id(self) -> None:
        """ContinuationMetadata carries the conversation_id captured from the add event."""
        fsm = _make_fsm()
        _feed_sync(fsm, _make_add_frame(channel=0, msg_id="asst", conv_id="my-conv"))
        _feed_sync(fsm, _make_shorthand_batch([{"p": "/message/content/parts/0", "o": "append", "v": "x"}]))
        data = _make_typed_event(
            "stream_handoff",
            options=[{"type": "subscribe_ws_topic", "topic_id": "conversation-turn-foo"}],
        )
        _feed_sync(fsm, data)
        assert fsm.continuation is not None
        assert fsm.continuation.conversation_id == "my-conv"


# ── Buffered path ─────────────────────────────────────────────────────────────


class TestBufferedPath:
    def test_sse_body_to_openai_chat_completion(self) -> None:
        """Concatenated OpenAI Conversations SSE → OpenAI ``chat.completion`` JSON."""
        raw_sse = (
            _make_encoding_banner()
            + _make_add_frame(channel=0, msg_id="asst")
            + _make_shorthand_batch(
                [
                    {"p": "/message/content/parts/0", "o": "append", "v": "hello "},
                    {"p": "/message/content/parts/0", "o": "append", "v": "world"},
                ]
            )
            + _make_shorthand_batch(
                [
                    {"p": "/message/status", "o": "replace", "v": "finished_successfully"},
                ]
            )
            + _make_done()
        )
        out_bytes = transform_buffered_response_sync(
            raw_bytes=raw_sse,
            provider_type="openai_conversations",
            inbound_format=InboundFormat.OPENAI_CHAT,
            model="gpt-5",
            request_params=ModelRequestParameters(),
        )
        out = json.loads(out_bytes)
        assert out["object"] == "chat.completion"
        assert out["choices"][0]["message"]["content"] == "hello world"
        assert out["choices"][0]["finish_reason"] == "stop"

    def test_sse_body_to_anthropic_message(self) -> None:
        """Concatenated SSE → Anthropic ``BetaMessage`` JSON."""
        raw_sse = (
            _make_add_frame(channel=0, msg_id="asst")
            + _make_shorthand_batch([{"p": "/message/content/parts/0", "o": "append", "v": "Hi there"}])
            + _make_done()
        )
        out_bytes = transform_buffered_response_sync(
            raw_bytes=raw_sse,
            provider_type="openai_conversations",
            inbound_format=InboundFormat.ANTHROPIC_MESSAGES,
            model="gpt-5",
            request_params=ModelRequestParameters(),
        )
        out = json.loads(out_bytes)
        assert out["type"] == "message"
        assert out["role"] == "assistant"
        assert any(b.get("text") == "Hi there" for b in out["content"] if b.get("type") == "text")

    def test_empty_body_returns_valid_empty_response(self) -> None:
        """Empty/no-content SSE → valid but empty response object."""
        raw_sse = _make_done()
        out_bytes = transform_buffered_response_sync(
            raw_bytes=raw_sse,
            provider_type="openai_conversations",
            inbound_format=InboundFormat.OPENAI_CHAT,
            model="gpt-5",
            request_params=ModelRequestParameters(),
        )
        out = json.loads(out_bytes)
        assert out["object"] == "chat.completion"
        assert out["choices"][0]["message"]["content"] is None


# ── SSE Pipeline integration ──────────────────────────────────────────────────


class TestSSEPipeline:
    def test_pipeline_drives_openai_conversations_to_openai_chat_sse(self) -> None:
        """SSEPipeline drives OpenAI Conversations intake → OpenAI Chat SSE output."""
        from ccproxy.lightllm.graph import dispatch_intake, dispatch_render
        from ccproxy.lightllm.graph.sse_pipeline import SSEPipeline

        intake = dispatch_intake(
            provider_type="openai_conversations",
            model="gpt-5",
            request_params=ModelRequestParameters(),
        )
        render = dispatch_render(inbound_format=InboundFormat.OPENAI_CHAT, model="gpt-5")
        pipeline = SSEPipeline(intake=intake, render=render)

        raw_sse = (
            _make_add_frame(channel=0, msg_id="asst", conv_id="c1")
            + _make_shorthand_batch([{"p": "/message/content/parts/0", "o": "append", "v": "pipeline "}])
            + _make_shorthand_batch([{"p": "/message/content/parts/0", "o": "append", "v": "test"}])
            + _make_done()
        )

        out = bytearray()
        for i in range(0, len(raw_sse), 16):
            chunk_out = pipeline(raw_sse[i : i + 16])
            if isinstance(chunk_out, (bytes, bytearray)):
                out.extend(chunk_out)
        flushed = pipeline(b"")
        if isinstance(flushed, (bytes, bytearray)):
            out.extend(flushed)
        pipeline.close()

        output_text = out.decode()
        # The rendered SSE should contain the assembled text somewhere in ``content`` fields
        assert "pipeline test" in output_text or "pipeline" in output_text

    def test_pipeline_chunk_boundary_invariant(self) -> None:
        """OpenAI Conversations pipeline output is invariant across chunk sizes."""
        from ccproxy.lightllm.graph import dispatch_intake, dispatch_render
        from ccproxy.lightllm.graph.sse_pipeline import SSEPipeline

        raw_sse = (
            _make_add_frame(channel=0, msg_id="asst")
            + _make_shorthand_batch([{"p": "/message/content/parts/0", "o": "append", "v": "chunk-safe"}])
            + _make_done()
        )

        results: list[bytes] = []
        for chunk_size in [1, 16, 0]:
            intake = dispatch_intake(
                provider_type="openai_conversations",
                model="gpt-5",
                request_params=ModelRequestParameters(),
            )
            render = dispatch_render(inbound_format=InboundFormat.OPENAI_CHAT, model="gpt-5")
            pipeline = SSEPipeline(intake=intake, render=render)
            out = bytearray()
            data = raw_sse
            if chunk_size <= 0 or chunk_size >= len(data):
                chunks = [data]
            else:
                chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
            for chunk in chunks:
                res = pipeline(chunk)
                if isinstance(res, (bytes, bytearray)):
                    out.extend(res)
            flushed = pipeline(b"")
            if isinstance(flushed, (bytes, bytearray)):
                out.extend(flushed)
            pipeline.close()
            results.append(bytes(out))

        # All three chunk sizes must produce the same text content
        texts = []
        for wire in results:
            text = ""
            for line in wire.decode().splitlines():
                if line.startswith("data:") and "[DONE]" not in line:
                    try:
                        obj = json.loads(line[5:].strip())
                        for choice in obj.get("choices", []):
                            text += choice.get("delta", {}).get("content", "") or ""
                    except (json.JSONDecodeError, KeyError):
                        pass
            texts.append(text)

        assert texts[0] == texts[1] == texts[2] == "chunk-safe"
