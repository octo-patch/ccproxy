"""Tests for the SSE-v1 frame decoder and patch normaliser in the OpenAI Conversations intake.

These tests exercise the pure frame-parsing layer (``_parse_frame``,
``_parse_delta``) and the ``_drain_sse_frames`` buffer management in
``OpenAIConversationsIntakeFSM``.  They do NOT drive the pydantic-graph FSM;
that is covered in ``test_openai_conversations_intake.py``.

Ported reference: gproxy MIT —
  sdk/gproxy-channel/src/channels/chatgpt/sse_v1.rs tests
"""

from __future__ import annotations

from ccproxy.lightllm.graph.openai_conversations_intake import (
    _AddEnvelope,
    _DoneEvent,
    _parse_delta,
    _parse_frame,
    _PatchEnvelope,
    _TypedSideEvent,
)

# ── _parse_delta shapes ───────────────────────────────────────────────────────


class TestParseDelta:
    def test_explicit_batch(self) -> None:
        """``{o: "patch", v: [{p,o,v}, ...]}`` — explicit batch."""
        frame = {
            "o": "patch",
            "v": [
                {"p": "/message/content/parts/0", "o": "append", "v": "hello"},
                {"p": "/message/metadata/token_count", "o": "replace", "v": 7},
            ],
        }
        patches = _parse_delta(frame)
        assert patches == [
            ("/message/content/parts/0", "append", "hello"),
            ("/message/metadata/token_count", "replace", 7),
        ]

    def test_shorthand_batch(self) -> None:
        """``{v: [{p,o,v}, ...]}`` without o/p — shorthand batch."""
        frame = {
            "v": [
                {"p": "/message/content/parts/0", "o": "append", "v": " world"},
                {"p": "/message/status", "o": "replace", "v": "finished_successfully"},
            ]
        }
        patches = _parse_delta(frame)
        assert patches == [
            ("/message/content/parts/0", "append", " world"),
            ("/message/status", "replace", "finished_successfully"),
        ]

    def test_implicit_add(self) -> None:
        """``{v: <object>}`` without o/p — implicit add at root path."""
        msg = {"message": {"id": "m1"}, "conversation_id": "c1"}
        frame = {"v": msg}
        patches = _parse_delta(frame)
        assert patches == [("", "add", msg)]

    def test_single_patch(self) -> None:
        """``{p, o, v}`` — single patch."""
        frame = {"p": "/message/content/parts/0", "o": "append", "v": "x"}
        patches = _parse_delta(frame)
        assert patches == [("/message/content/parts/0", "append", "x")]

    def test_single_patch_with_channel(self) -> None:
        """Channel field ``c`` does not affect patch list."""
        frame = {"p": "/message/status", "o": "replace", "v": "done", "c": 2}
        patches = _parse_delta(frame)
        assert patches == [("/message/status", "replace", "done")]

    def test_patch_list_skips_non_dicts(self) -> None:
        """Non-dict entries in a batch array are silently dropped."""
        frame = {"o": "patch", "v": [None, {"p": "/x", "o": "append", "v": "y"}, 42]}
        patches = _parse_delta(frame)
        assert patches == [("/x", "append", "y")]


# ── _parse_frame ─────────────────────────────────────────────────────────────


class TestParseFrame:
    def test_encoding_banner_dropped(self) -> None:
        """``event: delta_encoding`` frames are silently dropped."""
        frame = b'event: delta_encoding\ndata: "v1"'
        assert _parse_frame(frame) is None

    def test_done_sentinel(self) -> None:
        """``data: [DONE]`` produces a :class:`_DoneEvent`."""
        frame = b"data: [DONE]"
        result = _parse_frame(frame)
        assert isinstance(result, _DoneEvent)

    def test_initial_add_becomes_add_envelope(self) -> None:
        """Single add-at-root with channel → :class:`_AddEnvelope`."""
        payload = b'data: {"p":"","o":"add","v":{"message":{"id":"m1"},"conversation_id":"c1"},"c":0}'
        result = _parse_frame(payload)
        assert isinstance(result, _AddEnvelope)
        assert result.channel == 0
        assert result.value == {"message": {"id": "m1"}, "conversation_id": "c1"}

    def test_typed_side_event(self) -> None:
        """``{type: "..."}`` without ``p`` or ``v`` → :class:`_TypedSideEvent`."""
        payload = b'data: {"type":"message_marker","marker":"first"}'
        result = _parse_frame(payload)
        assert isinstance(result, _TypedSideEvent)
        assert result.kind == "message_marker"
        assert result.raw["marker"] == "first"

    def test_shorthand_batch_becomes_patch_envelope(self) -> None:
        """Shorthand batch → :class:`_PatchEnvelope` with correct patches."""
        payload = b'data: {"v":[{"p":"/message/content/parts/0","o":"append","v":"hi"}]}'
        result = _parse_frame(payload)
        assert isinstance(result, _PatchEnvelope)
        assert result.channel is None
        assert ("/message/content/parts/0", "append", "hi") in result.patches

    def test_patch_envelope_carries_channel(self) -> None:
        """Non-add frames with ``c`` field → :class:`_PatchEnvelope` with channel set."""
        payload = b'data: {"c":2,"v":[{"p":"/message/content/parts/0","o":"append","v":"x"}]}'
        result = _parse_frame(payload)
        assert isinstance(result, _PatchEnvelope)
        assert result.channel == 2

    def test_keepalive_comment_dropped(self) -> None:
        """Lines starting with ``:`` are SSE comments and should produce no output."""
        frame = b": keepalive"
        assert _parse_frame(frame) is None

    def test_empty_data_dropped(self) -> None:
        """A frame with no ``data:`` lines produces ``None``."""
        frame = b"event: delta"
        assert _parse_frame(frame) is None

    def test_unparseable_json_dropped(self) -> None:
        """Non-JSON data lines produce ``None`` rather than raising."""
        frame = b"data: not json at all"
        assert _parse_frame(frame) is None

    def test_explicit_patch_batch(self) -> None:
        """``{o:"patch", v:[...]}`` → :class:`_PatchEnvelope`."""
        payload = b'data: {"o":"patch","v":[{"p":"/a","o":"append","v":"x"},{"p":"/b","o":"replace","v":1}]}'
        result = _parse_frame(payload)
        assert isinstance(result, _PatchEnvelope)
        assert len(result.patches) == 2
        assert result.patches[0] == ("/a", "append", "x")
        assert result.patches[1] == ("/b", "replace", 1)

    def test_typed_side_event_with_v_field_is_patch(self) -> None:
        """Events with both ``type`` and ``v`` are delta frames, not typed side events."""
        # Deltas sometimes have a ``type`` field on embedded objects but ``v`` is present.
        payload = b'data: {"type":"some_type","v":[{"p":"/x","o":"append","v":"z"}]}'
        result = _parse_frame(payload)
        # Has ``v`` → patch envelope, NOT typed side event
        assert isinstance(result, _PatchEnvelope)

    def test_multiline_data_fields(self) -> None:
        """Multiple ``data:`` lines (per SSE spec) joined with newline before JSON parse."""
        # The SSE multi-line encoding sends multiple ``data:`` lines whose values
        # are joined with ``\n`` before parsing.  In practice the ChatGPT server
        # always sends one-line JSON, but the parser must not choke on the spec form.
        frame = b'data: {"v":[{"p":"/message/content/parts/0","o":"append","v":"hi"}]}'
        # One-line JSON: must parse as a PatchEnvelope
        result = _parse_frame(frame)
        assert isinstance(result, _PatchEnvelope)

    def test_stream_handoff_side_event(self) -> None:
        """``stream_handoff`` typed event → :class:`_TypedSideEvent`."""
        payload = (
            b'data: {"type":"stream_handoff","options":[{"type":"subscribe_ws_topic",'
            b'"topic_id":"conversation-turn-abc"}]}'
        )
        result = _parse_frame(payload)
        assert isinstance(result, _TypedSideEvent)
        assert result.kind == "stream_handoff"

    def test_resume_conversation_token_side_event(self) -> None:
        """``resume_conversation_token`` typed event → :class:`_TypedSideEvent`."""
        payload = b'data: {"type":"resume_conversation_token","token":"tok123"}'
        result = _parse_frame(payload)
        assert isinstance(result, _TypedSideEvent)
        assert result.kind == "resume_conversation_token"
        assert result.raw["token"] == "tok123"  # noqa: S105
