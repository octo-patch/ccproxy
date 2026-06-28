"""Tests for ``ccproxy.openai_conversations.ws_handoff`` and the sidecar
continuation seam.

Coverage:
- ``HandoffState.feed`` / ``detect_handoff`` — topic extraction from
  ``stream_handoff`` / ``server_ste_metadata`` / ``resume_conversation_token``.
- ``HandoffState.should_bridge`` — respects "no HTTP content" rule.
- ``_parse_ws_frames`` — JSON array vs single object.
- ``_sse_items_from_frame`` — ``encoded_item`` path and
  ``conversation-update`` path.
- ``_process_reply_frame`` — ``reply.topic_id`` filtering + ``catchups``
  iteration + ``[DONE]`` terminal.
- ``stream_handoff_sse`` using a fake in-process WebSocket server: frames
  fed by the server → SSE bytes yielded by the generator.
- End-to-end: fake HTTP SSE ending with a handoff marker + fake WS feeding
  frames → assembled SSE bytes piped through ``OpenAIConversationsIntakeFSM``
  → expected assistant text.
- Sidecar ``body_stream`` continuation: upstream body ends with a handoff
  marker + ``X-CCProxy-Continuation`` header → ``body_stream`` emits the
  WS-derived continuation.
- ``_refresh_sentinel`` chat-requirements expiry: decoded from the finalize
  token's JWT ``exp`` claim (unit test, no network).
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic_ai.messages import TextPart
from pydantic_ai.models import ModelRequestParameters
from websockets.asyncio.server import ServerConnection
from websockets.asyncio.server import serve as ws_serve

from ccproxy.config import CCProxyConfig, Provider, set_config_instance
from ccproxy.inspector.openai_conversations_addon import _refresh_sentinel
from ccproxy.inspector.transport_override_addon import TransportOverrideAddon
from ccproxy.lightllm.graph.openai_conversations_intake import OpenAIConversationsIntakeFSM
from ccproxy.openai_conversations.ws_handoff import (
    HandoffState,
    _conversation_update_item,
    _encoded_item,
    _is_done_item,
    _parse_ws_frames,
    _process_reply_frame,
    _sse_items_from_frame,
    detect_handoff,
    run_handoff_bridge,
    run_resume_bridge,
    stream_handoff_sse,
)
from ccproxy.transport import UnknownFingerprintProfileError
from ccproxy.transport.sidecar import CONTINUATION_HEADER, Sidecar

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_fsm(model: str = "gpt-5") -> OpenAIConversationsIntakeFSM:
    return OpenAIConversationsIntakeFSM(
        model=model,
        request_params=ModelRequestParameters(),
    )


_loop = asyncio.new_event_loop()


def _run(coro: Any) -> Any:
    return _loop.run_until_complete(coro)


def _sse(data: str) -> bytes:
    return f"data: {data}\n\n".encode()


def _stream_handoff_frame(topic_id: str) -> bytes:
    payload = {
        "type": "stream_handoff",
        "options": [
            {"type": "subscribe_ws_topic", "topic_id": topic_id},
        ],
    }
    return _sse(json.dumps(payload))


def _server_ste_frame(turn_exchange_id: str) -> bytes:
    payload = {"type": "server_ste_metadata", "turn_exchange_id": turn_exchange_id}
    return _sse(json.dumps(payload))


def _server_ste_nested_frame(turn_exchange_id: str) -> bytes:
    payload = {
        "type": "server_ste_metadata",
        "metadata": {"turn_exchange_id": turn_exchange_id},
    }
    return _sse(json.dumps(payload))


def _resume_token_frame(token: str = "tok123", conversation_id: str = "conv-resume") -> bytes:  # noqa: S107
    payload = {"type": "resume_conversation_token", "token": token, "conversation_id": conversation_id}
    return _sse(json.dumps(payload))


def _content_append_frame(channel: int, text: str) -> bytes:
    patch = {"p": "/message/content/parts/0", "o": "append", "v": text, "c": channel}
    return _sse(json.dumps(patch))


# ── HandoffState / detect_handoff ─────────────────────────────────────────────


class TestHandoffStateDetection:
    def test_no_handoff_initially(self) -> None:
        state = HandoffState()
        assert state.topic == ""
        assert not state.should_bridge()

    def test_stream_handoff_sets_topic(self) -> None:
        state = HandoffState()
        detect_handoff(state, _stream_handoff_frame("conversation-turn-abc123"))
        assert state.topic == "conversation-turn-abc123"

    def test_server_ste_metadata_is_not_a_handoff_topic(self) -> None:
        # server_ste_metadata is telemetry, not a WS handoff (it rides every turn);
        # it must not set a topic or trigger the WS bridge.
        state = HandoffState()
        detect_handoff(state, _server_ste_frame("xyz789"))
        assert state.topic == ""
        assert not state.should_bridge()

    def test_server_ste_metadata_nested_is_not_a_handoff_topic(self) -> None:
        state = HandoffState()
        detect_handoff(state, _server_ste_nested_frame("nested999"))
        assert state.topic == ""
        assert not state.should_bridge()

    def test_server_ste_with_resume_token_routes_to_resume(self) -> None:
        # The real shape: every turn carries server_ste_metadata + resume token.
        # No stream_handoff → no WS bridge → HTTP resume is the continuation.
        state = HandoffState()
        detect_handoff(state, _resume_token_frame(token="tokR", conversation_id="conv-R"))  # noqa: S106
        detect_handoff(state, _server_ste_frame("xyz789"))
        assert state.topic == ""
        assert state.should_bridge() is False
        assert state.should_resume() is True

    def test_resume_conversation_token_sets_resume_signal(self) -> None:
        state = HandoffState()
        detect_handoff(state, _resume_token_frame(token="tokR", conversation_id="conv-R"))  # noqa: S106
        assert state.resume_token == "tokR"  # noqa: S105
        assert state.conversation_id == "conv-R"
        assert state.topic == ""
        assert not state.should_bridge()
        assert state.should_resume() is True
        assert state.should_continue() is True

    def test_resume_not_triggered_without_conversation_id(self) -> None:
        state = HandoffState()
        # A resume token with no conversation id cannot build the resume body.
        detect_handoff(state, _sse(json.dumps({"type": "resume_conversation_token", "token": "t"})))
        assert state.resume_token == "t"  # noqa: S105
        assert state.conversation_id == ""
        assert state.should_resume() is False

    def test_resume_not_triggered_when_http_content_seen(self) -> None:
        state = HandoffState()
        detect_handoff(state, _resume_token_frame(token="tokR", conversation_id="conv-R"))  # noqa: S106
        detect_handoff(state, _content_append_frame(0, "inline answer"))
        assert state.http_content_seen is True
        assert state.should_resume() is False
        assert state.should_continue() is False

    def test_should_bridge_true_when_topic_and_no_http_content(self) -> None:
        state = HandoffState()
        detect_handoff(state, _stream_handoff_frame("conversation-turn-1"))
        assert state.should_bridge() is True

    def test_should_bridge_false_when_http_content_seen(self) -> None:
        state = HandoffState()
        detect_handoff(state, _stream_handoff_frame("conversation-turn-1"))
        # Simulate an HTTP content append arriving before the handoff.
        detect_handoff(state, _content_append_frame(1, "Hello"))
        assert state.http_content_seen is True
        assert state.should_bridge() is False

    def test_stream_handoff_wins_when_both_signals_present(self) -> None:
        state = HandoffState()
        # resume_conversation_token + stream_handoff together → the WS path wins
        # (the SPA's stream_handoff is the active-stream handler).
        detect_handoff(state, _resume_token_frame(token="tokR", conversation_id="conv-R"))  # noqa: S106
        detect_handoff(state, _stream_handoff_frame("conversation-turn-2"))
        assert state.should_bridge() is True
        assert state.should_resume() is False
        assert state.should_continue() is True

    def test_feed_handles_split_chunks(self) -> None:
        state = HandoffState()
        # topic_id in the frame is stored verbatim (no prefix added for stream_handoff).
        full = _stream_handoff_frame("turn-split-topic")
        mid = len(full) // 2
        detect_handoff(state, full[:mid])
        detect_handoff(state, full[mid:])
        assert state.topic == "turn-split-topic"

    def test_stream_handoff_with_multiple_options_picks_subscribe_ws_topic(self) -> None:
        payload = {
            "type": "stream_handoff",
            "options": [
                {"type": "other_option", "topic_id": "wrong"},
                {"type": "subscribe_ws_topic", "topic_id": "correct-topic"},
            ],
        }
        state = HandoffState()
        detect_handoff(state, _sse(json.dumps(payload)))
        assert state.topic == "correct-topic"

    def test_stream_handoff_with_no_matching_option_sets_empty_topic(self) -> None:
        payload = {"type": "stream_handoff", "options": [{"type": "other_option"}]}
        state = HandoffState()
        detect_handoff(state, _sse(json.dumps(payload)))
        # no subscribe_ws_topic → topic stays empty, should_bridge is False
        assert state.topic == ""
        assert not state.should_bridge()

    def test_batch_patch_triggers_http_content_seen(self) -> None:
        state = HandoffState()
        detect_handoff(state, _stream_handoff_frame("t1"))
        patch_batch = {
            "o": "patch",
            "v": [{"p": "/message/content/parts/0", "o": "append", "v": "Hi"}],
        }
        detect_handoff(state, _sse(json.dumps(patch_batch)))
        assert state.http_content_seen is True
        assert not state.should_bridge()


# ── Parametrized handoff extraction ──────────────────────────────────────────


@dataclass(frozen=True)
class HandoffExtractionCase:
    name: str
    """Scenario identifier."""

    frame_bytes: bytes
    """SSE bytes to feed."""

    expected_topic: str
    """Expected ``state.topic`` after feed."""

    expected_bridge: bool
    """Expected ``state.should_bridge()`` after feed."""


_HANDOFF_EXTRACTION_CASES: list[HandoffExtractionCase] = [
    HandoffExtractionCase(
        name="stream_handoff_basic",
        frame_bytes=_stream_handoff_frame("conversation-turn-basic"),
        expected_topic="conversation-turn-basic",
        expected_bridge=True,
    ),
    HandoffExtractionCase(
        name="server_ste_top_level_is_telemetry",
        frame_bytes=_server_ste_frame("top-level-id"),
        expected_topic="",
        expected_bridge=False,
    ),
    HandoffExtractionCase(
        name="server_ste_nested_is_telemetry",
        frame_bytes=_server_ste_nested_frame("nested-id"),
        expected_topic="",
        expected_bridge=False,
    ),
    HandoffExtractionCase(
        name="resume_token_no_bridge",
        frame_bytes=_resume_token_frame("resume123"),
        expected_topic="",
        expected_bridge=False,
    ),
]


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.name) for c in _HANDOFF_EXTRACTION_CASES],
)
def test_handoff_extraction(case: HandoffExtractionCase) -> None:
    state = HandoffState()
    detect_handoff(state, case.frame_bytes)
    assert state.topic == case.expected_topic
    assert state.should_bridge() == case.expected_bridge


# ── _parse_ws_frames ──────────────────────────────────────────────────────────


class TestParseWsFrames:
    def test_json_array_returns_list(self) -> None:
        frames = [{"type": "reply", "reply": {}}, {"type": "message"}]
        raw = json.dumps(frames).encode()
        result = _parse_ws_frames(raw)
        assert len(result) == 2
        assert result[0]["type"] == "reply"

    def test_single_object_returns_list_of_one(self) -> None:
        frame = {"type": "message", "topic_id": "conv-turn-1"}
        raw = json.dumps(frame).encode()
        result = _parse_ws_frames(raw)
        assert result == [frame]

    def test_empty_bytes_returns_empty(self) -> None:
        assert _parse_ws_frames(b"") == []

    def test_invalid_json_returns_empty(self) -> None:
        assert _parse_ws_frames(b"not json") == []

    def test_array_with_non_dict_items_filtered(self) -> None:
        raw = json.dumps([{"type": "reply"}, "not a dict", 42]).encode()
        result = _parse_ws_frames(raw)
        assert result == [{"type": "reply"}]


# ── _encoded_item and _conversation_update_item ───────────────────────────────


class TestSseItemExtraction:
    def test_encoded_item_extracted(self) -> None:
        topic = "conversation-turn-abc"
        frame = {
            "topic_id": topic,
            "payload": {"payload": {"encoded_item": "data: hello\n\n"}},
        }
        result = _encoded_item(frame, topic)
        assert result == "data: hello\n\n"

    def test_encoded_item_wrong_topic_returns_empty(self) -> None:
        frame = {
            "topic_id": "other-topic",
            "payload": {"payload": {"encoded_item": "data: hello\n\n"}},
        }
        result = _encoded_item(frame, "my-topic")
        assert result == ""

    def test_encoded_item_no_topic_in_frame_accepts_any_topic(self) -> None:
        # When ``topic_id`` is absent in the frame, any topic matches.
        frame = {"payload": {"payload": {"encoded_item": "data: ok\n"}}}
        result = _encoded_item(frame, "any-topic")
        assert result == "data: ok\n"

    def test_conversation_update_item_extracted(self) -> None:
        topic = "conversation-turn-x"
        frame = {
            "topic_id": topic,
            "payload": {"type": "conversation-update", "status": "done"},
        }
        result = _conversation_update_item(frame, topic)
        assert result.startswith("data: ")
        assert "conversation-update" in result

    def test_conversation_update_nested_payload(self) -> None:
        topic = "conversation-turn-y"
        frame = {
            "topic_id": topic,
            "payload": {"payload": {"type": "conversation-update", "id": "123"}},
        }
        result = _conversation_update_item(frame, topic)
        assert result.startswith("data: ")
        assert "conversation-update" in result

    def test_sse_items_from_frame_uses_encoded_first(self) -> None:
        topic = "t1"
        frame = {
            "topic_id": topic,
            "payload": {
                "payload": {"encoded_item": "data: encoded\n\n"},
                "type": "conversation-update",
            },
        }
        items = _sse_items_from_frame(frame, topic)
        assert len(items) == 1
        assert items[0] == "data: encoded\n\n"

    def test_sse_items_from_frame_falls_back_to_update(self) -> None:
        topic = "t2"
        frame = {
            "topic_id": topic,
            "payload": {"type": "conversation-update", "msg": "hi"},
        }
        items = _sse_items_from_frame(frame, topic)
        assert len(items) == 1
        assert "conversation-update" in items[0]


# ── _is_done_item ─────────────────────────────────────────────────────────────


class TestIsDoneItem:
    def test_done_with_space(self) -> None:
        assert _is_done_item("data: [DONE]\n\n") is True

    def test_done_without_space(self) -> None:
        assert _is_done_item("data:[DONE]\n") is True

    def test_non_done_item(self) -> None:
        assert _is_done_item("data: {}\n\n") is False


# ── _process_reply_frame ──────────────────────────────────────────────────────


class TestProcessReplyFrame:
    def test_reply_with_matching_topic_catchup_extracted(self) -> None:
        topic = "conversation-turn-reply"
        catchup = {
            "topic_id": topic,
            "payload": {"payload": {"encoded_item": "data: hello reply\n\n"}},
        }
        frame = {
            "type": "reply",
            "reply": {"topic_id": topic, "catchups": [catchup]},
        }
        items, done = _process_reply_frame(frame, topic)
        assert "data: hello reply\n\n" in items
        assert done is False

    def test_reply_with_done_in_catchup_signals_done(self) -> None:
        topic = "conversation-turn-done"
        catchup = {
            "topic_id": topic,
            "payload": {"payload": {"encoded_item": "data: [DONE]\n\n"}},
        }
        frame = {"type": "reply", "reply": {"topic_id": topic, "catchups": [catchup]}}
        _, done = _process_reply_frame(frame, topic)
        assert done is True

    def test_reply_with_wrong_topic_returns_empty(self) -> None:
        frame = {
            "type": "reply",
            "reply": {
                "topic_id": "wrong-topic",
                "catchups": [{"payload": {"payload": {"encoded_item": "data: x\n\n"}}}],
            },
        }
        items, done = _process_reply_frame(frame, "my-topic")
        assert items == []
        assert done is False

    def test_reply_with_no_reply_key_returns_empty(self) -> None:
        frame = {"type": "reply"}
        items, done = _process_reply_frame(frame, "any-topic")
        assert items == []
        assert done is False


# ── stream_handoff_sse — fake WS server ───────────────────────────────────────


def _make_encoded_sse_frame(topic_id: str, text: str) -> dict[str, Any]:
    """Build a WS message frame carrying an encoded SSE item."""
    return {
        "type": "message",
        "topic_id": topic_id,
        "payload": {"payload": {"encoded_item": f"data: {text}\n\n"}},
    }


def _make_done_frame(topic_id: str) -> dict[str, Any]:
    return {
        "type": "message",
        "topic_id": topic_id,
        "payload": {"payload": {"encoded_item": "data: [DONE]\n\n"}},
    }


async def _collect_ws_sse(ws_url: str, topic_id: str) -> list[bytes]:
    """Collect all bytes yielded by ``stream_handoff_sse`` into a list."""
    chunks: list[bytes] = []
    async for chunk in await stream_handoff_sse(ws_url=ws_url, topic_id=topic_id):
        chunks.append(chunk)
    return chunks


class TestStreamHandoffSseWithFakeServer:
    """Integration tests using a real in-process WebSocket server."""

    async def test_receives_encoded_items_from_ws(self) -> None:
        topic = "conversation-turn-ws1"
        received_texts: list[bytes] = []

        async def _handler(ws: ServerConnection) -> None:
            # Receive init + subscribe messages; ignore them.
            await ws.recv()  # init array
            await ws.recv()  # subscribe
            # Send two content frames then DONE.
            await ws.send(json.dumps([_make_encoded_sse_frame(topic, '{"delta": "Hello"}')]))
            await ws.send(json.dumps([_make_encoded_sse_frame(topic, '{"delta": " world"}')]))
            await ws.send(json.dumps([_make_done_frame(topic)]))

        async with ws_serve(_handler, "127.0.0.1", 0, ping_interval=None) as server:
            port = server.sockets[0].getsockname()[1]
            ws_url = f"ws://127.0.0.1:{port}"
            received_texts = await _collect_ws_sse(ws_url=ws_url, topic_id=topic)

        assert len(received_texts) >= 2
        joined = b"".join(received_texts)
        assert b'{"delta": "Hello"}' in joined
        assert b'{"delta": " world"}' in joined

    async def test_stops_on_done(self) -> None:
        topic = "conversation-turn-ws2"

        async def _handler(ws: ServerConnection) -> None:
            await ws.recv()
            await ws.recv()
            await ws.send(json.dumps([_make_done_frame(topic)]))
            # Additional frames after DONE should be ignored.
            await ws.send(json.dumps([_make_encoded_sse_frame(topic, '{"extra": true}')]))

        async with ws_serve(_handler, "127.0.0.1", 0, ping_interval=None) as server:
            port = server.sockets[0].getsockname()[1]
            ws_url = f"ws://127.0.0.1:{port}"
            chunks = await _collect_ws_sse(ws_url=ws_url, topic_id=topic)

        joined = b"".join(chunks)
        assert b"[DONE]" in joined
        assert b"extra" not in joined

    async def test_filters_wrong_topic_message_frames(self) -> None:
        topic = "conversation-turn-ws3"
        wrong_topic = "other-topic"

        async def _handler(ws: ServerConnection) -> None:
            await ws.recv()
            await ws.recv()
            # Wrong topic frame — should be ignored.
            await ws.send(json.dumps([_make_encoded_sse_frame(wrong_topic, '{"filtered": true}')]))
            # Correct topic frame.
            await ws.send(json.dumps([_make_encoded_sse_frame(topic, '{"correct": true}')]))
            await ws.send(json.dumps([_make_done_frame(topic)]))

        async with ws_serve(_handler, "127.0.0.1", 0, ping_interval=None) as server:
            port = server.sockets[0].getsockname()[1]
            ws_url = f"ws://127.0.0.1:{port}"
            chunks = await _collect_ws_sse(ws_url=ws_url, topic_id=topic)

        joined = b"".join(chunks)
        assert b"filtered" not in joined
        assert b"correct" in joined

    async def test_reply_frame_with_catchups_processed(self) -> None:
        topic = "conversation-turn-ws4"

        async def _handler(ws: ServerConnection) -> None:
            await ws.recv()
            await ws.recv()
            # Reply frame with catchups.
            reply_frame = {
                "type": "reply",
                "reply": {
                    "topic_id": topic,
                    "catchups": [
                        {
                            "topic_id": topic,
                            "payload": {"payload": {"encoded_item": 'data: {"from": "catchup"}\n\n'}},
                        }
                    ],
                },
            }
            await ws.send(json.dumps([reply_frame]))
            await ws.send(json.dumps([_make_done_frame(topic)]))

        async with ws_serve(_handler, "127.0.0.1", 0, ping_interval=None) as server:
            port = server.sockets[0].getsockname()[1]
            ws_url = f"ws://127.0.0.1:{port}"
            chunks = await _collect_ws_sse(ws_url=ws_url, topic_id=topic)

        joined = b"".join(chunks)
        assert b"catchup" in joined

    async def test_error_on_connect_yields_nothing(self) -> None:
        # Non-existent WS server — run_handoff_bridge should yield nothing.
        mock_client = AsyncMock(spec=httpx.AsyncClient)

        async def _bad_fetch(*args: object, **kwargs: object) -> str:
            raise RuntimeError("no server")

        with patch("ccproxy.openai_conversations.ws_handoff.fetch_ws_url", _bad_fetch):
            chunks: list[bytes] = []
            async for chunk in run_handoff_bridge(client=mock_client, topic_id="t1"):
                chunks.append(chunk)
        assert chunks == []


# ── run_resume_bridge (HTTP resume continuation) ──────────────────────────────


_CONTENT_SSE = 'data: {"p": "/message/content/parts/0", "o": "append", "v": "ANSWER"}\n\ndata: [DONE]\n\n'
"""A resume body that carries assistant content (yielded to the intake)."""


def _resume_token_sse(token: str, conversation_id: str) -> str:
    """A resume body that re-hands-off to a fresh conduit (no content)."""
    ev = json.dumps({"type": "resume_conversation_token", "token": token, "conversation_id": conversation_id})
    return f'event: delta_encoding\ndata: "v1"\n\ndata: {ev}\n\ndata: [DONE]\n\n'


class TestRunResumeBridge:
    """``run_resume_bridge`` — POST /f/conversation/resume, offset retry + re-handoff
    chain following.

    Uses a real ``httpx.MockTransport`` adapter (not patching) so the resume
    request shape (URL, ``x-conduit-token``, body, dropped sentinel headers) is
    asserted against actual httpx request objects.
    """

    async def _drive(
        self, handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
    ) -> tuple[list[bytes], list[httpx.Request]]:
        seen: list[httpx.Request] = []

        def _wrap(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(_wrap)) as client:
            chunks = [chunk async for chunk in run_resume_bridge(client=client, **kwargs)]
        return chunks, seen

    async def test_content_body_yielded(self) -> None:
        chunks, seen = await self._drive(
            lambda _r: httpx.Response(200, text=_CONTENT_SSE),
            conversation_id="conv-1",
            resume_token="tokZ",  # noqa: S106
            request_headers={"authorization": "Bearer x", "user-agent": "UA"},
        )
        assert b"ANSWER" in b"".join(chunks)
        assert len(seen) == 1
        assert str(seen[0].url) == "https://chatgpt.com/backend-api/f/conversation/resume"
        assert seen[0].headers["x-conduit-token"] == "tokZ"
        assert json.loads(seen[0].content) == {"conversation_id": "conv-1", "offset": 0}

    async def test_404_advances_offset_until_content(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            offset = json.loads(request.content)["offset"]
            if offset < 2:
                return httpx.Response(404)
            return httpx.Response(200, text=_CONTENT_SSE)

        chunks, seen = await self._drive(
            handler,
            conversation_id="c",
            resume_token="t",  # noqa: S106
            request_headers={},
        )
        assert [json.loads(r.content)["offset"] for r in seen] == [0, 1, 2]
        assert b"ANSWER" in b"".join(chunks)

    async def test_follows_rehandoff_chain_until_content(self) -> None:
        # First two resume responses re-hand-off to a fresh conduit (no content);
        # the third carries the answer. The bridge must follow the chain.
        responses = [
            _resume_token_sse(token="tok2", conversation_id="conv-1"),  # noqa: S106
            _resume_token_sse(token="tok3", conversation_id="conv-1"),  # noqa: S106
            _CONTENT_SSE,
        ]
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            # Each hop POSTs offset 0 first (content/re-handoff arrive at offset 0).
            body = responses[min(calls["n"], len(responses) - 1)]
            calls["n"] += 1
            return httpx.Response(200, text=body)

        chunks, seen = await self._drive(
            handler,
            conversation_id="conv-1",
            resume_token="tok1",  # noqa: S106
            request_headers={},
        )
        assert b"ANSWER" in b"".join(chunks)
        # 3 hops; each hop's conduit token chains tok1 → tok2 → tok3.
        tokens = [r.headers["x-conduit-token"] for r in seen]
        assert tokens == ["tok1", "tok2", "tok3"]

    async def test_rehandoff_without_new_token_stops(self) -> None:
        # A re-handoff body with no content and no fresh resume token ends the chain.
        chunks, seen = await self._drive(
            lambda _r: httpx.Response(200, text='data: {"type":"message_stream_complete"}\n\ndata: [DONE]\n\n'),
            conversation_id="c",
            resume_token="t",  # noqa: S106
            request_headers={},
        )
        assert chunks == []
        assert len(seen) == 1

    async def test_all_404_yields_nothing(self) -> None:
        chunks, seen = await self._drive(
            lambda _r: httpx.Response(404),
            conversation_id="c",
            resume_token="t",  # noqa: S106
            request_headers={},
        )
        assert chunks == []
        assert len(seen) == 3  # tried all offsets

    async def test_empty_body_not_yielded(self) -> None:
        chunks, _ = await self._drive(
            lambda _r: httpx.Response(200, text="   \n\n"),
            conversation_id="c",
            resume_token="t",  # noqa: S106
            request_headers={},
        )
        assert chunks == []

    async def test_non_404_error_stops_immediately(self) -> None:
        chunks, seen = await self._drive(
            lambda _r: httpx.Response(500),
            conversation_id="c",
            resume_token="t",  # noqa: S106
            request_headers={},
        )
        assert chunks == []
        assert len(seen) == 1  # 500 → give up, no further offsets

    async def test_sentinel_headers_dropped_bearer_kept(self) -> None:
        _, seen = await self._drive(
            lambda _r: httpx.Response(200, text=_CONTENT_SSE),
            conversation_id="c",
            resume_token="tok",  # noqa: S106
            request_headers={
                "authorization": "Bearer keep",
                "openai-sentinel-chat-requirements-token": "DROP",
                "openai-sentinel-proof-token": "DROP",
                "x-openai-target-path": "/backend-api/f/conversation",
            },
        )
        h = seen[0].headers
        assert h["authorization"] == "Bearer keep"
        assert "openai-sentinel-chat-requirements-token" not in h
        assert "openai-sentinel-proof-token" not in h
        assert h["x-conduit-token"] == "tok"
        assert h["x-openai-target-path"] == "/backend-api/f/conversation/resume"
        assert h["accept"] == "text/event-stream"


# ── End-to-end: HTTP SSE + WS frames → intake FSM → text ─────────────────────


class TestEndToEndHandoffToIntake:
    """Feed a full HTTP-handoff-SSE body + WS frames through the intake FSM."""

    async def test_full_handoff_flow_yields_assistant_text(self) -> None:
        topic = "conversation-turn-e2e"

        # 1. Build the HTTP SSE body: add frame + handoff marker.
        add_frame = json.dumps(
            {
                "p": "",
                "o": "add",
                "v": {
                    "message": {
                        "id": "msg-e2e-1",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": [""]},
                        "status": "in_progress",
                        "metadata": {"model_slug": "gpt-5"},
                    },
                    "conversation_id": "conv-e2e-1",
                },
                "c": 0,
            }
        )
        http_body = f"data: {add_frame}\n\n".encode() + _stream_handoff_frame(topic)

        # 2. The WS server sends the actual text content via SSE patches.

        async def _handler(ws: ServerConnection) -> None:
            await ws.recv()
            await ws.recv()
            await ws.send(
                json.dumps(
                    [
                        {
                            "type": "message",
                            "topic_id": topic,
                            "payload": {
                                "payload": {
                                    "encoded_item": (
                                        # An SSE item as produced by ChatGPT: a content-parts append.
                                        "data: "
                                        + json.dumps(
                                            {
                                                "p": "/message/content/parts/0",
                                                "o": "append",
                                                "v": "Hello from WS",
                                                "c": 0,
                                            }
                                        )
                                        + "\n\n"
                                    )
                                }
                            },
                        }
                    ]
                )
            )
            # Status patch → finish.
            await ws.send(
                json.dumps(
                    [
                        {
                            "type": "message",
                            "topic_id": topic,
                            "payload": {
                                "payload": {
                                    "encoded_item": (
                                        "data: "
                                        + json.dumps(
                                            {
                                                "p": "/message/status",
                                                "o": "replace",
                                                "v": "finished_successfully",
                                                "c": 0,
                                            }
                                        )
                                        + "\n\n"
                                    )
                                }
                            },
                        }
                    ]
                )
            )
            await ws.send(json.dumps([_make_done_frame(topic)]))

        async with ws_serve(_handler, "127.0.0.1", 0, ping_interval=None) as server:
            port = server.sockets[0].getsockname()[1]
            ws_url = f"ws://127.0.0.1:{port}"

            # Collect WS SSE bytes.
            ws_chunks: list[bytes] = await _collect_ws_sse(ws_url=ws_url, topic_id=topic)

        # 3. Feed HTTP body + WS continuation through the intake FSM.
        # The HTTP handoff side event is consumed silently (continuation metadata
        # set, no IR event, no raise); the WS chunks carry the real answer.
        fsm = _make_fsm()
        await fsm.feed(http_body)

        # Feed WS continuation bytes.
        for chunk in ws_chunks:
            await fsm.feed(chunk)
        await fsm.close()

        # 4. Assert we got the text from WS.
        parts = list(fsm.parts_manager.get_parts())
        text = "".join(p.content for p in parts if isinstance(p, TextPart))
        assert "Hello from WS" in text


# ── Sidecar continuation seam ─────────────────────────────────────────────────


class _AsyncChunkedStream(httpx.AsyncByteStream):
    """AsyncByteStream that yields pre-set chunks (mirrors test_transport_sidecar)."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class TestSidecarContinuationSeam:
    """body_stream() yields WS continuation bytes after HTTP body ends."""

    async def test_continuation_header_stripped_before_upstream(self) -> None:
        """The CONTINUATION_HEADER must not be forwarded to the upstream."""
        received_headers: list[dict[str, str]] = []

        class _RecordingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                received_headers.append({k.lower(): v for k, v in request.headers.items()})
                return httpx.Response(
                    200,
                    stream=_AsyncChunkedStream([b"ok"]),
                )

        mock_client = httpx.AsyncClient(transport=_RecordingTransport())
        sidecar = Sidecar()
        with patch("ccproxy.transport.sidecar.transport") as m:
            m.get_client = AsyncMock(return_value=mock_client)
            m.UnknownFingerprintProfileError = UnknownFingerprintProfileError
            m.resolve_captured_fingerprint = MagicMock(return_value=None)
            await sidecar.start()
            try:
                async with (
                    httpx.AsyncClient() as client,
                    client.stream(
                        "POST",
                        f"http://127.0.0.1:{sidecar.port}/v1/messages",
                        headers={
                            "x-ccproxy-target-url": "https://chatgpt.com/backend-api/f/conversation",
                            "x-ccproxy-impersonate": "chrome136",
                            CONTINUATION_HEADER: "openai_conversations",
                        },
                        content=b"{}",
                    ) as resp,
                ):
                    await resp.aread()
            finally:
                await sidecar.stop()
                await mock_client.aclose()

        assert len(received_headers) == 1
        assert CONTINUATION_HEADER not in received_headers[0]

    async def test_no_continuation_noop_for_non_oaic_providers(self) -> None:
        """Without CONTINUATION_HEADER the sidecar is byte-for-byte unchanged."""

        class _OkTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    stream=_AsyncChunkedStream([b"plain-body"]),
                )

        mock_client = httpx.AsyncClient(transport=_OkTransport())
        sidecar = Sidecar()
        with patch("ccproxy.transport.sidecar.transport") as m:
            m.get_client = AsyncMock(return_value=mock_client)
            m.UnknownFingerprintProfileError = UnknownFingerprintProfileError
            m.resolve_captured_fingerprint = MagicMock(return_value=None)
            await sidecar.start()
            try:
                async with (
                    httpx.AsyncClient() as client,
                    client.stream(
                        "POST",
                        f"http://127.0.0.1:{sidecar.port}/v1/messages",
                        headers={
                            "x-ccproxy-target-url": "https://api.anthropic.com/v1/messages",
                            "x-ccproxy-impersonate": "chrome131",
                            # No CONTINUATION_HEADER.
                        },
                        content=b"{}",
                    ) as resp,
                ):
                    body = await resp.aread()
            finally:
                await sidecar.stop()
                await mock_client.aclose()

        assert body == b"plain-body"

    async def test_continuation_bridges_after_http_body(self) -> None:
        """When CONTINUATION_HEADER is set and a handoff topic is found, the
        sidecar calls the continuation factory and appends its output."""
        topic = "conversation-turn-sidecar-test"
        ws_chunk = b'data: {"delta": "WS answer"}\n\n'

        class _HandoffBodyTransport(httpx.AsyncBaseTransport):
            """Returns HTTP body ending with a stream_handoff marker."""

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                payload = json.dumps(
                    {
                        "type": "stream_handoff",
                        "options": [{"type": "subscribe_ws_topic", "topic_id": topic}],
                    }
                )
                body = f"data: {payload}\n\n".encode()
                return httpx.Response(
                    200,
                    stream=_AsyncChunkedStream([body]),
                )

        mock_client = httpx.AsyncClient(transport=_HandoffBodyTransport())

        async def _fake_bridge(*, client: object, topic_id: str) -> AsyncIterator[bytes]:
            assert topic_id == topic
            yield ws_chunk

        sidecar = Sidecar()
        with (
            patch("ccproxy.transport.sidecar.transport") as m,
            patch("ccproxy.transport.sidecar.run_handoff_bridge", _fake_bridge),
        ):
            m.get_client = AsyncMock(return_value=mock_client)
            m.UnknownFingerprintProfileError = UnknownFingerprintProfileError
            m.resolve_captured_fingerprint = MagicMock(return_value=None)
            await sidecar.start()
            try:
                received = bytearray()
                async with (
                    httpx.AsyncClient() as client,
                    client.stream(
                        "POST",
                        f"http://127.0.0.1:{sidecar.port}/backend-api/f/conversation",
                        headers={
                            "x-ccproxy-target-url": "https://chatgpt.com/backend-api/f/conversation",
                            "x-ccproxy-impersonate": "chrome136",
                            CONTINUATION_HEADER: "openai_conversations",
                        },
                        content=b"{}",
                    ) as resp,
                ):
                    async for chunk in resp.aiter_bytes():
                        received.extend(chunk)
            finally:
                await sidecar.stop()
                await mock_client.aclose()

        assert ws_chunk in bytes(received)


# ── TransportOverrideAddon stamps CONTINUATION_HEADER ─────────────────────────


class TestTransportOverrideAddonContinuationHeader:
    """TransportOverrideAddon stamps X-CCProxy-Continuation for openai_conversations."""

    async def test_openai_conversations_provider_stamps_continuation_header(self) -> None:
        provider = Provider(
            host="chatgpt.com",
            type="openai_conversations",
            fingerprint_profile="chrome136",
        )
        cfg = CCProxyConfig(providers={"oaic": provider})
        set_config_instance(cfg)

        flow = MagicMock()
        flow.id = "flow-oaic"
        flow.metadata = {"ccproxy.auth_provider": "oaic"}
        flow.request.pretty_url = "https://chatgpt.com/backend-api/f/conversation"
        flow.request.headers = {}

        addon = TransportOverrideAddon(sidecar_port=19300)
        await addon.request(flow)

        assert flow.request.headers.get(CONTINUATION_HEADER) == "openai_conversations"

    async def test_non_openai_conversations_provider_no_continuation_header(self) -> None:
        provider = Provider(
            host="api.anthropic.com",
            type="anthropic",
            fingerprint_profile="chrome131",
        )
        cfg = CCProxyConfig(providers={"anthropic": provider})
        set_config_instance(cfg)

        flow = MagicMock()
        flow.id = "flow-anth"
        flow.metadata = {"ccproxy.auth_provider": "anthropic"}
        flow.request.pretty_url = "https://api.anthropic.com/v1/messages"
        flow.request.headers = {}

        addon = TransportOverrideAddon(sidecar_port=19300)
        await addon.request(flow)

        assert CONTINUATION_HEADER not in flow.request.headers


# ── expires_at unit guard ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExpiresAtCase:
    name: str
    """Scenario identifier."""

    exp_seconds: int
    """``exp`` claim (unix seconds) encoded into the finalize JWT."""

    expected_ms: int
    """Expected ``chat_req_token_expires_at_ms`` stored after decoding."""


_EXPIRES_AT_CASES: list[ExpiresAtCase] = [
    ExpiresAtCase(
        name="jwt_exp_seconds_to_ms",
        exp_seconds=1_750_000_000,
        expected_ms=1_750_000_000 * 1000,
    ),
    ExpiresAtCase(
        name="later_jwt_exp_seconds_to_ms",
        exp_seconds=1_900_000_000,
        expected_ms=1_900_000_000 * 1000,
    ),
]


def _jwt_with_exp(exp_seconds: int) -> str:
    """Build a minimal unsigned JWT carrying the given exp claim."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp_seconds}).encode()).decode().rstrip("=")
    return f"{header}.{payload}.sig"


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.name) for c in _EXPIRES_AT_CASES],
)
async def test_sentinel_expires_decoded_from_finalize_jwt(case: ExpiresAtCase) -> None:
    """_refresh_sentinel decodes chat_req_token_expires_at_ms from the finalize JWT exp."""
    stored: dict[str, object] = {}
    finalize_token = _jwt_with_exp(case.exp_seconds)

    async def _fake_post(url: str, **kwargs: object) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        if url.endswith("/finalize"):
            resp.json = MagicMock(return_value={"token": finalize_token, "persona": "chatgpt-paid"})
        else:
            resp.json = MagicMock(
                return_value={"prepare_token": "prep", "proofofwork": {"required": False}, "persona": "chatgpt-paid"}
            )
        return resp

    def _fake_update(path: str, **kwargs: object) -> None:
        stored.update(kwargs)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = _fake_post

    with (
        patch("ccproxy.inspector.openai_conversations_addon.update_sentinel_fields", _fake_update),
        patch("ccproxy.inspector.openai_conversations_addon.build_requirements_token", return_value="gAAAAACp"),
    ):
        result = await _refresh_sentinel(
            client=mock_client,
            credential_path="/tmp/fake-creds.json",  # noqa: S108
            access_token="bearer-jwt",  # noqa: S106
            device_id="device-1",
            timeout=5.0,
        )

    assert stored["chat_req_token_expires_at_ms"] == case.expected_ms
    assert stored["chat_req_token"] == finalize_token
    assert result.expires_at_ms == case.expected_ms
    assert result.persona == "chatgpt-paid"
