"""Handoff/resume continuation bridges for OpenAI Conversations.

When ChatGPT does not finish a turn inline, it ends the HTTP SSE early with one
of two continuation signals — exactly as the chatgpt.com SPA handles them:

* ``stream_handoff`` (carrying a ``subscribe_ws_topic``) → **WebSocket** handoff:
  ``GET /backend-api/celsius/ws/user`` → ``wss`` → subscribe to the topic
  (``run_handoff_bridge``). The SPA's ``celsius/ws/user`` path.
* ``resume_conversation_token`` (carrying a conduit JWT + conversation id) →
  **HTTP resume**: ``POST /backend-api/f/conversation/resume`` with
  ``x-conduit-token: <token>`` over offsets 0/1/2 (``run_resume_bridge``). The
  SPA's ``/f/conversation/resume`` path.

:class:`HandoffState` scans the streaming SSE bytes (``detect_handoff``) and
exposes :meth:`HandoffState.should_bridge` / :meth:`HandoffState.should_resume`
so the sidecar picks the right continuation; both yield SSE-v1 bytes that the
same intake FSM parses. The WS path wins when both signals appear.

MIT attribution: WS topic extraction + frame parsing adapted from aurora
(``internal/chatgpt/request.go``); the HTTP-resume offset/``x-conduit-token``
flow mirrors the chatgpt.com SPA (``sdk.js``) and pro-cli
(``src/transport.ts`` ``resumeHandoffStream``). All state-machine structure and
Python idioms are original.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from websockets.asyncio.client import connect as ws_connect

logger = logging.getLogger(__name__)

_WS_USER_PATH = "/backend-api/celsius/ws/user"
_WS_CHATGPT_ORIGIN = "https://chatgpt.com"
_WS_BASE_URL = "https://chatgpt.com"

# Only ``stream_handoff`` (with a ``subscribe_ws_topic`` option) is a real WS
# handoff directive. The SPA (``sdk.js``) treats ``server_ste_metadata`` as pure
# telemetry (``addServerSteMetadata``) — it carries a ``turn_exchange_id`` on
# every turn and is NOT a handoff topic; the actual continuation for those turns
# is ``resume_conversation_token`` (HTTP resume).
_HANDOFF_SIDE_EVENTS = frozenset({"stream_handoff"})

# Read timeout on the WS (aurora uses 120s).
_WS_READ_TIMEOUT = 120.0
# Give up if no answer item arrives on the handoff topic within this window, so a
# topic that never streams doesn't hold the client for the full read timeout.
_WS_FIRST_ITEM_TIMEOUT = 30.0
# Ping interval (aurora: 25s).
_WS_PING_INTERVAL = 25.0
# Browser-shape User-Agent string for the WS dial and /celsius/ws/user GET.
_WS_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"


# ── Handoff state ─────────────────────────────────────────────────────────────


@dataclass
class HandoffState:
    """Mutable accumulator for scanning SSE bytes for a handoff signal.

    Created once per request; ``feed(chunk)`` is called for each upstream SSE
    chunk; ``topic`` holds the WS topic id (empty string = no handoff detected
    yet); ``http_content_seen`` tracks whether any text-content patch arrived
    via HTTP so the caller can apply the ``shouldUseWebsocketHandoff`` rule.
    """

    topic: str = ""
    """WS topic id from a ``stream_handoff`` or ``server_ste_metadata`` event."""

    resume_token: str = ""
    """Conduit resume credential from a ``resume_conversation_token`` event; sent
    as ``x-conduit-token`` on ``POST /f/conversation/resume``."""

    conversation_id: str = ""
    """Conversation id for the HTTP-resume body (from the resume / handoff /
    ordinary message events, all of which carry it)."""

    http_content_seen: bool = False
    """True once an ``append``/``replace`` on ``/message/content/parts/0`` arrives."""

    _buf: bytearray = field(default_factory=bytearray)

    def feed(self, chunk: bytes) -> None:
        """Scan ``chunk`` for handoff signals and content patches."""
        self._buf.extend(chunk)
        self._drain()

    def _drain(self) -> None:
        while True:
            crlf = self._buf.find(b"\r\n\r\n")
            lf = self._buf.find(b"\n\n")
            if crlf == -1 and lf == -1:
                return
            if crlf != -1 and (lf == -1 or crlf < lf):
                sep_idx, sep_len = crlf, 4
            else:
                sep_idx, sep_len = lf, 2
            frame = bytes(self._buf[:sep_idx])
            del self._buf[: sep_idx + sep_len]
            self._process_frame(frame)

    def _process_frame(self, frame: bytes) -> None:
        data_lines: list[str] = []
        for raw in frame.split(b"\n"):
            line = raw.rstrip(b"\r").decode("utf-8", errors="replace")
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

        data = "\n".join(data_lines).strip()
        if not data or data == "[DONE]":
            return

        try:
            parsed = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(parsed, dict):
            return

        kind = parsed.get("type")
        if isinstance(kind, str):
            if kind == "resume_conversation_token":
                tok = parsed.get("token")
                if isinstance(tok, str) and tok:
                    self.resume_token = tok
                    # The JWT carries turn_topic_id (conversation-turn-<id>) — the
                    # WS topic for the conduit stream. Prefer it as the topic source.
                    if not self.topic:
                        jwt_topic = _topic_from_resume_jwt(tok)
                        if jwt_topic:
                            self.topic = jwt_topic
            elif kind in _HANDOFF_SIDE_EVENTS and not self.topic:
                topic = _extract_handoff_topic(parsed)
                if topic:
                    self.topic = topic

        # conversation_id rides on the resume / handoff / message events alike;
        # capture the first non-empty one for the HTTP-resume body.
        cid = parsed.get("conversation_id")
        if isinstance(cid, str) and cid and not self.conversation_id:
            self.conversation_id = cid

        # Detect HTTP text content (aurora ``shouldUseWebsocketHandoff`` rule).
        # Check ``o`` field for "append"/"replace" on the text path.
        op = parsed.get("o")
        if op in ("append", "replace") and parsed.get("p") == "/message/content/parts/0":
            if isinstance(parsed.get("v"), str) and parsed.get("v"):
                self.http_content_seen = True
            return

        # Also check batch patch arrays.
        if op == "patch" and isinstance(parsed.get("v"), list):
            for item in parsed["v"]:
                if not isinstance(item, dict):
                    continue
                if (
                    item.get("p") == "/message/content/parts/0"
                    and item.get("o") in ("append", "replace")
                    and isinstance(item.get("v"), str)
                    and item.get("v")
                ):
                    self.http_content_seen = True

    def should_bridge(self) -> bool:
        """True when a ``stream_handoff`` WS topic was found and no HTTP text arrived.

        The WebSocket path (``stream_handoff`` → ``/celsius/ws/user`` → wss) is the
        SPA's handler for that signal. A bare ``resume_conversation_token`` carries
        no WS topic and routes to :meth:`should_resume` instead.
        """
        return bool(self.topic) and not self.http_content_seen

    def should_resume(self) -> bool:
        """True when a ``resume_conversation_token`` (with conversation id) arrived,
        no inline content was seen, and no WS topic took precedence.

        Routes to the SPA's HTTP-resume path: ``POST /f/conversation/resume`` with
        ``x-conduit-token``. When both signals are present, the WS handoff wins
        (``should_bridge`` is checked first).
        """
        return bool(self.resume_token) and bool(self.conversation_id) and not self.http_content_seen and not self.topic

    def should_continue(self) -> bool:
        """True when either continuation path (WS bridge or HTTP resume) applies."""
        return self.should_bridge() or self.should_resume()


def detect_handoff(state: HandoffState, chunk: bytes) -> None:
    """Feed ``chunk`` through ``state``, accumulating handoff signals in-place."""
    state.feed(chunk)


def _topic_from_resume_jwt(token: str) -> str:
    """Decode the ``resume_conversation_token`` JWT and return its ``turn_topic_id``
    claim — the ``conversation-turn-<id>`` WebSocket topic carrying the conduit
    answer stream. Empty string on any decode failure or missing claim.

    The JWT payload also carries ``conduit_uuid`` / ``conduit_location`` (an
    internal IP, not directly reachable) / ``cluster``; the public route to the
    stream is ``/celsius/ws/user`` → wss, subscribed to ``turn_topic_id``.
    """
    parts = token.split(".")
    if len(parts) < 2 or not parts[1]:
        return ""
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(claims, dict):
        return ""
    topic = claims.get("turn_topic_id")
    return topic if isinstance(topic, str) else ""


def _extract_handoff_topic(raw: dict[str, Any]) -> str:
    """Extract the WS topic id from a ``stream_handoff`` event.

    The SPA (``sdk.js``) only treats a ``stream_handoff`` carrying a
    ``subscribe_ws_topic`` option as a WebSocket handoff (``stream_protocol =
    ws``); ``server_ste_metadata`` is telemetry, not a topic source.
    """
    options = raw.get("options")
    if isinstance(options, list):
        for option in options:
            if not isinstance(option, dict):
                continue
            if option.get("type") == "subscribe_ws_topic":
                topic = option.get("topic_id", "")
                if isinstance(topic, str) and topic:
                    return topic
    return ""


# ── WS URL fetch ──────────────────────────────────────────────────────────────


_WS_GET_DROP_HEADERS = frozenset(
    {
        "x-openai-target-path",
        "x-openai-target-route",
        "x-conduit-token",
        "content-length",
        "content-type",
        "accept",
        "host",
    }
)
"""Original /f/conversation headers the WS-url GET replaces (everything else —
bearer, browser shape, sentinel tokens — is reused so the GET is authenticated)."""


def _ws_get_headers(request_headers: dict[str, str]) -> dict[str, str]:
    """Build the ``/celsius/ws/user`` GET headers by reusing the original
    browser-shape + bearer + sentinel headers (aurora sends the full
    conversation header set; the minimal 3-header GET 403s)."""
    headers = {k: v for k, v in request_headers.items() if k.lower() not in _WS_GET_DROP_HEADERS}
    headers["accept"] = "*/*"
    headers["x-openai-target-path"] = _WS_USER_PATH
    headers["x-openai-target-route"] = _WS_USER_PATH
    return headers


async def fetch_ws_url(
    client: httpx.AsyncClient,
    *,
    request_headers: dict[str, str] | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> str:
    """GET /backend-api/celsius/ws/user and return the ``websocket_url`` value.

    Mirrors aurora ``getChatWebsocketURL`` (request.go:632-659, MIT-licensed),
    which sends the full conversation header set (bearer + sentinel + browser);
    a minimal-header GET is rejected with HTTP 403.

    Args:
        client: Cached httpx.AsyncClient (curl-cffi backed, authenticated).
        request_headers: The original forwarded ``/f/conversation`` headers,
            reused for the bearer + browser + sentinel block.
        extra_headers: Optional additional headers to stamp on the GET.
        timeout: Request timeout in seconds.

    Returns:
        Authenticated ``wss://`` URL.

    Raises:
        RuntimeError: On non-200 response or missing ``websocket_url`` field.
    """
    url = f"{_WS_BASE_URL}{_WS_USER_PATH}"
    headers = _ws_get_headers(request_headers or {})
    if extra_headers:
        headers.update(extra_headers)

    resp = await client.get(url=url, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"celsius ws/user failed: HTTP {resp.status_code} — {resp.text[:200]}")
    try:
        result = resp.json()
    except Exception as exc:
        raise RuntimeError(f"celsius ws/user: invalid JSON: {exc}") from exc

    ws_url = result.get("websocket_url")
    if not isinstance(ws_url, str) or not ws_url:
        raise RuntimeError(f"celsius ws/user: missing websocket_url in {result!r}")
    return ws_url


# ── WS frame helpers ──────────────────────────────────────────────────────────


def _parse_ws_frames(raw: bytes) -> list[dict[str, Any]]:
    """Parse one WebSocket message into a list of frame objects.

    A message is either a JSON array of frames or a single frame object.
    Mirrors aurora ``parseChatWebsocketFrames`` (request.go:721-737,
    MIT-licensed).
    """
    if not raw:
        return []
    try:
        if raw[0:1] == b"[":
            frames: Any = json.loads(raw)
            if isinstance(frames, list):
                return [f for f in frames if isinstance(f, dict)]
            return []
        frame: Any = json.loads(raw)
        if isinstance(frame, dict):
            return [frame]
        return []
    except (json.JSONDecodeError, ValueError):
        return []


def _sse_items_from_frame(frame: dict[str, Any], topic_id: str) -> list[str]:
    """Extract SSE data strings from one WS frame.

    Mirrors aurora ``chatWebsocketSSEItems`` (request.go:758-798, MIT-licensed):
    tries ``chatWebsocketEncodedItem`` first, then
    ``chatWebsocketConversationUpdateItem``.
    """
    items: list[str] = []
    encoded = _encoded_item(frame, topic_id)
    if encoded:
        items.append(encoded)
        return items
    update = _conversation_update_item(frame, topic_id)
    if update:
        items.append(update)
    return items


def _encoded_item(frame: dict[str, Any], topic_id: str) -> str:
    """Extract ``payload.payload.encoded_item`` from a frame matching ``topic_id``.

    Mirrors aurora ``chatWebsocketEncodedItem`` (request.go:739-756,
    MIT-licensed).
    """
    frame_topic = frame.get("topic_id")
    if isinstance(frame_topic, str) and frame_topic and frame_topic != topic_id:
        return ""
    payload = frame.get("payload")
    if not isinstance(payload, dict):
        return ""
    nested = payload.get("payload")
    if not isinstance(nested, dict):
        return ""
    encoded = nested.get("encoded_item")
    if not isinstance(encoded, str) or not encoded:
        return ""
    return encoded


def _conversation_update_item(frame: dict[str, Any], topic_id: str) -> str:
    """Extract a ``conversation-update`` payload as an SSE data line.

    Mirrors aurora ``chatWebsocketConversationUpdateItem``
    (request.go:768-797, MIT-licensed).
    """
    frame_topic = frame.get("topic_id")
    if isinstance(frame_topic, str) and frame_topic and frame_topic != topic_id and frame_topic != "conversations":
        return ""
    payload = frame.get("payload")
    if not isinstance(payload, dict):
        return ""

    # Unwrap nested payload if needed.
    if payload.get("type") != "conversation-update":
        nested = payload.get("payload")
        if isinstance(nested, dict) and nested.get("type") == "conversation-update":
            payload = nested

    if payload.get("type") != "conversation-update":
        return ""

    try:
        body = json.dumps(payload)
    except (TypeError, ValueError):
        return ""
    return "data: " + body + "\n"


def _is_done_item(item: str) -> bool:
    """True when ``item`` contains the ``[DONE]`` sentinel.

    Mirrors aurora ``chatWebsocketWriteEncodedItem`` done check
    (request.go:807, MIT-licensed).
    """
    return "data: [DONE]" in item or "data:[DONE]" in item


# ── Reply-type frame processing ───────────────────────────────────────────────


def _process_reply_frame(frame: dict[str, Any], topic_id: str) -> tuple[list[str], bool]:
    """Process a ``type==reply`` frame, yielding SSE items and done signal.

    A reply frame carries ``reply.topic_id`` and ``reply.catchups[]``.
    Only replies whose ``topic_id`` matches are processed.
    Mirrors aurora ``chatWebsocketStreamReader`` reply branch
    (request.go:841-857, MIT-licensed).

    Returns:
        (items, done) — list of SSE data strings and whether stream is terminal.
    """
    reply = frame.get("reply")
    if not isinstance(reply, dict):
        return [], False
    reply_topic = reply.get("topic_id")
    if reply_topic != topic_id:
        return [], False
    catchups = reply.get("catchups")
    if not isinstance(catchups, list):
        return [], False
    all_items: list[str] = []
    for catchup in catchups:
        if not isinstance(catchup, dict):
            continue
        for item in _sse_items_from_frame(catchup, topic_id):
            all_items.append(item)
            if _is_done_item(item):
                return all_items, True
    return all_items, False


# ── WS SSE streaming ─────────────────────────────────────────────────────────


async def stream_handoff_sse(
    *,
    ws_url: str,
    topic_id: str,
    user_agent: str = _WS_USER_AGENT,
    read_timeout: float = _WS_READ_TIMEOUT,
    ping_interval: float = _WS_PING_INTERVAL,
) -> AsyncIterator[bytes]:
    """Dial the WS, subscribe to ``topic_id``, and yield SSE bytes.

    The wss URL is already auth-bearing (from ``/celsius/ws/user``); no TLS
    fingerprint impersonation is needed — a plain ``websockets`` dial suffices.

    Each yielded ``bytes`` value is a complete SSE line of the form
    ``b"data: {...}\\n\\n"``.  The generator stops cleanly on ``[DONE]`` or any
    error; it never raises into the caller.

    Mirrors aurora ``chatWebsocketStreamReader`` (request.go:810-884,
    MIT-licensed) and ``DialChatWebsocket`` (request.go:661-708).
    """
    return _stream_handoff_sse_impl(
        ws_url=ws_url,
        topic_id=topic_id,
        user_agent=user_agent,
        read_timeout=read_timeout,
        ping_interval=ping_interval,
    )


async def _stream_handoff_sse_impl(
    *,
    ws_url: str,
    topic_id: str,
    user_agent: str,
    read_timeout: float,
    ping_interval: float,
) -> AsyncIterator[bytes]:
    # Aurora init array: connect + subscribe three known topics.
    init_msg = json.dumps(
        [
            {"id": 1, "command": {"type": "connect", "presence": {"type": "presence", "state": "background"}}},
            {"id": 2, "command": {"type": "subscribe", "topic_id": "calpico-chatgpt"}},
            {"id": 3, "command": {"type": "subscribe", "topic_id": "conversations"}},
            {"id": 4, "command": {"type": "subscribe", "topic_id": "app_notifications"}},
        ]
    )

    # Unique subscribe id — aurora uses an atomic counter starting at 5.
    sub_id = 5
    sub_msg = json.dumps([{"id": sub_id, "command": {"type": "subscribe", "topic_id": topic_id, "offset": "0"}}])

    try:
        async with ws_connect(
            ws_url,
            additional_headers={
                "User-Agent": user_agent,
                "Origin": _WS_CHATGPT_ORIGIN,
            },
            # Disable built-in ping; we handle it ourselves to mirror aurora.
            ping_interval=None,
            open_timeout=15.0,
        ) as ws:
            # Send init array (mirrors aurora DialChatWebsocketWithState).
            await ws.send(init_msg)
            # Subscribe to the handoff topic.
            await ws.send(sub_msg)

            # Ping task: every ping_interval seconds send a ping.
            async def _ping_loop() -> None:
                while True:
                    await asyncio.sleep(ping_interval)
                    try:
                        await ws.ping()
                    except Exception:
                        return

            ping_task = asyncio.create_task(_ping_loop(), name="ccproxy-ws-handoff-ping")
            logger.debug("ws_handoff: dialed wss, subscribed topic=%s", topic_id)
            yielded = 0
            # If no answer item arrives on the topic within this window, give up
            # rather than hold the client for the full per-message read timeout.
            first_item_deadline = asyncio.get_running_loop().time() + _WS_FIRST_ITEM_TIMEOUT

            try:
                async for raw_msg in _ws_read_loop(ws=ws, read_timeout=read_timeout):
                    frames = _parse_ws_frames(raw_msg if isinstance(raw_msg, bytes) else raw_msg.encode())
                    logger.debug(
                        "ws_handoff: WS msg frames=%s topics=%s",
                        [f.get("type") for f in frames],
                        [f.get("topic_id") or (f.get("reply") or {}).get("topic_id") for f in frames],
                    )
                    done = False
                    for frame in frames:
                        frame_type = frame.get("type")
                        if frame_type == "reply":
                            items, done = _process_reply_frame(frame, topic_id)
                        elif frame_type == "message":
                            items = _sse_items_from_frame(frame, topic_id)
                            done = any(_is_done_item(i) for i in items)
                        else:
                            continue

                        yielded += len(items)
                        for item in items:
                            if not item.endswith("\n"):
                                item += "\n"
                            # Ensure the SSE data line ends with the double newline
                            # separator expected by the intake FSM.
                            if not item.endswith("\n\n"):
                                item += "\n"
                            yield item.encode()
                        if done:
                            logger.debug("ws_handoff: WS done, yielded=%d items (topic=%s)", yielded, topic_id)
                            return
                    if yielded == 0 and asyncio.get_running_loop().time() > first_item_deadline:
                        logger.warning(
                            "ws_handoff: no answer on topic=%s within %.0fs; giving up",
                            topic_id,
                            _WS_FIRST_ITEM_TIMEOUT,
                        )
                        return
            finally:
                ping_task.cancel()
                logger.debug("ws_handoff: WS closed, yielded=%d items (topic=%s)", yielded, topic_id)
    except Exception as exc:
        logger.error("ws_handoff: WS bridge error (topic=%s): %s", topic_id, exc)


async def _ws_read_loop(
    *,
    ws: Any,
    read_timeout: float,
) -> AsyncIterator[str | bytes]:
    """Yield raw WebSocket messages with per-message read deadline."""
    while True:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=read_timeout)
        except TimeoutError:
            logger.warning("ws_handoff: read timeout after %.0fs", read_timeout)
            return
        except Exception as exc:
            logger.debug("ws_handoff: WS closed: %s", exc)
            return
        yield msg


# ── Public bridge entry point ─────────────────────────────────────────────────


async def run_handoff_bridge(
    *,
    client: httpx.AsyncClient,
    topic_id: str,
    request_headers: dict[str, str] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> AsyncIterator[bytes]:
    """Fetch WS URL, dial, stream topic, yield SSE bytes.

    On ANY error, logs and returns immediately — never raises into the
    ``body_stream`` caller.

    Args:
        client: Authenticated httpx.AsyncClient for the ``/celsius/ws/user`` GET.
        topic_id: WS topic id (from a ``stream_handoff`` option or the
            ``resume_conversation_token`` JWT's ``turn_topic_id``).
        request_headers: The original forwarded ``/f/conversation`` headers, reused
            so the ``/celsius/ws/user`` GET is authenticated (bearer + sentinel +
            browser); a minimal-header GET 403s.
        extra_headers: Optional additional headers for the ``/celsius/ws/user`` GET.
    """
    try:
        ws_url = await fetch_ws_url(client=client, request_headers=request_headers, extra_headers=extra_headers)
    except Exception as exc:
        logger.error("ws_handoff: failed to fetch WS URL (topic=%s): %s", topic_id, exc)
        return

    logger.debug("ws_handoff: bridging topic=%s via %s…", topic_id, ws_url[:60])

    try:
        async for chunk in _stream_handoff_sse_impl(
            ws_url=ws_url,
            topic_id=topic_id,
            user_agent=_WS_USER_AGENT,
            read_timeout=_WS_READ_TIMEOUT,
            ping_interval=_WS_PING_INTERVAL,
        ):
            yield chunk
    except Exception as exc:
        logger.error("ws_handoff: bridge stream error (topic=%s): %s", topic_id, exc)


# ── HTTP-resume continuation (resume_conversation_token) ──────────────────────

_RESUME_PATH = "/backend-api/f/conversation/resume"
_RESUME_OFFSETS = (0, 1, 2)
"""Offsets the ChatGPT SPA + pro-cli try in order; HTTP 404 means "next offset"."""

_RESUME_MAX_HOPS = 6
"""Max ``resume_conversation_token`` re-handoffs to follow before giving up."""

_RESUME_POLL_DELAY = 0.4
"""Delay between resume re-handoff hops while the conduit answer is generating."""

_RESUME_DROP_HEADERS = frozenset(
    {
        "openai-sentinel-chat-requirements-token",
        "openai-sentinel-proof-token",
        "x-openai-target-path",
        "x-openai-target-route",
        "x-conduit-token",
        "accept",
        "content-type",
        "content-length",
        "host",
    }
)
"""Original /f/conversation headers the resume call replaces or must not carry —
the sentinel chat-requirements tokens are conversation-submit-only."""


def _resume_headers(request_headers: dict[str, str], resume_token: str) -> dict[str, str]:
    """Build ``/f/conversation/resume`` headers by reusing the original browser-shape
    + bearer headers, swapping in the conduit token + resume target path."""
    headers = {k: v for k, v in request_headers.items() if k.lower() not in _RESUME_DROP_HEADERS}
    headers["accept"] = "text/event-stream"
    headers["content-type"] = "application/json"
    headers["x-conduit-token"] = resume_token
    headers["x-openai-target-path"] = _RESUME_PATH
    return headers


_CONTENT_PATCH_MARKER = b"/message/content/parts/0"
"""Substring present iff a resume body carries assistant content patches."""


def _resume_body_has_content(body: bytes) -> bool:
    return _CONTENT_PATCH_MARKER in body


def _read_resume_token(body: bytes) -> tuple[str, str]:
    """Extract ``(token, conversation_id)`` from a ``resume_conversation_token``
    event inside a resume body. Empty strings when absent."""
    for frame in body.split(b"\n\n"):
        for line in frame.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            try:
                obj = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict) and obj.get("type") == "resume_conversation_token":
                tok = obj.get("token")
                cid = obj.get("conversation_id")
                return (tok if isinstance(tok, str) else ""), (cid if isinstance(cid, str) else "")
    return "", ""


async def _resume_post_once(
    *,
    client: httpx.AsyncClient,
    headers: dict[str, str],
    conversation_id: str,
    token: str,
    timeout: float,
) -> bytes | None:
    """One resume POST trying offsets 0/1/2 (404 → next offset). Returns the 200
    body, or ``None`` on a non-200/404 status or transport error."""
    headers = {**headers, "x-conduit-token": token}
    for offset in _RESUME_OFFSETS:
        payload = json.dumps({"conversation_id": conversation_id, "offset": offset}).encode()
        try:
            resp = await client.post(
                url=f"{_WS_BASE_URL}{_RESUME_PATH}",
                headers=headers,
                content=payload,
                timeout=timeout,
            )
        except Exception as exc:
            logger.error("ws_handoff: resume POST failed (offset=%d): %s", offset, exc)
            return None
        if resp.status_code == 404:
            continue
        if resp.status_code != 200:
            logger.warning("ws_handoff: resume offset=%d → HTTP %d; giving up", offset, resp.status_code)
            return None
        try:
            return await resp.aread()
        except Exception as exc:
            logger.error("ws_handoff: resume body read failed (offset=%d): %s", offset, exc)
            return None
    return None


async def run_resume_bridge(
    *,
    client: httpx.AsyncClient,
    conversation_id: str,
    resume_token: str,
    request_headers: dict[str, str] | None = None,
    timeout: float = _WS_READ_TIMEOUT,
) -> AsyncIterator[bytes]:
    """HTTP-resume continuation for a ``resume_conversation_token`` turn.

    ``POST /backend-api/f/conversation/resume`` with ``x-conduit-token: <token>``
    and body ``{conversation_id, offset}`` (offsets 0/1/2, 404 → next), mirroring
    the ChatGPT SPA (``sdk.js`` ``RMn``/``nNn``) + pro-cli ``resumeHandoffStream``.

    The resume endpoint can answer one of two ways: a body carrying assistant
    content patches (the answer — yielded for the intake), or another
    ``resume_conversation_token`` re-pointing at a freshly-issued conduit (the
    answer isn't ready yet). In the latter case this follows the re-handoff chain
    up to :data:`_RESUME_MAX_HOPS` times, polling with a short delay, until content
    appears or the chain stalls. On ANY error this logs and returns — never raises
    into the ``body_stream`` caller.

    Args:
        client: Authenticated curl-cffi-backed AsyncClient (shares the cookie jar
            with the original ``/f/conversation`` request).
        conversation_id: Conversation id from the ``resume_conversation_token`` event.
        resume_token: The conduit resume credential (sent as ``x-conduit-token``).
        request_headers: The original forwarded ``/f/conversation`` headers, reused
            for the browser-shape + bearer block.
        timeout: Per-request timeout in seconds.
    """
    headers = _resume_headers(request_headers or {}, resume_token)
    cid, token = conversation_id, resume_token
    for hop in range(_RESUME_MAX_HOPS):
        body = await _resume_post_once(
            client=client, headers=headers, conversation_id=cid, token=token, timeout=timeout
        )
        if not body or not body.strip():
            return
        if _resume_body_has_content(body):
            logger.debug("ws_handoff: resume recovered content at hop=%d (%dB)", hop, len(body))
            yield body
            return
        next_token, next_cid = _read_resume_token(body)
        if not next_token or next_token == token:
            logger.debug("ws_handoff: resume hop=%d re-handoff carried no new conduit token; stopping", hop)
            return
        token, cid = next_token, (next_cid or cid)
        await asyncio.sleep(_RESUME_POLL_DELAY)
    logger.warning("ws_handoff: resume exhausted %d hops without content", _RESUME_MAX_HOPS)
