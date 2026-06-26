"""OpenAI Conversations SSE-v1 bytes → pydantic-ai IR events via FSM.

Ported from gproxy (MIT licence — original authors: gproxy contributors):
  sdk/gproxy-channel/src/channels/chatgpt/sse_v1.rs
  sdk/gproxy-channel/src/channels/chatgpt/sse_to_openai.rs

The ChatGPT ``/backend-api/f/conversation`` endpoint streams SSE-v1
JSON-patch deltas rather than OpenAI ``chat.completion.chunk`` frames.
This module parses those bytes directly — no dependency on the OpenAI
SDK shape — into pydantic-ai ``TextPart`` start + delta events compatible
with ``OpenAIResponseRenderFSM`` and ``AnthropicResponseRenderFSM``.

Wire shapes decoded:

* ``event: delta_encoding / data: "v1"`` — encoding banner, consumed silently.
* ``data: {type, ...}`` without ``p`` / ``v`` fields — typed side events:
  ``resume_conversation_token``, ``stream_handoff``, ``server_ste_metadata``.
  These are captured into ``state.continuation`` and surface a typed
  :class:`HandoffDetected` signal; they do NOT produce a false finish.
* ``data: {p, o, v, c}`` — initial "add" event declaring a new channel.
  The intake inspects the embedded message to decide whether this is the
  visible assistant final-answer channel.
* ``data: {o: "patch", v: [{p,o,v}, ...]}`` — explicit batch of patches on
  the current channel.
* ``data: {v: [{p,o,v}, ...]}`` (no ``o``/``p``) — shorthand batch, same
  semantics.
* ``data: {p, o, v}`` — single patch on the current channel.
* ``data: [DONE]`` — stream end.

Patch semantics on ``/message/content/parts/0``:

* ``o: "append"`` — emit the full value as a text delta.
* ``o: "replace"`` — compute the suffix not yet accumulated and emit only that.

Finish synthesis:

* ``o: "replace", p: "/message/status", v: "finished_successfully"`` triggers
  a synthetic finish event once content has been emitted.
* ``data: [DONE]`` after content also synthesises a finish, preventing
  duplication via the ``state.final_emitted`` flag.

Handoff detection (detection only — WS connection is CHATGPT-002):

When a typed side event with ``type`` in ``{resume_conversation_token,
stream_handoff, server_ste_metadata}`` arrives **before any visible content
has been accumulated**, ``state.continuation`` is populated and the FSM
surfaces a :class:`_HandoffDetected` envelope. The ``feed()`` method then
raises :class:`HandoffUnsupportedError`. When content is already flowing
these events are informational and intake continues.

MIT attribution:

    The SSE frame decoder (``_drain_sse_frames``), patch shape normalisation
    (``_parse_delta``), and channel-identification logic in ``handle_add`` are
    adapted from the gproxy Rust reference cited above. All behavioural
    decisions, state-machine structure, and Python idioms are original.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_graph import GraphBuilder, StepContext

import ccproxy.lightllm.graph._subgraph_patch  # noqa: F401  — installs add_subgraph

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestParameters

logger = logging.getLogger(__name__)

# ── Public typed signals ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContinuationMetadata:
    """Side-event metadata captured during intake.

    Populated when any of ``resume_conversation_token``, ``stream_handoff``,
    or ``server_ste_metadata`` typed events arrive.
    """

    conversation_id: str = ""
    """Conversation id extracted from the initial add or a side event."""

    resume_token: str = ""
    """Token from a ``resume_conversation_token`` event."""

    handoff_topic: str = ""
    """WS topic from a ``stream_handoff`` or ``server_ste_metadata`` event."""

    turn_exchange_id: str = ""
    """Turn exchange id from a ``server_ste_metadata`` event."""


class HandoffUnsupportedError(RuntimeError):
    """Raised when a handoff side event arrives before content and WS continuation is not yet wired.

    Carries the captured :class:`ContinuationMetadata` for inspection.
    """

    def __init__(self, meta: ContinuationMetadata) -> None:
        self.meta = meta
        super().__init__(
            f"openai_conversations handoff detected (topic={meta.handoff_topic!r}) "
            "but WS continuation is not yet implemented (CHATGPT-002); "
            "cannot continue stream."
        )


# ── Internal dispatch envelopes ───────────────────────────────────────────────


@dataclass(frozen=True)
class _AddEnvelope:
    """Initial channel-add event. Carries channel id and embedded message wrapper."""

    channel: int
    value: dict[str, Any]


@dataclass(frozen=True)
class _PatchEnvelope:
    """One or more patch operations on the current (or declared) channel.

    ``channel`` is ``None`` when the frame omits the ``c`` field; the handler
    uses ``state.current_channel`` in that case.
    """

    channel: int | None
    patches: list[tuple[str, str, Any]]
    """Normalised ``(path, op, value)`` tuples in application order."""


@dataclass(frozen=True)
class _TypedSideEvent:
    """Typed side event — ``{type, ...}`` without ``p`` or ``v`` fields."""

    kind: str
    raw: dict[str, Any]


class _DoneEvent:
    """``[DONE]`` sentinel from the SSE stream."""


class _FeedDone:
    """Queue exhausted — terminal step returns accumulated IR events."""


class _HandoffDetected:
    """Handoff side event arrived before any visible content."""


# ── State ─────────────────────────────────────────────────────────────────────

_ANSWER_VENDOR_ID = "oaicv-answer"
"""Stable vendor_part_id for the assistant final-answer ``TextPart``."""


@dataclass
class _ConversationsIntakeState:
    """FSM state for one OpenAI Conversations SSE-v1 stream.

    ``events_queue`` is populated by ``feed()`` before each graph run.
    ``out_events`` accumulates pydantic-ai IR events; the terminal step
    drains and returns it.

    Channel-tracking fields persist across ``feed()`` calls — they model
    the ongoing server-side JSON-patch state machine.
    """

    parts_manager: ModelResponsePartsManager

    # ── Channel tracking ──────────────────────────────────────────────────────
    current_channel: int | None = None
    """Most-recently-declared channel (applies to follow-up patches missing ``c``)."""

    final_channel: int | None = None
    """Channel carrying the assistant's visible final answer text."""

    message_id: str = ""
    """Assistant message id from the final-answer channel's initial add."""

    model_slug: str = ""
    """Model slug from the final-answer channel's message metadata."""

    conversation_id: str = ""
    """Conversation id from the first channel add."""

    # ── Accumulator ───────────────────────────────────────────────────────────
    accumulated_text: str = ""
    """Full text emitted for the final-answer channel; used for suffix diffing."""

    content_begun: bool = False
    """True once at least one text delta has been emitted."""

    final_emitted: bool = False
    """True once the finish reason has been recorded; prevents duplication."""

    finish_reason: str | None = None
    """Finish reason string (``"stop"``, ``"length"``, etc.) when known."""

    # ── Handoff ───────────────────────────────────────────────────────────────
    continuation: ContinuationMetadata | None = None
    """Populated when a handoff side event is received."""

    # ── Queue / output ────────────────────────────────────────────────────────
    events_queue: deque[Any] = field(default_factory=deque)
    out_events: list[ModelResponseStreamEvent] = field(default_factory=list)


# ── Pure helpers ──────────────────────────────────────────────────────────────


def _is_final_answer_message(msg: dict[str, Any]) -> bool:
    """True when ``msg`` is the visible assistant text message to track.

    Mirrors gproxy ``handle_add`` heuristics:
    - ``author.role == "assistant"``
    - ``content.content_type == "text"``
    - ``status != "finished_successfully"`` (completed messages are historical)
    - ``metadata.is_visually_hidden_from_conversation`` is falsy
    """
    role = msg.get("author", {}).get("role")
    content_type = msg.get("content", {}).get("content_type")
    status = msg.get("status")
    hidden = msg.get("metadata", {}).get("is_visually_hidden_from_conversation")
    return role == "assistant" and content_type == "text" and status != "finished_successfully" and not hidden


def _handoff_topic_from_side_event(kind: str, raw: dict[str, Any]) -> str:
    """Extract a WS handoff topic id from a typed side event.

    Mirrors aurora ``streamHandoffTopicFromEvent`` + ``streamHandoffTopicFromMetadata``.
    """
    if kind == "stream_handoff":
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

    if kind == "server_ste_metadata":
        turn_exchange_id = raw.get("turn_exchange_id")
        if isinstance(turn_exchange_id, str) and turn_exchange_id:
            return f"conversation-turn-{turn_exchange_id}"
        metadata = raw.get("metadata")
        if isinstance(metadata, dict):
            tei = metadata.get("turn_exchange_id")
            if isinstance(tei, str) and tei:
                return f"conversation-turn-{tei}"
        return ""

    return ""


def _parse_delta(frame_obj: dict[str, Any]) -> list[tuple[str, str, Any]]:
    """Normalise a frame object into ``(path, op, value)`` tuples.

    Four shapes (gproxy ``parse_delta``):

    - Explicit batch: ``{o: "patch", v: [{p,o,v}, ...]}``
    - Shorthand batch: ``{v: [{p,o,v}, ...]}`` (no ``o``/``p``)
    - Implicit add: ``{v: <object>}`` (no ``o``/``p``) → ``[("", "add", obj)]``
    - Single patch: ``{p, o, v}``
    """
    op_field = frame_obj.get("o")
    op_str = op_field if isinstance(op_field, str) else ""
    v_field = frame_obj.get("v")
    has_p = "p" in frame_obj

    if op_str == "patch" and isinstance(v_field, list):
        return _parse_patch_list(v_field)

    if not op_str and not has_p and v_field is not None:
        if isinstance(v_field, list):
            return _parse_patch_list(v_field)
        if isinstance(v_field, dict):
            return [("", "add", v_field)]

    path = frame_obj.get("p", "")
    path = path if isinstance(path, str) else ""
    return [(path, op_str, v_field)]


def _parse_patch_list(items: list[Any]) -> list[tuple[str, str, Any]]:
    result: list[tuple[str, str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        path = item.get("p", "")
        path = path if isinstance(path, str) else ""
        op = item.get("o", "")
        op = op if isinstance(op, str) else ""
        value = item.get("v")
        result.append((path, op, value))
    return result


def _channel_from_frame(frame_obj: dict[str, Any]) -> int | None:
    c = frame_obj.get("c")
    if c is None:
        return None
    try:
        return int(c)
    except (TypeError, ValueError):
        return None


# ── Outer intake graph ────────────────────────────────────────────────────────


_g: GraphBuilder[_ConversationsIntakeState, None, None, list[ModelResponseStreamEvent]] = GraphBuilder(
    name="openai_conversations_intake",
    state_type=_ConversationsIntakeState,
    output_type=list[ModelResponseStreamEvent],
)


@_g.step
async def frame_next_event(
    ctx: StepContext[_ConversationsIntakeState, None, None],
) -> Any:
    """Pop the next dispatch envelope from the events queue, or signal completion."""
    state = ctx.state
    if not state.events_queue:
        return _FeedDone()
    return state.events_queue.popleft()


@_g.step
async def handle_add(
    ctx: StepContext[_ConversationsIntakeState, None, _AddEnvelope],
) -> None:
    """Process an initial channel-add event.

    Inspects the embedded message to determine whether this channel carries
    the visible assistant final-answer text. Captures ``conversation_id`` and
    ``model_slug`` on first observation.
    """
    state = ctx.state
    env = ctx.inputs
    state.current_channel = env.channel

    msg_wrap = env.value
    conv_id = msg_wrap.get("conversation_id")
    if isinstance(conv_id, str) and conv_id and not state.conversation_id:
        state.conversation_id = conv_id

    msg = msg_wrap.get("message")
    if not isinstance(msg, dict):
        return

    if _is_final_answer_message(msg) and state.final_channel is None:
        state.final_channel = env.channel
        msg_id = msg.get("id")
        if isinstance(msg_id, str):
            state.message_id = msg_id
        slug = msg.get("metadata", {}).get("model_slug")
        if isinstance(slug, str):
            state.model_slug = slug


@_g.step
async def handle_patch(
    ctx: StepContext[_ConversationsIntakeState, None, _PatchEnvelope],
) -> None:
    """Apply patches, emitting IR text delta events for the final-answer channel.

    Updates ``state.current_channel`` when the envelope declares one.
    Only patches targeting the final-answer channel produce output:

    - ``append`` on ``/message/content/parts/0`` → text delta.
    - ``replace`` on ``/message/content/parts/0`` → suffix-only text delta.
    - ``replace`` on ``/message/status`` = ``finished_successfully`` → finish.
    """
    state = ctx.state
    env = ctx.inputs

    if env.channel is not None:
        state.current_channel = env.channel

    relevant = state.final_channel is not None and state.current_channel == state.final_channel
    if not relevant:
        return

    for path, op, value in env.patches:
        if path == "/message/content/parts/0":
            if op == "append" and isinstance(value, str) and value:
                state.accumulated_text += value
                state.out_events.extend(
                    state.parts_manager.handle_text_delta(
                        vendor_part_id=_ANSWER_VENDOR_ID,
                        content=value,
                    )
                )
                state.content_begun = True

            elif op == "replace" and isinstance(value, str):
                # Emit only the new suffix relative to already-accumulated text.
                suffix = value[len(state.accumulated_text) :] if value.startswith(state.accumulated_text) else value
                if suffix:
                    state.accumulated_text += suffix
                    state.out_events.extend(
                        state.parts_manager.handle_text_delta(
                            vendor_part_id=_ANSWER_VENDOR_ID,
                            content=suffix,
                        )
                    )
                    state.content_begun = True

        elif (
            path == "/message/status"
            and op == "replace"
            and value == "finished_successfully"
            and state.content_begun
            and not state.final_emitted
        ):
            state.final_emitted = True
            state.finish_reason = "stop"


@_g.step
async def handle_typed_side_event(
    ctx: StepContext[_ConversationsIntakeState, None, _TypedSideEvent],
) -> Any:
    """Process a typed side event.

    Known handoff types are captured into ``state.continuation``.
    When no content has yet been accumulated the router receives
    :class:`_HandoffDetected`. Unknown types are silently ignored.
    """
    state = ctx.state
    env = ctx.inputs
    kind = env.kind
    raw = env.raw

    _handoff_kinds = frozenset(
        {
            "resume_conversation_token",
            "stream_handoff",
            "server_ste_metadata",
        }
    )

    if kind not in _handoff_kinds:
        return None

    resume_token = ""
    handoff_topic = ""
    turn_exchange_id = ""

    if kind == "resume_conversation_token":
        tok = raw.get("token")
        resume_token = tok if isinstance(tok, str) else ""
    else:
        handoff_topic = _handoff_topic_from_side_event(kind, raw)
        if kind == "server_ste_metadata":
            tei = raw.get("turn_exchange_id")
            if isinstance(tei, str):
                turn_exchange_id = tei
            else:
                meta = raw.get("metadata")
                if isinstance(meta, dict):
                    tei2 = meta.get("turn_exchange_id")
                    if isinstance(tei2, str):
                        turn_exchange_id = tei2

    state.continuation = ContinuationMetadata(
        conversation_id=state.conversation_id,
        resume_token=resume_token,
        handoff_topic=handoff_topic,
        turn_exchange_id=turn_exchange_id,
    )

    if not state.content_begun:
        return _HandoffDetected()
    return None


@_g.step
async def handle_done(
    ctx: StepContext[_ConversationsIntakeState, None, _DoneEvent],
) -> None:
    """``[DONE]`` sentinel: record finish when content has arrived and not yet recorded."""
    state = ctx.state
    if state.content_begun and not state.final_emitted:
        state.final_emitted = True
        state.finish_reason = "stop"


@_g.step
async def handle_handoff_detected(
    ctx: StepContext[_ConversationsIntakeState, None, _HandoffDetected],
) -> None:
    """Handoff before content: no-op FSM step; caller reads ``state.continuation``."""


@_g.step
async def emit_done(
    ctx: StepContext[_ConversationsIntakeState, None, _FeedDone],
) -> list[ModelResponseStreamEvent]:
    """Terminal step — drain and return accumulated IR events."""
    out = ctx.state.out_events
    ctx.state.out_events = []
    return out


_g.add(
    _g.edge_from(_g.start_node).to(frame_next_event),
    _g.edge_from(frame_next_event).to(
        _g.decision()
        .branch(_g.match(_FeedDone).to(emit_done))
        .branch(_g.match(_AddEnvelope).to(handle_add))
        .branch(_g.match(_PatchEnvelope).to(handle_patch))
        .branch(_g.match(_TypedSideEvent).to(handle_typed_side_event))
        .branch(_g.match(_DoneEvent).to(handle_done))
    ),
    _g.edge_from(handle_add).to(frame_next_event),
    _g.edge_from(handle_patch).to(frame_next_event),
    _g.edge_from(handle_typed_side_event).to(
        _g.decision()
        .branch(_g.match(_HandoffDetected).to(handle_handoff_detected))
        .branch(_g.match(type(None)).to(frame_next_event))
    ),
    _g.edge_from(handle_handoff_detected).to(frame_next_event),
    _g.edge_from(handle_done).to(frame_next_event),
    _g.edge_from(emit_done).to(_g.end_node),
)

_intake_graph = _g.build()


# ── Public class ───────────────────────────────────────────────────────────────


class OpenAIConversationsIntakeFSM:
    """Async pydantic-graph-driven OpenAI Conversations SSE-v1 intake.

    Parses the ``/backend-api/f/conversation`` SSE-v1 JSON-patch stream
    into pydantic-ai ``TextPart`` events compatible with any render FSM
    (OpenAI Chat, Anthropic, Responses).

    Interface matches :class:`PerplexityResponseIntakeFSM` so
    :class:`SSEPipeline` drives it unchanged.
    """

    name = "openai_conversations"

    def __init__(self, *, model: str, request_params: ModelRequestParameters) -> None:
        self._model = model
        self._request_params = request_params
        self._sse_buffer = bytearray()
        self.upstream_raw_bytes = bytearray()
        self._state = _ConversationsIntakeState(
            parts_manager=ModelResponsePartsManager(model_request_parameters=request_params),
        )

    @property
    def parts_manager(self) -> ModelResponsePartsManager:
        return self._state.parts_manager

    @property
    def state(self) -> _ConversationsIntakeState:
        """Expose FSM state for tests and downstream consumers."""
        return self._state

    @property
    def conversation_id(self) -> str:
        return self._state.conversation_id

    @property
    def message_id(self) -> str:
        return self._state.message_id

    @property
    def continuation(self) -> ContinuationMetadata | None:
        return self._state.continuation

    @property
    def finish_reason(self) -> str | None:
        return self._state.finish_reason

    async def feed(self, data: bytes) -> list[ModelResponseStreamEvent]:
        """Buffer bytes, frame SSE events, drive FSM, return emitted IR events.

        Raises :class:`HandoffUnsupportedError` when a handoff side event
        arrives before any visible content has been accumulated.
        """
        if not data:
            return []
        self.upstream_raw_bytes.extend(data)
        self._sse_buffer.extend(data)
        for envelope in self._drain_sse_frames():
            self._state.events_queue.append(envelope)
        if not self._state.events_queue:
            return []
        result = await _intake_graph.run(state=self._state)
        # Post-run check: if a handoff was registered with no prior content, error.
        if self._state.continuation and not self._state.content_begun:
            raise HandoffUnsupportedError(self._state.continuation)
        return result

    async def close(self) -> list[ModelResponseStreamEvent]:
        """End of stream — no trailing events beyond what handle_done already emitted."""
        return []

    def _drain_sse_frames(self) -> Iterator[Any]:
        """Frame and normalise SSE events from ``self._sse_buffer``.

        Handles both ``\\r\\n\\r\\n`` (RFC standard) and ``\\n\\n`` separators;
        partial frames remain buffered for the next ``feed`` call.
        """
        while True:
            crlf = self._sse_buffer.find(b"\r\n\r\n")
            lf = self._sse_buffer.find(b"\n\n")
            if crlf == -1 and lf == -1:
                return
            if crlf != -1 and (lf == -1 or crlf < lf):
                sep_idx, sep_len = crlf, 4
            else:
                sep_idx, sep_len = lf, 2
            frame = bytes(self._sse_buffer[:sep_idx])
            del self._sse_buffer[: sep_idx + sep_len]
            envelope = _parse_frame(frame)
            if envelope is not None:
                yield envelope


def _parse_frame(frame: bytes) -> Any:
    """Parse one SSE frame bytes into a dispatch envelope.

    Returns one of :class:`_AddEnvelope`, :class:`_PatchEnvelope`,
    :class:`_TypedSideEvent`, :class:`_DoneEvent`, or ``None`` (silently
    dropped: encoding banner, keepalive comment, un-parseable data).
    """
    event_name: str | None = None
    data_lines: list[str] = []

    for raw_line in frame.split(b"\n"):
        line = raw_line.rstrip(b"\r").decode("utf-8", errors="replace")
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    data = "\n".join(data_lines).strip()
    if not data:
        return None

    # Encoding banner: drop silently.
    if event_name == "delta_encoding":
        return None

    if data == "[DONE]":
        return _DoneEvent()

    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, dict):
        return None

    # Typed side event: ``{type, ...}`` without ``p`` or ``v`` fields.
    kind = parsed.get("type")
    if isinstance(kind, str) and "v" not in parsed and "p" not in parsed:
        return _TypedSideEvent(kind=kind, raw=parsed)

    channel = _channel_from_frame(parsed)
    patches = _parse_delta(parsed)
    if not patches:
        return None

    # A single add-at-root with a channel declaration → AddEnvelope.
    if len(patches) == 1:
        path, op, value = patches[0]
        if path == "" and op == "add" and isinstance(value, dict):
            ch = channel if channel is not None else 0
            return _AddEnvelope(channel=ch, value=value)

    return _PatchEnvelope(channel=channel, patches=patches)
