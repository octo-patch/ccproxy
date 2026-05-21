"""IR events → Anthropic Messages SSE wire bytes (sync).

Inverse of :mod:`ccproxy.lightllm.response.intake_anthropic`. Consumes
``ModelResponseStreamEvent`` IR objects produced by any per-vendor
``ResponseIntake`` and serializes them to Anthropic Messages API SSE
frames suitable for clients that speak the Anthropic streaming wire
protocol.

Event sequence emitted per stream:

  1. ``message_start`` — once at stream start (synthesized on the first
     incoming ``PartStartEvent`` or, for an empty stream, in :meth:`close`).
  2. ``content_block_start`` — once per part, mapping the IR part class
     to the matching Anthropic block descriptor (text / thinking /
     redacted_thinking / tool_use).
  3. ``content_block_delta`` — once per ``PartDeltaEvent``; the delta
     subtype selects the wire delta type (text_delta / thinking_delta /
     signature_delta / input_json_delta).
  4. ``content_block_stop`` — once per ``PartEndEvent`` (and again from
     :meth:`close` if a block is still open at stream end).
  5. ``message_delta`` — stop_reason + usage placeholder. Emitted from
     :meth:`close`.
  6. ``message_stop`` — emitted from :meth:`close`.

The exhaustive ``isinstance`` ladder in :meth:`render` ends with
``assert_never(event)`` so mypy/ty catch any new
``ModelResponseStreamEvent`` variant that pydantic-ai adds upstream.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, assert_never

from pydantic_ai.messages import (
    BuiltinToolCallPart,
    FinalResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelResponseStreamEvent

logger = logging.getLogger(__name__)


class AnthropicResponseRender:
    """Sync renderer for the Anthropic Messages SSE wire format.

    State machine tracking one open content block at a time, mirroring the
    Anthropic streaming protocol's ``content_block_start`` /
    ``content_block_delta`` / ``content_block_stop`` envelope.
    """

    name = "anthropic_messages"

    def __init__(self, *, model: str = "unknown") -> None:
        self._message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self._model = model
        self._started = False
        self._open_block_index: int | None = None

    def render(self, event: ModelResponseStreamEvent) -> bytes:
        """One IR event → zero-or-more bytes of Anthropic SSE wire output."""
        if isinstance(event, PartStartEvent):
            return self._on_part_start(event)
        if isinstance(event, PartDeltaEvent):
            return self._on_part_delta(event)
        if isinstance(event, PartEndEvent):
            return self._on_part_end(event)
        if isinstance(event, FinalResultEvent):
            # Informational; no Anthropic wire equivalent.
            return b""
        assert_never(event)

    def close(self) -> bytes:
        """Flush any open block, then emit ``message_delta`` + ``message_stop``."""
        out = bytearray()
        if self._open_block_index is not None:
            out += self._emit_content_block_stop(self._open_block_index)
            self._open_block_index = None
        if not self._started:
            # Empty stream — still emit a valid envelope so the client sees a
            # parseable response.
            out += self._emit_message_start()
            self._started = True
        out += self._emit_message_delta()
        out += self._emit_message_stop()
        return bytes(out)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_part_start(self, event: PartStartEvent) -> bytes:
        out = bytearray()
        if not self._started:
            out += self._emit_message_start()
            self._started = True
        if self._open_block_index is not None:
            # New part start without an explicit PartEndEvent — close the previous
            # block before opening the new one. PartStartEvent.index is the IR
            # part index; we mirror it as the Anthropic block index.
            out += self._emit_content_block_stop(self._open_block_index)
        out += self._emit_content_block_start(event.index, event.part)
        self._open_block_index = event.index
        # If the start event already carries content (e.g. the intake collapsed an
        # empty content_block_start + the first delta into a single PartStartEvent
        # with a non-empty TextPart), emit that content as an initial delta so the
        # downstream client sees the same accumulated text.
        out += self._emit_initial_content_deltas(event.index, event.part)
        return bytes(out)

    def _on_part_delta(self, event: PartDeltaEvent) -> bytes:
        if self._open_block_index is None:
            # Defensive: a delta without an open block can't be expressed in
            # Anthropic's wire format.
            logger.debug("anthropic render: PartDeltaEvent with no open block; dropping")
            return b""
        return self._emit_content_block_delta(event.index, event.delta)

    def _on_part_end(self, event: PartEndEvent) -> bytes:
        if self._open_block_index is None:
            return b""
        out = self._emit_content_block_stop(event.index)
        self._open_block_index = None
        return out

    # ------------------------------------------------------------------
    # Wire emission helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _emit(event_name: str, body: dict[str, Any]) -> bytes:
        return f"event: {event_name}\ndata: {json.dumps(body, separators=(',', ':'))}\n\n".encode()

    def _emit_message_start(self) -> bytes:
        return self._emit(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self._message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self._model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

    def _emit_content_block_start(self, idx: int, part: Any) -> bytes:
        block: dict[str, Any]
        if isinstance(part, TextPart):
            block = {"type": "text", "text": ""}
        elif isinstance(part, ThinkingPart):
            if part.id == "redacted_thinking":
                # Anthropic redacted_thinking carries the opaque payload in `data`;
                # pydantic-ai stashes that on the part's `signature` field.
                block = {"type": "redacted_thinking", "data": part.signature or ""}
            else:
                block = {"type": "thinking", "thinking": "", "signature": ""}
        elif isinstance(part, ToolCallPart | BuiltinToolCallPart):
            block = {
                "type": "tool_use",
                "id": part.tool_call_id,
                "name": part.tool_name,
                "input": {},
            }
        else:
            # CompactionPart, FilePart, builtin-tool-return variants: no clean
            # Anthropic-streaming wire mapping; emit an empty text block so the
            # envelope stays well-formed.
            logger.debug(
                "anthropic render: no wire mapping for part %s; emitting empty text block",
                type(part).__name__,
            )
            block = {"type": "text", "text": ""}
        return self._emit(
            "content_block_start",
            {"type": "content_block_start", "index": idx, "content_block": block},
        )

    def _emit_initial_content_deltas(self, idx: int, part: Any) -> bytes:
        """Emit deltas for any non-empty content carried by a starting part.

        The intake collapses an Anthropic ``content_block_start`` whose initial
        content is non-empty (text/thinking) directly into a ``PartStartEvent``
        with that content already populated. On the wire, the equivalent
        Anthropic events are ``content_block_start`` (empty) + a single
        ``content_block_delta`` (with the initial value). Replay the deltas so
        the rendered stream preserves the full content.
        """
        out = bytearray()
        if isinstance(part, TextPart) and part.content:
            out += self._emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": part.content},
                },
            )
        elif isinstance(part, ThinkingPart) and part.id != "redacted_thinking":
            if part.content:
                out += self._emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "thinking_delta", "thinking": part.content},
                    },
                )
            if part.signature:
                out += self._emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "signature_delta", "signature": part.signature},
                    },
                )
        elif isinstance(part, ToolCallPart | BuiltinToolCallPart):
            partial_json = self._tool_args_to_json_string(part.args)
            if partial_json:
                out += self._emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "input_json_delta", "partial_json": partial_json},
                    },
                )
        return bytes(out)

    def _emit_content_block_delta(self, idx: int, delta: Any) -> bytes:
        wire_delta: dict[str, Any]
        if isinstance(delta, TextPartDelta):
            wire_delta = {"type": "text_delta", "text": delta.content_delta}
        elif isinstance(delta, ThinkingPartDelta):
            if delta.signature_delta is not None:
                wire_delta = {"type": "signature_delta", "signature": delta.signature_delta}
            elif delta.content_delta is not None:
                wire_delta = {"type": "thinking_delta", "thinking": delta.content_delta}
            else:
                logger.debug("anthropic render: empty ThinkingPartDelta; dropping")
                return b""
        elif isinstance(delta, ToolCallPartDelta):
            partial_json = self._tool_args_to_json_string(delta.args_delta)
            if partial_json is None:
                logger.debug("anthropic render: ToolCallPartDelta with no args_delta; dropping")
                return b""
            wire_delta = {"type": "input_json_delta", "partial_json": partial_json}
        else:
            logger.debug("anthropic render: unknown delta type %s; dropping", type(delta).__name__)
            return b""
        return self._emit(
            "content_block_delta",
            {"type": "content_block_delta", "index": idx, "delta": wire_delta},
        )

    @staticmethod
    def _tool_args_to_json_string(args_delta: str | dict[str, Any] | None) -> str | None:
        """Serialize a ``ToolCallPartDelta.args_delta`` to the wire ``partial_json`` shape.

        On the Anthropic wire ``input_json_delta.partial_json`` is always a string —
        the partially-arrived JSON. If the IR carries a dict (because the upstream
        intake already merged accumulated deltas), JSON-encode it.
        """
        if args_delta is None:
            return None
        if isinstance(args_delta, str):
            return args_delta
        return json.dumps(args_delta, separators=(",", ":"))

    def _emit_content_block_stop(self, idx: int) -> bytes:
        return self._emit("content_block_stop", {"type": "content_block_stop", "index": idx})

    def _emit_message_delta(self) -> bytes:
        return self._emit(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
        )

    def _emit_message_stop(self) -> bytes:
        return self._emit("message_stop", {"type": "message_stop"})
