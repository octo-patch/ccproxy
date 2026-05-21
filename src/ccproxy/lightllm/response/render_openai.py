"""IR events -> OpenAI Chat Completion SSE wire bytes (sync).

Inverse of :mod:`ccproxy.lightllm.response.intake_openai`. Consumes
``ModelResponseStreamEvent`` IR objects and emits ``chat.completion.chunk``
SSE wire bytes — the byte stream that a client polling
``POST /v1/chat/completions`` with ``stream=true`` expects.

Emission contract
-----------------

1. First chunk carries ``delta = {"role": "assistant"}`` (no content).
2. Text content arrives as ``delta = {"content": "<delta>"}``.
3. Tool calls land as ``delta = {"tool_calls": [{...}]}``:
   - First chunk per tool call: ``{index, id, type, function: {name, arguments}}``.
   - Subsequent chunks: ``{index, function: {arguments}}`` (partial args).
4. Final chunk has empty delta and ``finish_reason``.
5. ``data: [DONE]\\n\\n`` terminator from :meth:`close`.

The OpenAI ``tool_calls[].index`` is the position in the chunk's tool-call
array — not the IR ``part.index``. We map IR part indices onto a
monotonically-increasing OpenAI tool-call index so consecutive
``ToolCallPartDelta`` updates targeting the same IR part land in the same
OpenAI tool-call slot.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Literal, assert_never

from pydantic_ai.messages import (
    FinalResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelResponseStreamEvent


_FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


class OpenAIResponseRender:
    """Per-stream sync renderer for OpenAI Chat Completion SSE output."""

    name = "openai_chat"

    def __init__(self, *, model: str = "unknown") -> None:
        self._id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self._created = int(time.time())
        self._model = model
        self._role_emitted = False
        self._part_to_tool_call_index: dict[int, int] = {}
        self._next_tool_call_index = 0
        self._finish_reason: _FinishReason = "stop"

    def render(self, event: ModelResponseStreamEvent) -> bytes:
        """One IR event -> zero-or-more SSE wire bytes."""
        if isinstance(event, PartStartEvent):
            return self._on_part_start(event)
        if isinstance(event, PartDeltaEvent):
            return self._on_part_delta(event)
        if isinstance(event, PartEndEvent):
            return b""
        if isinstance(event, FinalResultEvent):
            return b""
        assert_never(event)

    def close(self) -> bytes:
        """Emit the final ``finish_reason`` chunk plus the ``[DONE]`` terminator."""
        out = bytearray()
        out += self._emit_chunk(delta={}, finish_reason=self._finish_reason)
        out += b"data: [DONE]\n\n"
        return bytes(out)

    def _ensure_role(self) -> bytes:
        """Emit the role chunk once, lazily, before any content chunk."""
        if self._role_emitted:
            return b""
        self._role_emitted = True
        return self._emit_chunk(delta={"role": "assistant"})

    def _on_part_start(self, event: PartStartEvent) -> bytes:
        out = bytearray()
        out += self._ensure_role()

        part = event.part
        if isinstance(part, TextPart):
            if part.content:
                out += self._emit_chunk(delta={"content": part.content})
        elif isinstance(part, ToolCallPart):
            tc_index = self._next_tool_call_index
            self._next_tool_call_index += 1
            self._part_to_tool_call_index[event.index] = tc_index
            out += self._emit_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": tc_index,
                            "id": part.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": part.tool_name,
                                "arguments": _args_to_str(part.args),
                            },
                        }
                    ]
                }
            )
            self._finish_reason = "tool_calls"
        # ThinkingPart, CompactionPart, FilePart, NativeToolCall* etc. have no
        # OpenAI Chat Completion wire surface — the role chunk above is the
        # only output. They're handled implicitly by falling through.
        return bytes(out)

    def _on_part_delta(self, event: PartDeltaEvent) -> bytes:
        delta = event.delta
        if isinstance(delta, TextPartDelta):
            out = bytearray()
            out += self._ensure_role()
            out += self._emit_chunk(delta={"content": delta.content_delta})
            return bytes(out)

        if isinstance(delta, ToolCallPartDelta):
            out = bytearray()
            out += self._ensure_role()
            tc_index = self._part_to_tool_call_index.get(event.index)
            if tc_index is None:
                # First sighting of this IR part via a delta — allocate an
                # OpenAI tool-call slot and emit the envelope (id + name + type).
                tc_index = self._next_tool_call_index
                self._next_tool_call_index += 1
                self._part_to_tool_call_index[event.index] = tc_index
                envelope: dict[str, Any] = {"index": tc_index, "type": "function"}
                if delta.tool_call_id is not None:
                    envelope["id"] = delta.tool_call_id
                fn: dict[str, Any] = {}
                if delta.tool_name_delta is not None:
                    fn["name"] = delta.tool_name_delta
                fn["arguments"] = _args_to_str(delta.args_delta)
                envelope["function"] = fn
                self._finish_reason = "tool_calls"
                out += self._emit_chunk(delta={"tool_calls": [envelope]})
                return bytes(out)

            self._finish_reason = "tool_calls"
            args_str = _args_to_str(delta.args_delta)
            out += self._emit_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": tc_index,
                            "function": {"arguments": args_str},
                        }
                    ]
                }
            )
            return bytes(out)

        if isinstance(delta, ThinkingPartDelta):
            # OpenAI Chat Completion SSE has no on-wire surface for thinking
            # content (the ``reasoning`` field is OpenAI Responses only).
            return b""

        # ``ModelResponsePartDelta`` is a closed union; if pydantic-ai ever
        # extends it the next mypy run flags this branch.
        assert_never(delta)

    def _emit_chunk(self, *, delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
        chunk: dict[str, Any] = {
            "id": self._id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
        }
        return f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode()


def _args_to_str(args: str | dict[str, Any] | None) -> str:
    """OpenAI Chat Completion wires tool-call arguments as a JSON string.

    pydantic-ai's IR holds either a string fragment (already-serialized
    JSON), a fully-formed dict, or ``None``. Normalize to the on-wire shape.
    """
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    return json.dumps(args, separators=(",", ":"))
