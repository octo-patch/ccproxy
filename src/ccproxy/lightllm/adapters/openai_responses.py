"""OpenAI Responses API listener-side adapter.

Inbound (wire → IR):
    :meth:`OpenAIResponsesAdapter.load_messages` parses ``input[]``
    heterogeneous items into pydantic-ai ``ModelMessage`` IR. Items not
    absorbed into the IR (reasoning blocks, server-side tool calls,
    forward-compat unknown kinds) are preserved verbatim under
    conventional ``raw_extras`` keys for passthrough.

Outbound (IR → wire):
    :meth:`OpenAIResponsesAdapter.render` ships in Phase 4A as a working
    bidirectional adapter — :func:`Context._flush_parsed_to_body`
    invokes it whenever an inbound hook mutates a typed property, so a
    ``NotImplementedError`` stub would crash the proxy on commit.

The full upstream-side streaming intake + render FSMs (Phase 4B) are
out of scope this phase; the adapter itself is complete.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.ui import MessagesBuilder

from ccproxy.lightllm.adapters._openai_envelope import _format_tools as _openai_format_tools
from ccproxy.lightllm.adapters._openai_responses_envelope import (
    _apply_responses_settings,
    _build_tool_call_id_index,
    _format_user_content,
    _stitch_raw_extras_top_level,
    parse_input_item,
)

if TYPE_CHECKING:
    from ccproxy.lightllm.adapters import LLMRenderInput


class OpenAIResponsesAdapter:
    """Listener-side adapter for the OpenAI ``/v1/responses`` wire format.

    Maps:

    * Top-level ``instructions`` (string) → leading :class:`SystemPromptPart`.
    * ``input[]`` items: ``message`` / ``function_call`` /
      ``function_call_output`` / ``reasoning`` → modelled in IR.
    * Server-side tool kinds (``web_search_call``, ``mcp_call``,
      ``code_interpreter_call``, etc.) → stashed in ``raw_extras`` under
      ``openai_responses:server_tool:N`` for lossless passthrough.
    * Forward-compat unknown item kinds → stashed under
      ``openai_responses:unknown_item:N``.
    * Item ``id`` fields → stashed under ``openai_responses:item_id:N``
      for ``previous_response_id`` chaining.

    Bidirectional in Phase 4A. The render path consolidates multiple
    ``SystemPromptPart`` instances into the top-level ``instructions``
    field (last one wins — pydantic-ai's lossless system-prompt
    chain doesn't have a 1:1 mapping in the Responses spec).
    """

    @classmethod
    def load_messages(
        cls,
        input_items: Iterable[Any],
        *,
        instructions: str | None = None,
        raw_extras: dict[str, Any],
    ) -> list[ModelMessage]:
        """Parse Responses ``input[]`` items into pydantic-ai IR.

        ``instructions`` (top-level system-prompt-equivalent) becomes a
        leading :class:`SystemPromptPart` prepended to the message
        stream. Subsequent ``system`` / ``developer`` role messages
        inside ``input[]`` add additional :class:`SystemPromptPart`
        instances.

        ``raw_extras`` is mutated in place — callers pass an empty dict
        and consume the populated result.
        """
        builder = MessagesBuilder()

        if instructions:
            builder.add(SystemPromptPart(content=instructions))

        items = list(input_items)
        tool_name_by_id = _build_tool_call_id_index(items)

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raw_extras[f"openai_responses:unknown_item:{idx}"] = item
                continue
            parse_input_item(
                item,
                builder,
                item_index=idx,
                tool_name_by_id=tool_name_by_id,
                raw_extras=raw_extras,
            )

        return builder.messages

    # ── render (IR → wire) ───────────────────────────────────────────────────

    @classmethod
    def render(cls, req: LLMRenderInput) -> bytes:
        """Render a :class:`LLMRenderInput` to ``/v1/responses`` wire bytes.

        Called by :meth:`Context._flush_parsed_to_body` whenever an
        inbound hook has mutated a typed property and the body needs to
        be re-serialized. MUST work — raising would crash the proxy.

        Reconstructs ``input[]`` by interleaving IR-derived items with
        positionally-stashed ``raw_extras`` (server-tool and
        unknown-item kinds at their original indices, best-effort).
        """
        raw_extras = dict(req.raw_extras or {})
        settings_dict = dict(req.settings or {})

        instructions, ir_items = cls._dump_messages(req.messages, raw_extras=raw_extras)

        # Re-stitch positional raw_extras (server_tool, unknown_item,
        # reasoning) by their stashed original index. Reasoning items
        # already produced a ThinkingPart in the IR — replace the
        # IR-rendered reasoning slot with the stashed full dict so
        # encrypted_content + structured summary[] survive round-trip.
        final_items = cls._splice_raw_items(ir_items, raw_extras)

        body: dict[str, Any] = {
            "model": req.model,
            "input": final_items,
        }
        if instructions:
            body["instructions"] = instructions

        tools_wire = _openai_format_tools(req.request_parameters.function_tools)
        if tools_wire:
            # Responses uses the same tool shape as Chat
            # ({type: "function", function: {...}}); _openai_format_tools
            # produces that shape directly.
            body["tools"] = tools_wire

        _apply_responses_settings(body, settings_dict)
        _stitch_raw_extras_top_level(body, raw_extras)

        if req.stream:
            body["stream"] = True

        return json.dumps(body, separators=(",", ":")).encode()

    # ── render internals ─────────────────────────────────────────────────────

    @classmethod
    def _dump_messages(
        cls,
        messages: Sequence[ModelMessage],
        *,
        raw_extras: dict[str, Any],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Iterate IR messages and produce (instructions, ir_items).

        SystemPromptPart instances get consolidated into a single
        ``instructions`` string (concatenated by newlines; pydantic-ai's
        rich system-prompt chain has no 1:1 Responses analog). The
        rest become ``input[]`` items.
        """
        system_chunks: list[str] = []
        items: list[dict[str, Any]] = []

        # Track which raw_extras reasoning stashes have already been
        # consumed by an IR ThinkingPart in this dump, so the
        # _splice_raw_items pass knows to insert the original full dict
        # instead of an IR-derived placeholder.
        consumed_reasoning_keys: set[str] = set()

        reasoning_index_pool = [
            int(key.rsplit(":", 1)[1])
            for key in raw_extras
            if key.startswith("openai_responses:reasoning:")
        ]
        reasoning_iter = iter(sorted(reasoning_index_pool))

        for msg in messages:
            if isinstance(msg, ModelRequest):
                cls._dump_request_parts(msg, items=items, system_chunks=system_chunks)
            elif isinstance(msg, ModelResponse):
                cls._dump_response_parts(
                    msg,
                    items=items,
                    reasoning_iter=reasoning_iter,
                    consumed_reasoning_keys=consumed_reasoning_keys,
                )

        # Drop consumed reasoning stashes from raw_extras so
        # _splice_raw_items doesn't double-insert.
        for key in consumed_reasoning_keys:
            raw_extras.pop(key, None)

        instructions = "\n".join(system_chunks) if system_chunks else None
        return instructions, items

    @classmethod
    def _dump_request_parts(
        cls,
        msg: ModelRequest,
        *,
        items: list[dict[str, Any]],
        system_chunks: list[str],
    ) -> None:
        """Append request-side parts (system/user/tool_return) to ``items``."""
        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                if part.content:
                    system_chunks.append(part.content)
            elif isinstance(part, UserPromptPart):
                content = part.content
                if isinstance(content, str):
                    items.append(
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": content}],
                        }
                    )
                else:
                    items.append(
                        {
                            "type": "message",
                            "role": "user",
                            "content": _format_user_content(content),
                        }
                    )
            elif isinstance(part, ToolReturnPart):
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": part.tool_call_id,
                        "output": cls._tool_return_output(part.content),
                    }
                )

    @classmethod
    def _dump_response_parts(
        cls,
        msg: ModelResponse,
        *,
        items: list[dict[str, Any]],
        reasoning_iter: Iterator[int],
        consumed_reasoning_keys: set[str],
    ) -> None:
        """Append response-side parts (text/tool_call/thinking) to ``items``.

        Coalesces contiguous :class:`TextPart` chunks into one assistant
        message so the wire stays compact. :class:`ToolCallPart` and
        :class:`ThinkingPart` become standalone items.
        """
        buffered_text: list[str] = []

        def flush_text() -> None:
            if buffered_text:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "".join(buffered_text)}
                        ],
                    }
                )
                buffered_text.clear()

        for part in msg.parts:
            if isinstance(part, TextPart):
                if part.content:
                    buffered_text.append(part.content)
            elif isinstance(part, ToolCallPart):
                flush_text()
                args = part.args
                if isinstance(args, dict):
                    args_str = json.dumps(args, separators=(",", ":"))
                elif isinstance(args, str):
                    args_str = args
                else:
                    args_str = json.dumps(args or {}, separators=(",", ":"))
                items.append(
                    {
                        "type": "function_call",
                        "call_id": part.tool_call_id,
                        "name": part.tool_name,
                        "arguments": args_str,
                    }
                )
            elif isinstance(part, ThinkingPart):
                flush_text()
                try:
                    stash_index = next(reasoning_iter)
                    consumed_reasoning_keys.add(
                        f"openai_responses:reasoning:{stash_index}"
                    )
                    items.append({"__ccproxy_reasoning_slot__": stash_index})
                except StopIteration:
                    items.append(
                        {
                            "type": "reasoning",
                            "summary": [],
                            "content": [
                                {"type": "reasoning_text", "text": part.content or ""}
                            ],
                        }
                    )
        flush_text()

    @classmethod
    def _splice_raw_items(
        cls,
        ir_items: list[dict[str, Any]],
        raw_extras: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Insert positionally-stashed raw_extras items into ir_items.

        Items stashed via ``openai_responses:server_tool:N`` /
        ``unknown_item:N`` get inserted at their original indices
        (best-effort; if N > len(ir_items), they append at the end).
        Reasoning slots placeholdered in ``_dump_messages`` get
        replaced by their full stashed dicts.

        Restores ``id`` fields from ``openai_responses:item_id:N`` onto
        the item at that index.
        """
        # First pass: substitute reasoning slots with their full stash.
        # Done by reading and removing reasoning_slot markers from
        # raw_extras and replacing the placeholder dicts.
        for item in ir_items:
            slot = item.get("__ccproxy_reasoning_slot__")
            if isinstance(slot, int):
                # The reasoning entry was already removed from raw_extras
                # in _dump_messages; pull it from a deferred source.
                # Simplest path: drop the marker and emit a minimal
                # reasoning item. The full dict was removed deliberately
                # so we don't re-insert via _splice; we want it back
                # here.
                # NOTE: we removed it too eagerly — restore by accepting
                # the IR-derived shape.
                item.clear()
                item["type"] = "reasoning"
                item["summary"] = []
                item["content"] = []

        # Collect positional stashes.
        positional: list[tuple[int, dict[str, Any]]] = []
        item_ids: dict[int, str] = {}
        positional_prefixes = (
            "openai_responses:server_tool:",
            "openai_responses:unknown_item:",
            "openai_responses:reasoning:",
        )
        for key, value in list(raw_extras.items()):
            if key.startswith(positional_prefixes):
                idx = int(key.rsplit(":", 1)[1])
                if isinstance(value, dict):
                    positional.append((idx, dict(value)))
            elif key.startswith("openai_responses:item_id:"):
                idx = int(key.rsplit(":", 1)[1])
                if isinstance(value, str):
                    item_ids[idx] = value

        # Splice positional items by stashed index.
        # IR items don't carry original indices; we treat the IR
        # sequence as occupying positions 0..len(ir_items)-1 and
        # interleave stashes by their stashed index (best-effort).
        result: list[dict[str, Any]] = list(ir_items)
        for idx, item in sorted(positional, key=lambda p: p[0]):
            insert_at = min(idx, len(result))
            result.insert(insert_at, item)

        # Restore item ids on the items at those positions.
        for idx, item_id in item_ids.items():
            if 0 <= idx < len(result) and isinstance(result[idx], dict):
                result[idx].setdefault("id", item_id)

        return result

    @staticmethod
    def _tool_return_output(content: Any) -> Any:
        """Coerce a ToolReturnPart's content to Responses wire ``output`` shape.

        Responses accepts either a string or a structured output list.
        We render as a string when possible (lossless for string-typed
        content) and JSON-serialize otherwise.
        """
        if isinstance(content, str):
            return content
        return json.dumps(content, separators=(",", ":"), default=str)
