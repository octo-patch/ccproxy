"""Anthropic Messages UIAdapter.

Converts Anthropic Messages request JSON to / from pydantic-ai's
``list[ModelMessage]`` IR. Reuses the SDK's `TypedDict`s
(``anthropic.types.beta.*``) for typed dispatch.

Replaces the two-FSM stack in ``ccproxy.lightllm.graph.anthropic_load``
plus ``ccproxy.lightllm.graph.anthropic_dump`` with a single procedural
adapter modeled on the pydantic-ai UI adapters in
``pydantic_ai.ui.{ag_ui,vercel_ai}``.

The Anthropic API uses a top-level ``system`` field separate from
``messages``; :meth:`dump_system` extracts it from IR, keeping
:meth:`dump_messages` returning only conversation turns. ``CachePoint``
items in IR are emitted as ``cache_control`` annotations on the
preceding block (or, for system blocks, on the matching system block).

``build_event_stream`` raises ``NotImplementedError``; streaming
intake/render still lives in ``ccproxy.lightllm.graph.anthropic_*``.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, cast

logger = logging.getLogger(__name__)

from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessageParam,
    BetaRedactedThinkingBlockParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from anthropic.types.beta.message_create_params import MessageCreateParamsBase
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.ui import MessagesBuilder, UIAdapter, UIEventStream

# pydantic-ai's CachePoint accepts only these two TTLs (Literal['5m', '1h']);
# anything else stashes in raw_extras via the per-block `cc:` key convention.
_SUPPORTED_TTLS: frozenset[str] = frozenset({"5m", "1h"})


@dataclass
class AnthropicAdapter(UIAdapter[MessageCreateParamsBase, BetaMessageParam, Any, AgentDepsT, OutputDataT]):
    """UIAdapter for the Anthropic Messages API wire format.

    Maps:

    * Top-level ``system`` (string or block array, possibly with
      ``cache_control``) → :class:`SystemPromptPart` chain (with sentinel
      :class:`UserPromptPart`-wrapped :class:`CachePoint` markers)
    * User turns: ``text`` / ``image`` / ``document`` / ``tool_result``
    * Assistant turns: ``text`` / ``thinking`` / ``redacted_thinking`` / ``tool_use``
    * ``cache_control`` on any block → :class:`CachePoint` appended after
      the matching content item
    * Base64 sources → :class:`BinaryContent`
    * URL sources → :class:`ImageUrl` / :class:`DocumentUrl`
    * File-ID sources → :class:`UploadedFile`

    :meth:`dump_messages` returns only the conversation turns; call
    :meth:`dump_system` separately to extract the ``system`` field.
    """

    @classmethod
    def build_run_input(cls, body: bytes) -> MessageCreateParamsBase:
        import json

        return cast(MessageCreateParamsBase, json.loads(body))

    @cached_property
    def messages(self) -> list[ModelMessage]:
        return self.load_messages(
            self.run_input["messages"],
            system=self.run_input.get("system"),
        )

    # ── load (wire → IR) ─────────────────────────────────────────────────────

    @classmethod
    def load_messages(  # noqa: PLR0912
        cls,
        messages: Iterable[BetaMessageParam],
        *,
        system: str | Iterable[BetaTextBlockParam] | None = None,
        raw_extras: dict[str, Any] | None = None,
    ) -> list[ModelMessage]:
        """Convert Anthropic ``messages`` + top-level ``system`` to IR.

        ``tool_result`` blocks don't carry the tool name — we scan all
        assistant turns first to build a ``{tool_use_id: tool_name}`` index.

        When ``raw_extras`` is provided, fields the IR doesn't model are
        stashed there for lossless round-trip:

        * ``cc:msg:N:block:M`` — non-standard ``cache_control`` TTLs
          (TTL ≠ ``5m``/``1h``)
        * ``unknown_block:msg:N:idx:M`` — unrecognized content blocks
        """
        messages = list(messages)

        tool_name_by_id: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, str) or content is None:
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                blk = cast(Mapping[str, Any], block)
                if blk.get("type") == "tool_use":
                    tool_name_by_id[blk["id"]] = blk["name"]

        builder = MessagesBuilder()

        if system is not None:
            if isinstance(system, str):
                if system:
                    builder.add(SystemPromptPart(content=system))
            else:
                for block in system:
                    builder.add(SystemPromptPart(content=block["text"]))
                    if cc := block.get("cache_control"):
                        # Sentinel UserPromptPart([CachePoint]) preserves the
                        # system-level cache marker; dump_system recovers it.
                        builder.add(UserPromptPart(content=[CachePoint(ttl=cls._cache_ttl(cc))]))

        for msg_index, msg in enumerate(messages):
            role = msg.get("role")
            if role == "user":
                cls._load_user_turn(
                    msg, builder, tool_name_by_id,
                    msg_index=msg_index, raw_extras=raw_extras,
                )
            elif role == "assistant":
                cls._load_assistant_turn(
                    msg, builder, msg_index=msg_index, raw_extras=raw_extras,
                )
            elif role == "system":
                # Some clients put system prompts inline in messages[] rather than
                # at the top-level `system` field. Surface them as SystemPromptParts.
                content = msg.get("content")
                if isinstance(content, str):
                    if content:
                        builder.add(SystemPromptPart(content=content))
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            builder.add(SystemPromptPart(content=block.get("text", "")))

        return builder.messages

    @classmethod
    def _load_user_turn(  # noqa: PLR0912, PLR0913
        cls,
        msg: BetaMessageParam,
        builder: MessagesBuilder,
        tool_name_by_id: dict[str, str],
        *,
        msg_index: int = 0,
        raw_extras: dict[str, Any] | None = None,
    ) -> None:
        """Process one Anthropic user turn into request parts.

        A single user turn may interleave regular content (text, image,
        document) with ``tool_result`` blocks. Regular content accumulates
        into one ``UserPromptPart``; each ``tool_result`` flushes the
        accumulator and becomes a standalone ``ToolReturnPart``.
        """
        content = msg.get("content")
        if isinstance(content, str):
            builder.add(UserPromptPart(content=content))
            return
        if not isinstance(content, list):
            # Defensive: non-list/non-string content (e.g., an integer) — emit
            # an empty UserPromptPart to keep the turn slot.
            return

        accumulated: list[UserContent] = []

        def flush() -> None:
            if not accumulated:
                return
            # Wire-side block was a list — keep IR content as a list to preserve
            # the round-trip shape (a single text item without cache markers
            # also stays a list, matching the legacy behavior).
            builder.add(UserPromptPart(content=list(accumulated)))
            accumulated.clear()

        def push_cache_marker(cc: Mapping[str, Any], block_index: int) -> None:
            # When ``cache_control`` is present without an explicit ``ttl``,
            # Anthropic defaults to ``5m``; mirror that so a present-but-empty
            # cc still produces a CachePoint.
            ttl = cc.get("ttl", "5m") if isinstance(cc, dict) else None
            if ttl in _SUPPORTED_TTLS:
                accumulated.append(CachePoint(ttl=cast(Literal["5m", "1h"], ttl)))
            elif raw_extras is not None and isinstance(cc, dict):
                raw_extras[f"cc:msg:{msg_index}:block:{block_index}"] = dict(cc)

        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                if raw_extras is not None:
                    raw_extras[f"unknown_block:msg:{msg_index}:idx:{block_index}"] = block
                accumulated.append(json.dumps(block))
                continue

            blk = cast(Mapping[str, Any], block)
            btype = blk.get("type")

            if btype == "text":
                accumulated.append(blk["text"])
                if cc := blk.get("cache_control"):
                    push_cache_marker(cc, block_index)

            elif btype == "image":
                accumulated.append(cls._load_image(blk.get("source") or {}))
                if cc := blk.get("cache_control"):
                    push_cache_marker(cc, block_index)

            elif btype == "document":
                accumulated.append(cls._load_document(blk.get("source") or {}, media_type=blk.get("media_type")))
                if cc := blk.get("cache_control"):
                    push_cache_marker(cc, block_index)

            elif btype == "tool_result":
                flush()
                tool_use_id = blk.get("tool_use_id", "")
                tool_name = tool_name_by_id.get(tool_use_id, "")
                if not tool_name and tool_use_id:
                    logger.debug(
                        "anthropic load: tool_result references unknown tool_use_id %r — leaving tool_name blank",
                        tool_use_id,
                    )
                outcome: Literal["success", "failed"] = "failed" if blk.get("is_error") else "success"
                builder.add(
                    ToolReturnPart(
                        tool_name=tool_name,
                        content=cls._flatten_tool_result_content(blk.get("content")),
                        tool_call_id=tool_use_id,
                        outcome=outcome,
                    )
                )

            else:
                # Unknown user-side block — stash + emit JSON-string placeholder.
                if raw_extras is not None:
                    raw_extras[f"unknown_block:msg:{msg_index}:idx:{block_index}"] = dict(blk)
                accumulated.append(json.dumps(dict(blk)))

        flush()

    @classmethod
    def _load_assistant_turn(  # noqa: PLR0912
        cls,
        msg: BetaMessageParam,
        builder: MessagesBuilder,
        *,
        msg_index: int = 0,
        raw_extras: dict[str, Any] | None = None,
    ) -> None:
        """Process one Anthropic assistant turn into response parts."""
        content = msg.get("content")
        if isinstance(content, str):
            builder.add(TextPart(content=content))
            return
        if not isinstance(content, list):
            builder.add(TextPart(content=""))
            return

        emitted = False
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                if raw_extras is not None:
                    raw_extras[f"unknown_block:msg:{msg_index}:idx:{block_index}"] = block
                builder.add(TextPart(content=json.dumps(block)))
                emitted = True
                continue

            blk = cast(Mapping[str, Any], block)
            btype = blk.get("type")

            if btype == "text":
                builder.add(TextPart(content=blk["text"]))
                emitted = True

            elif btype == "thinking":
                builder.add(
                    ThinkingPart(
                        content=blk["thinking"],
                        signature=blk["signature"],
                        provider_name="anthropic",
                    )
                )
                emitted = True

            elif btype == "redacted_thinking":
                builder.add(
                    ThinkingPart(
                        id="redacted_thinking",
                        content="",
                        signature=blk["data"],
                        provider_name="anthropic",
                    )
                )
                emitted = True

            elif btype == "tool_use":
                builder.add(
                    ToolCallPart(
                        tool_name=blk["name"],
                        args=cast(dict[str, Any], blk["input"]),
                        tool_call_id=blk["id"],
                    )
                )
                emitted = True

            else:
                if raw_extras is not None:
                    raw_extras[f"unknown_block:msg:{msg_index}:idx:{block_index}"] = dict(blk)
                builder.add(TextPart(content=json.dumps(dict(blk))))
                emitted = True

        if not emitted:
            builder.add(TextPart(content=""))

    # ── source helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_image(source: Mapping[str, Any]) -> UserContent:
        stype = source.get("type", "base64")
        if stype == "url":
            url = source.get("url", "")
            return ImageUrl(url=url, media_type=source.get("media_type")) if url else ""
        if stype == "file":
            return UploadedFile(
                file_id=source["file_id"],
                provider_name="anthropic",
                media_type=source.get("media_type") or "image/jpeg",
            )
        # default / "base64" — lenient: malformed base64 falls back to raw bytes
        # so a single bad image doesn't crash the whole load.
        media_type = source.get("media_type", "application/octet-stream")
        data_field = source.get("data", "")
        if isinstance(data_field, bytes):
            data_bytes = data_field
        elif data_field:
            try:
                data_bytes = base64.b64decode(data_field)
            except (ValueError, binascii.Error):
                data_bytes = data_field.encode("utf-8") if isinstance(data_field, str) else b""
        else:
            data_bytes = b""
        return BinaryContent(data=data_bytes, media_type=media_type)

    @staticmethod
    def _load_document(source: Mapping[str, Any], *, media_type: str | None) -> UserContent:
        stype = source.get("type")
        if stype == "url":
            return DocumentUrl(url=source["url"], media_type=media_type)
        elif stype == "base64":
            return BinaryContent(
                data=base64.b64decode(source["data"]),
                media_type=source["media_type"],
            )
        elif stype == "file":
            return UploadedFile(
                file_id=source["file_id"],
                provider_name="anthropic",
                media_type=source.get("media_type") or media_type or "application/octet-stream",
            )
        raise ValueError(f"Unknown document source type: {stype!r}")

    @staticmethod
    def _flatten_tool_result_content(content: Any) -> str:
        """Reduce tool_result content to a plain string.

        Anthropic allows ``content`` to be a list of text/image blocks; we
        extract the text parts and join them. Image blocks in tool results
        are dropped.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return "\n".join(b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text")

    @staticmethod
    def _cache_ttl(cache_control: Mapping[str, Any]) -> Literal["5m", "1h"]:
        ttl = cache_control.get("ttl")
        return ttl if ttl in ("5m", "1h") else "5m"

    # ── dump (IR → wire) ─────────────────────────────────────────────────────

    @classmethod
    def dump_system(cls, messages: Sequence[ModelMessage]) -> str | list[BetaTextBlockParam] | None:
        """Extract the system prompt from IR in Anthropic ``system`` format.

        A single bare ``SystemPromptPart`` becomes a plain string. Multiple
        parts, or any part with a following sentinel ``UserPromptPart([CachePoint])``,
        become a block array.
        """
        blocks: list[BetaTextBlockParam] = []
        parts = [p for m in messages if isinstance(m, ModelRequest) for p in m.parts]

        i = 0
        while i < len(parts):
            part = parts[i]
            if isinstance(part, SystemPromptPart):
                block: BetaTextBlockParam = {"type": "text", "text": part.content}
                if i + 1 < len(parts):
                    nxt = parts[i + 1]
                    if (
                        isinstance(nxt, UserPromptPart)
                        and isinstance(nxt.content, list)
                        and len(nxt.content) == 1
                        and isinstance(nxt.content[0], CachePoint)
                    ):
                        block["cache_control"] = {
                            "type": "ephemeral",
                            "ttl": nxt.content[0].ttl,
                        }
                        i += 1
                blocks.append(block)
            i += 1

        if not blocks:
            return None
        if len(blocks) == 1 and "cache_control" not in blocks[0]:
            return blocks[0]["text"]
        return blocks

    @classmethod
    def dump_messages(cls, messages: Sequence[ModelMessage]) -> list[BetaMessageParam]:
        """Convert IR to Anthropic conversation turns only.

        Call :meth:`dump_system` separately to extract the top-level ``system``
        field.
        """
        result: list[BetaMessageParam] = []
        # Skip sentinel UserPromptPart([CachePoint]) used as system-cache markers.
        for message in messages:
            if isinstance(message, ModelRequest):
                if (msg := cls._dump_request(message)) is not None:
                    result.append(msg)
            elif isinstance(message, ModelResponse) and (msg := cls._dump_response(message)) is not None:
                result.append(msg)
        return result

    @staticmethod
    def _dump_request(message: ModelRequest) -> BetaMessageParam | None:
        blocks: list[BetaContentBlockParam] = []

        def apply_cache_control(ttl: Literal["5m", "1h"]) -> None:
            if blocks:
                cast(dict[str, Any], blocks[-1])["cache_control"] = {
                    "type": "ephemeral",
                    "ttl": ttl,
                }

        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                # System prompt is dumped via dump_system, not here.
                continue

            elif isinstance(part, UserPromptPart):
                content = part.content
                # Skip sentinel UserPromptPart([CachePoint]) markers used by dump_system.
                if isinstance(content, list) and len(content) == 1 and isinstance(content[0], CachePoint):
                    continue
                if isinstance(content, str):
                    blocks.append({"type": "text", "text": content})
                else:
                    for item in content:
                        if isinstance(item, str):
                            blocks.append({"type": "text", "text": item})
                        elif isinstance(item, CachePoint):
                            apply_cache_control(item.ttl)
                        elif isinstance(item, BinaryContent):
                            source = {
                                "type": "base64",
                                "media_type": item.media_type,
                                "data": item.base64,
                            }
                            if item.is_image:
                                blocks.append(
                                    cast(
                                        BetaImageBlockParam,
                                        {"type": "image", "source": source},
                                    )
                                )
                            else:
                                blocks.append(
                                    cast(
                                        BetaContentBlockParam,
                                        {
                                            "type": "document",
                                            "source": source,
                                            "media_type": item.media_type,
                                        },
                                    )
                                )
                        elif isinstance(item, ImageUrl):
                            blocks.append(
                                cast(
                                    BetaImageBlockParam,
                                    {
                                        "type": "image",
                                        "source": {"type": "url", "url": item.url},
                                    },
                                )
                            )
                        elif isinstance(item, DocumentUrl):
                            blocks.append(
                                cast(
                                    BetaContentBlockParam,
                                    {
                                        "type": "document",
                                        "source": {"type": "url", "url": item.url},
                                        "media_type": item.media_type or "application/octet-stream",
                                    },
                                )
                            )
                        elif isinstance(item, AudioUrl):
                            # Anthropic Messages API has no audio block.
                            pass
                        elif isinstance(item, UploadedFile) and item.provider_name == "anthropic":
                            media = item.media_type or "application/octet-stream"
                            file_src = {
                                "type": "file",
                                "file_id": item.file_id,
                                "media_type": media,
                            }
                            kind = "image" if media.startswith("image/") else "document"
                            blk: dict[str, Any] = {"type": kind, "source": file_src}
                            if kind == "document":
                                blk["media_type"] = media
                            blocks.append(cast(BetaContentBlockParam, blk))

            elif isinstance(part, ToolReturnPart):
                tr: BetaToolResultBlockParam = {
                    "type": "tool_result",
                    "tool_use_id": part.tool_call_id,
                    "content": part.model_response_str(),
                }
                if part.outcome == "failed":
                    tr["is_error"] = True
                blocks.append(tr)

            elif isinstance(part, RetryPromptPart):
                if part.tool_name is not None:
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": part.tool_call_id,
                            "content": part.model_response(),
                            "is_error": True,
                        }
                    )
                else:
                    blocks.append({"type": "text", "text": part.model_response()})

        if not blocks:
            return None
        return {"role": "user", "content": blocks}

    @staticmethod
    def _dump_response(message: ModelResponse) -> BetaMessageParam | None:
        blocks: list[BetaContentBlockParam] = []

        for part in message.parts:
            if isinstance(part, TextPart):
                blocks.append({"type": "text", "text": part.content})

            elif isinstance(part, ThinkingPart):
                if part.id == "redacted_thinking":
                    blocks.append(
                        cast(
                            BetaRedactedThinkingBlockParam,
                            {
                                "type": "redacted_thinking",
                                "data": part.signature or "",
                            },
                        )
                    )
                else:
                    blocks.append(
                        {
                            "type": "thinking",
                            "thinking": part.content,
                            "signature": part.signature or "",
                        }
                    )

            elif isinstance(part, ToolCallPart):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": part.tool_call_id,
                        "name": part.tool_name,
                        "input": part.args_as_dict(),
                    }
                )

        if not blocks:
            return None
        return {"role": "assistant", "content": blocks}

    def build_event_stream(
        self,
    ) -> UIEventStream[MessageCreateParamsBase, Any, AgentDepsT, OutputDataT]:
        raise NotImplementedError("Implement a UIEventStream subclass to produce Anthropic SSE events.")
