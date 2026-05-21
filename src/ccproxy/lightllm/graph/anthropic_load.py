"""Parse an Anthropic Messages API request body to :class:`ParsedRequest` via FSM.

Inverse of :mod:`ccproxy.lightllm.graph.anthropic_dump`. Replaces the imperative
:mod:`ccproxy.lightllm.anthropic_inbound` parser with two per-message FSMs:

* ``_user_turn_graph`` walks a user-role message's content blocks, accumulating
  text / image / document items into a :class:`UserPromptPart` content list,
  and flushing the accumulator into a standalone :class:`ToolReturnPart` when a
  ``tool_result`` block interrupts it.
* ``_assistant_turn_graph`` walks an assistant-role message's content blocks,
  emitting one :class:`ModelResponsePart` per block.

The imperative envelope wrapper :func:`load_anthropic` handles tool_name two-pass
pre-scan, system extraction (with uniform-cache compression to
``settings['anthropic_cache_instructions']``), tools extraction (uniform-cache
compression to ``settings['anthropic_cache_tool_definitions']``), and raw_extras
accumulation. ``raw_extras`` keys mirror the legacy parser's conventions:

* ``cc:msg:{i}:block:{j}`` — non-standard cache_control TTL (anything but ``5m``/``1h``)
* ``unknown_block:msg:{i}:idx:{j}`` — unknown content block type
* ``system`` — non-uniform system cache_control (whole raw blocks list)
* ``tools`` — non-uniform tools cache_control (whole raw tools list)
* ``metadata`` — always preserved
* Any other unmodelled top-level wire field — copied verbatim under its wire name.
"""

from __future__ import annotations

import base64
import json
import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic_ai.messages import (
    BinaryContent,
    CachePoint,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from ccproxy.lightllm.parsed import ParsedRequest

logger = logging.getLogger(__name__)

# pydantic-ai's CachePoint accepts only these two TTLs (Literal['5m', '1h']).
_SUPPORTED_TTLS: frozenset[str] = frozenset({"5m", "1h"})

# Top-level Anthropic body fields the IR + ModelSettings absorb. Anything else
# in the body gets parked in ``raw_extras`` keyed by its wire name.
_ABSORBED_TOP_LEVEL: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "system",
        "tools",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "stream",
        "metadata",
    }
)


# ── User-turn FSM ──────────────────────────────────────────────────────────


@dataclass
class _UserTurnState:
    """State for one user (or system-role) message's load FSM.

    ``parts`` accumulates the final IR parts list. ``accumulator`` holds
    in-flight ``UserContent`` items for a :class:`UserPromptPart` that's still
    being built; it is flushed into ``parts`` either when a ``tool_result``
    block interrupts it or when the queue runs dry.
    """

    queue: deque[tuple[int, Any]] = field(default_factory=deque)
    parts: list[SystemPromptPart | UserPromptPart | ToolReturnPart] = field(default_factory=list)
    accumulator: list[UserContent] = field(default_factory=list)
    tool_name_lookup: dict[str, str] = field(default_factory=dict)
    msg_index: int = 0
    raw_extras: dict[str, Any] = field(default_factory=dict)


def _flush_accumulator(state: _UserTurnState) -> None:
    """Move in-flight content items into a ``UserPromptPart`` and clear the buffer."""
    if state.accumulator:
        state.parts.append(UserPromptPart(content=list(state.accumulator)))
        state.accumulator = []


def _emit_cache_control(
    cc: Any, *, items: list[UserContent], msg_index: int, block_index: int, raw_extras: dict[str, Any]
) -> None:
    """Append a :class:`CachePoint` after the just-added content item.

    Wire ``ttl`` values pydantic-ai cannot represent (anything other than ``5m``
    or ``1h``) are stashed in ``raw_extras`` and the IR marker is skipped.
    """
    if not isinstance(cc, dict):
        return
    cc_dict = cast(dict[str, Any], cc)
    ttl = cc_dict.get("ttl", "5m")
    if ttl in _SUPPORTED_TTLS:
        items.append(CachePoint(ttl=ttl))
        return
    raw_extras[f"cc:msg:{msg_index}:block:{block_index}"] = cc_dict


@dataclass
class FetchNextUserBlockNode(
    BaseNode[_UserTurnState, None, list[SystemPromptPart | UserPromptPart | ToolReturnPart]]
):
    """Pop the next content block from the user-turn queue and dispatch by ``type``."""

    async def run(
        self, ctx: GraphRunContext[_UserTurnState, None]
    ) -> (
        BaseNode[_UserTurnState, None, Any]
        | End[list[SystemPromptPart | UserPromptPart | ToolReturnPart]]
    ):
        if not ctx.state.queue:
            _flush_accumulator(ctx.state)
            return End(ctx.state.parts)

        block_index, raw_block = ctx.state.queue.popleft()
        if not isinstance(raw_block, dict):
            ctx.state.accumulator.append(json.dumps(raw_block))
            ctx.state.raw_extras[
                f"unknown_block:msg:{ctx.state.msg_index}:idx:{block_index}"
            ] = raw_block
            return FetchNextUserBlockNode()

        block: dict[str, Any] = raw_block

        match block.get("type", ""):
            case "text":
                return ParseUserTextNode(block_index=block_index, block=block)
            case "image":
                return ParseUserImageNode(block_index=block_index, block=block)
            case "tool_result":
                return ParseUserToolResultNode(block_index=block_index, block=block)
            case _:
                return ParseUserUnknownBlockNode(block_index=block_index, block=block)


@dataclass
class ParseUserTextNode(BaseNode[_UserTurnState, None]):
    """Append a text block's text to the accumulator and emit a CachePoint if applicable."""

    block_index: int
    block: dict[str, Any]

    async def run(self, ctx: GraphRunContext[_UserTurnState, None]) -> BaseNode[_UserTurnState, None, Any]:
        ctx.state.accumulator.append(self.block.get("text", ""))
        _emit_cache_control(
            self.block.get("cache_control"),
            items=ctx.state.accumulator,
            msg_index=ctx.state.msg_index,
            block_index=self.block_index,
            raw_extras=ctx.state.raw_extras,
        )
        return FetchNextUserBlockNode()


@dataclass
class ParseUserImageNode(BaseNode[_UserTurnState, None]):
    """Append an image block's payload (``BinaryContent`` or ``ImageUrl``) to the accumulator."""

    block_index: int
    block: dict[str, Any]

    async def run(self, ctx: GraphRunContext[_UserTurnState, None]) -> BaseNode[_UserTurnState, None, Any]:
        ctx.state.accumulator.append(_parse_image_source(self.block.get("source") or {}))
        _emit_cache_control(
            self.block.get("cache_control"),
            items=ctx.state.accumulator,
            msg_index=ctx.state.msg_index,
            block_index=self.block_index,
            raw_extras=ctx.state.raw_extras,
        )
        return FetchNextUserBlockNode()


@dataclass
class ParseUserToolResultNode(BaseNode[_UserTurnState, None]):
    """Flush the accumulator and emit a ``ToolReturnPart``.

    ``tool_name`` is resolved via the pre-scanned ``tool_name_lookup``; an
    orphan ``tool_use_id`` (no matching assistant ``tool_use``) leaves
    ``tool_name`` empty and logs a debug warning, matching the legacy parser.
    """

    block_index: int
    block: dict[str, Any]

    async def run(self, ctx: GraphRunContext[_UserTurnState, None]) -> BaseNode[_UserTurnState, None, Any]:
        _flush_accumulator(ctx.state)

        raw_content = self.block.get("content", "")
        if isinstance(raw_content, list):
            texts = [
                b.get("text", "") for b in raw_content if isinstance(b, dict) and b.get("type") == "text"
            ]
            content: Any = "\n".join(texts) if texts else str(raw_content)
        else:
            content = raw_content

        tool_use_id = self.block.get("tool_use_id", "")
        tool_name = ctx.state.tool_name_lookup.get(tool_use_id, "")
        if not tool_name and tool_use_id:
            logger.debug(
                "anthropic load: tool_result references unknown tool_use_id %r — leaving tool_name blank",
                tool_use_id,
            )

        ctx.state.parts.append(
            ToolReturnPart(tool_name=tool_name, content=content, tool_call_id=tool_use_id)
        )
        return FetchNextUserBlockNode()


@dataclass
class ParseUserUnknownBlockNode(BaseNode[_UserTurnState, None]):
    """Stash an unknown user-side block in ``raw_extras`` and feed its JSON into the accumulator."""

    block_index: int
    block: dict[str, Any]

    async def run(self, ctx: GraphRunContext[_UserTurnState, None]) -> BaseNode[_UserTurnState, None, Any]:
        ctx.state.accumulator.append(json.dumps(self.block))
        ctx.state.raw_extras[
            f"unknown_block:msg:{ctx.state.msg_index}:idx:{self.block_index}"
        ] = self.block
        return FetchNextUserBlockNode()


_user_turn_graph = Graph[
    _UserTurnState, None, list[SystemPromptPart | UserPromptPart | ToolReturnPart]
](
    nodes=(
        FetchNextUserBlockNode,
        ParseUserTextNode,
        ParseUserImageNode,
        ParseUserToolResultNode,
        ParseUserUnknownBlockNode,
    ),
)


# ── Assistant-turn FSM ─────────────────────────────────────────────────────


@dataclass
class _AssistantTurnState:
    """State for one assistant message's load FSM."""

    queue: deque[tuple[int, Any]] = field(default_factory=deque)
    parts: list[ModelResponsePart] = field(default_factory=list)
    msg_index: int = 0
    raw_extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class FetchNextAssistantBlockNode(BaseNode[_AssistantTurnState, None, list[ModelResponsePart]]):
    """Pop the next content block from the assistant-turn queue and dispatch by ``type``."""

    async def run(
        self, ctx: GraphRunContext[_AssistantTurnState, None]
    ) -> BaseNode[_AssistantTurnState, None, Any] | End[list[ModelResponsePart]]:
        if not ctx.state.queue:
            # Empty assistant content gets a sentinel empty TextPart so the
            # resulting ModelResponse has at least one part (matches legacy
            # parser behavior + downstream pydantic-ai expectations).
            if not ctx.state.parts:
                ctx.state.parts.append(TextPart(content=""))
            return End(ctx.state.parts)

        block_index, raw_block = ctx.state.queue.popleft()
        if not isinstance(raw_block, dict):
            ctx.state.parts.append(TextPart(content=json.dumps(raw_block)))
            ctx.state.raw_extras[
                f"unknown_block:msg:{ctx.state.msg_index}:idx:{block_index}"
            ] = raw_block
            return FetchNextAssistantBlockNode()

        block: dict[str, Any] = raw_block

        match block.get("type", ""):
            case "text":
                return ParseAssistantTextNode(block=block)
            case "tool_use":
                return ParseAssistantToolUseNode(block=block)
            case "thinking":
                return ParseAssistantThinkingNode(block=block)
            case "redacted_thinking":
                return ParseAssistantRedactedThinkingNode(block=block)
            case _:
                return ParseAssistantUnknownBlockNode(block_index=block_index, block=block)


@dataclass
class ParseAssistantTextNode(BaseNode[_AssistantTurnState, None]):
    """Emit a :class:`TextPart` from an assistant text block."""

    block: dict[str, Any]

    async def run(
        self, ctx: GraphRunContext[_AssistantTurnState, None]
    ) -> BaseNode[_AssistantTurnState, None, Any]:
        ctx.state.parts.append(TextPart(content=self.block.get("text", "")))
        return FetchNextAssistantBlockNode()


@dataclass
class ParseAssistantToolUseNode(BaseNode[_AssistantTurnState, None]):
    """Emit a :class:`ToolCallPart` from an assistant tool_use block."""

    block: dict[str, Any]

    async def run(
        self, ctx: GraphRunContext[_AssistantTurnState, None]
    ) -> BaseNode[_AssistantTurnState, None, Any]:
        ctx.state.parts.append(
            ToolCallPart(
                tool_name=self.block.get("name", ""),
                args=self.block.get("input"),
                tool_call_id=self.block.get("id", ""),
            )
        )
        return FetchNextAssistantBlockNode()


@dataclass
class ParseAssistantThinkingNode(BaseNode[_AssistantTurnState, None]):
    """Emit a :class:`ThinkingPart` from a thinking block."""

    block: dict[str, Any]

    async def run(
        self, ctx: GraphRunContext[_AssistantTurnState, None]
    ) -> BaseNode[_AssistantTurnState, None, Any]:
        ctx.state.parts.append(
            ThinkingPart(content=self.block.get("thinking", ""), signature=self.block.get("signature"))
        )
        return FetchNextAssistantBlockNode()


@dataclass
class ParseAssistantRedactedThinkingNode(BaseNode[_AssistantTurnState, None]):
    """Emit a :class:`ThinkingPart` with id=``redacted_thinking`` carrying opaque ciphertext."""

    block: dict[str, Any]

    async def run(
        self, ctx: GraphRunContext[_AssistantTurnState, None]
    ) -> BaseNode[_AssistantTurnState, None, Any]:
        ctx.state.parts.append(
            ThinkingPart(
                content="",
                id="redacted_thinking",
                signature=self.block.get("data"),
            )
        )
        return FetchNextAssistantBlockNode()


@dataclass
class ParseAssistantUnknownBlockNode(BaseNode[_AssistantTurnState, None]):
    """Stash unknown assistant blocks in raw_extras and feed JSON into a TextPart."""

    block_index: int
    block: dict[str, Any]

    async def run(
        self, ctx: GraphRunContext[_AssistantTurnState, None]
    ) -> BaseNode[_AssistantTurnState, None, Any]:
        ctx.state.parts.append(TextPart(content=json.dumps(self.block)))
        ctx.state.raw_extras[
            f"unknown_block:msg:{ctx.state.msg_index}:idx:{self.block_index}"
        ] = self.block
        return FetchNextAssistantBlockNode()


_assistant_turn_graph = Graph[_AssistantTurnState, None, list[ModelResponsePart]](
    nodes=(
        FetchNextAssistantBlockNode,
        ParseAssistantTextNode,
        ParseAssistantToolUseNode,
        ParseAssistantThinkingNode,
        ParseAssistantRedactedThinkingNode,
        ParseAssistantUnknownBlockNode,
    ),
)


# ── Source helpers (imperative — these are NOT FSM nodes) ──────────────────


def _parse_image_source(source: dict[str, Any]) -> UserContent:
    """Parse an Anthropic ``image`` block's ``source`` into a ``BinaryContent`` / ``ImageUrl``."""
    source_type = source.get("type", "base64")
    media_type = source.get("media_type", "application/octet-stream")

    if source_type == "url":
        url = source.get("url", "")
        return ImageUrl(url=url, media_type=media_type) if url else ""

    data_field = source.get("data", "")
    if isinstance(data_field, bytes):
        data_bytes = data_field
    else:
        try:
            data_bytes = base64.b64decode(data_field) if data_field else b""
        except (ValueError, TypeError):
            data_bytes = data_field.encode("utf-8") if isinstance(data_field, str) else b""

    return BinaryContent(data=data_bytes, media_type=media_type)


def _build_tool_name_lookup(raw_messages: Sequence[Any]) -> dict[str, str]:
    """Walk assistant messages to build a ``tool_use_id -> tool_name`` index."""
    lookup: dict[str, str] = {}
    for msg in raw_messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_id = block.get("id", "")
                if tool_id:
                    lookup[tool_id] = block.get("name", "")
    return lookup


# ── System + tools + settings (imperative envelope helpers) ────────────────


def _parse_system(
    raw_system: Any, *, settings: ModelSettings, raw_extras: dict[str, Any]
) -> list[SystemPromptPart]:
    """Parse the top-level ``system`` field into :class:`SystemPromptPart` entries.

    Uniform cache_control across blocks lifts to
    ``settings['anthropic_cache_instructions']``; non-uniform blocks land in
    ``raw_extras['system']`` for the outbound renderer to override.
    """
    if raw_system is None:
        return []
    if isinstance(raw_system, str):
        return [SystemPromptPart(content=raw_system)] if raw_system else []
    if not isinstance(raw_system, list):
        return []

    parts: list[SystemPromptPart] = []
    cache_ttls: list[str | None] = []
    for block in raw_system:
        if not isinstance(block, dict):
            continue
        parts.append(SystemPromptPart(content=block.get("text", "")))
        cc = block.get("cache_control")
        cache_ttls.append(cc.get("ttl", "5m") if isinstance(cc, dict) else None)

    cached_ttls = {ttl for ttl in cache_ttls if ttl is not None}
    if not cached_ttls:
        return parts

    if len(cached_ttls) == 1:
        only_ttl = next(iter(cached_ttls))
        if all(t is not None for t in cache_ttls) and only_ttl in _SUPPORTED_TTLS:
            cast(dict[str, Any], settings)["anthropic_cache_instructions"] = only_ttl
            return parts

    raw_extras["system"] = raw_system
    return parts


def _parse_tools(
    raw_tools: Sequence[Any], *, settings: ModelSettings
) -> tuple[list[ToolDefinition], bool]:
    """Parse Anthropic tool definitions.

    Returns the parsed tools and a flag indicating whether tools cache_control
    was non-uniform (the caller stashes the raw list in ``raw_extras['tools']``).
    """
    tools: list[ToolDefinition] = []
    cache_ttls: list[str | None] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        tools.append(
            ToolDefinition(
                name=tool.get("name", ""),
                description=tool.get("description"),
                parameters_json_schema=tool.get("input_schema") or {},
            )
        )
        cc = tool.get("cache_control")
        cache_ttls.append(cc.get("ttl", "5m") if isinstance(cc, dict) else None)

    cached_ttls = {ttl for ttl in cache_ttls if ttl is not None}
    if not cached_ttls:
        return tools, False
    if len(cached_ttls) == 1:
        only_ttl = next(iter(cached_ttls))
        if all(t is not None for t in cache_ttls) and only_ttl in _SUPPORTED_TTLS:
            cast(dict[str, Any], settings)["anthropic_cache_tool_definitions"] = only_ttl
            return tools, False
    return tools, True


def _build_settings(body: dict[str, Any], *, raw_extras: dict[str, Any]) -> ModelSettings:
    """Extract sampling + behavior settings from the wire body.

    ``metadata`` has no ``ModelSettings`` slot — preserved in ``raw_extras``.
    """
    settings: dict[str, Any] = {}
    for key in ("max_tokens", "temperature", "top_p", "stop_sequences", "top_k"):
        if key in body:
            settings[key] = body[key]
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        raw_extras["metadata"] = metadata
    return cast(ModelSettings, settings)


def _attach_system_prompts(
    messages: list[ModelMessage], system_parts: list[SystemPromptPart]
) -> list[ModelMessage]:
    """Prepend ``system_parts`` to the first ``ModelRequest`` in ``messages``."""
    if not system_parts:
        return messages
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest):
            new_parts: list[Any] = [*system_parts, *msg.parts]
            messages[i] = ModelRequest(parts=new_parts)
            return messages
    return [ModelRequest(parts=list(system_parts)), *messages]


# ── Per-message FSM drivers ────────────────────────────────────────────────


async def _load_user_message(
    content: Any, *, msg_index: int, role: str, tool_name_lookup: dict[str, str], raw_extras: dict[str, Any]
) -> ModelRequest:
    """Parse one user/system role message into a :class:`ModelRequest`."""
    if isinstance(content, str):
        if role == "system":
            return ModelRequest(parts=[SystemPromptPart(content=content)])
        return ModelRequest(parts=[UserPromptPart(content=content)])

    if not isinstance(content, list):
        return ModelRequest(parts=[])

    queue: deque[tuple[int, Any]] = deque(enumerate(content))
    state = _UserTurnState(
        queue=queue,
        tool_name_lookup=tool_name_lookup,
        msg_index=msg_index,
        raw_extras=raw_extras,
    )
    result = await _user_turn_graph.run(FetchNextUserBlockNode(), state=state)
    return ModelRequest(parts=list(result.output))


async def _load_assistant_message(
    content: Any, *, msg_index: int, raw_extras: dict[str, Any]
) -> ModelResponse:
    """Parse one assistant role message into a :class:`ModelResponse`."""
    if isinstance(content, str):
        return ModelResponse(parts=[TextPart(content=content)])
    if not isinstance(content, list):
        return ModelResponse(parts=[TextPart(content="")])

    queue: deque[tuple[int, Any]] = deque(enumerate(content))
    state = _AssistantTurnState(queue=queue, msg_index=msg_index, raw_extras=raw_extras)
    result = await _assistant_turn_graph.run(FetchNextAssistantBlockNode(), state=state)
    return ModelResponse(parts=list(result.output))


async def _load_messages(
    raw_messages: Sequence[Any], *, tool_name_lookup: dict[str, str], raw_extras: dict[str, Any]
) -> list[ModelMessage]:
    """Walk wire messages, dispatching each to the right per-message FSM."""
    result: list[ModelMessage] = []
    for i, msg in enumerate(raw_messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            result.append(await _load_assistant_message(content, msg_index=i, raw_extras=raw_extras))
        else:
            result.append(
                await _load_user_message(
                    content,
                    msg_index=i,
                    role=role,
                    tool_name_lookup=tool_name_lookup,
                    raw_extras=raw_extras,
                )
            )
    return result


# ── Public entrypoint ──────────────────────────────────────────────────────


async def load_anthropic(body: dict[str, Any]) -> ParsedRequest:
    """Parse an Anthropic Messages API request body into the IR via the FSM.

    Drop-in replacement for
    :func:`ccproxy.lightllm.anthropic_inbound.parse_anthropic_messages`.
    """
    raw_extras: dict[str, Any] = {}

    model = str(body.get("model", ""))
    stream = bool(body.get("stream", False))

    raw_messages = body.get("messages") or []
    tool_name_lookup = _build_tool_name_lookup(raw_messages)
    messages = await _load_messages(
        raw_messages, tool_name_lookup=tool_name_lookup, raw_extras=raw_extras
    )

    settings = _build_settings(body, raw_extras=raw_extras)

    raw_tools = body.get("tools") or []
    function_tools, has_mixed_cache = _parse_tools(raw_tools, settings=settings)
    if has_mixed_cache:
        raw_extras["tools"] = raw_tools
    request_parameters = ModelRequestParameters(function_tools=function_tools)

    system_parts = _parse_system(body.get("system"), settings=settings, raw_extras=raw_extras)
    if system_parts:
        messages = _attach_system_prompts(messages, system_parts)

    for key, value in body.items():
        if key in _ABSORBED_TOP_LEVEL:
            continue
        raw_extras.setdefault(key, value)

    return ParsedRequest(
        model=model,
        messages=messages,
        request_parameters=request_parameters,
        settings=settings,
        stream=stream,
        raw_extras=raw_extras,
    )
