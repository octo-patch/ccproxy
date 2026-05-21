"""Parse an Anthropic Messages API request body to :class:`ParsedRequest` via FSM.

Inverse of :mod:`ccproxy.lightllm.graph.anthropic_dump`. Replaces the imperative
:mod:`ccproxy.lightllm.anthropic_inbound` parser with two per-message FSMs
built atop :mod:`pydantic_graph.beta`'s ``GraphBuilder``:

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
from pydantic_graph.beta import GraphBuilder, StepContext

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


class _UserDone:
    """Marker for end of the user-turn queue."""


@dataclass
class _UserBlock:
    """A typed user-turn dispatch envelope keyed by block ``type``."""

    block_index: int
    block: dict[str, Any]


@dataclass
class _UserTextBlock(_UserBlock):
    pass


@dataclass
class _UserImageBlock(_UserBlock):
    pass


@dataclass
class _UserToolResultBlock(_UserBlock):
    pass


@dataclass
class _UserUnknownBlock(_UserBlock):
    pass


@dataclass
class _UserNonDictBlock:
    """A non-dict queue item (e.g. raw string fed in directly)."""

    block_index: int
    raw: Any


def _flush_accumulator(state: _UserTurnState) -> None:
    """Move in-flight content items into a ``UserPromptPart`` and clear the buffer."""
    if state.accumulator:
        state.parts.append(UserPromptPart(content=list(state.accumulator)))
        state.accumulator = []


def _emit_cache_control(
    cc: Any, *, items: list[UserContent], msg_index: int, block_index: int, raw_extras: dict[str, Any]
) -> None:
    """Append a :class:`CachePoint` after the just-added content item."""
    if not isinstance(cc, dict):
        return
    cc_dict = cast(dict[str, Any], cc)
    ttl = cc_dict.get("ttl", "5m")
    if ttl in _SUPPORTED_TTLS:
        items.append(CachePoint(ttl=ttl))
        return
    raw_extras[f"cc:msg:{msg_index}:block:{block_index}"] = cc_dict


_ug: GraphBuilder[
    _UserTurnState, None, None, list[SystemPromptPart | UserPromptPart | ToolReturnPart]
] = GraphBuilder(
    state_type=_UserTurnState,
    output_type=list[SystemPromptPart | UserPromptPart | ToolReturnPart],
)


@_ug.step
async def user_take_next(ctx: StepContext[_UserTurnState, None, None]) -> Any:
    """Router source: pop the next block and dispatch by ``type``."""
    if not ctx.state.queue:
        return _UserDone()
    block_index, raw_block = ctx.state.queue.popleft()
    if not isinstance(raw_block, dict):
        return _UserNonDictBlock(block_index=block_index, raw=raw_block)
    block: dict[str, Any] = raw_block
    block_type = block.get("type", "")
    if block_type == "text":
        return _UserTextBlock(block_index=block_index, block=block)
    if block_type == "image":
        return _UserImageBlock(block_index=block_index, block=block)
    if block_type == "tool_result":
        return _UserToolResultBlock(block_index=block_index, block=block)
    return _UserUnknownBlock(block_index=block_index, block=block)


@_ug.step
async def user_parse_text(ctx: StepContext[_UserTurnState, None, _UserTextBlock]) -> None:
    """Append a text block's text and emit a CachePoint if applicable."""
    payload = ctx.inputs
    ctx.state.accumulator.append(payload.block.get("text", ""))
    _emit_cache_control(
        payload.block.get("cache_control"),
        items=ctx.state.accumulator,
        msg_index=ctx.state.msg_index,
        block_index=payload.block_index,
        raw_extras=ctx.state.raw_extras,
    )


@_ug.step
async def user_parse_image(ctx: StepContext[_UserTurnState, None, _UserImageBlock]) -> None:
    """Append an image block's payload (``BinaryContent`` or ``ImageUrl``)."""
    payload = ctx.inputs
    ctx.state.accumulator.append(_parse_image_source(payload.block.get("source") or {}))
    _emit_cache_control(
        payload.block.get("cache_control"),
        items=ctx.state.accumulator,
        msg_index=ctx.state.msg_index,
        block_index=payload.block_index,
        raw_extras=ctx.state.raw_extras,
    )


@_ug.step
async def user_parse_tool_result(
    ctx: StepContext[_UserTurnState, None, _UserToolResultBlock],
) -> None:
    """Flush the accumulator and emit a ``ToolReturnPart``."""
    payload = ctx.inputs
    _flush_accumulator(ctx.state)

    raw_content = payload.block.get("content", "")
    if isinstance(raw_content, list):
        texts = [
            b.get("text", "") for b in raw_content if isinstance(b, dict) and b.get("type") == "text"
        ]
        content: Any = "\n".join(texts) if texts else str(raw_content)
    else:
        content = raw_content

    tool_use_id = payload.block.get("tool_use_id", "")
    tool_name = ctx.state.tool_name_lookup.get(tool_use_id, "")
    if not tool_name and tool_use_id:
        logger.debug(
            "anthropic load: tool_result references unknown tool_use_id %r — leaving tool_name blank",
            tool_use_id,
        )

    ctx.state.parts.append(
        ToolReturnPart(tool_name=tool_name, content=content, tool_call_id=tool_use_id)
    )


@_ug.step
async def user_parse_unknown(
    ctx: StepContext[_UserTurnState, None, _UserUnknownBlock],
) -> None:
    """Stash an unknown user-side block in ``raw_extras`` and feed its JSON into the accumulator."""
    payload = ctx.inputs
    ctx.state.accumulator.append(json.dumps(payload.block))
    ctx.state.raw_extras[
        f"unknown_block:msg:{ctx.state.msg_index}:idx:{payload.block_index}"
    ] = payload.block


@_ug.step
async def user_parse_non_dict(
    ctx: StepContext[_UserTurnState, None, _UserNonDictBlock],
) -> None:
    """Coerce a non-dict block to its JSON string and stash the raw value."""
    payload = ctx.inputs
    ctx.state.accumulator.append(json.dumps(payload.raw))
    ctx.state.raw_extras[
        f"unknown_block:msg:{ctx.state.msg_index}:idx:{payload.block_index}"
    ] = payload.raw


@_ug.step
async def user_emit(
    ctx: StepContext[_UserTurnState, None, _UserDone],
) -> list[SystemPromptPart | UserPromptPart | ToolReturnPart]:
    """Terminal step — flush the trailing accumulator and return all parts."""
    _flush_accumulator(ctx.state)
    return ctx.state.parts


_ug.add(
    _ug.edge_from(_ug.start_node).to(user_take_next),
    _ug.edge_from(user_take_next).to(
        _ug.decision()
        .branch(_ug.match(_UserDone).to(user_emit))
        .branch(_ug.match(_UserTextBlock).to(user_parse_text))
        .branch(_ug.match(_UserImageBlock).to(user_parse_image))
        .branch(_ug.match(_UserToolResultBlock).to(user_parse_tool_result))
        .branch(_ug.match(_UserUnknownBlock).to(user_parse_unknown))
        .branch(_ug.match(_UserNonDictBlock).to(user_parse_non_dict))
    ),
    _ug.edge_from(
        user_parse_text,
        user_parse_image,
        user_parse_tool_result,
        user_parse_unknown,
        user_parse_non_dict,
    ).to(user_take_next),
    _ug.edge_from(user_emit).to(_ug.end_node),
)


_user_turn_graph = _ug.build()


# ── Assistant-turn FSM ─────────────────────────────────────────────────────


@dataclass
class _AssistantTurnState:
    """State for one assistant message's load FSM."""

    queue: deque[tuple[int, Any]] = field(default_factory=deque)
    parts: list[ModelResponsePart] = field(default_factory=list)
    msg_index: int = 0
    raw_extras: dict[str, Any] = field(default_factory=dict)


class _AssistantDone:
    """Marker for end of the assistant-turn queue."""


@dataclass
class _AssistantBlock:
    """Typed assistant-turn dispatch envelope keyed by block ``type``."""

    block: dict[str, Any]


@dataclass
class _AssistantTextBlock(_AssistantBlock):
    pass


@dataclass
class _AssistantToolUseBlock(_AssistantBlock):
    pass


@dataclass
class _AssistantThinkingBlock(_AssistantBlock):
    pass


@dataclass
class _AssistantRedactedThinkingBlock(_AssistantBlock):
    pass


@dataclass
class _AssistantUnknownBlock:
    block_index: int
    block: dict[str, Any]


@dataclass
class _AssistantNonDictBlock:
    block_index: int
    raw: Any


_ag: GraphBuilder[_AssistantTurnState, None, None, list[ModelResponsePart]] = GraphBuilder(
    state_type=_AssistantTurnState,
    output_type=list[ModelResponsePart],
)


@_ag.step
async def assistant_take_next(ctx: StepContext[_AssistantTurnState, None, None]) -> Any:
    """Router source: pop the next block and dispatch by ``type``."""
    if not ctx.state.queue:
        return _AssistantDone()
    block_index, raw_block = ctx.state.queue.popleft()
    if not isinstance(raw_block, dict):
        return _AssistantNonDictBlock(block_index=block_index, raw=raw_block)
    block: dict[str, Any] = raw_block
    block_type = block.get("type", "")
    if block_type == "text":
        return _AssistantTextBlock(block=block)
    if block_type == "tool_use":
        return _AssistantToolUseBlock(block=block)
    if block_type == "thinking":
        return _AssistantThinkingBlock(block=block)
    if block_type == "redacted_thinking":
        return _AssistantRedactedThinkingBlock(block=block)
    return _AssistantUnknownBlock(block_index=block_index, block=block)


@_ag.step
async def assistant_parse_text(
    ctx: StepContext[_AssistantTurnState, None, _AssistantTextBlock],
) -> None:
    """Emit a :class:`TextPart` from an assistant text block."""
    ctx.state.parts.append(TextPart(content=ctx.inputs.block.get("text", "")))


@_ag.step
async def assistant_parse_tool_use(
    ctx: StepContext[_AssistantTurnState, None, _AssistantToolUseBlock],
) -> None:
    """Emit a :class:`ToolCallPart` from an assistant tool_use block."""
    block = ctx.inputs.block
    ctx.state.parts.append(
        ToolCallPart(
            tool_name=block.get("name", ""),
            args=block.get("input"),
            tool_call_id=block.get("id", ""),
        )
    )


@_ag.step
async def assistant_parse_thinking(
    ctx: StepContext[_AssistantTurnState, None, _AssistantThinkingBlock],
) -> None:
    """Emit a :class:`ThinkingPart` from a thinking block."""
    block = ctx.inputs.block
    ctx.state.parts.append(
        ThinkingPart(content=block.get("thinking", ""), signature=block.get("signature"))
    )


@_ag.step
async def assistant_parse_redacted_thinking(
    ctx: StepContext[_AssistantTurnState, None, _AssistantRedactedThinkingBlock],
) -> None:
    """Emit a :class:`ThinkingPart` with id=``redacted_thinking`` carrying opaque ciphertext."""
    ctx.state.parts.append(
        ThinkingPart(
            content="",
            id="redacted_thinking",
            signature=ctx.inputs.block.get("data"),
        )
    )


@_ag.step
async def assistant_parse_unknown(
    ctx: StepContext[_AssistantTurnState, None, _AssistantUnknownBlock],
) -> None:
    """Stash unknown assistant blocks in raw_extras and feed JSON into a TextPart."""
    payload = ctx.inputs
    ctx.state.parts.append(TextPart(content=json.dumps(payload.block)))
    ctx.state.raw_extras[
        f"unknown_block:msg:{ctx.state.msg_index}:idx:{payload.block_index}"
    ] = payload.block


@_ag.step
async def assistant_parse_non_dict(
    ctx: StepContext[_AssistantTurnState, None, _AssistantNonDictBlock],
) -> None:
    """Coerce a non-dict block to its JSON string and stash the raw value."""
    payload = ctx.inputs
    ctx.state.parts.append(TextPart(content=json.dumps(payload.raw)))
    ctx.state.raw_extras[
        f"unknown_block:msg:{ctx.state.msg_index}:idx:{payload.block_index}"
    ] = payload.raw


@_ag.step
async def assistant_emit(
    ctx: StepContext[_AssistantTurnState, None, _AssistantDone],
) -> list[ModelResponsePart]:
    """Terminal step — emit accumulated parts (with sentinel empty TextPart if none)."""
    if not ctx.state.parts:
        ctx.state.parts.append(TextPart(content=""))
    return ctx.state.parts


_ag.add(
    _ag.edge_from(_ag.start_node).to(assistant_take_next),
    _ag.edge_from(assistant_take_next).to(
        _ag.decision()
        .branch(_ag.match(_AssistantDone).to(assistant_emit))
        .branch(_ag.match(_AssistantTextBlock).to(assistant_parse_text))
        .branch(_ag.match(_AssistantToolUseBlock).to(assistant_parse_tool_use))
        .branch(_ag.match(_AssistantThinkingBlock).to(assistant_parse_thinking))
        .branch(_ag.match(_AssistantRedactedThinkingBlock).to(assistant_parse_redacted_thinking))
        .branch(_ag.match(_AssistantUnknownBlock).to(assistant_parse_unknown))
        .branch(_ag.match(_AssistantNonDictBlock).to(assistant_parse_non_dict))
    ),
    _ag.edge_from(
        assistant_parse_text,
        assistant_parse_tool_use,
        assistant_parse_thinking,
        assistant_parse_redacted_thinking,
        assistant_parse_unknown,
        assistant_parse_non_dict,
    ).to(assistant_take_next),
    _ag.edge_from(assistant_emit).to(_ag.end_node),
)


_assistant_turn_graph = _ag.build()


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
    """Parse the top-level ``system`` field into :class:`SystemPromptPart` entries."""
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
    """Parse Anthropic tool definitions."""
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
    """Extract sampling + behavior settings from the wire body."""
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
    parts = await _user_turn_graph.run(state=state)
    return ModelRequest(parts=list(parts))


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
    parts = await _assistant_turn_graph.run(state=state)
    return ModelResponse(parts=list(parts))


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
    """Parse an Anthropic Messages API request body into the IR via the FSM."""
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
