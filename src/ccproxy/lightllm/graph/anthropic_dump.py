"""Render a :class:`ParsedRequest` to Anthropic Messages wire bytes via FSM.

The flat-queue / decision-routed FSM (built with :mod:`pydantic_graph.beta`'s
``GraphBuilder``) replaces the ``CaptureSentinel``-driven ``AnthropicModel``
instantiation in :mod:`ccproxy.lightllm.outbound_anthropic`. One
:class:`AnthropicDumpState` + graph run per
:class:`pydantic_ai.messages.ModelMessage`; the imperative wrapper
:func:`render_anthropic_dump` assembles the static request envelope (model,
sampling settings, system blocks, tools, ``raw_extras`` stitch) around the
FSM-emitted content-block lists.

Cache control on per-content-block ``CachePoint`` markers is handled by
:func:`apply_cache` mutating the dict referenced by
``state.last_emitted_block``. Cache control on system blocks rides on
``settings['anthropic_cache_instructions']`` (uniform case) or
``raw_extras['system']`` (non-uniform case), matching the conventions the
inbound parser establishes. Same split for tools cache.

The output dicts use the SDK TypedDicts from ``anthropic.types.beta`` as the
typed wire boundary — no hand-rolled Pydantic mirror models, no
``dict[str, Any]`` in the emission path.
"""

from __future__ import annotations

import base64
import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessageParam,
    BetaRedactedThinkingBlockParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from pydantic_ai.messages import (
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
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition
from pydantic_graph.beta import GraphBuilder, StepContext, TypeExpression

from ccproxy.lightllm.parsed import ParsedRequest

# ── State ──────────────────────────────────────────────────────────────────


@dataclass
class AnthropicDumpState:
    """Per-message FSM state.

    The queue is the 1-D stream of pre-flattened IR items (parts + UserContent
    items) the FSM consumes. ``blocks`` accumulates the typed
    :class:`BetaContentBlockParam` dicts the queue items produce.
    ``last_emitted_block`` is the dict reference :func:`apply_cache` mutates
    to attach a ``cache_control`` field — kept as a separate field so that
    steps appending multiple blocks can update the reference deliberately
    rather than accidentally cache-tagging the wrong one.
    """

    queue: deque[Any] = field(default_factory=deque)
    blocks: list[BetaContentBlockParam] = field(default_factory=list)
    last_emitted_block: BetaContentBlockParam | None = None


class _DumpDone:
    """Marker returned by ``take_next`` when the queue is exhausted.

    The decision node routes this to ``emit_blocks``, which pulls the final
    block list out of state and hands it to the end node.
    """


class _Skip:
    """Marker for queue items with no Anthropic equivalent (audio, native tool parts)."""


def _append_block(state: AnthropicDumpState, block: BetaContentBlockParam) -> None:
    """Append a block AND update the cache-target reference in one step."""
    state.blocks.append(block)
    state.last_emitted_block = block


# ── Graph ──────────────────────────────────────────────────────────────────

_g: GraphBuilder[AnthropicDumpState, None, None, list[BetaContentBlockParam]] = GraphBuilder(
    state_type=AnthropicDumpState,
    output_type=list[BetaContentBlockParam],
)


@_g.step
async def take_next(
    ctx: StepContext[AnthropicDumpState, None, None],
) -> Any:
    """Router source: pop the next queue item, or signal end via :class:`_DumpDone`."""
    if not ctx.state.queue:
        return _DumpDone()
    item = ctx.state.queue.popleft()
    if isinstance(
        item,
        (
            str,
            CachePoint,
            BinaryContent,
            ImageUrl,
            DocumentUrl,
            UploadedFile,
            ToolReturnPart,
            RetryPromptPart,
            TextPart,
            ThinkingPart,
            ToolCallPart,
        ),
    ):
        return item
    # AudioUrl, NativeToolCallPart, NativeToolReturnPart, and anything else
    # with no Anthropic equivalent are dropped. (System parts are pre-stripped
    # by the wrapper.)
    return _Skip()


@_g.step
async def parse_text(ctx: StepContext[AnthropicDumpState, None, str]) -> None:
    """Emit a text content block from a bare string (or ``TextPart``-derived string)."""
    _append_block(ctx.state, {"type": "text", "text": ctx.inputs})


@_g.step
async def parse_text_part(ctx: StepContext[AnthropicDumpState, None, TextPart]) -> None:
    """Emit a text block from a :class:`TextPart` (assistant-turn text)."""
    _append_block(ctx.state, {"type": "text", "text": ctx.inputs.content})


@_g.step
async def parse_binary(ctx: StepContext[AnthropicDumpState, None, BinaryContent]) -> None:
    """Emit an image or document block from a :class:`BinaryContent` payload."""
    item = ctx.inputs
    media_type = item.media_type
    source: dict[str, Any] = {
        "type": "base64",
        "media_type": media_type,
        "data": base64.b64encode(item.data).decode("ascii"),
    }
    block: BetaContentBlockParam
    if media_type.startswith("image/"):
        block = cast(BetaImageBlockParam, {"type": "image", "source": source})
    else:
        block = cast(
            BetaContentBlockParam,
            {"type": "document", "source": source, "media_type": media_type},
        )
    _append_block(ctx.state, block)


@_g.step
async def parse_url(
    ctx: StepContext[AnthropicDumpState, None, ImageUrl | DocumentUrl],
) -> None:
    """Emit an image or document block from an ``ImageUrl`` / ``DocumentUrl``."""
    item = ctx.inputs
    block: BetaContentBlockParam
    if isinstance(item, ImageUrl):
        block = cast(
            BetaImageBlockParam,
            {"type": "image", "source": {"type": "url", "url": item.url}},
        )
    else:
        block = cast(
            BetaContentBlockParam,
            {
                "type": "document",
                "source": {"type": "url", "url": item.url},
                "media_type": item.media_type or "application/octet-stream",
            },
        )
    _append_block(ctx.state, block)


@_g.step
async def parse_uploaded_file(
    ctx: StepContext[AnthropicDumpState, None, UploadedFile],
) -> None:
    """Emit a file-source image/document block from an Anthropic ``UploadedFile``."""
    item = ctx.inputs
    if item.provider_name != "anthropic":
        return
    media_type = item.media_type or "application/octet-stream"
    file_src: dict[str, Any] = {
        "type": "file",
        "file_id": item.file_id,
        "media_type": media_type,
    }
    kind = "image" if media_type.startswith("image/") else "document"
    blk: dict[str, Any] = {"type": kind, "source": file_src}
    if kind == "document":
        blk["media_type"] = media_type
    _append_block(ctx.state, cast(BetaContentBlockParam, blk))


@_g.step
async def parse_tool_return(
    ctx: StepContext[AnthropicDumpState, None, ToolReturnPart],
) -> None:
    """Emit a ``tool_result`` block from a :class:`ToolReturnPart`."""
    part = ctx.inputs
    block: BetaToolResultBlockParam = {
        "type": "tool_result",
        "tool_use_id": part.tool_call_id,
        "content": [{"type": "text", "text": part.model_response_str()}],
    }
    if part.outcome == "failed":
        block["is_error"] = True
    _append_block(ctx.state, block)


@_g.step
async def parse_retry_prompt(
    ctx: StepContext[AnthropicDumpState, None, RetryPromptPart],
) -> None:
    """Emit a ``tool_result`` (with ``is_error``) or a plain text block."""
    part = ctx.inputs
    if part.tool_name is not None:
        block: BetaToolResultBlockParam = {
            "type": "tool_result",
            "tool_use_id": part.tool_call_id,
            "content": part.model_response(),
            "is_error": True,
        }
        _append_block(ctx.state, block)
    else:
        _append_block(ctx.state, {"type": "text", "text": part.model_response()})


@_g.step
async def parse_thinking_part(
    ctx: StepContext[AnthropicDumpState, None, ThinkingPart],
) -> None:
    """Emit a ``thinking`` or ``redacted_thinking`` block."""
    part = ctx.inputs
    block: BetaContentBlockParam
    if part.id == "redacted_thinking":
        block = cast(
            BetaRedactedThinkingBlockParam,
            {"type": "redacted_thinking", "data": part.signature or ""},
        )
    else:
        block = cast(
            BetaContentBlockParam,
            {
                "type": "thinking",
                "thinking": part.content,
                "signature": part.signature or "",
            },
        )
    _append_block(ctx.state, block)


@_g.step
async def parse_tool_call_part(
    ctx: StepContext[AnthropicDumpState, None, ToolCallPart],
) -> None:
    """Emit a ``tool_use`` block from a :class:`ToolCallPart`."""
    part = ctx.inputs
    _append_block(
        ctx.state,
        cast(
            BetaContentBlockParam,
            {
                "type": "tool_use",
                "id": part.tool_call_id,
                "name": part.tool_name,
                "input": part.args_as_dict(),
            },
        ),
    )


@_g.step
async def apply_cache(ctx: StepContext[AnthropicDumpState, None, CachePoint]) -> None:
    """Attach ``cache_control`` to the just-appended block."""
    if ctx.state.last_emitted_block is not None:
        cast(dict[str, Any], ctx.state.last_emitted_block)["cache_control"] = {
            "type": "ephemeral",
            "ttl": ctx.inputs.ttl,
        }


@_g.step
async def skip_item(ctx: StepContext[AnthropicDumpState, None, _Skip]) -> None:
    """No-op for queue items with no Anthropic equivalent."""
    del ctx  # protocol-required parameter; intentionally unused


@_g.step
async def emit_blocks(
    ctx: StepContext[AnthropicDumpState, None, _DumpDone],
) -> list[BetaContentBlockParam]:
    """Terminal step — hand the accumulated block list to the end node."""
    return ctx.state.blocks


_g.add(
    _g.edge_from(_g.start_node).to(take_next),
    _g.edge_from(take_next).to(
        _g.decision()
        .branch(_g.match(_DumpDone).to(emit_blocks))
        .branch(_g.match(_Skip).to(skip_item))
        .branch(_g.match(str).to(parse_text))
        .branch(_g.match(TextPart).to(parse_text_part))
        .branch(_g.match(CachePoint).to(apply_cache))
        .branch(_g.match(BinaryContent).to(parse_binary))
        .branch(_g.match(TypeExpression[ImageUrl | DocumentUrl]).to(parse_url))
        .branch(_g.match(UploadedFile).to(parse_uploaded_file))
        .branch(_g.match(ToolReturnPart).to(parse_tool_return))
        .branch(_g.match(RetryPromptPart).to(parse_retry_prompt))
        .branch(_g.match(ThinkingPart).to(parse_thinking_part))
        .branch(_g.match(ToolCallPart).to(parse_tool_call_part))
    ),
    _g.edge_from(
        parse_text,
        parse_text_part,
        apply_cache,
        parse_binary,
        parse_url,
        parse_uploaded_file,
        parse_tool_return,
        parse_retry_prompt,
        parse_thinking_part,
        parse_tool_call_part,
        skip_item,
    ).to(take_next),
    _g.edge_from(emit_blocks).to(_g.end_node),
)


_dump_graph = _g.build()


# ── Per-message FSM drivers ────────────────────────────────────────────────


async def _render_request_blocks(msg: ModelRequest) -> list[BetaContentBlockParam]:
    """Drive the FSM over one :class:`ModelRequest`'s parts."""
    flat: deque[Any] = deque()
    for part in msg.parts:
        if isinstance(part, SystemPromptPart):
            # Handled separately by _dump_system in the envelope wrapper.
            continue
        if isinstance(part, UserPromptPart):
            if isinstance(part.content, str):
                flat.append(part.content)
            else:
                # UserPromptPart([CachePoint]) sentinel: drop singleton CachePoint
                # lists since they carry no content block to attach to.
                if len(part.content) == 1 and isinstance(part.content[0], CachePoint):
                    continue
                flat.extend(part.content)
            continue
        # ToolReturnPart, RetryPromptPart — pass through to the FSM router.
        flat.append(part)

    if not flat:
        return []
    state = AnthropicDumpState(queue=flat)
    return await _dump_graph.run(state=state)


async def _render_response_blocks(msg: ModelResponse) -> list[BetaContentBlockParam]:
    """Drive the FSM over one :class:`ModelResponse`'s parts."""
    flat: deque[Any] = deque(msg.parts)
    if not flat:
        return []
    state = AnthropicDumpState(queue=flat)
    return await _dump_graph.run(state=state)


async def _render_messages(messages: Sequence[ModelMessage]) -> list[BetaMessageParam]:
    """Walk the IR conversation history into Anthropic ``BetaMessageParam`` turns."""
    out: list[BetaMessageParam] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            blocks = await _render_request_blocks(msg)
            if blocks:
                out.append({"role": "user", "content": blocks})
        elif isinstance(msg, ModelResponse):
            blocks = await _render_response_blocks(msg)
            if blocks:
                out.append({"role": "assistant", "content": blocks})
    return out


# ── Envelope helpers (imperative — these are NOT FSM nodes) ────────────────


def _dump_system(
    messages: Sequence[ModelMessage], settings: dict[str, Any]
) -> str | list[BetaTextBlockParam] | None:
    """Extract the top-level ``system`` field from the IR."""
    system_parts: list[SystemPromptPart] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    system_parts.append(part)
    if not system_parts:
        return None

    cache_ttl = settings.get("anthropic_cache_instructions")
    if not cache_ttl and len(system_parts) == 1:
        return system_parts[0].content

    blocks: list[BetaTextBlockParam] = []
    for part in system_parts:
        block: BetaTextBlockParam = {"type": "text", "text": part.content}
        if cache_ttl:
            block["cache_control"] = {"type": "ephemeral", "ttl": cache_ttl}
        blocks.append(block)
    return blocks


def _format_tools(tools: Sequence[ToolDefinition], settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Format :class:`ToolDefinition` entries as Anthropic tool dicts."""
    if not tools:
        return []
    cache_ttl = settings.get("anthropic_cache_tool_definitions")
    out: list[dict[str, Any]] = []
    for tool in tools:
        entry: dict[str, Any] = {
            "name": tool.name,
            "input_schema": tool.parameters_json_schema or {"type": "object"},
        }
        if tool.description:
            entry["description"] = tool.description
        if cache_ttl:
            entry["cache_control"] = {"type": "ephemeral", "ttl": cache_ttl}
        out.append(entry)
    return out


# Top-level wire fields the FSM + envelope wrapper own. ``raw_extras`` keys not
# in this set (and not IR-internal markers) get copied verbatim.
_IR_OWNED_TOP_LEVEL: frozenset[str] = frozenset(
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
    }
)


def _stitch_raw_extras(body: dict[str, Any], parsed: ParsedRequest) -> None:
    """Re-inject ``raw_extras`` entries onto the rendered body."""
    for key in ("system", "tools"):
        if key in parsed.raw_extras:
            body[key] = parsed.raw_extras[key]

    for key, value in parsed.raw_extras.items():
        if key in ("system", "tools"):
            continue
        if key.startswith(("cc:", "unknown_block:")):
            continue
        body.setdefault(key, value)


# ── Public entrypoint ──────────────────────────────────────────────────────


async def render_anthropic_dump(parsed: ParsedRequest) -> bytes:
    """Render a :class:`ParsedRequest` to Anthropic Messages wire bytes."""
    messages = await _render_messages(parsed.messages)
    settings_dict = cast(dict[str, Any], parsed.settings)
    system = _dump_system(parsed.messages, settings_dict)
    tools = _format_tools(parsed.request_parameters.function_tools, settings_dict)

    body: dict[str, Any] = {
        "model": parsed.model,
        "messages": messages,
    }
    for key in ("max_tokens", "temperature", "top_p", "top_k", "stop_sequences"):
        if key in settings_dict:
            body[key] = settings_dict[key]

    if system is not None:
        body["system"] = system
    if tools:
        body["tools"] = tools

    _stitch_raw_extras(body, parsed)

    if parsed.stream:
        body["stream"] = True

    return json.dumps(body, separators=(",", ":")).encode()
