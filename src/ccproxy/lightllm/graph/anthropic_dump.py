"""Render a :class:`ParsedRequest` to Anthropic Messages wire bytes via FSM.

The flat-queue / pattern-matched-router FSM replaces the
``CaptureSentinel``-driven ``AnthropicModel`` instantiation in
:mod:`ccproxy.lightllm.outbound_anthropic`. One :class:`AnthropicDumpState`
+ graph run per :class:`pydantic_ai.messages.ModelMessage`; the imperative
wrapper :func:`render_anthropic_dump` assembles the static request envelope
(model, sampling settings, system blocks, tools, ``raw_extras`` stitch) around
the FSM-emitted content-block lists.

Cache control on per-content-block ``CachePoint`` markers is handled by
:class:`ApplyCacheNode` mutating the dict referenced by
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
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from ccproxy.lightllm.parsed import ParsedRequest

# ── State ──────────────────────────────────────────────────────────────────


@dataclass
class AnthropicDumpState:
    """Per-message FSM state.

    The queue is the 1-D stream of pre-flattened IR items (parts + UserContent
    items) the FSM consumes. ``blocks`` accumulates the typed
    :class:`BetaContentBlockParam` dicts the queue items produce.
    ``last_emitted_block`` is the dict reference :class:`ApplyCacheNode` mutates
    to attach a ``cache_control`` field — kept as a separate field so that nodes
    appending multiple blocks can update the reference deliberately rather than
    accidentally cache-tagging the wrong one.
    """

    queue: deque[Any] = field(default_factory=deque)
    blocks: list[BetaContentBlockParam] = field(default_factory=list)
    last_emitted_block: BetaContentBlockParam | None = None


def _append_block(state: AnthropicDumpState, block: BetaContentBlockParam) -> None:
    """Append a block AND update the cache-target reference in one step.

    Every node that emits a block goes through this helper so the
    ``last_emitted_block`` invariant is centrally enforced.
    """
    state.blocks.append(block)
    state.last_emitted_block = block


# ── Nodes ──────────────────────────────────────────────────────────────────


@dataclass
class FetchNextNode(BaseNode[AnthropicDumpState, None, list[BetaContentBlockParam]]):
    """Router: pop the next queue item and dispatch by type via ``match``."""

    async def run(
        self, ctx: GraphRunContext[AnthropicDumpState, None]
    ) -> BaseNode[AnthropicDumpState, None, Any] | End[list[BetaContentBlockParam]]:
        if not ctx.state.queue:
            return End(ctx.state.blocks)

        item = ctx.state.queue.popleft()

        match item:
            case str():
                return ParseTextNode(text=item)
            case CachePoint():
                return ApplyCacheNode(cache=item)
            case BinaryContent():
                return ParseBinaryNode(item=item)
            case ImageUrl() | DocumentUrl():
                return ParseUrlNode(item=item)
            case UploadedFile():
                return ParseUploadedFileNode(item=item)
            case ToolReturnPart():
                return ParseToolReturnNode(part=item)
            case RetryPromptPart():
                return ParseRetryPromptNode(part=item)
            case TextPart():
                return ParseTextPartNode(part=item)
            case ThinkingPart():
                return ParseThinkingPartNode(part=item)
            case ToolCallPart():
                return ParseToolCallPartNode(part=item)
            case _:
                # AudioUrl, NativeToolCallPart, NativeToolReturnPart, and
                # anything else with no Anthropic equivalent are dropped.
                # (System parts are pre-stripped by the wrapper.)
                return FetchNextNode()


@dataclass
class ParseTextNode(BaseNode[AnthropicDumpState, None]):
    """Emit a text content block from a bare string (or ``TextPart``-derived string)."""

    text: str

    async def run(
        self, ctx: GraphRunContext[AnthropicDumpState, None]
    ) -> BaseNode[AnthropicDumpState, None, Any]:
        _append_block(ctx.state, {"type": "text", "text": self.text})
        return FetchNextNode()


@dataclass
class ParseTextPartNode(BaseNode[AnthropicDumpState, None]):
    """Emit a text block from a :class:`TextPart` (assistant-turn text)."""

    part: TextPart

    async def run(
        self, ctx: GraphRunContext[AnthropicDumpState, None]
    ) -> BaseNode[AnthropicDumpState, None, Any]:
        _append_block(ctx.state, {"type": "text", "text": self.part.content})
        return FetchNextNode()


@dataclass
class ParseBinaryNode(BaseNode[AnthropicDumpState, None]):
    """Emit an image or document block from a :class:`BinaryContent` payload.

    Bytes are base64-encoded eagerly into the source dict so the final
    ``json.dumps`` call doesn't need a ``default=`` fallback.
    """

    item: BinaryContent

    async def run(
        self, ctx: GraphRunContext[AnthropicDumpState, None]
    ) -> BaseNode[AnthropicDumpState, None, Any]:
        media_type = self.item.media_type
        source: dict[str, Any] = {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(self.item.data).decode("ascii"),
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
        return FetchNextNode()


@dataclass
class ParseUrlNode(BaseNode[AnthropicDumpState, None]):
    """Emit an image or document block from an ``ImageUrl`` / ``DocumentUrl``."""

    item: ImageUrl | DocumentUrl

    async def run(
        self, ctx: GraphRunContext[AnthropicDumpState, None]
    ) -> BaseNode[AnthropicDumpState, None, Any]:
        block: BetaContentBlockParam
        if isinstance(self.item, ImageUrl):
            block = cast(
                BetaImageBlockParam,
                {"type": "image", "source": {"type": "url", "url": self.item.url}},
            )
        else:
            block = cast(
                BetaContentBlockParam,
                {
                    "type": "document",
                    "source": {"type": "url", "url": self.item.url},
                    "media_type": self.item.media_type or "application/octet-stream",
                },
            )
        _append_block(ctx.state, block)
        return FetchNextNode()


@dataclass
class ParseUploadedFileNode(BaseNode[AnthropicDumpState, None]):
    """Emit a file-source image/document block from an Anthropic ``UploadedFile``."""

    item: UploadedFile

    async def run(
        self, ctx: GraphRunContext[AnthropicDumpState, None]
    ) -> BaseNode[AnthropicDumpState, None, Any]:
        if self.item.provider_name != "anthropic":
            # File from another provider — no Anthropic equivalent.
            return FetchNextNode()
        media_type = self.item.media_type or "application/octet-stream"
        file_src: dict[str, Any] = {
            "type": "file",
            "file_id": self.item.file_id,
            "media_type": media_type,
        }
        kind = "image" if media_type.startswith("image/") else "document"
        blk: dict[str, Any] = {"type": kind, "source": file_src}
        if kind == "document":
            blk["media_type"] = media_type
        _append_block(ctx.state, cast(BetaContentBlockParam, blk))
        return FetchNextNode()


@dataclass
class ParseToolReturnNode(BaseNode[AnthropicDumpState, None]):
    """Emit a ``tool_result`` block from a :class:`ToolReturnPart`."""

    part: ToolReturnPart

    async def run(
        self, ctx: GraphRunContext[AnthropicDumpState, None]
    ) -> BaseNode[AnthropicDumpState, None, Any]:
        # Emit list-of-text-blocks form to match pydantic-ai's AnthropicModel
        # output. Anthropic accepts both string and block list, but matching
        # the legacy renderer keeps byte-level diffs minimal during the
        # migration window.
        block: BetaToolResultBlockParam = {
            "type": "tool_result",
            "tool_use_id": self.part.tool_call_id,
            "content": [{"type": "text", "text": self.part.model_response_str()}],
        }
        if self.part.outcome == "failed":
            block["is_error"] = True
        _append_block(ctx.state, block)
        return FetchNextNode()


@dataclass
class ParseRetryPromptNode(BaseNode[AnthropicDumpState, None]):
    """Emit a ``tool_result`` (with ``is_error``) or a plain text block.

    When the retry carries a tool name it's a failed tool call response; with no
    tool name it's a synthesised user message asking the model to retry.
    """

    part: RetryPromptPart

    async def run(
        self, ctx: GraphRunContext[AnthropicDumpState, None]
    ) -> BaseNode[AnthropicDumpState, None, Any]:
        if self.part.tool_name is not None:
            block: BetaToolResultBlockParam = {
                "type": "tool_result",
                "tool_use_id": self.part.tool_call_id,
                "content": self.part.model_response(),
                "is_error": True,
            }
            _append_block(ctx.state, block)
        else:
            _append_block(ctx.state, {"type": "text", "text": self.part.model_response()})
        return FetchNextNode()


@dataclass
class ParseThinkingPartNode(BaseNode[AnthropicDumpState, None]):
    """Emit a ``thinking`` or ``redacted_thinking`` block."""

    part: ThinkingPart

    async def run(
        self, ctx: GraphRunContext[AnthropicDumpState, None]
    ) -> BaseNode[AnthropicDumpState, None, Any]:
        block: BetaContentBlockParam
        if self.part.id == "redacted_thinking":
            block = cast(
                BetaRedactedThinkingBlockParam,
                {"type": "redacted_thinking", "data": self.part.signature or ""},
            )
        else:
            block = cast(
                BetaContentBlockParam,
                {
                    "type": "thinking",
                    "thinking": self.part.content,
                    "signature": self.part.signature or "",
                },
            )
        _append_block(ctx.state, block)
        return FetchNextNode()


@dataclass
class ParseToolCallPartNode(BaseNode[AnthropicDumpState, None]):
    """Emit a ``tool_use`` block from a :class:`ToolCallPart`."""

    part: ToolCallPart

    async def run(
        self, ctx: GraphRunContext[AnthropicDumpState, None]
    ) -> BaseNode[AnthropicDumpState, None, Any]:
        _append_block(
            ctx.state,
            cast(
                BetaContentBlockParam,
                {
                    "type": "tool_use",
                    "id": self.part.tool_call_id,
                    "name": self.part.tool_name,
                    "input": self.part.args_as_dict(),
                },
            ),
        )
        return FetchNextNode()


@dataclass
class ApplyCacheNode(BaseNode[AnthropicDumpState, None]):
    """Attach ``cache_control`` to the just-appended block.

    A :class:`CachePoint` queue item arrives after the content item it caches;
    we mutate the dict referenced by ``state.last_emitted_block`` so the
    cache marker rides on the correct block in the final ``messages`` array.
    """

    cache: CachePoint

    async def run(
        self, ctx: GraphRunContext[AnthropicDumpState, None]
    ) -> BaseNode[AnthropicDumpState, None, Any]:
        if ctx.state.last_emitted_block is not None:
            # cache_control is allowed on every BetaContentBlockParam variant;
            # the cast is for the loose TypedDict union.
            cast(dict[str, Any], ctx.state.last_emitted_block)["cache_control"] = {
                "type": "ephemeral",
                "ttl": self.cache.ttl,
            }
        return FetchNextNode()


# ── Graph instance ─────────────────────────────────────────────────────────


_dump_graph = Graph[AnthropicDumpState, None, list[BetaContentBlockParam]](
    nodes=(
        FetchNextNode,
        ParseTextNode,
        ParseTextPartNode,
        ParseBinaryNode,
        ParseUrlNode,
        ParseUploadedFileNode,
        ParseToolReturnNode,
        ParseRetryPromptNode,
        ParseThinkingPartNode,
        ParseToolCallPartNode,
        ApplyCacheNode,
    ),
)


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
    result = await _dump_graph.run(FetchNextNode(), state=state)
    return result.output


async def _render_response_blocks(msg: ModelResponse) -> list[BetaContentBlockParam]:
    """Drive the FSM over one :class:`ModelResponse`'s parts."""
    flat: deque[Any] = deque(msg.parts)
    if not flat:
        return []
    state = AnthropicDumpState(queue=flat)
    result = await _dump_graph.run(FetchNextNode(), state=state)
    return result.output


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
    """Extract the top-level ``system`` field from the IR.

    Collects all :class:`SystemPromptPart` from :class:`ModelRequest` parts. If
    ``settings['anthropic_cache_instructions']`` is set, applies a uniform
    ``cache_control`` to every emitted block. The non-uniform case is handled
    downstream by :func:`_stitch_raw_extras` overriding with
    ``raw_extras['system']``.
    """
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
    """Format :class:`ToolDefinition` entries as Anthropic tool dicts.

    Applies uniform ``cache_control`` from ``settings['anthropic_cache_tool_definitions']``
    when set; the non-uniform case rides through ``raw_extras['tools']``.
    """
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
    """Re-inject ``raw_extras`` entries onto the rendered body.

    * ``raw_extras['system']`` and ``raw_extras['tools']`` override the
      FSM-rendered versions — populated by the inbound parser only when
      non-uniform ``cache_control`` couldn't be settings-compressed.
    * IR-internal markers (``cc:*``, ``unknown_block:*``) are skipped.
    * Other keys (``metadata``, etc.) are copied verbatim if they don't
      collide with a top-level field the FSM already produced.
    """
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
    """Render a :class:`ParsedRequest` to Anthropic Messages wire bytes.

    Drives the per-message FSM over ``parsed.messages`` to produce the typed
    ``messages`` array, then assembles the static envelope (model, sampling
    settings, system, tools, ``raw_extras`` stitch, stream flag) imperatively.
    Returns compact JSON bytes ready for the upstream ``POST /v1/messages``.
    """
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
