"""Perplexity Pro SSE → pydantic-ai IR events (sync).

Perplexity has no pydantic-ai model counterpart, so the Perplexity-specific
parsing logic is ported in-tree directly to emit pydantic-ai ``ModelResponseStreamEvent``
objects. The existing :class:`ccproxy.lightllm.pplx.PerplexityProIterator` —
deleted in Phase 9 — provided this functionality against LiteLLM's
``ModelResponseStream``; we replicate the same prefix-diffing, four-patch-mode
parser, step-rendering, and identifier-capture logic but route deltas through
:class:`pydantic_ai._parts_manager.ModelResponsePartsManager`.

Wire format quick reference (full coverage in ``docs/pplx.md``):

- Answer text arrives as JSON patches under ``blocks[].diff_block.patches[]``
  on ``markdown_block``. Four modes:
  - Mode A: ``path=""`` carrying cumulative ``answer`` string (prefix-diff)
  - Mode B: ``path=""`` carrying a ``chunks`` array (``chunk_starting_offset=0``)
  - Mode C: ``path="/chunks/N"`` carrying single new chunk string (append)
  - Mode D: ``path="/markdown_block"`` or ``"/markdown_block/answer"``
  (cumulative)
- Reasoning text arrives as ``plan_block.goals[].description`` (cumulative)
  plus rendered steps from ``plan_block.steps[]`` and the JSON-encoded
  ``event.text`` mirror.
- Identifier capture (``backend_uuid``, ``read_write_token``, ``context_uuid``,
  ``thread_url_slug``, ``thread_title``, ``display_model``) is independent of
  blocks — top-level event fields. ``upstream_raw_bytes`` carries the
  byte-for-byte tee so :class:`ccproxy.inspector.pplx_addon.PerplexityAddon`
  can do its own L1 cache extraction.
- ``intended_usage == "ask_text"`` is skipped to avoid double-emission against
  ``ask_text_0_markdown`` (the markdown-formatted parallel block).
- ``RESEARCH_CLARIFYING_QUESTIONS`` step is suppressed silently here; the
  request-side surfaces it as a 400 via the standalone iterator path. The
  intake's role is event emission only — error escalation lives outside
  the IR pipeline.

The intake emits two pydantic-ai part streams:

1. A :class:`pydantic_ai.messages.TextPart` for the answer (driven via
   ``handle_text_delta`` with a stable ``vendor_part_id="pplx-answer"``).
2. A :class:`pydantic_ai.messages.ThinkingPart` for reasoning + step
   rendering (driven via ``handle_thinking_delta`` with
   ``vendor_part_id="pplx-reasoning"``).

These remain available across the entire stream and are flushed (no
``PartEndEvent`` required) when ``close`` returns.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai._parts_manager import ModelResponsePartsManager

from ccproxy.lightllm.pplx_steps import _KNOWN_INTENDED_USAGES, render_step

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelResponseStreamEvent
    from pydantic_ai.models import ModelRequestParameters

logger = logging.getLogger(__name__)


_PPLX_ID_FIELDS: tuple[str, ...] = (
    "backend_uuid",
    "read_write_token",
    "context_uuid",
    "thread_url_slug",
    "thread_title",
    "display_model",
)
"""Top-level event fields captured into ``_ids`` whenever they appear."""

_ANSWER_VENDOR_ID = "pplx-answer"
"""Stable vendor_part_id for the answer ``TextPart``."""

_REASONING_VENDOR_ID = "pplx-reasoning"
"""Stable vendor_part_id for the reasoning ``ThinkingPart``."""


@dataclass
class _PerplexityStreamState:
    """Running state across SSE events for a single Perplexity response."""

    answer_seen: str = ""
    """Cumulative answer text seen so far — for prefix-diffing."""

    reasoning_seen: str = ""
    """Cumulative reasoning text from ``plan_block.goals[].description``."""

    ids: dict[str, str] = field(default_factory=dict)
    """Captured thread identifiers (last-write-wins)."""

    final: bool = False
    """``True`` once an event carries ``final_sse_message: true``."""

    seen_step_uuids: set[str] = field(default_factory=set)
    """Deduplication set for ``plan_block.steps[].uuid`` across cumulative events."""

    logged_unknown_intended_usages: set[str] = field(default_factory=set)
    """Per-stream dedup for the DEBUG log of unknown ``intended_usage`` values."""


class PerplexityResponseIntake:
    """Per-stream Perplexity SSE → pydantic-ai IR event dispatcher.

    Stateful. ``feed`` is called repeatedly with raw upstream bytes;
    framing of SSE events and prefix-diff state carry across calls.
    ``upstream_raw_bytes`` is a byte-for-byte tee for inspectors.
    """

    name = "perplexity_pro"

    def __init__(self, *, model: str, request_params: ModelRequestParameters) -> None:
        self._model = model
        self._request_params = request_params
        self._parts_manager = ModelResponsePartsManager()
        self._sse_buffer = bytearray()
        self.upstream_raw_bytes = bytearray()
        self._state = _PerplexityStreamState()

    # ---- public Protocol API ------------------------------------------------

    def feed(self, data: bytes) -> Iterator[ModelResponseStreamEvent]:
        """Process incoming bytes; yield zero-or-more IR events."""
        if not data:
            return
        self.upstream_raw_bytes.extend(data)
        self._sse_buffer.extend(data)
        for event_dict in self._drain_sse_events():
            yield from self._dispatch_event(event_dict)

    def close(self) -> Iterator[ModelResponseStreamEvent]:
        """Stream end. No trailing events required — parts_manager keeps state."""
        yield from ()

    # ---- SSE framing --------------------------------------------------------

    def _drain_sse_events(self) -> Iterator[dict[str, Any]]:
        """Frame ``data: <json>`` SSE events from the byte buffer.

        Standard SSE separators (``\\n\\n`` or ``\\r\\n\\r\\n``) terminate events.
        Partial frames remain in ``_sse_buffer`` for the next ``feed`` call.
        Non-JSON payloads and ``[DONE]`` sentinels are skipped silently.
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
            event_dict = self._parse_frame(frame)
            if event_dict is not None:
                yield event_dict

    @staticmethod
    def _parse_frame(frame: bytes) -> dict[str, Any] | None:
        """Extract the JSON payload from a single SSE frame.

        Walks lines looking for one starting with ``data:`` (per SSE spec).
        Returns ``None`` for keepalive comments, non-data frames, ``[DONE]``
        sentinels, and JSON parse failures.
        """
        for raw_line in frame.split(b"\n"):
            line = raw_line.rstrip(b"\r")
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].lstrip()
            if not payload or payload == b"[DONE]":
                return None
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    # ---- event dispatch -----------------------------------------------------

    def _dispatch_event(self, event: dict[str, Any]) -> Iterator[ModelResponseStreamEvent]:
        """Apply one Perplexity SSE event; yield resulting IR events.

        Capture identifiers, gate terminal flag, then walk the event for
        answer deltas (via ``markdown_block`` diff patches) and reasoning
        deltas (via ``plan_block.goals[].description``, ``plan_block.steps[]``,
        and the ``event.text`` JSON-encoded step mirror).
        """
        for key in _PPLX_ID_FIELDS:
            val = event.get(key)
            if isinstance(val, str) and val:
                self._state.ids[key] = val

        if event.get("final_sse_message"):
            self._state.final = True

        blocks_raw = event.get("blocks") or []
        blocks: list[dict[str, Any]] = (
            [b for b in blocks_raw if isinstance(b, dict)] if isinstance(blocks_raw, list) else []
        )

        reasoning_delta = ""
        answer_delta = ""

        # event.text mirror: walked only when no plan_block exists (avoids
        # double-emission against the structured channel). Clarifying questions
        # are silently suppressed here — the standalone Perplexity request
        # surface owns the 400 escalation.
        text = event.get("text")
        has_plan_block = any(isinstance(b.get("plan_block"), dict) for b in blocks)
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list) and not has_plan_block:
                for step in parsed:
                    if not isinstance(step, dict):
                        continue
                    if step.get("step_type") == "RESEARCH_CLARIFYING_QUESTIONS":
                        continue
                    rendered = self._consume_step(step)
                    if rendered:
                        reasoning_delta += rendered

        for block in blocks:
            intended_usage = block.get("intended_usage")

            if intended_usage in ("pro_search_steps", "plan", "reasoning_plan_block"):
                plan_block = block.get("plan_block") or {}
                if isinstance(plan_block, dict):
                    goals = plan_block.get("goals") or []
                    if isinstance(goals, list):
                        for goal in goals:
                            if not isinstance(goal, dict):
                                continue
                            desc = goal.get("description")
                            if isinstance(desc, str) and desc.startswith(self._state.reasoning_seen):
                                new = desc[len(self._state.reasoning_seen) :]
                                if new:
                                    reasoning_delta += new
                                    self._state.reasoning_seen = desc

                    for step in plan_block.get("steps") or []:
                        if not isinstance(step, dict):
                            continue
                        rendered = self._consume_step(step)
                        if rendered:
                            reasoning_delta += rendered

            # Bare ``markdown_block`` (no ``diff_block`` wrapper) — the terminal
            # event re-sends the full answer this way. Prefix-diff against
            # ``answer_seen`` surfaces any tail text not seen in earlier patches.
            mb = block.get("markdown_block")
            if isinstance(mb, dict) and not block.get("diff_block") and intended_usage != "ask_text":
                answer_str = mb.get("answer")
                if isinstance(answer_str, str) and answer_str and answer_str.startswith(self._state.answer_seen):
                    bare_delta = answer_str[len(self._state.answer_seen) :]
                    if bare_delta:
                        answer_delta += bare_delta
                    self._state.answer_seen = answer_str

            diff_block = block.get("diff_block")
            if not isinstance(diff_block, dict):
                if (
                    intended_usage
                    and intended_usage not in _KNOWN_INTENDED_USAGES
                    and intended_usage not in self._state.logged_unknown_intended_usages
                ):
                    self._state.logged_unknown_intended_usages.add(intended_usage)
                    logger.debug(
                        "pplx intake: unhandled intended_usage=%s keys=%s",
                        intended_usage,
                        list(block.keys()),
                    )
                continue

            # The ``ask_text`` block duplicates ``ask_text_0_markdown``'s
            # patches; processing both would double every chunk. Markdown wins.
            if intended_usage == "ask_text":
                continue

            field_name = diff_block.get("field")
            patches = diff_block.get("patches") or []
            if not isinstance(patches, list):
                continue

            for patch in patches:
                if not isinstance(patch, dict):
                    continue
                path = patch.get("path", "")
                value = patch.get("value")

                if path.startswith("/goals"):
                    if isinstance(value, str) and value.startswith(self._state.reasoning_seen):
                        new = value[len(self._state.reasoning_seen) :]
                        if new:
                            reasoning_delta += new
                            self._state.reasoning_seen = value
                    continue

                if path == "/progress":
                    continue

                if field_name != "markdown_block":
                    continue

                delta = self._apply_markdown_patch(path, value)
                if delta:
                    answer_delta += delta

        if reasoning_delta:
            yield from self._parts_manager.handle_thinking_delta(
                vendor_part_id=_REASONING_VENDOR_ID,
                content=reasoning_delta,
            )

        if answer_delta:
            yield from self._parts_manager.handle_text_delta(
                vendor_part_id=_ANSWER_VENDOR_ID,
                content=answer_delta,
            )

    def _apply_markdown_patch(self, path: str, value: Any) -> str:
        """Apply one ``diff_block.patches[]`` entry; return the answer delta string.

        Handles all four documented patch modes. Mutates
        ``self._state.answer_seen`` in place. Returns ``""`` when nothing
        new was extracted.
        """
        # Mode A/B — root patch carrying full markdown_block state (chunks
        # array with offset=0, and/or cumulative ``answer`` string).
        if path == "" and isinstance(value, dict):
            delta = ""
            chunks = value.get("chunks")
            if isinstance(chunks, list):
                offset = value.get("chunk_starting_offset")
                new_text = "".join(c for c in chunks if isinstance(c, str))
                if offset in (None, 0):
                    if new_text != self._state.answer_seen:
                        if new_text.startswith(self._state.answer_seen):
                            d = new_text[len(self._state.answer_seen) :]
                        else:
                            d = new_text
                        if d:
                            delta += d
                        self._state.answer_seen = new_text
                elif new_text:
                    delta += new_text
                    self._state.answer_seen += new_text
            answer_str = value.get("answer")
            if isinstance(answer_str, str) and answer_str and answer_str.startswith(self._state.answer_seen):
                d = answer_str[len(self._state.answer_seen) :]
                if d:
                    delta += d
                self._state.answer_seen = answer_str
            return delta

        # Mode C — incremental chunk append at ``/chunks/N``.
        if path.startswith("/chunks/") and isinstance(value, str):
            self._state.answer_seen += value
            return value

        # Mode D — cumulative answer at ``/markdown_block`` or
        # ``/markdown_block/answer``.
        if path == "/markdown_block" and isinstance(value, dict):
            answer_str = value.get("answer")
            if isinstance(answer_str, str) and answer_str:
                if answer_str.startswith(self._state.answer_seen):
                    d = answer_str[len(self._state.answer_seen) :]
                    self._state.answer_seen = answer_str
                    return d
                if answer_str != self._state.answer_seen:
                    self._state.answer_seen = answer_str
                    return answer_str
            return ""

        if path == "/markdown_block/answer" and isinstance(value, str):
            if value.startswith(self._state.answer_seen):
                d = value[len(self._state.answer_seen) :]
                self._state.answer_seen = value
                return d
            if value != self._state.answer_seen:
                self._state.answer_seen = value
                return value
            return ""

        return ""

    def _consume_step(self, step: dict[str, Any]) -> str:
        """Render one ``plan_block.steps[]`` entry; return reasoning text to emit.

        Dedup across SSE events via ``state.seen_step_uuids``. Unlike the
        legacy iterator path, the intake doesn't accumulate structured
        ``state.all_steps`` / ``state.mcp_steps`` lists — those exist only
        for the non-spec OpenAI response-side surface, which the render layer
        owns. We emit only the reasoning text into the IR's ThinkingPart.
        """
        uuid_raw = step.get("uuid") or ""
        uuid_ = uuid_raw if isinstance(uuid_raw, str) else ""
        if uuid_ and uuid_ in self._state.seen_step_uuids:
            return ""
        if uuid_:
            self._state.seen_step_uuids.add(uuid_)

        result = render_step(step)
        return result.reasoning_text
