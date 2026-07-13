"""Anthropic prompt-cache breakpoint policy engine.

Pure placement engine over the pydantic-ai IR: given a conversation
(``list[ModelMessage]``), model settings, and a :class:`CachePolicy`, it
computes where dynamic ``cache_control`` breakpoints should land and
returns new messages/settings plus a :class:`CacheBudgetReport`.

Anthropic semantics encoded here:

* A ``cache_control`` marker sits on a content block and caches the
  entire prefix up to and including that block (prefix order: tools →
  system → messages).
* A marker on the last tool definition caches the whole tool array —
  expressed IR-side as the ``anthropic_cache_tool_definitions`` settings
  knob, never as a message mutation.
* A system-level marker is the sentinel ``UserPromptPart([CachePoint])``
  immediately after the last :class:`SystemPromptPart` (the
  ``dump_system`` convention in :mod:`ccproxy.lightllm.adapters.anthropic`).
* **Maximum 4 breakpoints per request.** The engine censuses every
  marker that will reach the wire — authored ``CachePoint``s, sentinel
  system markers, ``cc:``-family ``raw_extras`` entries the dump path
  re-applies verbatim, and cache-related settings knobs — and admits
  policy placements in priority order (tools, system, user_tail
  newest-first) only while the budget holds.
* Authored always wins: the engine never removes or re-TTLs an existing
  marker; policy placements yield and the report says what yielded.

Scope: Anthropic cache semantics only. The engine does not check
providers — callers do (the ``cache_breakpoints`` pipeline hook guards
on the anthropic-compatible provider family; library callers know their
upstream).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from pydantic_ai.messages import (
    CachePoint,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    SystemPromptPart,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings

__all__ = ["CacheBudgetReport", "CachePolicy", "apply_cache_policy"]

MAX_BREAKPOINTS = 4
"""Anthropic's hard per-request breakpoint budget."""

_CACHE_SETTINGS_KEYS = (
    "anthropic_cache_tool_definitions",
    "anthropic_cache_instructions",
    "anthropic_cache_messages",
    "anthropic_cache",
)


@dataclass(frozen=True)
class CachePolicy:
    """Dynamic cache-breakpoint placements requested by a caller."""

    tools: Literal["5m", "1h"] | None = None
    """Marker on the tool set (settings ``anthropic_cache_tool_definitions``).

    The adapter stamps only the last non-deferred tool; deferred tools are
    excluded from prefix caching, and an all-deferred set makes this placement
    a wire no-op (the engine cannot see tools, so it still counts 1 breakpoint
    — a safe overestimate).
    """

    system: Literal["5m", "1h"] | None = None
    """Marker after the last system block (sentinel ``UserPromptPart``)."""

    user_tail: int = 0
    """Markers on the last N user messages (newest first). 0 = none."""

    user_tail_ttl: Literal["5m", "1h"] = "5m"
    """TTL for ``user_tail`` markers."""


@dataclass(frozen=True)
class CacheBudgetReport:
    """Outcome of one :func:`apply_cache_policy` run.

    Position labels are human-readable: ``"tools"``, ``"system"``,
    ``"user[-1]"`` (newest user message), ``"user[-2]"``, …
    """

    existing: int
    """Markers already present before policy placement (the census)."""

    placed: list[str]
    """Policy placements admitted within the budget."""

    dropped: list[str]
    """Policy placements refused because they would exceed the budget."""

    skipped: list[str]
    """Placements skipped as redundant or targetless (idempotence, shortfall)."""


def _is_sentinel(part: ModelRequestPart) -> bool:
    """True for the system-cache sentinel ``UserPromptPart([CachePoint])``."""
    return (
        isinstance(part, UserPromptPart)
        and isinstance(part.content, list)
        and len(part.content) == 1
        and isinstance(part.content[0], CachePoint)
    )


def _count_message_markers(messages: list[ModelMessage]) -> int:
    """Count every ``CachePoint`` in user-side content (sentinels included)."""
    count = 0
    for msg in messages:
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, list):
                count += sum(1 for item in part.content if isinstance(item, CachePoint))
    return count


def _census(
    messages: list[ModelMessage],
    settings: Mapping[str, Any],
    raw_extras: Mapping[str, Any] | None,
) -> int:
    """Count every marker that will reach the wire before policy placement."""
    count = _count_message_markers(messages)
    count += sum(1 for key in _CACHE_SETTINGS_KEYS if settings.get(key))
    if raw_extras:
        count += sum(1 for key in raw_extras if key.startswith("cc:"))
        # A verbatim tools override reaches the wire byte-faithfully, so any
        # cache_control it carries is an existing marker. Shallow scan only —
        # markers sit on the tool entries themselves, and fidelity-first means
        # even an invalid marker (e.g. on a deferred tool) is preserved and
        # therefore counted.
        tools_override = raw_extras.get("tools")
        if isinstance(tools_override, list):
            count += sum(1 for tool in tools_override if isinstance(tool, dict) and "cache_control" in tool)
    return count


def _last_system_position(messages: list[ModelMessage]) -> tuple[int, int] | None:
    """Return (message index, part index) of the last ``SystemPromptPart``."""
    position: tuple[int, int] | None = None
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, ModelRequest):
            continue
        for part_idx, part in enumerate(msg.parts):
            if isinstance(part, SystemPromptPart):
                position = (msg_idx, part_idx)
    return position


def _user_tail_targets(messages: list[ModelMessage], n: int) -> list[int]:
    """Message indices of the last ``n`` user-content-bearing requests, newest first."""
    targets: list[int] = []
    for msg_idx in range(len(messages) - 1, -1, -1):
        if len(targets) >= n:
            break
        msg = messages[msg_idx]
        if isinstance(msg, ModelRequest) and any(
            isinstance(p, UserPromptPart) and not _is_sentinel(p) for p in msg.parts
        ):
            targets.append(msg_idx)
    return targets


def _user_message_marked(msg: ModelRequest) -> bool:
    """True if the final real ``UserPromptPart`` already ends with a marker."""
    for part in reversed(msg.parts):
        if isinstance(part, UserPromptPart) and not _is_sentinel(part):
            content = part.content
            return isinstance(content, list) and bool(content) and isinstance(content[-1], CachePoint)
    return False


def _place_system_marker(
    messages: list[ModelMessage],
    position: tuple[int, int],
    ttl: Literal["5m", "1h"],
) -> None:
    """Insert the sentinel marker after the last system part (in place on the working list)."""
    msg_idx, part_idx = position
    msg = cast(ModelRequest, messages[msg_idx])
    sentinel = UserPromptPart(content=[CachePoint(ttl=ttl)])
    new_parts = [*msg.parts[: part_idx + 1], sentinel, *msg.parts[part_idx + 1 :]]
    messages[msg_idx] = replace(msg, parts=new_parts)


def _place_user_marker(
    messages: list[ModelMessage],
    msg_idx: int,
    ttl: Literal["5m", "1h"],
) -> None:
    """Append a ``CachePoint`` to the final real ``UserPromptPart`` of ``messages[msg_idx]``."""
    msg = cast(ModelRequest, messages[msg_idx])
    for part_idx in range(len(msg.parts) - 1, -1, -1):
        part = msg.parts[part_idx]
        if isinstance(part, UserPromptPart) and not _is_sentinel(part):
            content = part.content
            new_content: list[Any] = [content] if isinstance(content, str) else [*content]
            new_content.append(CachePoint(ttl=ttl))
            new_parts = list(msg.parts)
            new_parts[part_idx] = replace(part, content=new_content)
            messages[msg_idx] = replace(msg, parts=new_parts)
            return


def apply_cache_policy(
    messages: list[ModelMessage],
    settings: ModelSettings,
    policy: CachePolicy,
    *,
    raw_extras: Mapping[str, Any] | None = None,
) -> tuple[list[ModelMessage], ModelSettings, CacheBudgetReport]:
    """Apply ``policy`` to the IR, honoring Anthropic's 4-breakpoint budget.

    Args:
        messages: Conversation IR. Never mutated — touched requests are rebuilt.
        settings: Incoming model settings. Never mutated — copied.
        policy: Requested dynamic placements.
        raw_extras: Read-only census input; ``cc:``-family keys count as
            existing markers because the dump path re-applies them verbatim.

    Returns:
        ``(new_messages, new_settings, report)``. Re-running the same policy
        over its own output places nothing (idempotence).
    """
    settings_map = cast(Mapping[str, Any], settings)
    new_settings = cast(ModelSettings, dict(settings_map))
    new_messages: list[ModelMessage] = list(messages)

    existing = _census(new_messages, settings_map, raw_extras)
    total = existing
    placed: list[str] = []
    dropped: list[str] = []
    skipped: list[str] = []
    dropping = False

    def admit(label: str) -> bool:
        nonlocal total, dropping
        if dropping or total >= MAX_BREAKPOINTS:
            dropping = True
            dropped.append(label)
            return False
        total += 1
        placed.append(label)
        return True

    if policy.tools is not None:
        # A verbatim tools override overwrites the dump side's formatted tools
        # at stitch time, so a knob placement would burn budget for a wire
        # no-op — skip it whether or not the override carries markers.
        if settings_map.get("anthropic_cache_tool_definitions") or (raw_extras and "tools" in raw_extras):
            skipped.append("tools")
        elif admit("tools"):
            cast(dict[str, Any], new_settings)["anthropic_cache_tool_definitions"] = policy.tools

    if policy.system is not None:
        position = _last_system_position(new_messages)
        if position is None or settings_map.get("anthropic_cache_instructions"):
            skipped.append("system")
        else:
            msg = cast(ModelRequest, new_messages[position[0]])
            next_idx = position[1] + 1
            if next_idx < len(msg.parts) and _is_sentinel(msg.parts[next_idx]):
                skipped.append("system")
            elif admit("system"):
                _place_system_marker(new_messages, position, policy.system)

    if policy.user_tail > 0:
        targets = _user_tail_targets(new_messages, policy.user_tail)
        for slot in range(policy.user_tail):
            label = f"user[-{slot + 1}]"
            if slot >= len(targets):
                skipped.append(label)
                continue
            msg_idx = targets[slot]
            if _user_message_marked(cast(ModelRequest, new_messages[msg_idx])):
                skipped.append(label)
            elif admit(label):
                _place_user_marker(new_messages, msg_idx, policy.user_tail_ttl)

    report = CacheBudgetReport(existing=existing, placed=placed, dropped=dropped, skipped=skipped)
    return new_messages, new_settings, report
