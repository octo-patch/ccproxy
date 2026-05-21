"""Tests for ``ccproxy.lightllm.outbound_anthropic.render_anthropic``.

The acceptance criterion in the briefing is:

    ``render_anthropic(parse_anthropic_messages(b))`` matches
    ``json.loads(b)`` modulo field-order and ``null``/missing omission.

Where the IR normalizes the wire shape (e.g. a string ``content`` is
canonicalized to a single-element ``[{"type": "text", "text": ...}]``
list), we use the stronger IR-mediated equivalence:

    ``parse(render(parse(b))) == parse(b)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from ccproxy.lightllm.anthropic_inbound import parse_anthropic_messages
from ccproxy.lightllm.outbound_anthropic import (
    CaptureSentinel,
    render_anthropic,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Per-block fields pydantic-ai's outbound emits as defaults that have no
# semantic effect on the upstream API. Drop them when comparing.
_REDUNDANT_BLOCK_FIELDS: frozenset[str] = frozenset({"is_error"})


def _canonicalize_block(value: Any) -> Any:
    """Drop None values, drop redundant defaults, and recursively sort dict keys."""
    if isinstance(value, dict):
        return {
            k: _canonicalize_block(v)
            for k, v in sorted(value.items())
            if v is not None and not (k in _REDUNDANT_BLOCK_FIELDS and v is False)
        }
    if isinstance(value, list):
        return [_canonicalize_block(v) for v in value]
    return value


def _canonical_content(content: Any) -> list[dict[str, Any]]:
    """Normalize ``content`` to a list-of-blocks form.

    Anthropic accepts both ``"hello"`` and ``[{"type": "text", "text": "hello"}]``;
    we expand strings to the list form so semantic equality works across
    the round-trip's normalization.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [_canonicalize_block(block) for block in content]
    return [_canonicalize_block(content)]


def _canonical_messages(messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        out.append({
            "role": msg.get("role"),
            "content": _canonical_content(msg.get("content", "")),
        })
    return out


def _canonical_system(system: Any) -> list[dict[str, Any]]:
    """Normalize a wire ``system`` field to the list-of-blocks form.

    The IR collapses consecutive ``SystemPromptPart`` entries into a
    single block joined by ``\\n\\n`` when they share the same cache
    setting; round-tripping a uniform-cache multi-block input therefore
    produces one concatenated block. We fold consecutive blocks with
    identical ``cache_control`` here so the original and rendered forms
    compare equal in the uniform case.
    """
    if system is None:
        return []
    if isinstance(system, str):
        return [{"type": "text", "text": system}]
    if not isinstance(system, list):
        return []
    canonical = [_canonicalize_block(b) for b in system]
    folded: list[dict[str, Any]] = []
    for block in canonical:
        if (
            folded
            and block.get("type") == "text"
            and folded[-1].get("type") == "text"
            and block.get("cache_control") == folded[-1].get("cache_control")
        ):
            folded[-1] = {
                **folded[-1],
                "text": f"{folded[-1].get('text', '')}\n\n{block.get('text', '')}",
            }
        else:
            folded.append(block)
    return folded


_DEFAULT_TOOL_CHOICE = {"type": "auto"}


def _canonical_tool_choice(value: Any) -> dict[str, Any]:
    """``None`` and ``{'type': 'auto'}`` are semantically equivalent."""
    if value is None:
        return _DEFAULT_TOOL_CHOICE
    canonical = _canonicalize_block(value)
    return canonical if isinstance(canonical, dict) else _DEFAULT_TOOL_CHOICE


def _build_normalised_view(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": body.get("model"),
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "top_k": body.get("top_k"),
        "stop_sequences": body.get("stop_sequences"),
        "stream": body.get("stream", False),
        "messages": _canonical_messages(body.get("messages", [])),
        "system": _canonical_system(body.get("system")),
        "tools": [_canonicalize_block(t) for t in body.get("tools", [])],
        "tool_choice": _canonical_tool_choice(body.get("tool_choice")) if body.get("tools") else None,
        "metadata": _canonicalize_block(body.get("metadata")) if body.get("metadata") else None,
    }


def assert_anthropic_bodies_equivalent(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    """Semantic equality of two Anthropic Messages bodies.

    Tolerates: dict-key ordering, ``None``/missing-key swap, ``content``
    string ↔ single-block-list normalization, ``system`` string ↔
    block-list normalization (and uniform-cache concatenation), implicit
    ``tool_choice = auto`` when tools are present, redundant
    ``is_error: False`` defaults on tool_result blocks. Asserts equality
    on ``model``, ``max_tokens``, ``tools``, ``messages``, ``system``,
    and the sampling settings.
    """
    expected_norm = _build_normalised_view(expected)
    actual_norm = _build_normalised_view(actual)
    assert actual_norm == expected_norm, (
        f"Bodies differ:\nexpected={json.dumps(expected_norm, indent=2, sort_keys=True)}\n"
        f"actual={json.dumps(actual_norm, indent=2, sort_keys=True)}"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundtripCase:
    name: str
    """Test ID."""

    body: dict[str, Any]
    """Anthropic Messages body to roundtrip."""


_ROUNDTRIP_CASES: list[RoundtripCase] = [
    RoundtripCase(
        name="simple_text_user_message",
        body={
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
        },
    ),
    RoundtripCase(
        name="multi_turn_with_tool_use",
        body={
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 2048,
            "messages": [
                {"role": "user", "content": "what is 2+2?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me compute."},
                        {
                            "type": "tool_use",
                            "id": "tc_abc",
                            "name": "calc",
                            "input": {"a": 2, "b": 2},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tc_abc",
                            "content": [{"type": "text", "text": "4"}],
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "calc",
                    "description": "Add two numbers",
                    "input_schema": {
                        "type": "object",
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    },
                }
            ],
        },
    ),
    RoundtripCase(
        name="system_as_string",
        body={
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 1024,
            "system": "Be concise.",
            "messages": [{"role": "user", "content": "hi"}],
        },
    ),
    RoundtripCase(
        name="system_as_uniform_cache_blocks",
        body={
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "Block one.", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
                {"type": "text", "text": "Block two.", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
            ],
            "messages": [{"role": "user", "content": "go"}],
        },
    ),
    RoundtripCase(
        name="system_as_non_uniform_cache_blocks",
        body={
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "Cached block.", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
                {"type": "text", "text": "Uncached block."},
            ],
            "messages": [{"role": "user", "content": "go"}],
        },
    ),
    RoundtripCase(
        name="sampling_settings",
        body={
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["</done>"],
            "messages": [{"role": "user", "content": "x"}],
        },
    ),
    RoundtripCase(
        name="image_with_media_type",
        body={
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 256,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                # 1x1 transparent PNG
                                "data": (
                                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAA"
                                    "C0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
                                ),
                            },
                        },
                    ],
                }
            ],
        },
    ),
]


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.name) for c in _ROUNDTRIP_CASES],
)
async def test_roundtrip_semantic_equivalence(case: RoundtripCase) -> None:
    """``parse → render`` produces a body semantically equal to the input."""
    parsed = await parse_anthropic_messages(case.body)
    rendered = await render_anthropic(parsed)
    rebuilt = json.loads(rendered)
    assert_anthropic_bodies_equivalent(case.body, rebuilt)


def _summarise_part(part: Any) -> dict[str, Any]:
    """Return a timestamp-free summary of a pydantic-ai message part.

    The IR carries auto-generated ``timestamp`` fields that differ
    between parses; we strip them before comparing. ``UserPromptPart``
    normalises bare-string content into a single-item list so the
    string ↔ list-of-strings normalisation that the inbound parser
    performs after a round-trip doesn't trigger a false negative.
    """
    summary: dict[str, Any] = {"_type": type(part).__name__}
    for attr in ("content", "tool_name", "tool_call_id", "args", "signature"):
        if hasattr(part, attr):
            value = getattr(part, attr)
            summary[attr] = _summarise_value(value)
    if summary["_type"] == "UserPromptPart":
        content = summary.get("content")
        if isinstance(content, str):
            summary["content"] = [content]
    return summary


def _summarise_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_summarise_value(v) for v in value]
    # pydantic-ai content items (BinaryContent, ImageUrl, CachePoint, ...)
    # carry stable attributes — represent them by class + data fields.
    if hasattr(value, "__class__") and value.__class__.__module__.startswith("pydantic_ai"):
        out: dict[str, Any] = {"_type": type(value).__name__}
        for attr in ("data", "media_type", "url", "ttl"):
            if hasattr(value, attr):
                attr_value = getattr(value, attr)
                out[attr] = _summarise_value(attr_value)
        return out
    return value


def _fold_system_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse consecutive ``SystemPromptPart`` entries into one block.

    Uniform-cache system blocks compress into a single concatenated
    block at render time; reparsing produces one ``SystemPromptPart``
    versus the original's many. Folding here makes the IR-level
    comparison agnostic to the count.
    """
    folded: list[dict[str, Any]] = []
    for part in parts:
        if (
            folded
            and part.get("_type") == "SystemPromptPart"
            and folded[-1].get("_type") == "SystemPromptPart"
            and isinstance(part.get("content"), str)
            and isinstance(folded[-1].get("content"), str)
        ):
            folded[-1] = {
                **folded[-1],
                "content": f"{folded[-1]['content']}\n\n{part['content']}",
            }
        else:
            folded.append(part)
    return folded


def _summarise_messages(messages: list[Any]) -> list[Any]:
    return [
        {"_type": type(m).__name__, "parts": _fold_system_parts([_summarise_part(p) for p in m.parts])}
        for m in messages
    ]


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.name) for c in _ROUNDTRIP_CASES],
)
async def test_roundtrip_ir_idempotent(case: RoundtripCase) -> None:
    """Re-parsing the rendered body yields the same IR (timestamps stripped)."""
    parsed_original = await parse_anthropic_messages(case.body)
    rendered = await render_anthropic(parsed_original)
    parsed_again = await parse_anthropic_messages(json.loads(rendered))

    assert parsed_again.model == parsed_original.model
    assert _summarise_messages(parsed_again.messages) == _summarise_messages(parsed_original.messages)
    assert parsed_again.request_parameters == parsed_original.request_parameters


# ---------------------------------------------------------------------------
# Render output contract
# ---------------------------------------------------------------------------


async def test_render_returns_bytes() -> None:
    parsed = await parse_anthropic_messages(
        {"model": "claude-3-5-haiku-20241022", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    )
    rendered = await render_anthropic(parsed)
    assert isinstance(rendered, bytes)
    json.loads(rendered)  # well-formed JSON


async def test_render_compact_json() -> None:
    """Rendered output is compact JSON (no insignificant whitespace)."""
    parsed = await parse_anthropic_messages(
        {"model": "claude-3-5-haiku-20241022", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    )
    rendered = await render_anthropic(parsed)
    assert b": " not in rendered
    assert b", " not in rendered


async def test_render_strips_sdk_control_fields() -> None:
    """Rendered body never carries the SDK-only kwargs (extra_headers, betas, etc.)."""
    parsed = await parse_anthropic_messages(
        {"model": "claude-3-5-haiku-20241022", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    )
    rendered = json.loads(await render_anthropic(parsed))
    for forbidden in ("extra_headers", "extra_body", "extra_query", "timeout", "betas"):
        assert forbidden not in rendered, f"SDK control field {forbidden!r} leaked into body"


async def test_render_strips_omit_sentinels() -> None:
    """No anthropic.Omit / NotGiven sentinels survive into the JSON output."""
    parsed = await parse_anthropic_messages(
        {"model": "claude-3-5-haiku-20241022", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    )
    rendered = json.loads(await render_anthropic(parsed))
    # Top-level — only fields the user supplied should be present.
    # No empty/null leakage from the SDK Omit handling.
    for key, value in rendered.items():
        assert value is not None, f"Field {key!r} is None — Omit handling leaked"


# ---------------------------------------------------------------------------
# Raw extras overrides
# ---------------------------------------------------------------------------


async def test_non_uniform_system_cache_control_preserved() -> None:
    """Mixed system cache_control roundtrips via raw_extras['system']."""
    body = {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 256,
        "system": [
            {"type": "text", "text": "First", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
            {"type": "text", "text": "Second"},
        ],
        "messages": [{"role": "user", "content": "go"}],
    }
    parsed = await parse_anthropic_messages(body)
    # The inbound parser stashes the original blocks for non-uniform cache_control.
    assert "system" in parsed.raw_extras

    rendered = json.loads(await render_anthropic(parsed))
    assert rendered["system"] == body["system"]


async def test_metadata_preserved_via_raw_extras() -> None:
    body = {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"user_id": "alice"},
    }
    parsed = await parse_anthropic_messages(body)
    rendered = json.loads(await render_anthropic(parsed))
    assert rendered.get("metadata") == {"user_id": "alice"}


# ---------------------------------------------------------------------------
# CaptureSentinel
# ---------------------------------------------------------------------------


def test_capture_sentinel_carries_kwargs() -> None:
    kwargs = {"max_tokens": 1, "messages": []}
    sentinel = CaptureSentinel(kwargs)
    assert sentinel.kwargs is kwargs
    assert str(sentinel) == "captured"
