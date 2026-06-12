"""Codex/OpenAI Responses shape hooks."""

from __future__ import annotations

from typing import Any

from ccproxy.pipeline.context import Context
from ccproxy.pipeline.hook import hook


@hook(reads=["input"], writes=["input"])
def normalize_responses_input(ctx: Context, params: dict[str, Any]) -> Context:
    """Normalize public Responses SDK shorthand into Codex's list input form."""
    value = ctx._body.get("input")
    if not isinstance(value, str):
        return ctx

    ctx._body["input"] = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": value}],
        }
    ]
    return ctx


@hook(reads=["store"], writes=["store"])
def enforce_codex_store_false(ctx: Context, params: dict[str, Any]) -> Context:
    """Codex's ChatGPT backend requires non-stored Responses requests."""
    ctx._body["store"] = False
    return ctx
