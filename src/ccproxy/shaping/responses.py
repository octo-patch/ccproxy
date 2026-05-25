"""OpenAI Responses shape hooks."""

from __future__ import annotations

from typing import Any

from ccproxy.pipeline.context import Context
from ccproxy.pipeline.hook import hook


@hook(reads=["input"], writes=["input"])
def replace_body_from_incoming(ctx: Context, params: dict[str, Any]) -> Context:
    """Use the live Responses body while keeping the captured request envelope."""
    incoming_ctx = params.get("incoming_ctx")
    if incoming_ctx is None:
        return ctx
    ctx._body = dict(incoming_ctx._body) if isinstance(incoming_ctx._body, dict) else {}
    return ctx
