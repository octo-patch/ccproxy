"""Shared shape preparation helpers.

Runtime shaping and packaging both need the same apply-time preparation:
strip configured headers, inject incoming content fields, run the provider's
shape hooks, and commit the working request.
"""

from __future__ import annotations

from typing import Any

from glom import assign, delete

from ccproxy.config import ProviderShapingConfig
from ccproxy.pipeline.context import Context
from ccproxy.shaping.executor import execute_shape_hooks
from ccproxy.shaping.prepare import strip_headers


def prepare_shape(
    shape_ctx: Context,
    incoming_ctx: Context,
    profile: ProviderShapingConfig,
) -> Context:
    """Prepare a captured shape against an incoming request context."""
    strip_headers(shape_ctx, profile.strip_headers)
    inject_content(shape_ctx, incoming_ctx, profile)
    shape_ctx = execute_shape_hooks(shape_ctx, incoming_ctx, profile.shape_hooks)
    shape_ctx.commit()
    return shape_ctx


def parse_strategy(raw: str) -> tuple[str, int | None]:
    """Parse ``"prepend_shape:2"`` into ``("prepend_shape", 2)``."""
    if ":" in raw:
        name, _, param = raw.partition(":")
        return name, int(param)
    return raw, None


def inject_content(
    shape_ctx: Context,
    incoming_ctx: Context,
    profile: ProviderShapingConfig,
) -> None:
    """Strip content fields from shape, then fill from incoming per merge strategy."""
    shape_originals: dict[str, Any] = {}
    for key in profile.content_fields:
        strategy, _ = parse_strategy(profile.merge_strategies.get(key, "replace"))
        if strategy in ("prepend_shape", "append_shape") and key in shape_ctx._body:
            shape_originals[key] = shape_ctx._body[key]
        delete(shape_ctx._body, key, ignore_missing=True)

    for key in profile.content_fields:
        strategy, slice_n = parse_strategy(profile.merge_strategies.get(key, "replace"))
        if strategy == "replace":
            if key in incoming_ctx._body:
                assign(shape_ctx._body, key, incoming_ctx._body[key])
        elif strategy == "prepend_shape":
            incoming_val = incoming_ctx._body.get(key) or []
            shape_val = shape_originals.get(key) or []
            if isinstance(shape_val, str):
                shape_val = [{"type": "text", "text": shape_val}]
            if isinstance(incoming_val, str):
                incoming_val = [{"type": "text", "text": incoming_val}]
            if slice_n is not None:
                shape_val = shape_val[:slice_n]
            assign(shape_ctx._body, key, [*shape_val, *incoming_val])
        elif strategy == "append_shape":
            incoming_val = incoming_ctx._body.get(key) or []
            shape_val = shape_originals.get(key) or []
            if isinstance(shape_val, str):
                shape_val = [{"type": "text", "text": shape_val}]
            if isinstance(incoming_val, str):
                incoming_val = [{"type": "text", "text": incoming_val}]
            if slice_n is not None:
                shape_val = shape_val[:slice_n]
            assign(shape_ctx._body, key, [*incoming_val, *shape_val])
        elif strategy == "drop":
            pass
