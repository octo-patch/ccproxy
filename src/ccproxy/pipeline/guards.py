"""Shared guard functions for pipeline hooks.

These guards use header presence and protocol-level header shape where useful.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ccproxy.pipeline.context import Context


def is_auth_request(ctx: Context) -> bool:
    """Check if request uses an Authorization bearer token."""
    auth_header = ctx.authorization.lower()
    return auth_header.startswith("bearer ")


def is_anthropic_destination(ctx: Context) -> bool:
    """Check if the flow targets an Anthropic API endpoint."""
    return ctx.get_header("anthropic-version") != ""
