"""Apply the Anthropic cache-breakpoint policy engine to outbound requests.

Thin wire-side wrapper around
:func:`ccproxy.lightllm.cache_policy.apply_cache_policy`. Policy is
resolved per-request from the ``x-ccproxy-cache-policy`` control header,
falling back to hook ``params`` from ``ccproxy.yaml``; the guard returns
False when neither yields a policy or when the resolved upstream provider
is not in the anthropic-compatible family.

Header DSL — comma-separated ``key[=value]`` pairs:

* ``tools[=ttl]`` — marker on the tool set (default ``5m``)
* ``system[=ttl]`` — marker after the last system block (default ``5m``)
* ``user_tail=N[:ttl]`` — markers on the last N user messages (default ``5m``)

Example: ``x-ccproxy-cache-policy: tools=1h,system=1h,user_tail=2:5m``.
Parse errors raise ``ValueError`` naming the bad token — the executor
records an error ``HookResult``; there is no silent no-op. The control
header is deleted before the request leaves ccproxy.

DAG position: late in the outbound stage — after content-mutating hooks
(system injection, thread injection) so placements land on the final
conversation shape, before commit. ``reads``/``writes`` on ``messages``
and ``settings`` let the DAG order it after any hook that writes those
keys.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel

from ccproxy.config import get_config
from ccproxy.lightllm.cache_policy import CachePolicy, apply_cache_policy
from ccproxy.lightllm.graph import _ANTHROPIC_COMPATIBLE
from ccproxy.pipeline.hook import hook

if TYPE_CHECKING:
    from ccproxy.pipeline.context import Context

logger = logging.getLogger(__name__)

__all__ = ["cache_breakpoints", "cache_breakpoints_guard", "parse_policy_header"]

POLICY_HEADER = "x-ccproxy-cache-policy"

_TTLS = ("5m", "1h")


class CacheBreakpointsParams(BaseModel):
    """YAML ``params`` schema for the ``cache_breakpoints`` hook."""

    tools: Literal["5m", "1h"] | None = None
    system: Literal["5m", "1h"] | None = None
    user_tail: int = 0
    user_tail_ttl: Literal["5m", "1h"] = "5m"


def parse_policy_header(value: str) -> CachePolicy:
    """Parse the ``x-ccproxy-cache-policy`` header DSL into a :class:`CachePolicy`.

    Raises:
        ValueError: Naming the offending token on any malformed input.
    """
    tools: Literal["5m", "1h"] | None = None
    system: Literal["5m", "1h"] | None = None
    user_tail = 0
    user_tail_ttl: Literal["5m", "1h"] = "5m"

    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        key, _, val = token.partition("=")
        if key in ("tools", "system"):
            ttl = val or "5m"
            if ttl not in _TTLS:
                raise ValueError(f"invalid cache-policy token {token!r}: ttl must be 5m or 1h")
            if key == "tools":
                tools = cast(Literal["5m", "1h"], ttl)
            else:
                system = cast(Literal["5m", "1h"], ttl)
        elif key == "user_tail":
            count_str, _, ttl = val.partition(":")
            if not count_str.isdigit():
                raise ValueError(f"invalid cache-policy token {token!r}: expected user_tail=N[:ttl]")
            if ttl and ttl not in _TTLS:
                raise ValueError(f"invalid cache-policy token {token!r}: ttl must be 5m or 1h")
            user_tail = int(count_str)
            if ttl:
                user_tail_ttl = cast(Literal["5m", "1h"], ttl)
        else:
            raise ValueError(f"invalid cache-policy token {token!r}: unknown key")

    return CachePolicy(tools=tools, system=system, user_tail=user_tail, user_tail_ttl=user_tail_ttl)


def _config_params() -> dict[str, Any]:
    """Return this hook's ``params`` from the loaded config's hook lists."""
    for entries in get_config().hooks.values():
        for entry in entries:
            if isinstance(entry, dict) and str(entry.get("hook", "")).endswith("cache_breakpoints"):
                params = entry.get("params")
                if isinstance(params, dict):
                    return params
    return {}


def _provider_is_anthropic(ctx: Context) -> bool:
    provider = get_config().providers.get(ctx.metadata.auth_provider)
    return provider is not None and provider.type in _ANTHROPIC_COMPATIBLE


def cache_breakpoints_guard(ctx: Context) -> bool:
    """Run when a policy is resolvable and the upstream is anthropic-family."""
    has_header = bool(ctx.get_header(POLICY_HEADER))
    if not has_header and not _config_params():
        return False
    if not _provider_is_anthropic(ctx):
        if has_header:
            logger.info(
                "cache_breakpoints: skipping — provider %r is not anthropic-compatible",
                ctx.metadata.auth_provider,
            )
        return False
    return True


@hook(
    reads=["messages", "settings"],
    writes=["messages", "settings"],
    model=CacheBreakpointsParams,
)
def cache_breakpoints(ctx: Context, params: dict[str, Any]) -> Context:
    """Resolve the cache policy and apply the placement engine to the IR."""
    header_value = ctx.get_header(POLICY_HEADER)
    if header_value:
        ctx.set_header(POLICY_HEADER, "")
        policy = parse_policy_header(header_value)
    else:
        source = params or _config_params()
        validated = CacheBreakpointsParams(**source)
        policy = CachePolicy(
            tools=validated.tools,
            system=validated.system,
            user_tail=validated.user_tail,
            user_tail_ttl=validated.user_tail_ttl,
        )

    new_messages, new_settings, report = apply_cache_policy(
        ctx.messages,
        ctx.settings,
        policy,
        raw_extras=ctx.raw_extras,
    )
    ctx.messages = new_messages
    ctx.settings = new_settings
    logger.debug(
        "cache_breakpoints: existing=%d placed=%s dropped=%s skipped=%s",
        report.existing,
        report.placed,
        report.dropped,
        report.skipped,
    )
    return ctx
