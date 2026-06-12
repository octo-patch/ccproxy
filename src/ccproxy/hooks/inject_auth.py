"""Inject auth hook — sentinel key substitution and token injection.

Detects ``sk-ant-oat-ccproxy-{provider}`` sentinel keys on any inbound
auth header (``x-api-key``, ``x-goog-api-key``, or ``Authorization: Bearer``),
resolves the real auth token from ``CCProxyConfig.providers[provider]``,
and injects it via the header named on that Provider's ``auth.header``
(defaulting to ``Authorization: Bearer`` when unset). All non-target inbound
auth headers are cleared so the sentinel never leaks upstream.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ccproxy.config import get_config
from ccproxy.constants import AUTH_SENTINEL_PREFIX, AuthConfigError
from ccproxy.pipeline.hook import hook

if TYPE_CHECKING:
    from ccproxy.pipeline.context import Context

logger = logging.getLogger(__name__)


_INBOUND_AUTH_HEADERS: tuple[str, ...] = ("x-api-key", "x-goog-api-key", "authorization")
"""Headers checked inbound for a sentinel key, in priority order. ``authorization``
is matched against its bare token after stripping a ``Bearer `` prefix."""


def inject_auth_guard(ctx: Context) -> bool:
    """Guard: run if any inbound auth header carries a value."""
    return bool(ctx.x_api_key or ctx.authorization or ctx.get_header("x-goog-api-key") or ctx.get_header("api-key"))


def _bearer_token(value: str) -> str:
    """Strip a leading ``Bearer `` (case-insensitive) from an Authorization value."""
    if value.lower().startswith("bearer "):
        return value[7:].strip()
    return value


def _extract_sentinel(ctx: Context) -> str | None:
    """Return the sentinel-key value from any inbound auth header, or None."""
    for header in _INBOUND_AUTH_HEADERS:
        raw = ctx.get_header(header, "")
        candidate = _bearer_token(raw) if header == "authorization" else raw
        if candidate.startswith(AUTH_SENTINEL_PREFIX):
            return candidate
    return None


@hook(
    reads=["authorization", "x-api-key", "x-goog-api-key"],
    writes=["authorization", "x-api-key", "x-goog-api-key"],
)
def inject_auth(ctx: Context, _: dict[str, Any]) -> Context:
    """Forward an auth token to the provider, substituting a sentinel key."""
    sentinel = _extract_sentinel(ctx)
    if sentinel is None:
        return ctx

    provider = sentinel[len(AUTH_SENTINEL_PREFIX) :]
    token = _get_auth_token(provider)

    if not token:
        raise AuthConfigError(
            f"Sentinel key for provider '{provider}' but no matching providers entry. "
            f"Add 'providers.{provider}' to ccproxy.yaml."
        )

    _inject_token(ctx, provider, token)
    ctx.metadata.auth_provider = provider
    logger.info("Auth token injected for provider '%s' (sentinel)", provider)
    return ctx


def _get_auth_token(provider: str) -> str | None:
    """Resolve the provider's token; config failures are fatal, not silent.

    A config that cannot load or resolve must surface as ``AuthConfigError``
    (the one exception the pipeline executor propagates) rather than letting
    the request continue unauthenticated toward a deferred upstream 401.
    """
    try:
        config = get_config()
        return config.resolve_auth_token(provider)
    except AuthConfigError:
        raise
    except Exception as exc:
        raise AuthConfigError(f"Failed to load auth config for provider '{provider}': {exc}") from exc


def _inject_token(ctx: Context, provider: str, token: str) -> None:
    """Inject ``token`` into the configured outbound auth header.

    The provider's ``auth.header`` (None defaults to ``authorization``) wins.
    All other inbound auth headers are cleared so the sentinel never leaks
    upstream alongside the real token.
    """
    config = get_config()
    target_header = (config.get_auth_header(provider) or "authorization").lower()

    if target_header == "authorization":
        ctx.set_header("authorization", f"Bearer {token}")
    else:
        ctx.set_header(target_header, token)

    for header, value in config.get_auth_extra_headers(provider).items():
        ctx.set_header(header, value)

    for header in _INBOUND_AUTH_HEADERS:
        if header != target_header:
            ctx.set_header(header, "")

    ctx.metadata.auth_injected = True
