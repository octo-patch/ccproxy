"""Inject auth hook — sentinel key substitution and token injection.

Detects ``sk-ant-oat-ccproxy-{provider}`` sentinel keys on inbound auth
headers and applies the selected Provider's static request fields and
credential placement. LiteLLM model bindings use the same injection service,
so sentinel and model-selected routes share one auth implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ccproxy.config import Provider, get_config
from ccproxy.constants import AUTH_SENTINEL_PREFIX, AuthConfigError
from ccproxy.pipeline.hook import hook

if TYPE_CHECKING:
    from ccproxy.pipeline.context import Context

logger = logging.getLogger(__name__)


_INBOUND_AUTH_HEADERS: tuple[str, ...] = ("x-api-key", "x-goog-api-key", "api-key", "authorization")
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
    reads=["authorization", "x-api-key", "x-goog-api-key", "api-key"],
    writes=["authorization", "x-api-key", "x-goog-api-key", "api-key"],
)
def inject_auth(ctx: Context, _: dict[str, Any]) -> Context:
    """Forward an auth token to the provider, substituting a sentinel key."""
    sentinel = _extract_sentinel(ctx)
    if sentinel is None:
        return ctx

    provider = sentinel[len(AUTH_SENTINEL_PREFIX) :]
    inject_provider_auth(ctx, provider, require_auth=True)
    logger.info("Auth token injected for provider '%s' (sentinel)", provider)
    return ctx


def inject_provider_auth(
    ctx: Context,
    provider_name: str,
    provider: Provider | None = None,
    *,
    token: str | None = None,
    force: bool = False,
    require_auth: bool = False,
) -> None:
    """Apply one resolved Provider's static request fields and credentials.

    Sentinel-selected and LiteLLM-model-selected routes share this function.
    Header and query placement are both supported; all inbound credential
    headers are cleared before the configured value is stamped.
    """
    config = None
    if provider is None or token is None:
        try:
            config = get_config()
        except AuthConfigError:
            raise
        except Exception as exc:
            raise AuthConfigError(f"Failed to load auth config for provider '{provider_name}': {exc}") from exc
    resolved_provider = provider or (config.get_provider(provider_name) if config is not None else None)
    if resolved_provider is None:
        raise AuthConfigError(
            f"No provider configuration for '{provider_name}'. Add a matching provider to ccproxy.yaml."
        )
    if ctx.metadata.auth_injected and not force and ctx.metadata.auth_provider == provider_name:
        return

    previous_query_param = ctx.metadata.auth_query_param
    if previous_query_param and ctx.flow is not None:
        ctx.flow.request.query.pop(previous_query_param, None)
        ctx.metadata.auth_query_param = ""

    for header, value in resolved_provider.headers.items():
        ctx.set_header(header, value)
    if ctx.flow is not None:
        for key, value in resolved_provider.query.items():
            ctx.flow.request.query[key] = value

    if resolved_provider.auth is None:
        if ctx.metadata.auth_injected:
            for header in _INBOUND_AUTH_HEADERS:
                ctx.set_header(header, "")
            ctx.metadata.auth_injected = False
        if require_auth:
            raise AuthConfigError(f"Provider '{provider_name}' has no auth source for sentinel substitution")
        ctx.metadata.auth_provider = provider_name
        return
    resolved_token: str | None
    try:
        if token is not None:
            resolved_token = token
        else:
            assert config is not None
            resolved_token = config.resolve_provider_auth(provider_name, resolved_provider)
    except AuthConfigError:
        raise
    except Exception as exc:
        raise AuthConfigError(f"Failed to resolve auth for provider '{provider_name}': {exc}") from exc
    if not resolved_token:
        raise AuthConfigError(f"Provider '{provider_name}' has an auth source but it resolved no credential")

    for header in _INBOUND_AUTH_HEADERS:
        ctx.set_header(header, "")

    target_header = resolved_provider.auth.header
    target_query = resolved_provider.auth.query_param
    if target_query is not None:
        if ctx.flow is None:
            raise AuthConfigError(f"Provider '{provider_name}' uses query auth without an HTTP flow")
        ctx.flow.request.query[target_query] = resolved_token
        ctx.metadata.auth_query_param = target_query
    elif target_header is None or target_header.lower() == "authorization":
        ctx.set_header("authorization", f"Bearer {resolved_token}")
    else:
        ctx.set_header(target_header, resolved_token)

    for header, value in resolved_provider.auth.extra_headers(f"Auth/{provider_name}").items():
        ctx.set_header(header, value)

    ctx.metadata.auth_provider = provider_name
    ctx.metadata.auth_injected = True
