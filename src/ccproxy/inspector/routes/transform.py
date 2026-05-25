"""Transform route — sentinel-driven Provider routing + optional override layer.

Routing precedence on every inbound request:

    1. ``inspector.transforms`` — first regex-matched override wins.
    2. ccproxy metadata ``auth_provider`` — set by ``inject_auth`` when a
       sentinel key resolved. Looks up :class:`CCProxyConfig.providers`.
    3. None — :class:`mitmproxy.proxy.mode_specs.ReverseMode` flows return
       OpenAI-shape 501; WireGuard flows pass through unchanged.

Three actions:

    - ``transform``: rewrite the request body via lightllm dispatch (cross-format).
    - ``redirect``: rewrite destination only, preserve body (same-format).
    - ``passthrough``: forward unchanged.

For sentinel-resolved Provider targets, the action is auto-derived: when
``_detect_incoming_format`` matches ``provider.provider.value`` it's redirect,
otherwise transform.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Literal

from glom import glom
from mitmproxy.connection import Server
from mitmproxy.proxy.mode_specs import ReverseMode

from ccproxy.config import Provider, TransformOverride, get_config
from ccproxy.flows.store import TransformMeta
from ccproxy.lightllm.graph import _ANTHROPIC_COMPATIBLE
from ccproxy.pipeline.context import metadata_from_flow

if TYPE_CHECKING:
    from mitmproxy.http import HTTPFlow

    from ccproxy.inspector.router import InspectorRouter

logger = logging.getLogger(__name__)


_ACTION_RE = re.compile(r":(\w+)(?:$|\?)")
_MODEL_FROM_PATH_RE = re.compile(r"/models/([^/:]+)")

_FORMAT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^/v1/chat/completions(?:/|$)"), "openai"),
    (re.compile(r"^/(?:anthropic/)?v1/messages(?:/|$)"), "anthropic"),
    (re.compile(r"^/(?:gemini/)?v1beta/models/[^/]+:"), "gemini"),
    (re.compile(r"^/(?:gemini/)?v1alpha/models/[^/]+:"), "gemini"),
    (re.compile(r"^/v1internal:"), "gemini"),
    (re.compile(r"^/(?:v1/|backend-api/codex/)?responses(?:/|$)"), "openai_responses"),
)
"""URL-prefix patterns ccproxy recognises as a known wire format."""

_GEMINI_FORMATS: frozenset[str] = frozenset({"gemini", "vertex_ai", "vertex_ai_beta"})


def _openai_error(message: str, *, error_type: str, code: int) -> bytes:
    """Serialize an OpenAI-shape error envelope for synthetic responses."""
    return json.dumps(
        {
            "error": {"message": message, "type": error_type, "code": code},
        }
    ).encode()


def _detect_incoming_format(path: str) -> str | None:
    """Return the wire format ccproxy thinks the incoming request speaks.

    ``"openai"`` for OpenAI Chat Completions; ``"anthropic"`` for Messages
    (including DeepSeek's anthropic-compat endpoint); ``"gemini"`` for both
    v1beta and the cloudcode-pa v1internal envelope; ``None`` for unknown.
    """
    for pattern, name in _FORMAT_PATTERNS:
        if pattern.search(path):
            return name
    return None


def _flow_hosts(flow: HTTPFlow) -> set[str]:
    hosts: set[str] = {flow.request.pretty_host}
    for header in ("host", "x-forwarded-host"):
        value = flow.request.headers.get(header, "")
        if value:
            hosts.add(value.split(":")[0])
    return hosts


def _any_search(pattern: re.Pattern[str], values: set[str]) -> bool:
    return any(pattern.search(v) for v in values)


def _action_from_path(path: str) -> str | None:
    match = _ACTION_RE.search(path.split("?")[0])
    return match.group(1) if match else None


def _model_for_routing(body: dict[str, object], path: str) -> str:
    body_model = str(glom(body, "model", default=""))
    if body_model:
        return body_model
    match = _MODEL_FROM_PATH_RE.search(path)
    return match.group(1) if match else ""


def _apply_path_template(template: str, *, model: str, action: str | None) -> str:
    out = template
    if "{model}" in out:
        out = out.replace("{model}", model)
    if "{action}" in out:
        out = out.replace("{action}", action or "")
    return out


def _resolve_transform_target(
    flow: HTTPFlow,
    body: dict[str, object] | None = None,
) -> Provider | TransformOverride | None:
    """Pick the routing target. First match wins; None means no signal."""
    config = get_config()
    request_model = str(glom(body or {}, "model", default=""))

    for rule in config.inspector.transforms:
        if rule.match_host_re and not _any_search(rule.match_host_re, _flow_hosts(flow)):
            continue
        if not rule.match_path_re.search(flow.request.path):
            continue
        if rule.match_model_re and not rule.match_model_re.search(request_model):
            continue
        return rule

    auth_provider = metadata_from_flow(flow).auth_provider
    if auth_provider:
        return config.providers.get(auth_provider)

    return None


def _record_transform_meta(
    flow: HTTPFlow,
    *,
    provider_type: str,
    model: str,
    body: dict[str, object],
    is_streaming: bool,
    mode: Literal["redirect", "transform"],
) -> None:
    metadata = metadata_from_flow(flow)
    record = metadata.record
    if record is None:
        return
    record.transform = TransformMeta(
        provider_type=provider_type,
        model=model,
        request_data={**body},
        is_streaming=is_streaming,
        mode=mode,
        inbound_format=metadata.inbound_format,
        request_parameters=metadata.request_parameters,
    )


def _apply_destination(flow: HTTPFlow, host: str, path: str) -> None:
    flow.request.host = host
    flow.request.port = 443
    flow.request.scheme = "https"
    flow.request.path = path
    flow.server_conn = Server(address=(host, 443))


def _handle_passthrough(flow: HTTPFlow) -> None:
    logger.info(
        "transform passthrough: → %s:%d%s",
        flow.request.host,
        flow.request.port,
        flow.request.path,
    )


def _handle_redirect(
    flow: HTTPFlow,
    target: Provider | TransformOverride,
    body: dict[str, object],
) -> None:
    """Same-format redirect: rewrite host/path, preserve body."""
    is_streaming = bool(glom(body, "stream", default=False))
    action = _action_from_path(flow.request.path)
    config = get_config()

    host: str
    path: str
    if isinstance(target, Provider):
        provider_str = target.type
        model = _model_for_routing(body, flow.request.path)
        host = target.host
        path = _apply_path_template(target.path, model=model, action=action)
        api_key: str | None = None  # auth already stamped by inject_auth
    else:
        bound = config.providers.get(target.dest_provider) if target.dest_provider else None
        resolved_host = target.dest_host or (bound.host if bound else None)
        if resolved_host is None:
            logger.error(
                "redirect override missing dest_host and no resolvable dest_provider; passthrough",
            )
            return
        host = resolved_host
        provider_str = (bound.type if bound else target.dest_provider) or ""
        model = target.dest_model or _model_for_routing(body, flow.request.path)
        if target.dest_path:
            path = _apply_path_template(target.dest_path, model=model, action=action)
        elif bound is not None:
            path = _apply_path_template(bound.path, model=model, action=action)
        else:
            path = flow.request.path
        api_key = config.resolve_auth_token(target.dest_provider) if target.dest_provider else None

    _record_transform_meta(
        flow,
        provider_type=provider_str,
        model=model,
        body=body,
        is_streaming=is_streaming,
        mode="redirect",
    )

    _apply_destination(flow, host, path)
    if api_key:
        flow.request.headers["authorization"] = f"Bearer {api_key}"

    flow.comment = f"redirect → {provider_str}/{host}"
    logger.info("redirect: → %s %s%s", provider_str, host, path)


def _action_for_transform(provider_type: str, *, is_streaming: bool) -> str | None:
    """Resolve the ``{action}`` URL template substitution for a transform target.

    Gemini-family upstreams template the SDK action into their path
    (``:streamGenerateContent`` vs ``:generateContent``); other providers
    have no ``{action}`` slot so the resolved value is ``None`` (the path
    template's ``_apply_path_template`` no-ops in that case).
    """
    if provider_type in _GEMINI_FORMATS:
        return "streamGenerateContent" if is_streaming else "generateContent"
    return None


def _build_upstream_url_and_headers(
    *,
    target: Provider | TransformOverride,
    bound: Provider | None,
    model: str,
    provider_type: str,
    is_streaming: bool,
) -> tuple[str, dict[str, str]]:
    """Build the upstream ``(url, headers)`` for a transform-mode dispatch.

    Pulls host/path from the resolved target (``Provider`` or
    ``TransformOverride`` with optional ``dest_host`` / ``dest_path`` overrides
    falling back to the bound Provider). Auth headers are already stamped by
    the ``inject_auth`` inbound hook — this builder only adds the
    Anthropic-compat ``anthropic-version`` floor.
    """
    action = _action_for_transform(provider_type, is_streaming=is_streaming)

    host: str
    path_template: str
    if isinstance(target, Provider):
        host = target.host
        path_template = target.path
    else:
        resolved_host = target.dest_host or (bound.host if bound is not None else None)
        if resolved_host is None:
            raise ValueError(
                "transform override missing dest_host and no resolvable dest_provider",
            )
        host = resolved_host
        path_template = target.dest_path or (bound.path if bound is not None else "/")

    path = _apply_path_template(path_template, model=model, action=action)
    url = f"https://{host}{path}"

    headers: dict[str, str] = {}
    if provider_type in _ANTHROPIC_COMPATIBLE:
        # Defensive floor for cross-format flows targeting an Anthropic upstream
        # where no Anthropic shape replay runs. inject_auth has already stamped
        # auth; the shape hook adds the canonical Claude headers when present.
        headers["anthropic-version"] = "2023-06-01"
    return url, headers


def _handle_transform(
    flow: HTTPFlow,
    target: Provider | TransformOverride,
    body: dict[str, object],
) -> None:
    """Cross-format transform: render the body via ``dispatch_dump_sync`` and
    rewrite the destination.

    All providers (Anthropic-compatible, OpenAI, Gemini-family, Perplexity Pro)
    route through pydantic-ai's IR via :class:`~ccproxy.pipeline.context.Context.parse_sync`
    + :func:`dispatch_dump_sync`. URL + headers come from the resolved
    :class:`Provider` config (host/path with ``{model}`` / ``{action}`` templating)
    or the :class:`TransformOverride` overrides.
    """
    # deferred: avoid pulling pydantic-ai at module import time
    from ccproxy.lightllm.graph import dispatch_dump_sync
    from ccproxy.pipeline.context import Context

    is_streaming = bool(glom(body, "stream", default=False))
    config = get_config()

    bound: Provider | None
    if isinstance(target, Provider):
        provider_str = target.type
        model = _model_for_routing(body, flow.request.path)
        bound = target
    else:
        if target.dest_provider is None:
            logger.error("transform override missing dest_provider; passthrough")
            return
        bound = config.providers.get(target.dest_provider)
        if bound is None:
            logger.error(
                "transform override dest_provider '%s' not in config.providers; passthrough",
                target.dest_provider,
            )
            return
        provider_str = bound.type
        model = target.dest_model or _model_for_routing(body, flow.request.path)

    ctx = Context.from_flow(flow)
    if "inbound_format" not in ctx.metadata:
        ctx.metadata.inbound_format = ctx._inbound_format.value
    ctx.parse_sync()
    if model and model != ctx.model:
        ctx.model = model
    ctx.metadata.request_parameters = ctx.request_parameters
    new_body = dispatch_dump_sync(ctx, provider_type=provider_str)

    try:
        url, headers = _build_upstream_url_and_headers(
            target=target,
            bound=bound,
            model=model,
            provider_type=provider_str,
            is_streaming=is_streaming,
        )
    except ValueError as exc:
        logger.error("%s; passthrough", exc)
        return

    _record_transform_meta(
        flow,
        provider_type=provider_str,
        model=model,
        body=body,
        is_streaming=is_streaming,
        mode="transform",
    )

    from urllib.parse import urlparse

    parsed_url = urlparse(url)
    host = parsed_url.hostname or flow.request.host
    port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
    flow.request.host = host
    flow.request.port = port
    flow.request.scheme = parsed_url.scheme or "https"
    flow.request.path = parsed_url.path or "/"
    flow.server_conn = Server(address=(host, port))
    for k, v in headers.items():
        flow.request.headers[k] = v
    flow.request.content = new_body

    incoming_model = str(glom(body, "model", default="?"))
    flow.comment = f"{incoming_model} → {provider_str}/{model}"
    logger.info(
        "transform: %s → %s %s",
        incoming_model,
        provider_str,
        url.split("?")[0],
    )


def register_transform_routes(router: InspectorRouter) -> None:
    from ccproxy.inspector.router import RouteType

    @router.route("/{path}", rtype=RouteType.REQUEST, catch_error=False)  # ty: ignore[invalid-argument-type]
    def handle_transform(flow: HTTPFlow, **_kwargs: object) -> None:  # pyright: ignore[reportUnusedFunction]
        if metadata_from_flow(flow).direction != "inbound":
            return

        try:
            body = json.loads(flow.request.content or b"{}")
        except (json.JSONDecodeError, TypeError):
            body = {}

        target = _resolve_transform_target(flow, body)
        is_reverse = isinstance(flow.client_conn.proxy_mode, ReverseMode)

        if target is None:
            if is_reverse:
                # deferred: heavy mitmproxy Response import
                from mitmproxy.http import Response

                flow.response = Response.make(
                    501,
                    _openai_error(
                        "no provider or transform rule matched this request",
                        error_type="not_implemented_error",
                        code=501,
                    ),
                    {"Content-Type": "application/json"},
                )
            return

        action = target.action if isinstance(target, TransformOverride) else None

        if action == "passthrough":
            _handle_passthrough(flow)
        elif not is_reverse:
            # WireGuard flows already encode their destination.
            _handle_passthrough(flow)
        elif isinstance(target, Provider):
            incoming = _detect_incoming_format(flow.request.path)
            if incoming == target.type:
                _handle_redirect(flow, target, body)
            else:
                _handle_transform(flow, target, body)
        elif action == "redirect":
            _handle_redirect(flow, target, body)
        else:  # action == "transform"
            _handle_transform(flow, target, body)

        if is_reverse and flow.response is None and flow.request.host == "localhost" and flow.request.port == 1:
            from mitmproxy.http import Response

            flow.response = Response.make(
                502,
                _openai_error(
                    f"transform failed to rewrite destination (path={flow.request.path})",
                    error_type="api_error",
                    code=502,
                ),
                {"Content-Type": "application/json"},
            )
            logger.error(
                "Safety net: flow still targeting localhost:1 after transform (path=%s)",
                flow.request.path,
            )

    @router.route("/{path}", rtype=RouteType.RESPONSE, catch_error=False)  # ty: ignore[invalid-argument-type]
    def handle_transform_response(flow: HTTPFlow, **_kwargs: object) -> None:  # pyright: ignore[reportUnusedFunction]
        record = metadata_from_flow(flow).record
        if record is None or getattr(record, "transform", None) is None:
            return

        meta = record.transform
        if meta.mode != "transform":
            return
        if not flow.response or flow.response.status_code >= 400:
            return
        if meta.is_streaming:
            return

        try:
            # deferred: heavy FSM intake/render machinery
            from ccproxy.lightllm.graph.buffered import (
                transform_buffered_response_sync,
            )
            from ccproxy.lightllm.parsed import InboundFormat

            inbound_value = meta.inbound_format or "unknown"
            try:
                inbound_enum = InboundFormat(inbound_value)
            except ValueError:
                inbound_enum = InboundFormat.OPENAI_CHAT

            request_params = meta.request_parameters
            if request_params is None:
                from pydantic_ai.models import ModelRequestParameters

                request_params = ModelRequestParameters()

            new_body = transform_buffered_response_sync(
                raw_bytes=flow.response.content or b"",
                provider_type=meta.provider_type,
                inbound_format=inbound_enum,
                model=meta.model,
                request_params=request_params,
            )

            flow.response.content = new_body
            flow.response.headers["content-type"] = "application/json"
            flow.response.headers.pop("content-encoding", None)  # type: ignore[no-untyped-call]

            logger.info(
                "lightllm response transform: %s %s → %s",
                meta.provider_type,
                meta.model,
                inbound_enum.value,
            )
        except Exception:
            logger.warning("Response transform failed, passing through raw response", exc_info=True)
