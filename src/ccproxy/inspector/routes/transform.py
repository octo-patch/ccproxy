"""Transform route — model bindings, sentinel Providers, and explicit overrides.

Routing precedence on every inbound request:

    1. ``lightllm.transforms`` — first regex-matched override wins.
    2. Compiled LiteLLM model bindings — first configured pattern wins.
    3. ccproxy metadata ``auth_provider`` — set by ``inject_auth`` when a
       sentinel key resolved. Looks up :class:`CCProxyConfig.providers`.
    4. None — :class:`mitmproxy.proxy.mode_specs.ReverseMode` flows return
       OpenAI-shape 501; WireGuard flows pass through unchanged.

Three actions:

    - ``transform``: rewrite the request body via lightllm dispatch (cross-format).
    - ``redirect``: rewrite destination and selected model without wire conversion.
    - ``passthrough``: forward unchanged.

For sentinel-resolved Provider targets, the action is auto-derived from the
incoming wire format and the Provider's adapter type.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote, urlsplit

from glom import glom
from mitmproxy.connection import Server
from mitmproxy.proxy.mode_specs import ReverseMode

from ccproxy.config import ModelBinding, Provider, TransformOverride, get_config
from ccproxy.flows.store import TransformMeta
from ccproxy.lightllm.graph import _ANTHROPIC_COMPATIBLE, _GOOGLE_COMPATIBLE
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

_GEMINI_FORMATS = _GOOGLE_COMPATIBLE


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


def _wire_formats_match(incoming: str | None, provider_type: str) -> bool:
    if incoming == "anthropic":
        return provider_type in _ANTHROPIC_COMPATIBLE
    if incoming == "gemini":
        return provider_type in _GOOGLE_COMPATIBLE
    return incoming == provider_type


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
        out = out.replace("{model}", quote(model, safe=""))
    if "{action}" in out:
        out = out.replace("{action}", action or "")
    return out


def _resolve_transform_target(
    flow: HTTPFlow,
    body: dict[str, object] | None = None,
) -> Provider | TransformOverride | ModelBinding | None:
    """Pick the routing target. First match wins; None means no signal."""
    config = get_config()
    request_model = _model_for_routing(body or {}, flow.request.path)

    for rule in config.lightllm.transforms:
        if rule.match_host_re and not _any_search(rule.match_host_re, _flow_hosts(flow)):
            continue
        if not rule.match_path_re.search(flow.request.path):
            continue
        if rule.match_model_re and not rule.match_model_re.search(request_model):
            continue
        return rule

    if request_model:
        # LiteLLM resolves a concrete deployment before wildcard fallbacks,
        # regardless of declaration order. Preserve order within each tier.
        for wildcard in (False, True):
            for binding in config.model_bindings:
                if ("*" in binding.model_name) == wildcard and binding.matches(request_model):
                    return binding

    auth_provider = metadata_from_flow(flow).auth_provider
    if auth_provider:
        return config.get_provider(auth_provider)

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


def _apply_destination(flow: HTTPFlow, url: str) -> None:
    parsed = urlsplit(url)
    host = parsed.hostname
    if host is None:
        raise ValueError(f"destination URL has no hostname: {url!r}")
    scheme = parsed.scheme or "https"
    port = parsed.port or (443 if scheme == "https" else 80)
    flow.request.host = host
    flow.request.port = port
    flow.request.scheme = scheme
    flow.request.path = parsed.path or "/"
    if parsed.query:
        flow.request.path = f"{flow.request.path}?{parsed.query}"
    flow.server_conn = Server(address=(host, port))


def _endpoint_url(base_url: str, path: str) -> str:
    if not path:
        return base_url.rstrip("/")
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _provider_for_target(
    flow: HTTPFlow,
    target: Provider | TransformOverride | ModelBinding,
) -> tuple[str | None, Provider | None]:
    config = get_config()
    if isinstance(target, ModelBinding):
        return target.provider_name, target.provider
    if isinstance(target, Provider):
        provider_name = metadata_from_flow(flow).auth_provider or target.type
        return provider_name, target
    if target.dest_provider is None:
        return None, None
    return target.dest_provider, config.get_provider(target.dest_provider)


def _apply_provider_request(
    flow: HTTPFlow,
    *,
    provider_name: str | None,
    provider: Provider | None,
) -> None:
    if provider_name is None or provider is None:
        return
    from ccproxy.hooks.inject_auth import inject_provider_auth
    from ccproxy.pipeline.context import Context

    inject_provider_auth(Context.from_flow(flow), provider_name, provider)


def _handle_passthrough(flow: HTTPFlow) -> None:
    logger.info(
        "transform passthrough: → %s:%d%s",
        flow.request.host,
        flow.request.port,
        flow.request.path,
    )


def _handle_redirect(
    flow: HTTPFlow,
    target: Provider | TransformOverride | ModelBinding,
    body: dict[str, object],
) -> None:
    """Same-format redirect: rewrite destination and selected model."""
    is_streaming = bool(glom(body, "stream", default=False))
    action = _action_from_path(flow.request.path)
    provider_name, bound = _provider_for_target(flow, target)

    base_url: str
    path: str
    if isinstance(target, ModelBinding):
        provider_str = target.provider.type
        model = target.resolve_model(_model_for_routing(body, flow.request.path))
        base_url = target.provider.base_url
        path = _apply_path_template(target.provider.path, model=model, action=action)
    elif isinstance(target, Provider):
        provider_str = target.type
        model = _model_for_routing(body, flow.request.path)
        base_url = target.base_url
        path = _apply_path_template(target.path, model=model, action=action)
    else:
        resolved_base_url = target.dest_base_url or (bound.base_url if bound else None)
        if resolved_base_url is None:
            logger.error(
                "redirect override missing dest_base_url and no resolvable dest_provider; passthrough",
            )
            return
        base_url = resolved_base_url
        provider_str = (bound.type if bound else target.dest_provider) or ""
        model = target.dest_model or _model_for_routing(body, flow.request.path)
        if target.dest_path:
            path = _apply_path_template(target.dest_path, model=model, action=action)
        elif bound is not None:
            path = _apply_path_template(bound.path, model=model, action=action)
        else:
            path = flow.request.path

    if body.get("model") != model and model:
        body = {**body, "model": model}
        flow.request.content = json.dumps(body).encode()

    _record_transform_meta(
        flow,
        provider_type=provider_str,
        model=model,
        body=body,
        is_streaming=is_streaming,
        mode="redirect",
    )

    url = _endpoint_url(base_url, path)
    _apply_destination(flow, url)
    _apply_provider_request(flow, provider_name=provider_name, provider=bound)

    flow.comment = f"redirect → {provider_str}/{urlsplit(url).netloc}"
    logger.info("redirect: → %s %s", provider_str, url.split("?")[0])


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
    target: Provider | TransformOverride | ModelBinding,
    bound: Provider | None,
    model: str,
    provider_type: str,
    is_streaming: bool,
) -> tuple[str, dict[str, str]]:
    """Build the upstream ``(url, headers)`` for a transform-mode dispatch.

    Pulls base URL/path from the resolved target (``Provider`` or
    ``TransformOverride`` with optional ``dest_base_url`` / ``dest_path`` overrides
    falling back to the bound Provider). Credential placement happens after
    destination selection; this builder only adds the Anthropic-compat
    ``anthropic-version`` floor.
    """
    action = _action_for_transform(provider_type, is_streaming=is_streaming)

    base_url: str
    path_template: str
    if isinstance(target, ModelBinding):
        base_url = target.provider.base_url
        path_template = target.provider.path
    elif isinstance(target, Provider):
        base_url = target.base_url
        path_template = target.path
    else:
        resolved_base_url = target.dest_base_url or (bound.base_url if bound is not None else None)
        if resolved_base_url is None:
            raise ValueError(
                "transform override missing dest_base_url and no resolvable dest_provider",
            )
        base_url = resolved_base_url
        path_template = target.dest_path or (bound.path if bound is not None else "/")

    path = _apply_path_template(path_template, model=model, action=action)
    url = _endpoint_url(base_url, path)

    headers: dict[str, str] = {}
    if provider_type in _ANTHROPIC_COMPATIBLE:
        # Defensive floor for cross-format flows targeting an Anthropic upstream
        # where no Anthropic shape replay runs. The shape hook adds the
        # canonical Claude headers when present.
        headers["anthropic-version"] = "2023-06-01"
    return url, headers


def _handle_transform(
    flow: HTTPFlow,
    target: Provider | TransformOverride | ModelBinding,
    body: dict[str, object],
) -> None:
    """Cross-format transform: render the body via ``dispatch_dump_sync`` and
    rewrite the destination.

    All providers (Anthropic-compatible, OpenAI, Gemini-family, Perplexity Pro)
    route through pydantic-ai's IR via :class:`~ccproxy.pipeline.context.Context.parse_sync`
    + :func:`dispatch_dump_sync`. URL + headers come from the resolved
    :class:`Provider` config (base URL/path with ``{model}`` / ``{action}`` templating)
    or the :class:`TransformOverride` overrides.
    """
    # deferred: avoid pulling pydantic-ai at module import time
    from ccproxy.lightllm.graph import dispatch_dump_sync
    from ccproxy.pipeline.context import Context

    is_streaming = bool(glom(body, "stream", default=False))
    config = get_config()
    provider_name, resolved_provider = _provider_for_target(flow, target)

    bound: Provider | None
    if isinstance(target, ModelBinding):
        provider_str = target.provider.type
        model = target.resolve_model(_model_for_routing(body, flow.request.path))
        bound = target.provider
    elif isinstance(target, Provider):
        provider_str = target.type
        model = _model_for_routing(body, flow.request.path)
        bound = target
    else:
        if target.dest_provider is None:
            logger.error("transform override missing dest_provider; passthrough")
            return
        bound = config.get_provider(target.dest_provider)
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

    _apply_destination(flow, url)
    for k, v in headers.items():
        flow.request.headers[k] = v
    flow.request.content = new_body
    _apply_provider_request(
        flow,
        provider_name=provider_name,
        provider=resolved_provider,
    )

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

    @router.route("/{path}", rtype=RouteType.REQUEST, catch_error=False)
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

        if isinstance(target, ModelBinding) and is_reverse:
            body = target.apply_defaults(body)
            flow.request.content = json.dumps(body).encode()

        action = target.action if isinstance(target, TransformOverride) else None

        if action == "passthrough":
            _handle_passthrough(flow)
        elif not is_reverse:
            # WireGuard flows already encode their destination.
            _handle_passthrough(flow)
        elif isinstance(target, Provider | ModelBinding):
            incoming = _detect_incoming_format(flow.request.path)
            provider_type = target.provider.type if isinstance(target, ModelBinding) else target.type
            if _wire_formats_match(incoming, provider_type):
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

    @router.route("/{path}", rtype=RouteType.RESPONSE, catch_error=False)
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
        if meta.provider_type == "openai_conversations":
            # OpenAI Conversations is always force-streamed through SSEPipeline
            # (collect mode for non-streaming clients), which has already produced
            # the buffered JSON in flow.response.content. Re-running the buffered
            # transform here would double-transform it.
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
        except Exception as exc:
            # Passing the untransformed provider body through would hand the
            # client the wrong wire format; fail loudly in the format it expects.
            logger.warning("Response transform failed, returning 500 to client", exc_info=True)
            flow.response.status_code = 500
            flow.response.content = _openai_error(
                f"ccproxy response transform failed: {exc}",
                error_type="api_error",
                code=500,
            )
            flow.response.headers["content-type"] = "application/json"
            flow.response.headers.pop("content-encoding", None)  # type: ignore[no-untyped-call]
