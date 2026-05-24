"""Pipeline router — DAG-driven hook execution at the mitmproxy layer.

Builds PipelineExecutor instances from config and wires them as
mitmproxy addons. Two stages: inbound (pre-transform) and outbound
(post-transform), each with their own DAG.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

from ccproxy.flows.store import InspectorMeta
from ccproxy.lightllm import LightLLMError
from ccproxy.pipeline.executor import PipelineExecutor
from ccproxy.pipeline.loader import load_hooks

if TYPE_CHECKING:
    from mitmproxy.http import HTTPFlow

    from ccproxy.inspector.router import InspectorRouter

logger = logging.getLogger(__name__)


def _upstream_headers(response: httpx.Response) -> dict[str, str]:
    content_type = response.headers.get("content-type", "application/json")
    return {"Content-Type": content_type}


def _json_error_response(message: str, *, error_type: str, code: int) -> bytes:
    import json

    return json.dumps({"error": {"message": message, "type": error_type, "code": code}}).encode()


def build_executor(hook_entries: list[str | dict[str, Any]]) -> PipelineExecutor:
    specs = load_hooks(hook_entries)
    return PipelineExecutor(hooks=specs)


def register_pipeline_routes(
    router: InspectorRouter,
    executor: PipelineExecutor,
) -> None:
    from ccproxy.inspector.router import RouteType

    # Register both ``/`` and ``/{path}`` so flows targeting the root URL
    # match cleanly. ``parse.Parser("/{path}")`` does not match the bare
    # ``/`` (the ``{path}`` capture refuses empty segments), which would
    # otherwise leave root requests unhandled and trip xepor's
    # REQ_PASSTHROUGH behavior, blocking downstream synthetic routes.
    @router.route("/", rtype=RouteType.REQUEST)
    @router.route("/{path}", rtype=RouteType.REQUEST)
    def handle_pipeline(flow: HTTPFlow, **kwargs: object) -> None:  # pyright: ignore[reportUnusedFunction]
        if flow.metadata.get(InspectorMeta.DIRECTION) != "inbound":
            return

        try:
            executor.execute(flow)
        except httpx.HTTPStatusError as exc:
            from mitmproxy.http import Response

            upstream = exc.response
            flow.response = Response.make(upstream.status_code, upstream.content, _upstream_headers(upstream))
        except LightLLMError as exc:
            from mitmproxy.http import Response

            flow.response = Response.make(
                exc.status_code,
                _json_error_response(exc.message, error_type=exc.__class__.__name__, code=exc.status_code),
                {"Content-Type": "application/json"},
            )
