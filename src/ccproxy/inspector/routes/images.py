"""Route claims for OpenAI image generation / edit endpoints.

``POST /v1/images/generations`` and ``POST /v1/images/edits`` are not a known
chat wire format, so the transform router's ``/{path}`` catch-all would try to
parse them as chat requests and corrupt the body. These literal routes claim
the paths (registered before ``register_transform_routes`` so they win on exact
match) and simply stamp ``metadata.oaic_image_operation``.

All the real work — parsing, the 3-step image upload, rendering the
``/backend-api/f/conversation`` body, async polling, and the two-step download —
runs in :class:`~ccproxy.inspector.openai_conversations_addon.OpenAIConversationsAddon`,
which is async and owns the shared browser-fingerprinted transport. The route
handlers are synchronous (xepor), so they cannot perform those side trips here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ccproxy.pipeline.context import metadata_from_flow

if TYPE_CHECKING:
    from mitmproxy.http import HTTPFlow

    from ccproxy.inspector.router import InspectorRouter

logger = logging.getLogger(__name__)


def _claim_image_flow(flow: HTTPFlow, operation: str) -> None:
    metadata = metadata_from_flow(flow)
    if metadata.direction != "inbound":
        return
    metadata.oaic_image_operation = operation
    logger.debug("oaic image route claimed: op=%s path=%s", operation, flow.request.path)


def register_image_routes(router: InspectorRouter) -> None:
    """Register the ``/v1/images/{generations,edits}`` claim routes."""
    from ccproxy.inspector.router import RouteType

    @router.route("/v1/images/generations", rtype=RouteType.REQUEST, catch_error=False)
    def handle_image_generation(flow: HTTPFlow, **_kwargs: object) -> None:  # pyright: ignore[reportUnusedFunction]
        _claim_image_flow(flow, "generation")

    @router.route("/v1/images/edits", rtype=RouteType.REQUEST, catch_error=False)
    def handle_image_edit(flow: HTTPFlow, **_kwargs: object) -> None:  # pyright: ignore[reportUnusedFunction]
        _claim_image_flow(flow, "edit")
