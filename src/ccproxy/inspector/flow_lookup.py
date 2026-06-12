"""Mitmproxy flow lookup helpers shared by inspector addons."""

from __future__ import annotations

from mitmproxy import ctx, http


def find_http_flow(flow_id: str) -> http.HTTPFlow | None:
    view = ctx.master.addons.get("view")  # type: ignore[no-untyped-call]
    if view is None:
        return None
    found = view.get_by_id(flow_id)
    return found if isinstance(found, http.HTTPFlow) else None
