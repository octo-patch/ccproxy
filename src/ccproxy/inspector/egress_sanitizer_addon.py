"""Final-stage mitmproxy addon that scrubs ccproxy-internal correlation headers.

ccproxy uses ``x-ccproxy-flow-id`` (and ``x-ccproxy-hooks``,
``x-ccproxy-oauth-injected``) as cross-addon correlation keys on
:class:`mitmproxy.http.HTTPFlow.request`. These are infrastructure-only
— they have no purpose beyond the inspector pipeline and would otherwise
leak ccproxy's presence on every request (``x-ccproxy-*`` is a trivial
fingerprint for any provider to flag).

Not all ``x-ccproxy-*`` headers belong in the drop list. The sidecar
transport contract (``x-ccproxy-target-url`` and ``x-ccproxy-impersonate``)
needs to survive the egress hop from mitmproxy to the loopback sidecar
— the sidecar reads them and strips them itself before reaching upstream.
A blind prefix strip would break sidecar dispatch. So the drop set is
explicit: only headers we generated for our own correlation needs go away.

This addon registers last in :func:`ccproxy.inspector.process._build_addons`
so every prior addon has had a chance to read the header before we drop it.
mitmproxy then forwards the cleaned request to whichever transport is
bound (native, sidecar, or replay).
"""

from __future__ import annotations

import logging

from mitmproxy import http

logger = logging.getLogger(__name__)

_DROP_HEADERS = frozenset(
    {
        "x-ccproxy-flow-id",
        "x-ccproxy-hooks",
        "x-ccproxy-oauth-injected",
    }
)
"""ccproxy-internal correlation headers that must never reach the next hop.

Notable exclusions: ``x-ccproxy-target-url`` and ``x-ccproxy-impersonate``
are intentionally kept — they're the sidecar transport contract,
consumed by the sidecar on the loopback hop and stripped there before
egress to the real upstream."""


class EgressSanitizerAddon:
    """mitmproxy addon: strip ccproxy-internal correlation headers from outbound."""

    def request(self, flow: http.HTTPFlow) -> None:
        to_drop = [name for name in flow.request.headers if name.lower() in _DROP_HEADERS]
        for name in to_drop:
            flow.request.headers.pop(name, None)
