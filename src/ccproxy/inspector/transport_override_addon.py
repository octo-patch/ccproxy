"""Rewrite ``flow.request`` to the in-process sidecar for impersonated outbound.

Selection is keyed on the ccproxy metadata facade. Engagement precedence,
given a resolved :class:`~ccproxy.config.Provider`:

1. ``Provider.fingerprint_profile`` set in config — always wins. Used for
   browser-name overrides (``chrome131``, ``firefox144``) or to force a
   different provider's shape.
2. Unset, but ``ShapeStore.pick_fingerprint(provider.type)`` returns a
   :class:`~ccproxy.inspector.fingerprint.CapturedFingerprint` — the
   fingerprint is an inherent property of the captured shape, so sidecar
   engages implicitly with ``provider.type`` as the impersonate key.
3. Neither — mitmproxy's native transport is used unchanged.

When engaged, the addon stashes the real target in ``X-CCProxy-Target-Url``
and the profile in ``X-CCProxy-Impersonate``, then rewrites destination to
``127.0.0.1:<sidecar>``. The loopback hop is marked ``Connection: close`` so
mitmproxy never reuses stale sidecar keep-alive sockets; the sidecar's
``httpx-curl-cffi`` client still owns upstream connection reuse.
"""

from __future__ import annotations

import logging

from mitmproxy import http

from ccproxy.config import get_config
from ccproxy.flows.store import HttpSnapshot
from ccproxy.pipeline.context import metadata_from_flow
from ccproxy.transport.sidecar import IMPERSONATE_HEADER, TARGET_URL_HEADER

logger = logging.getLogger(__name__)


class TransportOverrideAddon:
    """mitmproxy addon: redirect to the impersonating sidecar."""

    def __init__(self, sidecar_port: int) -> None:
        self._sidecar_port = sidecar_port

    async def request(self, flow: http.HTTPFlow) -> None:
        metadata = metadata_from_flow(flow)
        provider_name = metadata.auth_provider
        if not provider_name:
            return

        provider = get_config().providers.get(provider_name)
        if provider is None:
            return

        profile = provider.fingerprint_profile
        if profile is None:
            from ccproxy.shaping.store import get_store

            if get_store().pick_fingerprint(provider.type) is None:
                return
            profile = provider.type

        target_url = flow.request.pretty_url

        record = metadata.record
        if record is not None:
            record.forwarded_request = HttpSnapshot(
                headers=dict(flow.request.headers.items()),  # type: ignore[no-untyped-call]
                body=flow.request.content or b"",
                method=flow.request.method,
                url=target_url,
            )

        flow.request.headers[TARGET_URL_HEADER] = target_url
        flow.request.headers[IMPERSONATE_HEADER] = profile

        flow.request.host = "127.0.0.1"
        flow.request.port = self._sidecar_port
        flow.request.scheme = "http"
        flow.request.headers["host"] = f"127.0.0.1:{self._sidecar_port}"
        flow.request.headers["connection"] = "close"

        metadata.transport_override = True
        metadata.fingerprint_profile = profile

        logger.debug(
            "sidecar override: flow=%s provider=%s profile=%s target=%s",
            flow.id,
            provider_name,
            profile,
            target_url,
        )
