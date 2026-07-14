"""Rewrite ``flow.request`` to the in-process sidecar for impersonated outbound.

Sidecar engagement is keyed exclusively on ``Provider.fingerprint_profile``
being set in config. When set, the named profile (a curl-cffi browser name
such as ``"chrome131"`` or a captured-fingerprint handle) drives TLS+HTTP/2
impersonation through the in-process curl-cffi transport.

When ``fingerprint_profile`` is unset, mitmproxy's native transport is used
unchanged — regardless of whether a shape file happens to carry a captured
fingerprint in its metadata. Implicit sidecar engagement from shape-embedded
fingerprints is intentionally absent: captured fingerprints must be opted
into explicitly via Provider config to avoid cipher-compatibility failures
on providers that don't require browser-level TLS impersonation.

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
from ccproxy.transport.sidecar import CONTINUATION_HEADER, IMPERSONATE_HEADER, TARGET_URL_HEADER

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

        provider = get_config().get_provider(provider_name)
        if provider is None:
            return

        profile = provider.fingerprint_profile
        if profile is None:
            return

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
        if provider.type == "openai_conversations":
            flow.request.headers[CONTINUATION_HEADER] = "openai_conversations"

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
