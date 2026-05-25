"""Response-side auth retry orchestration.

Detects 401 responses on flows where the request-side ``inject_auth`` hook
injected a provider auth token, resolves a fresh token, and transparently
replays the request. The credential source owns any underlying refresh logic;
this addon owns only the response-side detect/replay loop.
"""

from __future__ import annotations

import logging

from mitmproxy import http

from ccproxy import transport
from ccproxy.config import get_config
from ccproxy.inspector.fingerprint import CapturedFingerprint
from ccproxy.pipeline.context import metadata_from_flow

logger = logging.getLogger(__name__)


class AuthAddon:
    """mitmproxy addon: 401-detect → refresh → replay.

    Trigger contract: ``inject_auth`` stamps the ccproxy metadata facade.
    ``response()`` reads that state and replays the request when it sees a
    401 on a flow ccproxy injected.
    """

    async def response(self, flow: http.HTTPFlow) -> None:
        response = flow.response
        if not response or response.status_code != 401:
            return
        if not metadata_from_flow(flow).auth_injected:
            return

        try:
            await self._retry_with_refreshed_token(flow)
        except Exception:
            logger.error("Auth retry failed", exc_info=True)

    async def _retry_with_refreshed_token(self, flow: http.HTTPFlow) -> bool:
        metadata = metadata_from_flow(flow)
        provider = metadata.auth_provider
        if not provider:
            return False

        config = get_config()
        new_token = config.resolve_auth_token(provider)
        if not new_token:
            logger.warning("Auth 401 for provider '%s' — no token available, not retrying", provider)
            return False

        target_header = (config.get_auth_header(provider) or "authorization").lower()
        new_value = f"Bearer {new_token}" if target_header == "authorization" else new_token
        flow.request.headers[target_header] = new_value

        logger.info("Auth 401 for provider '%s' — token refreshed, retrying request", provider)

        headers = dict(flow.request.headers)
        headers.pop("x-ccproxy-auth-injected", None)

        profile = metadata.fingerprint_profile or transport.DEFAULT_PROFILE
        fingerprint = _resolve_captured_fingerprint(profile)
        if fingerprint is None:
            client = await transport.get_client(host=flow.request.pretty_host, profile=profile)
        else:
            client = await transport.get_client(
                host=flow.request.pretty_host,
                profile=profile,
                fingerprint=fingerprint,
            )
        retry_resp = await client.request(
            method=flow.request.method,
            url=flow.request.pretty_url,
            headers=headers,
            content=flow.request.content,
            timeout=config.provider_timeout,
        )
        metadata.retry_transport = "curl_cffi"
        metadata.retry_profile = profile

        assert flow.response is not None
        flow.response.status_code = retry_resp.status_code
        flow.response.headers.clear()
        for key, value in retry_resp.headers.multi_items():
            flow.response.headers.add(key, value)
        flow.response.content = retry_resp.content
        return True


def _resolve_captured_fingerprint(profile: str) -> CapturedFingerprint | None:
    if profile in transport.VALID_PROFILES:
        return None
    from ccproxy.shaping.store import get_store

    return get_store().pick_fingerprint(profile)
