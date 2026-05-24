"""Capture native TLS ClientHello fingerprints and attach them to HTTP flows."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

from mitmproxy import http, tls

from ccproxy.inspector.fingerprint import CLIENT_FINGERPRINT_METADATA, parse_client_hello_bytes

logger = logging.getLogger(__name__)

_MAX_CLIENT_HELLOS = 2048


class FingerprintCaptureAddon:
    """mitmproxy addon that bridges TLS ClientHello data to later HTTP flows."""

    def __init__(self, *, max_entries: int = _MAX_CLIENT_HELLOS) -> None:
        self._max_entries = max_entries
        self._by_client_id: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def tls_clienthello(self, data: tls.ClientHelloData) -> None:
        try:
            fingerprint = parse_client_hello_bytes(data.client_hello.raw_bytes(wrap_in_record=False))
        except Exception as exc:
            logger.debug("failed to parse ClientHello fingerprint: %s", exc)
            return

        client_id = data.context.client.id
        self._by_client_id[client_id] = fingerprint.to_dict()
        self._by_client_id.move_to_end(client_id)
        while len(self._by_client_id) > self._max_entries:
            self._by_client_id.popitem(last=False)

    def request(self, flow: http.HTTPFlow) -> None:
        fingerprint = self._by_client_id.get(flow.client_conn.id)
        if fingerprint is None:
            return
        flow.metadata[CLIENT_FINGERPRINT_METADATA] = fingerprint
