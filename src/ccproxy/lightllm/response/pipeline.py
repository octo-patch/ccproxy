"""Sync ``flow.response.stream`` callable bridging upstream wire → listener wire via IR.

``SSEPipeline`` is the sync class mitmproxy installs on
``flow.response.stream`` when the transform router decides a cross-format
response transform is needed. It wires:

  upstream bytes
    → ResponseIntake.feed         (vendor SSE → IR events)
    → ResponseRender.render       (IR events → listener wire bytes)
    → bytes returned to mitmproxy → client

A passthrough fast-path lives outside this pipeline: when the listener
format matches the upstream format, the inspector sets
``flow.response.stream = True`` and bytes flow through unchanged.

Exception handling: failures inside ``intake.feed()`` or ``render.render()``
are caught and the offending chunk is passed through unmodified so
mitmproxy doesn't stall. Catastrophic failures in ``close()`` still emit
the render's terminator so the client sees a well-formed end-of-stream.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ccproxy.lightllm.response.intake import ResponseIntake
    from ccproxy.lightllm.response.render import ResponseRender

logger = logging.getLogger(__name__)


class SSEPipeline:
    """Sync callable bridging upstream SSE → listener SSE via pydantic-ai IR."""

    def __init__(self, *, intake: ResponseIntake, render: ResponseRender) -> None:
        self._intake = intake
        self._render = render
        self._closed = False

    def __call__(self, data: bytes) -> bytes | list[bytes]:
        if data == b"":
            return self._flush_and_close()

        try:
            out = bytearray()
            for event in self._intake.feed(data):
                out.extend(self._render.render(event))
            return bytes(out) if out else []
        except Exception:
            logger.exception("SSEPipeline.feed failed mid-stream; passing chunk through")
            return data

    def _flush_and_close(self) -> bytes | list[bytes]:
        if self._closed:
            return []
        self._closed = True
        out = bytearray()
        try:
            for event in self._intake.close():
                out.extend(self._render.render(event))
        except Exception:
            logger.exception("SSEPipeline intake.close failed; emitting render terminator only")
        try:
            out.extend(self._render.close())
        except Exception:
            logger.exception("SSEPipeline render.close failed; no terminator emitted")
        return bytes(out) if out else []

    @property
    def upstream_raw_bytes(self) -> bytes:
        """Byte-for-byte tee of every chunk fed in (for pplx_addon etc.)."""
        return bytes(self._intake.upstream_raw_bytes)

    @property
    def raw_body(self) -> bytes:
        """Alias of ``upstream_raw_bytes`` for backward-compat with old ``SSETransformer.raw_body`` callsites."""
        return self.upstream_raw_bytes
