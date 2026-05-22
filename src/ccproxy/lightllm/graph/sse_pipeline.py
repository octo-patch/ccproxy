"""Sync ``flow.response.stream`` callable backed by a persistent asyncio loop.

The graph-side replacement for
:class:`ccproxy.lightllm.response.pipeline.SSEPipeline` (sync). The intakes /
renderers under :mod:`ccproxy.lightllm.graph` are async (each chunk drives one
``await graph.run(...)``), but mitmproxy installs sync callables on
``flow.response.stream``. This pipeline owns one daemon thread + one
:class:`asyncio.AbstractEventLoop` per instance and submits each chunk via
:func:`asyncio.run_coroutine_threadsafe`, paying ~10–50 µs of cross-thread
hop per chunk against an upstream-network-bound 10–100 ms-per-chunk floor.

Compare to the pathological pattern Phase Q replaces: the
``_GoogleSyncIntake`` / ``_PerplexitySyncIntake`` adapters in
``response/intake.py`` spawn one fresh ``asyncio.new_event_loop()`` per
``feed`` call — ~200 chunks in a 5-second stream means 200 fresh loops, each
allocating its own selectors, signal handlers, and task graph.

Exception handling: failures inside ``intake.feed()`` or ``render.render()``
are caught and the offending chunk is passed through unmodified so mitmproxy
doesn't stall. Catastrophic failures in :meth:`close` still emit the render's
terminator so the client sees a well-formed end-of-stream.

Lifecycle: the daemon thread dies with the process, so a missed
:meth:`close` won't leak — but explicit cleanup on
:meth:`InspectorAddon.response` / the ``done`` mitmproxy event is preferred
so the loop tears down promptly when a flow finishes.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ccproxy.lightllm.graph import AnyAsyncIntakeFSM, AnyAsyncRenderFSM

logger = logging.getLogger(__name__)


class SSEPipeline:
    """Sync mitmproxy stream callable bridging upstream SSE → listener SSE.

    Drives an async intake FSM + render FSM pair via a persistent asyncio loop
    in a dedicated daemon thread. Behavioral contract matches the legacy sync
    :class:`ccproxy.lightllm.response.pipeline.SSEPipeline`:

    * ``__call__(bytes) -> bytes | list[bytes]`` returns the rendered chunk;
      ``[]`` when nothing was emitted (no-op chunk like an incomplete SSE
      frame), ``bytes`` otherwise.
    * Empty ``data`` (``b""``) is mitmproxy's end-of-stream sentinel — drains
      the intake's :meth:`close`, renders any trailing IR events, then emits
      the render's :meth:`close` terminator.
    * :attr:`upstream_raw_bytes` byte-for-byte tee of every chunk fed in.
    * :attr:`raw_body` alias of :attr:`upstream_raw_bytes` (old
      ``SSETransformer`` callsites — e.g. :class:`PerplexityAddon`).
    * :meth:`close` explicit cleanup. Idempotent.
    """

    def __init__(
        self,
        *,
        intake: AnyAsyncIntakeFSM,
        render: AnyAsyncRenderFSM,
    ) -> None:
        self._intake = intake
        self._render = render
        self._closed = False
        self._terminator_emitted = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="ccproxy-sse-loop",
        )
        self._thread.start()

    def __call__(self, data: bytes) -> bytes | list[bytes]:
        if data == b"":
            return self._flush_and_close()

        if self._closed:
            # The loop has been torn down; pass the chunk through so we don't
            # silently drop bytes.
            logger.debug("SSEPipeline: chunk received after close; passing through")
            return data

        try:
            future: Future[bytes] = asyncio.run_coroutine_threadsafe(
                self._process_chunk(data), self._loop
            )
            out = future.result()
        except Exception:
            logger.exception(
                "SSEPipeline.feed failed mid-stream; passing chunk through"
            )
            return data
        return out if out else []

    async def _process_chunk(self, data: bytes) -> bytes:
        """Drive one chunk through intake → render. Runs on the persistent loop."""
        out = bytearray()
        for event in await self._intake.feed(data):
            out.extend(await self._render.render(event))
        return bytes(out)

    def _flush_and_close(self) -> bytes | list[bytes]:
        """Drain trailing IR events, emit the render terminator, tear down the loop."""
        if self._closed:
            return []

        out = bytearray()

        if self._loop.is_running():
            try:
                future: Future[bytes] = asyncio.run_coroutine_threadsafe(
                    self._drain_and_terminate(), self._loop
                )
                out.extend(future.result())
            except Exception:
                logger.exception(
                    "SSEPipeline.close failed mid-drain; emitting render terminator only"
                )
                # Fall through: still try to emit the render terminator below.

        # Tear down the loop regardless. ``self._closed`` is the gate for
        # idempotency; once True, further ``__call__`` invocations no-op.
        self._closed = True
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=1.0)
        except Exception:
            logger.exception("SSEPipeline: failed to tear down persistent loop")

        return bytes(out) if out else []

    async def _drain_and_terminate(self) -> bytes:
        """Async tail: ``intake.close()`` → render each trailing event → ``render.close()``."""
        out = bytearray()
        try:
            for event in await self._intake.close():
                out.extend(await self._render.render(event))
        except Exception:
            logger.exception(
                "SSEPipeline intake.close failed; emitting render terminator only"
            )
        if not self._terminator_emitted:
            self._terminator_emitted = True
            try:
                out.extend(await self._render.close())
            except Exception:
                logger.exception(
                    "SSEPipeline render.close failed; no terminator emitted"
                )
        return bytes(out)

    def close(self) -> None:
        """Explicit cleanup. Idempotent. Tears down the persistent loop.

        Does NOT emit a terminator — that's the EOS path. Use this when a
        flow is being abandoned (client disconnect, mitmproxy ``done`` event)
        and the bytes are no longer being delivered.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=1.0)
        except Exception:
            logger.exception("SSEPipeline.close: failed to tear down persistent loop")

    @property
    def upstream_raw_bytes(self) -> bytes:
        """Byte-for-byte tee of every chunk fed in (for pplx_addon etc.)."""
        return bytes(self._intake.upstream_raw_bytes)

    @property
    def raw_body(self) -> bytes:
        """Alias of :attr:`upstream_raw_bytes` for old ``SSETransformer.raw_body`` callsites."""
        return self.upstream_raw_bytes
