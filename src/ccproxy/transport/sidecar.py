"""In-process HTTP sidecar that forwards requests via curl-cffi impersonation.

mitmproxy reverse-proxies through this sidecar so provider egress has an
explicit TLS+HTTP/2 fingerprint policy. The request contract is:

- ``X-CCProxy-Target-Url`` — real upstream URL (scheme + host + path).
- ``X-CCProxy-Impersonate`` — ``curl-cffi`` impersonate profile name.
- ``X-CCProxy-Fingerprint`` — optional base64url JSON captured ClientHello
  profile for this flow.
- ``X-CCProxy-Continuation`` — optional continuation type name. When present,
  ``body_stream()`` tees chunks through a provider-specific continuation
  handler that may append further content after the upstream HTTP body ends
  (e.g. WebSocket handoff for ``openai_conversations``).

The sidecar strips those, forwards everything else through the cached
``httpx.AsyncClient`` from :mod:`ccproxy.transport.dispatch`, decodes any
upstream Content-Encoding, and streams the response body back chunk-by-chunk.
mitmproxy's existing streaming pipeline handles relaying chunks to the client.

Lifecycle: :class:`Sidecar` binds 127.0.0.1 on an OS-picked port at
:meth:`Sidecar.start`. :attr:`Sidecar.port` exposes the bound port for the
``TransportOverrideAddon`` to rewrite ``flow.request`` against.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import socket
from collections.abc import AsyncIterator
from typing import Protocol
from urllib.parse import urlsplit

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from ccproxy import transport
from ccproxy.inspector.fingerprint import CapturedFingerprint
from ccproxy.openai_conversations.session_ws import get_session_ws_manager
from ccproxy.openai_conversations.ws_handoff import (
    HandoffState,
    detect_handoff,
)

logger = logging.getLogger(__name__)

TARGET_URL_HEADER = "x-ccproxy-target-url"
IMPERSONATE_HEADER = "x-ccproxy-impersonate"
FINGERPRINT_HEADER = "x-ccproxy-fingerprint"
CONTINUATION_HEADER = "x-ccproxy-continuation"

_RELAY_EXCLUDED_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)
"""Headers the sidecar must not relay verbatim.

Includes RFC 7230 hop-by-hop headers plus ``host`` and ``content-length``,
which the outbound client recomputes from the rewritten target and body.
"""

_RELAY_RESPONSE_EXCLUDED_HEADERS = _RELAY_EXCLUDED_HEADERS | {"content-encoding"}
"""Response headers that no longer describe the sidecar-relayed body.

libcurl decodes ``Content-Encoding`` in the transport, so the relayed body is
always plaintext; the now-stale ``content-encoding`` header is dropped so
downstream clients don't try to decode it a second time.
"""


def _filter_headers(headers: list[tuple[bytes, bytes]], drop: frozenset[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in headers:
        name = k.decode("latin-1").lower()
        if name in drop:
            continue
        out[k.decode("latin-1")] = v.decode("latin-1")
    return out


def _filter_response_headers(
    headers: list[tuple[bytes, bytes]],
    *,
    drop: frozenset[str] = _RELAY_EXCLUDED_HEADERS,
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for k, v in headers:
        name = k.decode("latin-1").lower()
        if name in drop:
            continue
        out.append((k.decode("latin-1"), v.decode("latin-1")))
    return out


async def _handle(request: Request) -> Response:
    """Forward one request through the impersonating transport."""
    target_url = request.headers.get(TARGET_URL_HEADER)
    profile = request.headers.get(IMPERSONATE_HEADER)
    if not target_url or not profile:
        return Response(
            f"missing {TARGET_URL_HEADER} or {IMPERSONATE_HEADER}",
            status_code=400,
        )

    parsed = urlsplit(target_url)
    host = parsed.hostname
    if host is None:
        return Response(f"invalid target URL: {target_url!r}", status_code=400)

    continuation = request.headers.get(CONTINUATION_HEADER)
    drop = _RELAY_EXCLUDED_HEADERS | {TARGET_URL_HEADER, IMPERSONATE_HEADER, FINGERPRINT_HEADER, CONTINUATION_HEADER}
    fwd_headers = _filter_headers(list(request.headers.raw), drop)
    body = await request.body()

    try:
        fingerprint = _fingerprint_from_header(request.headers.get(FINGERPRINT_HEADER))
        if fingerprint is None:
            fingerprint = transport.resolve_captured_fingerprint(profile)
        client = await transport.get_client(host=host, profile=profile, fingerprint=fingerprint)
    except transport.UnknownFingerprintProfileError as e:
        return Response(str(e), status_code=400)
    except ValueError as e:
        return Response(str(e), status_code=400)

    try:
        upstream = await client.send(
            client.build_request(
                method=request.method,
                url=target_url,
                headers=fwd_headers,
                content=body,
            ),
            stream=True,
        )
    except Exception as e:
        # httpx-curl-cffi translates curl errors via assert exc.code, which
        # raises AssertionError("Curl error code undefined") when the underlying
        # ImpersonateError carries code=0. Unwrap the cause chain to recover
        # the real message (e.g. "Cipher 0xa3 is not found").
        real: BaseException = e
        if isinstance(e, AssertionError) and isinstance(e.__context__, Exception):
            real = e.__context__
        logger.warning("sidecar: transport error for %s: %s", target_url, real)
        return Response(f"transport error: {real}", status_code=502)

    # Resolve a continuation factory when the request opted in.
    continuation_factory = _CONTINUATION_FACTORIES.get(continuation) if continuation else None

    async def body_stream() -> AsyncIterator[bytes]:
        # libcurl already decoded any Content-Encoding, so chunks are plaintext.
        handoff_state: HandoffState | None = HandoffState() if continuation_factory is not None else None
        try:
            async for chunk in upstream.aiter_raw():
                if chunk:
                    if handoff_state is not None:
                        detect_handoff(handoff_state, chunk)
                    yield chunk
        finally:
            await upstream.aclose()

        # Continuation: if the HTTP body handed off without inline content, run
        # the WS bridge (the answer streams over the WebSocket) and stream its
        # bytes onto the same response.
        if continuation_factory is not None and handoff_state is not None and handoff_state.should_bridge():
            async for extra_chunk in continuation_factory(
                client=client,
                handoff_state=handoff_state,
                request_headers=fwd_headers,
            ):
                yield extra_chunk

    return StreamingResponse(
        body_stream(),
        status_code=upstream.status_code,
        headers=dict(
            _filter_response_headers(
                list(upstream.headers.raw),
                drop=_RELAY_RESPONSE_EXCLUDED_HEADERS,
            )
        ),
    )


async def _openai_conversations_continuation(
    *,
    client: httpx.AsyncClient,
    handoff_state: HandoffState,
    request_headers: dict[str, str],
) -> AsyncIterator[bytes]:
    """Continuation dispatcher for ``openai_conversations``.

    Routes the SPA's WebSocket handoff: a ``stream_handoff`` topic or a
    ``resume_conversation_token`` JWT's ``turn_topic_id`` through the persistent
    :class:`~ccproxy.openai_conversations.session_ws.SessionWSManager`, which
    keeps one ``/celsius/ws/user`` wss connection open per session and
    multiplexes each turn's frames onto it (falling back to the per-turn
    ``run_handoff_bridge`` on any failure of the persistent path).
    """
    if handoff_state.should_bridge():
        async for chunk in get_session_ws_manager().stream_turn(
            topic_id=handoff_state.topic, client=client, request_headers=request_headers
        ):
            yield chunk


class _ContinuationFactory(Protocol):
    """Callable protocol for continuation factories."""

    def __call__(
        self,
        *,
        client: httpx.AsyncClient,
        handoff_state: HandoffState,
        request_headers: dict[str, str],
    ) -> AsyncIterator[bytes]: ...


# Mapping from ``X-CCProxy-Continuation`` value to its async-generator factory.
# New providers add an entry here; the sidecar remains provider-agnostic.
_CONTINUATION_FACTORIES: dict[str, _ContinuationFactory] = {
    "openai_conversations": _openai_conversations_continuation,
}


def _fingerprint_from_header(value: str | None) -> CapturedFingerprint | None:
    if not value:
        return None
    try:
        padding = "=" * (-len(value) % 4)
        raw = base64.urlsafe_b64decode((value + padding).encode()).decode()
        payload = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"invalid {FINGERPRINT_HEADER}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {FINGERPRINT_HEADER}")
    return CapturedFingerprint.from_dict(payload)


def _build_app() -> Starlette:
    methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
    return Starlette(routes=[Route("/{path:path}", _handle, methods=methods)])


class Sidecar:
    """In-process HTTP sidecar lifecycle.

    Run :meth:`start` once during inspector boot; :attr:`port` is then the
    bound TCP port to rewrite ``flow.request`` destinations against. Call
    :meth:`stop` during shutdown — it ends the server cleanly and joins the
    background task.
    """

    def __init__(self) -> None:
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self._port: int | None = None
        self._sock: socket.socket | None = None

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("sidecar not started")
        return self._port

    async def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        self._sock = sock
        self._port = sock.getsockname()[1]

        # log_config=None: uvicorn's default LOGGING_CONFIG runs through
        # logging.config.dictConfig() which silently calls
        # _clearExistingHandlers() — closing every root-logger handler stream,
        # including the FileHandler ccproxy installed for ccproxy.log.
        # Setting log_config=None skips uvicorn's logging setup entirely;
        # ccproxy's setup_logging is the single source of truth.
        config = uvicorn.Config(
            app=_build_app(),
            log_level="warning",
            lifespan="off",
            ws="none",
            access_log=False,
            log_config=None,
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(
            self._server.serve(sockets=[sock]),
            name="ccproxy-sidecar",
        )

        deadline = asyncio.get_running_loop().time() + 5.0
        while not self._server.started:
            if asyncio.get_running_loop().time() > deadline:
                raise RuntimeError("sidecar failed to bind within 5s")
            if self._task.done():
                exc = self._task.exception()
                raise RuntimeError(f"sidecar serve() exited prematurely: {exc!r}") from exc
            await asyncio.sleep(0.01)

        logger.info("sidecar listening on 127.0.0.1:%d", self._port)

    async def stop(self) -> None:
        # Tear down every persistent session WebSocket before the loop stops, so
        # no read-loop task is left dangling across a sidecar restart.
        try:
            await get_session_ws_manager().shutdown()
        except Exception as exc:
            logger.warning("sidecar: session WS shutdown error: %s", exc)
        if self._server is None or self._task is None:
            return
        self._server.should_exit = True
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except TimeoutError:
            logger.warning("sidecar: shutdown timeout, cancelling")
            self._task.cancel()
        finally:
            self._server = None
            self._task = None
            self._sock = None
            self._port = None
