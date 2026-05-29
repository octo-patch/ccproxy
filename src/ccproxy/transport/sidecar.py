"""In-process HTTP sidecar that forwards requests via curl-cffi impersonation.

mitmproxy reverse-proxies through this sidecar when a flow needs TLS+HTTP/2
fingerprint impersonation. The two-header contract on the incoming request:

- ``X-CCProxy-Target-Url`` — real upstream URL (scheme + host + path).
- ``X-CCProxy-Impersonate`` — ``curl-cffi`` impersonate profile name.

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
import logging
import socket
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import uvicorn
from httpx import Headers
from httpx._decoders import SUPPORTED_DECODERS, ContentDecoder, DecodingError, MultiDecoder
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from ccproxy import transport
from ccproxy.inspector.fingerprint import CapturedFingerprint

logger = logging.getLogger(__name__)

TARGET_URL_HEADER = "x-ccproxy-target-url"
IMPERSONATE_HEADER = "x-ccproxy-impersonate"

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
"""Response headers that no longer describe the sidecar-relayed body."""


def _content_decodings(headers: Headers) -> list[str]:
    return [
        encoding.strip().lower()
        for value in headers.get_list("content-encoding")
        for encoding in value.split(",")
        if encoding.strip()
    ]


def _response_decoder(headers: Headers) -> ContentDecoder | None:
    decodings = [encoding for encoding in _content_decodings(headers) if encoding != "identity"]
    if not decodings:
        return None

    try:
        decoders = [SUPPORTED_DECODERS[encoding]() for encoding in decodings]
    except (KeyError, ImportError) as exc:
        logger.warning("sidecar: unsupported Content-Encoding %s: %s", ", ".join(decodings), exc)
        return None

    return MultiDecoder(decoders)


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

    drop = _RELAY_EXCLUDED_HEADERS | {TARGET_URL_HEADER, IMPERSONATE_HEADER}
    fwd_headers = _filter_headers(list(request.headers.raw), drop)
    body = await request.body()

    try:
        fingerprint = _resolve_captured_fingerprint(profile)
        client = await transport.get_client(host=host, profile=profile, fingerprint=fingerprint)
    except transport.UnknownFingerprintProfileError as e:
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
        logger.warning("sidecar: transport error for %s: %s", target_url, e)
        return Response(f"transport error: {e}", status_code=502)

    decoder = _response_decoder(upstream.headers)
    response_header_drop = _RELAY_RESPONSE_EXCLUDED_HEADERS if decoder is not None else _RELAY_EXCLUDED_HEADERS

    async def body_stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                if decoder is None:
                    yield chunk
                    continue
                try:
                    decoded = decoder.decode(chunk)
                except DecodingError as exc:
                    logger.warning("sidecar: failed to decode Content-Encoding for %s: %s", target_url, exc)
                    raise
                if decoded:
                    yield decoded
            if decoder is not None:
                flushed = decoder.flush()
                if flushed:
                    yield flushed
        finally:
            await upstream.aclose()

    return StreamingResponse(
        body_stream(),
        status_code=upstream.status_code,
        headers=dict(
            _filter_response_headers(
                list(upstream.headers.raw),
                drop=response_header_drop,
            )
        ),
    )


def _resolve_captured_fingerprint(profile: str) -> CapturedFingerprint | None:
    if profile in transport.VALID_PROFILES:
        return None
    from ccproxy.shaping.store import get_store

    return get_store().pick_fingerprint(profile)


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
