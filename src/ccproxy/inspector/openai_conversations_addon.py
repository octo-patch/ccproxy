"""OpenAI Conversations transport-preparation addon.

Responsibility: for flows whose ``metadata.auth_provider`` resolves to a
:class:`~ccproxy.config.Provider` with ``type == "openai_conversations"``,
this addon performs the browser-side pre-flight work that must share the same
cached curl-cffi client as the final forwarded request:

1. **Cookie-jar warmup** — ``GET /``, ``/api/auth/session``, ``/cdn-cgi/trace``
   through the cached client. Throttled to at most once per
   ``warmup_throttle_seconds`` when usable Cloudflare cookies already exist.
2. **Sentinel refresh** — ``POST /backend-api/sentinel/chat-requirements/prepare``
   → solve PoW locally → ``POST .../finalize`` when the persisted
   chat-requirements token is missing or within ``sentinel_skew_seconds`` of
   expiry. Persists the new chat_req_token + proof_token + expiry back to the
   credential file via :func:`update_sentinel_fields`.
3. **Conduit prewarm** — ``POST /backend-api/f/conversation/prepare`` as the
   browser does (HAR): first ``state=none`` with ``x-conduit-token: no-token``
   (before sentinel), then ``state=success`` looping — chaining each returned
   ``conduit_token`` until the server returns an empty one. Image turns use a
   single ``success`` prepare after rendering. Sentinel headers are never sent on
   prepare; fail-closed on any prepare error.
4. **Header stamping** — browser identity, the two Sentinel headers
   (chat-requirements + proof), turn trace, target-path, ``X-OAI-IS`` from the
   cookie jar (best-effort). The conduit token is NOT sent on the final
   ``/f/conversation`` (HAR). Clears stale downstream sec-*/oai-* headers first;
   keeps ``Authorization: Bearer <access_token>`` from ``inject_auth``.

``response()`` handles:
- **One-shot retry** on 401, 403, or Cloudflare challenge: invalidate cached
  Sentinel state, force warmup, refresh Sentinel, replay once via the same
  ``get_client(...)`` call, then mark the flow so ``AuthAddon`` skips its generic
  replay and this addon cannot loop.
- **ConversationStore write-back**: scan the upstream SSE body for
  ``conversation_id`` and the last assistant message id, then persist them for
  the next turn's ``openai_conversations_thread_inject`` hook.
- **X-OAI-IS-Update**: if the response carries this header, write its value back
  to the client cookie jar (best-effort).
- **WS handoff: DEFERRED** — The full WebSocket continuation
  (``GET /backend-api/celsius/ws/user`` → ``wss`` → subscribe → frame reader)
  is a joint capstone with CHATGPT-004 (intake FSM surfaces the handoff signal).
  A clearly commented stub is left here; wiring is done after CHATGPT-004 lands.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from http.cookiejar import LoadError, MozillaCookieJar
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from mitmproxy import http
from mitmproxy.connection import Server

if TYPE_CHECKING:
    from ccproxy.lightllm.adapters.openai_conversations import _PrepareState

from ccproxy import transport
from ccproxy.config import OpenAIConversationsConfig, Provider, get_config
from ccproxy.lightllm.openai import conversations_image_parse as image_parse
from ccproxy.openai_conversations.conversation_store import get_conversation_store
from ccproxy.openai_conversations.credentials import (
    load_credential_state,
    update_sentinel_fields,
)
from ccproxy.openai_conversations.pow import PowExhaustedError, solve_pow
from ccproxy.openai_conversations.prepare_p import build_requirements_token
from ccproxy.openai_conversations.profile import get_browser_headers
from ccproxy.openai_conversations.sentinel import (
    SentinelResult,
    build_finalize_body,
    build_prepare_body,
    decode_jwt_exp_ms,
    is_expired,
)
from ccproxy.pipeline.context import metadata_from_flow

logger = logging.getLogger(__name__)

# Per (provider_name, fingerprint_profile) → last warmup timestamp (monotonic).
_warmup_timestamps: dict[tuple[str, str], float] = {}

_BASE_URL = "https://chatgpt.com"
_CONVERSATION_PATH = "/backend-api/f/conversation"
_PREPARE_PATH = "/backend-api/f/conversation/prepare"
_SENTINEL_PREPARE_PATH = "/backend-api/sentinel/chat-requirements/prepare"
_SENTINEL_FINALIZE_PATH = "/backend-api/sentinel/chat-requirements/finalize"


def _bearer_from_flow(flow: http.HTTPFlow) -> str:
    """Return the bare bearer token stamped by ``inject_auth`` on the flow."""
    auth = str(flow.request.headers.get("authorization", "") or "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return auth.strip()


def _cookie_value(client: httpx.AsyncClient, name: str) -> str:
    """Return a chatgpt.com cookie value from the client jar (best-effort)."""
    try:
        return client.cookies.get(name, domain="chatgpt.com") or ""
    except Exception:
        return ""


def _load_cookies_from_file(client: httpx.AsyncClient, cookie_file: str) -> int:
    """Load Netscape-format cookies (gateau export) into the client jar.

    Supplies the Cloudflare ``cf_clearance`` and chatgpt.com session cookies
    that warmup cannot synthesize (Cloudflare-gated endpoints 403 without them).
    Returns the number of cookies loaded; a missing or malformed file is
    non-fatal (returns 0). Idempotent — re-setting the same cookies each request
    keeps the jar current when the file is refreshed.
    """
    path = Path(cookie_file).expanduser()
    if not path.is_file():
        return 0
    jar = MozillaCookieJar(str(path))
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except (OSError, LoadError) as exc:
        logger.warning("oaic cookie load failed for %s: %s", path, exc)
        return 0
    count = 0
    for cookie in jar:
        client.cookies.set(cookie.name, cookie.value or "", domain=cookie.domain, path=cookie.path or "/")
        count += 1
    return count


def _is_oaic_flow(flow: http.HTTPFlow) -> bool:
    """True when this flow belongs to the openai_conversations provider."""
    provider_name = metadata_from_flow(flow).auth_provider
    if not provider_name:
        return False
    provider = get_config().providers.get(provider_name)
    return provider is not None and provider.type == "openai_conversations"


def _has_usable_cookies(client: httpx.AsyncClient) -> bool:
    """Return True when the client cookie jar has CF-clearance or session tokens."""
    try:
        jar = client.cookies
        names = {c.name for c in jar.jar}
        return bool(names & {"cf_clearance", "__Secure-next-auth.session-token.0"})
    except Exception:
        return False


def _should_warmup(
    *,
    provider_name: str,
    profile: str,
    throttle_seconds: float,
    client: httpx.AsyncClient,
) -> bool:
    """Decide whether cookie-jar warmup should run.

    Skips warmup when all of:
    - The client already has usable cookies (cf_clearance or session-token), AND
    - The last warmup for (provider_name, profile) ran within ``throttle_seconds``.
    """
    key = (provider_name, profile)
    last = _warmup_timestamps.get(key, 0.0)
    elapsed = time.monotonic() - last
    return not (_has_usable_cookies(client) and elapsed < throttle_seconds)


def _mark_warmup(provider_name: str, profile: str) -> None:
    _warmup_timestamps[(provider_name, profile)] = time.monotonic()


async def _run_warmup(client: httpx.AsyncClient, timeout: float) -> None:
    """GET /, /api/auth/session, /cdn-cgi/trace to bootstrap Cloudflare cookies.

    Mirrors aurora cookie_bootstrap.go:bootstrapCookieJar (MIT-licensed).
    Failure on any individual URL is non-fatal — the cookie jar receives whatever
    Set-Cookie headers the server returned before the error.
    """
    urls = [
        f"{_BASE_URL}/",
        f"{_BASE_URL}/api/auth/session",
        f"{_BASE_URL}/cdn-cgi/trace",
    ]
    for url in urls:
        try:
            resp = await client.get(url, timeout=timeout)
            # Drain body so the connection can be reused.
            await resp.aread()
        except Exception as exc:
            logger.debug("oaic warmup %s failed (non-fatal): %s", url, exc)


async def _refresh_sentinel(
    *,
    client: httpx.AsyncClient,
    credential_path: str,
    access_token: str,
    device_id: str,
    timeout: float,
) -> SentinelResult:
    """Run the chat-requirements prepare→PoW→finalize cycle and persist tokens.

    Mirrors gproxy ``run_sentinel`` (sentinel.rs:74-110):

    1. ``POST /sentinel/chat-requirements/prepare`` with ``{"p": <requirements
       token>}`` carrying the bearer + Chrome browser headers (without these the
       server resolves an anonymous ``chatgpt-noauth`` persona).
    2. Solve the returned proof-of-work challenge locally.
    3. ``POST /sentinel/chat-requirements/finalize`` with the prepare_token and
       the solved proof (omitted when no PoW was required).

    Returns a :class:`SentinelResult`. ``chat_req_token`` is the finalize token
    (sent as ``openai-sentinel-chat-requirements-token``); ``proof_token`` is the
    solved PoW answer (sent as ``openai-sentinel-proof-token``) — the same string
    is used for finalize and the subsequent ``/f/conversation`` request. Persists
    the result to the credential file. Raises on any non-200 response.
    """
    headers = {
        **get_browser_headers(device_id=device_id, session_id=device_id, conversation_id="", final=False),
        "authorization": f"Bearer {access_token}",
        "content-type": "application/json",
        "accept": "*/*",
    }

    # 1. prepare
    p_token = build_requirements_token()
    prepare_resp = await client.post(
        f"{_BASE_URL}{_SENTINEL_PREPARE_PATH}",
        content=json.dumps(build_prepare_body(p_token)).encode(),
        headers=headers,
        timeout=timeout,
    )
    prepare_resp.raise_for_status()
    prepare = prepare_resp.json()
    prepare_token = str(prepare.get("prepare_token") or "")

    # 2. solve the proof-of-work challenge locally.
    proof_token = ""
    challenge = prepare.get("proofofwork")
    if isinstance(challenge, dict) and challenge.get("required"):
        seed = str(challenge.get("seed") or "")
        difficulty = str(challenge.get("difficulty") or "")
        if not (seed and difficulty):
            raise RuntimeError("sentinel prepare: proofofwork required but seed/difficulty missing")
        try:
            proof_token = solve_pow(seed, difficulty)
        except PowExhaustedError as exc:
            raise RuntimeError(f"sentinel proof-of-work exhausted: {exc}") from exc

    # 3. finalize — proof omitted when empty; turnstile deliberately not sent.
    finalize_resp = await client.post(
        f"{_BASE_URL}{_SENTINEL_FINALIZE_PATH}",
        content=json.dumps(build_finalize_body(prepare_token=prepare_token, proof=proof_token)).encode(),
        headers=headers,
        timeout=timeout,
    )
    finalize_resp.raise_for_status()
    finalize = finalize_resp.json()
    chat_req_token = str(finalize.get("token") or "")
    persona = str(finalize.get("persona") or prepare.get("persona") or "")
    expires_at_ms = decode_jwt_exp_ms(chat_req_token) or 0

    # Capture the turnstile + session-observer (so) bytecode challenges. Both are
    # base64 VM programs the browser executes (turnstile.dx → turnstile token,
    # so.collector_dx → so token). Not solved yet (test-without-turnstile first);
    # surfaced as raw material for the turnstile VM port and live analysis.
    turnstile = prepare.get("turnstile") if isinstance(prepare.get("turnstile"), dict) else {}
    turnstile_dx = str(turnstile.get("dx") or "")
    so = prepare.get("so") if isinstance(prepare.get("so"), dict) else {}
    so_collector_dx = str(so.get("collector_dx") or "")
    logger.info(
        "oaic sentinel: persona=%s turnstile_required=%s turnstile_dx=%dB so_collector_dx=%dB",
        persona,
        bool(turnstile.get("required")),
        len(turnstile_dx),
        len(so_collector_dx),
    )

    update_sentinel_fields(
        credential_path,
        chat_req_token=chat_req_token,
        proof_token=proof_token,
        chat_req_token_expires_at_ms=expires_at_ms,
        persona=persona,
        label="OpenAIConversations",
    )

    return SentinelResult(
        chat_req_token=chat_req_token,
        proof_token=proof_token,
        expires_at_ms=expires_at_ms,
        persona=persona,
        turnstile_dx=turnstile_dx,
        so_collector_dx=so_collector_dx,
    )


_CONDUIT_MAX_SUCCESS = 4
"""Defensive cap on the success-state conduit prewarm loop. The browser stops
when the server returns an empty ``conduit_token``; this bounds it regardless."""

_CONDUIT_SEED = "no-token"
"""Literal placeholder the browser sends as ``x-conduit-token`` on the first
(state=none) prepare call before any real token exists (HAR). Not a secret."""


async def _conduit_prepare_call(
    *,
    client: httpx.AsyncClient,
    final_body: dict[str, Any],
    state: _PrepareState,
    conduit_token: str,
    turn_trace_id: str,
    timeout: float,
    device_id: str,
    access_token: str,
) -> str:
    """One ``POST /f/conversation/prepare``; returns the response ``conduit_token``.

    Mirrors the browser conduit prewarm (HAR): the first call sends
    ``x-conduit-token: no-token`` with ``state=none``; subsequent ``state=success``
    calls send the token returned by the previous call, looping until the server
    returns an empty ``conduit_token``. The token is **not** stamped on the final
    ``/f/conversation`` — the chain is a server-side prewarm side effect.

    Sentinel headers are deliberately absent from prepare calls. The endpoint is
    Cloudflare-gated, so the full Chrome header shape + jar cookies are required.
    Raises on any non-2xx (fail-closed).
    """
    from ccproxy.lightllm.adapters.openai_conversations import build_conversation_prepare_body

    prepare_body = build_conversation_prepare_body(final_body=final_body, state=state)
    headers = {
        **get_browser_headers(device_id=device_id, session_id=device_id, conversation_id="", final=False),
        "authorization": f"Bearer {access_token}",
        "content-type": "application/json",
        "accept": "*/*",
        "x-oai-turn-trace-id": turn_trace_id,
        "x-openai-target-path": _PREPARE_PATH,
        "x-conduit-token": conduit_token,
    }
    resp = await client.post(
        f"{_BASE_URL}{_PREPARE_PATH}",
        content=json.dumps(prepare_body).encode(),
        headers=headers,
        timeout=timeout,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"conduit prepare({state}) failed: HTTP {resp.status_code} — {resp.text[:200]}")
    data = resp.json()
    return str(data.get("conduit_token") or "")


def _scan_sse_for_conversation_ids(raw_body: bytes) -> tuple[str, str]:
    """Scan OpenAI Conversations SSE body for conversation_id and last message id.

    Returns (conversation_id, parent_message_id). Either may be empty when not
    found. Late events overwrite earlier values.
    """
    conversation_id = ""
    parent_message_id = ""
    try:
        text = raw_body.decode("utf-8", errors="replace")
    except Exception:
        return conversation_id, parent_message_id

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data_part = line[len("data: ") :]
        if data_part == "[DONE]":
            continue
        try:
            event = json.loads(data_part)
        except (ValueError, json.JSONDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        cid = event.get("conversation_id")
        if isinstance(cid, str) and cid:
            conversation_id = cid
        msg = event.get("message")
        if isinstance(msg, dict):
            mid = msg.get("id")
            if isinstance(mid, str) and mid:
                role = (msg.get("author") or {}).get("role", "")
                if role == "assistant":
                    parent_message_id = mid

    return conversation_id, parent_message_id


class OpenAIConversationsAddon:
    """mitmproxy addon: browser pre-flight for OpenAI Conversations requests.

    Runs between the outbound pipeline and :class:`TransportOverrideAddon`.
    Every hook is gated on ``metadata.auth_provider`` resolving to a
    ``Provider`` with ``type == "openai_conversations"``; all other flows are
    byte-for-byte unaffected.
    """

    async def request(self, flow: http.HTTPFlow) -> None:
        if not _is_oaic_flow(flow):
            return
        try:
            await self._prepare_request(flow)
        except Exception:
            logger.error("OpenAIConversationsAddon.request failed", exc_info=True)
            raise

    async def responseheaders(self, flow: http.HTTPFlow) -> None:
        """Buffer image responses so ``response()`` can poll/download/repackage.

        Image flows carry no ``TransformMeta`` (no SSEPipeline is installed), so
        ``InspectorAddon.responseheaders`` would stream the raw ``/f/conversation``
        SSE straight to the client. Forcing ``stream = False`` here — this runs
        after ``InspectorAddon`` in the addon chain — buffers the body instead.
        """
        if not flow.response or not _is_oaic_flow(flow):
            return
        if not metadata_from_flow(flow).oaic_image_operation:
            return
        if "text/event-stream" in flow.response.headers.get("content-type", ""):
            flow.response.stream = False

    async def response(self, flow: http.HTTPFlow) -> None:
        if not flow.response or not _is_oaic_flow(flow):
            return
        try:
            await self._handle_response(flow)
        except Exception:
            logger.error("OpenAIConversationsAddon.response failed", exc_info=True)

    # ------------------------------------------------------------------
    # request-side orchestration
    # ------------------------------------------------------------------

    async def _prepare_request(self, flow: http.HTTPFlow) -> None:
        metadata = metadata_from_flow(flow)
        provider_name = metadata.auth_provider

        config = get_config()
        provider = config.providers[provider_name]
        oaic_cfg = config.lightllm.openai_conversations

        profile = provider.fingerprint_profile or transport.DEFAULT_PROFILE
        client = await transport.get_client(host=provider.host, profile=profile)

        # Load credential state — need device_id + chat-requirements fields.
        credential_path = getattr(provider.auth, "file_path", "") if provider.auth else ""
        state = load_credential_state(path=credential_path, label="OpenAIConversations") if credential_path else None
        if state is None:
            # No credential state — try to proceed without warmup/sentinel.
            logger.warning(
                "oaic: no credential state loaded for provider %s; skipping warmup/sentinel",
                provider_name,
            )
            device_id = str(uuid.uuid4())
            chat_req_token = ""
            proof_token = ""
            access_token = ""
        else:
            device_id = state.device_id or str(uuid.uuid4())
            chat_req_token = state.chat_req_token
            proof_token = state.proof_token
            access_token = state.access_token

        timeout = oaic_cfg.request_timeout_seconds

        # 0. Load browser cookies (cf_clearance + session) into the shared jar.
        # These supply the Cloudflare clearance warmup cannot synthesize; with
        # them present, warmup (which lacks browser headers and would 403) is
        # skipped.
        cookie_file = getattr(provider.auth, "cookie_file", "") if provider.auth else ""
        cookies_loaded = _load_cookies_from_file(client, cookie_file) if cookie_file else 0
        if cookies_loaded:
            _mark_warmup(provider_name=provider_name, profile=profile)
            logger.debug("oaic loaded %d browser cookies for provider=%s", cookies_loaded, provider_name)

        # Align the device id with the browser session (the oai-did cookie). A
        # mismatch between the oai-device-id header and the session's device id
        # trips OpenAI's "unusual activity has been detected" abuse check, so the
        # browser value must be used everywhere (Sentinel, conduit, headers).
        browser_did = _cookie_value(client, "oai-did")
        if browser_did:
            device_id = browser_did

        # 1. Warmup — bootstraps Cloudflare cookies into the shared client jar.
        if _should_warmup(
            provider_name=provider_name,
            profile=profile,
            throttle_seconds=oaic_cfg.warmup_throttle_seconds,
            client=client,
        ):
            try:
                await _run_warmup(client=client, timeout=timeout)
                _mark_warmup(provider_name=provider_name, profile=profile)
                logger.debug("oaic warmup complete for provider=%s", provider_name)
            except Exception as exc:
                logger.warning("oaic warmup failed (non-fatal): %s", exc)

        # Parse the request body up front (the conduit prepare bodies derive from
        # it) and mint the per-turn trace id shared across prepare + conversation.
        turn_trace_id = str(uuid.uuid4())
        try:
            final_body = json.loads(flow.request.content or b"{}")
        except (ValueError, json.JSONDecodeError):
            final_body = {}
        is_image = bool(metadata.oaic_image_operation)

        # 2. Conduit prewarm — phase 1 (text only): state=none seeded with the
        # "no-token" placeholder, BEFORE sentinel, mirroring the browser (HAR).
        # The returned token seeds the success loop. Image turns use a single
        # success prepare after rendering (aurora prepareImageConversation).
        conduit_seed = ""
        if not is_image:
            try:
                conduit_seed = await _conduit_prepare_call(
                    client=client,
                    final_body=final_body,
                    state="none",
                    conduit_token=_CONDUIT_SEED,
                    turn_trace_id=turn_trace_id,
                    device_id=device_id,
                    access_token=access_token,
                    timeout=timeout,
                )
            except Exception as exc:
                logger.error("oaic conduit prepare(none) failed: %s", exc)
                raise

        # 3. Sentinel refresh + PoW — when the chat-requirements token is missing
        # or within the skew window. chat_req_token + proof_token are persisted and
        # reused across the token's ~9 min TTL (cached turns skip the refresh).
        skew_ms = int(oaic_cfg.sentinel_skew_seconds * 1000)
        chat_req_expires_ms = state.chat_req_token_expires_at_ms if state else 0
        if is_expired(expiry_ms=chat_req_expires_ms, skew_ms=skew_ms):
            try:
                result = await _refresh_sentinel(
                    client=client,
                    credential_path=credential_path,
                    access_token=access_token,
                    device_id=device_id,
                    timeout=timeout,
                )
            except Exception as exc:
                logger.error("oaic sentinel refresh failed: %s", exc)
                raise
            chat_req_token = result.chat_req_token
            proof_token = result.proof_token
            logger.debug(
                "oaic sentinel refreshed for provider=%s persona=%s token=%s…",
                provider_name,
                result.persona,
                chat_req_token[:12] if chat_req_token else "",
            )

        # 4. Conduit prewarm — phase 2.
        if is_image:
            # Render the image body, then a single success-state prepare (aurora).
            await self._render_image_request(
                flow,
                client=client,
                provider=provider,
                device_id=device_id,
                oaic_cfg=oaic_cfg,
            )
            if flow.response is not None:
                # Rendering produced a synthetic error response; pass it through.
                return
            try:
                final_body = json.loads(flow.request.content or b"{}")
            except (ValueError, json.JSONDecodeError):
                final_body = {}
            try:
                await _conduit_prepare_call(
                    client=client,
                    final_body=final_body,
                    state="success",
                    conduit_token=_CONDUIT_SEED,
                    turn_trace_id=turn_trace_id,
                    device_id=device_id,
                    access_token=access_token,
                    timeout=timeout,
                )
            except Exception as exc:
                logger.error("oaic image conduit prepare failed: %s", exc)
                raise
        else:
            # Success loop: chain the conduit token until the server returns empty.
            tok = conduit_seed
            for _ in range(_CONDUIT_MAX_SUCCESS):
                if not tok:
                    break
                try:
                    tok = await _conduit_prepare_call(
                        client=client,
                        final_body=final_body,
                        state="success",
                        conduit_token=tok,
                        turn_trace_id=turn_trace_id,
                        device_id=device_id,
                        access_token=access_token,
                        timeout=timeout,
                    )
                except Exception as exc:
                    logger.error("oaic conduit prepare(success) failed: %s", exc)
                    raise

        # 4. Replace the request headers with a clean, Chrome-ordered set.
        # /f/conversation is Cloudflare-gated like prepare: cf_clearance is bound
        # to the request fingerprint. The original listener headers (appended to
        # in mitmproxy insertion order) fail the check, so the whole header block
        # is rebuilt in browser order — exactly like the conduit-prepare call.
        conversation_id = str(final_body.get("conversation_id") or "")
        target_path = flow.request.path

        final_headers: dict[str, str] = {
            **get_browser_headers(
                device_id=device_id,
                session_id=device_id,
                conversation_id=conversation_id,
                final=True,
            ),
            "authorization": f"Bearer {access_token}",
            "content-type": "application/json",
            "accept": "text/event-stream",
            # Sentinel goes as two separate headers (gproxy channel.rs:485-489),
            # NOT a single combined openai-sentinel-token blob: the
            # chat-requirements token is the finalize token, the proof token is
            # the solved PoW answer.
            "openai-sentinel-chat-requirements-token": chat_req_token,
            "openai-sentinel-proof-token": proof_token,
            "x-oai-turn-trace-id": turn_trace_id,
            "x-openai-target-path": target_path,
            # No x-conduit-token: the browser does NOT send it on /f/conversation
            # (HAR); the conduit prewarm chain is a server-side side effect.
        }
        oai_is_value = _get_oai_is_from_jar(client)
        if oai_is_value:
            final_headers["x-oai-is"] = oai_is_value

        # Replacing the header block drops Content-Length; re-apply the body so
        # mitmproxy re-stamps it (otherwise the upstream receives an empty body).
        body_bytes = flow.request.content or b""
        flow.request.headers = http.Headers([(k.encode(), v.encode()) for k, v in final_headers.items()])
        flow.request.content = body_bytes

        logger.debug(
            "oaic stamped: provider=%s profile=%s trace=%s persona_token=%s…",
            provider_name,
            profile,
            turn_trace_id[:8],
            chat_req_token[:12] if chat_req_token else "",
        )

    # ------------------------------------------------------------------
    # response-side orchestration
    # ------------------------------------------------------------------

    async def _handle_response(self, flow: http.HTTPFlow) -> None:
        assert flow.response is not None
        metadata = metadata_from_flow(flow)

        # X-OAI-IS-Update: write back to cookie jar before anything else.
        oai_is_update = flow.response.headers.get("x-oai-is-update") or flow.response.headers.get("X-OAI-IS-Update")
        if oai_is_update:
            await self._write_oai_is_to_jar(flow=flow, value=oai_is_update)

        # Image flows: poll/download/repackage into an OpenAI images.response.
        if metadata.oaic_image_operation:
            await self._handle_image_response(flow)
            return

        status = flow.response.status_code
        is_cf_challenge = bool(flow.response.headers.get("cf-mitigated"))

        if (status in (401, 403) or is_cf_challenge) and not metadata.oaic_retry_done:
            await self._retry_once(flow)
            return

        # ConversationStore write-back from completed SSE.
        await self._write_back_conversation(flow)

        # WS handoff: DEFERRED.
        # When CHATGPT-004 lands, the intake FSM surfaces a typed
        # ``stream_handoff`` event containing a topic_id.  The 002 addon
        # should then open ``GET /backend-api/celsius/ws/user`` → wss →
        # subscribe to the topic → pipe frames back through the intake FSM.
        # Implementation deferred to the CHATGPT-004 joint capstone.
        # See: aurora request.go:622-707, 810-891, 1213-1268.

    # ------------------------------------------------------------------
    # image generation / edit
    # ------------------------------------------------------------------

    async def _render_image_request(
        self,
        flow: http.HTTPFlow,
        *,
        client: httpx.AsyncClient,
        provider: Provider,
        device_id: str,
        oaic_cfg: OpenAIConversationsConfig,
    ) -> None:
        """Render an image request into a ``/backend-api/f/conversation`` body.

        Image edits upload the source image first. On any parse/upload error,
        sets a synthetic OpenAI error response and clears the image flag so the
        rest of the pipeline passes it through untouched.
        """
        from ccproxy.lightllm.openai import conversations_images as images

        metadata = metadata_from_flow(flow)
        operation = metadata.oaic_image_operation
        original = flow.request.content or b""
        content_type_in = flow.request.headers.get("content-type", "")
        access_token = _bearer_from_flow(flow)
        timeout = oaic_cfg.request_timeout_seconds

        try:
            if operation == "edit":
                parsed_edit = image_parse.parse_image_edit_request(body=original, content_type=content_type_in)
                uploaded = await images.upload_image(
                    client=client,
                    base_url=_BASE_URL,
                    access_token=access_token,
                    device_id=device_id,
                    parsed=parsed_edit,
                    timeout=timeout,
                )
                body = images.build_image_conversation_body(
                    prompt=parsed_edit.prompt,
                    model=parsed_edit.model,
                    uploaded_image=uploaded,
                )
            else:
                parsed_gen = image_parse.parse_image_gen_request(original)
                body = images.build_image_conversation_body(prompt=parsed_gen.prompt, model=parsed_gen.model)
        except image_parse.RemoteImageURLError as exc:
            metadata.oaic_image_operation = ""
            self._fail_image(flow, status=400, message=str(exc))
            return
        except (ValueError, images.ImageGenerationError) as exc:
            metadata.oaic_image_operation = ""
            self._fail_image(flow, status=502, message=f"image request preparation failed: {exc}")
            return

        target_path = provider.path or _CONVERSATION_PATH
        flow.request.method = "POST"
        flow.request.scheme = "https"
        flow.request.host = provider.host
        flow.request.port = 443
        flow.request.path = target_path
        flow.server_conn = Server(address=(provider.host, 443))
        flow.request.headers["content-type"] = "application/json"
        flow.request.content = json.dumps(body).encode()
        logger.debug("oaic image render: op=%s → %s%s", operation, provider.host, target_path)

    async def _handle_image_response(self, flow: http.HTTPFlow) -> None:
        """Poll/download chatgpt.com image assets → OpenAI ``images.response``."""
        from ccproxy.lightllm.openai import conversations_images as images

        assert flow.response is not None
        metadata = metadata_from_flow(flow)

        if flow.response.status_code >= 400:
            status = flow.response.status_code
            upstream_body = _extract_raw_body(flow)[:600].decode("utf-8", errors="replace")
            logger.warning("oaic image upstream HTTP %d body: %s", status, upstream_body)
            self._fail_image(
                flow,
                status=502,
                message=f"chatgpt.com HTTP {status} for image request: {upstream_body[:240]}",
            )
            return

        config = get_config()
        provider = config.providers.get(metadata.auth_provider)
        if provider is None:
            self._fail_image(flow, status=502, message="image provider is not configured")
            return

        oaic_cfg = config.lightllm.openai_conversations
        profile = provider.fingerprint_profile or transport.DEFAULT_PROFILE
        access_token = _bearer_from_flow(flow)
        device_id = flow.request.headers.get("oai-device-id", "")
        timeout = oaic_cfg.request_timeout_seconds

        raw_body = _extract_raw_body(flow)
        conversation_id, _ = _scan_sse_for_conversation_ids(raw_body)
        pointers = images.extract_pointers_from_sse(raw_body)

        try:
            client = await transport.get_client(host=provider.host, profile=profile)
            if not pointers:
                if not conversation_id:
                    raise images.ImageGenerationError("no conversation_id in image response; cannot poll for assets")
                pointers = await images.poll_conversation_for_pointers(
                    client=client,
                    base_url=_BASE_URL,
                    conversation_id=conversation_id,
                    access_token=access_token,
                    device_id=device_id,
                    interval_seconds=oaic_cfg.image_poll_interval_seconds,
                    max_attempts=oaic_cfg.image_poll_max_attempts,
                    timeout=timeout,
                )
            downloaded: list[bytes] = []
            for pointer in pointers:
                downloaded.append(
                    await images.download_image(
                        client=client,
                        base_url=_BASE_URL,
                        pointer=pointer,
                        conversation_id=conversation_id,
                        access_token=access_token,
                        device_id=device_id,
                        timeout=timeout,
                    )
                )
            if not downloaded:
                raise images.ImageGenerationError("image generation produced no downloadable assets")
            payload = images.build_images_response(downloaded)
        except images.ImageGenerationError as exc:
            self._fail_image(flow, status=502, message=str(exc))
            return
        except Exception as exc:
            # Convert any transport/parse failure into a clean client error
            # rather than leaking the raw ChatGPT SSE body downstream.
            logger.error("oaic image response handling failed", exc_info=True)
            self._fail_image(flow, status=502, message=f"image response handling failed: {exc}")
            return

        self._set_json_response(flow, status=200, payload=payload)
        logger.info(
            "oaic image response: provider=%s op=%s images=%d",
            metadata.auth_provider,
            metadata.oaic_image_operation,
            len(downloaded),
        )

    def _set_json_response(self, flow: http.HTTPFlow, *, status: int, payload: dict[str, Any]) -> None:
        assert flow.response is not None
        flow.response.status_code = status
        flow.response.content = json.dumps(payload).encode()
        flow.response.headers["content-type"] = "application/json"
        if "content-encoding" in flow.response.headers:
            del flow.response.headers["content-encoding"]

    def _fail_image(self, flow: http.HTTPFlow, *, status: int, message: str) -> None:
        """Set an OpenAI-shape error response for a failed image request."""
        payload: dict[str, Any] = {"error": {"message": message, "type": "api_error", "code": status}}
        if flow.response is None:
            flow.response = http.Response.make(
                status,
                json.dumps(payload).encode(),
                {"content-type": "application/json"},
            )
        else:
            self._set_json_response(flow, status=status, payload=payload)
        logger.warning("oaic image error (%d): %s", status, message)

    async def _retry_once(self, flow: http.HTTPFlow) -> None:
        """One-shot 401/403/CF-challenge retry.

        Invalidates the cached sentinel state, forces a warmup, refreshes the
        sentinel, then replays the request via the same cached client. Marks
        both the loop-guard and the AuthAddon-skip flags before returning.
        """
        metadata = metadata_from_flow(flow)
        metadata.oaic_retry_done = True
        # Prevent AuthAddon's generic 401 replay from firing on top of this one.
        metadata.auth_injected = False

        provider_name = metadata.auth_provider
        if not provider_name:
            return

        config = get_config()
        provider = config.providers.get(provider_name)
        if provider is None or provider.type != "openai_conversations":
            return

        oaic_cfg = config.lightllm.openai_conversations
        profile = provider.fingerprint_profile or transport.DEFAULT_PROFILE
        client = await transport.get_client(host=provider.host, profile=profile)
        credential_path = getattr(provider.auth, "file_path", "") if provider.auth else ""
        state = load_credential_state(path=credential_path, label="OpenAIConversations") if credential_path else None
        device_id = (state.device_id or str(uuid.uuid4())) if state else str(uuid.uuid4())
        access_token = state.access_token if state else _bearer_from_flow(flow)
        timeout = oaic_cfg.request_timeout_seconds

        # Force warmup.
        _warmup_timestamps.pop((provider_name, profile), None)
        try:
            await _run_warmup(client=client, timeout=timeout)
            _mark_warmup(provider_name=provider_name, profile=profile)
        except Exception as exc:
            logger.warning("oaic retry warmup failed (non-fatal): %s", exc)

        # Force sentinel refresh + PoW.
        try:
            result = await _refresh_sentinel(
                client=client,
                credential_path=credential_path,
                access_token=access_token,
                device_id=device_id,
                timeout=timeout,
            )
        except Exception as exc:
            logger.error("oaic retry sentinel refresh failed: %s", exc)
            return

        # Stamp the refreshed sentinel headers onto the request.
        flow.request.headers["openai-sentinel-chat-requirements-token"] = result.chat_req_token
        flow.request.headers["openai-sentinel-proof-token"] = result.proof_token

        # Replay via the same cached client (bypassing the sidecar rewrite).
        headers = dict(flow.request.headers)
        headers.pop("x-ccproxy-auth-injected", None)

        try:
            retry_resp = await client.request(
                method=flow.request.method,
                url=flow.request.pretty_url,
                headers=headers,
                content=flow.request.content,
                timeout=timeout,
            )
        except Exception as exc:
            logger.error("oaic one-shot retry request failed: %s", exc)
            return

        assert flow.response is not None
        flow.response.status_code = retry_resp.status_code
        flow.response.headers.clear()
        for key, value in retry_resp.headers.multi_items():
            flow.response.headers.add(key, value)
        flow.response.content = retry_resp.content

        logger.info(
            "oaic one-shot retry completed: provider=%s status=%d",
            provider_name,
            retry_resp.status_code,
        )

    async def _write_back_conversation(self, flow: http.HTTPFlow) -> None:
        """Scan the upstream SSE and persist conversation identifiers to the L1 store."""
        assert flow.response is not None
        metadata = metadata_from_flow(flow)

        if flow.response.status_code >= 400:
            return

        conv_id = metadata.conversation_id
        if not isinstance(conv_id, str) or not conv_id:
            return

        raw_body = _extract_raw_body(flow)
        if not raw_body:
            return

        chatgpt_conv_id, parent_message_id = _scan_sse_for_conversation_ids(raw_body)
        if not chatgpt_conv_id or not parent_message_id:
            logger.debug(
                "oaic write-back: no conversation_id/parent_message_id found in SSE (conv=%s)",
                conv_id[:8],
            )
            return

        store = get_conversation_store()
        store.save(
            key=conv_id,
            conversation_id=chatgpt_conv_id,
            parent_message_id=parent_message_id,
        )
        logger.debug(
            "oaic write-back: saved conv=%s chatgpt_id=%s parent=%s",
            conv_id[:8],
            chatgpt_conv_id[:8],
            parent_message_id[:8],
        )

    async def _write_oai_is_to_jar(self, *, flow: http.HTTPFlow, value: str) -> None:
        """Write X-OAI-IS-Update back to the shared client cookie jar (best-effort)."""
        metadata = metadata_from_flow(flow)
        provider_name = metadata.auth_provider
        if not provider_name:
            return
        provider = get_config().providers.get(provider_name)
        if provider is None:
            return
        profile = provider.fingerprint_profile or transport.DEFAULT_PROFILE
        try:
            client = await transport.get_client(host=provider.host, profile=profile)
            client.cookies.set("__Secure-oai-is", value, domain="chatgpt.com")
        except Exception as exc:
            logger.debug("oaic X-OAI-IS-Update jar write failed (non-fatal): %s", exc)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _get_oai_is_from_jar(client: httpx.AsyncClient) -> str:
    """Read ``__Secure-oai-is`` from the client cookie jar (best-effort).

    Returns an empty string when the cookie is absent or the jar is not
    accessible (curl-cffi may not expose a queryable jar in all configurations —
    see live-probe note in CHATGPT-002 acceptance checks).
    """
    try:
        return client.cookies.get("__Secure-oai-is", domain="chatgpt.com") or ""
    except Exception:
        return ""


def _extract_raw_body(flow: http.HTTPFlow) -> bytes:
    """Extract the raw upstream SSE body from the flow record or response.

    Mirrors :meth:`PerplexityAddon._extract_raw_body` exactly.
    """
    metadata = metadata_from_flow(flow)
    record = metadata.record
    provider_resp = getattr(record, "provider_response", None) if record else None
    if provider_resp is not None:
        body = getattr(provider_resp, "body", None)
        if isinstance(body, bytes) and body:
            return body
    transformer = metadata.sse_transformer
    if transformer is not None and hasattr(transformer, "raw_body"):
        raw = transformer.raw_body
        if isinstance(raw, bytes) and raw:
            return raw
    if flow.response is not None:
        try:
            return flow.response.content or b""
        except Exception:
            return b""
    return b""
