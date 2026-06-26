"""Current Sentinel /sentinel/req helpers for OpenAI Conversations.

Implements the 2026 single-call Sentinel protocol:
  POST /backend-api/sentinel/req
  Content-Type: text/plain;charset=UTF-8
  body: {"p":<requirementsToken>,"id":<deviceID>,"flow":<flow>}

  Response header: openai-sentinel-token = compact JSON {p,t,c,id,flow}

Cross-referenced against:
  aurora-develop/aurora  internal/chatgpt/request.go:436-479   (POSTSentinelReq)
  aurora-develop/aurora  internal/prooftoken/prooftoken.go:307-340  (BuildSentinelTokenHeader)
  basketikun/chatgpt2api utils/sentinel.py:96-159
  is7Qin/chatgpt-queue-reg backend/integrations/chatgpt/sentinel_token.py

No network calls are made in this module.
"""

from __future__ import annotations

import base64
import json
import time

_DEFAULT_FLOW = "conversation"


def build_sentinel_req_body(p: str, device_id: str, flow: str = _DEFAULT_FLOW) -> str:
    """Build the POST /sentinel/req request body as a JSON string.

    Serialized with ``separators=(",", ":")`` for compact output matching
    observed network captures. Sent with ``Content-Type: text/plain;charset=UTF-8``.

    Args:
        p: Requirements token (``gAAAAAC…~S``).
        device_id: OAI-Device-Id / oai-did UUID.
        flow: Sentinel flow identifier. Default: ``"conversation"``.

    Returns:
        Compact JSON string ``{"p":…,"id":…,"flow":…}``.
    """
    return json.dumps({"p": p, "id": device_id, "flow": flow}, separators=(",", ":"))


def build_sentinel_token_header(
    p: str,
    turnstile_token: str,
    sentinel_token: str,
    device_id: str,
    flow: str = _DEFAULT_FLOW,
) -> str:
    """Build the ``openai-sentinel-token`` header value as compact JSON.

    Key order is ``p, t, c, id, flow`` matching observed network specimens
    (aurora prooftoken.go:316-329, basketikun sentinel.py:144-149).

    Args:
        p: Requirements or proof token.
        turnstile_token: Turnstile token (empty string when not required).
        sentinel_token: Server-issued ``c`` token from /sentinel/req response.
        device_id: OAI-Device-Id / oai-did UUID.
        flow: Sentinel flow identifier. Default: ``"conversation"``.

    Returns:
        Compact JSON string ``{"p":…,"t":…,"c":…,"id":…,"flow":…}``.
    """
    return json.dumps(
        {"p": p, "t": turnstile_token, "c": sentinel_token, "id": device_id, "flow": flow},
        separators=(",", ":"),
    )


def decode_jwt_exp_ms(token: str) -> int | None:
    """Decode the ``exp`` claim from a JWT-shaped token, returned as unix milliseconds.

    Returns ``None`` when the token is not a valid JWT or has no ``exp`` claim.
    The ``exp`` claim is seconds-since-epoch per RFC 7519 §4.1.4.
    """
    parts = token.split(".")
    if len(parts) < 2 or not parts[1]:
        return None
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload + padding)
        claims = json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(claims, dict):
        return None
    exp = claims.get("exp")
    if isinstance(exp, int | float):
        return int(exp * 1000)
    return None


def is_expired(expiry_ms: int, skew_ms: int = 0) -> bool:
    """Return True when the sentinel token is expired or within the skew window.

    Args:
        expiry_ms: Token expiry as unix milliseconds (0 → always expired).
        skew_ms: Safety margin in milliseconds. The token is considered
            expired when ``now + skew_ms >= expiry_ms``.

    Returns:
        ``True`` if the token should be refreshed, ``False`` if it is valid.
    """
    if expiry_ms == 0:
        return True
    now_ms = int(time.time() * 1000)
    return now_ms + skew_ms >= expiry_ms
