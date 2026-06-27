"""Sentinel chat-requirements helpers for OpenAI Conversations.

Implements the two-call 2026 Sentinel proof-of-work protocol:

  POST /backend-api/sentinel/chat-requirements/prepare
    body: {"p": "gAAAAAC<base64(json(25-slot config))>"}
    ◀ {prepare_token, proofofwork:{required,seed,difficulty}, persona, turnstile}

  solve PoW locally

  POST /backend-api/sentinel/chat-requirements/finalize
    body: {"prepare_token": ..., "proofofwork": "gAAAAAB…~S"}  (proof omitted if empty)
    ◀ {token, persona}

  chat_req_token = finalize.token  → openai-sentinel-chat-requirements-token
  proof_token    = <PoW answer>    → openai-sentinel-proof-token
                   (the same string is sent to finalize AND on /f/conversation)

Cross-referenced against the gproxy chatgpt channel
(sdk/gproxy-channel/src/channels/chatgpt/sentinel.rs:74-183).

No network calls are made in this module — the addon composes these pure
helpers with the shared curl-cffi client.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


class SentinelPrepareBody(BaseModel):
    """``POST /sentinel/chat-requirements/prepare`` request body (sentinel.rs:118)."""

    p: str


class SentinelFinalizeBody(BaseModel):
    """``POST /sentinel/chat-requirements/finalize`` request body (sentinel.rs:128-139).

    ``proofofwork`` is ``None`` (omitted on serialization) when no PoW was required.
    """

    prepare_token: str
    proofofwork: str | None = None


@dataclass(frozen=True)
class SentinelResult:
    """Outcome of a chat-requirements prepare→PoW→finalize cycle."""

    chat_req_token: str
    """Finalize-issued token, sent as ``openai-sentinel-chat-requirements-token``."""

    proof_token: str
    """Solved proof-of-work answer, sent as ``openai-sentinel-proof-token``."""

    expires_at_ms: int
    """``chat_req_token`` expiry as unix milliseconds (0 = unknown)."""

    persona: str
    """Server-reported persona (e.g. ``"chatgpt-paid"``); ``""`` when absent."""

    turnstile_dx: str = ""
    """Turnstile VM bytecode challenge (base64) from the prepare response's
    ``turnstile.dx``; empty when turnstile is not required. Raw material for the
    (deferred) turnstile token solver."""

    so_collector_dx: str = ""
    """Session-observer collector VM bytecode (base64) from the prepare
    response's ``so.collector_dx``; empty when ``so`` is absent."""


def build_prepare_body(p: str) -> dict[str, Any]:
    """Build the ``/sentinel/chat-requirements/prepare`` request body.

    Args:
        p: Requirements token (``gAAAAAC…``).

    Returns:
        ``{"p": p}`` (sentinel.rs:118 — no device_id/flow fields).
    """
    return SentinelPrepareBody(p=p).model_dump()


def build_finalize_body(*, prepare_token: str, proof: str) -> dict[str, Any]:
    """Build the ``/sentinel/chat-requirements/finalize`` request body.

    The ``proofofwork`` field is included only when ``proof`` is non-empty
    (sentinel.rs:128-139). Turnstile is deliberately never sent.

    Args:
        prepare_token: ``prepare_token`` from the prepare response.
        proof: Solved proof-of-work answer (``gAAAAAB…~S``) or ``""``.

    Returns:
        ``{"prepare_token": …}`` plus ``"proofofwork": proof`` when ``proof``.
    """
    return SentinelFinalizeBody(prepare_token=prepare_token, proofofwork=proof or None).model_dump(exclude_none=True)


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
    """Return True when the chat-requirements token is expired or within the skew window.

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
