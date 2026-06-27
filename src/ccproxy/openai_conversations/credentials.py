"""OpenAI Conversations credential state — load/update around a flat JSON file.

Schema (current, no more and no less):
  access_token                 — ChatGPT web bearer JWT
  device_id                    — OAI-Device-Id UUID (stable per installation)
  persona                      — account persona (e.g. "chatgpt-paid")
  chat_req_token               — finalize-issued chat-requirements token, sent as
                                 the ``openai-sentinel-chat-requirements-token`` header
  proof_token                  — solved proof-of-work answer (``gAAAAAB…~S``), sent
                                 as the ``openai-sentinel-proof-token`` header
  chat_req_token_expires_at_ms — chat_req_token expiry as unix milliseconds

Explicitly excluded fields (request-derived, cookie-derived, or request-only):
  x_oai_is, x_conduit_token, turnstile tokens, OAI-Telemetry.

Reuses :func:`ccproxy.utils.atomic_write_back` for atomic writes;
no second atomic writer is implemented here.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ccproxy.utils import atomic_write_back

logger = logging.getLogger(__name__)

_FORBIDDEN_FIELDS = frozenset(
    {
        "x_oai_is",
        "x_conduit_token",
        "oai_is",
        "turnstile_token",
        "oai_telemetry",
        "OAI-Telemetry",
        "X-OAI-IS",
        "x-conduit-token",
    }
)


@dataclass
class OpenAIConversationsCredentialState:
    """Typed view of the flat credential JSON file.

    Constructed via :func:`load_credential_state`; written back via
    :func:`update_sentinel_fields`. Unknown sibling fields in the source
    JSON are preserved in ``_extra`` and round-tripped to disk.
    """

    access_token: str
    """ChatGPT web bearer JWT."""

    device_id: str = ""
    """OAI-Device-Id UUID (stable per installation)."""

    persona: str = "chatgpt-paid"
    """Account persona."""

    chat_req_token: str = ""
    """Finalize-issued chat-requirements token (sent as
    ``openai-sentinel-chat-requirements-token``)."""

    proof_token: str = ""
    """Solved proof-of-work answer ``gAAAAAB…~S`` (sent as
    ``openai-sentinel-proof-token``)."""

    chat_req_token_expires_at_ms: int = 0
    """chat_req_token expiry as unix milliseconds (0 = expired/unknown)."""

    _extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the flat JSON schema, merging back unknown siblings."""
        out: dict[str, Any] = dict(self._extra)
        out["access_token"] = self.access_token
        out["device_id"] = self.device_id
        out["persona"] = self.persona
        out["chat_req_token"] = self.chat_req_token
        out["proof_token"] = self.proof_token
        out["chat_req_token_expires_at_ms"] = self.chat_req_token_expires_at_ms
        return out


_KNOWN_KEYS = frozenset(
    {
        "access_token",
        "device_id",
        "persona",
        "chat_req_token",
        "proof_token",
        "chat_req_token_expires_at_ms",
    }
)


def load_credential_state(
    path: Path | str,
    label: str = "OpenAIConversations",
) -> OpenAIConversationsCredentialState | None:
    """Read the credential JSON file and return a typed state object.

    Args:
        path: Path to the credential JSON file.
        label: Log label prefix for error messages.

    Returns:
        ``OpenAIConversationsCredentialState`` on success, ``None`` on failure
        (missing file, parse error, missing access_token). Matches the None-on-
        failure convention of other ccproxy auth sources.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        logger.error("%s credential file not found: %s", label, resolved)
        return None
    try:
        raw: Any = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("%s could not read %s: %s", label, resolved, exc)
        return None
    if not isinstance(raw, dict):
        logger.error("%s credential file must contain a JSON object: %s", label, resolved)
        return None

    data: dict[str, Any] = raw
    access_token = data.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        logger.error("%s missing or empty access_token in %s", label, resolved)
        return None

    extra = {k: v for k, v in data.items() if k not in _KNOWN_KEYS}

    return OpenAIConversationsCredentialState(
        access_token=access_token,
        device_id=str(data.get("device_id") or ""),
        persona=str(data.get("persona") or "chatgpt-paid"),
        chat_req_token=str(data.get("chat_req_token") or ""),
        proof_token=str(data.get("proof_token") or ""),
        chat_req_token_expires_at_ms=int(data.get("chat_req_token_expires_at_ms") or 0),
        _extra=extra,
    )


def update_sentinel_fields(
    path: Path | str,
    *,
    chat_req_token: str,
    proof_token: str,
    chat_req_token_expires_at_ms: int,
    persona: str = "",
    label: str = "OpenAIConversations",
) -> bool:
    """Atomically update the chat-requirements fields in the credential file.

    Reads the current file, updates the chat-requirements token, proof token,
    expiry, and (when non-empty) persona, then atomically writes the result
    back. Unknown sibling fields are preserved.

    Args:
        path: Path to the credential JSON file.
        chat_req_token: New finalize-issued chat-requirements token.
        proof_token: New solved proof-of-work answer.
        chat_req_token_expires_at_ms: New expiry as unix milliseconds.
        persona: Server-reported persona; written only when non-empty.
        label: Log label prefix.

    Returns:
        ``True`` on success, ``False`` on any read/write failure.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        logger.error("%s credential file not found for chat-requirements update: %s", label, resolved)
        return False
    try:
        raw: Any = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("%s could not read %s for chat-requirements update: %s", label, resolved, exc)
        return False
    if not isinstance(raw, dict):
        logger.error("%s credential file must contain a JSON object: %s", label, resolved)
        return False

    merged: dict[str, Any] = copy.deepcopy(raw)
    merged["chat_req_token"] = chat_req_token
    merged["proof_token"] = proof_token
    merged["chat_req_token_expires_at_ms"] = chat_req_token_expires_at_ms
    if persona:
        merged["persona"] = persona

    try:
        atomic_write_back(resolved, merged)
    except Exception as exc:
        logger.error("%s failed to write chat-requirements fields to %s: %s", label, resolved, exc)
        return False
    return True
