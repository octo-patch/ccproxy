"""OpenAI Conversations credential state — load/update around a flat JSON file.

Schema (current, no more and no less):
  access_token           — ChatGPT web bearer JWT
  sentinel_token         — server-issued c token from /sentinel/req
  sentinel_p_token       — requirements or proof p token
  sentinel_expires_at_ms — sentinel token expiry as unix milliseconds
  sentinel_flow          — flow identifier ("conversation")
  sentinel_so_token      — session-observer token (optional, may be empty)
  persona                — account persona (e.g. "chatgpt-paid")
  device_id              — OAI-Device-Id UUID (stable per installation)

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

    sentinel_token: str = ""
    """Server-issued c token from /sentinel/req."""

    sentinel_p_token: str = ""
    """Requirements or proof p token."""

    sentinel_expires_at_ms: int = 0
    """Sentinel token expiry as unix milliseconds (0 = expired/unknown)."""

    sentinel_flow: str = "conversation"
    """Sentinel flow identifier."""

    sentinel_so_token: str = ""
    """Session-observer token (optional, may be empty)."""

    persona: str = "chatgpt-paid"
    """Account persona."""

    device_id: str = ""
    """OAI-Device-Id UUID (stable per installation)."""

    _extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the flat JSON schema, merging back unknown siblings."""
        out: dict[str, Any] = dict(self._extra)
        out["access_token"] = self.access_token
        out["sentinel_token"] = self.sentinel_token
        out["sentinel_p_token"] = self.sentinel_p_token
        out["sentinel_expires_at_ms"] = self.sentinel_expires_at_ms
        out["sentinel_flow"] = self.sentinel_flow
        out["sentinel_so_token"] = self.sentinel_so_token
        out["persona"] = self.persona
        out["device_id"] = self.device_id
        return out


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

    known_keys = {
        "access_token",
        "sentinel_token",
        "sentinel_p_token",
        "sentinel_expires_at_ms",
        "sentinel_flow",
        "sentinel_so_token",
        "persona",
        "device_id",
    }
    extra = {k: v for k, v in data.items() if k not in known_keys}

    return OpenAIConversationsCredentialState(
        access_token=access_token,
        sentinel_token=str(data.get("sentinel_token") or ""),
        sentinel_p_token=str(data.get("sentinel_p_token") or ""),
        sentinel_expires_at_ms=int(data.get("sentinel_expires_at_ms") or 0),
        sentinel_flow=str(data.get("sentinel_flow") or "conversation"),
        sentinel_so_token=str(data.get("sentinel_so_token") or ""),
        persona=str(data.get("persona") or "chatgpt-paid"),
        device_id=str(data.get("device_id") or ""),
        _extra=extra,
    )


def update_sentinel_fields(
    path: Path | str,
    *,
    sentinel_token: str,
    sentinel_p_token: str,
    sentinel_expires_at_ms: int,
    sentinel_flow: str = "conversation",
    sentinel_so_token: str = "",
    label: str = "OpenAIConversations",
) -> bool:
    """Atomically update only the Sentinel fields in the credential file.

    Reads the current file, updates the four Sentinel fields, and atomically
    writes the result back. Unknown sibling fields are preserved.

    Args:
        path: Path to the credential JSON file.
        sentinel_token: New server-issued c token.
        sentinel_p_token: New p token (requirements or proof).
        sentinel_expires_at_ms: New expiry as unix milliseconds.
        sentinel_flow: Flow identifier. Default: ``"conversation"``.
        sentinel_so_token: Session-observer token. Default: ``""``.
        label: Log label prefix.

    Returns:
        ``True`` on success, ``False`` on any read/write failure.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        logger.error("%s credential file not found for sentinel update: %s", label, resolved)
        return False
    try:
        raw: Any = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("%s could not read %s for sentinel update: %s", label, resolved, exc)
        return False
    if not isinstance(raw, dict):
        logger.error("%s credential file must contain a JSON object: %s", label, resolved)
        return False

    merged: dict[str, Any] = copy.deepcopy(raw)
    merged["sentinel_token"] = sentinel_token
    merged["sentinel_p_token"] = sentinel_p_token
    merged["sentinel_expires_at_ms"] = sentinel_expires_at_ms
    merged["sentinel_flow"] = sentinel_flow
    merged["sentinel_so_token"] = sentinel_so_token

    try:
        atomic_write_back(resolved, merged)
    except Exception as exc:
        logger.error("%s failed to write sentinel fields to %s: %s", label, resolved, exc)
        return False
    return True
