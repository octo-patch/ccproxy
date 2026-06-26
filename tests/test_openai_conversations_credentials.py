# ruff: noqa: S105, S106
"""Tests for OpenAI Conversations credentials and Sentinel helpers.

Covers:
  - build_sentinel_req_body: JSON shape with p, id, flow keys
  - build_sentinel_token_header: compact JSON with p, t, c, id, flow (in order)
  - decode_jwt_exp_ms + is_expired: JWT expiry decoding and skew
  - OpenAIConversationsCredentialState load / round-trip / unknown-sibling preservation
  - Schema has none of the forbidden fields (x_oai_is, x_conduit_token, etc.)
  - OpenAIConversationsAuthSource: parses through Provider.auth, resolve() returns token
  - parse_auth_source dispatches type: openai_conversations
"""

from __future__ import annotations

import base64
import json
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ccproxy.auth.sources import (
    OpenAIConversationsAuthSource,
    parse_auth_source,
)
from ccproxy.config import Provider
from ccproxy.openai_conversations.credentials import (
    OpenAIConversationsCredentialState,
    load_credential_state,
    update_sentinel_fields,
)
from ccproxy.openai_conversations.sentinel import (
    build_sentinel_req_body,
    build_sentinel_token_header,
    decode_jwt_exp_ms,
    is_expired,
)

# ---------------------------------------------------------------------------
# build_sentinel_req_body
# ---------------------------------------------------------------------------


def test_sentinel_req_body_has_p_id_flow_keys() -> None:
    """build_sentinel_req_body must return JSON with exactly p, id, flow."""
    body = build_sentinel_req_body(
        p="gAAAAACrequirements~S",
        device_id="device-uuid-1234",
        flow="conversation",
    )
    parsed = json.loads(body)
    assert set(parsed.keys()) == {"p", "id", "flow"}
    assert parsed["p"] == "gAAAAACrequirements~S"
    assert parsed["id"] == "device-uuid-1234"
    assert parsed["flow"] == "conversation"


def test_sentinel_req_body_default_flow() -> None:
    """Default flow is 'conversation'."""
    body = build_sentinel_req_body(p="tok", device_id="did")
    parsed = json.loads(body)
    assert parsed["flow"] == "conversation"


def test_sentinel_req_body_is_compact_json() -> None:
    """build_sentinel_req_body must not include extra whitespace."""
    body = build_sentinel_req_body(p="p", device_id="id", flow="conversation")
    assert " " not in body
    assert "\n" not in body


# ---------------------------------------------------------------------------
# build_sentinel_token_header
# ---------------------------------------------------------------------------


def test_sentinel_token_header_key_order_p_t_c_id_flow() -> None:
    """openai-sentinel-token must serialize as compact JSON with keys p,t,c,id,flow."""
    header = build_sentinel_token_header(
        p="gAAAAACrequirements~S",
        turnstile_token="",
        sentinel_token="server-token-abc",
        device_id="device-uuid-5678",
        flow="conversation",
    )
    parsed = json.loads(header)
    assert list(parsed.keys()) == ["p", "t", "c", "id", "flow"]
    assert parsed["p"] == "gAAAAACrequirements~S"
    assert parsed["t"] == ""
    assert parsed["c"] == "server-token-abc"
    assert parsed["id"] == "device-uuid-5678"
    assert parsed["flow"] == "conversation"


def test_sentinel_token_header_is_compact_json() -> None:
    """No whitespace in the serialized header value."""
    header = build_sentinel_token_header(p="p", turnstile_token="t", sentinel_token="c", device_id="id")
    assert " " not in header
    assert "\n" not in header


def test_sentinel_token_header_default_flow() -> None:
    """Default flow is 'conversation'."""
    header = build_sentinel_token_header(p="p", turnstile_token="", sentinel_token="c", device_id="id")
    parsed = json.loads(header)
    assert parsed["flow"] == "conversation"


# ---------------------------------------------------------------------------
# JWT expiry helpers
# ---------------------------------------------------------------------------


def _make_jwt(exp_unix: int) -> str:
    """Build a minimal JWT with the given exp claim (no signature)."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload_bytes = json.dumps({"exp": exp_unix}).encode()
    payload = base64.urlsafe_b64encode(payload_bytes).decode().rstrip("=")
    return f"{header}.{payload}.sig"


def test_decode_jwt_exp_ms_returns_unix_millis() -> None:
    """decode_jwt_exp_ms converts exp (seconds) to milliseconds."""
    exp_unix = 1_800_000_000
    token = _make_jwt(exp_unix=exp_unix)
    result = decode_jwt_exp_ms(token=token)
    assert result == exp_unix * 1000


def test_decode_jwt_exp_ms_returns_none_for_non_jwt() -> None:
    """Non-JWT strings return None without raising."""
    assert decode_jwt_exp_ms(token="not-a-jwt") is None
    assert decode_jwt_exp_ms(token="") is None
    assert decode_jwt_exp_ms(token="only.twoparts") is None


def test_decode_jwt_exp_ms_returns_none_when_no_exp_claim() -> None:
    """JWT without exp claim returns None."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(b'{"sub":"user"}').decode().rstrip("=")
    token = f"{header}.{payload}.sig"
    assert decode_jwt_exp_ms(token=token) is None


@dataclass(frozen=True)
class IsExpiredCase:
    """Test case for is_expired."""

    name: str
    expiry_ms: int
    skew_ms: int
    expected: bool


IS_EXPIRED_CASES: list[IsExpiredCase] = [
    IsExpiredCase(
        name="zero_expiry_always_expired",
        expiry_ms=0,
        skew_ms=0,
        expected=True,
    ),
    IsExpiredCase(
        name="far_future_not_expired",
        expiry_ms=9_999_999_999_000,
        skew_ms=0,
        expected=False,
    ),
    IsExpiredCase(
        name="past_expiry_is_expired",
        expiry_ms=1_000_000,
        skew_ms=0,
        expected=True,
    ),
]


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.name) for c in IS_EXPIRED_CASES],
)
def test_is_expired(case: IsExpiredCase) -> None:
    result = is_expired(expiry_ms=case.expiry_ms, skew_ms=case.skew_ms)
    assert result == case.expected


def test_is_expired_skew_pushes_valid_token_to_expired() -> None:
    """A token expiring in 30 s is considered expired with a 60 000 ms skew."""
    expiry_ms = int(time.time() * 1000) + 30_000
    assert not is_expired(expiry_ms=expiry_ms, skew_ms=0)
    assert is_expired(expiry_ms=expiry_ms, skew_ms=60_000)


# ---------------------------------------------------------------------------
# OpenAIConversationsCredentialState load / round-trip
# ---------------------------------------------------------------------------


def _write_creds(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data))


def test_load_credential_state_returns_state_for_valid_file(tmp_path: Path) -> None:
    """load_credential_state returns a typed state for a valid credential file."""
    creds_path = tmp_path / "creds.json"
    _write_creds(
        path=creds_path,
        data={
            "access_token": "bearer.jwt.token",
            "sentinel_token": "server-c-token",
            "sentinel_p_token": "gAAAAACrequirements~S",
            "sentinel_expires_at_ms": 1_800_000_000_000,
            "sentinel_flow": "conversation",
            "sentinel_so_token": "",
            "persona": "chatgpt-paid",
            "device_id": "uuid-1234-5678",
        },
    )
    state = load_credential_state(path=creds_path)
    assert state is not None
    assert state.access_token == "bearer.jwt.token"
    assert state.sentinel_token == "server-c-token"
    assert state.sentinel_p_token == "gAAAAACrequirements~S"
    assert state.sentinel_expires_at_ms == 1_800_000_000_000
    assert state.sentinel_flow == "conversation"
    assert state.sentinel_so_token == ""
    assert state.persona == "chatgpt-paid"
    assert state.device_id == "uuid-1234-5678"


def test_load_credential_state_missing_file_returns_none(tmp_path: Path) -> None:
    """Missing credential file returns None."""
    result = load_credential_state(path=tmp_path / "missing.json")
    assert result is None


def test_load_credential_state_corrupt_json_returns_none(tmp_path: Path) -> None:
    """Malformed JSON returns None."""
    creds_path = tmp_path / "bad.json"
    creds_path.write_text("not json{")
    assert load_credential_state(path=creds_path) is None


def test_load_credential_state_missing_access_token_returns_none(tmp_path: Path) -> None:
    """File without access_token returns None."""
    creds_path = tmp_path / "no-token.json"
    _write_creds(path=creds_path, data={"sentinel_token": "tok"})
    assert load_credential_state(path=creds_path) is None


def test_load_credential_state_preserves_unknown_sibling_fields(tmp_path: Path) -> None:
    """Unknown fields in the JSON file are preserved on round-trip."""
    creds_path = tmp_path / "creds.json"
    _write_creds(
        path=creds_path,
        data={
            "access_token": "jwt",
            "device_id": "uuid-x",
            "custom_flag": True,
            "extra_metadata": {"key": "value"},
        },
    )
    state = load_credential_state(path=creds_path)
    assert state is not None
    out = state.to_dict()
    assert out["custom_flag"] is True
    assert out["extra_metadata"] == {"key": "value"}
    assert out["access_token"] == "jwt"


def test_to_dict_round_trips_all_known_fields(tmp_path: Path) -> None:
    """to_dict() returns all eight canonical credential fields."""
    creds_path = tmp_path / "creds.json"
    _write_creds(
        path=creds_path,
        data={
            "access_token": "tok",
            "sentinel_token": "stok",
            "sentinel_p_token": "ptok",
            "sentinel_expires_at_ms": 12345,
            "sentinel_flow": "chatgpt",
            "sentinel_so_token": "so",
            "persona": "chatgpt-free",
            "device_id": "did-abc",
        },
    )
    state = load_credential_state(path=creds_path)
    assert state is not None
    out = state.to_dict()
    assert out["access_token"] == "tok"
    assert out["sentinel_token"] == "stok"
    assert out["sentinel_p_token"] == "ptok"
    assert out["sentinel_expires_at_ms"] == 12345
    assert out["sentinel_flow"] == "chatgpt"
    assert out["sentinel_so_token"] == "so"
    assert out["persona"] == "chatgpt-free"
    assert out["device_id"] == "did-abc"


def test_credential_state_has_no_forbidden_fields() -> None:
    """OpenAIConversationsCredentialState has none of the forbidden field names."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(OpenAIConversationsCredentialState)}
    forbidden = {
        "x_oai_is",
        "x_conduit_token",
        "oai_is",
        "turnstile_token",
        "oai_telemetry",
    }
    overlap = field_names & forbidden
    assert overlap == set(), f"Forbidden fields present: {overlap}"


def test_update_sentinel_fields_writes_and_preserves_siblings(tmp_path: Path) -> None:
    """update_sentinel_fields atomically updates sentinel fields, preserving sibling fields."""
    creds_path = tmp_path / "creds.json"
    _write_creds(
        path=creds_path,
        data={
            "access_token": "jwt-bearer",
            "device_id": "uuid-123",
            "persona": "chatgpt-paid",
            "sentinel_token": "",
            "sentinel_p_token": "",
            "sentinel_expires_at_ms": 0,
            "sentinel_flow": "conversation",
            "sentinel_so_token": "",
            "custom_extra": "preserve-me",
        },
    )
    result = update_sentinel_fields(
        path=creds_path,
        sentinel_token="new-server-token",
        sentinel_p_token="gAAAAABproof~S",
        sentinel_expires_at_ms=1_800_000_000_000,
        sentinel_flow="conversation",
        sentinel_so_token="so-tok",
    )
    assert result is True
    on_disk = json.loads(creds_path.read_text())
    assert on_disk["sentinel_token"] == "new-server-token"
    assert on_disk["sentinel_p_token"] == "gAAAAABproof~S"
    assert on_disk["sentinel_expires_at_ms"] == 1_800_000_000_000
    assert on_disk["sentinel_so_token"] == "so-tok"
    assert on_disk["access_token"] == "jwt-bearer"
    assert on_disk["custom_extra"] == "preserve-me"


def test_update_sentinel_fields_missing_file_returns_false(tmp_path: Path) -> None:
    """update_sentinel_fields returns False when the credential file is missing."""
    result = update_sentinel_fields(
        path=tmp_path / "missing.json",
        sentinel_token="tok",
        sentinel_p_token="p",
        sentinel_expires_at_ms=0,
    )
    assert result is False


def test_update_sentinel_fields_writes_with_mode_0600(tmp_path: Path) -> None:
    """Credential file written by update_sentinel_fields must have mode 0o600."""
    creds_path = tmp_path / "creds.json"
    _write_creds(
        path=creds_path,
        data={"access_token": "jwt", "device_id": "uuid", "persona": "chatgpt-paid"},
    )
    update_sentinel_fields(
        path=creds_path,
        sentinel_token="tok",
        sentinel_p_token="p",
        sentinel_expires_at_ms=12345,
    )
    mode = creds_path.stat().st_mode & 0o777
    assert mode == stat.S_IRUSR | stat.S_IWUSR


# ---------------------------------------------------------------------------
# OpenAIConversationsAuthSource — parse and resolve
# ---------------------------------------------------------------------------


def test_parse_auth_source_dispatches_openai_conversations(tmp_path: Path) -> None:
    """parse_auth_source must dispatch type: openai_conversations."""
    creds_path = str(tmp_path / "creds.json")
    source = parse_auth_source({"type": "openai_conversations", "file_path": creds_path})
    assert isinstance(source, OpenAIConversationsAuthSource)
    assert source.file_path == creds_path


def test_openai_conversations_auth_source_defaults() -> None:
    """OpenAIConversationsAuthSource has sensible defaults."""
    source = OpenAIConversationsAuthSource()
    assert source.type == "openai_conversations"
    assert "openai-conversations-credentials.json" in source.file_path


def test_openai_conversations_auth_source_resolve_returns_access_token(tmp_path: Path) -> None:
    """resolve() returns the access_token from the credential file."""
    creds_path = tmp_path / "creds.json"
    _write_creds(
        path=creds_path,
        data={"access_token": "my-bearer-jwt", "device_id": "uuid"},
    )
    source = OpenAIConversationsAuthSource(file_path=str(creds_path))
    assert source.resolve() == "my-bearer-jwt"


def test_openai_conversations_auth_source_resolve_missing_file_returns_none(tmp_path: Path) -> None:
    """resolve() returns None when the credential file does not exist."""
    source = OpenAIConversationsAuthSource(file_path=str(tmp_path / "missing.json"))
    assert source.resolve() is None


def test_openai_conversations_auth_source_resolve_missing_token_returns_none(tmp_path: Path) -> None:
    """resolve() returns None when access_token is absent from the credential file."""
    creds_path = tmp_path / "creds.json"
    _write_creds(path=creds_path, data={"device_id": "uuid"})
    source = OpenAIConversationsAuthSource(file_path=str(creds_path))
    assert source.resolve() is None


def test_openai_conversations_auth_source_in_provider_auth(tmp_path: Path) -> None:
    """OpenAIConversationsAuthSource parses through Provider.auth and resolve() works."""
    creds_path = tmp_path / "creds.json"
    _write_creds(path=creds_path, data={"access_token": "provider-jwt", "device_id": "uuid"})

    provider = Provider.model_validate(
        {
            "auth": {"type": "openai_conversations", "file_path": str(creds_path)},
            "host": "chatgpt.com",
            "path": "/backend-api/f/conversation",
            "type": "openai_conversations",
        }
    )
    assert provider.auth is not None
    assert isinstance(provider.auth, OpenAIConversationsAuthSource)
    assert provider.auth.resolve() == "provider-jwt"
