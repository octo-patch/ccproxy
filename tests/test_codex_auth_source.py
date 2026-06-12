"""Tests for CodexAuthSource over Codex's auth.json shape."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from ccproxy.auth.sources import CodexAuthSource

_TEST_ENDPOINT = "https://oauth.test.example/token"


def _b64_json(value: dict[str, Any]) -> str:
    raw = json.dumps(value, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _jwt(payload: dict[str, Any]) -> str:
    return f"{_b64_json({'alg': 'none', 'typ': 'JWT'})}.{_b64_json(payload)}.sig"


def _claims(*, exp_delta_seconds: int = 3600, account_id: str = "acct_test", fedramp: bool = False) -> dict[str, Any]:
    return {
        "exp": int(time.time()) + exp_delta_seconds,
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account_id,
            "chatgpt_account_is_fedramp": fedramp,
        },
    }


def test_cached_codex_access_token_and_companion_headers(tmp_path: Path) -> None:
    access_token = _jwt(_claims(account_id="acct_cached", fedramp=True))
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "rt_cached",
                    "id_token": _jwt(_claims(account_id="acct_id_token")),
                    "account_id": "acct_cached",
                },
                "last_refresh": "2026-06-01T00:00:00Z",
            }
        )
    )

    source = CodexAuthSource(file_path=str(auth_path), endpoint=_TEST_ENDPOINT)

    assert source.resolve("Auth/codex") == access_token
    assert source.extra_headers("Auth/codex") == {
        "ChatGPT-Account-ID": "acct_cached",
        "X-OpenAI-Fedramp": "true",
    }


def test_expired_codex_access_token_refreshes_and_updates_auth_json(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _jwt(_claims(exp_delta_seconds=-60, account_id="acct_old")),
                    "refresh_token": "rt_old",
                    "id_token": _jwt(_claims(account_id="acct_old")),
                },
                "last_refresh": "2026-06-01T00:00:00Z",
            }
        )
    )
    new_access = _jwt(_claims(account_id="acct_new"))
    new_id = _jwt(_claims(account_id="acct_new"))
    rotated_grant = "rt_new"
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "access_token": new_access,
                "refresh_token": rotated_grant,
                "id_token": new_id,
            },
        )

    source = CodexAuthSource(file_path=str(auth_path), endpoint=_TEST_ENDPOINT)
    original_refresh = CodexAuthSource._refresh_token

    def _wrapped(rt: str) -> Any:
        return original_refresh(source, rt, transport=httpx.MockTransport(handler))

    source._refresh_token = _wrapped  # type: ignore[method-assign]

    assert source.resolve("Auth/codex") == new_access
    assert captured["json"] == {
        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
        "grant_type": "refresh_token",
        "refresh_token": "rt_old",
    }

    updated = json.loads(auth_path.read_text())
    assert updated["tokens"]["access_token"] == new_access
    assert updated["tokens"]["refresh_token"] == rotated_grant
    assert updated["tokens"]["id_token"] == new_id
    assert updated["tokens"]["account_id"] == "acct_new"
    assert isinstance(updated["last_refresh"], str)


def test_missing_codex_auth_file_returns_none_and_no_extra_headers(tmp_path: Path) -> None:
    source = CodexAuthSource(file_path=str(tmp_path / "missing-auth.json"), endpoint=_TEST_ENDPOINT)

    assert source.resolve("Auth/codex") is None
    assert source.extra_headers("Auth/codex") == {}


@pytest.mark.parametrize(
    "body",
    [
        "{",
        "[]",
    ],
)
def test_invalid_codex_auth_file_returns_none(tmp_path: Path, body: str) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(body)
    source = CodexAuthSource(file_path=str(auth_path), endpoint=_TEST_ENDPOINT)

    assert source.resolve("Auth/codex") is None


def test_expired_codex_access_token_without_refresh_grant_returns_none(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _jwt(_claims(exp_delta_seconds=-60, account_id="acct_old")),
                }
            }
        )
    )
    source = CodexAuthSource(file_path=str(auth_path), endpoint=_TEST_ENDPOINT)

    assert source.resolve("Auth/codex") is None


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(401, json={"error": "invalid_grant"}), None),
        (httpx.Response(200, text="not json"), None),
        (httpx.Response(200, json={"refresh_token": "rt_new"}), None),
        (httpx.Response(200, json={"access_token": "fresh"}), {"access_token": "fresh"}),
    ],
)
def test_codex_refresh_token_returns_payload_or_none(response: httpx.Response, expected: dict[str, Any] | None) -> None:
    source = CodexAuthSource(file_path="/dev/null", endpoint=_TEST_ENDPOINT)
    payload = source._refresh_token("refresh-grant", transport=httpx.MockTransport(lambda _request: response))

    assert payload == expected


def test_codex_refresh_token_network_error_returns_none() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    source = CodexAuthSource(file_path="/dev/null", endpoint=_TEST_ENDPOINT)

    assert source._refresh_token("refresh-grant", transport=httpx.MockTransport(handler)) is None


def test_codex_extra_headers_derive_account_from_jwt_without_account_field(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _jwt(_claims(account_id="acct_access", fedramp=True)),
                    "refresh_token": "rt_cached",
                }
            }
        )
    )
    source = CodexAuthSource(file_path=str(auth_path), endpoint=_TEST_ENDPOINT)

    assert source.extra_headers("Auth/codex") == {
        "ChatGPT-Account-ID": "acct_access",
        "X-OpenAI-Fedramp": "true",
    }
