"""Header-builder tests for ``ccproxy.openai_conversations.profile``.

Locks in the canonical authenticated-request identity: every chatgpt.com
``/backend-api`` call (main turn, sentinel, conduit prepare, AND the image
side-trips) must present the same full browser shape + Bearer via
:func:`get_api_headers`, so a side-trip can never silently regress to a thin
2-header request (which the WAF 403s).
"""

from __future__ import annotations

from ccproxy.openai_conversations.profile import get_api_headers, get_browser_headers


def test_get_api_headers_is_full_browser_shape_plus_bearer() -> None:
    h = get_api_headers(access_token="tok123", device_id="dev-1", conversation_id="conv-9")  # noqa: S106
    assert h["user-agent"].startswith("Mozilla/5.0")
    assert h["oai-language"] == "en-US"
    assert h["oai-client-version"]
    assert h["origin"] == "https://chatgpt.com"
    assert h["oai-device-id"] == "dev-1"
    assert h["oai-session-id"] == "dev-1"  # session_id defaults to device_id
    assert h["referer"] == "https://chatgpt.com/c/conv-9"
    assert h["authorization"] == "Bearer tok123"


def test_get_api_headers_omits_bearer_when_token_empty() -> None:
    h = get_api_headers(access_token="", device_id="dev-1")
    assert "authorization" not in h
    assert h["user-agent"].startswith("Mozilla/5.0")  # still the full browser shape


def test_get_api_headers_is_browser_headers_plus_bearer() -> None:
    browser = get_browser_headers(device_id="d", session_id="d")
    api = get_api_headers(access_token="t", device_id="d")  # noqa: S106
    for key in browser:
        assert key in api  # api is a superset of the browser identity
    assert api["authorization"] == "Bearer t"
