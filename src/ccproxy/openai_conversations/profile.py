"""Browser identity constants for OpenAI Conversations (chatgpt.com) requests.

Ported from the MIT-licensed aurora reference:
  Copyright (c) 2026 aurora-develop
  https://github.com/aurora-develop/aurora  (MIT License)
  Source: internal/chatgpt/request.go:2715-2774 + util/useragent.go

These constants reflect 2026-06 network captures against Chrome 148 on Windows.
All header values must stay synchronized:
  - UA must match sec-ch-ua version numbers, and sec-ch-ua-platform must match.
  - Oai-Client-Version / Oai-Client-Build-Number come from the chatgpt.com build.
  - Oai-Echo-Logs / Oai-Telemetry go on the final /f/conversation only, not prepare.
"""

from __future__ import annotations

# Chrome 148 Windows — must stay in sync with sec-ch-ua-* below.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)

# Align conversation.txt 2026-06 captures: Chrome 148 Win64 English browser.
_BASE_HEADERS: dict[str, str] = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "oai-language": "en-US",
    "origin": "https://chatgpt.com",
    "priority": "u=1, i",
    "sec-ch-ua": '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
    "sec-ch-ua-arch": '"x86"',
    "sec-ch-ua-bitness": '"64"',
    "sec-ch-ua-full-version": '"148.0.7778.98"',
    "sec-ch-ua-full-version-list": (
        '"Chromium";v="148.0.7778.98", "Google Chrome";v="148.0.7778.98", "Not/A)Brand";v="99.0.0.0"'
    ),
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-model": '""',
    "sec-ch-ua-platform": '"Windows"',
    # Sec-Ch-Ua-Platform-Version reports "15.0.0" (Windows 11) — must match Windows NT 10.0
    # UA convention or Cloudflare flags the UA/platform-version cross-mismatch.
    "sec-ch-ua-platform-version": '"15.0.0"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": _USER_AGENT,
    # 2026-06 build from conversation.txt captures.
    "oai-client-version": "prod-497f333866796e100096ad083b51ca949d22e751",
    "oai-client-build-number": "7646290",
}

# Only on the final POST /f/conversation — not on prepare calls.
# Values from aurora request.go:580-581.
_FINAL_ONLY_HEADERS: dict[str, str] = {
    "oai-echo-logs": "0,3352,1,4100,0,6435,1,6501,0,6506,1,9918,0,11782,1,11804",
    "oai-telemetry": "[1,null]",
}

# Stale browser-identity / oai-* headers injected by downstream clients that must
# be cleared before stamping our own shape so the upstream sees one coherent form.
_HEADERS_TO_CLEAR: frozenset[str] = frozenset(
    {
        "user-agent",
        "sec-ch-ua",
        "sec-ch-ua-arch",
        "sec-ch-ua-bitness",
        "sec-ch-ua-full-version",
        "sec-ch-ua-full-version-list",
        "sec-ch-ua-mobile",
        "sec-ch-ua-model",
        "sec-ch-ua-platform",
        "sec-ch-ua-platform-version",
        "sec-fetch-dest",
        "sec-fetch-mode",
        "sec-fetch-site",
        "priority",
        "oai-language",
        "oai-client-version",
        "oai-client-build-number",
        "oai-device-id",
        "oai-session-id",
        "oai-echo-logs",
        "oai-telemetry",
        "openai-sentinel-token",
        "openai-sentinel-so-token",
        "x-oai-turn-trace-id",
        "x-openai-target-path",
        "x-openai-target-route",
        "x-conduit-token",
        "x-oai-is",
    }
)


def get_browser_headers(
    *,
    device_id: str,
    session_id: str,
    conversation_id: str = "",
    final: bool = False,
) -> dict[str, str]:
    """Return the browser identity header dict for a given request context.

    Args:
        device_id: Per-installation OAI-Device-Id UUID from credential state.
        session_id: Per-session OAI-Session-Id UUID (may be the same as device_id).
        conversation_id: ChatGPT conversation id for referer construction.
            When empty, Referer points to the chatgpt.com root.
        final: When ``True``, include ``Oai-Echo-Logs`` and ``Oai-Telemetry``
            (final ``/f/conversation`` only — not on prepare calls).

    Returns:
        Header dict (lowercase names) ready to merge onto a flow's headers.
    """
    headers = dict(_BASE_HEADERS)
    headers["oai-device-id"] = device_id
    headers["oai-session-id"] = session_id
    if conversation_id:
        headers["referer"] = f"https://chatgpt.com/c/{conversation_id}"
    else:
        headers["referer"] = "https://chatgpt.com/"
    if final:
        headers.update(_FINAL_ONLY_HEADERS)
    return headers


def headers_to_clear() -> frozenset[str]:
    """Header names (lowercase) that must be cleared before stamping browser headers."""
    return _HEADERS_TO_CLEAR
