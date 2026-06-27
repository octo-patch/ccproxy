#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["curl-cffi>=0.15.0"]
# ///
"""Acquire OpenAI Conversations credential state for the openai_conversations provider.

The script uses gateau as the browser-cookie source, then performs the
ChatGPT session request with curl-cffi browser impersonation. Plain curl can
read the cookies, but current chatgpt.com backend endpoints are Cloudflare
gated and require a browser-like TLS/HTTP fingerprint.

The bearer token is never printed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from typing import Any

from curl_cffi import requests

CHATGPT_ORIGIN = "https://chatgpt.com"
SESSION_ENDPOINT = f"{CHATGPT_ORIGIN}/api/auth/session"
VERIFY_ENDPOINT = f"{CHATGPT_ORIGIN}/backend-api/me"
DEFAULT_COOKIE_HOSTS = ("chatgpt.com", "auth.openai.com", "openai.com")
DEFAULT_BROWSER = "firefox"
DEFAULT_IMPERSONATE = "chrome136"
DEFAULT_PERSONA = "chatgpt-paid"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
OBSOLETE_SENTINEL_FIELDS = (
    "sentinel_token",
    "sentinel_p_token",
    "sentinel_expires_at_ms",
    "sentinel_flow",
    "sentinel_so_token",
)


def config_dir() -> Path:
    if value := os.environ.get("CCPROXY_CONFIG_DIR"):
        return Path(value).expanduser()
    if value := os.environ.get("XDG_CONFIG_HOME"):
        return Path(value).expanduser() / "ccproxy"
    return Path.home() / ".config" / "ccproxy"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "-b",
        "--browser",
        choices=("firefox", "chrome", "chromium", "edge"),
        default=DEFAULT_BROWSER,
        help=f"Browser profile to read with gateau. Default: {DEFAULT_BROWSER}.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=config_dir() / "openai-conversations-credentials.json",
        help="Credential JSON path. Default: $CCPROXY_CONFIG_DIR/openai-conversations-credentials.json.",
    )
    parser.add_argument(
        "-r",
        "--root-path",
        type=Path,
        help="Non-standard browser profile root to pass to gateau.",
    )
    parser.add_argument(
        "--fresh-session",
        action="store_true",
        help="Spawn a temporary browser profile and wait for ChatGPT login before reading cookies.",
    )
    parser.add_argument(
        "--session-url",
        default=f"{CHATGPT_ORIGIN}/",
        help="URL opened for --fresh-session. Default: https://chatgpt.com/.",
    )
    parser.add_argument(
        "--cookie-host",
        action="append",
        dest="cookie_hosts",
        help=(
            "Cookie host passed to gateau output. Repeatable. Defaults to chatgpt.com, auth.openai.com, and openai.com."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=SESSION_ENDPOINT,
        help=f"Session endpoint. Default: {SESSION_ENDPOINT}.",
    )
    parser.add_argument(
        "--verify-endpoint",
        default=VERIFY_ENDPOINT,
        help=f"Endpoint used to verify the bearer token. Default: {VERIFY_ENDPOINT}.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the bearer-token verification request.",
    )
    parser.add_argument(
        "--impersonate",
        default=DEFAULT_IMPERSONATE,
        help=f"curl-cffi impersonation profile. Default: {DEFAULT_IMPERSONATE}.",
    )
    parser.add_argument(
        "--no-bypass-lock",
        action="store_true",
        help="Do not pass --bypass-lock to gateau.",
    )
    parser.add_argument(
        "--preserve-sentinel",
        action="store_true",
        help="Preserve existing current Sentinel fields even if access_token changes.",
    )
    parser.add_argument(
        "--persona",
        default=DEFAULT_PERSONA,
        help=f"Persona placeholder to write when none exists. Default: {DEFAULT_PERSONA}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Acquire, parse, and verify the token, but do not write the output file.",
    )
    return parser.parse_args(argv)


def run_gateau(args: argparse.Namespace, hosts: Iterable[str]) -> str:
    cmd = ["gateau"]
    if not args.no_bypass_lock:
        cmd.append("--bypass-lock")
    if args.root_path:
        cmd.extend(["--root-path", str(args.root_path.expanduser())])
    if args.fresh_session:
        cmd.extend(["--session", f"--session-urls={args.session_url}"])
    cmd.extend(["--browser", args.browser, "output", *hosts])

    if args.fresh_session:
        print(
            "gateau will open a temporary browser profile. Log in to ChatGPT, then close that browser window.",
            file=sys.stderr,
        )

    try:
        proc = subprocess.run(cmd, capture_output=True, check=False, text=True)  # noqa: S603
    except FileNotFoundError as exc:
        raise SystemExit("required command not found: gateau") from exc

    if proc.returncode == 0:
        return proc.stdout

    message = [
        "failed to read ChatGPT cookies through gateau.",
        "",
        proc.stderr.strip(),
        "",
        "If Firefox is open, keep the default lock bypass behavior.",
        "If the selected profile is not logged in, use --fresh-session.",
        "Chrome/Chromium users may need to unlock the Secret Service keyring.",
    ]
    raise SystemExit("\n".join(part for part in message if part))


def load_cookie_jar(netscape_cookies: str) -> MozillaCookieJar:
    with tempfile.NamedTemporaryFile("w", delete=False) as handle:
        handle.write(netscape_cookies)
        cookie_path = Path(handle.name)
    try:
        jar = MozillaCookieJar(str(cookie_path))
        jar.load(ignore_discard=True, ignore_expires=True)
        return jar
    finally:
        cookie_path.unlink(missing_ok=True)


def build_session(jar: MozillaCookieJar, impersonate: str) -> requests.Session:
    session = requests.Session(impersonate=impersonate)
    for cookie in jar:
        session.cookies.set(cookie.name, cookie.value, domain=cookie.domain, path=cookie.path)
    return session


def request_headers(access_token: str | None = None) -> dict[str, str]:
    headers = {
        "accept": "application/json",
        "origin": CHATGPT_ORIGIN,
        "referer": f"{CHATGPT_ORIGIN}/",
        "user-agent": USER_AGENT,
    }
    if access_token:
        headers["authorization"] = f"Bearer {access_token}"
    return headers


def fetch_json(session: requests.Session, url: str, *, access_token: str | None = None) -> Any:
    response = session.get(
        url,
        headers=request_headers(access_token),
        timeout=30,
        allow_redirects=True,
    )
    content_type = response.headers.get("content-type", "")
    if response.status_code != 200:
        raise RuntimeError(f"{url} returned HTTP {response.status_code}")
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        prefix = response.text[:120].replace("\n", " ")
        raise RuntimeError(f"{url} did not return JSON ({content_type}): {prefix}") from exc


def looks_like_jwt(value: str) -> bool:
    return value.count(".") >= 2 and len(value) > 80


def find_access_token(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("accessToken", "access_token"):
            candidate = value.get(key)
            if isinstance(candidate, str) and looks_like_jwt(candidate):
                return candidate
        for child in value.values():
            if found := find_access_token(child):
                return found
    elif isinstance(value, list):
        for child in value:
            if found := find_access_token(child):
                return found
    return None


def decode_expiry_ms(token: str) -> int | None:
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, json.JSONDecodeError):
        return None
    exp = claims.get("exp")
    if isinstance(exp, int | float):
        return int(exp * 1000)
    return None


def load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"existing credential file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SystemExit(f"existing credential file must contain a JSON object: {path}")
    return loaded


def atomic_write_json(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        tmp_path.replace(path)
        dir_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def build_state(args: argparse.Namespace, access_token: str) -> tuple[dict[str, Any], bool]:
    output_path = args.output.expanduser()
    existing = load_existing(output_path)
    old_token = existing.get("access_token")
    token_changed = isinstance(old_token, str) and old_token != "" and old_token != access_token
    keep_sentinel = args.preserve_sentinel or not token_changed

    state = dict(existing)
    state["access_token"] = access_token
    state["device_id"] = str(state.get("device_id") or uuid.uuid4())
    state["persona"] = str(state.get("persona") or args.persona)
    for field in OBSOLETE_SENTINEL_FIELDS:
        state.pop(field, None)

    if keep_sentinel:
        state.setdefault("chat_req_token", "")
        state.setdefault("proof_token", "")
        state.setdefault("chat_req_token_expires_at_ms", 0)
    else:
        state["chat_req_token"] = ""
        state["proof_token"] = ""
        state["chat_req_token_expires_at_ms"] = 0

    return state, token_changed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hosts = tuple(args.cookie_hosts or DEFAULT_COOKIE_HOSTS)
    netscape_cookies = run_gateau(args, hosts)
    jar = load_cookie_jar(netscape_cookies)
    if not jar:
        raise SystemExit("gateau returned no cookies for the requested hosts")

    session = build_session(jar, args.impersonate)
    try:
        session_payload = fetch_json(session, args.endpoint)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    access_token = find_access_token(session_payload)
    if not access_token:
        raise SystemExit(
            "ChatGPT session JSON did not contain accessToken. "
            "Retry with --fresh-session or confirm the selected Firefox profile is logged in."
        )

    if not args.no_verify:
        try:
            fetch_json(session, args.verify_endpoint, access_token=access_token)
        except RuntimeError as exc:
            raise SystemExit(f"acquired accessToken but verification failed: {exc}") from exc

    state, token_changed = build_state(args, access_token)
    if not args.dry_run:
        atomic_write_json(args.output.expanduser(), state)

    action = "validated" if args.dry_run else "wrote"
    print(f"{action} OpenAI Conversations credential state: {args.output.expanduser()}")
    print(f"access_token bytes: {len(access_token)}")
    if expiry_ms := decode_expiry_ms(access_token):
        expiry = datetime.fromtimestamp(expiry_ms / 1000, tz=UTC).isoformat()
        print(f"access_token expiry: {expiry}")
    if token_changed and not args.preserve_sentinel:
        print("current Sentinel fields reset because the access_token changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
