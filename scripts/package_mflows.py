#!/usr/bin/env python3
"""Package built-in .mflow shapes from real captured provider traffic."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from mitmproxy import connection, http
from mitmproxy.io import FlowReader, FlowWriter

from ccproxy.config import clear_config_instance, get_config, get_config_dir
from ccproxy.flows import _make_client
from ccproxy.pipeline.context import Context
from ccproxy.shaping.apply import prepare_shape

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "src" / "ccproxy" / "templates" / "shapes"
DEFAULT_SOURCE_DIR = ROOT / ".ccproxy" / "package-mflows" / "source-shapes"
TEMPLATE_CONFIG = ROOT / "src" / "ccproxy" / "templates" / "ccproxy.yaml"


@dataclass(frozen=True)
class Capture:
    command: Callable[[], list[str]]
    selector: Callable[[dict[str, Any]], bool]
    inspect: bool = True


CAPTURES: dict[str, Capture] = {
    "anthropic": Capture(
        command=lambda: ["claude", "--model", "haiku", "-p", "Reply with exactly: packaged mflow ok"],
        selector=lambda flow: (
            _is_2xx(flow)
            and _request_host(flow) == "api.anthropic.com"
            and _request_path(flow).startswith("/v1/messages")
        ),
    ),
    "gemini": Capture(
        command=lambda: [
            "gemini",
            "-m",
            "gemini-3.1-pro-preview",
            "-p",
            "Reply with exactly: packaged mflow ok",
        ],
        selector=lambda flow: (
            _is_2xx(flow)
            and _request_host(flow) == "cloudcode-pa.googleapis.com"
            and _request_path(flow).startswith("/v1internal:")
        ),
    ),
    "openai_responses": Capture(
        command=lambda: [
            "codex",
            "exec",
            "--ephemeral",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "-C",
            tempfile.gettempdir(),
            "Reply with exactly: packaged mflow ok",
        ],
        selector=lambda flow: (
            _is_2xx(flow)
            and _request_host(flow) == "chatgpt.com"
            and _request_path(flow).startswith("/backend-api/codex/responses")
        ),
    ),
}

SENSITIVE_HEADERS = {
    "authorization",
    "cookie",
    "proxy-authorization",
    "x-api-key",
    "x-goog-api-key",
    "x-client-request-id",
    "session-id",
    "thread-id",
    "x-ccproxy-flow-id",
    "x-ccproxy-hooks",
    "x-ccproxy-auth-injected",
    "x-ccproxy-target-url",
    "x-ccproxy-impersonate",
    "chatgpt-account-id",
    "x-openai-fedramp",
    "x-codex-installation-id",
    "x-codex-turn-state",
    "x-codex-turn-metadata",
    "x-codex-parent-thread-id",
    "x-codex-window-id",
    "x-openai-memgen-request",
    "x-openai-subagent",
    "openai-organization",
    "openai-project",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir.resolve()
    source_dir = _source_dir(args.source_dir, args.skip_capture)

    with _package_config(source_dir, stop_process_compose=not args.skip_capture):
        if not args.skip_capture:
            _capture_all(args.providers)

        output_dir.mkdir(parents=True, exist_ok=True)
        for provider in args.providers:
            source = _read_latest(source_dir / f"{provider}.mflow")
            packaged = _package_flow(provider, source)
            _write_single(output_dir / f"{provider}.mflow", packaged)
            _audit_flow(provider, source, packaged)
            print(f"packaged {provider}: {output_dir / f'{provider}.mflow'}")

    print("package_mflows: ok")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-capture",
        action="store_true",
        help="Use existing source .mflow files instead of running the provider CLIs first.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory containing source {provider}.mflow files. Defaults to config.shaping.shapes_dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for packaged built-in .mflow files.",
    )
    parser.add_argument(
        "--provider",
        action="append",
        choices=sorted(CAPTURES),
        dest="providers",
        help="Provider key to capture/package. Repeatable. Defaults to every packaged provider.",
    )
    args = parser.parse_args(argv)
    args.providers = args.providers or list(CAPTURES)
    return args


def _source_dir(path: Path | None, skip_capture: bool) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    if not skip_capture:
        return DEFAULT_SOURCE_DIR.resolve()
    cfg = get_config()
    if cfg.shaping.shapes_dir:
        return Path(cfg.shaping.shapes_dir).expanduser().resolve()
    return (get_config_dir() / "shapes").resolve()


@contextmanager
def _package_config(source_dir: Path, *, stop_process_compose: bool):
    original_config_dir = os.environ.get("CCPROXY_CONFIG_DIR")
    with tempfile.TemporaryDirectory(prefix="ccproxy-package-mflows-") as tmp:
        config_dir = Path(tmp)
        _write_runtime_config(config_dir, source_dir)
        os.environ["CCPROXY_CONFIG_DIR"] = str(config_dir)
        clear_config_instance()
        try:
            yield
        finally:
            if stop_process_compose:
                _run(["process-compose", "down"], timeout=30, check=False)
            clear_config_instance()
            if original_config_dir is None:
                os.environ.pop("CCPROXY_CONFIG_DIR", None)
            else:
                os.environ["CCPROXY_CONFIG_DIR"] = original_config_dir


def _write_runtime_config(config_dir: Path, source_dir: Path) -> None:
    data = yaml.safe_load(TEMPLATE_CONFIG.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("ccproxy"), dict):
        raise ValueError(f"invalid template config: {TEMPLATE_CONFIG}")

    current_path = get_config_dir() / "ccproxy.yaml"
    if current_path.exists():
        current = yaml.safe_load(current_path.read_text())
        if isinstance(current, dict) and isinstance(current.get("ccproxy"), dict):
            data = current
            template = yaml.safe_load(TEMPLATE_CONFIG.read_text())
            data["ccproxy"]["hooks"] = template["ccproxy"]["hooks"]
            data["ccproxy"].setdefault("shaping", {})
            data["ccproxy"]["shaping"]["providers"] = template["ccproxy"]["shaping"]["providers"]

    ccproxy = data["ccproxy"]
    inspector = ccproxy.setdefault("inspector", {})
    inspector["cert_dir"] = str(config_dir)
    inspector["transforms"] = []
    ccproxy.setdefault("mcp", {}).setdefault("http", {})
    ccproxy.setdefault("shaping", {})["shapes_dir"] = str(source_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "ccproxy.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


def _capture_all(providers: list[str]) -> None:
    _run(["process-compose", "down"], timeout=30, check=False)
    _run(["process-compose", "up", "--detached"])
    _wait_for_proxy()
    for provider in providers:
        capture = CAPTURES[provider]
        _clear_flows()
        command = capture.command()
        if capture.inspect:
            _run(["uv", "run", "ccproxy", "run", "--inspect", "--", *command], timeout=240)
        else:
            _run(command, timeout=240)
        flow_id = _latest_matching_flow(capture.selector)
        with _make_client() as client:
            client.save_shape([flow_id], provider, mode="mflow")


def _wait_for_proxy() -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        proc = subprocess.run(["uv", "run", "ccproxy", "status", "--proxy"], check=False)  # noqa: S607
        if proc.returncode == 0:
            return
        time.sleep(2)
    raise RuntimeError("ccproxy did not become ready")


def _clear_flows() -> None:
    _run(["uv", "run", "ccproxy", "flows", "clear", "--all"])


def _latest_matching_flow(selector: Callable[[dict[str, Any]], bool]) -> str:
    with _make_client() as client:
        flows = [flow for flow in client.list_flows() if selector(flow)]
    if not flows:
        raise RuntimeError("no matching provider flow captured")
    return str(flows[-1]["id"])


def _run(command: list[str], *, timeout: int = 120, check: bool = True) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=check, timeout=timeout)  # noqa: S603


def _is_2xx(flow: dict[str, Any]) -> bool:
    response = flow.get("response") or {}
    status = response.get("status_code")
    return isinstance(status, int) and 200 <= status < 300


def _request_host(flow: dict[str, Any]) -> str:
    request = flow.get("request") or {}
    return str(request.get("pretty_host") or "")


def _request_path(flow: dict[str, Any]) -> str:
    request = flow.get("request") or {}
    return str(request.get("path") or "")


def _package_flow(provider: str, source: http.HTTPFlow) -> http.HTTPFlow:
    if source.request is None:
        raise ValueError(f"{provider} source shape has no request")
    profile = get_config().shaping.providers.get(provider)
    if profile is None:
        raise ValueError(f"no shaping profile configured for {provider}")

    working = http.Request.from_state(source.request.get_state())  # type: ignore[no-untyped-call]
    shape_ctx = Context.from_request(working)
    incoming_ctx = Context.from_request(_canonical_request(provider))
    prepare_shape(shape_ctx, incoming_ctx, profile)

    client_conn = connection.Client(peername=("127.0.0.1", 0), sockname=("127.0.0.1", 0))
    server_conn = connection.Server(address=(working.host, working.port))
    packaged = http.HTTPFlow(client_conn, server_conn)
    packaged.request = working
    packaged.comment = ""
    return packaged


def _canonical_request(provider: str) -> http.Request:
    if provider == "anthropic":
        body = {
            "model": "claude-haiku-4-5-20251001",
            "messages": [{"role": "user", "content": "Reply with exactly: packaged mflow ok"}],
            "max_tokens": 32,
            "stream": True,
        }
        return _json_request("https://api.anthropic.com/v1/messages", body)
    if provider == "gemini":
        body = {
            "model": "gemini-3.1-pro-preview",
            "request": {
                "session_id": str(uuid.uuid4()),
                "contents": [{"role": "user", "parts": [{"text": "Reply with exactly: packaged mflow ok"}]}],
                "generationConfig": {"maxOutputTokens": 32, "temperature": 0},
            },
        }
        return _json_request("https://cloudcode-pa.googleapis.com/v1internal:generateContent", body)
    if provider == "openai_responses":
        body = {
            "model": "gpt-5.5",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Reply with exactly: packaged mflow ok",
                        }
                    ],
                }
            ],
            "stream": True,
        }
        return _json_request("https://chatgpt.com/backend-api/codex/responses", body)
    raise ValueError(f"unsupported provider: {provider}")


def _json_request(url: str, body: dict[str, Any]) -> http.Request:
    return http.Request.make(
        "POST",
        url,
        json.dumps(body, separators=(",", ":")).encode(),
        {"content-type": "application/json", "user-agent": "ccproxy-package-mflows/1.0"},
    )


def _read_latest(path: Path) -> http.HTTPFlow:
    if not path.exists():
        raise FileNotFoundError(f"missing source shape: {path}")
    flows: list[http.HTTPFlow] = []
    with path.open("rb") as fo:
        for flow in FlowReader(fo).stream():  # type: ignore[no-untyped-call]
            if isinstance(flow, http.HTTPFlow):
                flows.append(flow)
    if not flows:
        raise ValueError(f"empty shape file: {path}")
    return flows[-1]


def _write_single(path: Path, flow: http.HTTPFlow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fo:
        FlowWriter(fo).add(flow)  # type: ignore[no-untyped-call]


def _audit_flow(provider: str, source: http.HTTPFlow, packaged: http.HTTPFlow) -> None:
    if packaged.response is not None:
        raise ValueError(f"{provider}: packaged flow has a response")
    if packaged.request is None:
        raise ValueError(f"{provider}: packaged flow has no request")

    for name in packaged.request.headers:
        if name.lower() in SENSITIVE_HEADERS:
            raise ValueError(f"{provider}: packaged flow kept sensitive header {name!r}")

    packaged_text = _serialized_search_text(packaged)
    packaged_text_lower = packaged_text.lower()
    for marker in _sensitive_state_markers():
        if marker in packaged_text_lower:
            raise ValueError(f"{provider}: packaged flow kept sensitive state marker {marker!r}")
    for value in _sensitive_source_values(provider, source):
        if value and value in packaged_text:
            raise ValueError(f"{provider}: source-sensitive value survived packaging")


def _serialized_search_text(flow: http.HTTPFlow) -> str:
    data = io.BytesIO()
    FlowWriter(data).add(flow)  # type: ignore[no-untyped-call]
    return data.getvalue().decode("utf-8", errors="replace")


def _sensitive_state_markers() -> set[str]:
    return {
        "ccproxy.record",
        "client_request",
        "provider_response",
        "authorization",
        "bearer ",
        "ya29.",
        "set-cookie",
        "cookie",
        "chatgpt-account-id",
        "x-openai-fedramp",
        "x-codex-installation-id",
        "x-codex-turn-state",
        "x-codex-turn-metadata",
        "x-codex-parent-thread-id",
        "x-codex-window-id",
        "openai-organization",
        "openai-project",
        "refresh_token",
        "access_token",
        "id_token",
        "account_id",
        "chatgpt_user_id",
        "cf_clearance",
    }


def _sensitive_source_values(provider: str, flow: http.HTTPFlow) -> set[str]:
    body = _body(flow)
    values: set[str] = set()
    for name, value in _all_headers(flow.get_state()):
        if name.lower() in SENSITIVE_HEADERS or value.startswith(("Bearer ", "ya29.")):
            values.add(value.removeprefix("Bearer "))
    if provider == "anthropic":
        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            _collect_strings(metadata, values)
        client_metadata = body.get("client_metadata")
        if isinstance(client_metadata, dict):
            _collect_strings(client_metadata, values)
        diagnostics = body.get("diagnostics")
        if isinstance(diagnostics, dict):
            _collect_strings(diagnostics, values)
    elif provider == "gemini":
        for value in (body.get("project"), body.get("user_prompt_id")):
            if isinstance(value, str):
                values.add(value)
        request = body.get("request")
        if isinstance(request, dict) and isinstance(request.get("session_id"), str):
            values.add(request["session_id"])
    elif provider == "openai_responses":
        for key in (
            "previous_response_id",
            "prompt_cache_key",
            "safety_identifier",
            "user",
        ):
            value = body.get(key)
            if isinstance(value, str):
                values.add(value)
        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            _collect_strings(metadata, values)
    return {value for value in values if len(value) >= 8}


def _all_headers(value: Any) -> list[tuple[str, str]]:
    headers: list[tuple[str, str]] = []
    if isinstance(value, dict):
        if all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            for key, item in value.items():
                headers.append((key, item))
        for item in value.values():
            headers.extend(_all_headers(item))
    elif isinstance(value, list):
        for item in value:
            headers.extend(_all_headers(item))
    return headers


def _body(flow: http.HTTPFlow) -> dict[str, Any]:
    if flow.request is None:
        return {}
    try:
        parsed = json.loads(flow.request.content or b"{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _collect_strings(value: Any, out: set[str]) -> None:
    if isinstance(value, str):
        out.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, out)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, out)


if __name__ == "__main__":
    sys.exit(main())
