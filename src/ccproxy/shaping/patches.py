"""Patch-series support for request shapes.

Shape patches use a quilt-style provider directory:

```
{shapes_dir}/{provider}/
├── series
└── 0001-example.patch
```

Each patch is a standard unified diff against the virtual file
``shape.json``. The default strip level is ``-p1``, so patches generated
with ``a/shape.json`` / ``b/shape.json`` paths apply directly.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any

from mitmproxy import http

logger = logging.getLogger(__name__)

PATCH_TARGET = "shape.json"
DEFAULT_STRIP_LEVEL = 1
_HUNK_RE = re.compile(r"@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@")


class ShapePatchError(RuntimeError):
    """Raised when a shape patch series cannot be loaded or applied."""


@dataclass(frozen=True)
class ShapePatch:
    """One patch file from a provider's ``series`` file."""

    path: Path
    strip: int = DEFAULT_STRIP_LEVEL


@dataclass(frozen=True)
class ShapePatchWriteResult:
    """Result of generating a provider patch against ``shape.json``."""

    path: Path
    changed: bool


@dataclass(frozen=True)
class _Hunk:
    old_start: int
    lines: list[str]


def apply_shape_patch_series(flow: http.HTTPFlow, provider: str, shapes_dir: Path | None) -> bool:
    """Apply the provider's patch series to ``flow.request``.

    Returns ``True`` when at least one patch was applied. Missing patch
    directories or missing ``series`` files are a no-op.
    """
    if shapes_dir is None or flow.request is None:
        return False

    provider_dir = shapes_dir / provider
    series_path = provider_dir / "series"
    if not series_path.exists():
        return False

    patches = _read_series(series_path)
    if not patches:
        return False

    text = _request_to_patch_text(flow.request)
    for patch in patches:
        text = _apply_unified_patch(
            text,
            patch.path.read_text(),
            strip=patch.strip,
            patch_name=str(patch.path),
        )
    _patch_text_to_request(flow.request, text)
    logger.info("Applied %d shape patch(es) for provider %s from %s", len(patches), provider, provider_dir)
    return bool(patches)


def write_shape_patch(
    base_request: http.Request,
    target_request: http.Request,
    provider_dir: Path,
    *,
    patch_name: str = "0001-local-shape.patch",
) -> ShapePatchWriteResult:
    """Write a standard unified diff from ``base_request`` to ``target_request``."""
    before = _request_to_patch_text(base_request)
    after = _request_to_patch_text(target_request)
    patch_path = provider_dir / patch_name

    if before == after:
        return ShapePatchWriteResult(path=patch_path, changed=False)

    provider_dir.mkdir(parents=True, exist_ok=True)
    patch = "\n".join(
        unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{PATCH_TARGET}",
            tofile=f"b/{PATCH_TARGET}",
            lineterm="",
        )
    )
    patch_path.write_text(patch + "\n")
    _ensure_series_entry(provider_dir / "series", patch_name)
    return ShapePatchWriteResult(path=patch_path, changed=True)


def _ensure_series_entry(series_path: Path, patch_name: str) -> None:
    if not series_path.exists():
        series_path.write_text(f"{patch_name}\n")
        return

    lines = series_path.read_text().splitlines()
    for raw_line in lines:
        tokens = shlex.split(raw_line, comments=True)
        if patch_name in tokens:
            return
    suffix = "" if not lines or lines[-1] == "" else "\n"
    series_path.write_text("\n".join(lines) + f"{suffix}{patch_name}\n")


def _read_series(series_path: Path) -> list[ShapePatch]:
    patches: list[ShapePatch] = []
    provider_dir = series_path.parent
    base = provider_dir.resolve()

    for line_number, raw_line in enumerate(series_path.read_text().splitlines(), start=1):
        tokens = shlex.split(raw_line, comments=True)
        if not tokens:
            continue

        patch_name: str | None = None
        strip = DEFAULT_STRIP_LEVEL
        idx = 0
        while idx < len(tokens):
            item = tokens[idx]
            if item == "--":
                idx += 1
                continue
            if item == "-p":
                idx += 1
                if idx >= len(tokens):
                    raise ShapePatchError(f"{series_path}:{line_number}: missing strip level after -p")
                strip = _parse_strip(tokens[idx], series_path, line_number)
            elif item.startswith("-p") and len(item) > 2:
                strip = _parse_strip(item[2:], series_path, line_number)
            elif item.startswith("-"):
                raise ShapePatchError(f"{series_path}:{line_number}: unsupported patch option {item!r}")
            elif patch_name is None:
                patch_name = item
            else:
                raise ShapePatchError(f"{series_path}:{line_number}: unexpected token {item!r}")
            idx += 1

        if patch_name is None:
            raise ShapePatchError(f"{series_path}:{line_number}: missing patch filename")
        patch_path = _resolve_patch_path(provider_dir, base, patch_name, series_path, line_number)
        patches.append(ShapePatch(path=patch_path, strip=strip))

    return patches


def _parse_strip(raw: str, series_path: Path, line_number: int) -> int:
    try:
        strip = int(raw)
    except ValueError as exc:
        raise ShapePatchError(f"{series_path}:{line_number}: invalid strip level {raw!r}") from exc
    if strip < 0:
        raise ShapePatchError(f"{series_path}:{line_number}: strip level must be non-negative")
    return strip


def _resolve_patch_path(
    provider_dir: Path,
    base: Path,
    patch_name: str,
    series_path: Path,
    line_number: int,
) -> Path:
    patch_path = (provider_dir / patch_name).resolve()
    try:
        patch_path.relative_to(base)
    except ValueError as exc:
        raise ShapePatchError(f"{series_path}:{line_number}: patch path escapes provider directory") from exc
    if not patch_path.is_file():
        raise ShapePatchError(f"{series_path}:{line_number}: patch file not found: {patch_name}")
    return patch_path


def _request_to_patch_text(request: http.Request) -> str:
    body = _parse_json_body(request.content)
    doc = {
        "body": body,
        "headers": {str(name): str(value) for name, value in request.headers.items()},  # type: ignore[no-untyped-call]
        "method": request.method,
        "url": request.url,
    }
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def _patch_text_to_request(request: http.Request, text: str) -> None:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ShapePatchError(f"patched {PATCH_TARGET} is not valid JSON: {exc}") from exc

    if not isinstance(doc, dict):
        raise ShapePatchError(f"patched {PATCH_TARGET} must be a JSON object")

    method = doc.get("method")
    url = doc.get("url")
    headers = doc.get("headers")
    body = doc.get("body")

    if not isinstance(method, str) or not method:
        raise ShapePatchError(f"patched {PATCH_TARGET} has invalid method")
    if not isinstance(url, str) or not url:
        raise ShapePatchError(f"patched {PATCH_TARGET} has invalid url")
    if not isinstance(headers, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
        raise ShapePatchError(f"patched {PATCH_TARGET} has invalid headers")
    if not isinstance(body, dict):
        raise ShapePatchError(f"patched {PATCH_TARGET} body must be a JSON object")

    request.method = method
    request.url = url
    request.headers.clear()
    for name, value in headers.items():
        request.headers[name] = value
    request.content = json.dumps(body).encode()


def _parse_json_body(content: bytes | None) -> dict[str, Any]:
    try:
        data = json.loads(content or b"{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _apply_unified_patch(source: str, patch_text: str, *, strip: int, patch_name: str) -> str:
    source_lines = source.splitlines()
    patch_lines = patch_text.splitlines()
    changed = False
    idx = 0

    while idx < len(patch_lines):
        if not patch_lines[idx].startswith("--- "):
            idx += 1
            continue

        old_path = _patch_header_path(patch_lines[idx])
        idx += 1
        if idx >= len(patch_lines) or not patch_lines[idx].startswith("+++ "):
            raise ShapePatchError(f"{patch_name}: missing +++ header after --- header")
        new_path = _patch_header_path(patch_lines[idx])
        idx += 1

        target = _strip_patch_path(new_path if new_path != "/dev/null" else old_path, strip)
        if target != PATCH_TARGET:
            raise ShapePatchError(f"{patch_name}: unsupported patch target {target!r}; expected {PATCH_TARGET!r}")

        hunks: list[_Hunk] = []
        while idx < len(patch_lines):
            line = patch_lines[idx]
            if line.startswith("--- "):
                break
            if line.startswith("diff --git "):
                idx += 1
                break
            if not line.startswith("@@ "):
                idx += 1
                continue

            match = _HUNK_RE.match(line)
            if match is None:
                raise ShapePatchError(f"{patch_name}: malformed hunk header: {line}")
            old_start = int(match.group("old_start"))
            idx += 1

            hunk_lines: list[str] = []
            while idx < len(patch_lines):
                hunk_line = patch_lines[idx]
                if hunk_line.startswith(("@@ ", "--- ", "diff --git ")):
                    break
                if hunk_line.startswith((" ", "-", "+", "\\")):
                    hunk_lines.append(hunk_line)
                    idx += 1
                    continue
                raise ShapePatchError(f"{patch_name}: malformed hunk line: {hunk_line!r}")
            hunks.append(_Hunk(old_start=old_start, lines=hunk_lines))

        source_lines = _apply_hunks(source_lines, hunks, patch_name)
        changed = True

    if not changed:
        raise ShapePatchError(f"{patch_name}: no patch for {PATCH_TARGET}")
    return "\n".join(source_lines) + "\n"


def _patch_header_path(line: str) -> str:
    path = line[4:].strip()
    return path.split("\t", 1)[0].split(" ", 1)[0]


def _strip_patch_path(path: str, strip: int) -> str:
    if path == "/dev/null":
        return path
    parts = [part for part in path.split("/") if part and part != "."]
    if strip > len(parts):
        return ""
    return "/".join(parts[strip:])


def _apply_hunks(source_lines: list[str], hunks: list[_Hunk], patch_name: str) -> list[str]:
    output: list[str] = []
    source_index = 0

    for hunk in hunks:
        target_index = max(hunk.old_start - 1, 0)
        if target_index < source_index or target_index > len(source_lines):
            raise ShapePatchError(f"{patch_name}: hunk location is out of range")

        output.extend(source_lines[source_index:target_index])
        source_index = target_index

        for line in hunk.lines:
            prefix = line[:1]
            content = line[1:]
            if prefix == "\\":
                continue
            if prefix in {" ", "-"}:
                if source_index >= len(source_lines) or source_lines[source_index] != content:
                    raise ShapePatchError(f"{patch_name}: hunk context does not match")
                if prefix == " ":
                    output.append(source_lines[source_index])
                source_index += 1
            elif prefix == "+":
                output.append(content)
            else:
                raise ShapePatchError(f"{patch_name}: malformed hunk line: {line!r}")

    output.extend(source_lines[source_index:])
    return output
