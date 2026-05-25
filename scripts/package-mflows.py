#!/usr/bin/env python3
"""Package a captured ``.mflow`` into a bundled template shape.

Bundled shapes ship in ``src/ccproxy/templates/shapes/`` as the working
baseline ccproxy uses out of the box. They MUST NOT carry capturer
identity (UUIDs, account IDs, device IDs, private session content) and
MUST NOT carry capture-time bookkeeping (correlation headers, internal
ccproxy flow metadata).

Users who want full impersonation stealth capture their own shape via
``ccproxy flows shape <provider>`` — personal captures land in
``~/.config/ccproxy/shapes/`` and retain everything observed. This
script is the one-way distillation: ``capture → personal use``;
``package → public shipping``.

Two run modes:

- ``package``::

      python scripts/package-mflows.py SRC.mflow --out DST.mflow

  Reads ``SRC``, applies the bundled-shape scrubber, writes ``DST``.

- ``verify`` (pre-commit gate)::

      python scripts/package-mflows.py --verify [PATH ...]

  Each ``PATH`` may be a file or directory. Without arguments, defaults
  to ``src/ccproxy/templates/shapes``. Every ``.mflow`` discovered is
  re-checked against the scrubber's expectations; any leftover identity
  artifact prints a violation list and exits non-zero.

Scrubber policy (applied to bundled output; personal captures untouched):

- **Request headers** dropped: ``X-Claude-Code-Session-Id``,
  ``x-client-request-id``, plus the ccproxy-internal correlation
  headers (``x-ccproxy-flow-id``, ``x-ccproxy-hooks``,
  ``x-ccproxy-oauth-injected``). Sidecar transport headers
  (``x-ccproxy-target-url``, ``x-ccproxy-impersonate``) are intentionally
  preserved — they're consumed on the loopback and stripped by the
  sidecar before reaching upstream.

- **Request body**:

  - ``metadata.user_id`` → all-zero UUID triple placeholder.
  - ``diagnostics.previous_message_id`` → ``None``.
  - ``messages`` → ``[]``. The apply-time ``content_fields`` injection
    rewrites this from the live request on every call; persisting the
    capturer's prompts is dead weight plus a private-content leak risk.
  - ``tools`` → ``[]``. Same logic — apply-time rewrite.
  - ``system`` → first 2 entries only. The
    ``merge_strategies.system = "prepend_shape:2"`` policy at apply
    time only consults the first 2 shape entries; the rest is dead
    weight.

- **Flow metadata**: every key dropped except
  ``ccproxy.fingerprint.profile`` (load-bearing for sidecar TLS replay).

- **Flow attributes**: ``response``, ``websocket``, ``error``,
  ``comment`` nulled.

What is intentionally **NOT** scrubbed: ``max_tokens``, ``stream``,
``thinking``, ``context_management``, ``model`` body fields;
``request.host``, ``request.path``, ``request.scheme``; any non-identity
request header (User-Agent, X-Stainless-*, anthropic-beta, anthropic-version,
content-type, accept, etc.); ``fingerprint.user_agent`` and
``fingerprint.runtime_version`` (CLI-version identifiers users need for
ccproxy to work).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mitmproxy import http
from mitmproxy.io import FlowReader, FlowWriter

ZERO_UUID = "00000000-0000-0000-0000-000000000000"

ZERO_USER_ID = json.dumps(
    {"account_uuid": ZERO_UUID, "device_id": ZERO_UUID, "session_id": ZERO_UUID},
)
"""Placeholder ``metadata.user_id`` value for bundled shapes."""

SCRUB_HEADERS = frozenset(
    {
        "x-claude-code-session-id",
        "x-client-request-id",
        "x-ccproxy-flow-id",
        "x-ccproxy-hooks",
        "x-ccproxy-oauth-injected",
    }
)
"""Explicit deny-list of headers stripped from bundled shapes.

The two ``x-claude-code-*`` / ``x-client-*`` headers are per-session/
per-request UUIDs set by Claude CLI — uniform across every replay would
be a correlation fingerprint, so they're dropped from the bundled.
The three ``x-ccproxy-*`` entries are our internal correlation IDs.

Notable exclusions: ``x-ccproxy-target-url`` and ``x-ccproxy-impersonate``
are kept — sidecar transport contract, stripped at the loopback hop by
the sidecar itself."""

SYSTEM_KEEP_COUNT = 2
"""Number of ``body.system`` entries to retain.

Matches ``shaping.providers.anthropic.merge_strategies.system =
``"prepend_shape:2"``: only the first two shape entries are consulted at
apply time. Everything past index 2 is dead weight on disk."""

PRESERVE_METADATA = frozenset({"ccproxy.fingerprint.profile"})
"""Flow-level metadata keys that survive scrubbing. Everything else dropped."""

DEFAULT_VERIFY_DIR = Path("src/ccproxy/templates/shapes")
"""Default directory walked by ``--verify`` when no PATH given."""


def _scrub_body(body: dict[str, Any]) -> dict[str, Any]:
    """Apply bundled-template policy to a parsed request body in-place."""
    md = body.get("metadata")
    if isinstance(md, dict) and "user_id" in md:
        md["user_id"] = ZERO_USER_ID

    diag = body.get("diagnostics")
    if isinstance(diag, dict) and "previous_message_id" in diag:
        diag["previous_message_id"] = None

    if "messages" in body:
        body["messages"] = []
    if "tools" in body:
        body["tools"] = []

    system = body.get("system")
    if isinstance(system, list) and len(system) > SYSTEM_KEEP_COUNT:
        body["system"] = system[:SYSTEM_KEEP_COUNT]

    return body


def _scrub_flow(flow: http.HTTPFlow) -> http.HTTPFlow:
    """Apply bundled-template policy to ``flow`` in-place."""
    for header_name in list(flow.request.headers.keys()):
        if header_name.lower() in SCRUB_HEADERS:
            del flow.request.headers[header_name]

    raw = flow.request.content or b""
    if raw:
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            body = None
        if isinstance(body, dict):
            flow.request.content = json.dumps(_scrub_body(body)).encode()

    metadata = dict(flow.metadata) if flow.metadata else {}
    flow.metadata = {k: v for k, v in metadata.items() if k in PRESERVE_METADATA}

    flow.response = None
    flow.websocket = None
    flow.error = None
    flow.comment = ""
    return flow


def _read_flows(path: Path) -> list[http.HTTPFlow]:
    with path.open("rb") as fo:
        return [f for f in FlowReader(fo).stream() if isinstance(f, http.HTTPFlow)]


def _verify_flow(flow: http.HTTPFlow) -> list[str]:
    """Return list of bundled-shape policy violations. Empty list means clean."""
    violations: list[str] = []

    for header_name in flow.request.headers:
        if header_name.lower() in SCRUB_HEADERS:
            violations.append(f"request header {header_name!r} present (should be stripped)")

    raw = flow.request.content or b""
    if raw:
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            body = None
        if isinstance(body, dict):
            md = body.get("metadata")
            if isinstance(md, dict):
                uid = md.get("user_id")
                if isinstance(uid, str) and uid != ZERO_USER_ID:
                    violations.append("metadata.user_id is not the zero-UUID placeholder")
            diag = body.get("diagnostics")
            if isinstance(diag, dict) and diag.get("previous_message_id") is not None:
                violations.append(f"diagnostics.previous_message_id = {diag['previous_message_id']!r}")
            if isinstance(body.get("messages"), list) and len(body["messages"]) > 0:
                violations.append(f"messages has {len(body['messages'])} entries (should be [])")
            if isinstance(body.get("tools"), list) and len(body["tools"]) > 0:
                violations.append(f"tools has {len(body['tools'])} entries (should be [])")
            system = body.get("system")
            if isinstance(system, list) and len(system) > SYSTEM_KEEP_COUNT:
                violations.append(f"system has {len(system)} entries (should be ≤ {SYSTEM_KEEP_COUNT})")

    for key in flow.metadata or {}:
        if key not in PRESERVE_METADATA:
            violations.append(f"flow metadata key {key!r} should be dropped")

    return violations


def package(src: Path, dst: Path) -> None:
    """Read ``src``, scrub, write to ``dst``."""
    flows = _read_flows(src)
    if not flows:
        raise SystemExit(f"no HTTPFlow in {src}")
    if len(flows) > 1:
        print(f"note: {src} contains {len(flows)} flows; using the last one", file=sys.stderr)
    flow = _scrub_flow(flows[-1])
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as fo:
        FlowWriter(fo).add(flow)
    print(f"packaged {src} -> {dst} ({dst.stat().st_size} bytes)")


def _iter_mflow_paths(targets: Iterable[Path]) -> Iterable[Path]:
    for target in targets:
        if target.is_dir():
            yield from sorted(target.rglob("*.mflow"))
        elif target.is_file():
            yield target


def verify(targets: list[Path]) -> int:
    """Verify every ``.mflow`` under ``targets``. Return count of failing flows."""
    paths = list(_iter_mflow_paths(targets))
    if not paths:
        print("no .mflow files to verify", file=sys.stderr)
        return 0

    fail = 0
    for path in paths:
        flows = _read_flows(path)
        if not flows:
            print(f"{path}: ERROR no HTTPFlow inside", file=sys.stderr)
            fail += 1
            continue
        for i, flow in enumerate(flows):
            violations = _verify_flow(flow)
            if violations:
                fail += 1
                print(f"{path}: FAIL flow[{i}]", file=sys.stderr)
                for v in violations:
                    print(f"  - {v}", file=sys.stderr)
            else:
                print(f"{path}: ok")
    return fail


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package or verify ccproxy bundled-shape .mflow files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("source", nargs="?", type=Path, help="source .mflow (for package mode)")
    parser.add_argument("--out", type=Path, help="destination .mflow (for package mode)")
    parser.add_argument(
        "--verify",
        nargs="*",
        type=Path,
        metavar="PATH",
        help=(
            "verify mode: each PATH may be a file or directory. "
            f"Defaults to {DEFAULT_VERIFY_DIR}/ when no PATH given."
        ),
    )
    args = parser.parse_args()

    if args.verify is not None:
        targets = args.verify or [DEFAULT_VERIFY_DIR]
        fails = verify(targets)
        if fails:
            raise SystemExit(f"{fails} flow(s) failed bundled-shape verification")
        return

    if args.source is None or args.out is None:
        parser.error("package mode requires both SRC and --out")
    package(args.source, args.out)


if __name__ == "__main__":
    main()
