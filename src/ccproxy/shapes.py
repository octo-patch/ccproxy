"""Shape CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

import httpx
import tyro
from mitmproxy import http
from mitmproxy.io import FlowReader
from pydantic import BaseModel
from rich.console import Console

from ccproxy.flows import MitmwebClient, _FlowsBase, _make_client, _resolve_flow_set
from ccproxy.utils import get_templates_dir


class ShapeSave(_FlowsBase):
    """Generate a provider shape patch from the resolved flow set.

    By default, writes a quilt-style patch queue under
    ``$CCPROXY_CONFIG_DIR/shapes/{provider}/``. Use ``--mflow`` to write
    an explicit request-only ``{provider}.mflow`` override.

        ccproxy shapes save anthropic
        ccproxy shapes save anthropic --mflow
    """

    provider: Annotated[str, tyro.conf.Positional, tyro.conf.arg(metavar="PROVIDER")]
    """Target provider type (e.g., 'anthropic', 'gemini')."""

    mflow: bool = False
    """Write a sanitized request-only .mflow override instead of a patch."""


class ShapeAudit(BaseModel):
    """Audit packaged shape files for basic artifact invariants."""

    directory: Path | None = None
    """Directory containing .mflow files. Defaults to packaged templates/shapes."""


Shapes = Annotated[
    Annotated[ShapeSave, tyro.conf.subcommand(name="save")]
    | Annotated[ShapeAudit, tyro.conf.subcommand(name="audit")],
    tyro.conf.subcommand(
        name="shapes",
        description="Manage provider shape artifacts.",
    ),
]

_SENSITIVE_HEADERS = {
    "authorization",
    "cookie",
    "proxy-authorization",
    "x-api-key",
    "x-goog-api-key",
    "x-ccproxy-flow-id",
    "x-ccproxy-hooks",
    "x-ccproxy-auth-injected",
    "x-ccproxy-target-url",
    "x-ccproxy-impersonate",
}


def _do_shape_save(
    console: Console,
    client: MitmwebClient,
    flow_set: list[dict[str, Any]],
    *,
    provider: str,
    mflow: bool,
) -> None:
    """Save a shape artifact from the flow set."""
    if not flow_set:
        console.print("[red]No flows in set.[/red]")
        sys.exit(1)
    if not mflow and len(flow_set) != 1:
        console.print("[red]Patch shape generation requires exactly one flow in the set.[/red]")
        sys.exit(1)
    flow_ids = [f["id"] for f in flow_set]
    mode = "mflow" if mflow else "patch"
    result = client.save_shape(flow_ids, provider, mode=mode)
    if mode == "patch":
        status = str(result.get("status", "ok"))
        patch = result.get("patch")
        if status == "unchanged":
            console.print(f"Shape patch for [bold]{result['provider']}[/bold] is unchanged.")
            return
        console.print(f"Saved shape patch for [bold]{result['provider']}[/bold]: {patch}")
        return
    console.print(
        f"Saved .mflow shape for [bold]{result['provider']}[/bold]: "
        f"{result['flows_saved']} flow(s) saved"
        + (f", {len(result.get('missing', []))} missing" if result.get("missing") else "")
    )


def _do_shape_audit(console: Console, directory: Path | None) -> None:
    """Audit packaged shape files for readability and sensitive headers."""
    shape_dir = directory if directory is not None else get_templates_dir() / "shapes"
    if not shape_dir.exists():
        console.print(f"[red]Shape directory missing: {shape_dir}[/red]")
        sys.exit(1)

    count = 0
    failures: list[str] = []
    for path in sorted(shape_dir.glob("*.mflow")):
        count += 1
        try:
            flow = _read_latest(path)
        except Exception as exc:
            failures.append(f"{path.name}: unreadable ({exc})")
            continue
        if flow.response is not None:
            failures.append(f"{path.name}: response is present")
        for name in flow.request.headers:
            if name.lower() in _SENSITIVE_HEADERS:
                failures.append(f"{path.name}: sensitive header {name!r}")
    if failures:
        for failure in failures:
            console.print(f"[red]{failure}[/red]")
        sys.exit(1)
    console.print(f"Audited {count} shape file(s).")


def _read_latest(path: Path) -> http.HTTPFlow:
    flows: list[http.HTTPFlow] = []
    with path.open("rb") as fo:
        for flow in FlowReader(fo).stream():  # type: ignore[no-untyped-call]
            if isinstance(flow, http.HTTPFlow):
                flows.append(flow)
    if not flows:
        raise ValueError("empty mflow")
    return flows[-1]


def handle_shapes(cmd: ShapeSave | ShapeAudit, _config_dir: Path) -> None:
    """Dispatch shapes subcommands."""
    from ccproxy.config import get_config

    err = Console(stderr=True)
    if isinstance(cmd, ShapeAudit):
        _do_shape_audit(err, cmd.directory)
        return

    config = get_config()
    try:
        with _make_client() as client:
            flow_set = _resolve_flow_set(client, cmd, config.flows)
            _do_shape_save(err, client, flow_set, provider=cmd.provider, mflow=cmd.mflow)
    except httpx.ConnectError:
        err.print("[red]Cannot connect to mitmweb. Is ccproxy running?[/red]")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        err.print(f"[red]HTTP {e.response.status_code}: {e.response.text[:200]}[/red]")
        sys.exit(1)
    except ValueError as e:
        err.print(f"[red]{e}[/red]")
        sys.exit(1)
