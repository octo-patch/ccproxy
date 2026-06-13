"""Query mitmweb flows REST API for debugging LLM request pipelines.

All ``flows`` subcommands operate on a **set** of flows built by:

    GET /flows → config.flows.default_jq_filters → CLI --jq filters → final set

CLI subcommands:

    ccproxy flows list     [--json] [--jq FILTER]...
    ccproxy flows dump              [--jq FILTER]...
    ccproxy flows diff              [--jq FILTER]...
    ccproxy flows compare           [--jq FILTER]...
    ccproxy flows clear    [--all]  [--jq FILTER]...

HAR output from ``dump`` is built server-side by the ``ccproxy.dump`` mitmproxy
command (registered by ``MultiHARSaver`` in ``ccproxy.inspector.multi_har_saver``).
"""

from __future__ import annotations

import atexit
import code
import contextlib
import importlib
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, cast

import httpx
import humanize
import tyro
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table


class MitmwebClient:
    """Sync client for the mitmweb REST API."""

    def __init__(self, host: str, port: int, token: str) -> None:
        self._base = f"http://{host}:{port}"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(
            base_url=self._base,
            headers=headers,
            timeout=10.0,
        )
        self._xsrf: str | None = None

    def list_flows(self) -> list[dict[str, Any]]:
        resp = self._client.get("/flows")
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def get_request_body(self, flow_id: str) -> bytes:
        resp = self._client.get(f"/flows/{flow_id}/request/content.data")
        resp.raise_for_status()
        return resp.content

    def get_response_body(self, flow_id: str) -> bytes:
        """Fetch the response body for a flow as raw bytes."""
        resp = self._client.get(f"/flows/{flow_id}/response/content.data")
        resp.raise_for_status()
        return resp.content

    def dump_har(self, flow_ids: list[str]) -> str:
        """Invoke ``ccproxy.dump`` with one or more flow ids; returns HAR JSON string."""
        if not flow_ids:
            raise ValueError("dump_har: flow_ids must be non-empty")
        resp = self._post(
            "/commands/ccproxy.dump",
            json_body={"arguments": [",".join(flow_ids)]},
        )
        payload = resp.json()
        if "error" in payload:
            raise ValueError(payload["error"])
        return str(payload["value"])

    def delete_flow(self, flow_id: str) -> None:
        """DELETE /flows/{id} — remove a single flow from mitmweb."""
        import secrets as _secrets

        if not self._xsrf:
            self._xsrf = _secrets.token_hex(16)
        self._client.cookies.set("_xsrf", self._xsrf)
        resp = self._client.delete(
            f"/flows/{flow_id}",
            headers={"X-XSRFToken": self._xsrf},
        )
        resp.raise_for_status()

    def clear(self) -> None:
        self._post("/clear")

    def _post(
        self,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """POST with synthetic XSRF token pair (cookie + header), optional JSON body."""
        import secrets as _secrets

        if not self._xsrf:
            self._xsrf = _secrets.token_hex(16)
        self._client.cookies.set("_xsrf", self._xsrf)
        resp = self._client.post(
            path,
            headers={"X-XSRFToken": self._xsrf},
            json=json_body,
        )
        resp.raise_for_status()
        return resp

    def save_shape(self, flow_ids: list[str], provider: str, *, mode: str = "patch") -> dict[str, Any]:
        """Invoke ``ccproxy.shape`` with flow ids and provider; returns summary dict."""
        if not flow_ids:
            raise ValueError("save_shape: flow_ids must be non-empty")
        resp = self._post(
            "/commands/ccproxy.shape",
            json_body={"arguments": [",".join(flow_ids), provider, mode]},
        )
        payload = resp.json()
        if "error" in payload:
            raise ValueError(payload["error"])
        return json.loads(payload["value"])  # type: ignore[no-any-return]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MitmwebClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# --- CLI subcommand classes ---


class _FlowsBase(BaseModel):
    """Shared fields for every ``flows`` subcommand."""

    jq_filter: Annotated[list[str], tyro.conf.arg(name="jq")] = Field(
        default_factory=list,
    )
    """Repeatable jq filter expression. Each must consume and produce a JSON array."""


class FlowsList(_FlowsBase):
    """Tabular listing of the resolved flow set."""

    json_output: Annotated[bool, tyro.conf.arg(name="json")] = False
    """Emit raw JSON instead of a rendered table."""


class FlowsDump(_FlowsBase):
    """Dump the resolved flow set as a multi-page HAR 1.2 file.

    Output contains one page per flow (pageref = flow.id), each page
    containing two HAR entries:

      entries[2i]     [fwdreq, provider_response]  forwarded request + raw provider response
      entries[2i+1]   [clireq, client_response]   client request + post-transform response

    Pipe to a file and open in Chrome DevTools / Charles / Fiddler:

        ccproxy flows dump > all.har
        ccproxy flows dump --jq 'map(select(.id | startswith("abc")))' > one.har
    """


class FlowsDiff(_FlowsBase):
    """Sliding-window unified diff over the resolved flow set.

    For a set [f0, f1, f2, f3], emits 3 diffs: f0->f1, f1->f2, f2->f3.
    Narrow to exactly 2 flows for a classic pairwise diff.
    """


class FlowsCompare(_FlowsBase):
    """Per-flow client-request vs forwarded-request diff.

    For each flow in the set, shows what the ccproxy pipeline changed:
    diffs the pre-pipeline client request against the post-pipeline
    forwarded request.

    Supports 1+ flows. Each flow produces one diff panel.

        ccproxy flows compare
        ccproxy flows compare --jq 'map(select(.id | startswith("abc")))'
    """


class FlowsRepl(_FlowsBase):
    """Open an interactive Python REPL over the resolved flow set."""


class FlowsClear(_FlowsBase):
    """Clear the resolved flow set (or everything with --all)."""

    all: Annotated[bool, tyro.conf.arg(name="all")] = False
    """Bypass the filter pipeline and clear every flow."""


Flows = Annotated[
    Annotated[FlowsList, tyro.conf.subcommand(name="list")]
    | Annotated[FlowsDump, tyro.conf.subcommand(name="dump")]
    | Annotated[FlowsDiff, tyro.conf.subcommand(name="diff")]
    | Annotated[FlowsCompare, tyro.conf.subcommand(name="compare")]
    | Annotated[FlowsRepl, tyro.conf.subcommand(name="repl")]
    | Annotated[FlowsClear, tyro.conf.subcommand(name="clear")],
    tyro.conf.subcommand(
        name="flows",
        description="Inspect mitmweb flows. All commands operate on a set "
        "narrowed by --jq filters + config default_jq_filters.",
    ),
]


# --- Helpers ---


def _make_client() -> MitmwebClient:
    from ccproxy.config import get_config

    cfg = get_config()
    inspector = cfg.inspector
    host = inspector.mitmproxy.web_host
    port = inspector.port

    token = _resolve_web_password(inspector.mitmproxy.web_password)
    return MitmwebClient(host=host, port=port, token=token)


def _resolve_web_password(cfg: Any) -> str:
    if cfg is None:
        return ""
    if isinstance(cfg, str):
        return cfg
    return cfg.resolve("mitmweb web_password") or ""


def _header_value(headers: list[list[str]], name: str) -> str:
    """Extract a header value from the mitmweb headers array [[name, value], ...]."""
    for pair in headers:
        if pair[0].lower() == name.lower():
            return pair[1]
    return ""


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)


FlowRef = int | str | dict[str, Any]


# --- JQ filter pipeline ---


def _run_jq(
    flows: list[dict[str, Any]],
    filter_str: str,
) -> list[Any]:
    """Run a jq filter over a flows list. Filter must produce a JSON array."""
    proc = subprocess.run(  # noqa: S603
        ["jq", "-c", filter_str],  # noqa: S607
        input=json.dumps(flows).encode(),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(f"jq filter failed: {proc.stderr.decode().strip()}")
    try:
        output = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ValueError(f"jq output is not valid JSON: {e}") from e
    if not isinstance(output, list):
        raise ValueError(
            f"jq filter must produce a JSON array, got {type(output).__name__}",
        )
    return cast(list[Any], output)


def _resolve_flow_set(
    client: MitmwebClient,
    cmd: _FlowsBase,
    flows_cfg: Any,
) -> list[dict[str, Any]]:
    """Build the operating set: raw -> default filters -> CLI filters."""
    raw = client.list_flows()
    filters = [*flows_cfg.default_jq_filters, *cmd.jq_filter]
    if not filters:
        return raw
    return cast(list[dict[str, Any]], _run_jq(raw, " | ".join(filters)))


def _resolve_flow_ref(flow_set: list[dict[str, Any]], ref: FlowRef) -> dict[str, Any]:
    """Resolve an index, exact id, id prefix, or flow dict to a flow from the current set."""
    if isinstance(ref, dict):
        flow_id = ref.get("id")
        if isinstance(flow_id, str):
            for flow in flow_set:
                if flow.get("id") == flow_id:
                    return flow
        raise ValueError("flow dict is not in the current set")

    if isinstance(ref, int):
        try:
            return flow_set[ref]
        except IndexError as e:
            raise ValueError(f"flow index {ref} is out of range") from e

    matches = [flow for flow in flow_set if str(flow.get("id", "")).startswith(ref)]
    if not matches:
        raise ValueError(f"no flow matches {ref!r}")
    if len(matches) > 1:
        ids = ", ".join(str(flow["id"])[:8] for flow in matches[:5])
        raise ValueError(f"flow prefix {ref!r} is ambiguous: {ids}")
    return matches[0]


def _select_flows(
    flow_set: list[dict[str, Any]],
    refs: Sequence[FlowRef] | None,
) -> list[dict[str, Any]]:
    """Return selected flows, preserving set order when refs is None."""
    if refs is None:
        return list(flow_set)
    return [_resolve_flow_ref(flow_set, ref) for ref in refs]


class FlowReplSession:
    """Mutable REPL facade over a resolved mitmweb flow set."""

    def __init__(
        self,
        client: MitmwebClient,
        flow_set: list[dict[str, Any]],
        *,
        flows_cfg: Any | None = None,
        jq_filter: Sequence[str] | None = None,
    ) -> None:
        default_filters = getattr(flows_cfg, "default_jq_filters", []) if flows_cfg is not None else []
        self.client = client
        self.default_jq_filters = [str(filter_str) for filter_str in default_filters]
        self.jq_filter = [str(filter_str) for filter_str in (jq_filter or [])]
        self.flows: list[dict[str, Any]] = []
        self.ids: list[str] = []
        self._set_flows(flow_set)

    def __repr__(self) -> str:
        return f"FlowReplSession(flows={len(self.flows)})"

    def _set_flows(self, flow_set: list[dict[str, Any]]) -> None:
        self.flows[:] = flow_set
        self.ids[:] = [str(flow["id"]) for flow in flow_set]

    def _selected(self, refs: Sequence[FlowRef]) -> list[dict[str, Any]]:
        return _select_flows(self.flows, refs or None)

    def flow(self, ref: FlowRef = 0) -> dict[str, Any]:
        """Return a flow dict by index, exact id, id prefix, or existing flow dict."""
        return _resolve_flow_ref(self.flows, ref)

    def flow_id(self, ref: FlowRef = 0) -> str:
        """Return a full flow id from any accepted flow reference."""
        return str(self.flow(ref)["id"])

    def show(self, *, json_output: bool = False) -> None:
        """Render the current flow set with the same table used by ``flows list``."""
        _do_list(Console(), self.flows, json_output=json_output)

    def refresh(self) -> list[dict[str, Any]]:
        """Reload flows from mitmweb and reapply config + CLI filters."""
        flow_set = self.client.list_flows()
        for filter_str in [*self.default_jq_filters, *self.jq_filter]:
            flow_set = cast(list[dict[str, Any]], _run_jq(flow_set, filter_str))
        self._set_flows(list(flow_set))
        return self.flows

    def apply(self, filter_str: str) -> list[dict[str, Any]]:
        """Apply a jq array filter to the current in-memory flow set."""
        self._set_flows(cast(list[dict[str, Any]], _run_jq(self.flows, filter_str)))
        return self.flows

    def request(self, ref: FlowRef = 0, *, pretty: bool = True) -> str:
        """Return a flow's request body."""
        text = self.client.get_request_body(self.flow_id(ref)).decode("utf-8", errors="replace")
        return _format_body(text) if pretty else text

    def response(self, ref: FlowRef = 0, *, pretty: bool = True) -> str:
        """Return a flow's response body."""
        text = self.client.get_response_body(self.flow_id(ref)).decode("utf-8", errors="replace")
        return _format_body(text) if pretty else text

    def diff(self, left: FlowRef = 0, right: FlowRef = 1) -> None:
        """Diff request bodies for two flows."""
        left_id = self.flow_id(left)
        right_id = self.flow_id(right)
        _git_diff(
            self.request(left, pretty=True),
            self.request(right, pretty=True),
            f"flow:{left_id[:8]}",
            f"flow:{right_id[:8]}",
        )

    def compare(self, *refs: FlowRef) -> None:
        """Diff client-vs-forwarded request and provider-vs-client response for selected flows."""
        _do_compare(self.client, self._selected(refs))

    def dump(self, *refs: FlowRef, path: str | Path | None = None) -> str | Path:
        """Dump selected flows as HAR JSON, optionally writing it to ``path``."""
        flow_ids = [str(flow["id"]) for flow in self._selected(refs)]
        har = self.client.dump_har(flow_ids)
        if path is None:
            print(har)
            return har
        output_path = Path(path)
        output_path.write_text(har)
        return output_path

    def shape(self, provider: str, *refs: FlowRef, mflow: bool = False) -> dict[str, Any]:
        """Save selected flows as a provider shape and return the mitmproxy command summary."""
        flow_ids = [str(flow["id"]) for flow in self._selected(refs)]
        mode = "mflow" if mflow else "patch"
        return self.client.save_shape(flow_ids, provider, mode=mode)

    def clear(self, *refs: FlowRef) -> int:
        """Delete selected flows from mitmweb and refresh the current set."""
        selected = self._selected(refs)
        for flow in selected:
            self.client.delete_flow(str(flow["id"]))
        self.refresh()
        return len(selected)

    def save_request(self, ref: FlowRef = 0, path: str | Path | None = None) -> Path:
        """Write a pretty request body to disk."""
        flow_id = self.flow_id(ref)
        output_path = Path(path) if path is not None else Path(f"{flow_id[:8]}-request.json")
        output_path.write_text(self.request(ref, pretty=True))
        return output_path

    def save_response(self, ref: FlowRef = 0, path: str | Path | None = None) -> Path:
        """Write a pretty response body to disk."""
        flow_id = self.flow_id(ref)
        output_path = Path(path) if path is not None else Path(f"{flow_id[:8]}-response.json")
        output_path.write_text(self.response(ref, pretty=True))
        return output_path


# --- Per-command handlers ---


def _do_list(
    console: Console,
    flow_set: list[dict[str, Any]],
    *,
    json_output: bool = False,
) -> None:
    """Render a pre-resolved flow set as a table or JSON."""
    if json_output:
        for f in flow_set:
            ts = f["request"].get("timestamp_start")
            if ts:
                f["time"] = _dt(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(json.dumps(flow_set, indent=2))
        return

    if not flow_set:
        console.print("[dim]No flows.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", width=8)
    table.add_column("Method", width=7)
    table.add_column("Code", width=5, justify="right")
    table.add_column("Host", max_width=35)
    table.add_column("Path", max_width=60)
    table.add_column("UA", max_width=30)
    table.add_column("Time", width=12)

    for f in flow_set:
        req = f["request"]
        res = f.get("response") or {}
        code = str(res.get("status_code", "-"))
        code_style = "green" if code.startswith("2") else "red" if code != "-" else "dim"
        ua = _header_value(req.get("headers", []), "user-agent")
        ts = req.get("timestamp_start")
        rel_time = humanize.naturaltime(_dt(ts)) if ts else "-"

        table.add_row(
            f["id"][:8],
            req["method"],
            f"[{code_style}]{code}[/{code_style}]",
            req["pretty_host"],
            req["path"][:60],
            ua[:30] if ua else "[dim]-[/dim]",
            f"[dim]{rel_time}[/dim]",
        )

    console.print(table)


def _do_dump(client: MitmwebClient, flow_set: list[dict[str, Any]]) -> None:
    """Dump all flows in the set as a multi-page HAR."""
    if not flow_set:
        print("No flows in set.", file=sys.stderr)
        sys.exit(1)
    flow_ids = [f["id"] for f in flow_set]
    print(client.dump_har(flow_ids))


def _format_body(text: str | None) -> str:
    """Try to pretty-format a body string as JSON; fall back to raw."""
    if not text:
        return ""
    with contextlib.suppress(json.JSONDecodeError, ValueError):
        return json.dumps(json.loads(text), indent=2)
    return text


def _git_diff(text_a: str, text_b: str, label_a: str, label_b: str) -> None:
    """Diff two strings via git diff --no-index. Output goes directly to stdout."""
    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix=f"{label_a}_", delete=True) as fa,
        tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix=f"{label_b}_", delete=True) as fb,
    ):
        fa.write(text_a)
        fa.flush()
        fb.write(text_b)
        fb.flush()
        subprocess.run(  # noqa: S603
            [  # noqa: S607
                "git",
                "--no-pager",
                "diff",
                "--no-index",
                "--color=auto",
                f"--src-prefix={label_a}/",
                f"--dst-prefix={label_b}/",
                "--",
                fa.name,
                fb.name,
            ],
            check=False,
        )


def _do_diff(
    client: MitmwebClient,
    flow_set: list[dict[str, Any]],
) -> None:
    """Sliding-window diff over the set."""
    if len(flow_set) < 2:
        print(
            f"diff needs at least 2 flows in the set (got {len(flow_set)})",
            file=sys.stderr,
        )
        sys.exit(1)

    for i in range(len(flow_set) - 1):
        a, b = flow_set[i], flow_set[i + 1]
        id_a, id_b = a["id"], b["id"]

        body_a = client.get_request_body(id_a).decode("utf-8", errors="replace")
        body_b = client.get_request_body(id_b).decode("utf-8", errors="replace")

        body_a = _format_body(body_a) or body_a
        body_b = _format_body(body_b) or body_b

        if i > 0:
            print()

        _git_diff(body_a, body_b, f"flow:{id_a[:8]}", f"flow:{id_b[:8]}")


def _do_compare(
    client: MitmwebClient,
    flow_set: list[dict[str, Any]],
) -> None:
    """Per-flow client-request vs forwarded-request diff."""
    if not flow_set:
        print("No flows in set.", file=sys.stderr)
        sys.exit(1)

    flow_ids = [f["id"] for f in flow_set]
    har = json.loads(client.dump_har(flow_ids))
    entries = har["log"]["entries"]

    for i in range(0, len(entries), 2):
        fwd_entry = entries[i]
        cli_entry = entries[i + 1]
        flow_id = har["log"]["pages"][i // 2]["id"]

        fwd_url = fwd_entry["request"]["url"]
        cli_url = cli_entry["request"]["url"]
        fwd_body = _format_body(fwd_entry["request"].get("postData", {}).get("text"))
        cli_body = _format_body(cli_entry["request"].get("postData", {}).get("text"))

        if i > 0:
            print()

        if cli_url != fwd_url:
            print(f"--- URL change: {flow_id[:8]} ---")
            print(f"- {cli_url}")
            print(f"+ {fwd_url}")

        _git_diff(cli_body, fwd_body, f"client:{flow_id[:8]}", f"forwarded:{flow_id[:8]}")

        fwd_response = _format_body(fwd_entry["response"].get("content", {}).get("text"))
        cli_response = _format_body(cli_entry["response"].get("content", {}).get("text"))
        _git_diff(fwd_response, cli_response, f"provider:{flow_id[:8]}", f"client:{flow_id[:8]}")


def _do_clear(
    console: Console,
    client: MitmwebClient,
    flow_set: list[dict[str, Any]],
    *,
    clear_all: bool,
) -> None:
    """Clear the set (or everything if --all)."""
    if clear_all:
        client.clear()
        console.print("All flows cleared.")
        return
    if not flow_set:
        console.print("No flows in set.")
        return
    for flow in flow_set:
        client.delete_flow(flow["id"])
    console.print(f"Cleared {len(flow_set)} flow(s).")


def _repl_namespace(session: FlowReplSession) -> dict[str, Any]:
    """Build the user namespace for ``flows repl``."""
    return {
        "session": session,
        "client": session.client,
        "flows": session.flows,
        "ids": session.ids,
        "show": session.show,
        "jq": session.apply,
        "refresh": session.refresh,
        "reload": session.refresh,
        "flow": session.flow,
        "flow_id": session.flow_id,
        "request": session.request,
        "response": session.response,
        "diff": session.diff,
        "compare": session.compare,
        "dump": session.dump,
        "shape": session.shape,
        "clear": session.clear,
        "save_request": session.save_request,
        "save_response": session.save_response,
    }


def _repl_banner(session: FlowReplSession) -> str:
    helper_names = (
        "show",
        "jq",
        "refresh",
        "flow",
        "request",
        "response",
        "diff",
        "compare",
        "dump",
        "shape",
        "clear",
        "save_request",
        "save_response",
    )
    helpers = ", ".join(helper_names)
    return (
        f"ccproxy flows repl: {len(session.flows)} flow(s) loaded\n"
        f"session, client, flows, ids, and helpers are available: {helpers}\n"
        "Examples: show(); request(0); diff(0, 1); jq('map(select(.response.status_code == 500))')"
    )


def _install_repl_history(history_path: Path) -> None:
    with contextlib.suppress(ImportError):
        import readline

        history_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            readline.read_history_file(str(history_path))
        atexit.register(readline.write_history_file, str(history_path))


def _embed_repl(namespace: dict[str, Any], banner: str) -> None:
    """Launch IPython when present, falling back to the stdlib interactive console."""
    _install_repl_history(Path.home() / ".ccproxy-flows-repl-history")
    with contextlib.suppress(ImportError):
        ipython = importlib.import_module("IPython")
        embed = getattr(ipython, "embed", None)
        if callable(embed):
            cast(Callable[..., None], embed)(user_ns=namespace, banner1=banner)
            return

    console = code.InteractiveConsole(locals=namespace)
    console.interact(banner=banner, exitmsg="")


def _do_repl(
    client: MitmwebClient,
    flow_set: list[dict[str, Any]],
    *,
    flows_cfg: Any,
    jq_filter: Sequence[str],
) -> None:
    """Start the interactive flows REPL."""
    session = FlowReplSession(client, flow_set, flows_cfg=flows_cfg, jq_filter=jq_filter)
    _embed_repl(_repl_namespace(session), _repl_banner(session))


# --- Dispatch ---


def handle_flows(
    cmd: FlowsList | FlowsDump | FlowsDiff | FlowsCompare | FlowsRepl | FlowsClear,
    _config_dir: Path,
) -> None:
    """Dispatch flows subcommand actions by isinstance."""
    from ccproxy.config import get_config

    err = Console(stderr=True)
    config = get_config()
    try:
        with _make_client() as client:
            flow_set = _resolve_flow_set(client, cmd, config.flows)
            if isinstance(cmd, FlowsList):
                _do_list(Console(), flow_set, json_output=cmd.json_output)
            elif isinstance(cmd, FlowsDump):
                _do_dump(client, flow_set)
            elif isinstance(cmd, FlowsDiff):
                _do_diff(client, flow_set)
            elif isinstance(cmd, FlowsCompare):
                _do_compare(client, flow_set)
            elif isinstance(cmd, FlowsRepl):
                _do_repl(client, flow_set, flows_cfg=config.flows, jq_filter=cmd.jq_filter)
            elif isinstance(cmd, FlowsClear):
                _do_clear(err, client, flow_set, clear_all=cmd.all)
    except httpx.ConnectError:
        err.print("[red]Cannot connect to mitmweb. Is ccproxy running?[/red]")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        err.print(f"[red]HTTP {e.response.status_code}: {e.response.text[:200]}[/red]")
        sys.exit(1)
    except ValueError as e:
        err.print(f"[red]{e}[/red]")
        sys.exit(1)
