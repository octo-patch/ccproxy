"""Tests for shape CLI subcommands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from mitmproxy import http
from mitmproxy.io import FlowWriter
from mitmproxy.test import tflow

from ccproxy.shapes import (
    ShapeAudit,
    ShapeSave,
    _do_shape_audit,
    _do_shape_save,
    _read_latest,
    handle_shapes,
)


class TestDoShapeSave:
    def test_patch_mode_requires_single_flow(self) -> None:
        console = MagicMock()
        client = MagicMock()

        with pytest.raises(SystemExit):
            _do_shape_save(console, client, [{"id": "a"}, {"id": "b"}], provider="anthropic", mflow=False)

        client.save_shape.assert_not_called()

    def test_patch_mode_calls_client(self) -> None:
        console = MagicMock()
        client = MagicMock()
        client.save_shape.return_value = {"provider": "anthropic", "status": "ok", "patch": "shape.patch"}

        _do_shape_save(console, client, [{"id": "a"}], provider="anthropic", mflow=False)

        client.save_shape.assert_called_once_with(["a"], "anthropic", mode="patch")
        assert "Saved shape patch" in str(console.print.call_args)

    def test_mflow_mode_accepts_multiple_flows(self) -> None:
        console = MagicMock()
        client = MagicMock()
        client.save_shape.return_value = {"provider": "anthropic", "flows_saved": 2, "missing": []}

        _do_shape_save(console, client, [{"id": "a"}, {"id": "b"}], provider="anthropic", mflow=True)

        client.save_shape.assert_called_once_with(["a", "b"], "anthropic", mode="mflow")
        assert "Saved .mflow shape" in str(console.print.call_args)


class TestHandleShapes:
    @patch("ccproxy.config.get_config")
    @patch("ccproxy.shapes._make_client")
    @patch("ccproxy.shapes._resolve_flow_set")
    @patch("ccproxy.shapes._do_shape_save")
    def test_save_subcommand(
        self,
        mock_shape: MagicMock,
        mock_resolve: MagicMock,
        mock_client: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        mock_ctx = MagicMock()
        mock_client.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        flow_set = [{"id": "a"}]
        mock_resolve.return_value = flow_set

        handle_shapes(ShapeSave(provider="anthropic"), Path("/tmp"))  # noqa: S108

        mock_shape.assert_called_once()
        assert mock_shape.call_args.kwargs["provider"] == "anthropic"
        assert mock_shape.call_args.kwargs["mflow"] is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_mflow(path: Path, *, sensitive_header: str | None = None, with_response: bool = False) -> None:
    """Write a minimal .mflow file for testing _read_latest and _do_shape_audit."""
    f = tflow.tflow()
    headers: dict[str, str] = {"content-type": "application/json"}
    if sensitive_header:
        headers[sensitive_header] = "secret"
    f.request = http.Request.make(
        "POST",
        "https://api.anthropic.com/v1/messages",
        b'{"messages":[]}',
        headers,
    )
    if with_response:
        f.response = http.Response.make(200, b"ok", {"content-type": "text/plain"})
    else:
        f.response = None
    with path.open("wb") as fout:
        FlowWriter(fout).add(f)


# ---------------------------------------------------------------------------
# _do_shape_save — additional branches
# ---------------------------------------------------------------------------


class TestDoShapeSaveAdditional:
    def test_empty_flow_set_exits(self) -> None:
        console = MagicMock()
        client = MagicMock()
        with pytest.raises(SystemExit):
            _do_shape_save(console, client, [], provider="anthropic", mflow=False)
        client.save_shape.assert_not_called()

    def test_patch_mode_unchanged_status_prints_unchanged(self) -> None:
        console = MagicMock()
        client = MagicMock()
        client.save_shape.return_value = {"provider": "anthropic", "status": "unchanged", "patch": None}

        _do_shape_save(console, client, [{"id": "a"}], provider="anthropic", mflow=False)

        assert "unchanged" in str(console.print.call_args_list).lower()

    def test_mflow_mode_with_missing_flows_prints_missing_count(self) -> None:
        console = MagicMock()
        client = MagicMock()
        client.save_shape.return_value = {
            "provider": "anthropic",
            "flows_saved": 1,
            "missing": ["b"],
        }

        _do_shape_save(console, client, [{"id": "a"}], provider="anthropic", mflow=True)

        output = str(console.print.call_args_list)
        assert "1 missing" in output

    def test_mflow_mode_no_missing_omits_missing_suffix(self) -> None:
        console = MagicMock()
        client = MagicMock()
        client.save_shape.return_value = {"provider": "anthropic", "flows_saved": 2, "missing": []}

        _do_shape_save(console, client, [{"id": "a"}, {"id": "b"}], provider="anthropic", mflow=True)

        output = str(console.print.call_args_list)
        assert "missing" not in output


# ---------------------------------------------------------------------------
# _read_latest
# ---------------------------------------------------------------------------


class TestReadLatest:
    def test_returns_last_flow_when_multiple_present(self, tmp_path: Path) -> None:
        path = tmp_path / "test.mflow"
        f1 = tflow.tflow()
        f1.request = http.Request.make("GET", "https://example.com/first", b"", {})
        f1.response = None
        f2 = tflow.tflow()
        f2.request = http.Request.make("GET", "https://example.com/second", b"", {})
        f2.response = None
        with path.open("wb") as fout:
            w = FlowWriter(fout)
            w.add(f1)
            w.add(f2)

        flow = _read_latest(path)
        assert flow.request.path == "/second"

    def test_empty_mflow_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.mflow"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="empty mflow"):
            _read_latest(path)


# ---------------------------------------------------------------------------
# _do_shape_audit
# ---------------------------------------------------------------------------


class TestDoShapeAudit:
    def test_good_shape_file_prints_audited_count(self, tmp_path: Path) -> None:
        _write_mflow(tmp_path / "anthropic.mflow")
        console = MagicMock()
        _do_shape_audit(console, tmp_path)
        assert "Audited 1 shape file(s)" in str(console.print.call_args_list)

    def test_missing_directory_exits(self) -> None:
        console = MagicMock()
        with pytest.raises(SystemExit):
            _do_shape_audit(console, Path("/nonexistent/path/xyz"))
        assert "missing" in str(console.print.call_args_list).lower()

    def test_sensitive_header_causes_failure_exit(self, tmp_path: Path) -> None:
        _write_mflow(tmp_path / "anthropic.mflow", sensitive_header="authorization")
        console = MagicMock()
        with pytest.raises(SystemExit):
            _do_shape_audit(console, tmp_path)
        output = str(console.print.call_args_list)
        assert "authorization" in output.lower()

    def test_response_present_causes_failure_exit(self, tmp_path: Path) -> None:
        _write_mflow(tmp_path / "gemini.mflow", with_response=True)
        console = MagicMock()
        with pytest.raises(SystemExit):
            _do_shape_audit(console, tmp_path)
        assert "response is present" in str(console.print.call_args_list)

    def test_cookie_header_is_sensitive(self, tmp_path: Path) -> None:
        _write_mflow(tmp_path / "bad.mflow", sensitive_header="cookie")
        console = MagicMock()
        with pytest.raises(SystemExit):
            _do_shape_audit(console, tmp_path)

    def test_x_api_key_header_is_sensitive(self, tmp_path: Path) -> None:
        _write_mflow(tmp_path / "bad.mflow", sensitive_header="x-api-key")
        console = MagicMock()
        with pytest.raises(SystemExit):
            _do_shape_audit(console, tmp_path)

    def test_multiple_files_audited_count_correct(self, tmp_path: Path) -> None:
        _write_mflow(tmp_path / "a.mflow")
        _write_mflow(tmp_path / "b.mflow")
        console = MagicMock()
        _do_shape_audit(console, tmp_path)
        assert "Audited 2 shape file(s)" in str(console.print.call_args_list)

    def test_unreadable_mflow_reports_failure_and_exits(self, tmp_path: Path) -> None:
        (tmp_path / "corrupt.mflow").write_bytes(b"")
        console = MagicMock()
        with pytest.raises(SystemExit):
            _do_shape_audit(console, tmp_path)
        assert "unreadable" in str(console.print.call_args_list)

    def test_defaults_to_packaged_templates_directory(self) -> None:
        """_do_shape_audit(console, None) uses the packaged shapes dir (may be empty)."""
        console = MagicMock()
        with (
            patch("ccproxy.shapes.get_templates_dir") as mock_get_tpl,
            patch("pathlib.Path.exists", return_value=False),
        ):
            mock_get_tpl.return_value = Path("/nonexistent/templates")
            with pytest.raises(SystemExit):
                _do_shape_audit(console, None)


# ---------------------------------------------------------------------------
# handle_shapes — audit and error branches
# ---------------------------------------------------------------------------


class TestHandleShapesAudit:
    def test_audit_subcommand_dispatches_to_do_shape_audit(self, tmp_path: Path) -> None:
        _write_mflow(tmp_path / "anthropic.mflow")
        with patch("ccproxy.shapes.Console") as mock_console_cls:
            mock_console = MagicMock()
            mock_console_cls.return_value = mock_console
            handle_shapes(ShapeAudit(directory=tmp_path), tmp_path)
        assert "Audited 1 shape file(s)" in str(mock_console.print.call_args_list)

    @patch("ccproxy.config.get_config")
    @patch("ccproxy.shapes._make_client")
    @patch("ccproxy.shapes.Console")
    def test_save_connect_error_exits_with_message(
        self,
        mock_console_cls: MagicMock,
        mock_mc: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        mock_err = MagicMock()
        mock_console_cls.return_value = mock_err
        mock_config.return_value = MagicMock(flows=MagicMock())
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(side_effect=httpx.ConnectError("refused"))
        ctx.__exit__ = MagicMock(return_value=False)
        mock_mc.return_value = ctx

        with pytest.raises(SystemExit):
            handle_shapes(ShapeSave(provider="anthropic"), Path("/tmp"))  # noqa: S108

        assert "Cannot connect" in str(mock_err.print.call_args_list)

    @patch("ccproxy.config.get_config")
    @patch("ccproxy.shapes._make_client")
    @patch("ccproxy.shapes.Console")
    def test_save_http_status_error_exits_with_message(
        self,
        mock_console_cls: MagicMock,
        mock_mc: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        mock_err = MagicMock()
        mock_console_cls.return_value = mock_err
        mock_config.return_value = MagicMock(flows=MagicMock())
        resp = MagicMock()
        resp.status_code = 404
        resp.text = "not found here"
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(
            side_effect=httpx.HTTPStatusError("msg", request=MagicMock(), response=resp)
        )
        ctx.__exit__ = MagicMock(return_value=False)
        mock_mc.return_value = ctx

        with pytest.raises(SystemExit):
            handle_shapes(ShapeSave(provider="anthropic"), Path("/tmp"))  # noqa: S108

        assert "404" in str(mock_err.print.call_args_list)

    @patch("ccproxy.config.get_config")
    @patch("ccproxy.shapes._make_client")
    @patch("ccproxy.shapes.Console")
    def test_save_value_error_exits_with_message(
        self,
        mock_console_cls: MagicMock,
        mock_mc: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        mock_err = MagicMock()
        mock_console_cls.return_value = mock_err
        mock_config.return_value = MagicMock(flows=MagicMock())
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(side_effect=ValueError("bad shape"))
        ctx.__exit__ = MagicMock(return_value=False)
        mock_mc.return_value = ctx

        with pytest.raises(SystemExit):
            handle_shapes(ShapeSave(provider="anthropic"), Path("/tmp"))  # noqa: S108

        assert "bad shape" in str(mock_err.print.call_args_list)
