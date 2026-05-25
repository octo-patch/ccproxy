"""Tests for shape CLI subcommands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ccproxy.shapes import ShapeSave, _do_shape_save, handle_shapes


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
