"""Tests for ShapeCaptureAddon shape artifact generation."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from mitmproxy import http
from mitmproxy.io import FlowReader
from mitmproxy.test import tflow

from ccproxy.inspector.fingerprint import CLIENT_FINGERPRINT_METADATA, REPLAY_FINGERPRINT_METADATA
from ccproxy.inspector.shape_capturer import ShapeCaptureAddon
from ccproxy.shaping.store import ShapeStore, clear_store_instance


@pytest.fixture()
def store(tmp_path: Path) -> Generator[ShapeStore]:
    from ccproxy.config import CCProxyConfig, set_config_instance
    from ccproxy.shaping.store import _store_lock

    set_config_instance(CCProxyConfig())
    shape_store = ShapeStore(tmp_path / "shapes")

    import ccproxy.shaping.store as store_mod

    with _store_lock:
        store_mod._store_instance = shape_store
    yield shape_store
    clear_store_instance()


def _flow(flow_id: str = "abc123") -> http.HTTPFlow:
    f = tflow.tflow()
    f.id = flow_id
    f.request = http.Request.make(
        "POST",
        "https://api.anthropic.com/v1/messages",
        b'{"model": "claude", "messages": [{"role": "user", "content": "hi"}]}',
        cast(
            dict[str | bytes, str | bytes],
            {"x-app": "cli", "user-agent": "test-cli/1.0", "content-type": "application/json"},
        ),
    )
    return f


def _fingerprint_dict() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "test",
        "captured_at": "2026-05-24T00:00:00+00:00",
        "sni": "api.anthropic.com",
        "alpn_protocols": ["http/1.1"],
        "legacy_version": 771,
        "supported_versions": ["0304", "0303"],
        "cipher_suites": ["1301", "1302"],
        "extensions": ["0000", "0010"],
        "supported_groups": ["001d"],
        "ec_point_formats": ["00"],
        "signature_algorithms": ["0403"],
        "signature_algorithm_names": ["ecdsa_secp256r1_sha256"],
        "ja3": "ja3-test",
        "ja3_full": "771,4865-4866,0-16,29,0",
        "ja4": "ja4-test",
        "ja4_r": "ja4-r-test",
        "http_version": "v1_1",
    }


def _read_raw_shape(store: ShapeStore, provider: str) -> http.HTTPFlow:
    path = store._path(provider)
    with path.open("rb") as fo:
        flows = [flow for flow in FlowReader(fo).stream() if isinstance(flow, http.HTTPFlow)]  # type: ignore[no-untyped-call]
    assert flows
    return flows[-1]


def _run_shape(
    capturer: ShapeCaptureAddon,
    flows_by_id: dict[str, http.HTTPFlow],
    ids: str,
    provider: str,
    mode: str = "mflow",
) -> dict[str, Any]:
    with patch.object(
        capturer,
        "_find_http_flow",
        side_effect=lambda fid: flows_by_id.get(fid),
    ):
        result = capturer.save_shape_artifact(ids, provider, mode)
    return cast(dict[str, Any], json.loads(result))


class TestShapeCaptureAddon:
    def test_single_flow(self, store: ShapeStore) -> None:
        capturer = ShapeCaptureAddon()
        result = _run_shape(capturer, {"abc123": _flow("abc123")}, "abc123", "anthropic")
        assert result["status"] == "ok"
        assert result["provider"] == "anthropic"
        assert result["mode"] == "mflow"
        assert result["flows_saved"] == 1
        assert result["missing"] == []
        assert store.pick("anthropic") is not None

    def test_multiple_flows(self, store: ShapeStore) -> None:
        flows = {fid: _flow(fid) for fid in ("f1", "f2", "f3")}
        capturer = ShapeCaptureAddon()
        result = _run_shape(capturer, flows, "f1,f2,f3", "anthropic")
        assert result["flows_saved"] == 3

    def test_skips_missing_flows(self, store: ShapeStore) -> None:
        capturer = ShapeCaptureAddon()
        result = _run_shape(
            capturer,
            {"exists": _flow("exists")},
            "exists,missing",
            "anthropic",
        )
        assert result["flows_saved"] == 1
        assert result["missing"] == ["missing"]

    def test_empty_ids_raises(self) -> None:
        capturer = ShapeCaptureAddon()
        with pytest.raises(ValueError, match="no flow ids"):
            capturer.save_shape_artifact("", "anthropic")

    def test_all_missing_reports_empty(self, store: ShapeStore) -> None:
        capturer = ShapeCaptureAddon()
        result = _run_shape(capturer, {}, "missing", "anthropic")
        assert result["status"] == "empty"
        assert result["flows_saved"] == 0
        assert result["missing"] == ["missing"]

    def test_strips_whitespace_and_empty_tokens(self, store: ShapeStore) -> None:
        capturer = ShapeCaptureAddon()
        result = _run_shape(
            capturer,
            {"f1": _flow("f1")},
            " f1 , ,",
            "anthropic",
        )
        assert result["flows_saved"] == 1

    def test_default_mode_writes_patch_queue(self, store: ShapeStore) -> None:
        capturer = ShapeCaptureAddon()
        base = _flow("base")
        target = _flow("target")
        target.request.content = b'{"model": "claude", "messages": [{"role": "user", "content": "patched"}]}'
        store.add("anthropic", base)

        result = _run_shape(capturer, {"target": target}, "target", "anthropic", mode="patch")

        assert result["status"] == "ok"
        assert result["mode"] == "patch"
        assert result["patches_written"] == 1
        patch_path = Path(result["patch"])
        assert patch_path.name == "0001-local-shape.patch"
        assert (patch_path.parent / "series").read_text() == "0001-local-shape.patch\n"
        picked = store.pick("anthropic")
        assert picked is not None
        assert picked.request is not None
        assert json.loads(picked.request.content or b"{}")["messages"][0]["content"] == "patched"

    def test_patch_mode_requires_one_flow(self, store: ShapeStore) -> None:
        capturer = ShapeCaptureAddon()
        store.add("anthropic", _flow("base"))

        with pytest.raises(ValueError, match="exactly one flow"):
            _run_shape(capturer, {"f1": _flow("f1"), "f2": _flow("f2")}, "f1,f2", "anthropic", mode="patch")

    def test_mflow_override_is_request_only_and_sanitized(self, store: ShapeStore) -> None:
        capturer = ShapeCaptureAddon()
        flow = _flow("abc123")
        flow.response = http.Response.make(200, b'{"ok": true}')
        flow.metadata["ccproxy.runtime"] = "value"
        flow.request.headers["authorization"] = "Bearer secret"
        flow.request.headers["cookie"] = "session=secret"
        _run_shape(capturer, {"abc123": flow}, "abc123", "anthropic")
        picked = store.pick("anthropic")
        assert picked is not None
        assert picked.request is not None
        assert picked.response is None
        assert picked.metadata["ccproxy.runtime"] == "value"
        assert picked.request.method == "POST"
        assert picked.request.pretty_host == "api.anthropic.com"
        assert picked.request.headers.get("user-agent") == "test-cli/1.0"
        assert "authorization" not in picked.request.headers
        assert "cookie" not in picked.request.headers
        raw = _read_raw_shape(store, "anthropic")
        assert raw.metadata["ccproxy.runtime"] == "value"

    def test_mflow_mode_embeds_captured_fingerprint(self, store: ShapeStore) -> None:
        capturer = ShapeCaptureAddon()
        flow = _flow("abc123")
        flow.metadata[CLIENT_FINGERPRINT_METADATA] = _fingerprint_dict()

        result = _run_shape(capturer, {"abc123": flow}, "abc123", "anthropic")

        assert result["fingerprint"] == "embedded"
        raw = _read_raw_shape(store, "anthropic")
        fingerprint = raw.metadata[REPLAY_FINGERPRINT_METADATA]
        assert fingerprint["provider"] == "anthropic"
        assert fingerprint["user_agent"] == "test-cli/1.0"
        assert fingerprint["runtime_version"] is None
        assert store.pick_fingerprint("anthropic") is not None

    def test_patch_mode_embeds_captured_fingerprint(self, store: ShapeStore) -> None:
        capturer = ShapeCaptureAddon()
        base = _flow("base")
        target = _flow("target")
        target.metadata[CLIENT_FINGERPRINT_METADATA] = _fingerprint_dict()
        store.add("anthropic", base)

        result = _run_shape(capturer, {"target": target}, "target", "anthropic", mode="patch")

        assert result["fingerprint"] == "embedded"
        raw = _read_raw_shape(store, "anthropic")
        fingerprint = raw.metadata[REPLAY_FINGERPRINT_METADATA]
        assert fingerprint["ja3"] == "ja3-test"


class TestFindHttpFlow:
    def test_returns_none_when_view_missing(self) -> None:
        master = MagicMock()
        master.addons.get.return_value = None
        with patch("ccproxy.inspector.flow_lookup.ctx") as mock_ctx:
            mock_ctx.master = master
            assert ShapeCaptureAddon._find_http_flow("x") is None

    def test_returns_flow_when_found(self) -> None:
        flow = _flow("abc")
        view = MagicMock()
        view.get_by_id.return_value = flow
        master = MagicMock()
        master.addons.get.return_value = view
        with patch("ccproxy.inspector.flow_lookup.ctx") as mock_ctx:
            mock_ctx.master = master
            assert ShapeCaptureAddon._find_http_flow("abc") is flow

    def test_returns_none_for_non_http_flow(self) -> None:
        view = MagicMock()
        view.get_by_id.return_value = object()
        master = MagicMock()
        master.addons.get.return_value = view
        with patch("ccproxy.inspector.flow_lookup.ctx") as mock_ctx:
            mock_ctx.master = master
            assert ShapeCaptureAddon._find_http_flow("x") is None
