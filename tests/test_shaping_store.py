"""Tests for ccproxy.shaping.store.ShapeStore."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from mitmproxy import http
from mitmproxy.io import FlowReader
from mitmproxy.test import tflow

from ccproxy.inspector.fingerprint import REPLAY_FINGERPRINT_METADATA, CapturedFingerprint
from ccproxy.shaping.store import ShapeStore


@pytest.fixture()
def seeds_dir(tmp_path: Path) -> Path:
    return tmp_path / "seeds"


def _flow(host: str = "api.anthropic.com", path: str = "/v1/messages") -> http.HTTPFlow:
    f = tflow.tflow()
    f.request = http.Request.make(
        "POST",
        f"https://{host}{path}",
        b'{"hello": "world"}',
        {"x-custom": "v"},
    )
    return f


def _fingerprint() -> CapturedFingerprint:
    return CapturedFingerprint(
        schema_version=1,
        source="test",
        captured_at="2026-05-24T00:00:00+00:00",
        sni="api.anthropic.com",
        alpn_protocols=("http/1.1",),
        legacy_version=771,
        supported_versions=("0304", "0303"),
        cipher_suites=("1301", "1302"),
        extensions=("0000", "0010"),
        supported_groups=("001d",),
        ec_point_formats=("00",),
        signature_algorithms=("0403",),
        signature_algorithm_names=("ecdsa_secp256r1_sha256",),
        ja3="ja3-test",
        ja3_full="771,4865-4866,0-16,29,0",
        ja4="ja4-test",
        ja4_r="ja4-r-test",
        http_version="v1_1",
        provider="anthropic",
    )


def _read_shape(path: Path) -> http.HTTPFlow:
    with path.open("rb") as fo:
        flows = [flow for flow in FlowReader(fo).stream() if isinstance(flow, http.HTTPFlow)]  # type: ignore[no-untyped-call]
    assert flows
    return flows[-1]


class TestShapeStore:
    def test_init_creates_directory(self, seeds_dir: Path) -> None:
        assert not seeds_dir.exists()
        ShapeStore(seeds_dir)
        assert seeds_dir.is_dir()

    def test_add_and_pick_roundtrip(self, seeds_dir: Path) -> None:
        store = ShapeStore(seeds_dir)
        store.add("anthropic", _flow())
        picked = store.pick("anthropic")
        assert picked is not None
        assert picked.request is not None
        assert picked.request.pretty_host == "api.anthropic.com"

    def test_pick_returns_none_when_missing(self, seeds_dir: Path) -> None:
        store = ShapeStore(seeds_dir)
        assert store.pick("anthropic") is None

    def test_pick_uses_fallback_when_user_shape_missing(self, tmp_path: Path) -> None:
        user_dir = tmp_path / "user"
        fallback_dir = tmp_path / "fallback"
        ShapeStore(fallback_dir).add("anthropic", _flow(host="fallback.example"))

        picked = ShapeStore(user_dir, fallback_dir=fallback_dir).pick("anthropic")

        assert picked is not None
        assert picked.request is not None
        assert picked.request.pretty_host == "fallback.example"

    def test_pick_prefers_user_shape_over_fallback(self, tmp_path: Path) -> None:
        user_dir = tmp_path / "user"
        fallback_dir = tmp_path / "fallback"
        ShapeStore(fallback_dir).add("anthropic", _flow(host="fallback.example"))
        store = ShapeStore(user_dir, fallback_dir=fallback_dir)
        store.add("anthropic", _flow(host="user.example"))

        picked = store.pick("anthropic")

        assert picked is not None
        assert picked.request is not None
        assert picked.request.pretty_host == "user.example"

    def test_pick_returns_most_recent(self, seeds_dir: Path) -> None:
        store = ShapeStore(seeds_dir)
        store.add("anthropic", _flow(host="old.example"))
        store.add("anthropic", _flow(host="new.example"))
        picked = store.pick("anthropic")
        assert picked is not None
        assert picked.request is not None
        assert picked.request.pretty_host == "new.example"

    def test_clear_removes_seed_file(self, seeds_dir: Path) -> None:
        store = ShapeStore(seeds_dir)
        store.add("anthropic", _flow())
        patch_dir = seeds_dir / "anthropic"
        patch_dir.mkdir()
        (patch_dir / "series").write_text("0001-local.patch\n")
        assert (seeds_dir / "anthropic.mflow").exists()
        store.clear("anthropic")
        assert not (seeds_dir / "anthropic.mflow").exists()
        assert not patch_dir.exists()

    def test_clear_reveals_fallback_shape(self, tmp_path: Path) -> None:
        user_dir = tmp_path / "user"
        fallback_dir = tmp_path / "fallback"
        ShapeStore(fallback_dir).add("anthropic", _flow(host="fallback.example"))
        store = ShapeStore(user_dir, fallback_dir=fallback_dir)
        store.add("anthropic", _flow(host="user.example"))

        store.clear("anthropic")
        picked = store.pick("anthropic")

        assert not (user_dir / "anthropic.mflow").exists()
        assert (fallback_dir / "anthropic.mflow").exists()
        assert picked is not None
        assert picked.request is not None
        assert picked.request.pretty_host == "fallback.example"

    def test_clear_is_idempotent(self, seeds_dir: Path) -> None:
        ShapeStore(seeds_dir).clear("never-seeded")

    def test_list_providers(self, seeds_dir: Path) -> None:
        store = ShapeStore(seeds_dir)
        store.add("anthropic", _flow())
        store.add("gemini", _flow())
        assert store.list_providers() == ["anthropic", "gemini"]

    def test_list_providers_includes_fallbacks(self, tmp_path: Path) -> None:
        user_dir = tmp_path / "user"
        fallback_dir = tmp_path / "fallback"
        ShapeStore(fallback_dir).add("anthropic", _flow())
        ShapeStore(fallback_dir).add("gemini", _flow())
        store = ShapeStore(user_dir, fallback_dir=fallback_dir)
        store.add("anthropic", _flow(host="user.example"))

        assert store.list_providers() == ["anthropic", "gemini"]

    def test_isolates_per_provider(self, seeds_dir: Path) -> None:
        store = ShapeStore(seeds_dir)
        store.add("anthropic", _flow(host="a.example"))
        store.add("gemini", _flow(host="g.example"))
        a = store.pick("anthropic")
        g = store.pick("gemini")
        assert a is not None and a.request is not None
        assert g is not None and g.request is not None
        assert a.request.pretty_host == "a.example"
        assert g.request.pretty_host == "g.example"

    def test_persists_across_instances(self, seeds_dir: Path) -> None:
        ShapeStore(seeds_dir).add("anthropic", _flow())
        picked = ShapeStore(seeds_dir).pick("anthropic")
        assert picked is not None

    def test_pick_preserves_shape_metadata(self, seeds_dir: Path) -> None:
        store = ShapeStore(seeds_dir)
        flow = _flow()
        flow.metadata["ccproxy.shape"] = "persisted"
        flow.metadata[REPLAY_FINGERPRINT_METADATA] = _fingerprint().to_dict()
        store.add("anthropic", flow)

        picked = store.pick("anthropic")
        fingerprint = store.pick_fingerprint("anthropic")
        raw = _read_shape(seeds_dir / "anthropic.mflow")

        assert picked is not None
        assert picked.metadata["ccproxy.shape"] == "persisted"
        assert raw.metadata["ccproxy.shape"] == "persisted"
        assert fingerprint is not None
        assert fingerprint.ja3 == "ja3-test"

    def test_pick_fingerprint_falls_back_when_user_shape_lacks_profile(self, tmp_path: Path) -> None:
        user_dir = tmp_path / "user"
        fallback_dir = tmp_path / "fallback"
        fallback = _flow(host="fallback.example")
        fallback.metadata[REPLAY_FINGERPRINT_METADATA] = _fingerprint().to_dict()
        ShapeStore(fallback_dir).add("anthropic", fallback)
        store = ShapeStore(user_dir, fallback_dir=fallback_dir)
        store.add("anthropic", _flow(host="user.example"))

        fingerprint = store.pick_fingerprint("anthropic")

        assert fingerprint is not None
        assert fingerprint.ja4 == "ja4-test"

    def test_write_fingerprint_copies_fallback_shape_to_user_file(self, tmp_path: Path) -> None:
        user_dir = tmp_path / "user"
        fallback_dir = tmp_path / "fallback"
        ShapeStore(fallback_dir).add("anthropic", _flow(host="fallback.example"))
        store = ShapeStore(user_dir, fallback_dir=fallback_dir)

        store.write_fingerprint("anthropic", _fingerprint())

        picked = store.pick("anthropic")
        raw = _read_shape(user_dir / "anthropic.mflow")
        assert picked is not None
        assert picked.request is not None
        assert picked.request.pretty_host == "fallback.example"
        assert raw.metadata[REPLAY_FINGERPRINT_METADATA]["ja3"] == "ja3-test"


class TestGetStoreSingleton:
    def test_get_store_uses_configured_seeds_dir(self, tmp_path: Path) -> None:
        from ccproxy.config import CCProxyConfig, set_config_instance
        from ccproxy.shaping.store import clear_store_instance, get_store

        explicit_dir = tmp_path / "custom-seeds"
        config = CCProxyConfig()
        config.shaping.shapes_dir = str(explicit_dir)
        set_config_instance(config)
        clear_store_instance()

        store = get_store()
        store.add("anthropic", _flow())
        assert (explicit_dir / "anthropic.mflow").exists()
        clear_store_instance()

    def test_get_store_falls_back_to_config_dir(self, tmp_path: Path, monkeypatch: Any) -> None:
        from ccproxy.config import CCProxyConfig, set_config_instance
        from ccproxy.shaping.store import clear_store_instance, get_store

        monkeypatch.setenv("CCPROXY_CONFIG_DIR", str(tmp_path))
        set_config_instance(CCProxyConfig())
        clear_store_instance()

        store = get_store()
        store.add("anthropic", _flow())
        assert (tmp_path / "shapes" / "anthropic.mflow").exists()
        clear_store_instance()

    def test_get_store_is_a_singleton(self, tmp_path: Path, monkeypatch: Any) -> None:
        from ccproxy.config import CCProxyConfig, set_config_instance
        from ccproxy.shaping.store import clear_store_instance, get_store

        monkeypatch.setenv("CCPROXY_CONFIG_DIR", str(tmp_path))
        set_config_instance(CCProxyConfig())
        clear_store_instance()

        assert get_store() is get_store()
        clear_store_instance()

    def test_get_store_uses_bundled_fallback_dir(self, tmp_path: Path, monkeypatch: Any) -> None:
        from ccproxy.config import CCProxyConfig, set_config_instance
        from ccproxy.shaping.store import clear_store_instance, get_store

        config_dir = tmp_path / "config"
        templates_dir = tmp_path / "templates"
        fallback_dir = templates_dir / "shapes"
        ShapeStore(fallback_dir).add("anthropic", _flow(host="fallback.example"))
        monkeypatch.setenv("CCPROXY_CONFIG_DIR", str(config_dir))
        monkeypatch.setattr("ccproxy.shaping.store.get_templates_dir", lambda: templates_dir)
        set_config_instance(CCProxyConfig())
        clear_store_instance()

        picked = get_store().pick("anthropic")

        assert picked is not None
        assert picked.request is not None
        assert picked.request.pretty_host == "fallback.example"
        clear_store_instance()
