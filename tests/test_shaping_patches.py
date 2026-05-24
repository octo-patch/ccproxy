"""Tests for quilt-style shape patch series."""

from __future__ import annotations

import difflib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from mitmproxy import http
from mitmproxy.test import tflow

from ccproxy.shaping.patches import (
    ShapePatchError,
    _request_to_patch_text,
    apply_shape_patch_series,
)
from ccproxy.shaping.store import ShapeStore, clear_store_instance, get_store


def _flow(
    *,
    host: str = "api.example.com",
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> http.HTTPFlow:
    flow = tflow.tflow()
    flow.request = http.Request.make(
        "POST",
        f"https://{host}/v1/messages",
        json.dumps(body or {"seed": "old"}).encode(),
        headers or {"content-type": "application/json", "x-seed": "old"},
    )
    return flow


def _patch_text(
    before: str,
    mutator: Callable[[dict[str, Any]], None],
    *,
    fromfile: str = "a/shape.json",
    tofile: str = "b/shape.json",
) -> tuple[str, str]:
    doc = json.loads(before)
    mutator(doc)
    after = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    patch = "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
        )
    )
    return patch + "\n", after


def _write_series(provider_dir: Path, entries: dict[str, str], series: str | None = None) -> None:
    provider_dir.mkdir(parents=True)
    for name, text in entries.items():
        (provider_dir / name).write_text(text)
    (provider_dir / "series").write_text(series or "".join(f"{name}\n" for name in entries))


def test_applies_series_in_order(tmp_path: Path) -> None:
    flow = _flow()
    first_patch, first_text = _patch_text(
        _request_to_patch_text(flow.request),
        lambda doc: doc["body"].update({"seed": "patched"}),
    )
    second_patch, _ = _patch_text(
        first_text,
        lambda doc: doc["headers"].update({"x-seed": "patched"}),
    )
    patches_dir = tmp_path / "patches"
    _write_series(
        patches_dir / "anthropic",
        {
            "0001-body.patch": first_patch,
            "0002-headers.patch": second_patch,
        },
    )

    assert apply_shape_patch_series(flow, "anthropic", patches_dir) is True

    body = json.loads(flow.request.content or b"{}")
    assert body["seed"] == "patched"
    assert flow.request.headers["x-seed"] == "patched"


def test_series_supports_p0_patch_paths(tmp_path: Path) -> None:
    flow = _flow()
    patch, _ = _patch_text(
        _request_to_patch_text(flow.request),
        lambda doc: doc.update({"url": "https://patched.example/v1/messages?beta=true"}),
        fromfile="shape.json",
        tofile="shape.json",
    )
    patches_dir = tmp_path / "patches"
    _write_series(patches_dir / "anthropic", {"0001-url.patch": patch}, series="0001-url.patch -p0\n")

    assert apply_shape_patch_series(flow, "anthropic", patches_dir) is True

    assert flow.request.pretty_host == "patched.example"
    assert flow.request.query["beta"] == "true"


def test_missing_series_is_noop(tmp_path: Path) -> None:
    flow = _flow()

    assert apply_shape_patch_series(flow, "anthropic", tmp_path / "patches") is False

    assert json.loads(flow.request.content or b"{}") == {"seed": "old"}


def test_bad_patch_context_raises(tmp_path: Path) -> None:
    patches_dir = tmp_path / "patches"
    _write_series(
        patches_dir / "anthropic",
        {
            "0001-bad.patch": "\n".join(
                [
                    "--- a/shape.json",
                    "+++ b/shape.json",
                    "@@ -1,1 +1,1 @@",
                    "-not the shape document",
                    "+replacement",
                    "",
                ]
            ),
        },
    )

    with pytest.raises(ShapePatchError, match="hunk context"):
        apply_shape_patch_series(_flow(), "anthropic", patches_dir)


def test_store_applies_user_patch_to_fallback_shape(tmp_path: Path) -> None:
    fallback_flow = _flow(body={"seed": "fallback"})
    fallback_dir = tmp_path / "fallback"
    ShapeStore(fallback_dir).add("anthropic", fallback_flow)

    patch, _ = _patch_text(
        _request_to_patch_text(fallback_flow.request),
        lambda doc: doc["body"].update({"seed": "user-patched"}),
    )
    patches_dir = tmp_path / "patches"
    _write_series(patches_dir / "anthropic", {"0001-user.patch": patch})

    store = ShapeStore(tmp_path / "user", fallback_dir=fallback_dir, patches_dir=patches_dir)
    picked = store.pick("anthropic")

    assert picked is not None
    assert picked.request is not None
    assert json.loads(picked.request.content or b"{}")["seed"] == "user-patched"


def test_get_store_uses_configured_patch_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ccproxy.config import CCProxyConfig, set_config_instance

    config_dir = tmp_path / "config"
    shapes_dir = tmp_path / "shapes"
    patches_dir = tmp_path / "patches"
    flow = _flow(body={"seed": "configured"})
    ShapeStore(shapes_dir).add("anthropic", flow)

    patch, _ = _patch_text(
        _request_to_patch_text(flow.request),
        lambda doc: doc["body"].update({"seed": "patched-by-config"}),
    )
    _write_series(patches_dir / "anthropic", {"0001-config.patch": patch})

    monkeypatch.setenv("CCPROXY_CONFIG_DIR", str(config_dir))
    set_config_instance(
        CCProxyConfig(shaping={"shapes_dir": str(shapes_dir), "patches_dir": str(patches_dir)}),
    )
    clear_store_instance()

    picked = get_store().pick("anthropic")

    assert picked is not None
    assert picked.request is not None
    assert json.loads(picked.request.content or b"{}")["seed"] == "patched-by-config"
