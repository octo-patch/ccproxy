"""ShapeStore — per-provider on-disk store of request shapes.

One writable ``.mflow`` override per provider may live under ``shapes_dir``.
Provider patch queues live next to those overrides as ``{provider}/series``.
Optional package defaults are read from a fallback directory.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mitmproxy import http
from mitmproxy.io import FlowReader, FlowWriter

from ccproxy.config import get_config, get_config_dir
from ccproxy.inspector.fingerprint import REPLAY_FINGERPRINT_METADATA, CapturedFingerprint
from ccproxy.shaping.patches import ShapePatchWriteResult, apply_shape_patch_series, write_shape_patch
from ccproxy.utils import get_templates_dir

logger = logging.getLogger(__name__)


class ShapeStore:
    """Thread-safe per-provider store of captured and bundled request shapes."""

    def __init__(
        self,
        shapes_dir: Path,
        fallback_dir: Path | None = None,
    ) -> None:
        self._dir = shapes_dir
        self._fallback_dir = fallback_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def add(self, provider: str, flow: http.HTTPFlow) -> None:
        """Append a flow to the provider's shape file."""
        path = self._path(provider)
        writable = _prepare_flow_for_write(flow)
        with self._lock, path.open("ab") as fo:
            FlowWriter(fo).add(writable)  # type: ignore[no-untyped-call]
        logger.info("Saved shape for flow %s under provider %s", flow.id, provider)

    def pick(self, provider: str) -> http.HTTPFlow | None:
        """Return the most recent user shape, then the bundled default."""
        with self._lock:
            flow = self._pick_base(provider)
            if flow is None:
                return None
            apply_shape_patch_series(flow, provider, self._dir)
            return flow

    def pick_base(self, provider: str) -> http.HTTPFlow | None:
        """Return the most recent user shape or bundled default without patches."""
        with self._lock:
            return self._pick_base(provider)

    def write_patch(
        self,
        provider: str,
        target_flow: http.HTTPFlow,
        *,
        patch_name: str = "0001-local-shape.patch",
    ) -> ShapePatchWriteResult:
        """Write a patch queue entry from the provider base to ``target_flow``."""
        with self._lock:
            base_flow = self._pick_base(provider)
            if base_flow is None or base_flow.request is None:
                raise ValueError(f"no base shape available for provider {provider}")
            if target_flow.request is None:
                raise ValueError("target flow has no request")
            return write_shape_patch(
                base_flow.request,
                target_flow.request,
                self._patch_dir(provider),
                patch_name=patch_name,
            )

    def write_fingerprint(self, provider: str, fingerprint: CapturedFingerprint) -> Path:
        """Embed the provider's captured native TLS fingerprint profile in its ``.mflow`` metadata."""
        path = self._path(provider)
        with self._lock:
            flow = self._pick_base(provider)
            if flow is None:
                raise ValueError(f"no base shape available for provider {provider}")
            flow.metadata[REPLAY_FINGERPRINT_METADATA] = fingerprint.to_dict()
            self._write_single(path, flow)
        logger.info("Saved fingerprint profile for provider %s at %s", provider, path)
        return path

    def pick_fingerprint(self, provider: str) -> CapturedFingerprint | None:
        """Return the fingerprint profile embedded in the user shape, then bundled default."""
        with self._lock:
            for flow in (self._pick_from(self._path(provider)), self._pick_from(self._fallback_path(provider))):
                fingerprint = _fingerprint_from_metadata(provider, flow)
                if fingerprint is not None:
                    return fingerprint
            return None

    def clear(self, provider: str) -> None:
        """Delete the provider's user override and patch queue, if any."""
        with self._lock:
            self._path(provider).unlink(missing_ok=True)
            shutil.rmtree(self._patch_dir(provider), ignore_errors=True)

    def list_providers(self) -> list[str]:
        """Return sorted list of providers with at least one shape file."""
        with self._lock:
            providers = {p.stem for p in self._dir.glob("*.mflow")}
            providers.update(p.name for p in self._dir.iterdir() if p.is_dir() and (p / "series").exists())
            if self._fallback_dir is not None and self._fallback_dir.exists():
                providers.update(p.stem for p in self._fallback_dir.glob("*.mflow"))
            return sorted(providers)

    def _path(self, provider: str) -> Path:
        return self._dir / f"{provider}.mflow"

    def _fallback_path(self, provider: str) -> Path | None:
        if self._fallback_dir is None:
            return None
        return self._fallback_dir / f"{provider}.mflow"

    def _patch_dir(self, provider: str) -> Path:
        return self._dir / provider

    def _pick_base(self, provider: str) -> http.HTTPFlow | None:
        user_flow = self._pick_from(self._path(provider))
        if user_flow is not None:
            return user_flow
        return self._pick_from(self._fallback_path(provider))

    @staticmethod
    def _write_single(path: Path, flow: http.HTTPFlow) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fo:
            FlowWriter(fo).add(_prepare_flow_for_write(flow))  # type: ignore[no-untyped-call]

    @staticmethod
    def _pick_from(path: Path | None) -> http.HTTPFlow | None:
        if path is None or not path.exists():
            return None
        flows: list[http.HTTPFlow] = []
        try:
            with path.open("rb") as fo:
                for f in FlowReader(fo).stream():  # type: ignore[no-untyped-call]
                    if isinstance(f, http.HTTPFlow):
                        flows.append(f)
        except Exception as exc:
            logger.warning("Failed to read shape file %s: %s", path, exc)
            return None
        return flows[-1] if flows else None


# --- Singleton ---

_store_instance: ShapeStore | None = None
_store_lock = threading.Lock()


def get_store() -> ShapeStore:
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = _create_store()
    return _store_instance


def _create_store() -> ShapeStore:
    config = get_config()
    config_dir = get_config_dir()

    shapes_dir = Path(config.shaping.shapes_dir).expanduser() if config.shaping.shapes_dir else config_dir / "shapes"

    fallback_dir: Path | None = None
    try:
        templates_dir = get_templates_dir()
    except RuntimeError:
        templates_dir = None
    if templates_dir is not None:
        candidate = templates_dir / "shapes"
        if candidate.exists():
            fallback_dir = candidate

    return ShapeStore(
        shapes_dir=shapes_dir,
        fallback_dir=fallback_dir,
    )


def clear_store_instance() -> None:
    """Reset the singleton (for tests)."""
    global _store_instance
    _store_instance = None


def _prepare_flow_for_write(flow: http.HTTPFlow) -> http.HTTPFlow:
    clone: http.HTTPFlow = flow.copy()  # type: ignore[no-untyped-call]
    clone.metadata = {str(key): _metadata_to_state(value) for key, value in clone.metadata.items()}
    return clone


def _metadata_to_state(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _metadata_to_state(dataclasses.asdict(value))
    if hasattr(value, "get_state"):
        try:
            return _metadata_to_state(value.get_state())
        except Exception as exc:
            logger.debug("Failed to serialize metadata value via get_state(): %s", exc)
    if isinstance(value, Mapping):
        return {str(k): _metadata_to_state(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_metadata_to_state(item) for item in value]
    return repr(value)


def _fingerprint_from_metadata(provider: str, flow: http.HTTPFlow | None) -> CapturedFingerprint | None:
    if flow is None:
        return None
    raw = flow.metadata.get(REPLAY_FINGERPRINT_METADATA)
    if not isinstance(raw, dict):
        return None
    try:
        return CapturedFingerprint.from_dict(raw)
    except Exception as exc:
        logger.warning("Failed to load fingerprint profile for provider %s: %s", provider, exc)
        return None
