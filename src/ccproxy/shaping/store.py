"""ShapeStore — per-provider on-disk store of request shapes.

One writable ``.mflow`` file per provider under ``shapes_dir``. Optional
package defaults are read from a fallback directory. Files are native
mitmproxy tnetstring dumps, openable in ``mitmweb --rfile``.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from mitmproxy import http
from mitmproxy.io import FlowReader, FlowWriter

from ccproxy.config import get_config, get_config_dir
from ccproxy.shaping.patches import apply_shape_patch_series
from ccproxy.utils import get_templates_dir

logger = logging.getLogger(__name__)


class ShapeStore:
    """Thread-safe per-provider store of captured and bundled request shapes."""

    def __init__(
        self,
        shapes_dir: Path,
        fallback_dir: Path | None = None,
        patches_dir: Path | None = None,
        fallback_patches_dir: Path | None = None,
    ) -> None:
        self._dir = shapes_dir
        self._fallback_dir = fallback_dir
        self._patches_dir = patches_dir
        self._fallback_patches_dir = fallback_patches_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def add(self, provider: str, flow: http.HTTPFlow) -> None:
        """Append a flow to the provider's shape file."""
        path = self._path(provider)
        with self._lock, path.open("ab") as fo:
            FlowWriter(fo).add(flow)  # type: ignore[no-untyped-call]
        logger.info("Saved shape for flow %s under provider %s", flow.id, provider)

    def pick(self, provider: str) -> http.HTTPFlow | None:
        """Return the most recent user shape, then the bundled default."""
        with self._lock:
            user_flow = self._pick_from(self._path(provider))
            if user_flow is not None:
                self._apply_patch_dirs(user_flow, provider, [self._patches_dir])
                return user_flow

            fallback_flow = self._pick_from(self._fallback_path(provider))
            if fallback_flow is not None:
                self._apply_patch_dirs(fallback_flow, provider, [self._fallback_patches_dir, self._patches_dir])
                return fallback_flow

            return None

    def clear(self, provider: str) -> None:
        """Delete the provider's shape file, if any."""
        with self._lock:
            self._path(provider).unlink(missing_ok=True)

    def list_providers(self) -> list[str]:
        """Return sorted list of providers with at least one shape file."""
        with self._lock:
            providers = {p.stem for p in self._dir.glob("*.mflow")}
            if self._fallback_dir is not None and self._fallback_dir.exists():
                providers.update(p.stem for p in self._fallback_dir.glob("*.mflow"))
            return sorted(providers)

    def _path(self, provider: str) -> Path:
        return self._dir / f"{provider}.mflow"

    def _fallback_path(self, provider: str) -> Path | None:
        if self._fallback_dir is None:
            return None
        return self._fallback_dir / f"{provider}.mflow"

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

    @staticmethod
    def _apply_patch_dirs(flow: http.HTTPFlow, provider: str, patch_dirs: list[Path | None]) -> None:
        for patch_dir in patch_dirs:
            apply_shape_patch_series(flow, provider, patch_dir)


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

    if config.shaping.shapes_dir:
        shapes_dir = Path(config.shaping.shapes_dir).expanduser()
    else:
        shapes_dir = config_dir / "shaping" / "shapes"

    fallback_dir: Path | None = None
    fallback_patches_dir: Path | None = None
    try:
        templates_dir = get_templates_dir()
    except RuntimeError:
        templates_dir = None
    if templates_dir is not None:
        candidate = templates_dir / "shapes"
        if candidate.exists():
            fallback_dir = candidate
        patches_candidate = templates_dir / "shapes" / "patches"
        if patches_candidate.exists():
            fallback_patches_dir = patches_candidate

    if config.shaping.patches_dir:
        patches_dir = Path(config.shaping.patches_dir).expanduser()
    else:
        patches_dir = config_dir / "shaping" / "patches"

    return ShapeStore(
        shapes_dir=shapes_dir,
        fallback_dir=fallback_dir,
        patches_dir=patches_dir,
        fallback_patches_dir=fallback_patches_dir,
    )


def clear_store_instance() -> None:
    """Reset the singleton (for tests)."""
    global _store_instance
    _store_instance = None
