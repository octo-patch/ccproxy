"""Bounded in-memory capture of every inbound conduit-WebSocket frame.

The "never silently drop" guarantee for the handoff bridge: every raw WS message
the bridge receives is recorded here, whether or not its content was forwarded to
the answer stream. Non-turn frames (``app_notifications``, other conversations,
envelope shapes we have no handler for) therefore stay inspectable rather than
being discarded.

This is the ephemeral cache (a bounded ring). Durable JSONL persistence and
cross-turn retention belong to the future session-scoped WS manager
(CHATGPT-008 Increment 2).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class WSFrameRecord:
    """One captured inbound WebSocket message."""

    received_at: float
    """Wall-clock timestamp (``time.time()``) when the frame arrived."""

    topic: str
    """The turn topic the bridge was reading for when this frame arrived."""

    raw: str
    """The raw WebSocket message text, verbatim."""

    forwarded: int
    """Count of SSE items forwarded from this message to the answer stream."""


class WSFrameCapture:
    """Thread-safe bounded ring of inbound WS frames."""

    def __init__(self, *, max_frames: int = 4096) -> None:
        self._frames: deque[WSFrameRecord] = deque(maxlen=max_frames)
        self._lock = threading.Lock()

    def record(self, *, topic: str, raw: str, forwarded: int) -> None:
        """Append one captured frame. Never raises into the caller."""
        with self._lock:
            self._frames.append(WSFrameRecord(received_at=time.time(), topic=topic, raw=raw, forwarded=forwarded))

    def dump(self) -> list[WSFrameRecord]:
        """Return a snapshot of the captured frames in arrival order."""
        with self._lock:
            return list(self._frames)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()


_capture: WSFrameCapture | None = None
_capture_lock = threading.Lock()


def get_ws_capture() -> WSFrameCapture:
    """Return the process-wide WS frame capture singleton."""
    global _capture
    if _capture is None:
        with _capture_lock:
            if _capture is None:
                _capture = WSFrameCapture()
    return _capture


def clear_ws_capture() -> None:
    """Reset the capture singleton (test-suite teardown)."""
    get_ws_capture().clear()
