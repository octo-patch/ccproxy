"""Thread-safe notification buffer for MCP terminal events."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MAX_EVENTS = 64 * 1024
DEFAULT_TTL_SECONDS = 600


@dataclass
class TaskBuffer:
    """Buffer for a single task's events."""

    task_id: str
    """MCP task identifier."""

    session_id: str
    """Claude Code session this task belongs to."""

    events: list[dict[str, Any]] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    """Buffered notification events for this task."""

    last_seen: float = field(default_factory=time.time)
    """Timestamp of the most recent event (for TTL expiry)."""


class NotificationBuffer:
    """Thread-safe buffer for MCP notification events, keyed by task_id."""

    def __init__(self, max_events: int = DEFAULT_MAX_EVENTS) -> None:
        if max_events < 0:
            raise ValueError("max_events must be non-negative")
        self._buffers: dict[str, TaskBuffer] = {}
        self._lock = threading.Lock()
        self._max_events = max_events

    def append(self, task_id: str, session_id: str, event: dict[str, Any]) -> None:
        """Append an event to the buffer for a task. Creates buffer if needed."""
        with self._lock:
            buf = self._buffers.get(task_id)
            if buf is None:
                buf = TaskBuffer(task_id=task_id, session_id=session_id)
            self._buffers[task_id] = buf
            buf.events.append(event)
            buf.last_seen = time.time()
            if len(buf.events) > self._max_events:
                if self._max_events > 0:
                    old_dropped = 0
                    actual_events = buf.events
                    first = actual_events[0] if actual_events else None
                    if isinstance(first, dict) and first.get("type") == "ccproxy_buffer_overflow":
                        old_dropped = int(first.get("dropped_events") or 0)
                        actual_events = actual_events[1:]
                    tail_count = self._max_events - 1
                    tail = actual_events[-tail_count:] if tail_count > 0 else []
                    marker = {
                        "type": "ccproxy_buffer_overflow",
                        "dropped_events": old_dropped + len(actual_events) - len(tail),
                        "max_events": self._max_events,
                    }
                    buf.events = [marker, *tail]
                else:
                    buf.events = []
            if not buf.events:
                del self._buffers[task_id]

    def drain_session(self, session_id: str) -> dict[str, list[dict[str, Any]]]:
        """Atomically drain all events for a session. Returns {task_id: events}."""
        result: dict[str, list[dict[str, Any]]] = {}
        with self._lock:
            to_remove: list[str] = []
            for task_id, buf in self._buffers.items():
                if buf.session_id == session_id and buf.events:
                    result[task_id] = buf.events
                    buf.events = []
                    to_remove.append(task_id)
            for task_id in to_remove:
                del self._buffers[task_id]
        return result

    def expire(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> int:
        """Remove entries older than ttl_seconds. Returns count removed."""
        now = time.time()
        removed = 0
        with self._lock:
            expired = [tid for tid, buf in self._buffers.items() if now - buf.last_seen > ttl_seconds]
            for tid in expired:
                del self._buffers[tid]
                removed += 1
        return removed

    def has_events_for_session(self, session_id: str) -> bool:
        """Check if any task with matching session_id has buffered events."""
        with self._lock:
            return any(buf.session_id == session_id and buf.events for buf in self._buffers.values())

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffers) == 0


_buffer: NotificationBuffer | None = None
_buffer_lock = threading.Lock()


def get_buffer() -> NotificationBuffer:
    """Creates buffer if needed."""
    global _buffer
    if _buffer is None:
        with _buffer_lock:
            if _buffer is None:
                try:
                    from ccproxy.config import get_config

                    max_events = get_config().mcp.buffer.max_events_per_task
                except Exception:
                    max_events = DEFAULT_MAX_EVENTS
                _buffer = NotificationBuffer(max_events=max_events)
    return _buffer


def clear_buffer() -> None:
    """Reset the singleton buffer. For testing."""
    global _buffer
    with _buffer_lock:
        _buffer = None
