from __future__ import annotations

import threading
from typing import Any
from unittest.mock import patch

from ccproxy.mcp.buffer import NotificationBuffer, clear_buffer, get_buffer


def test_drain_session_single_task() -> None:
    buf = NotificationBuffer()
    buf.append("task-1", "session-a", {"type": "progress"})
    result = buf.drain_session("session-a")
    assert result == {"task-1": [{"type": "progress"}]}


def test_drain_session_multiple_tasks_same_session() -> None:
    buf = NotificationBuffer()
    buf.append("task-1", "session-a", {"type": "start"})
    buf.append("task-2", "session-a", {"type": "end"})
    result = buf.drain_session("session-a")
    assert set(result.keys()) == {"task-1", "task-2"}
    assert result["task-1"] == [{"type": "start"}]
    assert result["task-2"] == [{"type": "end"}]


def test_drain_session_isolates_other_sessions() -> None:
    buf = NotificationBuffer()
    buf.append("task-1", "session-a", {"type": "ping"})
    buf.append("task-2", "session-b", {"type": "pong"})
    result = buf.drain_session("session-a")
    assert "task-1" in result
    assert "task-2" not in result
    assert buf.has_events_for_session("session-b")


def test_overflow_drops_oldest_events() -> None:
    buf = NotificationBuffer(max_events=3)
    for i in range(5):
        buf.append("task-1", "session-a", {"seq": i})
    result = buf.drain_session("session-a")
    events = result["task-1"]
    assert len(events) == 3
    assert events[0]["type"] == "ccproxy_buffer_overflow"
    assert events[0]["dropped_events"] == 3
    assert [e["seq"] for e in events[1:]] == [3, 4]


def test_zero_max_events_keeps_no_events() -> None:
    buf = NotificationBuffer(max_events=0)
    buf.append("task-1", "session-a", {"seq": 0})
    assert buf.drain_session("session-a") == {}
    assert buf.is_empty() is True


def test_negative_max_events_rejected() -> None:
    try:
        NotificationBuffer(max_events=-1)
    except ValueError as exc:
        assert "max_events" in str(exc)
    else:
        raise AssertionError("negative max_events should fail")


def test_ttl_expiry_removes_stale_entries() -> None:
    buf = NotificationBuffer()
    with patch("ccproxy.mcp.buffer.time") as mock_time:
        mock_time.time.return_value = 1000.0
        buf.append("task-1", "session-a", {"type": "event"})
        mock_time.time.return_value = 1700.0
        removed = buf.expire(ttl_seconds=600)
    assert removed == 1
    assert buf.is_empty()


def test_drain_session_empty_buffer() -> None:
    buf = NotificationBuffer()
    result = buf.drain_session("session-x")
    assert result == {}


def test_has_events_for_session_true() -> None:
    buf = NotificationBuffer()
    buf.append("task-1", "session-a", {"type": "event"})
    assert buf.has_events_for_session("session-a") is True


def test_has_events_for_session_false_no_match() -> None:
    buf = NotificationBuffer()
    buf.append("task-1", "session-a", {"type": "event"})
    assert buf.has_events_for_session("session-z") is False


def test_has_events_for_session_false_after_drain() -> None:
    buf = NotificationBuffer()
    buf.append("task-1", "session-a", {"type": "event"})
    buf.drain_session("session-a")
    assert buf.has_events_for_session("session-a") is False


def test_concurrent_drain_disjoint_results() -> None:
    buf = NotificationBuffer()
    for i in range(10):
        buf.append(f"task-{i}", "session-a", {"seq": i})

    results: list[dict[str, list[dict[str, Any]]]] = [{}, {}]

    def drain(index: int) -> None:
        results[index] = buf.drain_session("session-a")

    t1 = threading.Thread(target=drain, args=(0,))
    t2 = threading.Thread(target=drain, args=(1,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    combined = {**results[0], **results[1]}
    assert set(combined.keys()) == {f"task-{i}" for i in range(10)}
    assert len(results[0]) + len(results[1]) == 10


def test_clear_buffer_resets_singleton() -> None:
    b1 = get_buffer()
    b1.append("task-1", "session-a", {"type": "event"})
    clear_buffer()
    b2 = get_buffer()
    assert b2 is not b1
    assert b2.is_empty()


def test_is_empty_false_after_append() -> None:
    buf = NotificationBuffer()
    buf.append("task-1", "session-a", {"type": "event"})
    assert buf.is_empty() is False


def test_is_empty_true_after_drain() -> None:
    buf = NotificationBuffer()
    buf.append("task-1", "session-a", {"type": "event"})
    buf.drain_session("session-a")
    assert buf.is_empty() is True
