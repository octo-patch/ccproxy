"""Tests for hook result discriminated union."""

from __future__ import annotations

import json

import pytest
from pydantic_core import to_jsonable_python

from ccproxy.pipeline.context import Context
from ccproxy.pipeline.results import (
    _HookDeferred,
    _HookError,
    _HookSkipped,
    _HookSuccess,
    unwrap_hook_result,
    wrap_hook_call,
)


def test_hook_success_construction():
    """Test _HookSuccess constructs correctly."""
    result = _HookSuccess()
    assert result.kind == "success"


def test_hook_skipped_construction():
    """Test _HookSkipped constructs correctly."""
    result = _HookSkipped(reason="guard returned False")
    assert result.kind == "skipped"
    assert result.reason == "guard returned False"


def test_hook_error_construction():
    """Test _HookError constructs correctly."""
    result = _HookError(
        hook_name="test_hook",
        exc_type="ValueError",
        message="something went wrong",
        traceback="Traceback...",
    )
    assert result.kind == "error"
    assert result.hook_name == "test_hook"
    assert result.exc_type == "ValueError"
    assert result.message == "something went wrong"
    assert result.traceback == "Traceback..."


def test_hook_deferred_construction():
    """Test _HookDeferred constructs correctly."""
    result = _HookDeferred(
        hook_name="test_hook",
        reason="waiting for dependency",
    )
    assert result.kind == "deferred"
    assert result.hook_name == "test_hook"
    assert result.reason == "waiting for dependency"


def test_json_serialization_success():
    """Test _HookSuccess round-trips through JSON serialization."""
    result = _HookSuccess()
    json_data = to_jsonable_python(result)
    assert json_data == {"kind": "success"}

    json_str = json.dumps(json_data)
    parsed = json.loads(json_str)
    assert parsed == {"kind": "success"}


def test_json_serialization_skipped():
    """Test _HookSkipped round-trips through JSON serialization."""
    result = _HookSkipped(reason="guard failed")
    json_data = to_jsonable_python(result)
    assert json_data == {"kind": "skipped", "reason": "guard failed"}

    json_str = json.dumps(json_data)
    parsed = json.loads(json_str)
    assert parsed == {"kind": "skipped", "reason": "guard failed"}


def test_json_serialization_error():
    """Test _HookError round-trips through JSON serialization."""
    result = _HookError(
        hook_name="test_hook",
        exc_type="ValueError",
        message="error message",
        traceback="traceback...",
    )
    json_data = to_jsonable_python(result)
    expected = {
        "kind": "error",
        "hook_name": "test_hook",
        "exc_type": "ValueError",
        "message": "error message",
        "traceback": "traceback...",
    }
    assert json_data == expected

    json_str = json.dumps(json_data)
    parsed = json.loads(json_str)
    assert parsed == expected


def test_json_serialization_deferred():
    """Test _HookDeferred round-trips through JSON serialization."""
    result = _HookDeferred(
        hook_name="test_hook",
        reason="waiting",
    )
    json_data = to_jsonable_python(result)
    assert json_data == {
        "kind": "deferred",
        "hook_name": "test_hook",
        "reason": "waiting",
    }

    json_str = json.dumps(json_data)
    parsed = json.loads(json_str)
    assert parsed == {
        "kind": "deferred",
        "hook_name": "test_hook",
        "reason": "waiting",
    }


def test_wrap_hook_call_sync_success(mock_flow):
    """Test wrap_hook_call returns _HookSuccess for successful sync hook."""

    def successful_hook(ctx: Context) -> None:
        ctx.set_header("x-test", "value")

    wrapped = wrap_hook_call(successful_hook, hook_name="test_hook")
    ctx = Context.from_flow(mock_flow)
    result = wrapped(ctx)

    assert isinstance(result, _HookSuccess)
    assert result.kind == "success"


def test_wrap_hook_call_sync_error(mock_flow):
    """Test wrap_hook_call converts raising sync hook to _HookError."""

    def failing_hook(ctx: Context) -> None:
        raise ValueError("test error")

    wrapped = wrap_hook_call(failing_hook, hook_name="failing_hook")
    ctx = Context.from_flow(mock_flow)
    result = wrapped(ctx)

    assert isinstance(result, _HookError)
    assert result.kind == "error"
    assert result.hook_name == "failing_hook"
    assert result.exc_type == "ValueError"
    assert result.message == "test error"
    assert result.traceback is not None
    assert "ValueError: test error" in result.traceback


@pytest.mark.asyncio
async def test_wrap_hook_call_async_success(mock_flow):
    """Test wrap_hook_call returns _HookSuccess for successful async hook."""

    async def successful_async_hook(ctx: Context) -> None:
        ctx.set_header("x-test", "value")

    wrapped = wrap_hook_call(successful_async_hook, hook_name="test_hook")
    ctx = Context.from_flow(mock_flow)
    result = await wrapped(ctx)

    assert isinstance(result, _HookSuccess)
    assert result.kind == "success"


@pytest.mark.asyncio
async def test_wrap_hook_call_async_error(mock_flow):
    """Test wrap_hook_call converts raising async hook to _HookError."""

    async def failing_async_hook(ctx: Context) -> None:
        raise RuntimeError("async error")

    wrapped = wrap_hook_call(failing_async_hook, hook_name="failing_async_hook")
    ctx = Context.from_flow(mock_flow)
    result = await wrapped(ctx)

    assert isinstance(result, _HookError)
    assert result.kind == "error"
    assert result.hook_name == "failing_async_hook"
    assert result.exc_type == "RuntimeError"
    assert result.message == "async error"
    assert result.traceback is not None
    assert "RuntimeError: async error" in result.traceback


def test_unwrap_hook_result_success_no_raise():
    """Test unwrap_hook_result no-ops on success when raise_on_error=False."""
    result = _HookSuccess()
    unwrap_hook_result(result, raise_on_error=False)


def test_unwrap_hook_result_error_no_raise():
    """Test unwrap_hook_result no-ops on error when raise_on_error=False."""
    result = _HookError(
        hook_name="test_hook",
        exc_type="ValueError",
        message="error",
    )
    unwrap_hook_result(result, raise_on_error=False)


def test_unwrap_hook_result_error_with_raise():
    """Test unwrap_hook_result re-raises RuntimeError when raise_on_error=True."""
    result = _HookError(
        hook_name="test_hook",
        exc_type="ValueError",
        message="error message",
    )
    with pytest.raises(RuntimeError, match=r"Hook 'test_hook' failed: ValueError: error message"):
        unwrap_hook_result(result, raise_on_error=True)


def test_unwrap_hook_result_skipped_with_raise():
    """Test unwrap_hook_result no-ops on skipped even when raise_on_error=True."""
    result = _HookSkipped(reason="guard failed")
    unwrap_hook_result(result, raise_on_error=True)


def test_unwrap_hook_result_deferred_with_raise():
    """Test unwrap_hook_result no-ops on deferred even when raise_on_error=True."""
    result = _HookDeferred(hook_name="test_hook", reason="waiting")
    unwrap_hook_result(result, raise_on_error=True)


def test_executor_adds_success_result_to_metadata():
    """Test that executor records _HookSuccess in flow.metadata."""
    from ccproxy.pipeline.executor import PipelineExecutor
    from ccproxy.pipeline.hook import HookSpec

    def successful_hook(ctx: Context, params: dict) -> Context:
        return ctx

    flow = _make_flow()
    spec = HookSpec(
        name="test_hook",
        handler=successful_hook,
        reads=frozenset(),
        writes=frozenset(),
    )
    executor = PipelineExecutor(hooks=[spec])
    executor.execute(flow)

    assert "ccproxy.hook_results" in flow.metadata
    results = flow.metadata["ccproxy.hook_results"]
    assert len(results) == 1
    assert isinstance(results[0], _HookSuccess)


def test_executor_adds_error_result_on_failure():
    """Test that executor records _HookError when hook raises."""
    from ccproxy.pipeline.executor import PipelineExecutor
    from ccproxy.pipeline.hook import HookSpec

    def failing_hook(ctx: Context, params: dict) -> Context:
        raise ValueError("test error")

    flow = _make_flow()
    spec = HookSpec(
        name="failing_hook",
        handler=failing_hook,
        reads=frozenset(),
        writes=frozenset(),
    )
    executor = PipelineExecutor(hooks=[spec])
    executor.execute(flow)

    assert "ccproxy.hook_results" in flow.metadata
    results = flow.metadata["ccproxy.hook_results"]
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, _HookError)
    assert result.hook_name == "failing_hook"
    assert result.exc_type == "ValueError"
    assert result.message == "test error"


def test_executor_adds_skipped_result_for_guard():
    """Test that executor records _HookSkipped when guard returns False."""
    from ccproxy.pipeline.executor import PipelineExecutor
    from ccproxy.pipeline.hook import HookSpec

    def never_run_guard(ctx: Context) -> bool:
        return False

    def hook_handler(ctx: Context, params: dict) -> Context:
        return ctx

    flow = _make_flow()
    spec = HookSpec(
        name="skipped_hook",
        handler=hook_handler,
        guard=never_run_guard,
        reads=frozenset(),
        writes=frozenset(),
    )
    executor = PipelineExecutor(hooks=[spec])
    executor.execute(flow)

    assert "ccproxy.hook_results" in flow.metadata
    results = flow.metadata["ccproxy.hook_results"]
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, _HookSkipped)
    assert result.reason == "guard"


def test_executor_preserves_error_isolation():
    """Test that hook errors don't abort the DAG."""
    from ccproxy.pipeline.executor import PipelineExecutor
    from ccproxy.pipeline.hook import HookSpec

    def failing_hook(ctx: Context, params: dict) -> Context:
        raise RuntimeError("fail")

    def succeeding_hook(ctx: Context, params: dict) -> Context:
        return ctx

    flow = _make_flow()
    specs = [
        HookSpec(
            name="failing",
            handler=failing_hook,
            reads=frozenset(),
            writes=frozenset(),
        ),
        HookSpec(
            name="succeeding",
            handler=succeeding_hook,
            reads=frozenset(),
            writes=frozenset(),
        ),
    ]
    executor = PipelineExecutor(hooks=specs)
    executor.execute(flow)

    results = flow.metadata["ccproxy.hook_results"]
    assert len(results) == 2
    assert isinstance(results[0], _HookError)
    assert isinstance(results[1], _HookSuccess)


def _make_flow(body: dict | None = None):
    """Create a mock HTTPFlow for testing."""
    import json
    from unittest.mock import MagicMock

    flow = MagicMock()
    flow.id = "test-flow-id"
    flow.metadata = {}
    flow.request.content = json.dumps(
        body
        or {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).encode()
    flow.request.headers = {}
    flow.request.path = "/v1/messages"
    return flow


@pytest.fixture
def mock_flow():
    """Create a mock HTTPFlow for testing."""
    from unittest.mock import MagicMock

    from mitmproxy.connection import Server
    from mitmproxy.http import HTTPFlow, Request, Response
    from mitmproxy.proxy.mode_specs import ProxyMode

    flow = MagicMock(spec=HTTPFlow)
    flow.id = "test-flow-id"
    flow.metadata = {}

    request = MagicMock(spec=Request)
    request.method = "POST"
    request.scheme = "https"
    request.host = "api.anthropic.com"
    request.port = 443
    request.path = "/v1/messages"
    request.headers = {}
    request.content = b'{"model": "claude-3-5-sonnet-20241022", "messages": []}'
    flow.request = request

    response = MagicMock(spec=Response)
    response.status_code = 200
    response.headers = {}
    response.content = b"{}"
    flow.response = response

    server = MagicMock(spec=Server)
    server.address = ("api.anthropic.com", 443)
    flow.server_conn = server

    flow.mode = ProxyMode.parse("reverse:https://api.anthropic.com@443")

    return flow
