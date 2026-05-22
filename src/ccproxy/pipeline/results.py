"""Hook execution result types.

Discriminated union for hook execution outcomes, following the Temporal
pattern from pydantic-ai. Each variant is a frozen dataclass with a
``kind`` discriminator field.
"""

from __future__ import annotations

import inspect
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import Discriminator

from ccproxy.pipeline.context import Context


@dataclass(frozen=True)
class _HookSuccess:
    """Hook executed successfully."""

    kind: Literal["success"] = "success"


@dataclass(frozen=True)
class _HookSkipped:
    """Hook skipped due to guard or override."""

    reason: str
    """Reason the hook was skipped."""

    kind: Literal["skipped"] = "skipped"


@dataclass(frozen=True)
class _HookError:
    """Hook raised an exception."""

    hook_name: str
    """Name of the hook that failed."""

    exc_type: str
    """Exception type name."""

    message: str
    """Exception message."""

    traceback: str | None = None
    """Full traceback string if available."""

    kind: Literal["error"] = "error"


@dataclass(frozen=True)
class _HookDeferred:
    """Hook deferred for later execution."""

    hook_name: str
    """Name of the hook that was deferred."""

    reason: str
    """Reason for deferral."""

    kind: Literal["deferred"] = "deferred"


HookResult = Annotated[
    _HookSuccess | _HookSkipped | _HookError | _HookDeferred,
    Discriminator("kind"),
]


def wrap_hook_call(
    hook_callable: Callable[[Context], Any],
    *,
    hook_name: str,
) -> Callable[[Context], HookResult] | Callable[[Context], Awaitable[HookResult]]:
    """Wrap a hook callable to catch exceptions and return HookResult.

    Args:
        hook_callable: The hook function to wrap (sync or async).
        hook_name: Name of the hook for error reporting.

    Returns:
        A wrapped callable that returns HookResult instead of raising.
    """
    if inspect.iscoroutinefunction(hook_callable):

        async def async_wrapper(ctx: Context) -> HookResult:
            try:
                await hook_callable(ctx)
                return _HookSuccess()
            except Exception as e:
                return _HookError(
                    hook_name=hook_name,
                    exc_type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                )

        return async_wrapper
    else:

        def sync_wrapper(ctx: Context) -> HookResult:
            try:
                hook_callable(ctx)
                return _HookSuccess()
            except Exception as e:
                return _HookError(
                    hook_name=hook_name,
                    exc_type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                )

        return sync_wrapper


def unwrap_hook_result(result: HookResult, *, raise_on_error: bool = False) -> None:
    """Re-raise a synthetic RuntimeError when result is error and raise_on_error is True.

    Args:
        result: The HookResult to potentially unwrap.
        raise_on_error: If True, re-raise errors; otherwise no-op.

    Raises:
        RuntimeError: When raise_on_error is True and result is _HookError.
    """
    if raise_on_error and isinstance(result, _HookError):
        raise RuntimeError(f"Hook '{result.hook_name}' failed: {result.exc_type}: {result.message}")
