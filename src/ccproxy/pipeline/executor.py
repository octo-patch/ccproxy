"""Pipeline executor with DAG-ordered execution.

Executes hooks in dependency-safe order with override support.
"""

from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING, Any

from ccproxy.constants import OAuthConfigError
from ccproxy.pipeline.context import Context
from ccproxy.pipeline.dag import HookDAG
from ccproxy.pipeline.keyspace import extract_available_keys
from ccproxy.pipeline.overrides import (
    HookOverride,
    OverrideSet,
    extract_overrides_from_context,
)
from ccproxy.pipeline.results import (
    HookResult,
    _HookError,
    _HookSkipped,
    _HookSuccess,
)

if TYPE_CHECKING:
    from mitmproxy.http import HTTPFlow

    from ccproxy.pipeline.hook import HookSpec

logger = logging.getLogger(__name__)

_HOOK_RESULTS_KEY = "ccproxy.hook_results"


class PipelineExecutor:
    """Executes hooks in DAG-ordered sequence with override support."""

    def __init__(
        self,
        hooks: list[HookSpec],
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        self.dag = HookDAG(hooks)
        self.extra_params = extra_params or {}

        order = self.dag.execution_order
        logger.info("Pipeline execution order: %s", " → ".join(order))

        groups = self.dag.parallel_groups
        if any(len(g) > 1 for g in groups):
            logger.info(
                "Parallel execution groups: %s",
                [sorted(g) for g in groups],
            )

    def execute(self, flow: HTTPFlow) -> None:
        """Execute the hook pipeline against a mitmproxy flow.

        Builds a Context from the flow, runs all hooks in DAG order,
        then commits body mutations back to the flow. Header mutations
        are applied live during hook execution.

        Per-hook runtime validation: before each hook runs, checks that
        its declared ``reads`` are satisfied by either the initial flow
        vocabulary (request body keys, header names) or by earlier hooks'
        ``writes``. Missing reads emit a WARNING with the request path
        and trace_id, but do not block execution.

        Hook results (success, skip, error) are accumulated in
        flow.metadata["ccproxy.hook_results"] as a list of HookResult.
        """
        ctx = Context.from_flow(flow)
        flow.metadata["ccproxy.listener_format"] = ctx._listener_format.value

        # Initialize hook results storage
        if _HOOK_RESULTS_KEY not in flow.metadata:
            flow.metadata[_HOOK_RESULTS_KEY] = []

        available = extract_available_keys(ctx)

        overrides = extract_overrides_from_context(ctx.headers)
        if overrides.raw_header:
            logger.debug("Hook overrides: %s", overrides.raw_header)

        for hook_name in self.dag.execution_order:
            spec = self.dag.get_hook(hook_name)

            missing = spec.reads - available
            if missing:
                logger.warning(
                    "Hook '%s' reads unavailable keys: %s (path=%s, trace_id=%s)",
                    hook_name,
                    sorted(missing),
                    flow.request.path,
                    flow.id,
                )

            result = self._execute_hook(ctx, spec, overrides, self.extra_params)
            flow.metadata[_HOOK_RESULTS_KEY].append(result)

            # Only update available keys if hook succeeded
            if isinstance(result, _HookSuccess):
                available |= set(spec.writes)

        ctx.commit()

    def _execute_hook(
        self,
        ctx: Context,
        spec: HookSpec,
        overrides: OverrideSet,
        params: dict[str, Any],
    ) -> HookResult:
        """Execute a single hook with error isolation.

        Returns:
            HookResult indicating success, skip, or error.

        Raises:
            OAuthConfigError: Fatal error that should propagate.
        """
        hook_name = spec.name

        try:
            override = overrides.get_override(hook_name)

            if override == HookOverride.FORCE_SKIP:
                logger.debug("Hook '%s' skipped (override)", hook_name)
                return _HookSkipped(reason="override")

            if override != HookOverride.FORCE_RUN and not spec.should_run(ctx):
                logger.debug("Hook '%s' skipped (guard)", hook_name)
                return _HookSkipped(reason="guard")

            logger.debug("Executing hook '%s'", hook_name)
            spec.execute(ctx, params)
            return _HookSuccess()

        except OAuthConfigError:
            raise
        except Exception as e:
            logger.error(
                "Hook '%s' failed: %s: %s",
                hook_name,
                type(e).__name__,
                str(e),
            )
            return _HookError(
                hook_name=hook_name,
                exc_type=type(e).__name__,
                message=str(e),
                traceback=traceback.format_exc(),
            )

    def get_execution_order(self) -> list[str]:
        return self.dag.execution_order

    def get_parallel_groups(self) -> list[set[str]]:
        return self.dag.parallel_groups
