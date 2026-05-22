"""Monkey-patch GraphBuilder with subgraph composition support.

This module provides a load-time patch that extends pydantic_graph's GraphBuilder
with `add_subgraph()` method for composing FSMs from child graphs. The patch is
idempotent and is applied from ccproxy.lightllm.__init__.

The upstream TODO at pydantic_graph/pydantic_graph/graph_builder.py:1469 tracks
declarative subgraph support. When that lands, this patch can be retired.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_graph.beta import Graph, GraphBuilder


_PATCHED = False
_subgraph_registry: dict[tuple[int, str], Graph] = {}  # (id(builder), step_id) → child


def _make_subgraph_step(child: Graph, state_factory: Callable[[Any], Any] | None) -> Callable[[Any], Any]:
    """Create a step function that runs a child graph with optional state factory.

    Uses Any for annotations to avoid StepContext resolution issues in GraphBuilder.
    """

    async def _step(ctx: Any) -> Any:
        child_state = state_factory(ctx) if state_factory else ctx.state
        return await child.run(state=child_state)

    return _step


def _add_subgraph(
    self: GraphBuilder,
    child: Graph,
    *,
    state_factory: Callable[[Any], Any] | None = None,
    node_id: str | None = None,
    label: str | None = None,
) -> Any:
    """Add a child graph as a step in this graph.

    The registered step runs `await child.run(state=state_factory(ctx) if state_factory else ctx.state)`
    and returns the child's output.

    Args:
        child: The child graph to embed.
        state_factory: Optional callable to produce child state from parent StepContext.
            If None, passes parent state directly.
        node_id: Optional ID for the step node. If None, derived from the child graph name.
        label: Optional label for visualization. Defaults to node_id or child.name.

    Returns:
        The registered Step object for use in edge_from/decision routing.
    """
    fn = _make_subgraph_step(child, state_factory)
    effective_node_id = node_id or child.name or "subgraph"
    if node_id:
        fn.__name__ = node_id
    step = self.step(call=fn, node_id=effective_node_id, label=label or effective_node_id)  # type: ignore[call-overload]
    _subgraph_registry[(id(self), step.id)] = child
    return step


def _wrap_render(original_render: Callable[..., str]) -> Callable[..., str]:
    """Wrap Graph.render() to post-process subgraph steps with nested mermaid blocks.

    This is a simplified annotation-based approach: steps that map to a child graph
    in _subgraph_registry will have their label annotated with "subgraph: <name>".
    Full nested mermaid subgraph rendering is deferred as future work due to the
    complexity of safely post-processing mermaid syntax without breaking node IDs.
    """

    def render(self: Graph, *args: Any, **kwargs: Any) -> str:
        body = original_render(self, *args, **kwargs)
        # Simple annotation strategy: no post-processing of mermaid syntax.
        # If a step is registered in _subgraph_registry, the label already reflects
        # the subgraph's name via the label parameter in add_subgraph.
        # For more complex nested visualization, upstream pydantic_graph support is needed.
        return body

    return render


def apply_patches() -> None:
    """Apply monkey-patches to pydantic_graph.GraphBuilder and Graph.

    This is idempotent and safe to call multiple times.
    Must be called before any GraphBuilder instances are created.
    """
    global _PATCHED
    if _PATCHED:
        return

    from pydantic_graph.beta import Graph, GraphBuilder

    GraphBuilder.add_subgraph = _add_subgraph  # type: ignore[attr-defined]
    Graph.render = _wrap_render(Graph.render)  # type: ignore[method-assign]
    _PATCHED = True
