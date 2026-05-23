"""Monkey-patch :class:`pydantic_graph.GraphBuilder` with subgraph composition.

Upstream TODO at ``pydantic_graph/graph_builder.py:1469``::

    # TODO(DavidM): Support adding subgraphs; I think this behaves like a step
    # with the same inputs/outputs but gets rendered as a subgraph in mermaid

Importing this module installs :meth:`GraphBuilder.add_subgraph`. Delete this
file and remove its imports the day ``pydantic_graph`` ships native subgraph
composition; the call sites should work unchanged (or trivially adapt if
upstream picks a different method name).

The patched method wraps a built :class:`pydantic_graph.graph_builder.Graph`
in a synthetic :class:`pydantic_graph.Step` whose body awaits
``subgraph.run(state=ctx.state, deps=ctx.deps, inputs=ctx.inputs)``. The
returned ``Step`` is usable in ``edge_from(...).to(...)`` like any other
step the builder produces. Shared ``StateT``/``DepsT`` flow through
unchanged — the inner graph sees and mutates the same state instance as
the parent, which is how Phase F preserves cross-block invariants like
``state.answer_seen`` prefix accumulation.

Sequencing: the subgraph runs to completion before the outer step's
downstream edges fire. No fork/parallel semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_graph import GraphBuilder, Step, StepContext
from pydantic_graph.graph_builder import Graph

if TYPE_CHECKING:
    from typing import TypeVar

    StateT = TypeVar("StateT")
    DepsT = TypeVar("DepsT")
    SubInputT = TypeVar("SubInputT")
    SubOutputT = TypeVar("SubOutputT")


def _add_subgraph(
    self: GraphBuilder[object, object, object, object],
    subgraph: Graph[object, object, object, object],
    *,
    node_id: str | None = None,
    label: str | None = None,
) -> Step[object, object, object, object]:
    """Register ``subgraph`` as a composable step inside this builder.

    Args:
        subgraph: A built :class:`Graph` whose ``state_type``/``deps_type``
            match this builder's. Its ``input_type`` becomes the new step's
            input type; its ``output_type`` becomes the step's output type.
        node_id: Optional override for the step's node id. Defaults to
            ``"subgraph_" + subgraph.name``.
        label: Optional human-readable label rendered in mermaid output.

    Returns:
        A :class:`Step` referencing the subgraph. Use it in
        ``edge_from(...).to(...)`` like any other step.
    """

    async def _run_subgraph(ctx: StepContext[object, object, object]) -> object:
        return await subgraph.run(
            state=ctx.state,
            deps=ctx.deps,
            inputs=ctx.inputs,
            infer_name=False,
        )

    resolved_id = node_id or f"subgraph_{subgraph.name or 'unnamed'}"
    return self.step(call=_run_subgraph, node_id=resolved_id, label=label)


GraphBuilder.add_subgraph = _add_subgraph  # ty: ignore[unresolved-attribute]
