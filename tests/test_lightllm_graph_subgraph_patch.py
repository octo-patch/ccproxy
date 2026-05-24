"""Tests for the :class:`GraphBuilder.add_subgraph` monkey-patch.

Covers:

- ``add_subgraph`` registers a callable :class:`Step` usable in
  ``edge_from(...).to(...)``.
- State mutations performed inside the subgraph are visible to the
  parent graph after the subgraph step returns (shared ``StateT``).
- A subgraph's typed output threads through to the parent's downstream
  node — the parent step receives the subgraph's return value as its
  input.

The patch itself lives in
:mod:`ccproxy.lightllm.graph._subgraph_patch`; importing it once installs
``GraphBuilder.add_subgraph``. Subsequent test modules see the method
without re-importing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pydantic_graph import GraphBuilder, Step, StepContext

import ccproxy.lightllm.graph._subgraph_patch  # noqa: F401  — installs add_subgraph

# ---------------------------------------------------------------------------
# Shared state for the composition tests
# ---------------------------------------------------------------------------


@dataclass
class _State:
    """Mutable state shared between parent and subgraph in the tests."""

    outer_log: list[str] = field(default_factory=list)
    inner_log: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Trigger:
    """Input envelope for the inner subgraph."""

    payload: str


@dataclass(frozen=True)
class _SubgraphResult:
    """Typed output of the inner subgraph."""

    echo: str
    count: int


# ---------------------------------------------------------------------------
# Test 1 — add_subgraph returns a Step usable in edges
# ---------------------------------------------------------------------------


def test_add_subgraph_returns_step() -> None:
    """``add_subgraph`` registers a :class:`Step` so the result is wireable."""

    sub: GraphBuilder[_State, None, _Trigger, _SubgraphResult] = GraphBuilder(
        state_type=_State,
        input_type=_Trigger,
        output_type=_SubgraphResult,
    )

    @sub.step
    async def echo_step(ctx: StepContext[_State, None, _Trigger]) -> _SubgraphResult:
        return _SubgraphResult(echo=ctx.inputs.payload, count=1)

    sub.add(sub.edge_from(sub.start_node).to(echo_step))
    sub.add(sub.edge_from(echo_step).to(sub.end_node))
    sub_graph = sub.build()

    parent: GraphBuilder[_State, None, _Trigger, _SubgraphResult] = GraphBuilder(
        state_type=_State,
        input_type=_Trigger,
        output_type=_SubgraphResult,
    )
    sub_step = parent.add_subgraph(sub_graph, label="echo_subgraph")  # ty: ignore[unresolved-attribute]

    assert isinstance(sub_step, Step)
    assert sub_step.label == "echo_subgraph"
    assert sub_step.id.startswith("subgraph_")


# ---------------------------------------------------------------------------
# Test 2 — state mutations from inner are visible to parent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subgraph_shared_state_mutation_visible_to_parent() -> None:
    """Inner steps mutating the shared state instance are observed by the parent."""

    sub: GraphBuilder[_State, None, _Trigger, _SubgraphResult] = GraphBuilder(
        state_type=_State,
        input_type=_Trigger,
        output_type=_SubgraphResult,
    )

    @sub.step
    async def inner_mutate(ctx: StepContext[_State, None, _Trigger]) -> _SubgraphResult:
        ctx.state.inner_log.append(f"inner saw payload={ctx.inputs.payload}")
        return _SubgraphResult(echo=ctx.inputs.payload, count=len(ctx.state.inner_log))

    sub.add(sub.edge_from(sub.start_node).to(inner_mutate))
    sub.add(sub.edge_from(inner_mutate).to(sub.end_node))
    sub_graph = sub.build()

    parent: GraphBuilder[_State, None, _Trigger, _SubgraphResult] = GraphBuilder(
        state_type=_State,
        input_type=_Trigger,
        output_type=_SubgraphResult,
    )
    sub_step = parent.add_subgraph(sub_graph)  # ty: ignore[unresolved-attribute]

    @parent.step
    async def parent_after(ctx: StepContext[_State, None, _SubgraphResult]) -> _SubgraphResult:
        ctx.state.outer_log.append(f"parent saw inner_log_len={len(ctx.state.inner_log)} echo={ctx.inputs.echo}")
        return ctx.inputs

    parent.add(parent.edge_from(parent.start_node).to(sub_step))
    parent.add(parent.edge_from(sub_step).to(parent_after))
    parent.add(parent.edge_from(parent_after).to(parent.end_node))
    parent_graph = parent.build()

    state = _State()
    result = await parent_graph.run(state=state, inputs=_Trigger(payload="hello"))

    assert result == _SubgraphResult(echo="hello", count=1)
    assert state.inner_log == ["inner saw payload=hello"]
    assert state.outer_log == ["parent saw inner_log_len=1 echo=hello"]


# ---------------------------------------------------------------------------
# Test 3 — subgraph's typed output threads through to the parent's next node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subgraph_output_threads_to_parent_downstream() -> None:
    """The parent step downstream of the subgraph receives the subgraph's output as input."""

    sub: GraphBuilder[_State, None, _Trigger, _SubgraphResult] = GraphBuilder(
        state_type=_State,
        input_type=_Trigger,
        output_type=_SubgraphResult,
    )

    @sub.step
    async def inner(ctx: StepContext[_State, None, _Trigger]) -> _SubgraphResult:
        return _SubgraphResult(echo=ctx.inputs.payload.upper(), count=len(ctx.inputs.payload))

    sub.add(sub.edge_from(sub.start_node).to(inner))
    sub.add(sub.edge_from(inner).to(sub.end_node))
    sub_graph = sub.build()

    parent: GraphBuilder[_State, None, _Trigger, str] = GraphBuilder(
        state_type=_State,
        input_type=_Trigger,
        output_type=str,
    )
    sub_step = parent.add_subgraph(sub_graph)  # ty: ignore[unresolved-attribute]

    @parent.step
    async def stringify(ctx: StepContext[_State, None, _SubgraphResult]) -> str:
        return f"{ctx.inputs.echo}|{ctx.inputs.count}"

    parent.add(parent.edge_from(parent.start_node).to(sub_step))
    parent.add(parent.edge_from(sub_step).to(stringify))
    parent.add(parent.edge_from(stringify).to(parent.end_node))
    parent_graph = parent.build()

    result = await parent_graph.run(state=_State(), inputs=_Trigger(payload="abc"))
    assert result == "ABC|3"
