"""Tests for graph_ext monkey-patch functionality."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic_graph.beta import GraphBuilder

from ccproxy.lightllm.graph_ext import apply_patches


@dataclass
class ParentState:
    counter: int
    result: str | None = None


@dataclass
class ChildState:
    multiplier: int


@pytest.fixture(autouse=True)
def ensure_patched() -> None:
    """Ensure patches are applied before each test."""
    apply_patches()


def test_apply_patches_is_idempotent() -> None:
    """Calling apply_patches multiple times should not raise."""
    apply_patches()
    apply_patches()
    apply_patches()


def test_graphbuilder_has_add_subgraph_method() -> None:
    """After patching, GraphBuilder should have add_subgraph method."""
    builder = GraphBuilder(state_type=ParentState, output_type=str)
    assert hasattr(builder, "add_subgraph")
    assert callable(builder.add_subgraph)


async def test_subgraph_step_runs_child_graph() -> None:
    """A subgraph step should invoke the child graph and return its output."""
    # Build child graph that doubles the counter
    child_builder = GraphBuilder[ChildState, None, None, int](state_type=ChildState, output_type=int)

    @child_builder.step
    async def double_counter(ctx):
        return ctx.state.multiplier * 2

    child_builder.add(
        child_builder.edge_from(child_builder.start_node).to(double_counter),
        child_builder.edge_from(double_counter).to(child_builder.end_node),
    )

    child_graph = child_builder.build()

    # Build parent graph that uses child as a subgraph
    parent_builder = GraphBuilder[ParentState, None, None, str](state_type=ParentState, output_type=str)

    def state_factory(ctx):
        return ChildState(multiplier=ctx.state.counter)

    subgraph_step = parent_builder.add_subgraph(child_graph, state_factory=state_factory, node_id="double_via_child")

    @parent_builder.step
    async def format_result(ctx):
        doubled = ctx.inputs
        return f"Result: {doubled}"

    parent_builder.add(
        parent_builder.edge_from(parent_builder.start_node).to(subgraph_step),
        parent_builder.edge_from(subgraph_step).to(format_result),
        parent_builder.edge_from(format_result).to(parent_builder.end_node),
    )

    parent_graph = parent_builder.build()

    # Run parent graph
    result = await parent_graph.run(state=ParentState(counter=5))
    assert result == "Result: 10"


async def test_subgraph_without_state_factory() -> None:
    """Subgraph with no state_factory should receive parent state directly."""
    # Build child graph that reads parent state
    child_builder = GraphBuilder[ParentState, None, None, str](state_type=ParentState, output_type=str)

    @child_builder.step
    async def read_counter(ctx):
        return f"Counter was {ctx.state.counter}"

    child_builder.add(
        child_builder.edge_from(child_builder.start_node).to(read_counter),
        child_builder.edge_from(read_counter).to(child_builder.end_node),
    )

    child_graph = child_builder.build()

    # Build parent graph
    parent_builder = GraphBuilder[ParentState, None, None, str](state_type=ParentState, output_type=str)

    subgraph_step = parent_builder.add_subgraph(child_graph, node_id="read_child")

    parent_builder.add(
        parent_builder.edge_from(parent_builder.start_node).to(subgraph_step),
        parent_builder.edge_from(subgraph_step).to(parent_builder.end_node),
    )

    parent_graph = parent_builder.build()

    result = await parent_graph.run(state=ParentState(counter=42))
    assert result == "Counter was 42"


async def test_graph_render_includes_subgraph_annotation() -> None:
    """Graph.render() should produce valid mermaid output with subgraph steps."""
    # Build simple child
    child_builder = GraphBuilder[ChildState, None, None, int](state_type=ChildState, output_type=int)

    @child_builder.step
    async def child_step(ctx):
        return ctx.state.multiplier

    child_builder.add(
        child_builder.edge_from(child_builder.start_node).to(child_step),
        child_builder.edge_from(child_step).to(child_builder.end_node),
    )

    child_graph = child_builder.build()

    # Build parent with subgraph
    parent_builder = GraphBuilder[ParentState, None, None, int](state_type=ParentState, output_type=int)

    subgraph_step = parent_builder.add_subgraph(child_graph, node_id="embedded_child", label="EmbeddedChild")

    parent_builder.add(
        parent_builder.edge_from(parent_builder.start_node).to(subgraph_step),
        parent_builder.edge_from(subgraph_step).to(parent_builder.end_node),
    )

    parent_graph = parent_builder.build()

    # Render should not raise and should produce valid mermaid
    mermaid = parent_graph.render()
    assert isinstance(mermaid, str)
    assert len(mermaid) > 0
    # The subgraph step should appear with its label
    assert "EmbeddedChild" in mermaid or "embedded_child" in mermaid
