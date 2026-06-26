"""Mermaid render smoke test for every lightllm graph (outer FSM + inner subgraphs).

Each `*_intake.py` / `*_render.py` module builds one or more module-level
`pydantic_graph` `Graph` objects: the outer FSM plus the inner-dispatch
subgraphs that graph-ify per-event variant dispatch. Rendering every one
confirms the topology is well-formed — a mis-wired edge or an unmatched
decision branch fails at ``render()`` time — and exercises the GraphBuilder
mermaid benefit that the FSMs otherwise leave uncollected.

This is the visualization sanity check referenced in ``docs/lightllm.md``.
"""

from __future__ import annotations

import importlib

import pytest
from pydantic_graph.graph_builder import Graph

GRAPH_MODULES: list[str] = [
    "ccproxy.lightllm.graph.anthropic_intake",
    "ccproxy.lightllm.graph.openai_intake",
    "ccproxy.lightllm.graph.openai_responses_intake",
    "ccproxy.lightllm.graph.google_intake",
    "ccproxy.lightllm.graph.perplexity_intake",
    "ccproxy.lightllm.graph.anthropic_render",
    "ccproxy.lightllm.graph.openai_render",
    "ccproxy.lightllm.graph.openai_responses_render",
]


def _module_graphs(module_name: str) -> list[tuple[str, Graph]]:
    """Every module-level built ``Graph`` in ``module_name``."""
    module = importlib.import_module(module_name)
    return [(name, obj) for name, obj in vars(module).items() if isinstance(obj, Graph)]


@pytest.mark.parametrize("module_name", [pytest.param(m, id=m.rsplit(".", 1)[-1]) for m in GRAPH_MODULES])
def test_module_graphs_render_to_mermaid(module_name: str) -> None:
    """Every built graph in the module renders to a non-empty named stateDiagram-v2.

    The outer FSM plus at least one inner-dispatch subgraph must be present —
    inner per-event dispatch is graph-ified, so each module exposes >= 2 graphs.
    """
    graphs = _module_graphs(module_name)
    found = ", ".join(name for name, _ in graphs)
    assert len(graphs) >= 2, f"{module_name}: expected outer FSM + inner subgraph(s), found {len(graphs)}: {found}"
    for attr_name, graph in graphs:
        assert graph.name, f"{module_name}.{attr_name}: built graph has no name (needed for mermaid/tracing labels)"
        diagram = graph.render()
        assert "stateDiagram-v2" in diagram, f"{module_name}.{attr_name}: render() produced no stateDiagram-v2"
        assert "[*] -->" in diagram, f"{module_name}.{attr_name}: render() has no start-node edge"
