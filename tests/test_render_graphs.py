"""The graph-render tool renders every intake + render FSM to valid mermaid."""

from __future__ import annotations

from ccproxy.lightllm.graph.render_graphs import _GROUPS, render_all


def test_render_all_emits_mermaid_for_every_graph() -> None:
    out = render_all(markdown=True)
    expected_blocks = sum(len(graphs) for _, _, graphs in _GROUPS)
    assert expected_blocks == 22  # 9 main graphs + 13 subgraphs
    assert out.count("```mermaid") == expected_blocks
    assert out.count("stateDiagram-v2") == expected_blocks


def test_render_all_covers_each_provider_intake_and_render() -> None:
    out = render_all()
    for provider in ("Anthropic", "OpenAI Chat", "OpenAI Responses", "Google", "Perplexity", "OpenAI Conversations"):
        assert provider in out
    # the OpenAI Conversations intake carries its handoff sub-decision branch
    assert "handle_handoff_detected" in out
    # renders expose the part_start / part_delta subgraphs
    assert "part_start" in out
    assert "part_delta" in out
