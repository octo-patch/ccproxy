"""Design-render tool: every lightllm intake + render graph → mermaid.

Single source of truth — imports the built, behavior-bearing graphs from the
per-provider modules and emits pydantic-graph's native ``render()`` mermaid.
Subgraphs (composed via :mod:`ccproxy.lightllm.graph._subgraph_patch`, which the
provider modules install at import) appear inside their parent as labeled step
nodes (``subgraph_<name>: <label>``) and are each rendered on their own below the
parent so the inner structure is visible too.

Run:

    uv run python -m ccproxy.lightllm.graph.render_graphs              # to stdout
    uv run python -m ccproxy.lightllm.graph.render_graphs --markdown  # ``` fenced
"""

from __future__ import annotations

import sys
from typing import Any

from ccproxy.lightllm.graph import (
    anthropic_intake,
    anthropic_render,
    google_intake,
    openai_conversations_intake,
    openai_intake,
    openai_render,
    openai_responses_intake,
    openai_responses_render,
    perplexity_intake,
)

# (kind, provider, [(label, graph), …]) — the main graph first, then its subgraphs.
_GROUPS: list[tuple[str, str, list[tuple[str, Any]]]] = [
    (
        "intake",
        "Anthropic / DeepSeek / Z.ai",
        [
            ("Anthropic intake", anthropic_intake._intake_graph),
            ("↳ block_start subgraph", anthropic_intake._block_start_graph),
            ("↳ block_delta subgraph", anthropic_intake._block_delta_graph),
        ],
    ),
    (
        "intake",
        "OpenAI Chat",
        [
            ("OpenAI Chat intake", openai_intake._intake_graph),
            ("↳ tool_calls subgraph", openai_intake._tool_calls_graph),
        ],
    ),
    (
        "intake",
        "OpenAI Responses",
        [
            ("OpenAI Responses intake", openai_responses_intake._intake_graph),
            ("↳ item_added subgraph", openai_responses_intake._item_added_graph),
            ("↳ item_done subgraph", openai_responses_intake._item_done_graph),
        ],
    ),
    (
        "intake",
        "Google / Gemini / Vertex",
        [
            ("Google intake", google_intake._intake_graph),
            ("↳ chunk_dispatch subgraph", google_intake._chunk_dispatch_graph),
        ],
    ),
    (
        "intake",
        "Perplexity",
        [
            ("Perplexity intake", perplexity_intake._intake_graph),
            ("↳ event_dispatch subgraph", perplexity_intake._event_dispatch_graph),
        ],
    ),
    (
        "intake",
        "OpenAI Conversations (ChatGPT)",
        [
            ("OpenAI Conversations intake", openai_conversations_intake._intake_graph),
        ],
    ),
    (
        "render",
        "Anthropic Messages",
        [
            ("Anthropic render", anthropic_render._render_graph),
            ("↳ part_start subgraph", anthropic_render._part_start_graph),
            ("↳ part_delta subgraph", anthropic_render._part_delta_graph),
        ],
    ),
    (
        "render",
        "OpenAI Chat Completions",
        [
            ("OpenAI Chat render", openai_render._render_graph),
            ("↳ part_start subgraph", openai_render._part_start_graph),
            ("↳ part_delta subgraph", openai_render._part_delta_graph),
        ],
    ),
    (
        "render",
        "OpenAI Responses",
        [
            ("OpenAI Responses render", openai_responses_render._render_graph),
            ("↳ part_start subgraph", openai_responses_render._part_start_graph),
            ("↳ part_delta subgraph", openai_responses_render._part_delta_graph),
        ],
    ),
]


def render_all(*, markdown: bool = False) -> str:
    """Return every graph's mermaid, grouped by provider. Markdown fences each block."""
    out: list[str] = []
    for kind, provider, graphs in _GROUPS:
        out.append(f"{'## ' if markdown else '# '}{kind.upper()} — {provider}")
        for label, graph in graphs:
            body = graph.render(title=label, direction="LR")
            out.append(f"\n```mermaid\n{body}\n```" if markdown else f"\n{body}")
        out.append("")
    return "\n".join(out)


def main() -> None:
    print(render_all(markdown="--markdown" in sys.argv[1:]))


if __name__ == "__main__":
    main()
