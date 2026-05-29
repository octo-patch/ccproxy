#!/usr/bin/env python3
"""Probe: discover the SSE wire format for Perplexity's server-side MCP tools.

The user has connected a GitHub MCP server to their perplexity.ai account
via the connectors UI. When a query needs GitHub data, Perplexity's
backend exposes those MCP tools to the model. We send a question that
should trigger an MCP tool call, then dump the SSE stream to see what
block types and ``intended_usage`` values appear.

This does NOT send OpenAI ``tools=[...]`` — that's the user-defined-tools
path (which is currently broken on frontier models). We want the
*server-side* MCP path.

Usage:
    uv run python docs/examples/pplx_mcp_probe.py
    ccproxy flows list                  # find the flow id
    ccproxy flows dump > /tmp/probe.har # raw SSE captured

Then `pplx_mcp_probe_analyze.py` (or manual jq) extracts unique
``intended_usage`` values from the SSE.
"""

import os

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

console = Console()
err_console = Console(stderr=True)

BASE_URL = f"{os.environ.get('CCPROXY_BASE_URL', 'http://127.0.0.1:4000')}/v1"
SENTINEL_KEY = "sk-ant-oat-ccproxy-perplexity_pro"
MODEL = os.environ.get("CCPROXY_PPLX_MODEL", "anthropic/claude-sonnet-4.6")

# Vary so we don't hit Mode 2 L1 cache (which reuses a thread across runs).
NONCE = os.urandom(4).hex()


def main() -> None:
    console.print(Panel(f"[cyan]MCP probe — model={MODEL}[/cyan]", border_style="blue"))
    console.print(f"[yellow]Base URL:[/yellow] {BASE_URL}")

    client = OpenAI(base_url=BASE_URL, api_key=SENTINEL_KEY)

    user_text = (
        f"[probe {NONCE}] Use the GitHub connector to list my five most recent "
        "pull requests across all my repositories. For each, include the PR title, "
        "the repository name, the PR number, and the current state (open/closed/merged)."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": user_text}],
        stream=False,
    )

    choice = response.choices[0]
    console.print("\n[green]Content:[/green]")
    console.print(choice.message.content)
    console.print(f"\n[dim]finish_reason:[/dim] [bold]{choice.finish_reason}[/bold]")
    slug = getattr(response, "pplx_thread_url_slug", None)
    if slug:
        console.print(f"[dim]slug:[/dim] {slug}")
    if getattr(choice.message, "tool_calls", None):
        console.print("\n[dim]tool_calls (from our parser):[/dim]")
        for tc in choice.message.tool_calls:
            console.print(f"  - {tc.function.name}({tc.function.arguments})")


if __name__ == "__main__":
    main()
