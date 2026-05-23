# ccproxy refactor — remaining items + verification suite

## Outstanding / deferred

- [x] **Phase H**: typed promotion via newer `ModelResponsePartsManager`
  API (1.99+). The boundary fix landed in
  `src/ccproxy/lightllm/adapters/_anthropic_envelope.py:_parse_tools`
  and `_openai_envelope.py:_parse_tools` — they now consult
  `_tool_kinds.ANTHROPIC_TYPED_TOOLS` / `OPENAI_TYPED_TOOLS` to set
  `ToolDefinition.tool_kind` from the wire `type` discriminator.
  Regression test at `tests/test_lightllm_graph_intake_anthropic.py`
  (`test_typed_search_tool_promotes_tool_call_part`) asserts a
  `web_search_20250305` tool flow promotes to `ToolSearchCallPart`.
- [x] **SSE intake decomposition into per-step subgraphs**
  (deferred Phase F Stages 2-5). Implemented via a temporary
  `GraphBuilder.add_subgraph` monkey-patch
  (`src/ccproxy/lightllm/graph/_subgraph_patch.py`) that tracks the
  upstream TODO at `pydantic_graph/graph_builder.py:1469`. Perplexity's
  142-line `_dispatch_one_event` is gone — replaced by a per-event
  subgraph that pops blocks one at a time and routes through three
  arms (plan / bare-markdown / diff-block). Google's
  `handle_generate_chunk` is gone — replaced by a per-chunk subgraph
  that classifies parts via a typed-marker decision. Outer topology
  for both is unchanged: events queue → dispatch via subgraph → loop.
  Patch is removable when pydantic-graph ships native subgraphs.
- [ ] **Push to `origin/dev`** (Kyle does manually). Now 22+N commits
  ahead (refactor + this PR's work).
- [ ] **Production rollout**: when ready, `nh os switch ~/.config/nixos`
  on gaiagear picks up the path-flake input automatically. Restart unit
  fires via `X-Restart-Triggers` on YAML change; otherwise
  `systemctl --user restart ccproxy`.

## Verification suite — perform against dev daemon (port 4001)

### Static gates

```bash
just up                                      # daemon
uv run pytest tests/ --no-cov -q             # expect 1659 passed
uv run mypy src/ccproxy                      # expect Success
uv run ruff check src/ccproxy                # expect All checks passed!
uv run pytest tests/ --no-cov -q \
  -W "error::DeprecationWarning:ccproxy" \
  -W "error::pydantic_graph.PydanticGraphDeprecationWarning"   # zero ccproxy/pydantic-graph deprecations
```

### Inspector smoke matrix

For each row: run the command, then `ccproxy flows compare --jq` on the
resulting /v1/messages or /chat/completions flow. Confirm 200 status,
non-empty response, and the forwarded body carries the expected shape.

| # | Listener | Upstream | Test command | What to verify |
|---|---|---|---|---|
| 1 | Anthropic | Anthropic | `ccproxy run --inspect -- claude --model haiku -p "2+2"` | Native passthrough — claude CLI baseline (always works) |
| 2 | Anthropic | Anthropic | `CCPROXY_BASE_URL=http://127.0.0.1:4001 uv run python docs/sdk/anthropic_sdk.py` | Shape stamps full Claude Code envelope (system + metadata + billing header + `?beta=true`). This was the 429 reproducer; now 200. |
| 3 | OpenAI | Anthropic | Use `docs/sdk/openai_sdk.py` (or equivalent OpenAI client → `:4001/v1/chat/completions` with `model=claude-...` + sentinel key) | Cross-format transform: OpenAI listener parses → IR → AnthropicAdapter.render to wire. Forwarded body should be Anthropic-shaped. |
| 4 | Anthropic | Anthropic | Multi-turn conversation (system prompt + 2 user turns) | System prompt survives shape's `prepend_shape:N` strategy |
| 5 | Anthropic | Anthropic | Tool use roundtrip (declare a tool, model calls it, send tool_result) | `tool_use_id` preserved; `tool_result.content` wrapped in `[{type: text, text: ...}]` array |
| 6 | Anthropic | Anthropic | Image content (BinaryContent or ImageUrl) | `media_type` preserved on round-trip, base64 inline data intact |
| 7 | Anthropic | Anthropic | Prompt caching: send same prefix twice with `cache_control` | Second request reports cache_read_input_tokens > 0 |
| 8 | Anthropic | DeepSeek (anthropic-compatible) | SDK call with `sk-ant-oat-ccproxy-deepseek` (if configured) | Routes to deepseek host, Anthropic wire format |
| 9 | Anthropic | ZAI (anthropic-compatible) | SDK call with `sk-ant-oat-ccproxy-zai` (if configured) | Routes to zai host, Anthropic wire format |
| 10 | OpenAI | OpenAI | OpenAI SDK call with `sk-ant-oat-ccproxy-openai` (if configured) | Native passthrough — no shape, no transform |
| 11 | OpenAI | Google/Gemini | Cross-format with `sk-ant-oat-ccproxy-gemini` | GoogleAdapter.render emits camelCase + generationConfig |
| 12 | OpenAI | Perplexity Pro | SDK call with `sk-ant-oat-ccproxy-perplexity_pro` | PerplexityAdapter.render emits 28-field payload |

### Flow inspection helpers

```bash
# List all /v1/messages flows
ccproxy flows list --jq 'map(select(.request.path | tostring | test("messages")))'

# Compare client vs forwarded for a specific flow
ccproxy flows compare --jq 'map(select(.id | startswith("PREFIX")))'

# Pull request body
ccproxy flows dump --jq 'map(select(.id == "FULL_ID"))'

# Tail log for hook activity / errors
ccproxy logs -n 100 | grep -iE "error|exception|shape|warning"

# Watch hook_results in real time
ccproxy logs -f | grep "hook_results"
```

### Negative-path / regression checks

- [ ] Send a request with NO Claude CLI UA via SDK → should get 200
  (shape masks identity)
- [ ] Send a request from `claude` CLI → should get 200 with
  `_ua_matches` triggering "skipping shaping" in logs
- [ ] Verify `ctx.invalidate_parsed()` is called by apply_shape: edit
  `apply_shape` to NOT invalidate, re-run SDK test → should reproduce 429
- [ ] OAuth 401 path: corrupt the cached token, send request, verify
  OAuthAddon refreshes and retries (1 retry, then succeeds)
- [ ] Capacity 429 path: deliberately overload (or mock) → verify
  GeminiAddon capacity fallback walks the fallback_models chain

### Visual / mermaid sanity

```bash
# Print every built FSM as mermaid to confirm no orphan nodes
uv run python -c "
from ccproxy.lightllm.graph.anthropic_intake import _intake_graph as ai
from ccproxy.lightllm.graph.anthropic_render import _render_graph as ar
from ccproxy.lightllm.graph.openai_intake import _intake_graph as oi
from ccproxy.lightllm.graph.openai_render import _render_graph as or_
from ccproxy.lightllm.graph.google_intake import _intake_graph as gi
from ccproxy.lightllm.graph.perplexity_intake import _intake_graph as pi
for name, g in [('anthropic_intake', ai), ('anthropic_render', ar),
                ('openai_intake', oi), ('openai_render', or_),
                ('google_intake', gi), ('perplexity_intake', pi)]:
    print(f'=== {name} ===')
    print(g.render(title=name, direction='LR'))
    print()
"
```

### Final acceptance

- [ ] All rows in the matrix pass
- [ ] No new `ERROR` or `Traceback` in `ccproxy logs` after the run
- [ ] `git log` clean — no unintended commits
- [ ] `nh os switch ~/.config/nixos` (production rollout) when ready

