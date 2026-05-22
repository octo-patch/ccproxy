# lightllm — wire translation layer

`ccproxy.lightllm` is the IR ↔ wire translation layer. It is what turns an
incoming request body (Anthropic Messages, OpenAI Chat Completions) into an
intermediate representation that ccproxy's hook pipeline can manipulate, and
back into a request body for whatever upstream provider the router resolves
to (Anthropic, OpenAI, Google Gemini, Perplexity Pro, plus the
Anthropic-compatible forks DeepSeek and ZAI).

Today it is **bi-modal**: the request side is fully FSM-based using
`pydantic_graph.beta.GraphBuilder`, and the response side is still
hand-rolled stateful classes (with LiteLLM doing some of the lifting). The
response-side migration is planned in `nextplan.md`; the end state is full
symmetry — same FSM idiom in both directions and `litellm` removed from
`pyproject.toml`.

This doc covers what's currently shipping. Read `nextplan.md` for what
changes next.

---

## Architecture

### The system at a glance

```
Client                              ccproxy                                Provider
  │                                    │                                      │
  │── REQUEST (listener wire) ────────▶│                                      │
  │                                    │  ┌─────────────────────────────┐     │
  │                                    │  │ Context.from_flow(flow)     │     │
  │                                    │  │   ↓                         │     │
  │                                    │  │ Context.parse_sync()        │     │
  │                                    │  │   → _run_coro_sync(...)     │     │
  │                                    │  │     ↓                       │     │
  │                                    │  │   await dispatch_load(      │     │
  │                                    │  │     body, listener_format=) │     │
  │                                    │  │     ↓                       │     │
  │                                    │  │   ParsedRequest (IR)        │     │
  │                                    │  └──────────┬──────────────────┘     │
  │                                    │             ↓                        │
  │                                    │  ┌──────────────────────┐            │
  │                                    │  │ Pipeline hooks (DAG) │            │
  │                                    │  └──────────┬───────────┘            │
  │                                    │             ↓                        │
  │                                    │  ┌──────────────────────────────┐    │
  │                                    │  │ dispatch_dump_sync(          │    │
  │                                    │  │   parsed, provider=)         │    │
  │                                    │  │   → _run_coro_sync(...)      │    │
  │                                    │  │     ↓                        │    │
  │                                    │  │   await dispatch_dump(...)   │    │
  │                                    │  │     ↓                        │    │
  │                                    │  │   provider wire bytes ──────────▶│
  │                                    │  └──────────────────────────────┘    │
  │                                    │                                      │
  │                                    │◀── provider wire (buffered or SSE) ──│
  │                                    │  ┌──────────────────────────────┐    │
  │                                    │  │ response/intake_<provider>.py│    │
  │                                    │  │   stateful, hand-rolled,     │    │
  │                                    │  │   drives ModelResponseParts… │    │
  │                                    │  │   ↓ ModelResponseStreamEvent │    │
  │                                    │  │ response/render_<listener>.py│    │
  │                                    │  │   ↓                          │    │
  │                                    │  │ listener wire bytes          │    │
  │◀── RESPONSE (listener wire) ───────│  └──────────────────────────────┘    │
  │                                    │                                      │
```

The thick line between the two halves is `pydantic_ai.messages.ModelMessage`
(and `ParsedRequest`) — the canonical IR that the pipeline hooks operate on.

### Module layout

```
src/ccproxy/lightllm/
├── parsed.py             ParsedRequest dataclass, ListenerFormat enum
├── registry.py           Provider name → BaseConfig resolver (local + LiteLLM)
├── dispatch.py           [LiteLLM-mediated response transform + Gemini req
│                         transform; scheduled for replacement, see nextplan.md]
├── context_cache.py      [Gemini cachedContents API; scheduled for replacement]
├── noop_logging.py       [LiteLLM Logging stub; scheduled for deletion]
├── pplx.py               Perplexity Pro BaseConfig subclass + iterator
├── pplx_steps.py         Perplexity step trail renderer
├── pplx_threads.py       Perplexity thread continuation helpers
│
├── graph/                ← REQUEST-SIDE FSM (canonical)
│   ├── __init__.py       dispatch_load, dispatch_dump, dispatch_dump_sync
│   ├── anthropic_dump.py IR → Anthropic Messages wire
│   ├── anthropic_load.py Anthropic Messages wire → IR
│   ├── openai_dump.py    IR → OpenAI Chat Completions wire
│   ├── openai_load.py    OpenAI Chat Completions wire → IR
│   ├── google_dump.py    IR → Google Gemini generateContent (wraps GoogleModel)
│   └── perplexity_dump.py IR → Perplexity Pro wire (wraps pplx.py helpers)
│
└── response/             ← RESPONSE-SIDE (hand-rolled; FSM migration pending)
    ├── intake.py         ResponseIntake protocol
    ├── intake_anthropic.py  Anthropic Messages SSE → IR events
    ├── intake_openai.py     OpenAI Chat SSE → IR events
    ├── intake_google.py     Google streamGenerateContent → IR events (NOT WIRED)
    ├── intake_perplexity.py Perplexity SSE → IR events
    ├── render.py         ResponseRender protocol
    ├── render_anthropic.py  IR events → Anthropic Messages SSE
    ├── render_openai.py     IR events → OpenAI Chat Completions SSE
    ├── pipeline.py       SsePipeline (sync mitmproxy.stream callable)
    └── buffered.py       Buffered (non-streaming) wrapper
```

### Bi-modal split — why and where

The request side migrated to a `pydantic-graph` FSM in commit
`refactor(ccproxy): migrate lightllm wire layer to pydantic-graph FSM` and
then to the `GraphBuilder` API in `4dd9765` / `d6007ea`. The response side
predates both and still uses hand-rolled stateful classes + LiteLLM's
per-provider iterators.

Why the split exists today:

1. **Cross-format request transform is the architectural pain.** Before the
   FSM, the outbound renderers instantiated `AnthropicModel` / `OpenAIChatModel`
   / `GoogleModel` from pydantic-ai with a fake provider client that raised a
   `CaptureSentinel` exception to extract the kwargs that would have hit the
   SDK. Brittle, abused control flow. The FSM rewrite directly emits typed
   SDK TypedDicts (`anthropic.types.beta.BetaMessageParam`,
   `openai.types.chat.ChatCompletionMessageParam`, etc.) — no capture, no
   exception flow.

2. **Response transform is mechanical conversion**, and pydantic-ai's
   `ModelResponsePartsManager` plus LiteLLM's per-provider chunk parsers were
   already doing the work correctly. The hand-rolled intake/render classes
   in `response/` are imperative but not architecturally smelly the way
   `CaptureSentinel` was. Replacing them is symmetry work, not bug-fix work.

The plan in `nextplan.md` describes the response-side migration. After it
lands, `dispatch.py`, `context_cache.py`, `noop_logging.py`, and the
`pplx.py` LiteLLM inheritance all delete; the response/ subpackage is
replaced by `lightllm/graph/*_intake.py` + `*_render.py`; `litellm` is
removed from `pyproject.toml`.

---

## The IR

### `ParsedRequest` — the request envelope

`src/ccproxy/lightllm/parsed.py`:

```python
@dataclass(frozen=True)
class ParsedRequest:
    model: str                            # model name from the listener body
    messages: list[ModelMessage]          # pydantic-ai IR conversation
    request_parameters: ModelRequestParameters  # tools, output config
    settings: ModelSettings               # max_tokens, temperature, top_p, ...
    stream: bool = False                  # listener requested SSE
    raw_extras: dict[str, Any] = field(default_factory=dict)
```

`raw_extras` is the load-bearing field for round-trip fidelity (see
"raw_extras contract" below).

### `ModelMessage` — the conversation IR

From `pydantic_ai.messages`. Each message is either:

* **`ModelRequest(parts=[...])`** — a user turn (or system turn). Parts:
  - `SystemPromptPart(content: str)`
  - `UserPromptPart(content: str | list[UserContent])` where `UserContent`
    is one of `str`, `BinaryContent`, `ImageUrl`, `DocumentUrl`, `AudioUrl`,
    `UploadedFile`, `CachePoint`
  - `ToolReturnPart(tool_name, content, tool_call_id, outcome=)` — a
    tool-result message
  - `RetryPromptPart(...)` — synthetic retry prompts

* **`ModelResponse(parts=[...])`** — an assistant turn. Parts:
  - `TextPart(content)`
  - `ToolCallPart(tool_name, args, tool_call_id)`
  - `ThinkingPart(content, signature, id=)` — including
    `id="redacted_thinking"` for opaque ciphertext

The conversation is a flat `list[ModelMessage]`; multi-turn ordering is
position-significant.

### `ListenerFormat` — what the client sent

```python
class ListenerFormat(str, Enum):
    UNKNOWN = "unknown"
    ANTHROPIC_MESSAGES = "anthropic_messages"   # /v1/messages
    OPENAI_CHAT = "openai_chat"                 # /v1/chat/completions
```

Pinned at `Context` construction from path + headers. Drives the choice of
inbound parser (`dispatch_load`). The **provider** the request routes to is
a separate decision (made by the transform router via sentinel-key or
`TransformOverride` rule); the listener format is purely "what did the
client send."

---

## The FSM pattern

Every file under `lightllm/graph/*_dump.py` and `*_load.py` (except the
google/perplexity wrappers) follows the same shape. Reading
`anthropic_dump.py` end-to-end is the fastest way to understand the
pattern.

### Anatomy of one FSM

```python
from pydantic_graph.beta import GraphBuilder, StepContext

# 1. State — a mutable dataclass carrying everything the FSM needs across steps.
@dataclass
class AnthropicDumpState:
    queue: deque[Any] = field(default_factory=deque)
    blocks: list[BetaContentBlockParam] = field(default_factory=list)
    last_emitted_block: BetaContentBlockParam | None = None

# 2. End-of-graph sentinel — a marker class routed to a terminal step.
class _DumpDone:
    """Marker returned when the queue is exhausted."""

# 3. GraphBuilder — the type parameters describe the FSM's runtime signature.
_g: GraphBuilder[AnthropicDumpState, None, None, list[BetaContentBlockParam]] = GraphBuilder(
    state_type=AnthropicDumpState,
    output_type=list[BetaContentBlockParam],
)

# 4. Router step — pops the next item OR signals done.
@_g.step
async def take_next(ctx: StepContext[AnthropicDumpState, None, None]) -> Any:
    if not ctx.state.queue:
        return _DumpDone()
    return ctx.state.queue.popleft()

# 5. Per-type handler steps — one per IR-part type.
@_g.step
async def parse_text(ctx: StepContext[AnthropicDumpState, None, str]) -> None:
    block: BetaTextBlockParam = {"type": "text", "text": ctx.inputs}
    ctx.state.blocks.append(block)
    ctx.state.last_emitted_block = block

@_g.step
async def apply_cache(ctx: StepContext[AnthropicDumpState, None, CachePoint]) -> None:
    if ctx.state.last_emitted_block is not None:
        cast(dict, ctx.state.last_emitted_block)["cache_control"] = {
            "type": "ephemeral", "ttl": ctx.inputs.ttl,
        }

# (... per-type steps for BinaryContent, ImageUrl, ToolReturnPart, etc.)

# 6. Terminal step — pulls the result out of state and hands it to end_node.
@_g.step
async def emit_blocks(ctx: StepContext[AnthropicDumpState, None, _DumpDone]) -> list[BetaContentBlockParam]:
    return ctx.state.blocks

# 7. Wire the topology — declarative edges with a single decision fan-out.
_g.add(
    _g.edge_from(_g.start_node).to(take_next),
    _g.edge_from(take_next).to(
        _g.decision()
        .branch(_g.match(_DumpDone).to(emit_blocks))
        .branch(_g.match(str).to(parse_text))
        .branch(_g.match(CachePoint).to(apply_cache))
        .branch(_g.match(BinaryContent).to(parse_binary))
        # ... per-IR-part-type branches
    ),
    # Loop-back: every parse_* step feeds back into take_next.
    _g.edge_from(parse_text, apply_cache, parse_binary, ...).to(take_next),
    _g.edge_from(emit_blocks).to(_g.end_node),
)

# 8. Build once at import time.
_dump_graph = _g.build()

# 9. Public entrypoint — drives the graph from imperative wrapper code.
async def render_anthropic_dump(parsed: ParsedRequest) -> bytes:
    # ... assemble static envelope (model, tools, system, settings, raw_extras)
    state = AnthropicDumpState(queue=deque(flatten_messages_to_items(parsed.messages)))
    blocks = await _dump_graph.run(state=state)
    # ... stitch blocks into the BetaMessageParam list and serialize
    return json.dumps(body, separators=(",", ":")).encode()
```

### Why this shape

| Concern | Solution |
|---|---|
| **Polymorphic walk** over heterogeneous IR parts | One router step (`take_next`) + a decision with a branch per type. Replaces an imperative `match` statement that would otherwise live inside the step body. |
| **End-of-graph from a router** | A marker class (e.g. `_DumpDone`) routed via `g.match(_DumpDone).to(terminal_step)`. The terminal step returns the accumulated state — that value becomes the graph's output. |
| **Typed dispatch on string-discriminated unions** (load side) | Wrap the runtime-string-tagged dicts in one frozen dataclass per discriminator value (`_UserTextBlock`, `_UserImageUrlBlock`, …). The router inspects the discriminator once and emits the matching envelope; the decision routes by Python type. |
| **Centralized middleware** (e.g. `cache_control` attachment) | A dedicated step that mutates state side-effectfully. Every other step that emits a block updates a `state.last_emitted_block` reference; the middleware step mutates the dict that reference points to. |
| **Side-effect-only no-ops** (items with no provider equivalent) | A `skip_item` step matched by a `_Skip` marker that loops back to the router. Keeps each per-type branch single-purpose. |
| **End-of-stream variant flushing** (load side: `UserPromptPart` accumulator with mid-stream `tool_result` flushes) | The accumulator lives on state; the per-block parse step pushes to it; the `tool_result` parse step flushes it; the terminal step flushes any remaining accumulator before emitting. |
| **Mermaid visualization** | Free via `graph.render(title=..., direction='LR')`. Every FSM file can produce its diagram on demand. |

### What's in each file

| File | What its FSM does | Key marker classes |
|---|---|---|
| `anthropic_dump.py` | IR → Anthropic `BetaMessageParam` content blocks | `_DumpDone`, `_Skip` |
| `anthropic_load.py` | Anthropic content block dict → IR (one user-turn FSM + one assistant-turn FSM, both per-message) | `_UserDone`, `_AssistantDone`, plus envelope dataclasses per wire `type` |
| `openai_dump.py` | IR → OpenAI `ChatCompletionContentPartParam` content parts (one FSM, per-`UserPromptPart` content list only — rest is imperative because OpenAI's per-role message shape isn't polymorphic) | `_OpenAIDone`, `_OpenAISkip` |
| `openai_load.py` | OpenAI user-content list → IR (one FSM; system/tool/assistant role dispatch is imperative) | `_UserDone`, envelope dataclasses |
| `google_dump.py` | **Not really an FSM** — wraps pydantic-ai's `GoogleModel` via the `CaptureSentinel` pattern. Lives in `graph/` for uniformity. Migration to a real FSM is Phase O of `nextplan.md`. | — |
| `perplexity_dump.py` | **Not really an FSM** — wraps `pplx.py:_build_pplx_payload` and friends. Lives in `graph/` for uniformity. | — |

---

## Public API

### `dispatch_load` — wire → IR

```python
from ccproxy.lightllm.graph import dispatch_load
from ccproxy.lightllm.parsed import ListenerFormat

parsed: ParsedRequest = await dispatch_load(
    body_dict,
    listener_format=ListenerFormat.ANTHROPIC_MESSAGES,
)
```

Routes by `listener_format`:
* `ANTHROPIC_MESSAGES` → `load_anthropic`
* `OPENAI_CHAT` → `load_openai_chat`
* `UNKNOWN` → raises `ValueError`

Async because the FSM nodes are async. Drive it via the worker-thread
bridge if you're calling from sync code (see "The worker-thread bridge"
below).

### `dispatch_dump` / `dispatch_dump_sync` — IR → wire

```python
from ccproxy.lightllm.graph import dispatch_dump, dispatch_dump_sync

# Async
wire_bytes: bytes = await dispatch_dump(parsed, provider="anthropic")

# Sync (use this from mitmproxy hooks, pipeline executors, anywhere
# you're outside an event-loop context OR inside one and need a sync
# result)
wire_bytes: bytes = dispatch_dump_sync(parsed, provider="anthropic")
```

Routes by `provider`:
* `anthropic` / `deepseek` / `zai` → `render_anthropic_dump`
* `openai` → `render_openai_chat_dump`
* `google` / `gemini` / `vertex_ai` → `render_google_dump`
* `perplexity_pro` → `render_perplexity_pro_dump`
* anything else → `UnsupportedUpstreamError`

The Anthropic-compatible forks (`deepseek`, `zai`) deliberately share the
Anthropic renderer — their wire format is identical, only the upstream URL
and auth differ (and those are handled by the `Provider` config, not by
lightllm).

### `ParsedRequest` — direct construction

You don't normally build `ParsedRequest` by hand — `dispatch_load` does it.
But for tests and tooling, the dataclass is plain:

```python
from ccproxy.lightllm.parsed import ParsedRequest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

parsed = ParsedRequest(
    model="claude-3-5-haiku-20241022",
    messages=[ModelRequest(parts=[UserPromptPart(content="hello")])],
    request_parameters=ModelRequestParameters(),
    settings={"max_tokens": 1024},
    stream=False,
    raw_extras={},
)
```

---

## The worker-thread bridge

### Why it exists

`pydantic_graph.Graph.run_sync` is deprecated (see
`pydantic_graph/graph.py:160-191` upstream). Its implementation is:

```python
return _utils.get_event_loop().run_until_complete(self.run(...))
```

Calling that from inside an already-running asyncio loop — which is what
happens inside every mitmproxy addon hook — raises
`RuntimeError: This event loop is already running`.

Commit `016d7d1` fixed this for the inbound parser by spinning a worker
thread per invocation:

```python
# src/ccproxy/pipeline/context.py:27-53
def _run_coro_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running → use a private loop on this thread.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    # Loop already running → spawn a worker thread that owns its own loop.
    def _worker() -> Any:
        worker_loop = asyncio.new_event_loop()
        try:
            return worker_loop.run_until_complete(coro)
        finally:
            worker_loop.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_worker).result()
```

`dispatch_dump_sync` in `lightllm/graph/__init__.py` does the same pattern
for the outbound renderer.

### When to use which

* **From async code** (other async FSMs, async hooks, async tests): use
  `await dispatch_load(...)` and `await dispatch_dump(...)`.
* **From sync code inside mitmproxy hooks** or anywhere on the addon
  event loop: use `Context.parse_sync()` or `dispatch_dump_sync(...)`.
* **Never** call `dispatch_dump(...)` or `dispatch_load(...)` from a sync
  context that has a running asyncio loop. The `_run_coro_sync` bridge
  is the only safe way.

### Streaming responses are different

The same per-invocation worker-thread pattern would be pathological for
streaming responses — mitmproxy delivers SSE in many small chunks per
stream, and you don't want to spawn one thread per chunk. The
response-side migration in `nextplan.md` introduces a persistent asyncio
loop per `SSEPipeline` instance (one thread per stream, not one per
chunk). See `nextplan.md` § "Sync vs async at the response boundary" for
the full design.

---

## raw_extras contract

`raw_extras` is the lossless-passthrough mechanism. Anything the IR
doesn't natively model gets stashed here under a conventional key, and the
outbound renderer stitches it back onto the wire body.

### Conventions per provider

**Anthropic load** (`anthropic_load.py`):

| Key | What | Why |
|---|---|---|
| `cc:msg:{i}:block:{j}` | Original `cache_control` dict from a content block | TTL wasn't `5m` or `1h` (the only values pydantic-ai's `CachePoint` accepts) — preserved so dump can re-apply verbatim |
| `unknown_block:msg:{i}:idx:{j}` | Original wire-block dict | Block had a `type` we don't recognize — preserved so dump can emit it back |
| `system` | The original `system` list from the body | Non-uniform `cache_control` across system blocks — can't be expressed via `settings['anthropic_cache_instructions']` (which is uniform-only) |
| `tools` | The original `tools` list from the body | Non-uniform `cache_control` across tools — same reason |
| `metadata` | The body's `metadata` dict | Anthropic-specific; no IR slot |
| Other unmodeled top-level keys | Copied verbatim under their wire name | E.g. `service_tier` |

**OpenAI load** (`openai_load.py`):

| Key | What | Why |
|---|---|---|
| `image_detail:msg:{i}:block:{j}` | The `image_url.detail` string | Not currently part of the `ImageUrl` IR |
| `file:msg:{i}:block:{j}` | Original `file` content block | Preserved verbatim |
| `unknown_block:msg:{i}:block:{j}` | Unknown content block | Same as Anthropic |
| `refusal:msg:{i}` | Refusal text | Assistant refusal isn't in the IR |
| `function_call:msg:{i}` | Legacy `function_call` field | Pre-`tool_calls` OpenAI format |
| `tool_choice` | The body's `tool_choice` | IR has no slot |
| `response_format` | The body's `response_format` | IR has no slot |

### Round-trip contract

Both dumps strip IR-internal markers (anything starting with `cc:`,
`unknown_block:`, `refusal:`, `file:`, `image_detail:`, `function_call:`)
when stitching `raw_extras` back onto the body. Override keys (`system`,
`tools`, `tool_choice`, `response_format`) win over whatever the FSM
produced. Everything else is `setdefault`'d onto the body.

### What this guarantees

If a client sends a request to ccproxy, the inbound parser produces an IR,
the outbound renderer produces a wire body — the round-trip should be
**semantically equivalent** to the original. The `tests/test_lightllm_graph_*`
tests assert this via canonicalization helpers
(`assert_anthropic_bodies_equivalent`) for every shape in the test corpus.

The lossiness regressions specifically called out in the refactor plan:
* `ToolReturnPart.tool_name` populated via two-pass lookup (was hardcoded
  to `""` in the wire.py predecessor).
* Image `media_type` preserved on `BinaryContent` (was defaulted).
* `cache_control` TTLs pydantic-ai can't represent stashed in `raw_extras`
  (were silently coerced).
* Unknown content blocks preserved in `raw_extras` (were dropped).

---

## How Context wires it together

`src/ccproxy/pipeline/context.py:Context` is the per-request envelope hooks
and inspector routes operate on. The lightllm integration is three calls:

### Inbound — parsing

```python
ctx = Context.from_flow(flow)        # builds Context with _listener_format
parsed = ctx.parse_sync()            # → dispatch_load(body, listener_format=...)
# ctx._parsed is now populated; subsequent access reads the cache.
```

The typed property accessors (`ctx.messages`, `ctx.system`, `ctx.tools`)
all funnel through `ctx.parse_sync()`. They return mutable IR objects;
hooks can edit them in place.

### Outbound — committing

```python
ctx.messages = new_messages          # mutate via setter (rebuilds IR)
ctx.system = new_system_parts
ctx.tools = new_tool_definitions
ctx.commit()                         # → dispatch_dump_sync(parsed, provider=...)
                                     # body is re-rendered, written back to flow.request
```

`commit()` is what hook executors call after the DAG runs. It rebuilds
`ParsedRequest` from any mutated typed properties, runs the outbound
renderer for the listener format, and writes the resulting bytes back to
`flow.request.content`.

The provider name passed to `dispatch_dump_sync` is the **listener
format**, not the upstream provider — the transform router decides the
upstream separately. Listener `anthropic_messages` → renderer
`anthropic`; listener `openai_chat` → renderer `openai`. Cross-format
transformation happens upstream of `commit()` — by then, the IR is in the
target format already.

---

## How the inspector wires it together

`src/ccproxy/inspector/routes/transform.py:_handle_transform` is the
inspector's transform route handler. The lightllm interaction:

```python
ctx = Context.from_flow(flow)
parsed = ctx.parse_sync()
if model and model != parsed.model:
    parsed = dataclasses.replace(parsed, model=model)
new_body = dispatch_dump_sync(parsed, provider=provider_str)
```

Where `provider_str` comes from `TransformOverride.dest_provider` or
sentinel-key resolution. The body is then written to `flow.request.content`
and the URL/headers are rewritten via `_resolve_upstream_url_and_headers`.

The Gemini branch in the same handler (lines 321-351) still uses the
legacy `transform_to_provider` from `dispatch.py` because the
cachedContents resolution happens there. That fold-in is Phase O of
`nextplan.md`.

---

## Adding a new provider

Suppose you're adding a new upstream provider — say "MyVendor" — that
accepts an Anthropic-compatible wire format. Walkthrough:

### 1. Configure the provider

In `ccproxy.yaml`:

```yaml
providers:
  myvendor:
    auth:
      type: file
      file: ~/.myvendor/token
    host: api.myvendor.com
    path: /v1/messages
    provider: anthropic    # ← wire format = anthropic-compatible
```

Done. Sentinel key `sk-ant-oat-ccproxy-myvendor` now routes to
`api.myvendor.com` with the Anthropic renderer, because `provider:
anthropic` and `_ANTHROPIC_COMPATIBLE` includes it.

If the wire is OpenAI-compatible, use `provider: openai`. If it's
Google-compatible, `provider: google`.

### 2. If the wire format is genuinely new

Then you need a new FSM. Files to add:

* `src/ccproxy/lightllm/graph/myvendor_dump.py` — pattern from
  `anthropic_dump.py`. State + steps + decision + terminal step + envelope
  wrapper.
* `src/ccproxy/lightllm/graph/myvendor_load.py` (only if listener format
  is also new — i.e. ccproxy needs to ACCEPT requests in MyVendor's wire
  format. Most new providers are upstream-only.)
* Update `src/ccproxy/lightllm/graph/__init__.py:dispatch_dump` to add the
  provider branch:
  ```python
  if provider == "myvendor":
      return await render_myvendor_dump(parsed)
  ```
* Add a `__all__` export entry in `__init__.py`.

### 3. Write the tests

Copy a `tests/test_lightllm_graph_*_dump.py` file and adapt:
* A `Render` type alias and fixture pointing at your new entrypoint.
* Roundtrip cases — at minimum: simple_text, multi_turn_with_tool_use,
  system_as_string, image_with_media_type, sampling_settings.
* Lossiness regressions: `test_metadata_preserved_via_raw_extras`,
  `test_render_returns_bytes`, `test_render_compact_json`.
* Run `uv run pytest tests/test_lightllm_graph_myvendor_dump.py -q --no-cov`.

### 4. Wire mypy

If your new file is the first user of a new pydantic-graph beta API, you
may need to extend the per-module mypy override in `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = [
  "ccproxy.lightllm.graph.anthropic_dump",
  "ccproxy.lightllm.graph.anthropic_load",
  "ccproxy.lightllm.graph.openai_dump",
  "ccproxy.lightllm.graph.openai_load",
  "ccproxy.lightllm.graph.myvendor_dump",   # ← add here
]
disable_error_code = ["type-arg", "attr-defined", "no-any-return",
                       "misc", "index", "arg-type", "unreachable"]
```

This compensates for pydantic_graph.beta's `TypeVar(infer_variance=True)`
which mypy 1.19 doesn't recognize. Pyright handles it correctly so editor
IntelliSense is unaffected.

---

## Testing

### The parametrize-then-collapse pattern

During the request-side FSM migration, each test file had two
implementations to compare:

```python
@pytest.fixture(params=["legacy", "fsm"])
def render(request) -> Render:
    if request.param == "legacy":
        return render_anthropic        # the old CaptureSentinel path
    return render_anthropic_dump       # the new FSM
```

Every test ran twice; both implementations had to satisfy the same
assertion contract. Once parity was proven, the `legacy` branch was
deleted along with the legacy file, and the fixture collapsed to:

```python
@pytest.fixture
def render() -> Render:
    return render_anthropic_dump
```

Use this same pattern for any further migrations (the response-side phase
will use it; the per-provider FSM additions can use it if you keep a
reference implementation around for comparison).

### Lossiness assertions

The `tests/test_lightllm_graph_anthropic_load.py:TestLossinessRegressions`
class has four asserts that the dump can't drop:

* `tool_name` populated for `ToolReturnPart` via two-pass lookup
* `BinaryContent.media_type` preserved
* Non-standard `cache_control.ttl` stashed in `raw_extras["cc:msg:N:block:M"]`
* Unknown content blocks stashed in `raw_extras["unknown_block:msg:N:idx:M"]`

Mirror these for any new provider's load FSM.

### Roundtrip semantic equivalence

`tests/test_lightllm_graph_anthropic_dump.py:test_roundtrip_semantic_equivalence`
asserts:

```python
parsed = await load_anthropic(case.body)
rendered = await render_anthropic_dump(parsed)
rebuilt = json.loads(rendered)
assert_anthropic_bodies_equivalent(case.body, rebuilt)
```

The `assert_anthropic_bodies_equivalent` helper tolerates field ordering,
`null` vs missing, `content` string ↔ single-block-list normalization,
`system` string ↔ block-list normalization, uniform-cache block
concatenation, default `tool_choice = auto`, and redundant
`is_error: False` defaults on tool_result blocks. Asserts equality on
`model`, `max_tokens`, `tools`, `messages`, `system`, and the sampling
settings.

---

## Visualization

Every FSM in `lightllm/graph/` can render itself as a mermaid diagram:

```python
from ccproxy.lightllm.graph.anthropic_dump import _dump_graph
print(_dump_graph.render(title="anthropic_dump", direction="LR"))
```

Produces (excerpt):

```
---
title: anthropic_dump
---
stateDiagram-v2
  direction LR
  take_next
  state decision <<choice>>
  apply_cache
  emit_blocks
  parse_binary
  parse_text
  parse_tool_call_part
  parse_tool_return
  parse_url
  skip_item

  [*] --> take_next
  take_next --> decision
  decision --> apply_cache
  decision --> emit_blocks
  decision --> parse_binary
  decision --> parse_text
  decision --> parse_tool_call_part
  decision --> parse_tool_return
  decision --> parse_url
  decision --> skip_item
  apply_cache --> take_next
  parse_binary --> take_next
  parse_text --> take_next
  parse_tool_call_part --> take_next
  parse_tool_return --> take_next
  parse_url --> take_next
  skip_item --> take_next
  emit_blocks --> [*]
```

Useful for debugging surprising routing, for code reviews, and for
keeping docs in sync.

---

## Troubleshooting

### `RuntimeError: This event loop is already running`

You called `dispatch_load(...)` or `dispatch_dump(...)` from sync code
inside a running asyncio loop. Use `Context.parse_sync()` or
`dispatch_dump_sync()` — they bridge through `_run_coro_sync`.

### `UnsupportedUpstreamError: no outbound renderer for provider='X'`

Either the provider name is misspelled in `providers.X.provider` (config),
or you're trying to route to a provider that has no dump FSM. Add the
provider branch in `lightllm/graph/__init__.py:dispatch_dump`.

### `ValueError: no IR parser for listener_format=UNKNOWN`

The listener-format detection in `Context.from_flow` didn't match the
request path or headers. Check `_select_listener_format` in
`pipeline/context.py:86-100`. Usual cause: a path that's neither
`/v1/messages` nor `/v1/chat/completions` and no `anthropic-version`
header.

### A test passes for the legacy parser but fails for the FSM (or vice versa)

You're mid-migration. Check the parametrize fixture in the test file — if
one of the two implementations behaves differently, the FSM has a bug or
the legacy had a bug the FSM doesn't reproduce. Use `pytest -vv` to see
the full diff; the canonicalization helpers print expected vs actual as
sorted JSON.

### `mypy: type-arg ... cannot be parameterized`

You're touching a file that uses `pydantic_graph.beta` types and your
module isn't in the `pyproject.toml` mypy override list. Add it to the
relevant `[[tool.mypy.overrides]]` block.

### Lossiness regression test failed

A specific behavioral contract that's documented in the test docstring
just broke. Look at `tests/test_lightllm_graph_{anthropic,openai}_load.py:TestLossinessRegressions`.
Restore the behavior — these are non-negotiable round-trip invariants.

### Streaming response is malformed / cut off

You're hitting the hand-rolled response side (`response/intake_*.py`,
`response/render_*.py`, `response/pipeline.py`). The FSM doesn't own this
yet. Check `inspector/addon.py:_install_sse_transformer` to see which
intake/render pair was selected; check `ccproxy logs -f` for warnings
about chunk parse failures.

---

## File map

| Component | Path |
|---|---|
| Request envelope | `src/ccproxy/lightllm/parsed.py` |
| Public dispatchers | `src/ccproxy/lightllm/graph/__init__.py` |
| Anthropic FSMs | `src/ccproxy/lightllm/graph/anthropic_{dump,load}.py` |
| OpenAI FSMs | `src/ccproxy/lightllm/graph/openai_{dump,load}.py` |
| Google dump (wraps GoogleModel) | `src/ccproxy/lightllm/graph/google_dump.py` |
| Perplexity dump (wraps pplx.py) | `src/ccproxy/lightllm/graph/perplexity_dump.py` |
| Worker-thread bridge (inbound) | `src/ccproxy/pipeline/context.py:_run_coro_sync` |
| Worker-thread bridge (outbound) | `src/ccproxy/lightllm/graph/__init__.py:dispatch_dump_sync` |
| Inspector call site | `src/ccproxy/inspector/routes/transform.py:_handle_transform` |
| Tests | `tests/test_lightllm_graph_*.py` |
| Response-side intake (hand-rolled) | `src/ccproxy/lightllm/response/intake_*.py` |
| Response-side render (hand-rolled) | `src/ccproxy/lightllm/response/render_*.py` |
| Response-side pipeline + buffered wrappers | `src/ccproxy/lightllm/response/{pipeline,buffered}.py` |
| Legacy LiteLLM-mediated paths (scheduled for deletion) | `src/ccproxy/lightllm/{dispatch,context_cache,noop_logging}.py` |
| Perplexity provider (LiteLLM BaseConfig subclass) | `src/ccproxy/lightllm/pplx.py` |
| Perplexity business logic | `src/ccproxy/lightllm/pplx_steps.py`, `pplx_threads.py` |
| Provider registry | `src/ccproxy/lightllm/registry.py` |
| Plan for the next phase | `nextplan.md` |
