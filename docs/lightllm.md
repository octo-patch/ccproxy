# lightllm — wire translation layer

`ccproxy.lightllm` is the IR ↔ wire translation layer. It turns an incoming
request body (Anthropic Messages, OpenAI Chat Completions) into an
intermediate representation that ccproxy's hook pipeline can manipulate, and
back into a request body for whatever upstream provider the router resolves
to (Anthropic, OpenAI, Google Gemini, Perplexity Pro, plus the
Anthropic-compatible forks DeepSeek and ZAI). On the response side the same
package turns upstream SSE bytes (or buffered JSON) back into IR events and
re-renders to the listener's wire format.

Both directions share one FSM idiom built on
`pydantic_graph.beta.GraphBuilder`: one `*_load.py` / `*_dump.py` /
`*_intake.py` / `*_render.py` module per provider/listener-format. There is
no LiteLLM dependency; `rg "litellm" src/` returns empty.

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
  │                                    │  │   → provider wire bytes ────────▶│
  │                                    │  └──────────────────────────────┘    │
  │                                    │                                      │
  │                                    │◀── provider wire (buffered or SSE) ──│
  │                                    │  ┌──────────────────────────────┐    │
  │                                    │  │ SSE: SSEPipeline (sync       │    │
  │                                    │  │   mitmproxy stream callable) │    │
  │                                    │  │   → persistent asyncio loop  │    │
  │                                    │  │     ↓                        │    │
  │                                    │  │   dispatch_intake(provider=) │    │
  │                                    │  │     → ModelResponseStream    │    │
  │                                    │  │       Event (IR)             │    │
  │                                    │  │     ↓                        │    │
  │                                    │  │   dispatch_render(listener=) │    │
  │                                    │  │     ↓                        │    │
  │                                    │  │ Buffered: transform_buffered │    │
  │                                    │  │   _response_sync(...) drives │    │
  │                                    │  │   intake once + emits        │    │
  │                                    │  │   listener-shape JSON        │    │
  │                                    │  └──────────────────────────────┘    │
  │◀── RESPONSE (listener wire) ───────│                                      │
  │                                    │                                      │
```

The thick line through the middle is `pydantic_ai.messages` — `ModelMessage`
+ `ModelResponseStreamEvent` are the canonical IR types the pipeline hooks
operate on.

### Module layout

```
src/ccproxy/lightllm/
├── parsed.py             ParsedRequest, ParsedResponse, ListenerFormat
├── registry.py           Local Perplexity Pro registration (no LiteLLM fallback)
├── pplx.py               Perplexity Pro config + exceptions (no LiteLLM bases)
├── pplx_steps.py         Perplexity step trail renderer
├── pplx_threads.py       Perplexity thread continuation helpers
│
└── graph/                ← FSM modules (canonical)
    ├── __init__.py       dispatch_load, dispatch_dump, dispatch_dump_sync,
    │                      dispatch_intake, dispatch_render
    │
    ├── anthropic_dump.py   IR → Anthropic Messages wire
    ├── anthropic_load.py   Anthropic Messages wire → IR
    ├── anthropic_intake.py Anthropic SSE → IR events
    ├── anthropic_render.py IR events → Anthropic SSE
    │
    ├── openai_dump.py    IR → OpenAI Chat Completions wire
    ├── openai_load.py    OpenAI Chat Completions wire → IR
    ├── openai_intake.py  OpenAI SSE → IR events
    ├── openai_render.py  IR events → OpenAI SSE
    │
    ├── google_dump.py    IR → Google Gemini generateContent (wraps GoogleModel)
    ├── google_intake.py  Google streamGenerateContent SSE → IR events
    │                      (cloudcode-pa envelope unwrap folded in)
    │
    ├── perplexity_dump.py   IR → Perplexity Pro wire (wraps pplx.py helpers)
    ├── perplexity_intake.py Perplexity Pro SSE → IR events
    │
    ├── sse_pipeline.py   SSEPipeline — persistent asyncio loop per stream
    └── buffered.py       transform_buffered_response_sync — non-streaming
                          cross-format transform via FSM
```

There is no `response/` subpackage anymore (deleted), no `dispatch.py`
(deleted), no `context_cache.py` (deleted — Gemini cachedContents is
unsupported via the OAuth path the production deployment uses; restore it as
an outbound hook if API-key Gemini ever needs it).

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

### `ParsedResponse` — the response envelope

```python
@dataclass(frozen=True)
class ParsedResponse:
    model: str                            # model from upstream response
    response: ModelResponse               # pydantic-ai IR (TextPart/ToolCallPart/...)
    stream: bool = False                  # was the response streamed?
    raw_extras: dict[str, Any] = field(default_factory=dict)
```

Mirrors `ParsedRequest`. Used by the buffered path; streaming flows pass
`ModelResponseStreamEvent` directly between intake and render FSMs.

### `ModelMessage` and `ModelResponseStreamEvent` — the conversation IR

From `pydantic_ai.messages`.

* **`ModelRequest(parts=[...])`** — user/system turn. Parts:
  `SystemPromptPart`, `UserPromptPart(content=str | list[UserContent])`
  where `UserContent` is one of `str`, `BinaryContent`, `ImageUrl`,
  `DocumentUrl`, `AudioUrl`, `UploadedFile`, `CachePoint`; plus
  `ToolReturnPart`, `RetryPromptPart`.

* **`ModelResponse(parts=[...])`** — assistant turn. Parts: `TextPart`,
  `ToolCallPart`, `ThinkingPart` (including `id="redacted_thinking"` for
  opaque ciphertext).

Streaming uses `ModelResponseStreamEvent` — a union of `PartStartEvent`,
`PartDeltaEvent`, `PartEndEvent`, `FinalResultEvent`. The intake FSM drives
pydantic-ai's `ModelResponsePartsManager` and yields these events; the
render FSM consumes them.

### `ListenerFormat` — what the client sent

```python
class ListenerFormat(str, Enum):
    UNKNOWN = "unknown"
    ANTHROPIC_MESSAGES = "anthropic_messages"   # /v1/messages
    OPENAI_CHAT = "openai_chat"                 # /v1/chat/completions
```

Pinned at `Context` construction from path + headers. Drives the choice of
inbound parser (`dispatch_load`) AND the choice of response renderer
(`dispatch_render`). The **upstream provider** the request routes to is a
separate decision (made by the transform router via sentinel-key or
`TransformOverride` rule).

---

## The FSM pattern

Every file under `lightllm/graph/*_{dump,load,intake,render}.py` (except the
google/perplexity dump wrappers) follows the same shape. Reading
`anthropic_dump.py` end-to-end is the fastest way to understand it; the
other 11 modules echo its idioms.

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

# 3. GraphBuilder — type parameters describe the FSM's runtime signature.
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

# ... (more per-type steps for BinaryContent, ImageUrl, ToolReturnPart, etc.)

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
        .branch(_g.match(BinaryContent).to(parse_binary))
        # ... per-IR-part-type branches
    ),
    # Loop-back: every parse_* step feeds back into take_next.
    _g.edge_from(parse_text, parse_binary, ...).to(take_next),
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
| **Polymorphic walk** over heterogeneous IR parts | One router step (`take_next`) + a decision with a branch per type. |
| **End-of-graph from a router** | A marker class (e.g. `_DumpDone`) routed via `g.match(_DumpDone).to(terminal_step)`. The terminal step returns the accumulated state — that value becomes the graph's output. |
| **Typed dispatch on string-discriminated unions** (load + intake side) | Wrap the runtime-string-tagged dicts in one frozen dataclass per discriminator value (`_UserTextBlock`, `_MessageStartEvent`, …). The router inspects the discriminator once and emits the matching envelope; the decision routes by Python type. |
| **Centralized middleware** (e.g. `cache_control` attachment) | A dedicated step that mutates state side-effectfully. Every other step that emits a block updates a `state.last_emitted_block` reference; the middleware step mutates the dict that reference points to. |
| **Side-effect-only no-ops** | A `skip_item` step matched by a `_Skip` marker that loops back to the router. |
| **Mermaid visualization** | Free via `graph.render(title=..., direction='LR')`. Every FSM file can produce its diagram on demand. |

### What each file does

| File | What its FSM does | Key marker classes |
|---|---|---|
| `anthropic_dump.py` | IR → Anthropic `BetaMessageParam` content blocks | `_DumpDone`, `_Skip` |
| `anthropic_load.py` | Anthropic content block dict → IR (user-turn FSM + assistant-turn FSM, both per-message) | `_UserDone`, `_AssistantDone`, envelope dataclasses |
| `anthropic_intake.py` | Anthropic SSE → IR `ModelResponseStreamEvent` (typed dispatch on `BetaRawMessageStreamEvent` union) | `_FeedDone`, `_IgnoredEvent` |
| `anthropic_render.py` | IR `ModelResponseStreamEvent` → Anthropic SSE wire bytes | `_RenderDone` |
| `openai_dump.py` | IR → OpenAI content parts (per-`UserPromptPart` only — rest is imperative because OpenAI's per-role message shape isn't polymorphic) | `_OpenAIDone`, `_OpenAISkip` |
| `openai_load.py` | OpenAI user-content list → IR (system/tool/assistant role dispatch is imperative) | `_UserDone`, envelope dataclasses |
| `openai_intake.py` | OpenAI Chat Completions SSE → IR (per-chunk envelope dispatch on content/tool_call/refusal shapes) | `_FeedDone`, `_RefusalChunk`, `_StandardChunk`, `_EmptyChoicesChunk` |
| `openai_render.py` | IR → OpenAI Chat Completions SSE | `_RenderDone` |
| `google_dump.py` | **Not really an FSM** — wraps pydantic-ai's `GoogleModel` via the `CaptureSentinel` pattern. Lives in `graph/` for uniformity. | — |
| `google_intake.py` | Google `streamGenerateContent` chunks → IR (envelope unwrap of `{response: {...}}` from cloudcode-pa folded in) | `_FeedDone` |
| `perplexity_dump.py` | **Not really an FSM** — wraps `pplx.py:_build_pplx_payload` and friends. | — |
| `perplexity_intake.py` | Perplexity Pro SSE → IR (per-event-type dispatch driving `_extract_deltas`) | `_FeedDone`, `_PerplexityEventEnvelope` |
| `sse_pipeline.py` | Sync mitmproxy stream callable backed by a persistent asyncio loop + daemon thread; drives an intake + render FSM pair per stream | — |
| `buffered.py` | Non-streaming buffered-body cross-format transform; synthesizes streaming events from buffered JSON per provider, drives the intake FSM, emits listener-shape JSON | — |

---

## Public API

### Request side

```python
from ccproxy.lightllm.graph import dispatch_load, dispatch_dump, dispatch_dump_sync
from ccproxy.lightllm.parsed import ListenerFormat

# Inbound: wire → IR
parsed: ParsedRequest = await dispatch_load(
    body_dict, listener_format=ListenerFormat.ANTHROPIC_MESSAGES,
)

# Outbound (async)
wire_bytes: bytes = await dispatch_dump(parsed, provider="anthropic")

# Outbound (sync — from inside mitmproxy hooks or pipeline executors)
wire_bytes: bytes = dispatch_dump_sync(parsed, provider="anthropic")
```

`dispatch_dump` routes by upstream provider:
* `anthropic` / `deepseek` / `zai` → `render_anthropic_dump`
* `openai` → `render_openai_chat_dump`
* `google` / `gemini` / `vertex_ai` / `vertex_ai_beta` → `render_google_dump`
* `perplexity_pro` → `render_perplexity_pro_dump`
* anything else → `UnsupportedUpstreamError`

The Anthropic-compatible forks (`deepseek`, `zai`) deliberately share the
Anthropic renderer — their wire format is identical, only the upstream URL
and auth differ (and those are handled by the `Provider` config).

### Response side

```python
from ccproxy.lightllm.graph import dispatch_intake, dispatch_render
from ccproxy.lightllm.graph.sse_pipeline import SSEPipeline
from ccproxy.lightllm.graph.buffered import transform_buffered_response_sync

# Streaming (mitmproxy installs this on flow.response.stream)
intake = dispatch_intake(
    upstream_provider="anthropic", model="claude-...", request_params=...,
)
render = dispatch_render(listener_format=ListenerFormat.OPENAI_CHAT, model="claude-...")
pipeline = SSEPipeline(intake=intake, render=render)
flow.response.stream = pipeline

# Buffered (one-shot from inspector route handler)
listener_body: bytes = transform_buffered_response_sync(
    raw_bytes=flow.response.content,
    upstream_provider="anthropic",
    listener_format=ListenerFormat.OPENAI_CHAT,
    model="claude-...",
    request_params=...,
)
```

`dispatch_intake` and `dispatch_render` return async FSM instances. The
`SSEPipeline` adapts them to mitmproxy's sync stream callable contract.

### `ParsedRequest` / `ParsedResponse` — direct construction

You don't normally build these by hand — `dispatch_load` and `buffered.py`
do it. For tests and tooling, the dataclasses are plain:

```python
from ccproxy.lightllm.parsed import ParsedRequest, ParsedResponse
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

req = ParsedRequest(
    model="claude-3-5-haiku-20241022",
    messages=[ModelRequest(parts=[UserPromptPart(content="hello")])],
    request_parameters=ModelRequestParameters(),
    settings={"max_tokens": 1024},
)
resp = ParsedResponse(
    model="claude-3-5-haiku-20241022",
    response=ModelResponse(parts=[TextPart(content="hi")]),
    stream=False,
)
```

---

## The sync/async bridges

### Request-side worker thread (`dispatch_dump_sync`)

`pydantic_graph.Graph.run_sync` is deprecated. Its implementation is:

```python
return _utils.get_event_loop().run_until_complete(self.run(...))
```

Calling that from inside an already-running asyncio loop — which is what
happens inside every mitmproxy addon hook — raises `RuntimeError: This
event loop is already running`.

`Context._run_coro_sync` (`pipeline/context.py:27-53`) spins a worker
thread per invocation: a `ThreadPoolExecutor(max_workers=1)` that owns a
fresh asyncio loop, runs the coro to completion, then tears down.
`dispatch_dump_sync` in `lightllm/graph/__init__.py` does the same pattern
for the outbound renderer.

Use cases:
* **From async code** (other async FSMs, async hooks, async tests): use
  `await dispatch_load(...)` and `await dispatch_dump(...)`.
* **From sync code inside mitmproxy hooks** or anywhere on the addon
  event loop: use `Context.parse_sync()` or `dispatch_dump_sync(...)`.
* **Never** call `dispatch_dump(...)` or `dispatch_load(...)` from a sync
  context that has a running asyncio loop. The `_run_coro_sync` bridge
  is the only safe way.

### Response-side persistent loop (`SSEPipeline`)

The per-invocation worker-thread pattern would be pathological for
streaming responses — mitmproxy delivers SSE in many small chunks per
stream, and spawning one thread + fresh loop per chunk would mean ~200
fresh loops in a 5-second stream.

`SSEPipeline` (`lightllm/graph/sse_pipeline.py`) instead owns one
persistent `asyncio.AbstractEventLoop` running in a daemon thread per
instance. Each chunk is submitted to that loop via
`asyncio.run_coroutine_threadsafe` and the result awaited synchronously:

```python
class SSEPipeline:
    def __init__(self, *, intake, render):
        self._intake = intake
        self._render = render
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="ccproxy-sse-loop",
        )
        self._thread.start()

    def __call__(self, data: bytes) -> bytes | list[bytes]:
        if data == b"":
            return self._flush_and_close()
        future = asyncio.run_coroutine_threadsafe(self._process_chunk(data), self._loop)
        return future.result() or []

    async def _process_chunk(self, data: bytes) -> bytes:
        out = bytearray()
        for event in await self._intake.feed(data):
            out.extend(await self._render.render(event))
        return bytes(out)
```

Per-chunk overhead is ~10-50 µs of cross-thread hop, negligible against
the ~10-100 ms-per-chunk network-I/O floor.

Lifecycle: the daemon thread dies with the process, so a missed `close()`
won't leak — but `InspectorAddon.response` calls `pipeline.close()`
explicitly on flow finalization for tidiness. `close()` is idempotent.

### Buffered transforms use a simpler per-call loop

`transform_buffered_response_sync` in `lightllm/graph/buffered.py` is
one-shot per response (no streaming) so it just uses the per-call
asyncio-loop pattern. No persistent thread, no overhead.

---

## raw_extras contract

`raw_extras` is the lossless-passthrough mechanism. Anything the IR
doesn't natively model gets stashed here under a conventional key, and the
outbound renderer (or response render) stitches it back onto the wire body.

### Request-side conventions

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

### Response-side conventions

Streaming intakes drive `ModelResponsePartsManager` directly and don't
currently surface per-message metadata via `raw_extras`. The buffered
transform parses metadata into the listener-format envelope fields (usage,
finish_reason, model) at serialization time. If you need response-side
`raw_extras` (e.g., for citations, safety, groundingMetadata
preservation), add a `state.raw_extras` field to the per-provider intake's
FSM state and stitch it back on the buffered side — the pattern is
symmetric with the request side.

### Round-trip contract

Both request-side dumps strip IR-internal markers (anything starting with
`cc:`, `unknown_block:`, `refusal:`, `file:`, `image_detail:`,
`function_call:`) when stitching `raw_extras` back onto the body. Override
keys (`system`, `tools`, `tool_choice`, `response_format`) win over
whatever the FSM produced. Everything else is `setdefault`'d onto the
body.

### What this guarantees

If a client sends a request to ccproxy, the inbound parser produces an IR,
the outbound renderer produces a wire body — the round-trip should be
**semantically equivalent** to the original. The `tests/test_lightllm_graph_*`
tests assert this via canonicalization helpers
(`assert_anthropic_bodies_equivalent`) for every shape in the test corpus.

The lossiness regressions specifically called out:
* `ToolReturnPart.tool_name` populated via two-pass lookup (was hardcoded
  to `""` in the pre-FSM wire.py predecessor).
* Image `media_type` preserved on `BinaryContent` (was defaulted).
* `cache_control` TTLs pydantic-ai can't represent stashed in `raw_extras`
  (were silently coerced).
* Unknown content blocks preserved in `raw_extras` (were dropped).

---

## How Context wires the request side

`src/ccproxy/pipeline/context.py:Context` is the per-request envelope
hooks and inspector routes operate on. The lightllm integration is three
calls:

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

## How the inspector wires the response side

`src/ccproxy/inspector/addon.py:InspectorAddon` installs the streaming
pipeline in `responseheaders`:

```python
def _install_streaming_transformer(self, flow, transform):
    listener_format = ListenerFormat(transform.listener_format)
    intake = dispatch_intake(
        upstream_provider=transform.provider,
        model=transform.model,
        request_params=transform.request_parameters,
    )
    render = dispatch_render(listener_format=listener_format, model=transform.model)
    pipeline = SSEPipeline(intake=intake, render=render)
    flow.response.stream = pipeline
    flow.metadata["ccproxy.sse_transformer"] = pipeline
```

`InspectorAddon.response` calls `pipeline.close()` on flow finalization to
tear down the daemon thread promptly.

For non-streaming flows, `inspector/routes/transform.py:handle_transform_response`
calls `transform_buffered_response_sync` instead — same `dispatch_intake`
under the hood, plus per-provider buffered-body-to-streaming-events
synthesis where the upstream's buffered shape differs from its streaming
shape (Anthropic, OpenAI, Google) or direct feed where it doesn't
(Perplexity Pro always streams, so its buffered body IS concatenated SSE).

`GeminiAddon.responseheaders` backs off from installing its
`EnvelopeUnwrapStream` when `flow.response.stream` is already a callable
(i.e., when `InspectorAddon` installed an `SSEPipeline`). The unwrap is
folded into `google_intake.py` for that path; the addon-installed
`EnvelopeUnwrapStream` still handles passthrough Gemini flows.

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
`api.myvendor.com` with the Anthropic renderer + intake + render, because
`provider: anthropic` and `_ANTHROPIC_COMPATIBLE` includes it.

If the wire is OpenAI-compatible, use `provider: openai`. If it's
Google-compatible, `provider: google`.

### 2. If the wire format is genuinely new

Then you need a new set of FSMs. Files to add:

* `src/ccproxy/lightllm/graph/myvendor_dump.py` — IR → wire bytes. Pattern
  from `anthropic_dump.py`.
* `src/ccproxy/lightllm/graph/myvendor_intake.py` — wire SSE → IR events.
  Pattern from `anthropic_intake.py`.
* `src/ccproxy/lightllm/graph/myvendor_load.py` (only if listener format
  is also new — i.e. ccproxy needs to ACCEPT requests in MyVendor's wire
  format. Most new providers are upstream-only.)
* `src/ccproxy/lightllm/graph/myvendor_render.py` (only if listener
  format is new — same reason.)
* Update `src/ccproxy/lightllm/graph/__init__.py`:
  * Add `myvendor` to the dispatch branches in `dispatch_dump`,
    `dispatch_intake`, and `dispatch_render` (the last two only if the
    listener format is also new).
  * Add `MyVendorResponseIntakeFSM` to the `AnyAsyncIntakeFSM` union and
    `MyVendorResponseRenderFSM` to `AnyAsyncRenderFSM`.
  * Add `__all__` exports.

If the new provider just needs buffered response support, add a synthesis
branch to `buffered.py:_synthesize_chunks_for` covering its buffered-body
shape.

### 3. Write the tests

Copy a `tests/test_lightllm_graph_*.py` file and adapt:
* Roundtrip cases — at minimum: simple_text, multi_turn_with_tool_use,
  system_as_string, image_with_media_type, sampling_settings.
* Lossiness regressions: `test_metadata_preserved_via_raw_extras`,
  `test_render_returns_bytes`, `test_render_compact_json`.
* Run `uv run pytest tests/test_lightllm_graph_myvendor_*.py -q --no-cov`.

### 4. Wire mypy

If your new file is the first user of a new pydantic-graph beta API, you
may need to extend the per-module mypy override in `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = [
  "ccproxy.lightllm.graph.anthropic_dump",
  "ccproxy.lightllm.graph.anthropic_load",
  # ... existing entries
  "ccproxy.lightllm.graph.myvendor_dump",   # ← add here
  "ccproxy.lightllm.graph.myvendor_intake",
]
disable_error_code = ["type-arg", "attr-defined", "no-any-return",
                       "misc", "index", "arg-type", "unreachable"]
```

This compensates for `pydantic_graph.beta`'s `TypeVar(infer_variance=True)`
which mypy 1.19 doesn't recognize. Pyright handles it correctly so editor
IntelliSense is unaffected.

---

## Testing

### Roundtrip semantic equivalence (request side)

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

### Roundtrip event-sequence equivalence (response side)

`tests/test_lightllm_graph_render_anthropic.py:test_roundtrip_*` feeds a
canonical SSE byte stream through the intake FSM, captures the resulting
IR event sequence, drives it back through the render FSM, parses the
result back into IR via a fresh intake — and asserts structural equality.
Same shape as the request-side roundtrip; the render's terminator bytes
are excluded from the round-trip target since the intake doesn't re-emit
them.

### Cross-impl streaming parity

`tests/test_lightllm_graph_sse_pipeline.py` exercises the persistent-loop
`SSEPipeline` against canonical fixtures:
* Anthropic → Anthropic same-format: render produces byte-equivalent SSE
  (after canonical normalization of random ids and `created` timestamps).
* Anthropic → OpenAI cross-format: render produces parseable OpenAI SSE
  whose IR re-parse matches the input.
* Chunk-boundary robustness: same wire output under 1-byte, 16-byte,
  64-byte, and all-at-once chunking.
* Concurrent independent pipelines on the same thread don't share state.

### Lossiness assertions

`tests/test_lightllm_graph_intake_anthropic.py:TestLossinessRegressions`
has four asserts that the dump can't drop:

* `tool_name` populated for `ToolReturnPart` via two-pass lookup
* `BinaryContent.media_type` preserved
* Non-standard `cache_control.ttl` stashed in `raw_extras["cc:msg:N:block:M"]`
* Unknown content blocks stashed in `raw_extras["unknown_block:msg:N:idx:M"]`

Mirror these for any new provider's load FSM.

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
`dispatch_dump_sync()` — they bridge through `_run_coro_sync`. For
streaming response work, the `SSEPipeline`'s persistent loop handles
this automatically.

### `UnsupportedUpstreamError: no outbound renderer for provider='X'`

Either the provider name is misspelled in `providers.X.provider` (config),
or you're trying to route to a provider that has no dump FSM. Add the
provider branch in `lightllm/graph/__init__.py:dispatch_dump`.

### `UnsupportedUpstreamError: no response intake for upstream_provider='X'`

Same diagnosis, but for the response side. Add a branch in
`dispatch_intake` plus the per-provider intake FSM module.

### `UnsupportedListenerError: no response render for listener_format=X`

The listener format wasn't recognized by `dispatch_render`. Add a render
FSM module + a branch in `dispatch_render`.

### `ValueError: no IR parser for listener_format=UNKNOWN`

The listener-format detection in `Context.from_flow` didn't match the
request path or headers. Check `_select_listener_format` in
`pipeline/context.py:86-100`. Usual cause: a path that's neither
`/v1/messages` nor `/v1/chat/completions` and no `anthropic-version`
header.

### `mypy: type-arg ... cannot be parameterized`

You're touching a file that uses `pydantic_graph.beta` types and your
module isn't in the `pyproject.toml` mypy override list. Add it to the
relevant `[[tool.mypy.overrides]]` block.

### Lossiness regression test failed

A specific behavioral contract that's documented in the test docstring
just broke. Look at `tests/test_lightllm_graph_intake_{anthropic,openai}.py:TestLossinessRegressions`.
Restore the behavior — these are non-negotiable round-trip invariants.

### Streaming response is malformed / cut off

* Check `inspector/addon.py:_install_streaming_transformer` ran — search
  the logs for "SSEPipeline missing listener_format / request_parameters".
  The pipeline only installs when both are stamped on the `TransformMeta`.
* Check the persistent loop is alive — `pipeline.close()` shouldn't have
  fired before EOS. `InspectorAddon.response` is the explicit-close
  callsite.
* Check `flow.response.stream` is the `SSEPipeline` instance, not
  overwritten by `GeminiAddon.responseheaders` (which has a back-off
  guard — investigate if the guard mis-fired).

### Buffered response is malformed

`transform_buffered_response_sync` failed silently — check the inspector
log for "Response transform failed, passing through raw response". Common
causes: synthesizing the per-block synthetic SSE for Anthropic when a
content block has an unexpected `type`; the buffered Gemini body wasn't a
`GenerateContentResponse` instance (cloudcode-pa returned an error
envelope without unwrap).

---

## File map

| Component | Path |
|---|---|
| Request envelope | `src/ccproxy/lightllm/parsed.py` (`ParsedRequest`) |
| Response envelope | `src/ccproxy/lightllm/parsed.py` (`ParsedResponse`) |
| Public dispatchers | `src/ccproxy/lightllm/graph/__init__.py` |
| Anthropic FSMs | `src/ccproxy/lightllm/graph/anthropic_{dump,load,intake,render}.py` |
| OpenAI FSMs | `src/ccproxy/lightllm/graph/openai_{dump,load,intake,render}.py` |
| Google FSMs | `src/ccproxy/lightllm/graph/google_{dump,intake}.py` (dump wraps `GoogleModel`) |
| Perplexity FSMs | `src/ccproxy/lightllm/graph/perplexity_{dump,intake}.py` (dump wraps `pplx.py`) |
| Streaming response pipeline | `src/ccproxy/lightllm/graph/sse_pipeline.py` |
| Buffered response transform | `src/ccproxy/lightllm/graph/buffered.py` |
| Worker-thread bridge (inbound) | `src/ccproxy/pipeline/context.py:_run_coro_sync` |
| Worker-thread bridge (outbound) | `src/ccproxy/lightllm/graph/__init__.py:dispatch_dump_sync` |
| Persistent-loop bridge (response stream) | `src/ccproxy/lightllm/graph/sse_pipeline.py:SSEPipeline` |
| Inspector streaming call site | `src/ccproxy/inspector/addon.py:_install_streaming_transformer` |
| Inspector buffered call site | `src/ccproxy/inspector/routes/transform.py:handle_transform_response` |
| Inspector transform call site | `src/ccproxy/inspector/routes/transform.py:_handle_transform` |
| Tests | `tests/test_lightllm_graph_*.py` |
| Perplexity Pro provider config + exceptions | `src/ccproxy/lightllm/pplx.py` |
| Perplexity business logic | `src/ccproxy/lightllm/pplx_steps.py`, `pplx_threads.py` |
| Provider registry | `src/ccproxy/lightllm/registry.py` |
