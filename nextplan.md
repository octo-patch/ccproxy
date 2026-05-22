# Next session: symmetric pydantic-graph FSM for response side

## Why this plan exists

The request-side FSM rewrite landed in commit `<sha>` ("refactor(ccproxy): migrate lightllm wire layer to pydantic-graph FSM"). What it accomplished:

* `lightllm/graph/` owns IR ↔ wire translation for **REQUEST** bodies across all four providers (Anthropic / OpenAI / Google / Perplexity).
* `dispatch_load` (wire → IR) and `dispatch_dump_sync` (IR → wire) are the public entry points; `Context.parse_sync` and `inspector/routes/transform.py:_handle_transform` are wired through them.
* The `CaptureSentinel` + `AnthropicModel` / `OpenAIChatModel` / `GoogleModel` instantiation hack is gone for Anthropic and OpenAI dumps. Google + Perplexity dumps still use their original mechanisms but live inside `lightllm/graph/` for uniformity.
* The worker-thread bridge (`Context._run_coro_sync`, `dispatch_dump_sync`) is preserved because pydantic-graph's `Graph.run_sync` is deprecated and event-loop-bound (verified at `graph.py:160-191`).
* 1689 tests pass (matches baseline at `9e8aa30`).

But the architecture is **bi-modal**: REQUEST goes through the FSM + pydantic-ai IR, RESPONSE is still LiteLLM-mediated for the buffered path and Gemini streaming, and hand-rolled stateful classes for the Anthropic/OpenAI/Perplexity streaming intake. The next step makes it **symmetric**: FSM in both directions, LiteLLM excised everywhere we can do without it.

## Goal — symmetric bidirectional FSM

```
Client                              ccproxy                                Provider
  │                                    │                                      │
  │── REQUEST ─────────────────────────▶│                                      │
  │   (listener wire bytes)            │                                      │
  │                                    │ FSM dispatch_load (per-listener)     │
  │                                    │    ↓                                 │
  │                                    │ ParsedRequest (pydantic-ai IR)       │
  │                                    │    ↓                                 │
  │                                    │ pipeline hooks (DAG)                 │
  │                                    │    ↓                                 │
  │                                    │ FSM dispatch_dump (per-provider) ───▶│
  │                                    │                                      │
  │                                    │◀── provider wire bytes ──────────────│
  │                                    │   (buffered or streaming SSE)        │
  │                                    │ FSM dispatch_intake (per-provider)   │
  │                                    │    ↓                                 │
  │                                    │ ParsedResponse (pydantic-ai IR,      │
  │                                    │  streaming or buffered)              │
  │                                    │    ↓                                 │
  │                                    │ response hooks (DAG, future)         │
  │                                    │    ↓                                 │
  │◀── RESPONSE ───────────────────────│ FSM dispatch_render (per-listener)   │
  │   (listener wire bytes)            │                                      │
```

When this lands, **`litellm` is removed from `pyproject.toml` entirely.** Every LiteLLM import in the codebase (`dispatch.py`, `context_cache.py`, `noop_logging.py`, `pplx.py`'s `BaseConfig`/`BaseModelResponseIterator` inheritance, `registry.py`'s `ProviderConfigManager` fallback) is replaced by native ccproxy code or direct vendor-SDK calls. The dep tree shrinks dramatically — `litellm` pulls in dozens of provider SDKs plus `tokenizers` and per-provider `httpx` clients, none of which ccproxy uses for anything but the small `BaseConfig` contract surface.

## Two reference artifacts to read first

1. **The completed request-side FSM** (`src/ccproxy/lightllm/graph/`, 7 modules, ~2580 lines):
   * `anthropic_dump.py` — canonical FSM topology: state with queue + last-emitted-block reference, `FetchNextNode` router with structural `match`, per-IR-part nodes, `ApplyCacheNode` middleware.
   * `anthropic_load.py` — inverse direction: two-phase per-message FSM (user-turn accumulator-flush, assistant-turn straightforward emission), pre-pass for two-pass tool_name lookup.
   * The same shapes apply on the response side — the topology is mature.

2. **The existing hand-rolled response scaffold** (`src/ccproxy/lightllm/response/`, 11 modules, ~1880 lines):
   * `intake.py` defines the `ResponseIntake` protocol (sync, stateful, `feed(bytes) → Iterator[ModelResponseStreamEvent]`).
   * `intake_{anthropic,openai,google,perplexity}.py` — concrete implementations. Anthropic intake drives `ModelResponsePartsManager` from `pydantic_ai._parts_manager`. The Google intake is implemented but NOT wired (addon still routes Gemini through `dispatch.py:make_sse_transformer`).
   * `render.py` + `render_{anthropic,openai}.py` — symmetric IR → listener-wire intake side.
   * `pipeline.py` — `SSEPipeline` is the sync callable installed on `flow.response.stream`. Already exists; the FSM port slots in underneath.
   * `buffered.py` — non-streaming entry point.

These are the surfaces being FSM-ified.

## Current state — what stays, what's replaced, what's deleted

### Keep (no FSM rewrite needed)

| File | Why |
|---|---|
| `pydantic_ai.messages.*` IR types | Canonical IR remains. Streaming uses `ModelResponseStreamEvent` and `ModelResponsePartsManager`. |
| `lightllm/parsed.py` | `ParsedRequest`. We'll add a sibling `ParsedResponse` envelope for the response side. |
| `lightllm/graph/*` (current 7 modules) | The completed request-side FSM stays exactly as committed. New response-side modules join it. |
| `lightllm/pplx_steps.py`, `lightllm/pplx_threads.py` | Perplexity business logic — pure Python, no LiteLLM. Untouched. |
| `Context._run_coro_sync`, `Context.parse_sync` | Worker-thread bridge — MUST stay. Same correction as Phase H of the request-side plan. |
| `inspector/addon.py` SSE-installation framework | The mechanism stays; the callable installed on `flow.response.stream` swaps. |

### Replace (FSM takes over)

| Current | Replaced by |
|---|---|
| `lightllm/response/intake_anthropic.py` (339 lines, hand-rolled state machine) | `lightllm/graph/anthropic_intake.py` — pydantic-graph FSM. State = SSE buffer + `ModelResponsePartsManager`; nodes per Anthropic SSE event type (`message_start`, `content_block_start/stop`, `content_block_delta` with text/input_json/thinking variants, `message_delta`, `message_stop`, `error`). |
| `lightllm/response/intake_openai.py` (190 lines) | `lightllm/graph/openai_intake.py` — same shape, OpenAI Chat Completions chunks. |
| `lightllm/response/intake_google.py` (148 lines, **currently dormant**) | `lightllm/graph/google_intake.py` — same shape, Google `streamGenerateContent` events. Wired into the addon, displacing `dispatch.py:make_sse_transformer` for Gemini. |
| `lightllm/response/intake_perplexity.py` (413 lines, uses `pplx_steps.render_step`) | `lightllm/graph/perplexity_intake.py` — Perplexity-specific event types. The `pplx_steps`/`render_step` helpers stay; the FSM wraps them. |
| `lightllm/response/render_anthropic.py` (303 lines) | `lightllm/graph/anthropic_render.py` — IR streaming events → Anthropic SSE wire. Symmetric to dump. |
| `lightllm/response/render_openai.py` (206 lines) | `lightllm/graph/openai_render.py` — IR streaming events → OpenAI Chat SSE wire. |
| `lightllm/response/intake.py`, `lightllm/response/render.py` (dispatchers) | Fold into `lightllm/graph/__init__.py` as `dispatch_intake` / `dispatch_render`, matching the request-side dispatcher shape. |
| `lightllm/response/pipeline.py` (`SSEPipeline`) | Move to `lightllm/graph/sse_pipeline.py`. Same mitmproxy-stream callable contract; internal driver swaps to FSM intake + render. |
| `lightllm/response/buffered.py` | Move to `lightllm/graph/buffered.py`. Buffered (non-streaming) variant. |
| `lightllm/dispatch.py:transform_to_openai` (buffered response) | `lightllm/graph/buffered.py` provides the cross-provider buffered transform via FSM intake + render. Same call site contract for `inspector/routes/transform.py:494`. |
| `lightllm/dispatch.py:SSETransformer`, `make_sse_transformer` | Deleted once Gemini intake is wired through the FSM. |
| `lightllm/dispatch.py:transform_to_provider` (Gemini request, with cachedContents) | Folded into `lightllm/graph/google_dump.py` plus a new `lightllm/graph/google_cache.py` for the `cachedContents` API. The `context_cache.py` helpers fold in too. |

### Delete outright when the FSM lands

* `lightllm/response/` subpackage — all 11 files, replaced by `lightllm/graph/*_intake.py` + `*_render.py` + `sse_pipeline.py` + `buffered.py`.
* `lightllm/dispatch.py` — all three top-level functions and the supporting classes (`MitmResponseShim`, `SSETransformer`, `make_sse_transformer`).
* `lightllm/context_cache.py` — Gemini context-caching helpers; logic folds into `lightllm/graph/google_cache.py` (or `google_dump.py` as a sub-helper).
* `lightllm/noop_logging.py` — only exists to feed LiteLLM's `Logging` interface, which `dispatch.py` is the only caller of.
* `tests/test_lightllm_dispatch.py` — replaced by graph-driven tests.
* `tests/test_response_transform.py` — replaced by graph-driven tests.
* `tests/test_sse_pipeline.py` (if present) — re-cast.

### Stays under LiteLLM

**Nothing.** After Phase S, `rg "litellm" src/` returns empty and `litellm` is dropped from `pyproject.toml`.

The previous deferral on Perplexity is reversed (see Open Design Point #5): the `BaseConfig`/`BaseModelResponseIterator` inheritance is structural-only and disappears for free once `dispatch.py` is deleted.

## Implementation order

### Phase J — Add response-side IR scaffold

* Define `ParsedResponse` dataclass in `lightllm/parsed.py`, mirroring `ParsedRequest`:
  ```python
  @dataclass
  class ParsedResponse:
      model: str
      response: ModelResponse          # pydantic-ai IR
      stream: bool                     # was the response streamed?
      raw_extras: dict[str, Any]       # provider-side fields not absorbed
  ```
* Add a streaming variant: `StreamingParsedResponse` carrying a `ModelResponsePartsManager` plus accumulated state for emitting `ModelResponseStreamEvent` per chunk.
* Decide the streaming-IR contract: directly emit pydantic-ai's `ModelResponseStreamEvent` from intake nodes, or define a thinner `RenderableEvent` enum that's easier to FSM over. Recommendation: use pydantic-ai's events directly — they're well-typed and the render side can `match` on them.

### Phase K — Anthropic response intake FSM

`lightllm/graph/anthropic_intake.py`. The Anthropic Messages SSE event types are:
* `message_start` — opens the response, carries `usage.input_tokens`.
* `content_block_start` — opens a block (text / tool_use / thinking).
* `content_block_delta` — incremental update (text delta / input_json delta / thinking delta).
* `content_block_stop` — closes the block.
* `message_delta` — usage update.
* `message_stop` — closes the response.
* `error` — error event.
* `ping` — keepalive (ignored).

**FSM topology** (GraphBuilder, mirroring the request-side load shape):

* State carries `sse_buffer: bytearray`, `parts_manager: ModelResponsePartsManager`, `current_block_index: int`, `tool_call_state: dict[int, ToolCallAccumulator]`, `raw_extras: dict[str, Any]`, and an output event queue.
* A typed dispatch envelope per Anthropic event type (`_MessageStartEvent`, `_ContentBlockStartEvent`, `_ContentBlockDeltaEvent`, `_ContentBlockStopEvent`, `_MessageDeltaEvent`, `_MessageStopEvent`, `_ErrorEvent`, `_PingEvent`, `_DoneMarker`) — Anthropic's wire types are string-discriminated, so the router `frame_next_event` reads the discriminator once and wraps each event in the matching dataclass.
* `g.decision().branch(g.match(_EventType).to(handler_step))` routes per envelope type.
* Each handler step mutates `state.parts_manager` and pushes any emitted `ModelResponseStreamEvent` into `state.events_queue`. All handlers loop back to `frame_next_event`.
* `_DoneMarker` (queue exhausted) routes to a terminal `emit_events` step that pulls the queue into the output.

**Public callable shape**: `IntakeFSM` exposes a `feed(chunk: bytes) → list[ModelResponseStreamEvent]` method. Internally each `feed` call drives one FSM run (since each chunk may contain 0+ complete events). The persistent-loop pattern in `SsePipeline` (see "Sync vs async at the response boundary" below) drives the FSM via `await intake_graph.run(state=state)`.

**Verification gate K**: parametrize the existing `tests/test_*intake_anthropic*.py` over the new FSM intake; assert identical event sequence on every fixture against the hand-rolled `response/intake_anthropic.py` until parity is verified, then collapse to FSM-only per the Phase H pattern.

### Phase L — Anthropic response render FSM

`lightllm/graph/anthropic_render.py`. Inverse direction. State: emitted byte buffer, message-id counter, current content block index. Nodes per IR `ModelResponseStreamEvent` variant (`PartStartEvent`, `PartDeltaEvent`, `FinalResultEvent`, `BuiltinToolCallEvent`). The router `take_next_event` pops from a queue of pending `ModelResponseStreamEvent`s; the decision matches on `match(PartStartEvent)`, `match(PartDeltaEvent)`, etc., routing to per-variant emitter steps that append SSE frames to a `state.out: bytearray`. Terminal step (`_RenderDone` marker) hands the accumulated bytes to `g.end_node`.

Public callable: `RenderFSM.render(events: Iterable[ModelResponseStreamEvent]) → bytes` (drives one graph run per `render` call from inside `SsePipeline._process_chunk`).

**Verification gate L**: roundtrip through Anthropic intake → Anthropic render produces byte-equivalent SSE up to canonical normalization. Same parametrize-then-collapse pattern as Phase B.

### Phase M — OpenAI response intake + render FSM

Symmetric to K + L. OpenAI Chat Completions SSE is simpler (no per-event "block lifecycle" — just `choices[].delta.{content, tool_calls}` accumulation), so the FSM has fewer per-type branches. Same `take_next_event` → `decision()` → per-variant-step → loop-back topology.

### Phase N — Google response intake FSM (wire Gemini through the graph package)

`lightllm/graph/google_intake.py`. Google `streamGenerateContent` events: each chunk is a `GenerateContentResponse` with `candidates[].content.parts` deltas, `usageMetadata`, optional `cachedContent`, `safetyRatings`, `groundingMetadata`.

The cloudcode-pa envelope (`{response: {...}}`) unwrap moves into the intake — currently it's handled twice (once in `inspector/gemini_addon.py:EnvelopeUnwrapStream` for streaming, once in `hooks/gemini_envelope.py:unwrap_buffered` for buffered). After this phase, the intake handles unwrapping uniformly via a `_GeminiUnwrap` envelope step that consumes the outer `response` wrapper before the per-part dispatch runs.

**Bonus opportunity — capacity fallback as reducer**: `inspector/gemini_addon.py` currently sticky-retries on 429/503 then walks `fallback_models`. With GraphBuilder, this becomes a `g.join(ReduceFirstValue, ...)` where the join races the original model + fallback models in parallel, and the first successful response wins via `ReducerContext.cancel_sibling_tasks()`. Defer to a Phase O.5 — not strictly needed for the FSM migration, but the primitive is now available.

**Critical**: this phase deletes `dispatch.py:SSETransformer` + `make_sse_transformer` since Gemini was the last caller (Anthropic and OpenAI already use `SSEPipeline` from `response/`). Update `inspector/addon.py:233` to remove the Gemini branch and route everything through the unified `dispatch_intake`.

### Phase O — Google + Gemini request fold-in

Per `docs/gemini.md`, the Gemini surface is overwhelmingly hooks-driven:
* **Sentinel-key flows** (Gemini SDK, Glass) — `gemini_cli` hook does the v1internal envelope wrap + path rewrite + header masquerade. **No `dispatch.py` involvement.**
* **Response unwrap** — `hooks/gemini_envelope.py` (buffered + streaming). **No `dispatch.py` involvement.**
* **Capacity fallback** — `inspector/gemini_addon.py`. **No `dispatch.py` involvement.**
* **Cross-format transform** (scenario 3: OpenAI-format client → Gemini upstream) — this is the ONE Gemini path that goes through `dispatch.py:transform_to_provider` → `_transform_gemini` (line 82 imports `_get_gemini_url` and `_transform_request_body` from LiteLLM).

So Phase O is small: route the cross-format Gemini transform through the existing `render_google_dump` (already in `lightllm/graph/google_dump.py`, already uses pydantic-ai's `GoogleModel` not LiteLLM), and inline the cachedContents helpers.

Specifically:
* Update `inspector/routes/transform.py:321-351` Gemini branch to call `dispatch_dump_sync(parsed, provider="gemini")` (matches the non-Gemini branch). The `google_dump.py` FSM already produces the right body shape for Gemini's standard `generateContent`; the `gemini_cli` outbound hook handles the v1internal envelope wrap downstream.
* Inline `context_cache.py`'s LiteLLM helpers (`is_cached_message`, `is_prompt_caching_valid_prompt`, `ContextCachedContent`) into `lightllm/graph/google_cache.py` as ~30 lines of native code. The cachedContents API itself (`POST /v1beta/cachedContents`) is callable directly via httpx — no LiteLLM intermediary needed.
* Add a `cached_content` hook in the Gemini outbound chain that resolves the cached resource ID and stamps it onto the body before `gemini_cli` runs, OR fold the resolution into `google_dump.py` directly (Recommendation: hook — keeps the FSM stateless and matches the existing hook-pipeline architecture).

After this phase: `dispatch.py:_transform_gemini`, `dispatch.py:transform_to_provider`'s Gemini branch, `context_cache.py`, and `noop_logging.py` all delete. `registry.py`'s `ProviderConfigManager` fallback deletes too.

**Verification gate O**: smoke an OpenAI-format request hitting a Gemini-back provider via transform rule; assert the upstream-bound body matches the pre-refactor wire shape (use `ccproxy flows compare`).

### Phase P — Perplexity LiteLLM removal + response intake FSM

`pplx.py:PerplexityProConfig` inherits `BaseConfig` and overrides 7 methods (`get_supported_openai_params`, `map_openai_params`, `validate_environment`, `get_complete_url`, `transform_request`, `transform_response`, `get_model_response_iterator`). `pplx.py:PerplexityProIterator` inherits `BaseModelResponseIterator` and overrides 1 method (`chunk_parser`). **Every reachable method is overridden** — the inheritance is structural-only, present so `dispatch.py` could call methods uniformly across Perplexity and upstream LiteLLM providers.

When `dispatch.py` deletes (Phase R), nothing calls those methods through the BaseConfig contract anymore. The FSM intake calls `chunk_parser` directly; the FSM dump (Phase G `render_perplexity_pro_dump` already exists) calls `_build_pplx_payload` directly. So:

* Drop `class PerplexityProConfig(BaseConfig)` → `class PerplexityProConfig` (plain class). Keep all method bodies; they don't depend on any inherited behavior.
* Drop `class PerplexityProIterator(BaseModelResponseIterator)` → `class PerplexityProIterator` (plain class). `chunk_parser` becomes a plain method (or moves into the FSM intake nodes directly).
* `PerplexityException(BaseLLMException)` → swap base to a local `LightllmException(Exception)` carrying `status_code`. Same 5-line definition we'd otherwise import.
* Build `lightllm/graph/perplexity_intake.py` — the Perplexity SSE has its own JSONL-over-SSE shape with step events, file attachments, citation metadata. The existing `intake_perplexity.py` (413 lines, uses `pplx_steps.render_step`) defines the chunk parsing rules; the FSM ports it to per-step-type nodes routed by `match` on the chunk's `type` field. The `pplx_steps` and `pplx_threads` helpers stay untouched — pure Python business logic, no LiteLLM.

**Why this is trivial**: every LiteLLM symbol in `pplx.py` is structural. `BaseConfig` gives us nothing we use — every relevant method is overridden. `BaseModelResponseIterator` gives us a `chunk_parser` slot, but ccproxy is the only caller and the FSM intake replaces it. Net change to `pplx.py`: ~10 line diff to drop two `(BaseConfig)` and `(BaseModelResponseIterator)` annotations and replace `BaseLLMException` with our own.

### Phase Q — Unified dispatcher in `lightllm/graph/__init__.py`

After all per-provider intakes/renders exist, expose:
```python
def dispatch_intake(
    *, upstream_provider: str, model: str, request_params: ModelRequestParameters
) -> ResponseIntakeFSM: ...

def dispatch_render(*, listener_format: ListenerFormat) -> ResponseRenderFSM: ...
```
These mirror `dispatch_load` / `dispatch_dump_sync` and let `inspector/addon.py` install the streaming pipeline with one entry point:
```python
intake = dispatch_intake(upstream_provider=..., model=..., request_params=...)
render = dispatch_render(listener_format=...)
pipeline = SSEPipeline(intake=intake, render=render)
flow.response.stream = pipeline
```

### Phase R — Buffered response transform FSM

`lightllm/graph/buffered.py` provides `transform_buffered_response_sync(*, raw_bytes, upstream_provider, listener_format, model, request_params) → bytes`. Drives an intake FSM on the full response body (no streaming), then a render FSM to emit the listener-wire body. Replaces `dispatch.py:transform_to_openai` at `inspector/routes/transform.py:494`.

### Phase S — Delete the response/ subpackage, dispatch.py, and litellm itself

Once Phases K–R are green:
* Delete `lightllm/response/` (all 11 files).
* Delete `lightllm/dispatch.py`, `lightllm/context_cache.py`, `lightllm/noop_logging.py`.
* Drop the `(BaseConfig)` / `(BaseModelResponseIterator)` / `(BaseLLMException)` bases in `lightllm/pplx.py`. Replace `BaseLLMException` with a local `LightllmException(Exception)`.
* Simplify `lightllm/registry.py` — only Perplexity is local-registered, no more LiteLLM `ProviderConfigManager` fallback.
* Update `lightllm/__init__.py` exports.
* **Remove `litellm` from `pyproject.toml [project.dependencies]`.** Run `uv sync` and verify nothing imports `litellm.*` anymore (`rg "^(from|import) litellm" src/ tests/` must return empty).
* Delete `tests/test_lightllm_dispatch.py`, `tests/test_response_transform.py`.
* Re-point any remaining test mocks (likely in `tests/test_transform_routes.py` and `tests/test_inspector_*.py`).

### Phase T — End-to-end smoke

* Anthropic via inspector (the same scenario validated in the request-side Phase I).
* Gemini via inspector — first smoke after Phase N+O, then again after Phase R.
* OpenAI-format listener → Anthropic upstream (cross-format transform of both request AND response).
* Cross-format response: send an Anthropic request to ccproxy with `?listener=openai`-equivalent path (or use a transform rule), assert the response comes back in OpenAI Chat Completions SSE format.
* Perplexity Pro via the OpenAI SDK pointing at ccproxy — full request + response roundtrip.

## Architectural recipe (response-side specifics)

### GraphBuilder is the FSM idiom

The request-side phase migrated to `pydantic_graph.beta.GraphBuilder` (see the request-side plan at `/home/***/.claude/plans/here-i-ve-done-a-ticklish-torvalds.md` and `lightllm/graph/anthropic_dump.py` as the canonical reference). Every response-side FSM in this plan follows the same idiom:

* State as a plain `@dataclass` with mutable accumulators.
* `@g.step` async functions taking `StepContext[State, None, InputT]` and returning the next typed value (or a sentinel marker for end-of-graph).
* `g.decision().branch(g.match(Type).to(step))` for type-discriminated routing.
* Typed dispatch envelopes (one frozen dataclass per discriminator value) when the wire uses string-discriminated unions — pydantic-graph matches on Python types, not runtime strings.
* `g.add(g.edge_from(...).to(...))` for explicit edges + loop-back.
* `graph.render(title=..., direction='LR')` for mermaid diagrams in docs and debugging.
* `g.join(reducer, initial=)` for parallel aggregation (used in Phase O.5 capacity fallback).

The 4 migrated request-side files are the reference; the response-side files should mirror their shape exactly.

### Streaming state shape

The response intake is *append-only-with-lookback* — chunks arrive in order and the FSM must accumulate parts incrementally without seeing the future. The state owns:
* `sse_buffer: bytearray` — incomplete SSE frame bytes between feed() calls.
* `parts_manager: ModelResponsePartsManager` — pydantic-ai's helper for streaming part accumulation.
* `current_block_index: int` — which content block is being assembled (Anthropic only).
* `tool_call_state: dict[int, ToolCallAccumulator]` — per-tool-call argument accumulators (OpenAI delta-as-string-fragment pattern).
* `raw_extras: dict[str, Any]` — provider-side response metadata (usage, citations, safety, groundingMetadata) that the IR doesn't absorb.

A `_emit_event(state, event)` helper appends to an internal event queue that `feed()` drains and yields. This is the dual of the request-side `_append_block` helper.

### Cache-control on the response side

Responses don't carry `cache_control` markers; they carry `cache_creation_input_tokens` / `cache_read_input_tokens` in `usage`. These ride on `raw_extras["usage"]` and the render side decides how to surface them in the listener wire format.

### `raw_extras` parity

The intake's `raw_extras` mirror the request-side conventions:
* `usage:msg:0` — Anthropic per-message usage delta.
* `safety:msg:0:rating:0` — Gemini safety ratings.
* `citations:msg:0` — Perplexity per-message citations.
* Unknown event types: `unknown_event:msg:0:event:N` → stash whole event dict.

Render-side stitches them back onto the wire body (matching how `_stitch_raw_extras` works on the request side).

### Sync vs async at the response boundary — the streaming overhead problem

**The trap**: the request-side worker-thread bridge (`_run_coro_sync` in `pipeline/context.py:27-53`) spawns a `ThreadPoolExecutor(max_workers=1)` and tears it down per invocation. That's fine for `Context.parse_sync` (one call per request). Applied per-chunk on a streaming response, it's pathological — ~200 chunks in a 5-second stream means 200 thread spawns plus 200 fresh asyncio loops.

**Architectural decision (baked, not deferred)**: **persistent asyncio loop in a dedicated daemon thread per `SSEPipeline` instance.** Lifecycle:

```python
import asyncio, threading
from concurrent.futures import Future

class SSEPipeline:
    """Sync mitmproxy stream callable backed by a persistent asyncio loop."""

    def __init__(self, intake: IntakeFSM, render: RenderFSM) -> None:
        self._intake = intake
        self._render = render
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="ccproxy-sse-loop"
        )
        self._thread.start()
        self._closed = False

    def __call__(self, data: bytes) -> bytes | Iterable[bytes]:
        # Submit to the persistent loop; block until result.
        future: Future[bytes] = asyncio.run_coroutine_threadsafe(
            self._process_chunk(data), self._loop
        )
        return future.result()

    async def _process_chunk(self, data: bytes) -> bytes:
        out = bytearray()
        async for event in self._intake.feed(data):
            out += await self._render.render(event)
        if not data:  # mitmproxy's end-of-stream sentinel
            out += await self._render.terminator()
        return bytes(out)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1.0)
```

**Per-chunk overhead**: cross-thread future submission + wait, ~10-50µs typical. Negligible against the ~10-100ms-per-chunk floor set by upstream network I/O. SSE delivery is dominated by upstream response timing, not parsing.

**Why not just keep the existing sync intake/render classes** (today's `response/intake_*.py` and `response/render_*.py`): they DO work, and we could keep them sync forever. But then the response side has a different aesthetic than the request side — no `match`-based router, no `ApplyDeltaNode` middleware, no `GraphRunContext.history` for debugging streaming bugs. The user's stated goal is symmetric FSM in both directions; the persistent-loop pattern is what makes that affordable.

**Why not a custom sync FSM mimicking pydantic-graph**: we'd lose pydantic-graph's mermaid diagram generation, persistence interface, and the muscle memory the team just built in the request-side phase. Not worth the code we'd write to dodge ~10µs.

**Fallback path** if profiling shows Option A is meaningful (it shouldn't be): drop the streaming intake/render back to plain sync classes (today's shape). Buffered (non-streaming) response transform always uses the FSM via `dispatch_dump_sync`-style bridge — those are one-shot like the request side.

**Lifecycle wiring**: `inspector/addon.py:_install_sse_transformer` already creates one `SSEPipeline` per request. The `.close()` call goes onto `done` event or in `responsebody` hook when mitmproxy signals end-of-stream. Belt-and-suspenders: daemon thread means a missed `.close()` won't leak (the thread dies with the process), but explicit cleanup is preferred.

## Open design points

1. **Streaming IR — pydantic-ai's events or our own enum?** `pydantic_ai.messages.ModelResponseStreamEvent` is the union we'd consume from intake. It's: `PartStartEvent | PartDeltaEvent | FinalResultEvent | BuiltinToolCallEvent`. Render-side has to pattern-match on these. Risk: pydantic-ai may evolve the event shape and break us. Mitigation: pin pydantic-ai version (already a direct dep) and add a thin event-adapter layer if drift becomes a problem.

2. **Per-chunk FSM run vs single-FSM-spanning-the-stream**. Two options:
   * Option A: one graph run per `feed(bytes)` call. State persists across calls outside the graph.
   * Option B: one graph run for the whole stream, with `feed()` pushing onto an async queue the FSM consumes.
   Option A is simpler and matches the request-side per-message pattern. Option B gives full `GraphRunContext.history` traceability for the whole response. Recommendation: start with A; switch to B if debugging benefits show up.

3. ~~Worker-thread overhead per chunk~~ **Resolved**: persistent-loop pattern (one asyncio loop in a daemon thread per `SSEPipeline` instance, cross-thread future submission per chunk). See "Sync vs async at the response boundary" above. ~10-50µs per chunk is well below the network-I/O floor; no per-chunk thread spawn.

4. **Should the response side have its own pipeline-hooks DAG?** The request side has DAG-driven hooks between IR creation and dump (forward_oauth, gemini_cli, shape, etc.). A symmetric response-side DAG could fold the response unwrap, capacity fallback, and OAuth 401-retry logic into hooks. Out of scope for this plan but the FSM architecture invites it.

5. ~~Perplexity LiteLLM coupling — keep or replace?~~ **Resolved**: replace. `PerplexityProConfig` and `PerplexityProIterator` override every reachable method of their respective LiteLLM bases (7 + 1). The inheritance is structural-only — once `dispatch.py` is gone, nothing calls through `BaseConfig`. Drop the bases; the classes become standalone with their existing method bodies intact. See Phase P.

6. **`ParsedResponse` envelope shape**. Mirror `ParsedRequest` (model, IR, stream, raw_extras) or carry richer metadata (provider, request_params back-reference, OTel span context)? Recommendation: mirror; the rest is sidecar state on the FSM run.

7. **Buffered vs streaming code-path unification**. Currently `response/buffered.py` and `response/pipeline.py` are separate. The FSM intake can be driven for either case (one-shot for buffered, chunk-fed for streaming). Phase R could unify them under one entry point that takes a `bytes | AsyncIterator[bytes]`.

## Reference: current commit history

```
4dd9765   chore: disable mypy errors for pydantic_graph TypeVar inference
d6007ea   refactor(ccproxy): replace user-turn nodes with GraphBuilder functions
<base sha> refactor(ccproxy): migrate lightllm wire layer to pydantic-graph FSM
9e8aa30   cleaned up old plan files
016d7d1   fix(ccproxy): worker-thread fallback for sync IR bridges in async hooks
6e3fc46   refactor(ccproxy): migrate Context typed properties to IR, delete wire.py
710761e   feat(ccproxy): rewire inspector to use pydantic-ai-mediated wire layer
819e9cb   feat(ccproxy): add SSEPipeline, buffered renderer, and TransformMeta fields
43ad06c   feat(ccproxy): introduce pydantic-ai-mediated wire layer in lightllm/
```

The request-side FSM landed across two commits: the BaseNode-style initial drop, then the GraphBuilder migration. The 4 FSM files in `lightllm/graph/` (`anthropic_dump.py`, `anthropic_load.py`, `openai_dump.py`, `openai_load.py`) now use `pydantic_graph.beta.GraphBuilder` and are the canonical pattern for everything in this plan. This plan picks up from there.

## Notes for the lead next session

* The plan file from the request-side phase is at `/home/***/.claude/plans/here-i-ve-done-a-ticklish-torvalds.md`. The Wire-type discipline section + branch coverage matrix from that doc apply verbatim to the response side — copy the patterns.
* Test count baseline: **1689 passing**, 2 pre-existing failures (`test_fastmcp_instructions_block_configured`, `test_blacklisted_domain_gets_default_response`).
* mypy: 11 pre-existing errors in `pplx.py`, `addon.py`, `pplx_thread_inject.py`. Not caused by FSM work; not blocking but worth fixing in a side-pass before Phase J starts.
* `--cov-fail-under=90` is currently failing at 82.41% (baseline 82.38%). Not caused by FSM work either. The response-side rewrite will likely move it further if test parity isn't preserved at parametrize-then-collapse — apply the same discipline as Phase H.
* `~/.claude/.credentials.json` confirmed present in the dev environment; Phase T smoke 1 (Anthropic) already verified working post-Phase I.
* The biggest payoff after this plan is shipping: **single IR boundary in both directions, single FSM idiom, single dispatcher pattern, LiteLLM gone except for the locally-registered Perplexity provider's iterator contract**. The bi-modal cognitive tax disappears; new providers add via a uniform "add four files: load, dump, intake, render" recipe.
