# picture_v2 Research Report — ChatGPT Image Generation via `/backend-api/f/conversation`

**Purpose**: Decision-support comparison of how four reference implementations drive ChatGPT
image generation and image edit through `POST /backend-api/f/conversation`, for ccproxy's
`openai_conversations` provider (CHATGPT-006).

**References**:

| Ref | Language | On-disk | Kitstore |
|---|---|---|---|
| **gproxy** | Rust | `/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/` | `.kitstore/gproxy` |
| **gproxy-protocol** | Rust | n/a | `.kitstore/gproxy-protocol` |
| **pro-cli** | TypeScript | `/home/eigenmage/dev/scratch/ccproxy/refs/pro-cli/` | `.kitstore/pro-cli` |
| **aurora** | Go | `/home/eigenmage/dev/scratch/ccproxy/refs/aurora/` | `.kitstore/aurora` |

---

## Executive Summary

| Decision point | gproxy | pro-cli | aurora | **Recommended for ccproxy** |
|---|---|---|---|---|
| **Generation trigger** | `system_hints: ["picture_v2"]` (from `tools[{type:"image_generation"}]` mapping or native pass-through) | Prompt prefix only; `system_hints: []` | `system_hints: ["picture_v2"]` at top-level **and** `messages[0].metadata.system_hints` | **Both**: `system_hints: ["picture_v2"]` + pro-cli prefix prepended to prompt |
| **Prefix string** | None | `"Use ChatGPT image generation tools to create image(s) from the user's prompt. Do not answer with only a revised prompt. After generation, keep any text brief."` | None | Adopt pro-cli prefix verbatim (defense-in-depth; gproxy/aurora have direct API access, ccproxy does not) |
| **Model routing** | Pass client model slug through `resolve_model()` (passthrough; `""` → `"gpt-5-4"`) | `DEFAULT_MODEL` (the default chat model) regardless of client-requested model | `imageModelSlug()`: `gpt-image-*` / `dall-e-*` → `"auto"` | Map `gpt-image-*` → `"auto"`; otherwise use configured `default_model` (`gpt-5-5-pro`) |
| **Saved vs temporary** | `temporary_chat` is caller-controlled; NOT forced for images | Images explicitly excluded from `temporary_chat` (`temporary && !isImage` guard); error detection for temporary-chat blocks | No explicit guard (images always use saved conversation implicitly) | Force `temporary_chat: false` for all image requests (polling 404s on temporary conversations) |
| **Poll cadence** | 3s interval, 180s deadline | 3s interval (30s on 429), caller-supplied timeout | 2s interval, 45 retries (90s max) | 3s interval, 120s deadline |
| **Poll node filters** | `role == "tool"` + `async_task_type == "image_gen"` + `content_type == "multimodal_text"` | `content_type == "image_asset_pointer"` (part-level) + `async_status` terminal check | Generic deep-walk of all JSON values; no role/task filter | gproxy filters (most precise and explicit) |
| **Download Step 1 routing** | By id prefix: `file_` / `file-` → `/backend-api/files/download/{id}?conversation_id={cid}&inline=false`; other (sediment bare id) → `/backend-api/conversation/{cid}/attachment/{id}/download` | Always `/backend-api/files/download/{encodeURIComponent(fileId)}` (strips `sediment://` prefix) | Single path: `/backend-api/files/{asset_after_scheme}/download` for all schemes | gproxy routing (handles both schemes; most precise) |
| **Authorization on presigned GET** | OMITTED — explicit code comment: "pre-signed sig= in the querystring and no Authorization header; sending Bearer here causes the server to 403" | Uses same-origin browser credentials (no explicit Authorization omission) | INCLUDED (sends `Authorization: Bearer` on download bytes) — probable bug | OMIT Authorization on presigned download (gproxy comment is authoritative) |
| **Upload Step 3 endpoint** | `POST /backend-api/files/process_upload_stream` with `{file_id, use_case, index_for_retrieval, file_name, library_persistence_mode, metadata}` | Not implemented | `POST /backend-api/files/{file_id}/uploaded` with `{}` body | gproxy `/process_upload_stream` (more explicit; aurora's endpoint may be a newer but less-documented variant — flag as risk) |
| **Output shape** | `{"created": <unix>, "data": [{"b64_json": "...", "revised_prompt": ""}]}` | Per-file artifact on disk; no OpenAI envelope returned to caller (CLI-only) | `ImageGenerationResult{URL, B64JSON}` — caller wraps | OpenAI `images.response`: `{"created": <unix>, "data": [{"b64_json": "..."}]}` |

---

## §1 Generation Trigger

The central question: what makes ChatGPT's `/f/conversation` actually invoke the image-generation tool?

### gproxy

**Mechanism**: `system_hints: ["picture_v2"]` injected into the top-level conversation body. The hint is derived from the incoming OpenAI request in three ways (in priority order):

1. Native pass-through: `body.system_hints` → forwarded as-is.
2. `extra_body.system_hints` → forwarded.
3. OpenAI Responses `tools[{type: "image_generation"}]` → mapped to `"picture_v2"` via `openai_tool_to_hint()`.

```rust
// request_builder.rs:301-307
fn openai_tool_to_hint(tool_type: &str) -> Option<&'static str> {
    match tool_type {
        "image_generation" => Some("picture_v2"),
        "web_search" | "web_search_preview" | "web_search_preview_2025_03_11" => Some("search"),
        "deep_research" => Some("connector:connector_openai_deep_research"),
        _ => None,
    }
}
```

`extract_system_hints()` deduplicates all three sources into a single `Vec<String>`:
`/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/request_builder.rs:263-295`

For image gen, `body_map.insert("system_hints", json!(["picture_v2"]))` only appears when the client sends a tool or native hint. **No prompt prefix is prepended.** No explicit `tools` declaration in the outbound body.

The model slug for image requests: the incoming model (`gpt-image-1`, etc.) passes through `resolve_model()` verbatim — no mapping to `"auto"`:

```rust
// request_builder.rs:243-251
pub fn resolve_model(requested: &str) -> String {
    const DEFAULT: &str = "gpt-5-4";
    let trimmed = requested.trim();
    if trimmed.is_empty() { DEFAULT.to_string() } else { trimmed.to_string() }
}
```

`/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/request_builder.rs:243-251`

**`temporary_chat`** is NOT forced for images; it is a caller-controlled setting. Image routes use whatever the client provides:
`/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/channel.rs:423`

### pro-cli

**Mechanism**: Prompt prefix only. `system_hints` is hardcoded to `[]` for all non-research requests, including images:

```typescript
// transport.ts:1026
system_hints: isResearch ? [`connector:${DEEP_RESEARCH_CONNECTOR_ID}`] : [],
```

`/home/eigenmage/dev/scratch/ccproxy/refs/pro-cli/src/transport.ts:1026`

The prompt is rewritten by `buildImagePrompt()`:

```typescript
// transport.ts:1074-1083
function buildImagePrompt(job: JobRecord): string {
  const prompt = job.prompt.trim();
  const imageInstructions =
    "Use ChatGPT image generation tools to create image(s) from the user's prompt. Do not answer with only a revised prompt. After generation, keep any text brief.";
  const instructions = [
    imageInstructions,
    stringOption(job.options.instructions),
  ].filter((part): part is string => Boolean(part && part.trim())).join("\n\n");
  return `${instructions.trim()}\n\n${prompt}`;
}
```

`/home/eigenmage/dev/scratch/ccproxy/refs/pro-cli/src/transport.ts:1074-1083`

The model for image requests is `DEFAULT_MODEL` (the default chat model), **not** the client-requested image model:

```typescript
// transport.ts:998
const model = isResearch ? DEEP_RESEARCH_ROUTER_MODEL : isImage ? DEFAULT_MODEL : normalizeModel(job.model);
```

`/home/eigenmage/dev/scratch/ccproxy/refs/pro-cli/src/transport.ts:998`

Image explicitly excluded from `temporary_chat`:

```typescript
// transport.ts:1037-1040
if (temporary && !isImage) {
  body.history_and_training_disabled = true;
  body.client_contextual_info = { app_name: "chatgpt.com" };
}
```

`/home/eigenmage/dev/scratch/ccproxy/refs/pro-cli/src/transport.ts:1037-1040`

Also detects upstream error when image generation is attempted in a temporary chat:

```typescript
// transport.ts:223-226
if (parsed.text.toLowerCase().includes("image generation isn") && parsed.text.toLowerCase().includes("temporary chat")) {
  throw new ProError("IMAGE_TEMPORARY_UNAVAILABLE", "ChatGPT image generation is not available in temporary chats.", {
    suggestions: ["Use a saved ChatGPT conversation for image generation; omit --temporary."],
  });
```

`/home/eigenmage/dev/scratch/ccproxy/refs/pro-cli/src/transport.ts:223-226`

### aurora

**Mechanism**: `system_hints: ["picture_v2"]` injected at **two** locations:

1. Top-level conversation body field.
2. **Also** in `messages[0].metadata.system_hints`.

```go
// request.go:2083-2114
payload := map[string]interface{}{
    ...
    "messages": []map[string]interface{}{{
        ...
        "metadata": map[string]interface{}{
            "developer_mode_connector_ids": []interface{}{},
            "selected_github_repos":        []interface{}{},
            "selected_all_github_repos":    false,
            "system_hints":                 []string{"picture_v2"},   // ← in message metadata
            "serialization_metadata":       map[string]interface{}{"custom_symbol_offsets": []interface{}{}},
        },
    }},
    ...
    "system_hints": []string{"picture_v2"},   // ← also top-level
    ...
}
```

`/home/eigenmage/dev/scratch/ccproxy/refs/aurora/internal/chatgpt/request.go:2083-2114`

**No prompt prefix.** No `tools` declaration in the outbound body. Also uses a separate `prepareImageConversation()` call that sends `system_hints: ["picture_v2"]` to `POST /f/conversation/prepare` as well:
`/home/eigenmage/dev/scratch/ccproxy/refs/aurora/internal/chatgpt/request.go:2022-2071`

### gproxy-protocol

Not relevant to image generation via `/f/conversation`. gproxy-protocol defines SSE event types for the **OpenAI Responses API** `image_generation_call` output item path (used when the client sends `tools: [{type: "image_generation"}]` to a Responses-API-capable upstream). This is a different pathway (Codex/Responses API) rather than the ChatGPT Conversations wire. Reference for completeness only:
`.kitstore/gproxy-protocol/src/openai/create_image/stream.rs:46-56`

### Side-by-Side

| Mechanism | gproxy | pro-cli | aurora |
|---|---|---|---|
| `system_hints: ["picture_v2"]` in body | Yes (derived from tools or native pass-through) | **No** (`system_hints: []`) | Yes (at top-level) |
| `system_hints: ["picture_v2"]` in `messages[0].metadata` | No | No | **Yes** (defense-in-depth) |
| `system_hints: ["picture_v2"]` in `/f/conversation/prepare` | No separate prepare for images | No prepare call | **Yes** |
| Prompt prefix injected | No | **Yes** — verbatim string cited above | No |
| `tools` declaration in outbound body | No | No | No |
| Most current / reliable signal | Relies on accurate tool mapping | Browser-context CDP; known to work in production | Direct API; dual injection maximizes signal |

**Assessment**: gproxy and aurora rely on `system_hints: ["picture_v2"]` from a privileged API context where token is freshly minted and PoW is high-trust. pro-cli runs in a real browser context where `system_hints` may be ignored for untrusted clients and uses the prompt prefix as the signal instead. ccproxy is in a position analogous to gproxy/aurora (direct API with bearer token + PoW), so the hint should work — but since we cannot live-test, combining both maximizes reliability.

---

## §2 Image Model Routing

### gproxy

`resolve_model()` is a pass-through; empty string defaults to `"gpt-5-4"`:
`/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/request_builder.rs:243-251`

For image requests, the `CreateImage` path extracts only `prompt` from the OpenAI body and builds a bare chat-like body without inspecting the requested model:
`/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/channel.rs:395-413`

Result: `gpt-image-1` would be passed through to `/f/conversation` verbatim, which is likely incorrect (chatgpt.com's model catalog doesn't include `gpt-image-*` slugs). This is a gap in gproxy.

### pro-cli

Image requests always use `DEFAULT_MODEL` (the default chat model) regardless of what the client requested:

```typescript
// transport.ts:998
const model = isResearch ? DEEP_RESEARCH_ROUTER_MODEL : isImage ? DEFAULT_MODEL : normalizeModel(job.model);
```

`/home/eigenmage/dev/scratch/ccproxy/refs/pro-cli/src/transport.ts:998`

`DEFAULT_MODEL` is whatever the chat default is (e.g. `gpt-4o`, `gpt-5-5-pro`). chatgpt.com's image generation is triggered by the `picture_v2` hint — the model slug just selects which chat model serves the turn; the image tool call is a side effect.

### aurora

Explicitly maps `gpt-image-*` → `"auto"`:

```go
// request.go:1981-1989
func imageModelSlug(model string) string {
    if model == "" || strings.HasPrefix(model, "dall-e") {
        model = "gpt-image-2"
    }
    if model == "gpt-image-2" || strings.HasPrefix(model, "gpt-image") {
        return "auto"
    }
    return model
}
```

`/home/eigenmage/dev/scratch/ccproxy/refs/aurora/internal/chatgpt/request.go:1981-1989`

`"auto"` in this context is chatgpt.com's signal to let the server choose the best model for the task. aurora's mapping: `dall-e-*` → normalizes to `gpt-image-2` → maps to `"auto"`; any other `gpt-image-*` → `"auto"`.

### Assessment

aurora's mapping is the most correct: `gpt-image-*` models are OpenAI API slugs that chatgpt.com's conversation endpoint doesn't recognize; sending `"auto"` lets chatgpt.com route to its image generation pipeline. For non-image-model slugs (e.g. `gpt-5-5-pro`), pass through unchanged — this is what the configured `default_model` would be.

---

## §3 Full Image-Generation Request Body

### gproxy

`build_conversation_body()` at `/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/request_builder.rs:12-71`:

```json
{
  "action": "next",
  "messages": [{"id": "<uuid>", "author": {"role": "user"}, "create_time": <unix>, "content": {"content_type": "text", "parts": ["<prompt>"]}, "metadata": {...}}],
  "parent_message_id": "client-created-root",
  "model": "<resolved_model>",
  "client_prepare_state": "sent",
  "timezone_offset_min": -480,
  "timezone": "Asia/Shanghai",
  "conversation_mode": {"kind": "primary_assistant"},
  "enable_message_followups": true,
  "system_hints": ["picture_v2"],
  "supports_buffering": true,
  "supported_encodings": ["v1"],
  "client_contextual_info": {
    "is_dark_mode": false, "time_since_loaded": 5000,
    "page_height": 1039, "page_width": 1237, "pixel_ratio": 1.35,
    "screen_height": 1067, "screen_width": 1707, "app_name": "chatgpt.com"
  },
  "paragen_cot_summary_display_override": "allow",
  "force_parallel_switch": "auto"
}
```

Image-specific fields vs text: `system_hints: ["picture_v2"]` (if client sends `tools: [{type: "image_generation"}]`). `history_and_training_disabled` only present when `temporary_chat: true`, which the image path does NOT force. No `conversation_id`.

### pro-cli

From `buildRequestBody()` at `/home/eigenmage/dev/scratch/ccproxy/refs/pro-cli/src/transport.ts:993-1048`:

```json
{
  "action": "next",
  "messages": [{"id": "<uuid>", "author": {"role": "user"}, "create_time": <unix_sec>, "content": {"content_type": "text", "parts": ["<image_instructions>\n\n<prompt>"]}, "metadata": {}}],
  "model": "<DEFAULT_MODEL>",
  "parent_message_id": "client-created-root",
  "client_prepare_state": "none",
  "timezone_offset_min": <local_tz_offset>,
  "timezone": "<local_tz>",
  "conversation_mode": {"kind": "primary_assistant"},
  "enable_message_followups": true,
  "system_hints": [],
  "supports_buffering": true,
  "supported_encodings": ["v1"],
  "client_contextual_info": {"app_name": "chatgpt"},
  "paragen_cot_summary_display_override": "allow",
  "force_parallel_switch": "auto"
}
```

Image-specific: `system_hints: []` (empty), `client_prepare_state: "none"`, prompt rewritten by `buildImagePrompt()`. `history_and_training_disabled` NOT set for images (`temporary && !isImage` guard excludes it). `client_contextual_info.app_name` is `"chatgpt"` (not `"chatgpt.com"` — that only applies to temporary chat).

### aurora

From `GeneratePictureConversationImages()` at `/home/eigenmage/dev/scratch/ccproxy/refs/aurora/internal/chatgpt/request.go:2083-2114`:

```json
{
  "action": "next",
  "messages": [{
    "id": "<uuid>",
    "author": {"role": "user"},
    "create_time": <unix_sec>,
    "content": {"content_type": "text", "parts": ["<prompt>"]},
    "metadata": {
      "developer_mode_connector_ids": [],
      "selected_github_repos": [],
      "selected_all_github_repos": false,
      "system_hints": ["picture_v2"],
      "serialization_metadata": {"custom_symbol_offsets": []}
    }
  }],
  "parent_message_id": "<state.ParentMessageID>",
  "model": "auto",
  "client_prepare_state": "sent",
  "timezone_offset_min": 420,
  "timezone": "America/Los_Angeles",
  "conversation_mode": {"kind": "primary_assistant"},
  "enable_message_followups": true,
  "system_hints": ["picture_v2"],
  "supports_buffering": true,
  "supported_encodings": ["v1"],
  "client_contextual_info": {"<from state>": "..."},
  "paragen_cot_summary_display_override": "allow",
  "force_parallel_switch": "auto",
  "thinking_effort": "standard"
}
```

Image-specific: `system_hints: ["picture_v2"]` at top-level AND in `messages[0].metadata.system_hints`; model is `"auto"` (mapped from any `gpt-image-*`); `thinking_effort: "standard"` hardcoded; no `history_and_training_disabled`.

---

## §4 SSE Pointer Extraction + Async Polling

### gproxy

**SSE extraction** (`extract_image_pointers()` at `/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/image.rs:28-95`):

Walks all SSE-v1 delta events. For each patch:
- `op: "add", path: ""` — whole-message add event: extracts `conversation_id`, walks `message.content.parts` for `asset_pointer` fields via `collect_pointers_from_parts()`.
- `op: "append" | "replace", path: "/message/content/parts/0"` — text delta: scans for `file-service://` or `sediment://` substrings via `scan_text_for_pointers()`.
- `op: "replace", path: "/message/content/parts"` — parts wholesale replace: walks entire parts array.

Pointer normalization (`push_pointer()` at lines 128-134): `file-service://` → bare id; `sediment://` → `sed:<id>` prefix (routing tag).

**Test fixtures** at lines 362-393 confirm:
- `asset_pointer: "file-service://file_abc123"` → id `"file_abc123"`
- `asset_pointer: "sediment://sedfoo_bar"` → id `"sed:sedfoo_bar"`

**Polling** (`poll_conversation_for_images()` at `/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/image.rs:244-334`):

URL: `GET https://chatgpt.com/backend-api/conversation/{conversation_id}` (lines 251-254).

Headers: `standard_headers(access_token)` + `oai-device-id` (lines 257-264).

Node walk filters (lines 286-318):
```rust
if role != Some("tool") { continue; }         // author.role
if async_kind != Some("image_gen") { continue; }  // metadata.async_task_type
if content_type != Some("multimodal_text") { continue; }  // content.content_type
// then collect asset_pointer from parts
```

Poll interval: **3 seconds** (`tokio::time::sleep(Duration::from_secs(3))` at line 332).
Deadline: `deadline_secs` parameter (caller passes 180 at line 1047 in channel.rs).
Termination: non-empty pointer list OR deadline elapsed.

**SSE signal for async state**: No explicit `async_status` check. The "Processing image" tool message appears in SSE with empty `multimodal_text` parts before the real pointer arrives:
```
// channel.rs:1035-1057 (comment):
// Image generation on chatgpt.com is ASYNC: the initial SSE
// only emits a "Processing image" tool message and returns
// BEFORE the file-service pointer appears. Poll the
// conversation endpoint until the real pointers show up.
```
`/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/channel.rs:1035-1057`

### pro-cli

**SSE extraction**: Not direct SSE scanning. pro-cli runs in a CDP browser context. The conversation poll happens via `buildImageAssetsFetchExpression()` which injects JavaScript into the ChatGPT page. The JS:
- Fetches `GET /backend-api/conversation/{conversationId}` with `Authorization: Bearer {accessToken}`.
- Walks `mapping` values, checks each part for `image.content_type !== "image_asset_pointer"`.
- Strips `sediment://` prefix: `fileId = assetPointer.replace(/^sediment:\/\//, "")`.
- Reads `record.async_status`.

`/home/eigenmage/dev/scratch/ccproxy/refs/pro-cli/src/transport.ts:2070-2116`

**`async_status` semantics**:
```typescript
// transport.ts:1595
const status = result.asyncStatus === 4 ? "final_without_assets" : "running";
```
`asyncStatus === 4` = terminal without images (generation failed without producing assets). Any non-4 status with assets present → succeeded.

**Poll interval**: `DEFAULT_IMAGE_POLL_MS = 3_000` (3s); `DEFAULT_IMAGE_RATE_LIMIT_POLL_MS = 30_000` (30s on 429):
`/home/eigenmage/dev/scratch/ccproxy/refs/pro-cli/src/transport.ts:22-23`

**Polling loop**: Continues until `result.assets.length > 0` or `asyncStatus === 4` or caller-supplied timeout expired. Line 1595 in transport.ts; loop at lines ~1540-1606.

**Note**: pro-cli only recognizes `sediment://` in the asset pointer strip (`replace(/^sediment:\/\//, "")`); it does NOT handle `file-service://`. This may be intentional (modern chatgpt.com image gen always uses `sediment://`) or a gap.
`/home/eigenmage/dev/scratch/ccproxy/refs/pro-cli/src/transport.ts:2083`

### aurora

**SSE extraction** (`CollectImageResults()` at `/home/eigenmage/dev/scratch/ccproxy/refs/aurora/internal/chatgpt/request.go:1810-1873`):

- Reads SSE line by line; strips `data: ` prefix.
- For each line: calls `collectImageResultsFromValue()` (deep-walk of all JSON values) AND tries typed unmarshal into `ChatGPTResponse`.
- On typed path: if `Content.ContentType == "multimodal_text"`, calls `appendAssetPointerResult()` for each part with non-empty `AssetPointer`.
- `collectImageResultsFromValue()` scans ALL JSON recursively for `asset_pointer`, `assetPointer`, `file_id`/`fileId`/`id` (with `file-` prefix check), `download_url`/`downloadUrl`/`url` keys.

No `author.role` or `async_task_type` filtering on the SSE side — aurora collects any asset pointer it finds.

**Polling** (`PollImageResults()` at `/home/eigenmage/dev/scratch/ccproxy/refs/aurora/internal/chatgpt/request.go:1953-1979`):

URL: `GET {BaseURL}/conversation/{conversationID}` (via `getConversation()` at lines 1889-1911).

Early exit: if `initial` is non-empty, skip polling entirely.
Poll: up to **45 iterations**, **2s sleep** between each (lines 1958-1960) → max 90s.
Termination: non-empty results from `collectImageResultsFromConversation()` OR content policy error detected by `findImageGenerationError()`.

No `async_status` check. No explicit `async_task_type` filter on polling — uses the same generic deep-walk as SSE extraction.

---

## §5 Two-Step Download

### gproxy

**Step 1 — authenticated metadata fetch** (`download_image_b64()` at `/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/image.rs:144-237`):

Routing by id prefix:
```rust
// image.rs:150-166
let (endpoint_id, is_sediment) = if let Some(rest) = ptr.id.strip_prefix("sed:") {
    (rest.to_string(), true)
} else {
    (ptr.id.clone(), false)
};
let download_url_body = if is_sediment {
    format!("https://chatgpt.com/backend-api/conversation/{}/attachment/{}/download",
        ptr.conversation_id, endpoint_id)
} else {
    format!("https://chatgpt.com/backend-api/files/download/{}?conversation_id={}&inline=false",
        endpoint_id, ptr.conversation_id)
};
```

Headers for Step 1: `standard_headers(access_token)` (Bearer) + `oai-device-id` (line 171).
Response: JSON with `download_url` field (line 193-204).

**Step 2 — presigned fetch** (lines 207-236):

```rust
// image.rs:207-218
let step2_resp = client
    .get(download_url)
    // Browser fetch of the estuary URL uses pre-signed sig= in the
    // querystring and no Authorization header; sending Bearer here
    // causes the server to 403 with "File stream access denied".
    .header("accept", "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8")
    .header("referer", "https://chatgpt.com/")
    .send()
    .await
```

Authorization is **explicitly omitted** with an inline explanation. Referer is set. No `oai-device-id`.

Output: `base64::STANDARD.encode(&bytes)` → `String` (line 236).

### pro-cli

**Step 1** (`buildImageDownloadExpression()` at `/home/eigenmage/dev/scratch/ccproxy/refs/pro-cli/src/transport.ts:2119-2171`):

URL: `GET /backend-api/files/download/{encodeURIComponent(id)}` (line 2128).
Headers: `credentials: "include"`, `referrer: "https://chatgpt.com/"`, `Authorization: Bearer {accessToken}` if available (line 2134).
Response field: `metadata.download_url` (line 2141).

**Step 2** (lines 2153-2156):

```typescript
const imageResponse = await fetch(metadata.download_url, {
  credentials: "include",
  referrer: "https://chatgpt.com/",
});
```

Browser `credentials: "include"` carries cookies. No explicit Authorization header on step 2 — browser sends cookies instead (same-origin context). No comment about Authorization causing 403 because in the CDP browser context same-origin credentials handle authentication naturally.

Output: `btoa(binary)` → base64 (lines 2160-2168).

**Only handles** `sediment://` scheme (strips prefix); no `file-service://`/`sed:` routing split.

### aurora

**`GetImageDownloadURL()`** at `/home/eigenmage/dev/scratch/ccproxy/refs/aurora/internal/chatgpt/request.go:1658-1688`:

Sends `Authorization: Bearer` on the metadata fetch.
Response field: `info.DownloadURL` or falls back to `info.URL` if `DownloadURL` is empty (line 1682).

`appendAssetPointerResult()` at lines 1747-1759:
```go
assetParts := strings.Split(assetPointer, "//")
// assetParts[1] is everything after "://"
downloadURL, err := GetImageDownloadURL(client, fileDownloadBaseURL()+assetParts[1]+"/download", secret)
```
This uses the **same download path** regardless of whether the scheme is `file-service://` or `sediment://` — just takes the part after `://` and appends `/download`. `fileDownloadBaseURL()` is `/backend-api/files/` (line 1740).

**`DownloadImageBytes()`** at lines 1690-1715:
```go
header.Set("Authorization", "Bearer "+secret.Token)
```
Authorization is **included** on the presigned download. This is **likely a bug** or works because aurora's production deployment uses the same-domain responses API where the presigned URL may accept bearer tokens. gproxy's explicit comment ("causes 403") is the authoritative signal.

---

## §6 Image Edit — 3-Step Upload

### gproxy

**Parsing** (`parse_edit_body()` at `/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/image_edit.rs:41-47`):

Autodetects multipart vs JSON:
- `is_multipart()` checks `body.starts_with(b"--")` (line 49-51).
- Multipart parser: `parse_multipart()` handles `image`, `image[]`, `image[0]` field names; extracts binary `image_bytes`, `filename`, `mime_type`, `prompt` (lines 53-138).
- JSON parser: `parse_json()` handles `image` (data URL), `images[0].image_url`, `image_url` fields (lines 184-231).

Remote URL rejection:
```rust
// image_edit.rs:226-230
} else if image_ref.starts_with("http://") || image_ref.starts_with("https://") {
    Err("edit body: remote image_url not yet supported".into())
} else {
    Err("edit body: image_url must be data URL or https URL".into())
}
```
`/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/image_edit.rs:226-230`

**Dimension probing** (`probe_png_dimensions()` at `/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/image_edit.rs:391-425`):
- PNG: magic bytes `\x89PNG\r\n\x1a\n`, width at offset 16-20, height at 20-24 (big-endian).
- JPEG: SOF0/SOF2 markers (0xFFC0/0xFFC2), height at i+5, width at i+7.
- GIF: `GIF87a`/`GIF89a`, width/height little-endian at bytes 6-10.
- Returns `(1024, 1024)` as fallback (line 285).

**3-step upload** (`upload_image_to_chatgpt()` at `/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/image_edit.rs:280-384`):

Step 1 body (lines 289-297):
```json
{
  "file_name": "<filename>",
  "file_size": <size_bytes>,
  "use_case": "multimodal",
  "timezone_offset_min": -480,
  "reset_rate_limits": false,
  "store_in_library": true,
  "library_persistence_mode": "opportunistic"
}
```

Step 2 (lines 326-344): `PUT <upload_url>` with headers:
```
Content-Type: <mime_type>
x-ms-blob-type: BlockBlob
```

**Step 3** (lines 346-374): `POST /backend-api/files/process_upload_stream` with body:
```json
{
  "file_id": "<file_id>",
  "use_case": "multimodal",
  "index_for_retrieval": false,
  "file_name": "<filename>",
  "library_persistence_mode": "opportunistic",
  "metadata": {"store_in_library": true}
}
```

**Attachment** (`attach_uploaded_image()` at `/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/channel.rs:796-857`):

```rust
// channel.rs:817-823
let asset = serde_json::json!({
    "content_type": "image_asset_pointer",
    "asset_pointer": format!("sediment://{}", upload.file_id),
    "size_bytes": upload.size_bytes,
    "width": upload.width,
    "height": upload.height,
});
// content.content_type → "multimodal_text"
// content.parts → [asset, prompt_text_string]
```

Attachment metadata (lines 840-856):
```json
{
  "id": "<file_id>",
  "size": <size_bytes>,
  "name": "<filename>",
  "mime_type": "<mime>",
  "width": <w>,
  "height": <h>,
  "source": "library",
  "is_big_paste": false
}
```

### pro-cli

No image edit implementation. pro-cli is a CLI for text and image generation only; edit path is not present.

### aurora

**Parsing**: No multipart parser. Aurora converts images from the OpenAI `image_url` attachment pattern upstream in `buildMessageParts()`. The upload takes raw bytes + mime hint.

**Dimension probing**: Uses Go's `image.DecodeConfig()` stdlib (import `image`, `image/png`, `image/jpeg`, `image/gif`) at `/home/eigenmage/dev/scratch/ccproxy/refs/aurora/internal/chatgpt/files.go:87-91`.

**3-step upload** (`UploadFile()` at `/home/eigenmage/dev/scratch/ccproxy/refs/aurora/internal/chatgpt/files.go:64-134`):

Step 1 body (lines 94-105):
```json
{
  "file_name": "<filename>",
  "file_size": <len(data)>,
  "use_case": "multimodal",
  "mime_type": "<mime>",
  "store_in_library": true,
  "library_persistence_mode": "opportunistic",
  "width": <w>,    // only if > 0
  "height": <h>    // only if > 0
}
```

Note: aurora omits `timezone_offset_min` and `reset_rate_limits` from Step 1; gproxy includes them.

Step 2 (lines 171-191): `PUT <upload_url>` with headers:
```
Content-Type: <mime>
X-Ms-Blob-Type: BlockBlob
X-Ms-Version: 2020-04-08    ← aurora adds this; gproxy does not
Origin: https://chatgpt.com
Referer: https://chatgpt.com/
User-Agent: <aurora default>
Accept: application/json, text/plain, */*
Accept-Language: en-US,en;q=0.8
```

**Step 3 divergence** (line 204):
```go
// files.go:204
response, err := client.Request(http.MethodPost, BaseURL+"/files/"+fileID+"/uploaded", header, nil, strings.NewReader("{}"))
```
aurora uses `POST /backend-api/files/{file_id}/uploaded` with empty body `{}`.

**gproxy uses `POST /backend-api/files/process_upload_stream`** with the explicit JSON body shown above.

**Which is more current?** These are different API endpoints with different semantics:
- `/files/{id}/uploaded` (aurora): simpler "confirm this upload is done" signal; older/simpler API variant.
- `/files/process_upload_stream` (gproxy): more descriptive; carries `use_case`, `index_for_retrieval`, `library_persistence_mode`, and nested `metadata` — richer activation context.

gproxy's endpoint is the **recommended** choice because: (1) it carries necessary activation metadata that controls how the file is indexed; (2) gproxy is the most actively maintained Rust SDK targeting the same audience; (3) aurora's `{}` body provides no context for the server to correctly file-type the upload. However, this is a genuine risk — if aurora's endpoint reflects a more recent API change, gproxy's endpoint may return 404. Flagged as an open question for live probe.

### Uploaded File Pointer Scheme

Both gproxy and aurora agree: uploaded reference images use `sediment://{file_id}` (not `file-service://`):
- gproxy: `format!("sediment://{}", upload.file_id)` at `channel.rs:819`
- aurora conversion layer: `"file-service://" + fileID` for upstream OpenAI image_url attachments, but after `UploadFile()` the returned `UploadedFile.FileID` is stored as bare id; the `sediment://` wrapper is added by the caller.

---

## §7 OpenAI `images.response` Output Shape

### gproxy

`build_openai_images_response()` at `/home/eigenmage/dev/src/gproxy/sdk/gproxy-channel/src/channels/chatgpt/image.rs:337-354`:

```rust
json!({
    "created": <unix_secs>,
    "data": [{"b64_json": b64, "revised_prompt": revised_prompt}]
})
```

`revised_prompt` is an empty string `""` (not `null`) when no revised prompt is available (line 1074 in channel.rs).

### pro-cli

Does not return OpenAI `images.response`. Returns a custom CLI JSON:
```json
{
  "type": "image_generation",
  "conversationId": "<id>",
  "text": "<assistant_text_or_null>",
  "images": [{"fileId": "...", "path": "...", "contentType": "...", "bytes": 0, "width": null, "height": null, "title": null, "fileName": null, "genId": null}]
}
```
`/home/eigenmage/dev/scratch/ccproxy/refs/pro-cli/src/transport.ts:1586-1592`

### aurora

Returns `ImageGenerationResult{URL, B64JSON}` structs; the caller (`initialize/handlers.go`) wraps into the output format. Does not produce a standard OpenAI `images.response` — URL-only results are returned as URLs:

```go
// request.go:1627-1630
type ImageGenerationResult struct {
    URL     string
    B64JSON string
}
```

### Recommendation

OpenAI `images.response` format matching gproxy:
```json
{"created": <unix_secs>, "data": [{"b64_json": "<base64>"}]}
```
`revised_prompt` field: include as `null` (JSON null, not empty string) when not available, to match OpenAI's documented API shape. gproxy uses `""` but `null` is more correct per OpenAI spec.

---

## §8 Recommendation for ccproxy

### Generation Trigger

**Use both**: `system_hints: ["picture_v2"]` AND the pro-cli prompt prefix.

Rationale:
- `system_hints: ["picture_v2"]` is the primary, cleanest signal. gproxy and aurora both use it, and it's already mapped in ccproxy's `_TOOL_HINT_MAP["image_generation"] → "picture_v2"` and `_extract_system_hints()` in `openai_conversations.py`. However: ccproxy's caller is an OpenAI SDK client that typically sends `tools: [{type: "image_generation"}]` — the hint will be extracted automatically from that.
- The prompt prefix is a safety net for cases where the hint alone is insufficient (e.g., if chatgpt.com's `system_hints` verification changes). pro-cli's production use confirms the prefix works standalone.

**Exact prefix string** (cite: `/home/eigenmage/dev/scratch/ccproxy/refs/pro-cli/src/transport.ts:1077`):
```
Use ChatGPT image generation tools to create image(s) from the user's prompt. Do not answer with only a revised prompt. After generation, keep any text brief.
```

**Also inject into `messages[0].metadata.system_hints`** (aurora-style, aurora line 2095) as additional signal.

**Do NOT** prepend the prefix to the system prompt or to non-image requests. It belongs only in the first user message text for image requests.

### Model Routing

Apply aurora's `imageModelSlug()` mapping:
- `gpt-image-*`, `dall-e-*` → `"auto"` (chatgpt.com routes image generation internally).
- Anything else: pass through unchanged (the configured `default_model: gpt-5-5-pro` will be used for most callers who don't specify a model).

### Saved vs Temporary

Force `temporary_chat: False` (i.e., omit or set `history_and_training_disabled: false`) for **all** image requests. This is a hard invariant:
- gproxy: comment at `channel.rs:1035-1037` makes the async nature explicit.
- pro-cli: explicit `temporary && !isImage` guard at `transport.ts:1037`.
- Polling `GET /backend-api/conversation/{id}` returns 404 for temporary conversations.

### Divergence Risks

| Risk | Details | Hedge |
|---|---|---|
| **Upload Step 3 endpoint** | gproxy: `/files/process_upload_stream` with rich body. aurora: `/files/{id}/uploaded` with `{}`. Unknown which is live. | Try gproxy's endpoint first (more complete); fall back to aurora's on 404/405. Log the response body to distinguish cases. |
| **sediment vs file-service download routing** | gproxy splits on id prefix (`sed:` tag for `sediment://`, bare id for `file-service://`). aurora/pro-cli use single path (all through `/files/`). | Implement gproxy's routing. If `sediment://` bare id resolves through `/backend-api/files/download/{id}` (as pro-cli does), also include it as fallback. |
| **Authorization on presigned download** | gproxy explicitly omits Bearer (with 403 warning). aurora includes Bearer (probable bug). pro-cli uses browser cookies. | OMIT Authorization on presigned GET. The gproxy comment is authoritative and the mechanism is clear: presigned URL has `sig=` in query string; providing bearer causes 403. |
| **`async_status` semantics** | pro-cli uses `asyncStatus === 4` as "final without assets". gproxy/aurora don't check async_status. | Read `record.async_status` from the poll response; treat `4` as a terminal-without-assets signal (stop polling, return error). For other non-zero values or no assets: continue polling until deadline. |
| **Pro-cli only handles `sediment://`** | `assetPointer.replace(/^sediment:\/\//, "")` does not handle `file-service://`. | Handle both schemes in extraction AND download routing (per gproxy). Recent chatgpt.com image gen appears to always use `sediment://`, but `file-service://` appeared in older flows and tests. |

### Open Questions for First Credentialed Live Probe

1. Which `system_hints` injection is load-bearing? Does `system_hints: ["picture_v2"]` at top-level alone work, or is `messages[0].metadata.system_hints` (aurora-style) also needed? Or does the prompt prefix alone suffice (pro-cli style)?
2. Does `POST /backend-api/files/process_upload_stream` (gproxy) still return 200, or has the API migrated to `POST /backend-api/files/{id}/uploaded` (aurora)? Check the response status and body.
3. What is the real `async_status` value when images are available? Confirm whether `4` means "done without assets" or "done with assets" (pro-cli's `final_without_assets` semantics).
4. Does the presigned `download_url` have `sig=` in the query string? If so, confirm that omitting `Authorization` returns 200 (gproxy's claim). If the URL is now a same-host `chatgpt.com/backend-api/estuary/content` URL, does it still reject Bearer?
5. What does the `sediment://` file id look like in a real `/f/conversation` response? Does it have a `file_` or `file-` prefix (which would trigger gproxy's `/files/download` path) or is it a different format (which would trigger the `/conversation/{cid}/attachment/{id}/download` path)?
6. Does the `/f/conversation/prepare` call for image requests need `system_hints: ["picture_v2"]` (aurora includes it), or is the prepare body for images identical to text (gproxy/ccproxy's current `build_conversation_prepare_body` does not force it)?
7. Is `thinking_effort: "standard"` (aurora hardcodes this) required for image gen requests? Or is it a no-op?
