# OpenAI Conversations (ChatGPT) — Usage Guide

The `openai_conversations` provider lets you drive a **logged-in chatgpt.com consumer
account** through ccproxy's OpenAI-compatible API. You talk to ccproxy with the standard
OpenAI Chat Completions API (or any SDK that speaks it — this guide uses **Pydantic AI**);
ccproxy does all the ChatGPT-web heavy lifting for you:

```
your script ──OpenAI Chat API──▶ ccproxy ──(bearer + cookies + Sentinel PoW
  (Pydantic AI)   :4001/v1          │         + conduit prewarm + browser TLS)──▶ chatgpt.com
                                    └──────────── chat.completion ◀── SSE / WebSocket ──┘
```

You never deal with the bearer JWT, the Cloudflare `cf_clearance`, the Sentinel proof-of-work,
the conduit handshake, or the TLS fingerprint — ccproxy owns all of it. You send messages and
get back a normal `chat.completion`.

---

## 1. Prerequisites

- A running ccproxy with the `openai_conversations` provider configured (below). This guide
  assumes the proxy is at `http://127.0.0.1:4001` (the dev rig); production is `:4000`.
- A ChatGPT account you are **logged into in a local browser** (Firefox in the examples).
- `gateau` for exporting browser cookies, and `uv` to run the example.

---

## 2. Configure the provider

Add this to your `ccproxy.yaml` under `providers:` (it is also shipped as a packaged default in
`nix/defaults.nix`, so on the Home-Manager module you only need to supply the two files below):

```yaml
providers:
  openai_conversations:
    base_url: https://chatgpt.com
    type: openai_conversations
    fingerprint_profile: chrome136          # browser TLS+HTTP/2 impersonation
    auth:
      type: openai_conversations
      file_path: ~/.config/ccproxy/openai-conversations-credentials.json
      cookie_file: ~/.config/ccproxy/openai-conversations-cookies.txt
```

You supply two files:

### 2a. The credential file (the account bearer)

A flat JSON file. The only field you must provide is `access_token` (the ChatGPT web bearer
JWT); `device_id` is strongly recommended. ccproxy fills in and refreshes the Sentinel
proof-of-work fields (`chat_req_token`, `proof_token`, …) itself.

```json
{
  "access_token": "eyJhbGciOi...<ChatGPT web bearer JWT>",
  "device_id": "f1e2d3c4-....-............"
}
```

Where to get them (logged into chatgpt.com in your browser):
- **`access_token`** — DevTools → Network → any `https://chatgpt.com/backend-api/...` request →
  request header `Authorization: Bearer <…>`. (Equivalently, `GET /api/auth/session` returns it
  as `accessToken`.) These rotate every few hours; ccproxy uses it until it 401s.
- **`device_id`** — the `oai-did` cookie value. Using the browser's own device id avoids
  ChatGPT's "unusual activity" abuse check.

### 2b. The cookie file (Cloudflare clearance + session)

ccproxy needs your browser's `cf_clearance` and session cookies to pass Cloudflare. Export them
with `gateau` into the `cookie_file` path:

```bash
gateau --browser firefox output chatgpt.com auth.openai.com openai.com \
  > ~/.config/ccproxy/openai-conversations-cookies.txt
```

> **`cf_clearance` expires every ~15–30 minutes.** Re-run the export whenever requests start
> returning 403 / Cloudflare challenges. ccproxy reloads the file on every request, so you do
> not need to restart it.

---

## 3. The sentinel key

You authenticate to **ccproxy** (not OpenAI) with the ccproxy *sentinel key*:

```
sk-ant-oat-ccproxy-openai_conversations
```

This is the universal pattern — `sk-ant-oat-ccproxy-<provider-name>`. The sentinel key is what
triggers the `openai_conversations` routing + the whole auth/shaping pipeline. **Never** put a
real OpenAI key here; there is no real OpenAI API key involved at all.

---

## 4. Example: chat + multi-turn sessions with Pydantic AI

Point Pydantic AI's `OpenAIChatModel` at ccproxy's `/v1` endpoint with the sentinel key. That's
the entire integration — everything else is normal Pydantic AI.

Save as `chatgpt_example.py` and run it with `uv run chatgpt_example.py` (the inline
script-metadata block makes uv install Pydantic AI on the fly — no project needed):

```python
# /// script
# requires-python = ">=3.13"
# dependencies = ["pydantic-ai>=1.0"]
# ///
"""Talk to a logged-in ChatGPT account through ccproxy, via Pydantic AI."""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Point Pydantic AI at ccproxy. The api_key is the ccproxy SENTINEL key, not an
# OpenAI key — it selects the openai_conversations provider.
model = OpenAIChatModel(
    "gpt-5-5",  # the ChatGPT model slug; "gpt-5-5-pro" is the deeper-reasoning default
    provider=OpenAIProvider(
        base_url="http://127.0.0.1:4001/v1",
        api_key="sk-ant-oat-ccproxy-openai_conversations",
    ),
)
agent = Agent(model)

# --- Single turn ---------------------------------------------------------------
first = agent.run_sync("My name is Kyle. Reply with exactly: noted")
print("turn 1:", first.output)        # -> noted

# --- Multi-turn session --------------------------------------------------------
# Pass the prior messages back via message_history; the model sees the full
# context and answers in-conversation. Sessions need NOTHING special — this is
# the standard Pydantic AI multi-turn pattern.
second = agent.run_sync(
    "What is my name? One word.",
    message_history=first.all_messages(),
)
print("turn 2:", second.output)       # -> Kyle
```

Expected output (verified live against a real chatgpt-paid account):

```
turn 1: noted
turn 2: Kyle
```

### Streaming

`agent.run_stream(...)` works the same way — ccproxy streams the ChatGPT response back as
OpenAI `chat.completion.chunk`s. `stream=False` (the default `run_sync`) returns one buffered
`chat.completion`; ccproxy collects the upstream stream and renders a single object for you.

---

## 5. How sessions work (and what you don't have to do)

**You do not manage a ChatGPT conversation id.** From the client side, a "session" is just the
ordinary OpenAI pattern: keep the message history and resend it (Pydantic AI's `message_history`,
or appending to your `messages` list). ccproxy forwards the conversation so the model sees the
full context — that is what made `turn 2` answer "Kyle" above.

Under the hood ccproxy keys continuity on a hash of your **first user message**, so keep that
first message stable across a conversation's turns (the standard growing-history pattern does
this automatically).

> **Optional — true server-side threading.** ccproxy can also continue the *ChatGPT-side*
> conversation (sending only the new turn instead of re-flattening the history each time) via the
> `ccproxy.hooks.openai_conversations_thread_inject` inbound hook backed by a TTL conversation
> store. It is **not** enabled by default. Add it to your `hooks.inbound` list (after
> `inject_auth`) if you want ccproxy to thread on the server rather than resend history. With the
> default config, message-history threading (above) is all you need.

---

## 6. Images

Image **generation** and **editing** go through the standard OpenAI Images API on the same proxy,
with the same sentinel key. ccproxy renders the request into a ChatGPT image turn, runs the
upload/poll/download side-trips, and returns an OpenAI `images.response`. (Pydantic AI is a
chat/agent SDK and doesn't generate images, so use the `openai` SDK or plain HTTP here.)

**Generate** — `POST /v1/images/generations`:

```bash
curl -sS http://127.0.0.1:4001/v1/images/generations \
  -H "Authorization: Bearer sk-ant-oat-ccproxy-openai_conversations" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a watercolor fox reading a book", "n": 1}'
```

Or with the `openai` SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:4001/v1",
    api_key="sk-ant-oat-ccproxy-openai_conversations",
)
img = client.images.generate(prompt="a watercolor fox reading a book")
print(img.data[0].url or img.data[0].b64_json[:40])
```

**Edit** — `POST /v1/images/edits` (multipart, with an `image` file + `prompt`) is handled the
same way: ccproxy uploads the source image to ChatGPT first, then renders the edit turn.

Notes on the image hints:
- The prompt is sent **verbatim** — ccproxy does not prepend its own style/system text.
- ChatGPT's image turns are asynchronous; the request can take noticeably longer than text while
  ccproxy polls for the rendered asset, then downloads it and returns the OpenAI shape.

---

## 7. Models

Pass the ChatGPT model slug as the `model`:

| slug | notes |
|---|---|
| `gpt-5-5-pro` | deeper-reasoning default |
| `gpt-5-5` | faster |

ChatGPT exposes no compatible upstream discovery endpoint for wildcard refresh.
Declare account-supported slugs in sibling `config.yaml` to advertise them from
ccproxy's synthetic `/v1/models`, or pass a supported slug directly when using
sentinel routing. An unknown slug is rejected by ChatGPT upstream.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `403` / Cloudflare challenge | `cf_clearance` expired — re-run the `gateau` export (§2b). No restart needed. |
| `401` from ChatGPT | `access_token` expired — refresh it in the credential file (§2a). |
| `"unusual activity"` | `device_id` mismatch — set it to the browser's `oai-did` cookie value. |
| Empty / odd answer | Rare: the turn took the *conduit* (WebSocket) path. ccproxy handles it, but if you ever see an empty answer, run the proxy at `log_level: DEBUG` and check `ccproxy logs` — the intake telemetry (`produced NO text …`) and `ws_handoff: RAWFRAME` lines will say exactly what happened. |
| All requests fail at startup | Check the credential file is present and `access_token` is non-empty (`ccproxy logs` will say `oaic: no credential state loaded`). |

Triage principle: **every** failure through ccproxy is ours to explain first — the logs
(`ccproxy logs -f`) show exactly what was injected, stamped, and forwarded.
