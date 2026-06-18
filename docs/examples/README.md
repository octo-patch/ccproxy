# Examples

This directory contains runnable examples for routing SDK clients through ccproxy.

## Overview

These examples show how to route SDK requests through ccproxy to leverage provider routing, auth substitution, and observability. They default to the production listener at `http://127.0.0.1:4000`; set `CCPROXY_BASE_URL=http://127.0.0.1:4001` for the dev instance.

To install all SDK dependencies needed by these examples:

```bash
uv add ai-ccproxy[sdk]
```

## Auth Sentinel Key

ccproxy supports a **sentinel API key** that triggers managed auth substitution. This allows SDK clients to use ccproxy's configured provider credentials without carrying a real upstream API key.

**Format:** `sk-ant-oat-ccproxy-{provider}`

**Example for Anthropic:**
```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-ant-oat-ccproxy-anthropic",  # Sentinel key
    base_url="http://localhost:4000",
)
```

When ccproxy sees this sentinel key, it:
1. Looks up the token for the specified provider from the `providers` map
2. Substitutes the sentinel with the real token (and routes the request to the matching `Provider`'s `host`/`path`)
3. If shaping is enabled, stamps the packaged compliance envelope (beta flags, user-agent, etc.) onto the request

**Requirements:**
- A `providers` entry configured in `~/.config/ccproxy/ccproxy.yaml` for the sentinel suffix
- Pipeline hooks enabled: `inject_auth`, `shape`

```bash
# Start ccproxy (foreground — use process-compose or systemd for background)
ccproxy start
```

## Examples

### anthropic_sdk.py

Direct usage of the Anthropic SDK with ccproxy using managed credential forwarding.

**Purpose:**
- Demonstrate non-streaming and streaming requests via Anthropic SDK
- Show proxy-based authentication using a sentinel key
- Simple request/response pattern

**Prerequisites:**
```bash
# anthropic is a core dep of ccproxy — no extra install needed

# Configure OAuth credentials in ~/.config/ccproxy/ccproxy.yaml
# Start ccproxy
ccproxy start
```

**Usage:**
```bash
# Run both simple and streaming examples
uv run python docs/examples/anthropic_sdk.py
```

**Features:**
- Uses sentinel API key (`sk-ant-oat-ccproxy-anthropic`) - proxy substitutes the real auth token
- Base URL: `http://localhost:4000`
- Demonstrates both `messages.create()` and `messages.stream()` patterns
- Shape replay supplies the required native-client compliance envelope

---

### litellm_sdk.py

Using LiteLLM's Python SDK with async completion API.

**Purpose:**
- Show async request patterns with `litellm.acompletion()`
- Demonstrate streaming and non-streaming modes
- Illustrate proxy-based credential handling

**Prerequisites:**
```bash
# litellm is a client-side choice — install it where you're running the example
uv pip install litellm

# Configure credentials in ~/.config/ccproxy/ccproxy.yaml
# Start ccproxy
ccproxy start
```

**Usage:**
```bash
# Run both simple and streaming examples
uv run python docs/examples/litellm_sdk.py
```

**Features:**
- Uses `litellm.acompletion()` interface (works with proxies)
- Async/await patterns for concurrent requests
- Sentinel key with proxy authentication

**Note:** The `litellm.anthropic.messages` interface bypasses proxies, so this example uses the standard completion interface instead.

---

### zai_anthropic_sdk.py

Using Anthropic SDK to access Z.AI GLM models via ccproxy.

**Purpose:**
- Demonstrate Anthropic SDK with GLM-4.7 routed through ccproxy
- Show non-streaming and streaming patterns with messages API
- Proxy handles authentication via `os.environ/ZAI_API_KEY` in ccproxy.yaml

**Prerequisites:**
```bash
# Ensure ZAI_API_KEY is in environment (for ccproxy.yaml)
export ZAI_API_KEY="your-api-key"

# Start ccproxy
ccproxy start
```

**Usage:**
```bash
uv run python docs/examples/zai_anthropic_sdk.py
```

**Features:**
- Routes through ccproxy at `http://127.0.0.1:4000`
- Model: `glm-4.7` (resolved via `providers.zai` in `~/.config/ccproxy/ccproxy.yaml`)
- Sentinel API key — ccproxy substitutes the real auth token via `inject_auth`

---

### gemini_sdk.py

google-genai SDK through ccproxy using the Gemini sentinel key.

**Purpose:**
- Demonstrate non-streaming and streaming content generation via google-genai SDK
- Show proxy-based authentication using the Gemini sentinel key
- The `gemini_cli` outbound hook wraps standard Gemini bodies in the v1internal envelope

**Prerequisites:**
```bash
# Install google-genai (included in ccproxy[sdk])
uv add ai-ccproxy[sdk]

# Ensure Gemini OAuth credentials exist
gemini -p ""

# Start ccproxy
ccproxy start
```

**Usage:**
```bash
uv run python docs/examples/gemini_sdk.py
```

**Features:**
- Uses sentinel key `sk-ant-oat-ccproxy-gemini` — proxy substitutes the real auth token
- Base URL: `http://127.0.0.1:4000/gemini`
- Demonstrates both `generate_content()` and `generate_content_stream()` patterns
- Same-format redirect — no body transformation needed

---

### gemini_sdk_image_via_ccproxy.py

google-genai SDK through ccproxy with an inline image payload.

**Purpose:**
- Demonstrate multi-MB inline image payloads through the Gemini SDK path
- Verify ccproxy preserves `inlineData` payloads while wrapping the request for `cloudcode-pa`

**Usage:**
```bash
uv run python docs/examples/gemini_sdk_image_via_ccproxy.py ~/pictures/screenshot.png
```

---

### deepseek_sdk.py

Anthropic SDK through ccproxy to DeepSeek using the sentinel key.

**Purpose:**
- Demonstrate using the Anthropic SDK with DeepSeek models
- DeepSeek exposes an Anthropic-compatible API — same wire format, same SDK
- ccproxy handles `x-api-key` header injection via `inject_auth` hook

**Prerequisites:**
```bash
# anthropic is a core dep of ccproxy — no extra install needed

# Configure providers.deepseek in ccproxy.yaml
# Start ccproxy
ccproxy start
```

**Usage:**
```bash
uv run python docs/examples/deepseek_sdk.py
```

**Features:**
- Uses sentinel key `sk-ant-oat-ccproxy-deepseek`
- Same SDK as `anthropic_sdk.py` — just a different sentinel key
- Same-format redirect — no body transformation needed
- Demonstrates both `messages.create()` and `messages.stream()` patterns

---

### lightllm_transform.py

Demonstrates ccproxy's lightllm cross-format transformation by using the OpenAI SDK
to call Anthropic and Gemini models through the transform pipeline.

**Purpose:**
- Show how ccproxy rewrites OpenAI-format requests into provider-native format
- Demonstrate the lightllm request adapter plus response intake/render path
- For Gemini: show the Google adapter plus `gemini_cli` envelope-wrap path
- Prove the same OpenAI SDK code can reach any provider ccproxy knows about

**Prerequisites:**
```bash
# Install openai (included in ccproxy[sdk])
uv add ai-ccproxy[sdk]

# Start ccproxy
ccproxy start
```

**Usage:**
```bash
uv run python docs/examples/lightllm_transform.py
```

**Features:**
- Uses OpenAI SDK (`openai.OpenAI`) — single client, multiple backends
- Sentinel keys: `sk-ant-oat-ccproxy-anthropic` and `sk-ant-oat-ccproxy-gemini`
- ccproxy auto-detects OpenAI format from `/v1/chat/completions` path
- Format mismatch triggers transform automatically (no config needed)
- ``SSEPipeline`` handles cross-provider streaming: parses provider-native SSE
  chunks into ccproxy's response IR and re-serializes them as OpenAI SSE
- Demonstrates both non-streaming and streaming for each provider direction

---

### pplx_mcp_probe.py

OpenAI SDK probe for Perplexity Pro server-side MCP connector traffic.

**Purpose:**
- Exercise the Perplexity Pro provider via the OpenAI SDK
- Capture a real flow for inspecting Perplexity's server-side MCP SSE blocks

**Usage:**
```bash
uv run python docs/examples/pplx_mcp_probe.py
```

## Common Setup

All examples require ccproxy to be running:

```bash
# Start ccproxy (foreground — use process-compose or systemd for background)
ccproxy start

# Monitor logs (optional)
ccproxy logs -f

# Check status
ccproxy status
```

## Configuration

Examples expect ccproxy running with:
- **Proxy port**: 4000 (default)
- **OAuth credentials**: Configured in `~/.config/ccproxy/ccproxy.yaml` under `providers`
- **Model routing**: Driven by sentinel-key resolution against `providers`. Use `lightllm.transforms` (`TransformOverride` entries) only for edge cases — bypassing auth for a host or forcing a specific destination for a path/model combo.

### Example ccproxy.yaml Provider Configuration

```yaml
ccproxy:
  providers:
    anthropic:
      auth:
        type: command
        command: "jq -r '.claudeAiOauth.accessToken' ~/.claude/.credentials.json"
      host: api.anthropic.com
      path: /v1/messages
      type: anthropic
```

## Troubleshooting

If examples fail:

1. **Verify ccproxy is running**: `ccproxy status`
2. **Check provider configuration**: Verify the relevant entry under `providers` in `~/.config/ccproxy/ccproxy.yaml`
3. **Review logs**: `ccproxy logs -f` for detailed error messages
4. **Check pipeline hooks**: Ensure `inject_auth` and `shape` are enabled in hooks configuration
5. **Verify port**: Default is 4000, ensure it's not blocked or in use

### Common Errors

- **"This credential is only authorized for use with Claude Code"**: Auth/shaping pipeline hooks are not configured. Verify `inject_auth` and `shape` hooks are enabled, and that a packaged or user shape exists for the provider.
- **"invalid x-api-key"**: Auth headers not being set correctly. Check `inject_auth` hook configuration and logs.
- **Connection refused**: ccproxy not running. Check `ccproxy status`.
- **Transform returning unexpected format**: Verify the sentinel key resolves to a provider with a different wire format. Check `ccproxy flows compare` to see the pre-transform client request and post-transform forwarded request side-by-side.

## Additional Resources

- [ccproxy Documentation](../../README.md)
- [Anthropic SDK Documentation](https://github.com/anthropics/anthropic-sdk-python)
- [OpenAI SDK Documentation](https://github.com/openai/openai-python)
- [google-genai SDK Documentation](https://github.com/googleapis/python-genai)
