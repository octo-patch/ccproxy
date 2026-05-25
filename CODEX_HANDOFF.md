# Codex Handoff: bundled shapes via the existing apply-time machinery

## The lesson from this session

`scripts/package-mflows.py` was built as a "scrubber" that reimplemented
the existing shaping system at packaging time: it deleted body fields,
emptied arrays, stripped headers, sanitized connection state. Every
single one of those operations is **already configurable at apply
time** via the shaping framework. The packager was redundant and
wrong-shaped. It has been reverted.

The user's repeated direction was: **save it all at capture time,
selectively apply at runtime.** The bundled `.mflow` should be a
faithful capture; the existing apply-time machinery handles the rest.

That machinery, all in `nix/defaults.nix → shaping.providers.<name>`:

- **`content_fields`** — body keys overridden by incoming request at apply
  time. Anything listed here gets the capturer's value erased and the
  live request's value injected. This is the canonical answer for any
  body field that's per-user (`metadata.user_id`, `project`,
  `user_prompt_id`, `messages`, `tools`, `system`, `diagnostics`, etc).
- **`merge_strategies`** — per-field merge override (`replace`,
  `prepend_shape`, `append_shape`, `drop`, with `:N` slice). E.g.
  `merge_strategies.system = "prepend_shape:2"` keeps the first 2 shape
  blocks and prepends them onto incoming. Anything past index 2 is dead
  weight at apply time.
- **`shape_hooks`** — DAG-ordered inner hooks that mutate the shape
  working copy before stamping. Already used: `regenerate_user_prompt_id`,
  `regenerate_session_id` (body-level `metadata.user_id.session_id`),
  `regenerate_billing_header`, `caching.strip`, `caching.insert`,
  `inject_gemini_content`, `strip_unset_content`. Add more here for
  any per-request derivation that can't be expressed as a field
  injection.
- **`strip_headers`** — headers removed from the shape working copy at
  apply time. Auth tokens, transport headers.
- **`preserve_headers`** — headers from the live target that survive
  the shape stamping (auth headers set by `forward_oauth`, host set by
  the transform router).

So a bundled `.mflow` that's a faithful capture from Claude CLI / Gemini
CLI is fine to ship **provided the shaping config covers every
identifying field**. Where it doesn't, the answer is to extend the
shaping config — not to write a custom scrubber script that operates
out-of-band of the shaping system.

## What's been kept from this session (apply-time / capture-time fixes)

These are real fixes, aligned with the "selectively apply" principle.
Leave them in:

- `src/ccproxy/inspector/shape_capturer.py` — `_STRIP_SHAPE_HEADERS`
  now includes `x-ccproxy-flow-id` (the ccproxy correlation header has
  no meaning outside a running process; strip at capture time so it
  doesn't even land in personal shapes).
- `src/ccproxy/inspector/egress_sanitizer_addon.py` — new mitmproxy
  addon registered last in `_build_addons`. Explicit deny-list:
  `x-ccproxy-flow-id`, `x-ccproxy-hooks`, `x-ccproxy-oauth-injected`.
  Sidecar transport headers (`x-ccproxy-target-url`,
  `x-ccproxy-impersonate`) are intentionally kept — they're consumed
  by the sidecar on the loopback hop and stripped there.
- `nix/defaults.nix` + regenerated `src/ccproxy/templates/ccproxy.yaml`
  — `diagnostics` added to anthropic `content_fields` so the live
  request's `previous_message_id` wins at apply time.
- `src/ccproxy/inspector/fingerprint.py` +
  `src/ccproxy/transport/dispatch.py` — `CurlOpt.HTTP_CONTENT_DECODING = 0`
  in `transport_kwargs` and the browser-impersonate branch. Disables
  curl-cffi's auto-decompression so the sidecar streams compressed
  bytes verbatim and mitmproxy's existing decoder handles
  `Content-Encoding` for both the response to the client and the
  inspector capture (eliminated the "decode response gzip" errors in
  the daemon log).
- `src/ccproxy/config.py` — extracted `_default_hooks()` helper to
  resolve ty diagnostic on `Field(default_factory=lambda: ...)`
  invariant mismatch.

## What's been reverted

- `scripts/package-mflows.py` — deleted. Bundled scrubbing as a
  pre-packaging step is the wrong design.
- `.pre-commit-config.yaml` — `package-mflows-verify` hook removed.
- `docs/fingerprint.md` — "Bundled vs personal shapes" section
  removed (it described the deleted script's policy).
- `src/ccproxy/templates/shapes/anthropic.mflow` — deleted. Filter-repo
  corrupted the original tnetstring encoding. Needs re-capture.
- `src/ccproxy/templates/shapes/gemini.mflow` — already deleted
  earlier in the session for the same reason.

## What Codex needs to do

### 1. Re-capture both bundled shapes

`anthropic.mflow` and `gemini.mflow` both need to be re-captured from a
real CLI session and committed to `src/ccproxy/templates/shapes/`.
Capture via `ccproxy run --inspect -- <cli> -p "<prompt>"`, identify
the matching flow, `ccproxy flows shape <provider> --mflow`. Copy the
resulting `~/.config/ccproxy/<config-dir>/shapes/<provider>.mflow` into
the source tree.

**Before committing**, audit the shape for residual PII using the
existing apply-time strip lists as the spec — anything that *would*
leak after going through `content_fields` + `strip_headers` + the
shape hooks at apply time. The captured user_agent / device_id will
appear in the bundled but apply-time machinery handles them; the
specific identifiers below need to be either added to that machinery
or absent from the capture itself.

### 2. Extend shaping config to cover per-user fields

The following per-user body / header fields should be added to the
appropriate `shaping.providers.<name>` config so apply-time wins
without needing pre-packaging scrub:

**Anthropic** (`nix/defaults.nix:shaping.providers.anthropic`):

- `content_fields`: add `metadata` (top-level). The current entry
  doesn't override `metadata.user_id`, so the bundled's value (which
  has the capturer's `account_uuid` and `device_id`) replays on every
  request. Adding `metadata` to `content_fields` means the live
  request's metadata wins. If the live request doesn't carry
  `metadata` (e.g. raw curl), the apply will inject the bundled
  capture — for that gap there's already a `regenerate_session_id`
  shape hook (rolls just the session_id portion), but `account_uuid`
  and `device_id` will still leak from the bundled. Options:
  - Extend `regenerate_session_id` to also null the other two fields
    when the incoming request has no `metadata`.
  - Add a new shape inner-DAG hook
    `scrub_persistent_user_id_when_incoming_absent` that wipes the
    triple unless the live request provides its own.
  - Per-provider configuration on this is the user's preferred direction.

**Gemini** (`nix/defaults.nix:shaping.providers.gemini`):

- `content_fields`: already lists `model` and `project`, which covers
  the cloud project ID. But `user_prompt_id`, `request.session_id`,
  `request.contents`, `request.systemInstruction`, `request.tools` —
  these aren't expressible as top-level `content_fields` entries
  because they're nested under `request`. The existing
  `inject_gemini_content` and `strip_unset_content` hooks handle
  `contents` / `systemInstruction` / `tools` already. Need a similar
  approach for `request.session_id` and top-level `user_prompt_id` —
  either extend an existing hook or add new ones.

**For header-level UUIDs** (`X-Claude-Code-Session-Id`,
`x-client-request-id`): these come from the captured shape's headers
and currently replay verbatim. The user previously flagged this as
the "header session_id + request_id regen" task (originally task #2
in earlier plans, parked). A shape inner-DAG hook that rolls those
header values per request is the right fit — analogous to how
`regenerate_session_id` rolls the body-level session_id.

### 3. Provider-SDK e2e tests against the dev daemon

The user explicitly asked for tests that exercise each provider's
bundled default shape end-to-end against the live `process-compose`
dev daemon. Acceptance: for each provider declared in
`nix/defaults.nix`, build a minimal SDK request, send it through
the dev daemon at `http://127.0.0.1:4001`, assert 200 + parseable
response.

Suggested file: `tests/e2e/test_bundled_shapes_e2e.py`, marked
`pytest.mark.e2e` (excluded from default suite per pyproject's
`addopts`).

| Provider | SDK | Sentinel | Required env |
|---|---|---|---|
| `anthropic` | `anthropic` Python SDK | `sk-ant-oat-ccproxy-anthropic` | `CLAUDE_CODE_OAUTH_TOKEN` |
| `gemini` | `google-genai` SDK | `sk-ant-oat-ccproxy-gemini` | `~/.gemini/oauth_creds.json` |
| `deepseek` | `anthropic` SDK (type: anthropic) | `sk-ant-oat-ccproxy-deepseek` | `DEEPSEEK_API_KEY` |
| `codex` | `openai` SDK | `sk-ant-oat-ccproxy-codex` | `~/.codex/auth.json` |
| `perplexity_pro` | direct HTTP | `sk-ant-oat-ccproxy-perplexity_pro` | `~/.opnix/secrets/perplexity-pro-api-key` |

Skip a test if the required credential isn't available (don't fail).
Skip the whole module if the dev daemon isn't reachable.

Each test:

```python
@pytest.mark.e2e
def test_anthropic_default_shape_round_trip(dev_daemon_url):
    client = anthropic.Anthropic(
        api_key="sk-ant-oat-ccproxy-anthropic",
        base_url=dev_daemon_url,
    )
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=24,
        messages=[{"role": "user", "content": "Reply with: e2e ok"}],
    )
    assert resp.content[0].text.strip() == "e2e ok"
```

The tests' real job is regression-catching: when someone updates a
bundled `.mflow` (because a CLI shipped a new version) or changes the
shaping config, these confirm the apply path still gets a real 200
from the real upstream for every provider.

### 4. (Optional, separately scoped) `ccproxy providers init/list/save/load`

User-mentioned UX for the "capture all default shapes" workflow:

- `ccproxy providers list` — configured providers + whether a personal
  shape exists.
- `ccproxy providers init [--provider=<name>]` — run the canonical
  capture command(s) per provider; save personal shape.
- `ccproxy providers save <name>` — explicit "capture from a running
  flow you specify" variant.
- `ccproxy providers load <name>` — bundled re-import.

Its own design pass. Not blocking on the other work.

## Constraints / things to not redo

- **Don't reinvent the shaping system.** Capture-time strips (the
  `_STRIP_SHAPE_HEADERS` set in `inspector/shape_capturer.py`) are
  fine for unambiguous transport / auth headers. Anything beyond
  that — body fields, identity headers, per-request derivations —
  belongs in `nix/defaults.nix:shaping.providers.<name>` so the
  existing apply-time machinery handles it.
- **No hand-curated literal-string PII blocklists in tests.** The
  previous `BODY_LEAK_MARKERS` list in
  `tests/test_shaping_defaults.py` doxxed the maintainer in their
  own public test file. That test has been deleted. Any future
  safety check must be structural, not literal-string-based.
- **Don't re-introduce `metadata.user_id` zero-UUID placeholders, "seed"
  message placeholders, or hardcoded `max_tokens` defaults** into a
  packaging script. The user explicitly rejected each of those.
- **The bundled `.mflow` is a faithful capture, not a synthesized
  artifact.** Sanitization belongs in apply-time configuration.

## Open follow-ups carried from earlier

- `tests/test_lightllm_graph_openai_load.py` still contains the
  string `kyle` — flagged but not touched in this session.
- Public forks of `starbaser/ccproxy` may retain pre-rewrite state
  with the original PII. GitHub PII removal request is the only way
  to address those; not a code task.
- `transport/sidecar.py:_HOP_BY_HOP` set is misnamed (includes
  `host` / `content-length` which aren't strictly RFC 7230 hop-by-hop).
  Cosmetic cleanup.

## Verification ledger at handoff

`just lint` + `just typecheck` clean; `uv run pytest --no-cov`
passes (will land at 1783 tests with `test_shaping_defaults.py`
deleted). `origin/dev` and `origin/main` both PII-scrubbed via
filter-repo + force-push.
