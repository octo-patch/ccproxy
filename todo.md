## 🔴 Silent data drops (highest priority — losing user content)

| Y/N | File:Line | Issue |
| --- | --- | --- |
| | `lightllm/pplx_steps.py:145` | `urls[:3]` — drops search-result URLs beyond first 3 |
| | `lightllm/pplx_threads.py:75` + `inspector/routes/pplx.py:109` | `limit=100` thread fetch — threads with >100 turns silently truncated |
| | `hooks/gemini_envelope.py:358-362` | Multimodal parts dropped on no-token path, warning only |
| | `mcp/buffer.py:10,49-50` | `DEFAULT_MAX_EVENTS=50` + drop-oldest without notification |
| | `hooks/pplx_preflight.py:41` | `_PREFLIGHT_MAX_QUERY=2000` arbitrary query truncation |
| | `oauth/sources.py:271` | `resp.text[:500]` — error body truncated, full detail lost |
| | `pipeline/wire.py:340` | Non-dict tool args silently dropped |
| | `pipeline/wire.py:257,304` | TTL silently coerced to `"5m"` if not `"5m"`/`"1h"` |
| | `utils.py:334,337,346` | Debug-value truncation at width 50/60 |
| | `lightllm/pplx.py:239-241` | `skip_search_enabled`, `is_nav_suggestions_disabled`, `always_search_override` hardcoded (no opt-out for users who want search) |

## 🟡 Useful features gated OFF by default

| Y/N | File:Line | Issue |
| --- | --- | --- |
| | `config.py:226` | `otel.enabled=False` — span data silently dropped unless user knows to flip |
| | `config.py:241` | `GeminiCapacityFallbackConfig.enabled=False` — capacity fallback off |
| | `specs/model_catalog.py` | `refresh=False` default — live catalog refresh requires code change |
| | `lightllm/pplx.py:204` | `save_to_library=True` default — inverse problem (no opt-out for incognito) |

## 🟡 Arbitrary timeouts / hardcoded magic numbers (not configurable)

| Y/N | File:Line | Issue |
| --- | --- | --- |
| | `cli.py:72` | `lines=100` default for `logs` |
| | `cli.py:536` | MCP shutdown 5s hardcoded |
| | `cli.py:689` | TCP probe 0.5s — slow VMs/SSH false-negative |
| | `hooks/gemini_cli.py:82` | Prewarm 10s |
| | `inspector/oauth_addon.py:97` | `INTERNAL` allowlist too broad |
| | `inspector/oauth_addon.py:257` | Exponential backoff base `2` hardcoded |
| | `inspector/oauth_addon.py:291` | 1 retry per fallback model hardcoded |
| | `inspector/namespace.py:152,176,210,488,501,524,541,544` | 7+ hardcoded slirp/curl/warmup/wait timeouts |
| | `inspector/process.py:354,356,390,399` | MCP bind/start/shutdown 5s/15s/2s hardcoded |
| | `lightllm/context_cache.py:27,29` | `timeout=30.0`, `_MAX_PAGINATION_PAGES=100` |
| | `oauth/sources.py:67,119,416` | Credential cmd 5s, refresh 15s, refresh headroom 60s |
| | `specs/model_catalog.py:96` | Fetch timeout 5s |
| | `transport/dispatch.py:35,38` | `MAX_SESSIONS=16`, `IDLE_TIMEOUT=60.0s` |
| | `utils.py:160` | `find_available_port` hardcoded 100 attempts |
| | `inspector/gemini_envelope.py:55-57` | 10s/60s/120s fetch/upload/subscribe |

## 🟢 TTLs without rationale

| Y/N | File:Line | Issue |
| --- | --- | --- |
| | `config.py:288` | `ttl_seconds=1800` (30min L1 cache) |
| | `flows/store.py:170` | `_STORE_TTL=3600` (1h flow store) |
| | `mcp/buffer.py:66` | `DEFAULT_TTL_SECONDS=600` (10min) |

## 🟢 Validator caps, version pins, cosmetic

| Y/N | File:Line | Issue |
| --- | --- | --- |
| | `config.py:250` | `sticky_retry_attempts: le=10` arbitrary upper bound |
| | `inspector/gemini_envelope.py:60,339` | `"2.18"` API version pinned twice |
| | `inspector/addon.py:116,124` | `[:12]` SHA truncation (collision risk at scale) |
| | `inspector/namespace.py:159,191` | `cmdline[:80]` debug truncation |
| | `pipeline/render.py:32` | `MAX_PANEL_WIDTH=60` |
| | `preflight.py:50` | `uuid.uuid4().hex[:13]` arbitrary |
