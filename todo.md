## 🔴 Silent data drops (highest priority — losing user content)

| Y/N | File:Line | Issue |
| --- | --- | --- |
| | `lightllm/pplx_steps.py:164` | `urls[:3]` — drops search-result URLs beyond first 3 |
| | `lightllm/pplx_threads.py:75` + `inspector/routes/pplx.py:109` | `limit=100` thread fetch — threads with >100 turns silently truncated |
| | `mcp/buffer.py:10,49-50` | `DEFAULT_MAX_EVENTS=50` + drop-oldest without notification |
| | `hooks/pplx_preflight.py:41` | `_PREFLIGHT_MAX_QUERY=2000` arbitrary query truncation |
| | `oauth/sources.py:274` | `resp.text[:500]` — error body truncated, full detail lost |
| | `utils.py:333,337,346` | Debug-value truncation at width 50/60 |
| | `lightllm/pplx.py:242-244` | `skip_search_enabled`, `is_nav_suggestions_disabled`, `always_search_override` hardcoded (no opt-out for users who want search) |

## 🟡 Useful features gated OFF by default

| Y/N | File:Line | Issue |
| --- | --- | --- |
| | `config.py:226` | `otel.enabled=False` — span data silently dropped unless user knows to flip |
| | `config.py:241` | `GeminiCapacityFallbackConfig.enabled=False` — capacity fallback off |
| | `specs/model_catalog.py:137` | `refresh=False` default — live catalog refresh requires code change |
| | `lightllm/pplx.py:208` | `save_to_library=True` default — inverse problem (no opt-out for incognito) |

## 🟡 Arbitrary timeouts / hardcoded magic numbers (not configurable)

| Y/N | File:Line | Issue |
| --- | --- | --- |
| | `cli.py:71` | `lines=100` default for `logs` |
| | `cli.py:538` | MCP shutdown 5s hardcoded |
| | `cli.py:692` | TCP probe 0.5s — slow VMs/SSH false-negative |
| | `hooks/gemini_cli.py:82` | Prewarm 10s |
| | `inspector/namespace.py:152,176,210,488,501,524,541,544` | 7+ hardcoded slirp/curl/warmup/wait timeouts |
| | `inspector/process.py:356,362,390,398` | MCP bind/start/shutdown 5s/15s/2s hardcoded |
| | `oauth/sources.py:67,119,416` | Credential cmd 5s, refresh 15s, refresh headroom 60s |
| | `specs/model_catalog.py:96` | Fetch timeout 5s |
| | `transport/dispatch.py:35,38` | `MAX_SESSIONS=16`, `IDLE_TIMEOUT=60.0s` |
| | `utils.py:160` | `find_available_port` hardcoded 100 attempts |

## 🟢 TTLs without rationale

| Y/N | File:Line | Issue |
| --- | --- | --- |
| | `config.py:288` | `ttl_seconds=1800` (30min L1 cache) |
| | `flows/store.py:198` | `_STORE_TTL=3600` (1h flow store) |
| | `mcp/buffer.py:66` | `DEFAULT_TTL_SECONDS=600` (10min) |

## 🟢 Validator caps, version pins, cosmetic

| Y/N | File:Line | Issue |
| --- | --- | --- |
| | `config.py:250` | `sticky_retry_attempts: le=10` arbitrary upper bound |
| | `inspector/addon.py:117,124` | `[:12]` SHA truncation (collision risk at scale) |
| | `inspector/namespace.py:159,191` | `cmdline[:80]` debug truncation |
| | `pipeline/render.py:32` | `MAX_PANEL_WIDTH=60` |
| | `preflight.py:50` | `uuid.uuid4().hex[:13]` arbitrary |
