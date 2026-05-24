"""Resolve Perplexity thread continuation state and inject into the request.

ccproxy holds no authoritative thread state — Perplexity's server-side
thread library is the source of truth (see ``threads-history.md``). This
hook implements the three-mode resolution chain:

1. **Body metadata** — ``body.metadata.session_id = "<slug-or-uuid>"``
   wins; we ``GET /rest/thread/{value}`` to fetch the latest
   ``backend_uuid`` + ``read_write_token`` + ``context_uuid`` from the
   thread's most recent entry. Upstream errors are returned with
   Perplexity's status/body intact. Divergence between OpenAI history and
   server state is detected here.

2. **Organic L1 cache hit** — when no explicit slug is provided but the
   ``ccproxy.conversation_id`` flow-metadata key matches an entry in the
   :class:`PerplexityThreadStore` populated by a prior turn's
   :class:`PerplexityAddon`. Hot path; no server round-trip.

3. **Pass-through** — nothing matched; the payload builder emits
   ``query_source: "home"`` (fresh thread).

Resolved identifiers go into ``ctx._body["pplx"]`` so they flow through
:class:`~ccproxy.lightllm.adapters.perplexity.PerplexityAdapter.render` →
``_build_pplx_payload(extras=ctx.raw_extras["pplx"])`` chain.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx
from glom import glom

from ccproxy.config import get_config
from ccproxy.lightllm.pplx import (
    PERPLEXITY_BLOCK_USE_CASES,
    PERPLEXITY_BROWSER_UA,
    PERPLEXITY_PROVIDER_NAME,
    PERPLEXITY_SESSION_COOKIE,
    PERPLEXITY_URL_BASE,
    PerplexityError,
)
from ccproxy.lightllm.pplx_threads import get_pplx_thread_store
from ccproxy.pipeline.hook import hook

if TYPE_CHECKING:
    from ccproxy.pipeline.context import Context

logger = logging.getLogger(__name__)

__all__ = ["pplx_thread_inject", "pplx_thread_inject_guard"]


def pplx_thread_inject_guard(ctx: Context) -> bool:
    """Run only when forward_oauth resolved the Perplexity sentinel."""
    assert ctx.flow is not None
    return ctx.flow.metadata.get("ccproxy.oauth_provider") == PERPLEXITY_PROVIDER_NAME


def _thread_fetch_params(*, limit: int, cursor: str | None) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = [
        ("version", "2.18"),
        ("source", "default"),
        ("limit", str(limit)),
        ("from_first", "true"),
        ("with_parent_info", "true"),
        ("with_schematized_response", "true"),
    ]
    if cursor is not None:
        params.append(("cursor", cursor))
    params.extend(("supported_block_use_cases", uc) for uc in PERPLEXITY_BLOCK_USE_CASES)
    return params


def _fetch_thread_page(slug: str, token: str, *, limit: int, cursor: str | None, timeout: float) -> dict[str, Any]:
    """Fetch one ``GET /rest/thread/{slug}`` page."""
    url = f"{PERPLEXITY_URL_BASE}/rest/thread/{slug}"
    headers = {
        "Cookie": f"{PERPLEXITY_SESSION_COOKIE}={token}",
        "User-Agent": PERPLEXITY_BROWSER_UA,
        "Origin": PERPLEXITY_URL_BASE,
        "Referer": f"{PERPLEXITY_URL_BASE}/",
        "Accept": "application/json",
        "x-app-apiclient": "default",
        "x-app-apiversion": "2.18",
        "x-perplexity-request-reason": "perplexity-query-state-provider",
        "x-perplexity-request-endpoint": url,
    }

    resp = httpx.get(
        url,
        params=tuple(_thread_fetch_params(limit=limit, cursor=cursor)),
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    parsed: dict[str, Any] = resp.json()
    return parsed


def _merge_thread_page(base: dict[str, Any], page: dict[str, Any]) -> None:
    entries = base.get("entries")
    page_entries = page.get("entries")
    if isinstance(entries, list) and isinstance(page_entries, list):
        entries.extend(page_entries)


def _fetch_thread(slug: str, token: str) -> dict[str, Any]:
    """``GET /rest/thread/{slug}`` for all available entries.

    Returns the parsed thread dict on success. Upstream non-2xx responses
    raise ``httpx.HTTPStatusError`` with Perplexity's response attached.
    """
    fetch_config = get_config().pplx.thread
    page_size = fetch_config.fetch_page_size
    timeout = fetch_config.fetch_timeout_seconds
    merged: dict[str, Any] | None = None
    cursor: str | None = None
    seen_cursors: set[str] = set()
    pages_fetched = 0

    while True:
        page = _fetch_thread_page(slug, token, limit=page_size, cursor=cursor, timeout=timeout)
        if merged is None:
            merged = page
        else:
            _merge_thread_page(merged, page)

        pages_fetched += 1
        has_next = bool(page.get("has_next") or page.get("has_next_page"))
        if not has_next:
            break
        next_cursor = page.get("end_cursor") or page.get("next_cursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            raise PerplexityError(
                status_code=502,
                message=f"Perplexity thread {slug!r} reported additional entries without a pagination cursor.",
            )
        if next_cursor in seen_cursors:
            raise PerplexityError(
                status_code=502,
                message=f"Perplexity thread {slug!r} repeated pagination cursor {next_cursor!r}.",
            )
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    assert merged is not None
    merged["has_next"] = False
    merged["has_next_page"] = False
    merged["ccproxy_pages_fetched"] = pages_fetched

    return merged


def _extract_latest_identifiers(thread: dict[str, Any]) -> dict[str, str | None] | None:
    """Pull the most recent entry's identifiers from a thread detail response."""
    entries = thread.get("entries")
    if not isinstance(entries, list) or not entries:
        return None
    last = entries[-1]
    if not isinstance(last, dict):
        return None
    backend_uuid = last.get("backend_uuid") or last.get("uuid")
    context_uuid = last.get("context_uuid")
    read_write_token = last.get("read_write_token")
    if not isinstance(backend_uuid, str) or not isinstance(context_uuid, str):
        return None
    return {
        "backend_uuid": backend_uuid,
        "context_uuid": context_uuid,
        "read_write_token": read_write_token if isinstance(read_write_token, str) else None,
    }


def _count_client_user_turns(messages: list[Any]) -> int:
    """Count user-role messages in the incoming OpenAI history (excluding the
    final new user turn). Per the thinkdeep correction, dividing total
    message count by 2 breaks when clients interleave system messages or
    tool turns — counting user roles directly is robust to those shapes.
    """
    if len(messages) < 2:
        return 0
    history = messages[:-1]
    count = 0
    for m in history:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "user":
            count += 1
    return count


@hook(
    reads=["metadata.session_id"],
    writes=["pplx"],
)
def pplx_thread_inject(ctx: Context, _: dict[str, Any]) -> Context:
    """Resolve thread continuation state and inject into ``ctx._body["pplx"]``."""
    assert ctx.flow is not None
    flow = ctx.flow
    body = ctx._body if isinstance(ctx._body, dict) else {}

    slug = glom(body, "metadata.session_id", default=None)
    resolved: dict[str, str | None] | None = None
    resolved_via: str | None = None
    thread_entry_count: int | None = None

    if isinstance(slug, str) and slug:
        config = get_config()
        token = config.resolve_oauth_token(PERPLEXITY_PROVIDER_NAME)
        if not token:
            raise PerplexityError(
                status_code=503,
                message=f"Perplexity thread {slug!r} cannot be resolved because no session token is configured.",
            )
        else:
            try:
                thread = _fetch_thread(slug, token)
            except httpx.HTTPStatusError:
                raise
            except httpx.HTTPError as e:
                raise PerplexityError(
                    status_code=502,
                    message=f"Perplexity thread fetch failed for {slug!r}: {e}",
                ) from e
            ids = _extract_latest_identifiers(thread)
            if ids is not None:
                resolved = ids
                resolved_via = "metadata"
                entries = thread.get("entries")
                if isinstance(entries, list):
                    thread_entry_count = len(entries)
            else:
                raise PerplexityError(
                    status_code=502,
                    message=f"Perplexity thread {slug!r} returned no usable continuation identifiers.",
                )

    if resolved is None:
        conv_id = flow.metadata.get("ccproxy.conversation_id")
        if isinstance(conv_id, str) and conv_id:
            store = get_pplx_thread_store()
            cached = store.get(conv_id)
            if cached is not None:
                resolved = {
                    "backend_uuid": cached.backend_uuid,
                    "context_uuid": cached.context_uuid,
                    "read_write_token": cached.read_write_token,
                }
                resolved_via = "l1_cache"

    if resolved is None:
        return ctx

    if resolved_via == "metadata" and thread_entry_count is not None and isinstance(body.get("messages"), list):
        client_user_turns = _count_client_user_turns(body["messages"])
        if client_user_turns != thread_entry_count:
            mode = get_config().pplx.thread.consistency_mode
            divergence = f"turn_count_mismatch: client={client_user_turns} server={thread_entry_count}"
            if mode == "strict":
                raise PerplexityError(
                    status_code=409,
                    message=(
                        f"Perplexity thread {slug!r} diverged from incoming history "
                        f"({divergence}). Re-import the thread or remove "
                        f"metadata.session_id."
                    ),
                )
            if mode == "warn":
                flow.metadata["ccproxy.pplx.divergence"] = divergence
                logger.warning("pplx_thread_inject: divergence (warn): %s", divergence)

    pplx_extras = body.get("pplx")
    if not isinstance(pplx_extras, dict):
        pplx_extras = {}
    pplx_extras["last_backend_uuid"] = resolved["backend_uuid"]
    pplx_extras["frontend_context_uuid"] = resolved["context_uuid"]
    if resolved.get("read_write_token"):
        pplx_extras["read_write_token"] = resolved["read_write_token"]
    body["pplx"] = pplx_extras
    ctx._body = body

    flow.metadata["ccproxy.pplx.resolved_via"] = resolved_via
    logger.info(
        "pplx_thread_inject: resolved_via=%s backend_uuid=%s%s",
        resolved_via,
        resolved["backend_uuid"][:8] if resolved["backend_uuid"] else "",
        " (slug=" + (slug or "") + ")" if resolved_via == "metadata" else "",
    )

    return ctx
