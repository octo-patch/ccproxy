"""Response-side token-usage capture and wire-shape projection.

The lightllm response FSMs drive pydantic-ai's ``ModelResponsePartsManager``
directly, which — by design — carries no token usage: in pydantic-ai proper,
usage rides ``StreamedResponse._usage`` as side-channel state, never as a part
or stream event. These helpers give ccproxy's intake FSMs the accumulator it
otherwise skips.

Two directions:

* ``usage_from_*`` read the raw provider usage object (dict or SDK model) off an
  already-parsed wire event into the canonical :class:`pydantic_ai.usage.RequestUsage`.
  Canonical ``input_tokens`` EXCLUDES cache tokens; ``cache_read_tokens`` and
  ``cache_write_tokens`` are separate additive fields (matching pydantic-ai /
  genai-prices semantics).
* ``to_*`` project a captured ``RequestUsage`` back into each listener's native
  usage shape.

The field translation is hand-written rather than delegated to
``RequestUsage.extract`` (genai-prices). ``extract`` swallows every exception
and returns an empty ``RequestUsage`` on any provider-resolution miss — which
would silently re-zero usage, the exact failure this module exists to remove —
and it runs a pricing-snapshot lookup per event on the streaming hot path. The
mapping for our known providers is small and deterministic.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.usage import RequestUsage


def _int(value: Any) -> int:
    """Coerce a possibly-``None`` wire token count to ``int``."""
    return int(value) if isinstance(value, (int, float)) else 0


def _get(raw: Any, key: str) -> int:
    """Read one integer field off a dict or an SDK model, defaulting to 0."""
    if isinstance(raw, dict):
        return _int(raw.get(key))
    return _int(getattr(raw, key, 0))


def _get_any(raw: Any, *keys: str) -> int:
    """First non-zero of several key spellings (wire camelCase vs SDK snake_case)."""
    for key in keys:
        value = _get(raw, key)
        if value:
            return value
    return 0


def _nested(raw: Any, outer: str, inner: str) -> int:
    """Read ``raw[outer][inner]`` off a dict or SDK model, defaulting to 0."""
    container = raw.get(outer) if isinstance(raw, dict) else getattr(raw, outer, None)
    if container is None:
        return 0
    return _get(container, inner)


def usage_is_empty(usage: RequestUsage | None) -> bool:
    """True when there is no token information worth emitting.

    An absent-or-zero usage block is deliberately omitted rather than emitted as
    zeros: a present-but-zero usage reads to consumers as "this call was free"
    and suppresses their tokenizer fallback, whereas an absent block triggers it.
    """
    if usage is None:
        return True
    return not (usage.input_tokens or usage.output_tokens or usage.cache_read_tokens or usage.cache_write_tokens)


# ── Capture: raw provider usage → canonical RequestUsage ────────────────────


def usage_from_anthropic(raw: Any, *, existing: RequestUsage | None = None) -> RequestUsage:
    """Map an Anthropic ``usage`` object/dict to canonical ``RequestUsage``.

    Anthropic streams usage across two events — ``message_start`` (input + cache)
    and ``message_delta`` (cumulative output) — so the numbers are cumulative and
    each event should REPLACE, not increment. Because ``message_delta`` omits the
    input/cache fields, we merge field-by-field: a present (non-zero) wire value
    wins; an absent one keeps whatever a prior event already established.
    Anthropic's wire ``input_tokens`` already excludes cache, so it maps across.
    """
    base = existing or RequestUsage()
    return RequestUsage(
        input_tokens=_get(raw, "input_tokens") or base.input_tokens,
        output_tokens=_get(raw, "output_tokens") or base.output_tokens,
        cache_write_tokens=_get(raw, "cache_creation_input_tokens") or base.cache_write_tokens,
        cache_read_tokens=_get(raw, "cache_read_input_tokens") or base.cache_read_tokens,
    )


def usage_from_openai_chat(raw: Any) -> RequestUsage:
    """Map an OpenAI Chat ``usage`` object/dict to canonical ``RequestUsage``.

    OpenAI's ``prompt_tokens`` INCLUDES cached tokens, so the canonical
    (cache-excluding) ``input_tokens`` is ``prompt_tokens - cached_tokens``.
    OpenAI streams a single terminal usage chunk, so this is authoritative.
    """
    cached = _nested(raw, "prompt_tokens_details", "cached_tokens")
    prompt = _get(raw, "prompt_tokens")
    return RequestUsage(
        input_tokens=max(prompt - cached, 0),
        output_tokens=_get(raw, "completion_tokens"),
        cache_read_tokens=cached,
    )


def usage_from_openai_responses(raw: Any) -> RequestUsage:
    """Map an OpenAI Responses ``usage`` object/dict to canonical ``RequestUsage``.

    Responses ``input_tokens`` includes cached; ``input_tokens_details.cached_tokens``
    breaks it out. Canonical ``input_tokens`` excludes cache.
    """
    cached = _nested(raw, "input_tokens_details", "cached_tokens")
    input_total = _get(raw, "input_tokens")
    return RequestUsage(
        input_tokens=max(input_total - cached, 0),
        output_tokens=_get(raw, "output_tokens"),
        cache_read_tokens=cached,
    )


def usage_from_google(raw: Any) -> RequestUsage:
    """Map a Google ``usageMetadata`` object/dict to canonical ``RequestUsage``.

    ``promptTokenCount`` includes cached content; ``cachedContentTokenCount``
    breaks it out. ``thoughtsTokenCount`` (thinking) is folded into output.
    Accepts both the wire camelCase and the google-genai SDK snake_case spellings.
    """
    cached = _get_any(raw, "cachedContentTokenCount", "cached_content_token_count")
    prompt = _get_any(raw, "promptTokenCount", "prompt_token_count")
    output = _get_any(raw, "candidatesTokenCount", "candidates_token_count") + _get_any(
        raw, "thoughtsTokenCount", "thoughts_token_count"
    )
    return RequestUsage(
        input_tokens=max(prompt - cached, 0),
        output_tokens=output,
        cache_read_tokens=cached,
    )


# ── Projection: canonical RequestUsage → listener wire shape ────────────────


def to_openai_chat(usage: RequestUsage) -> dict[str, Any]:
    """Project to an OpenAI Chat Completions ``usage`` block.

    OpenAI ``prompt_tokens`` includes cache (read + write); ``cached_tokens`` in
    ``prompt_tokens_details`` carries the cache-read breakdown.
    """
    prompt = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
    completion = usage.output_tokens
    block: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if usage.cache_read_tokens:
        block["prompt_tokens_details"] = {"cached_tokens": usage.cache_read_tokens}
    return block


def to_openai_responses(usage: RequestUsage) -> dict[str, Any]:
    """Project to an OpenAI Responses ``usage`` block (``input_tokens`` includes cache)."""
    input_total = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
    output = usage.output_tokens
    block: dict[str, Any] = {
        "input_tokens": input_total,
        "output_tokens": output,
        "total_tokens": input_total + output,
    }
    if usage.cache_read_tokens:
        block["input_tokens_details"] = {"cached_tokens": usage.cache_read_tokens}
    return block


def to_anthropic(usage: RequestUsage) -> dict[str, Any]:
    """Project to an Anthropic ``usage`` block (cache classes are separate fields)."""
    block: dict[str, Any] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
    if usage.cache_read_tokens:
        block["cache_read_input_tokens"] = usage.cache_read_tokens
    if usage.cache_write_tokens:
        block["cache_creation_input_tokens"] = usage.cache_write_tokens
    return block


def to_anthropic_message_start(usage: RequestUsage) -> dict[str, Any]:
    """Anthropic ``message_start.usage``: input + cache, output not yet counted."""
    block = to_anthropic(usage)
    block["output_tokens"] = 0
    return block
