"""Perplexity Pro WebUI subscription as a LiteLLM ``BaseConfig``.

Routes OpenAI ``/v1/chat/completions`` requests to Perplexity's internal
``POST https://www.perplexity.ai/rest/sse/perplexity_ask`` endpoint using
a ``__Secure-next-auth.session-token`` cookie for auth (Pro subscription).

The Perplexity wire format is not chat-completions-shaped: a single
``query_str`` plus a ``params`` block carrying model preference, search
focus, sources, etc. Streaming responses arrive as schematized SSE events
(``use_schematized_api: true``, ``send_back_text_in_streaming_api: false``)
delivering cumulative answer text via ``diff_block.patches[]`` patches on
``/markdown_block`` and reasoning text via ``plan_block.goals[].description``.
``PerplexityProIterator`` prefix-diffs both streams independently and emits
OpenAI-format delta chunks (``content`` + ``reasoning_content``).

Thread continuation: the inbound ``pplx_thread_inject`` hook resolves
``body.metadata.session_id`` (or an L1 cache hit) to identifiers
and writes them into ``optional_params["pplx"]`` as ``last_backend_uuid``
+ ``read_write_token`` + ``frontend_context_uuid``. The payload builder
honors these to emit ``query_source: "followup"``. The final SSE event's
``thread_url_slug`` is echoed back to the client on the terminal chunk so
cooperating clients can capture it for the next turn's metadata field.

Model catalog vendored in ``ccproxy/specs/perplexity_models.json``.

Credits to https://henrique-coder.github.io/perplexity-webui-scraper for
the original wire-format reconnaissance.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Any

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.utils import ModelResponse, ModelResponseStream

from ccproxy.lightllm.pplx_steps import _KNOWN_INTENDED_USAGES, render_step

if TYPE_CHECKING:
    import httpx
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import AllMessageValues

logger = logging.getLogger(__name__)


PERPLEXITY_URL_BASE = "https://www.perplexity.ai"
PERPLEXITY_URL = f"{PERPLEXITY_URL_BASE}/rest/sse/perplexity_ask"
PERPLEXITY_PREFLIGHT_URL = f"{PERPLEXITY_URL_BASE}/search/new"
PERPLEXITY_API_VERSION = "2.18"
PERPLEXITY_BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
PERPLEXITY_SESSION_COOKIE = "__Secure-next-auth.session-token"
PERPLEXITY_PROVIDER_NAME = "perplexity_pro"

PERPLEXITY_FEATURES: list[str] = ["browser_agent_permission_banner_v1.1"]

PERPLEXITY_BLOCK_USE_CASES: list[str] = [
    "answer_modes",
    "media_items",
    "diff_blocks",
    "preserve_latex",
    "inline_claims",
]


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def load_pplx_models() -> dict[str, dict[str, str]]:
    """Load the vendored Perplexity model catalog keyed by public model id."""
    raw: bytes = files("ccproxy.specs").joinpath("perplexity_models.json").read_bytes()  # type: ignore[arg-type]
    data: list[dict[str, str]] = json.loads(raw)
    return {m["id"]: {"identifier": m["identifier"], "mode": m["mode"]} for m in data}


PERPLEXITY_MODELS: dict[str, dict[str, str]] = load_pplx_models()


_SOURCE_MAP: dict[str, str] = {
    "web": "web",
    "academic": "scholar",
    "social": "social",
    "finance": "edgar",
    "all": "web",
}

_SEARCH_MAP: dict[str, str] = {
    "web": "internet",
    "writing": "writing",
}

_TIME_MAP: dict[str, str] = {
    "all": "",
    "day": "DAY",
    "week": "WEEK",
    "month": "MONTH",
    "year": "YEAR",
}


def _flatten_messages(messages: list[Any]) -> str:
    """Flatten OpenAI-style chat messages into a single Perplexity ``query_str``."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )

        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str):
                        text_parts.append(t)
            text = "\n".join(text_parts)

        if not text:
            continue
        if role == "system":
            parts.insert(0, f"[System]: {text}")
        else:
            parts.append(text)

    return "\n\n".join(parts)


def _flatten_last_user_turn(messages: list[Any]) -> str:
    """Extract text from the last ``role == "user"`` message.

    Followup requests identify the thread via ``last_backend_uuid``; the
    Perplexity server already holds the full conversation, so ``dsl_query``
    must carry only the new user turn — not the flattened history.
    """
    for msg in reversed(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role != "user":
            continue
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str):
                        text_parts.append(t)
            return "\n".join(text_parts)
        return ""
    return ""


def _build_pplx_payload(
    query: str,
    model_id: str,
    extras: dict[str, Any],
) -> dict[str, Any]:
    """Build the Perplexity SSE ask payload per core-query.md:147-241.

    ``extras`` is sourced from ``optional_params["pplx"]`` — the merger of
    OpenAI ``extra_body.pplx.*`` from the client and identifiers injected
    by the ``pplx_thread_inject`` hook (``last_backend_uuid``,
    ``read_write_token``, ``frontend_context_uuid``).
    """
    meta = PERPLEXITY_MODELS.get(model_id)
    if meta is None:
        available = ", ".join(sorted(PERPLEXITY_MODELS))
        raise ValueError(
            f"Unknown Perplexity model {model_id!r}. Available: {available}"
        )

    raw_sources = extras.get("source_focus", "web")
    if not isinstance(raw_sources, list):
        raw_sources = [raw_sources]
    sources = [_SOURCE_MAP.get(s, "web") for s in raw_sources]

    coordinates = extras.get("coordinates")
    client_coords: dict[str, Any] | None = None
    if isinstance(coordinates, dict):
        client_coords = {
            "location_lat": coordinates.get("latitude"),
            "location_lng": coordinates.get("longitude"),
            "name": "",
        }

    save_to_library = bool(extras.get("save_to_library", True))

    last_backend_uuid = extras.get("last_backend_uuid") or extras.get("thread_uuid")
    is_followup = last_backend_uuid is not None

    frontend_uuid = str(uuid.uuid4())
    frontend_context_uuid = extras.get("frontend_context_uuid") or str(uuid.uuid4())

    # TODO: determine field requirements/usage, then properly parameterize.
    params: dict[str, Any] = {
        "version": PERPLEXITY_API_VERSION,
        "source": "default",
        "language": extras.get("language", "en-US"),
        "timezone": extras.get("timezone", "America/Los_Angeles"),
        "search_focus": _SEARCH_MAP.get(extras.get("search_focus", "web"), "internet"),
        "sources": sources,
        "search_recency_filter": _TIME_MAP.get(extras.get("time_range", "all"), "")
        or None,
        "mode": meta["mode"],
        "model_preference": meta["identifier"],
        "frontend_uuid": frontend_uuid,
        "frontend_context_uuid": frontend_context_uuid,
        "is_incognito": not save_to_library,
        "use_schematized_api": True,
        "send_back_text_in_streaming_api": False,
        "prompt_source": "user",
        "dsl_query": query,
        "is_related_query": False,
        "is_sponsored": False,
        "time_from_first_type": 8758 if is_followup else 18361,
        "local_search_enabled": client_coords is not None,
        "client_coordinates": client_coords,
        "mentions": extras.get("mentions", []),
        "attachments": extras.get("attachments", []),
        "skip_search_enabled": True,
        "is_nav_suggestions_disabled": True,
        "always_search_override": False,
        "override_no_search": False,
        "should_ask_for_mcp_tool_confirmation": True,
        "browser_agent_allow_once_from_toggle": False,
        "force_enable_browser_agent": False,
        "supported_features": PERPLEXITY_FEATURES,
        "supported_block_use_cases": PERPLEXITY_BLOCK_USE_CASES,
    }

    space_uuid = extras.get("space_uuid")
    if space_uuid:
        params["target_collection_uuid"] = space_uuid
        params["target_thread_access_level"] = 1
        params["query_source"] = "collection"
        params["is_incognito"] = False
    elif is_followup:
        params["query_source"] = "followup"
        params["followup_source"] = "link"
        params["last_backend_uuid"] = last_backend_uuid
        read_write_token = extras.get("read_write_token")
        if read_write_token:
            params["read_write_token"] = read_write_token
    else:
        params["query_source"] = "home"

    return {"params": params, "query_str": query}


@dataclass
class StreamState:
    """Running state across SSE events for a single Perplexity response."""

    answer_seen: str = ""
    reasoning_seen: str = ""
    ids: dict[str, str] = field(default_factory=dict)
    followups: list[str] = field(default_factory=list)
    final: bool = False
    # Step rendering — populated by `render_step` via `_extract_deltas`.
    # See `pplx_steps.py` for the renderer dispatch.
    mcp_steps: list[dict[str, Any]] = field(default_factory=list)
    all_steps: list[dict[str, Any]] = field(default_factory=list)
    goals: list[dict[str, Any]] = field(default_factory=list)
    seen_step_uuids: set[str] = field(default_factory=set)
    logged_unknown_intended_usages: set[str] = field(default_factory=set)
    # Per-step reasoning accumulator (separate from `reasoning_seen` which
    # tracks cumulative goal description text). Streaming path emits via
    # `reasoning_delta`; non-streaming reads this accumulator at finalize.
    step_reasoning: str = ""


_PPLX_ID_FIELDS: tuple[str, ...] = (
    "backend_uuid",
    "read_write_token",
    "context_uuid",
    "thread_url_slug",
    "thread_title",
    "display_model",
)


def _parse_sse_line(line: str | bytes) -> dict[str, Any] | None:
    """Parse a single SSE ``data:`` line. Returns None for non-data lines."""
    if isinstance(line, bytes):
        if not line.startswith(b"data: "):
            return None
        payload = line[6:]
    elif isinstance(line, str):
        if not line.startswith("data: "):
            return None
        payload = line[6:]
    else:
        return None

    if not payload or payload.strip() in (b"[DONE]", "[DONE]"):
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _attach_non_spec_fields(response: Any, state: StreamState) -> None:
    """Stamp Perplexity-only fields onto the OpenAI response object.

    Mirrors how ``pplx_thread_url_slug`` was previously attached: best-effort
    setattr on a Pydantic model that doesn't declare the field. LiteLLM
    serialises unknown attrs into the response JSON; standard OpenAI clients
    ignore them.
    """
    slug = state.ids.get("thread_url_slug")
    if slug:
        try:
            response.pplx_thread_url_slug = slug
        except Exception:
            pass
    if state.ids.get("thread_title"):
        try:
            response.pplx_thread_title = state.ids["thread_title"]
        except Exception:
            pass
    if state.mcp_steps:
        try:
            response.pplx_mcp_steps = state.mcp_steps
        except Exception:
            pass
    if state.all_steps:
        try:
            response.pplx_steps = state.all_steps
        except Exception:
            pass
    if state.goals:
        try:
            response.pplx_goals = state.goals
        except Exception:
            pass
    if state.followups:
        try:
            response.pplx_pending_followups = state.followups
        except Exception:
            pass


def _consume_step(step: dict[str, Any], state: StreamState) -> str:
    """Render one step and route into StreamState. Returns reasoning text to emit.

    Dedups across SSE events via ``state.seen_step_uuids``. Pushes structured
    data into ``state.all_steps`` (every step), ``state.mcp_steps`` (MCP only),
    and accumulates rendered text into ``state.step_reasoning`` for the
    non-streaming finalize path.

    Pre-rendering, MCP_TOOL_OUTPUT steps borrow ``tool_name`` from the
    matching MCP_TOOL_INPUT by ``goal_id`` — the structured channel omits
    tool_name on outputs, so without this pairing the renderer would fall
    back to the generic "tool" placeholder.
    """
    uuid_ = step.get("uuid") or ""
    if uuid_ and uuid_ in state.seen_step_uuids:
        return ""
    if uuid_:
        state.seen_step_uuids.add(uuid_)

    if step.get("step_type") == "MCP_TOOL_OUTPUT":
        content = step.get("mcp_tool_output_content") or step.get("content") or {}
        if isinstance(content, dict) and not content.get("tool_name"):
            goal_id = content.get("goal_id")
            if goal_id is not None:
                for prior in reversed(state.mcp_steps):
                    if prior.get("phase") == "input" and prior.get("goal_id") == goal_id:
                        # Mutate a copy of step so render_step sees tool_name
                        step = {**step, "tool_name": prior.get("tool_name")}
                        break

    result = render_step(step)
    if result.structured:
        step_type = step.get("step_type") or "UNKNOWN"
        state.all_steps.append({"step_type": step_type, **result.structured})
        if "mcp_step" in result.structured:
            state.mcp_steps.append(result.structured["mcp_step"])
    if result.reasoning_text:
        state.step_reasoning += result.reasoning_text
    return result.reasoning_text


def _extract_deltas(
    event: dict[str, Any], state: StreamState
) -> tuple[str | None, str | None]:
    """Apply one SSE event to ``state``; return new (answer_delta, reasoning_delta).

    Walks ``event["blocks"][*]``:
    - ``diff_block.patches[]`` on a ``markdown_block`` field carries the
      cumulative answer; emit prefix-diff against ``state.answer_seen``.
    - ``plan_block.goals[].description`` (in ``pro_search_steps`` / ``plan``
      blocks) carries cumulative reasoning text; emit prefix-diff against
      ``state.reasoning_seen``.
    - ``pending_followups_block.followups[]`` populates ``state.followups``.

    Captures the six thread-identifying fields from the event top level
    into ``state.ids`` lazily — they arrive on different events per
    ``core-query.md:1260-1273``.

    Raises ``PerplexityClarifyingQuestionsError`` when a
    ``RESEARCH_CLARIFYING_QUESTIONS`` step block appears (Deep Research mode).
    """
    for key in _PPLX_ID_FIELDS:
        val = event.get(key)
        if isinstance(val, str) and val:
            state.ids[key] = val

    # ``final_sse_message=true`` is set on exactly one event — the true
    # terminator. ``final=true`` may appear on the second-to-last event too,
    # but that one still carries meaningful blocks; gating only on
    # ``final_sse_message`` prevents emitting ``finish_reason=stop`` early.
    if event.get("final_sse_message"):
        state.final = True

    answer_delta: str | None = None
    reasoning_delta: str | None = None

    blocks = event.get("blocks") or []
    if not isinstance(blocks, list):
        blocks = []

    # The top-level ``text`` field carries the same step list as
    # ``plan_block.steps[]``, but JSON-encoded. We always raise on
    # RESEARCH_CLARIFYING_QUESTIONS (it surfaces as a 400 to the client),
    # but for other step types we only walk this fallback channel when the
    # event has no ``plan_block`` blocks — otherwise we'd double-emit
    # whatever the structured channel will also emit below.
    text = event.get("text")
    has_plan_block_this_event = any(
        isinstance(b, dict) and isinstance(b.get("plan_block"), dict) for b in blocks
    )
    if isinstance(text, str):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            for step in parsed:
                if not isinstance(step, dict):
                    continue
                st = step.get("step_type")
                if st == "RESEARCH_CLARIFYING_QUESTIONS":
                    raise PerplexityClarifyingQuestionsError(
                        _extract_clarifying_questions(step)
                    )
                if has_plan_block_this_event:
                    continue
                rendered = _consume_step(step, state)
                if rendered:
                    reasoning_delta = (reasoning_delta or "") + rendered

    for block in blocks:
        if not isinstance(block, dict):
            continue

        intended_usage = block.get("intended_usage")

        if intended_usage in ("pro_search_steps", "plan", "reasoning_plan_block"):
            plan_block = block.get("plan_block") or {}
            goals = plan_block.get("goals") or []
            if isinstance(goals, list):
                # Snapshot the latest goals[] for the non-spec response field
                # (server sends cumulative; last write wins).
                cleaned: list[dict[str, Any]] = []
                for goal in goals:
                    if not isinstance(goal, dict):
                        continue
                    cleaned.append(goal)
                    desc = goal.get("description")
                    if isinstance(desc, str) and desc.startswith(state.reasoning_seen):
                        new = desc[len(state.reasoning_seen) :]
                        if new:
                            reasoning_delta = (reasoning_delta or "") + new
                            state.reasoning_seen = desc
                if cleaned:
                    state.goals = cleaned

            # Walk plan_block.steps[] for the full step inventory: MCP tool
            # calls, web searches, browser-agent actions, image generation, etc.
            # See pplx_steps.py for renderer dispatch.
            for step in (plan_block.get("steps") or []):
                if not isinstance(step, dict):
                    continue
                rendered = _consume_step(step, state)
                if rendered:
                    reasoning_delta = (reasoning_delta or "") + rendered

        if intended_usage == "pending_followups":
            fb = block.get("pending_followups_block") or {}
            ups = fb.get("followups") or []
            if isinstance(ups, list):
                captured: list[str] = []
                for u in ups:
                    if isinstance(u, dict):
                        t = u.get("text")
                        if isinstance(t, str) and t:
                            captured.append(t)
                if captured:
                    state.followups = captured

        # Bare ``markdown_block`` (no ``diff_block`` wrapper) — the terminal
        # event re-sends the full answer this way. Usually redundant because
        # the diff_block stream has already accumulated the same content,
        # but Mode A-style prefix-diff keeps it safe and surfaces any tail
        # text we'd otherwise drop.
        mb = block.get("markdown_block")
        if isinstance(mb, dict) and not block.get("diff_block") and intended_usage != "ask_text":
            answer_str = mb.get("answer")
            if isinstance(answer_str, str) and answer_str:
                if answer_str.startswith(state.answer_seen):
                    bare_delta = answer_str[len(state.answer_seen) :]
                    if bare_delta:
                        answer_delta = (answer_delta or "") + bare_delta
                    state.answer_seen = answer_str

        diff_block = block.get("diff_block")
        if not isinstance(diff_block, dict):
            # No diff_block on this block — log unknown intended_usage so we
            # discover new block types instead of silently dropping them.
            if intended_usage and intended_usage not in _KNOWN_INTENDED_USAGES:
                if intended_usage not in state.logged_unknown_intended_usages:
                    state.logged_unknown_intended_usages.add(intended_usage)
                    logger.debug(
                        "pplx: unhandled intended_usage=%s keys=%s",
                        intended_usage,
                        list(block.keys()),
                    )
            continue

        # Perplexity sends the answer in two parallel blocks: ``ask_text_0_markdown``
        # (markdown-formatted) and ``ask_text`` (plain text). They carry identical
        # patches; processing both would double every chunk. Markdown wins.
        if intended_usage == "ask_text":
            continue

        field_name = diff_block.get("field")
        patches = diff_block.get("patches") or []
        if not isinstance(patches, list):
            continue

        for patch in patches:
            if not isinstance(patch, dict):
                continue
            path = patch.get("path", "")
            value = patch.get("value")

            if path.startswith("/goals"):
                if isinstance(value, str) and value.startswith(state.reasoning_seen):
                    new = value[len(state.reasoning_seen) :]
                    if new:
                        reasoning_delta = (reasoning_delta or "") + new
                        state.reasoning_seen = value
                continue

            if path == "/progress":
                continue

            if field_name != "markdown_block":
                continue

            # Mode A — root patch with the full markdown_block state. Carries
            # either a fresh ``chunks`` array (``chunk_starting_offset=0``) or
            # a cumulative ``answer`` string. Per core-query.md:716-757.
            if path == "" and isinstance(value, dict):
                chunks = value.get("chunks")
                if isinstance(chunks, list):
                    offset = value.get("chunk_starting_offset")
                    new_text = "".join(c for c in chunks if isinstance(c, str))
                    if offset in (None, 0):
                        if new_text != state.answer_seen:
                            if new_text.startswith(state.answer_seen):
                                delta = new_text[len(state.answer_seen) :]
                            else:
                                delta = new_text
                            if delta:
                                answer_delta = (answer_delta or "") + delta
                            state.answer_seen = new_text
                    elif new_text:
                        answer_delta = (answer_delta or "") + new_text
                        state.answer_seen += new_text
                answer_str = value.get("answer")
                if isinstance(answer_str, str) and answer_str:
                    if answer_str.startswith(state.answer_seen):
                        delta = answer_str[len(state.answer_seen) :]
                        if delta:
                            answer_delta = (answer_delta or "") + delta
                        state.answer_seen = answer_str
                continue

            # Mode B — incremental chunk append at ``/chunks/N``. Each patch
            # carries one new chunk as a string value.
            if path.startswith("/chunks/") and isinstance(value, str):
                state.answer_seen += value
                answer_delta = (answer_delta or "") + value
                continue

            # Mode C — cumulative answer at ``/markdown_block`` (legacy path).
            if path == "/markdown_block" and isinstance(value, dict):
                answer_str = value.get("answer")
                if isinstance(answer_str, str) and answer_str:
                    if answer_str.startswith(state.answer_seen):
                        delta = answer_str[len(state.answer_seen) :]
                        if delta:
                            answer_delta = (answer_delta or "") + delta
                        state.answer_seen = answer_str
                    elif answer_str != state.answer_seen:
                        answer_delta = (answer_delta or "") + answer_str
                        state.answer_seen = answer_str
                continue

            # Mode D — direct string at ``/markdown_block/answer``.
            if path == "/markdown_block/answer" and isinstance(value, str):
                if value.startswith(state.answer_seen):
                    delta = value[len(state.answer_seen) :]
                    if delta:
                        answer_delta = (answer_delta or "") + delta
                    state.answer_seen = value
                elif value != state.answer_seen:
                    answer_delta = (answer_delta or "") + value
                    state.answer_seen = value
                continue

    return answer_delta, reasoning_delta


def _extract_clarifying_questions(step: dict[str, Any]) -> list[str]:
    """Pull question strings from a RESEARCH_CLARIFYING_QUESTIONS step block."""
    questions: list[str] = []
    content = step.get("content")
    if isinstance(content, dict):
        for key in ("questions", "clarifying_questions"):
            raw = content.get(key)
            if isinstance(raw, list):
                questions.extend(str(q) for q in raw if q)
        if not questions:
            for value in content.values():
                if isinstance(value, str) and "?" in value:
                    questions.append(value)
    elif isinstance(content, list):
        questions = [str(q) for q in content if q]
    elif isinstance(content, str):
        questions = [content]
    return questions


def _format_citations(
    text: str | None,
    citation_mode: str,
    web_results: list[dict[str, Any]] | None,
) -> str | None:
    """Apply citation formatting to answer text.

    Modes per ``core-query.md:153-192``:
    - ``"markdown"`` (default): ``[N]`` → ``[N](url)`` using ``web_results``.
    - ``"default"``: preserve markers verbatim.
    - ``"clean"``: strip markers entirely.
    """
    if not text or citation_mode == "default":
        return text
    results = web_results or []

    def replacer(m: re.Match[str]) -> str:
        num = m.group(1)
        if not num.isdigit():
            return m.group(0)
        if citation_mode == "clean":
            return ""
        idx = int(num) - 1
        if 0 <= idx < len(results):
            url = results[idx].get("url") if isinstance(results[idx], dict) else None
            if citation_mode == "markdown" and url:
                return f"[{num}]({url})"
        return m.group(0)

    return _CITATION_PATTERN.sub(replacer, text)


def _extract_answer_from_entry(
    entry: dict[str, Any],
    citation_mode: str = "markdown",
) -> tuple[str, list[dict[str, Any]]]:
    """Pull the answer markdown + web_results from a thread entry's ``blocks[]``.

    Reads:
    - ``entry.structured_answer_block_usages`` (e.g. ``["ask_text_0_markdown"]``)
      names the block carrying the canonical answer; default to that name.
    - That block's ``markdown_block.answer`` is the raw answer string.
    - The first ``intended_usage == "web_results"`` block carries
      ``web_result_block.web_results[]`` for citation numbering.
    """
    blocks = entry.get("blocks") or []
    if not isinstance(blocks, list):
        return "", []

    usages = entry.get("structured_answer_block_usages")
    answer_iu = (
        usages[0]
        if isinstance(usages, list) and usages and isinstance(usages[0], str)
        else "ask_text_0_markdown"
    )

    raw_answer = ""
    web_results: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        iu = block.get("intended_usage")
        if iu == answer_iu and not raw_answer:
            mb = block.get("markdown_block") or {}
            if isinstance(mb, dict):
                ans = mb.get("answer")
                if isinstance(ans, str):
                    raw_answer = ans
        elif iu == "web_results" and not web_results:
            wrb = block.get("web_result_block") or {}
            if isinstance(wrb, dict):
                wrs = wrb.get("web_results") or []
                if isinstance(wrs, list):
                    web_results = [w for w in wrs if isinstance(w, dict)]

    text = _format_citations(raw_answer, citation_mode, web_results)
    return (text or ""), web_results


def _thread_to_openai_messages(
    thread: dict[str, Any],
    citation_mode: str = "markdown",
    include_reasoning: bool = False,
) -> list[dict[str, str]]:
    """Convert a Perplexity thread (``GET /rest/thread/{slug}`` response) to
    an OpenAI ``messages[]`` array.

    Each thread entry produces a ``(user, assistant)`` pair. Attachments
    become a ``[Attached: filename...]`` trailer on the user content (S3
    URLs are session-bearer-scoped and would not work outside Perplexity).
    Reasoning is omitted by default; if ``include_reasoning=True``, the
    plan_block goals descriptions are appended as a markdown footnote.
    """
    out: list[dict[str, str]] = []
    entries = thread.get("entries") or []
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        user_text = entry.get("query_str") or ""
        attachments = entry.get("attachments") or []
        if isinstance(attachments, list) and attachments:
            names = [str(a) for a in attachments if a]
            if names:
                user_text = f"{user_text}\n\n[Attached: {', '.join(names)}]"
        out.append({"role": "user", "content": user_text})

        answer_text, _web = _extract_answer_from_entry(entry, citation_mode)

        if include_reasoning:
            reasoning_lines: list[str] = []
            for block in entry.get("blocks") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("intended_usage") not in (
                    "pro_search_steps",
                    "plan",
                    "reasoning_plan_block",
                ):
                    continue
                plan = block.get("plan_block") or {}
                goals = plan.get("goals") or []
                if not isinstance(goals, list):
                    continue
                for g in goals:
                    if isinstance(g, dict):
                        d = g.get("description")
                        if isinstance(d, str) and d:
                            reasoning_lines.append(d)
            if reasoning_lines:
                answer_text = (
                    f"{answer_text}\n\n---\n**Reasoning:**\n\n- "
                    + "\n- ".join(reasoning_lines)
                )

        out.append({"role": "assistant", "content": answer_text})
    return out


class PerplexityException(BaseLLMException):
    pass


class PerplexityThreadNotFoundError(PerplexityException):
    pass


class PerplexityClarifyingQuestionsError(PerplexityException):
    """Deep Research returned clarifying questions instead of an answer."""

    def __init__(self, questions: list[str]) -> None:
        message = "Perplexity Deep Research requires clarification: " + "; ".join(
            questions
        )
        super().__init__(status_code=400, message=message, headers=None)
        self.questions = questions


class PerplexityProConfig(BaseConfig):
    """LiteLLM ``BaseConfig`` for the Perplexity Pro WebUI subscription path."""

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        return False

    def get_supported_openai_params(self, model: str) -> list[str]:
        return ["stream"]

    def map_openai_params(
        self,
        non_default_params: dict[str, Any],
        optional_params: dict[str, Any],
        model: str,
        drop_params: bool,
    ) -> dict[str, Any]:
        out = dict(optional_params)
        if "pplx" in non_default_params:
            out["pplx"] = non_default_params["pplx"]
        return out

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, Any],
        litellm_params: dict[str, Any],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        if not api_key:
            raise ValueError(
                "Perplexity Pro requires the session-token cookie value as api_key"
            )
        out = dict(headers)
        out["Cookie"] = f"{PERPLEXITY_SESSION_COOKIE}={api_key}"
        out["User-Agent"] = PERPLEXITY_BROWSER_UA
        out["Origin"] = PERPLEXITY_URL_BASE
        out["Referer"] = f"{PERPLEXITY_URL_BASE}/"
        out["Accept"] = "text/event-stream, application/json"
        out["Content-Type"] = "application/json"
        out["x-perplexity-request-reason"] = "perplexity-query-state-provider"
        out["x-app-apiversion"] = PERPLEXITY_API_VERSION
        out["x-app-apiclient"] = "default"
        out["x-request-id"] = str(uuid.uuid4())
        out["sec-fetch-dest"] = "empty"
        out["sec-fetch-mode"] = "cors"
        out["sec-fetch-site"] = "same-origin"
        return out

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, Any],
        litellm_params: dict[str, Any],
        stream: bool | None = None,
    ) -> str:
        return PERPLEXITY_URL

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, Any],
        litellm_params: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        raw_extras = optional_params.get("pplx") or {}
        extras: dict[str, Any] = raw_extras if isinstance(raw_extras, dict) else {}
        is_followup = bool(
            extras.get("last_backend_uuid") or extras.get("thread_uuid")
        )
        query = (
            _flatten_last_user_turn(messages)
            if is_followup
            else _flatten_messages(messages)
        )
        return _build_pplx_payload(
            query=query,
            model_id=model,
            extras=extras,
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict[str, Any],
        messages: list[AllMessageValues],
        optional_params: dict[str, Any],
        litellm_params: dict[str, Any],
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        state = StreamState()
        for raw_line in raw_response.text.splitlines():
            event = _parse_sse_line(raw_line)
            if event is None:
                continue
            try:
                _extract_deltas(event, state)
            except PerplexityClarifyingQuestionsError:
                raise

        from litellm.types.utils import Choices, Message

        message = Message(role="assistant", content=state.answer_seen)
        combined_reasoning = "\n".join(
            part for part in (state.reasoning_seen, state.step_reasoning.strip()) if part
        )
        if combined_reasoning:
            try:
                message.reasoning_content = combined_reasoning  # type: ignore[attr-defined]
            except Exception:
                pass

        model_response.id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        # Use the upstream-reported `display_model` so clients see which actual
        # model fired (e.g. "claude46sonnet") instead of the requested alias.
        model_response.model = state.ids.get("display_model") or model
        model_response.choices = [
            Choices(index=0, message=message, finish_reason="stop")
        ]

        _attach_non_spec_fields(model_response, state)
        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Any,
    ) -> BaseLLMException:
        return PerplexityException(
            status_code=status_code, message=error_message, headers=headers
        )

    def get_model_response_iterator(
        self,
        streaming_response: Any,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> Any:
        return PerplexityProIterator(
            streaming_response=iter([]),
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class PerplexityProIterator(BaseModelResponseIterator):
    """Stateful Perplexity SSE → OpenAI delta chunk parser.

    Each upstream event is parsed by ``_extract_deltas`` against ``_state``;
    the resulting ``(answer_delta, reasoning_delta)`` becomes one OpenAI
    ``ModelResponseStream`` chunk. On the final event (``final_sse_message``
    or ``final``), the captured ``thread_url_slug`` is stamped as a non-spec
    top-level field on the response so cooperating clients can echo it back
    via ``metadata.session_id`` on the next turn.
    """

    def __init__(
        self,
        streaming_response: Any,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> None:
        super().__init__(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
        self._state = StreamState()
        self._terminated = False

    def chunk_parser(self, chunk: dict[str, Any]) -> ModelResponseStream | None:
        if self._terminated:
            return None

        try:
            answer_delta, reasoning_delta = _extract_deltas(chunk, self._state)
        except PerplexityClarifyingQuestionsError as e:
            answer_delta = e.message
            reasoning_delta = None
            self._state.final = True

        from litellm.types.utils import Delta, StreamingChoices

        delta = Delta()
        if answer_delta:
            delta.content = answer_delta
        if reasoning_delta:
            try:
                delta.reasoning_content = reasoning_delta  # type: ignore[attr-defined]
            except Exception:
                pass

        if self._state.final:
            finish_reason: str | None = "stop"
            self._terminated = True
        else:
            finish_reason = None

        choice = StreamingChoices(
            index=0,
            delta=delta,
            finish_reason=finish_reason,
        )
        response = ModelResponseStream(choices=[choice])

        if self._state.final:
            # Stamp the upstream-reported model so clients see what actually fired
            display_model = self._state.ids.get("display_model")
            if display_model:
                try:
                    response.model = display_model  # type: ignore[assignment]
                except Exception:
                    pass
            _attach_non_spec_fields(response, self._state)
        return response
