"""Render Perplexity SSE step events into reasoning text + structured data.

Perplexity's `plan_block.steps[]` and the parallel top-level `text`-field
JSON channel both carry the same `step_type`-tagged step objects with
typed `*_content` fields. There are 65+ step_type values in the SPA bundle (see `docs/pplx/step_types.md`); we
ship specialized renderers for the common categories (MCP tool calls,
web search, browser agent, calendar/email, image generation, etc.) and
a generic fallback that captures unknown step types as structured data
plus a DEBUG log so we discover new ones in the wild instead of silently
dropping them.

The naming convention is regular: `UPPER_SNAKE_CASE` step_type ↔
`lower_snake_case_content` typed field
(e.g. ``MCP_TOOL_INPUT`` → ``mcp_tool_input_content``). ``render_step``
tolerates both the structured shape (typed `*_content` field) and the
text-field shape (generic `content` key).

Render results are consumed by ``_extract_deltas`` in ``pplx.py`` and
flow into ``delta.reasoning_content`` (Claude-style thinking blocks) +
non-spec response fields (``pplx_mcp_steps``, ``pplx_steps``,
``pplx_goals``, etc.).
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "StepRenderResult",
    "content_field_for",
    "render_step",
]


@dataclass
class StepRenderResult:
    """Output of a step renderer.

    ``reasoning_text`` is appended to ``delta.reasoning_content`` (or
    accumulated into ``Message.reasoning_content`` for non-streaming).
    ``structured`` carries an optional dict that's appended to
    ``state.all_steps`` and, when keyed ``"mcp_step"``, additionally to
    ``state.mcp_steps`` for the non-spec ``pplx_mcp_steps`` response field.
    """

    reasoning_text: str = ""
    structured: dict[str, Any] | None = None


def content_field_for(step_type: str) -> str:
    """Map ``MCP_TOOL_INPUT`` → ``mcp_tool_input_content``.

    Reverse-engineered from the SPA bundle's ``??`` fallback chain in
    ``ThreadEntryContext-hgdcVwpW.js`` — every step_type uses the
    lowercase-underscore form of its enum name plus ``_content``.
    """
    return step_type.lower() + "_content"


def render_step(step: dict[str, Any]) -> StepRenderResult:
    """Dispatch a step to its renderer.

    Reads ``step["step_type"]``, finds the typed content field via the
    naming convention, falls back to a generic ``content`` key for the
    text-field JSON shape, and dispatches to the matching renderer. Unknown
    step types route to ``_render_generic`` which captures the full
    content dict as structured data and logs at DEBUG.

    Outer-level fields like ``tool_name`` and ``tool_input_summary`` on the
    step itself (observed on ``MCP_TOOL_OUTPUT`` wire shape) are merged
    into the content dict as defaults so renderers don't have to special-case
    where they live.
    """
    step_type = step.get("step_type") or "UNKNOWN"
    uuid_ = step.get("uuid", "")
    content_key = content_field_for(step_type)
    content_obj = step.get(content_key)
    if not isinstance(content_obj, dict):
        fallback = step.get("content")
        content_obj = fallback if isinstance(fallback, dict) else {}
    # Merge outer-level metadata into content as defaults — Perplexity puts
    # tool_name + tool_input_summary at the OUTER level on MCP_TOOL_OUTPUT.
    merged: dict[str, Any] = dict(content_obj)
    for outer_key in ("tool_name", "tool_input_summary"):
        if outer_key not in merged and step.get(outer_key) is not None:
            merged[outer_key] = step[outer_key]
    renderer = _RENDERERS.get(step_type, _render_generic)
    return renderer(step_type, merged, uuid_)


# ---- Suppressed (redundant with other channels) -------------------------


def _render_suppressed(
    _step_type: str, _content: dict[str, Any], _uuid: str
) -> StepRenderResult:
    """INITIAL_QUERY (already in user msg) and FINAL (already in markdown_block)."""
    return StepRenderResult()


# ---- Core / control ----------------------------------------------------


def _render_terminate(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    reason = content.get("reason") or content.get("message") or ""
    text = "✓ Done" + (f" — {reason}" if reason else "") + "\n"
    return StepRenderResult(
        text, {"phase": "terminate", "step_uuid": uuid, "reason": reason}
    )


def _render_attachment(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    name = content.get("name") or content.get("filename") or "attachment"
    text = f"📎 Processing attachment: {name}\n"
    return StepRenderResult(
        text, {"phase": "attachment", "step_uuid": uuid, "name": name}
    )


# ---- Web search --------------------------------------------------------


def _render_search_web(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    queries = content.get("queries") or []
    if isinstance(queries, list) and queries:
        q_str = " · ".join(str(q) for q in queries if q)
    else:
        q_str = str(content.get("query") or "")
    text = f"→ Web search: {q_str}\n" if q_str else "→ Web search\n"
    return StepRenderResult(
        text, {"phase": "search", "step_uuid": uuid, "queries": queries or [q_str]}
    )


def _render_web_results(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    results = content.get("web_results") or content.get("results") or []
    n = len(results) if isinstance(results, list) else 0
    text = f"← {n} web result{'s' if n != 1 else ''}\n"
    return StepRenderResult(
        text, {"phase": "web_results", "step_uuid": uuid, "count": n}
    )


def _render_read_results(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    urls = content.get("urls") or []
    n = len(urls) if isinstance(urls, list) else 0
    sample = urls[:3] if isinstance(urls, list) else []
    text = f"← Read {n} result{'s' if n != 1 else ''}"
    if sample:
        text += (
            " (" + ", ".join(str(u) for u in sample) + (", …" if n > 3 else "") + ")"
        )
    text += "\n"
    return StepRenderResult(
        text, {"phase": "read_results", "step_uuid": uuid, "urls": urls or []}
    )


def _render_get_url_content(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    url = content.get("url") or ""
    text = f"→ Fetch URL: {url}\n"
    return StepRenderResult(text, {"phase": "fetch_url", "step_uuid": uuid, "url": url})


# ---- MCP tool calls ----------------------------------------------------


def _render_mcp_tool_input(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    app = content.get("app") or "unknown"
    tool_name = content.get("tool_name") or content.get("tool_id") or "unknown"
    tool_args = (
        content.get("tool_args") if isinstance(content.get("tool_args"), dict) else {}
    )
    summary = content.get("tool_input_summary") or ""
    args_repr = json.dumps(tool_args, separators=(",", ":")) if tool_args else "{}"
    text = f"→ [{app}] {tool_name}({args_repr})"
    if summary:
        text += f": {summary}"
    text += "\n"

    rua = content.get("request_user_approval") or {}
    needs_approval = bool(rua.get("request_user_approval"))

    structured: dict[str, Any] = {
        "phase": "input",
        "step_uuid": uuid,
        "app": app,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "goal_id": content.get("goal_id"),
        "summary": summary,
        "needs_user_approval": needs_approval,
        "approval_result": content.get("approval_result"),
        "mcp_server_type": content.get("mcp_server_type"),
        "source_type": content.get("source_type"),
        "authenticated": content.get("authenticated"),
        "logo_url": content.get("logo_url"),
    }
    return StepRenderResult(text, {"mcp_step": structured})


def _render_mcp_tool_output(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    tool_name = content.get("tool_name") or content.get("tool_id") or "tool"
    status = content.get("status") or "unknown"
    text = f"← {tool_name} ({status})\n"

    raw_content = content.get("content")
    parsed_content: Any = raw_content
    if isinstance(raw_content, str):
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            parsed_content = json.loads(raw_content)

    structured: dict[str, Any] = {
        "phase": "output",
        "step_uuid": uuid,
        "tool_name": tool_name,
        "status": status,
        "content": parsed_content,
        "goal_id": content.get("goal_id"),
        "app": content.get("app"),
        "authenticated": content.get("authenticated"),
        "should_rerun_query": content.get("should_rerun_query"),
        "data_is_redacted": content.get("data_is_redacted"),
    }
    return StepRenderResult(text, {"mcp_step": structured})


# ---- Comet agent (Perplexity browser agent) ----------------------------


def _render_comet_agent_input(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    task = content.get("task_uuid") or content.get("task") or ""
    text = f"→ Comet agent: {task}\n" if task else "→ Comet agent\n"
    return StepRenderResult(
        text, {"phase": "comet_input", "step_uuid": uuid, "task": task}
    )


def _render_comet_agent_output(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    status = content.get("status") or "done"
    text = f"← Comet agent ({status})\n"
    return StepRenderResult(
        text, {"phase": "comet_output", "step_uuid": uuid, "status": status}
    )


# ---- Browser agent (Deep Research browser mode) ------------------------


def _render_browser_search(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    q = content.get("query") or content.get("queries") or ""
    text = f"→ Browser search: {q}\n" if q else "→ Browser search\n"
    return StepRenderResult(
        text, {"phase": "browser_search", "step_uuid": uuid, "query": q}
    )


def _render_url_navigate(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    url = content.get("url") or ""
    text = f"→ Browser navigate: {url}\n"
    return StepRenderResult(
        text, {"phase": "browser_navigate", "step_uuid": uuid, "url": url}
    )


def _render_browser_open_tab(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    url = content.get("url") or ""
    text = f"→ Browser open tab: {url}\n"
    return StepRenderResult(
        text, {"phase": "browser_open_tab", "step_uuid": uuid, "url": url}
    )


def _render_browser_get_site_content(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    url = content.get("url") or ""
    text = f"← Read page: {url}\n" if url else "← Read page\n"
    return StepRenderResult(
        text, {"phase": "browser_get_content", "step_uuid": uuid, "url": url}
    )


# ---- Productivity / agent steps ----------------------------------------


def _render_code(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    lang = content.get("language") or ""
    text = f"💻 Code execution{f' ({lang})' if lang else ''}\n"
    return StepRenderResult(
        text, {"phase": "code", "step_uuid": uuid, "language": lang, "content": content}
    )


def _render_generate_image(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    prompt = content.get("prompt") or ""
    text = f"🎨 Generating image: {prompt}\n" if prompt else "🎨 Generating image\n"
    return StepRenderResult(
        text, {"phase": "image_gen", "step_uuid": uuid, "prompt": prompt}
    )


def _render_generate_image_results(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    results = content.get("image_results") or content.get("images") or []
    n = len(results) if isinstance(results, list) else 0
    text = f"← {n} image{'s' if n != 1 else ''} generated\n"
    return StepRenderResult(
        text, {"phase": "image_results", "step_uuid": uuid, "results": results or []}
    )


def _render_create_chart(
    _step_type: str, _content: dict[str, Any], uuid: str
) -> StepRenderResult:
    text = "📊 Creating chart\n"
    return StepRenderResult(text, {"phase": "create_chart", "step_uuid": uuid})


def _render_create_tasks(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    tasks = content.get("tasks") or []
    n = len(tasks) if isinstance(tasks, list) else 0
    text = f"📋 Creating {n} task{'s' if n != 1 else ''}\n"
    return StepRenderResult(
        text, {"phase": "create_tasks", "step_uuid": uuid, "tasks": tasks or []}
    )


# ---- Calendar / Email agent (legacy connectors) ------------------------


def _render_read_calendar(
    _step_type: str, _content: dict[str, Any], uuid: str
) -> StepRenderResult:
    return StepRenderResult(
        "→ Calendar: read\n", {"phase": "calendar_read", "step_uuid": uuid}
    )


def _render_update_calendar(
    _step_type: str, _content: dict[str, Any], uuid: str
) -> StepRenderResult:
    return StepRenderResult(
        "→ Calendar: update\n", {"phase": "calendar_update", "step_uuid": uuid}
    )


def _render_read_email(
    _step_type: str, _content: dict[str, Any], uuid: str
) -> StepRenderResult:
    return StepRenderResult(
        "→ Email: read\n", {"phase": "email_read", "step_uuid": uuid}
    )


def _render_send_email(
    _step_type: str, _content: dict[str, Any], uuid: str
) -> StepRenderResult:
    return StepRenderResult(
        "→ Email: send\n", {"phase": "email_send", "step_uuid": uuid}
    )


# ---- Clarifying questions ----------------------------------------------


def _render_clarifying_questions(
    _step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    qs = content.get("questions") or []
    n = len(qs) if isinstance(qs, list) else 0
    text = f"❓ Clarifying questions ({n})\n"
    return StepRenderResult(
        text, {"phase": "clarifying", "step_uuid": uuid, "questions": qs or []}
    )


# ---- Generic fallback (DEBUG-logs unknowns) ----------------------------


def _render_generic(
    step_type: str, content: dict[str, Any], uuid: str
) -> StepRenderResult:
    """Catch-all for unmapped step types.

    Renders a minimal `[STEP_TYPE]` line + any obvious summary field, and
    captures the full content dict as structured data so nothing is
    silently dropped. Logs at DEBUG so unknowns surface in dev logs the
    first time they appear.
    """
    summary = (
        content.get("summary")
        or content.get("description")
        or content.get("query")
        or content.get("title")
        or ""
    )
    text = f"[{step_type}]" + (f" {summary}" if summary else "") + "\n"
    structured = {
        "phase": "unmapped",
        "step_type": step_type,
        "step_uuid": uuid,
        "content": content,
    }
    logger.debug(
        "pplx_steps: unmapped step_type=%s uuid=%s content_keys=%s",
        step_type,
        uuid,
        list(content.keys()) if content else [],
    )
    return StepRenderResult(text, {"unmapped_step": structured})


_Renderer = Callable[[str, dict[str, Any], str], StepRenderResult]


_RENDERERS: dict[str, _Renderer] = {
    # Suppressed (redundant)
    "INITIAL_QUERY": _render_suppressed,
    "FINAL": _render_suppressed,
    # Control
    "TERMINATE": _render_terminate,
    "ATTACHMENT": _render_attachment,
    # Web search
    "SEARCH_WEB": _render_search_web,
    "WEB_RESULTS": _render_web_results,
    "READ_RESULTS": _render_read_results,
    "GET_URL_CONTENT": _render_get_url_content,
    # MCP tool calls (the headline use case)
    "MCP_TOOL_INPUT": _render_mcp_tool_input,
    "MCP_TOOL_OUTPUT": _render_mcp_tool_output,
    # Comet agent
    "COMET_AGENT_TOOL_INPUT": _render_comet_agent_input,
    "COMET_AGENT_TOOL_OUTPUT": _render_comet_agent_output,
    # Browser agent
    "BROWSER_SEARCH": _render_browser_search,
    "SEARCH_BROWSER": _render_browser_search,
    "URL_NAVIGATE": _render_url_navigate,
    "BROWSER_OPEN_TAB": _render_browser_open_tab,
    "BROWSER_GET_SITE_CONTENT": _render_browser_get_site_content,
    # Productivity / agents
    "CODE": _render_code,
    "GENERATE_IMAGE": _render_generate_image,
    "GENERATE_IMAGE_RESULTS": _render_generate_image_results,
    "CREATE_CHART": _render_create_chart,
    "CREATE_TASKS": _render_create_tasks,
    # Calendar / Email connectors (legacy direct calls before MCP unification)
    "READ_CALENDAR": _render_read_calendar,
    "UPDATE_CALENDAR": _render_update_calendar,
    "READ_EMAIL": _render_read_email,
    "SEND_EMAIL": _render_send_email,
    # Clarifying questions (the non-raising one — RESEARCH_CLARIFYING_QUESTIONS
    # still raises in pplx._extract_deltas to surface as 400)
    "CLARIFYING_QUESTIONS": _render_clarifying_questions,
    # `_render_generic` handles every other step_type
}


_KNOWN_INTENDED_USAGES: frozenset[str] = frozenset(
    {
        "ask_text_0_markdown",
        "ask_text",
        "pro_search_steps",
        "plan",
        "reasoning_plan_block",
        "pending_followups",
        "sources_answer_mode",
        "web_results",
        "media_items",
        "image_answer_mode",
        "video_answer_mode",
        "answer_modes",
        "knowledge_cards",
        "inline_entity_cards",
        "place_widgets",
        "finance_widgets",
        "sports_widgets",
        "shopping_widgets",
        "jobs_widgets",
        "search_result_widgets",
        "diff_blocks",
        "inline_images",
        "inline_assets",
        "placeholder_cards",
        "inline_knowledge_cards",
        "entity_group_v2",
        "refinement_filters",
        "canvas_mode",
        "maps_preview",
        "answer_tabs",
        "price_comparison_widgets",
        "preserve_latex",
        "generic_onboarding_widgets",
        "in_context_suggestions",
        "inline_claims",
        "prediction_market_widgets",
        "flight_status_widgets",
        "news_widgets",
        "image_answer_generated",
        "answer_generated_image",
    }
)
