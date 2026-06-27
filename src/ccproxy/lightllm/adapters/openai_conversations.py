"""OpenAI Conversations UIAdapter (outbound-only).

Renders a :class:`~ccproxy.lightllm.adapters.LLMRenderInput` (typically a
:class:`~ccproxy.pipeline.context.Context`) to the ``/backend-api/f/conversation``
wire format used by chatgpt.com's web SPA.

Two pure body builders are exported for the CHATGPT-002 addon to reuse:

- :func:`build_conversation_body` — the final ``POST /f/conversation`` body.
- :func:`build_conversation_prepare_body` — the ``POST /f/conversation/prepare``
  body at each of the three prepare states (``none`` / ``sent`` / ``success``).

Threading semantics (ADR-0002):

- **New conversation** (``openai_conversations`` key absent from ``raw_extras``
  or ``is_continuation`` is ``False``): flatten the entire IR history into one
  user turn, emit ``parent_message_id: "client-created-root"``, omit
  ``conversation_id``.
- **Continuation** (``is_continuation`` is ``True``): emit only the latest user
  turn, set ``parent_message_id`` and ``conversation_id`` from ``raw_extras``.

Default model is ``gpt-5-5-pro`` read from
``config.lightllm.openai_conversations.default_model`` when the request carries
no model (ADR-0002 §3).

This is an OUTBOUND-ONLY adapter — :meth:`load_messages` and
:meth:`build_event_stream` raise :class:`NotImplementedError`.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextContent,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.ui import UIAdapter, UIEventStream

_PrepareState = Literal["none", "sent", "success"]

_EFFORT_MAP: dict[str, str] = {
    "low": "standard",
    "medium": "extended",
    "high": "max",
}

_TOOL_HINT_MAP: dict[str, str] = {
    "image_generation": "picture_v2",
    "web_search": "search",
    "web_search_preview": "search",
    "web_search_preview_2025_03_11": "search",
    "deep_research": "connector:connector_openai_deep_research",
}

_PREPARE_STRIP_KEYS: frozenset[str] = frozenset(
    {
        "messages",
        "enable_message_followups",
        "paragen_cot_summary_display_override",
        "force_parallel_switch",
        "client_contextual_info",
        "client_prepare_state",
    }
)


@dataclass
class OpenAIConversationsAdapter(UIAdapter[Any, dict[str, Any], Any, AgentDepsT, OutputDataT]):
    """Outbound-only UIAdapter for ChatGPT web ``/backend-api/f/conversation``.

    :meth:`load_messages` raises :class:`NotImplementedError` because ccproxy
    does not host an OpenAI Conversations listener. :meth:`render` projects IR
    messages to the ChatGPT web wire format with threading support.

    :meth:`build_event_stream` raises :class:`NotImplementedError`; streaming
    intake lives in :mod:`ccproxy.lightllm.graph.openai_conversations_intake`.
    """

    @classmethod
    def load_messages(cls, *_args: Any, **_kwargs: Any) -> list[ModelMessage]:
        raise NotImplementedError(
            "ccproxy does not host an OpenAI Conversations listener; OpenAIConversationsAdapter is outbound-only."
        )

    def build_event_stream(
        self,
    ) -> UIEventStream[Any, Any, AgentDepsT, OutputDataT]:
        raise NotImplementedError(
            "OpenAI Conversations streaming intake lives in ccproxy.lightllm.graph.openai_conversations_intake."
        )

    @classmethod
    def render(cls, req: Any) -> bytes:
        """Render an :class:`LLMRenderInput` to OpenAI Conversations wire bytes.

        Resolves the model (falling back to the configured default), reads
        threading state from ``req.raw_extras["openai_conversations"]``, and
        invokes :func:`build_conversation_body`.

        Args:
            req: The :class:`LLMRenderInput` (typically a Context) to render.

        Returns:
            JSON-encoded OpenAI Conversations wire body as bytes.
        """
        thread_info = _resolve_thread_info(raw_extras=req.raw_extras)
        is_continuation = bool(thread_info.get("is_continuation", False))
        conversation_id = str(thread_info.get("conversation_id", "")) if is_continuation else ""
        parent_message_id = str(thread_info.get("parent_message_id", "")) if is_continuation else ""

        model = req.model or _default_model()

        thinking_effort = _extract_thinking_effort(raw_extras=req.raw_extras)
        system_hints = _extract_system_hints(
            raw_extras=req.raw_extras,
            request_parameters=req.request_parameters,
        )
        temporary_chat: bool = bool(req.raw_extras.get("temporary_chat", False))

        body = build_conversation_body(
            messages_ir=req.messages,
            model=model,
            thinking_effort=thinking_effort,
            system_hints=system_hints,
            temporary_chat=temporary_chat,
            conversation_id=conversation_id,
            parent_message_id=parent_message_id,
            is_continuation=is_continuation,
        )
        return json.dumps(body).encode()


def build_conversation_body(
    *,
    messages_ir: list[ModelMessage],
    model: str,
    thinking_effort: str | None = None,
    system_hints: list[str] | None = None,
    temporary_chat: bool = False,
    conversation_id: str = "",
    parent_message_id: str = "",
    is_continuation: bool = False,
    prepared: bool = True,
) -> dict[str, Any]:
    """Build the ``POST /backend-api/f/conversation`` request body.

    Args:
        messages_ir: Pydantic-AI IR conversation history.
        model: Resolved ChatGPT model slug (must not be empty).
        thinking_effort: ChatGPT-native reasoning effort level
            (``standard`` / ``extended`` / ``max`` / ``min``), or ``None``
            to omit the field entirely.
        system_hints: ChatGPT system hint ids (e.g. ``["picture_v2", "search"]``).
            Defaults to an empty list when ``None``.
        temporary_chat: When ``True``, sets ``history_and_training_disabled: true``
            so the turn is excluded from the user's ChatGPT history. Defaults
            ``False`` (saved conversations per ADR-0002).
        conversation_id: ChatGPT server conversation id to continue.
            Omitted when empty or when ``is_continuation`` is ``False``.
        parent_message_id: Parent message id for this turn. When empty and
            ``is_continuation`` is ``False`` this defaults to
            ``"client-created-root"``.
        is_continuation: ``True`` to continue an existing conversation (emit only
            the latest user turn). ``False`` (default) flattens full history.
        prepared: When ``True`` (default), sets ``client_prepare_state: "sent"``
            to signal that a conduit prepare handshake was completed. The
            CHATGPT-002 addon owns the conduit chain; this flag marks the body
            as "prepared" for the addon's final request.

    Returns:
        Dict ready for ``json.dumps``.
    """
    hints: list[str] = system_hints if system_hints is not None else []
    effective_parent = parent_message_id or "client-created-root"

    if is_continuation:
        messages_block = _build_continuation_message(messages_ir=messages_ir)
    else:
        messages_block = _build_new_conversation_message(messages_ir=messages_ir)

    body: dict[str, Any] = {
        "action": "next",
        "messages": messages_block,
        "parent_message_id": effective_parent,
        "model": model,
        "client_prepare_state": "sent" if prepared else "none",
        "timezone_offset_min": -480,
        "timezone": "Asia/Shanghai",
        "conversation_mode": {"kind": "primary_assistant"},
        "enable_message_followups": True,
        "system_hints": hints,
        "supports_buffering": True,
        "supported_encodings": ["v1"],
        "client_contextual_info": {
            "is_dark_mode": False,
            "time_since_loaded": 5000,
            "page_height": 1039,
            "page_width": 1237,
            "pixel_ratio": 1.35,
            "screen_height": 1067,
            "screen_width": 1707,
            "app_name": "chatgpt.com",
        },
        "paragen_cot_summary_display_override": "allow",
        "force_parallel_switch": "auto",
    }

    # Emit history_and_training_disabled only when the temporary-chat toggle is
    # on (manual §5.4) — it protects the operator's ChatGPT history.
    if temporary_chat:
        body["history_and_training_disabled"] = True

    if thinking_effort is not None:
        body["thinking_effort"] = thinking_effort

    if is_continuation and conversation_id:
        body["conversation_id"] = conversation_id

    return body


def build_conversation_prepare_body(
    final_body: dict[str, Any],
    state: _PrepareState,
) -> dict[str, Any]:
    """Derive an ``/f/conversation/prepare`` body from the final conversation body.

    Strips fields that the prepare endpoint does not accept, adds
    ``fork_from_shared_post: false``, replaces ``client_contextual_info`` with the
    slim ``{app_name}`` form, and sets state-specific fields:

    - ``"none"``: omit ``partial_query``; model defaults to ``"auto"`` when empty.
    - ``"sent"`` / ``"success"``: include ``partial_query`` with the first 5 chars
      of the last user message.

    Args:
        final_body: The complete body produced by :func:`build_conversation_body`.
        state: One of ``"none"``, ``"sent"``, or ``"success"``.

    Returns:
        Dict ready for ``json.dumps`` as the prepare request body.
    """
    prepare: dict[str, Any] = {k: v for k, v in final_body.items() if k not in _PREPARE_STRIP_KEYS}
    prepare["fork_from_shared_post"] = False
    prepare["client_prepare_state"] = state
    prepare["client_contextual_info"] = {"app_name": "chatgpt.com"}

    model = prepare.get("model", "")
    if not model:
        prepare["model"] = "auto"

    if state in ("sent", "success"):
        partial_text = _extract_partial_query(final_body=final_body)
        prepare["partial_query"] = {
            "id": str(uuid.uuid4()),
            "author": {"role": "user"},
            "content": {
                "content_type": "text",
                "parts": [partial_text],
            },
        }

    return prepare


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_model() -> str:
    """Read the configured default model slug.

    Falls back to ``"gpt-5-5-pro"`` when the config singleton is unavailable
    (early startup or tests without a loaded config).
    """
    try:
        from ccproxy.config import get_config

        return get_config().lightllm.openai_conversations.default_model
    except Exception:
        return "gpt-5-5-pro"


def _resolve_thread_info(*, raw_extras: dict[str, Any]) -> dict[str, Any]:
    """Extract the ``openai_conversations`` threading block from ``raw_extras``."""
    raw = raw_extras.get("openai_conversations")
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}
    return {}


def _extract_thinking_effort(*, raw_extras: dict[str, Any]) -> str | None:
    """Pull and map ``thinking_effort`` from ``raw_extras`` or nested extras.

    Priority order mirrors gproxy ``extract_thinking_effort``:
    1. ``raw_extras["thinking_effort"]`` (ChatGPT-native top-level)
    2. ``raw_extras["extra_body"]["thinking_effort"]``
    3. ``raw_extras["extra_body"]["reasoning"]["effort"]`` — OpenAI Responses
    4. ``raw_extras["reasoning"]["effort"]``

    ``low`` / ``medium`` / ``high`` are mapped to ``standard`` / ``extended`` /
    ``max`` respectively. Other values pass through unchanged.
    """
    raw: str | None = None

    if isinstance(raw_extras.get("thinking_effort"), str):
        raw = raw_extras["thinking_effort"]
    else:
        extra_body = raw_extras.get("extra_body")
        if isinstance(extra_body, dict):
            if isinstance(extra_body.get("thinking_effort"), str):
                raw = extra_body["thinking_effort"]
            elif isinstance(extra_body.get("reasoning"), dict):
                effort = extra_body["reasoning"].get("effort")
                if isinstance(effort, str):
                    raw = effort
        if raw is None:
            reasoning = raw_extras.get("reasoning")
            if isinstance(reasoning, dict):
                effort = reasoning.get("effort")
                if isinstance(effort, str):
                    raw = effort

    if raw is None:
        return None
    return _EFFORT_MAP.get(raw, raw)


def _extract_system_hints(
    *,
    raw_extras: dict[str, Any],
    request_parameters: Any,
) -> list[str]:
    """Collect system hints from multiple sources, deduplicating in order.

    Sources (matching gproxy ``extract_system_hints``):
    1. ``raw_extras["system_hints"]`` — upstream-native hint ids
    2. ``raw_extras["extra_body"]["system_hints"]``
    3. ``request_parameters.function_tools`` — OpenAI Responses tool types mapped
       via :data:`_TOOL_HINT_MAP`
    """
    hints: list[str] = []

    def _push(s: str) -> None:
        if s and s not in hints:
            hints.append(s)

    for source in [
        raw_extras.get("system_hints"),
        (raw_extras.get("extra_body") or {}).get("system_hints"),
    ]:
        if isinstance(source, list):
            for item in source:
                if isinstance(item, str):
                    _push(item)

    tools = getattr(request_parameters, "function_tools", None) or []
    for tool in tools:
        tool_name = getattr(tool, "name", None) or ""
        hint = _TOOL_HINT_MAP.get(tool_name)
        if hint:
            _push(hint)

    return hints


def _extract_text_from_ir_messages(*, messages_ir: list[ModelMessage]) -> list[tuple[str, str]]:
    """Flatten IR messages into ``(role, text)`` pairs.

    Handles :class:`~pydantic_ai.messages.ModelRequest` (system + user parts)
    and :class:`~pydantic_ai.messages.ModelResponse` (text parts only).
    Non-text content is collapsed to an empty string fragment.
    """
    result: list[tuple[str, str]] = []
    for msg in messages_ir:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    result.append(("system", part.content))
                elif isinstance(part, UserPromptPart):
                    result.append(("user", _coerce_user_content(part.content)))
        elif isinstance(msg, ModelResponse):
            text_chunks: list[str] = []
            for resp_part in msg.parts:
                if isinstance(resp_part, TextPart):
                    text_chunks.append(resp_part.content)
            result.append(("assistant", "".join(text_chunks)))
    return result


def _coerce_user_content(content: Any) -> str:
    """Collapse user content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list | tuple):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, TextContent):
                parts.append(item.content)
        return "\n".join(parts)
    return str(content)


def _build_new_conversation_message(*, messages_ir: list[ModelMessage]) -> list[dict[str, Any]]:
    """Flatten the full IR history into a single ChatGPT user message.

    Non-last turns are prefixed as ``"<role>: <text>"`` lines. The last
    user turn is appended as the final instruction.

    Mirrors gproxy ``messages_to_chatgpt`` exactly.
    """
    pairs = _extract_text_from_ir_messages(messages_ir=messages_ir)

    prompt = ""
    last_user_text: str | None = None
    for role, text in pairs:
        if role == "user" and not prompt:
            last_user_text = text
            continue
        if prompt:
            prompt += "\n"
        prompt += f"{role}: {text}"

    if not prompt and last_user_text is not None:
        final_prompt = last_user_text
    elif prompt and last_user_text is not None:
        final_prompt = prompt + f"\nuser: {last_user_text}"
    else:
        final_prompt = prompt

    return [_make_user_message(text=final_prompt)]


def _build_continuation_message(*, messages_ir: list[ModelMessage]) -> list[dict[str, Any]]:
    """Emit only the latest user turn for a continuation request.

    ChatGPT already holds prior turns server-side; only the new user message
    is sent.
    """
    last_user_text = ""
    for msg in reversed(messages_ir):
        if isinstance(msg, ModelRequest):
            for part in reversed(msg.parts):
                if isinstance(part, UserPromptPart):
                    last_user_text = _coerce_user_content(part.content)
                    break
            if last_user_text:
                break

    return [_make_user_message(text=last_user_text)]


def _make_user_message(*, text: str) -> dict[str, Any]:
    """Build the ChatGPT user message object for the ``messages`` array."""
    return {
        "id": str(uuid.uuid4()),
        "author": {"role": "user"},
        "create_time": time.time(),
        "content": {"content_type": "text", "parts": [text]},
        "metadata": {
            "developer_mode_connector_ids": [],
            "selected_connector_ids": [],
            "selected_sync_knowledge_store_ids": [],
            "selected_sources": [],
            "selected_github_repos": [],
            "selected_all_github_repos": False,
            "serialization_metadata": {"custom_symbol_offsets": []},
        },
    }


def _extract_partial_query(*, final_body: dict[str, Any]) -> str:
    """Extract the first 5 rune-codepoints of the last user message text.

    Used for ``partial_query`` in the prepare bodies. Mirrors gproxy
    ``conversationPartialText`` / ``runeSlice``.
    """
    messages = final_body.get("messages", [])
    if not isinstance(messages, list):
        return "h"
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        author = msg.get("author", {})
        if not isinstance(author, dict) or author.get("role") != "user":
            continue
        content = msg.get("content", {})
        if not isinstance(content, dict):
            continue
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, str) and part.strip():
                runes = list(part)
                return "".join(runes[:5])
    return "h"
