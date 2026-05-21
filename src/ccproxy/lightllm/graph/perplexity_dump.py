"""Render :class:`ParsedRequest` to Perplexity Pro wire bytes.

Perplexity Pro has no pydantic-ai counterpart — its wire shape is not
chat-completions-shaped, it's a Perplexity-specific
``{params: {...28 fields...}, query_str: "..."}`` payload posted to
``POST https://www.perplexity.ai/rest/sse/perplexity_ask``. This module
adapts the existing ``_build_pplx_payload`` machinery in :mod:`pplx`
to consume the pydantic-ai IR instead of OpenAI-format dicts.

Conversion strategy (Option A): walk the IR messages, project each one
back to its OpenAI-format dict equivalent (the inverse of
``openai_inbound.parse_openai_chat``), then hand the result to the
existing ``_flatten_messages`` / ``_flatten_last_user_turn`` /
``_build_pplx_payload`` helpers. The Perplexity-specific
``params`` block (sources, search focus, attachments, thread
continuation) is sourced from ``parsed.raw_extras["pplx"]`` — the same
top-level wire field that the inbound hooks (``extract_pplx_files``,
``pplx_thread_inject``) write to.

Why Option A: the existing ``_build_pplx_payload`` is the source of
truth for the 28-field Perplexity production payload. Re-implementing
it against IR walks would invite drift; the conversion to OpenAI-format
dicts is lossless for the fields Perplexity actually consumes
(``role`` + ``content`` text — images are already stripped to S3
attachments upstream of the IR by the ``extract_pplx_files`` hook).

The output is JSON-encoded bytes ready for the outbound wire.
"""

from __future__ import annotations

import json
from typing import Any, cast

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)

from ccproxy.lightllm.parsed import ParsedRequest
from ccproxy.lightllm.pplx import (
    _build_pplx_payload,
    _flatten_last_user_turn,
    _flatten_messages,
)


async def render_perplexity_pro_dump(parsed: ParsedRequest) -> bytes:
    """Render IR back to Perplexity Pro wire bytes.

    Walks ``parsed.messages`` into OpenAI-format chat messages, then
    invokes the existing ``_build_pplx_payload`` helper with the
    appropriate query string (flattened full history for first turn,
    last user turn only for followup). The Perplexity ``pplx`` block
    (attachments, last_backend_uuid, read_write_token, etc.) is read
    from ``parsed.raw_extras["pplx"]``.
    """
    messages_openai = _ir_to_openai_messages(messages=parsed.messages)
    extras = _resolve_pplx_extras(raw_extras=parsed.raw_extras)
    is_followup = bool(
        extras.get("last_backend_uuid") or extras.get("thread_uuid")
    )
    query = (
        _flatten_last_user_turn(messages_openai)
        if is_followup
        else _flatten_messages(messages_openai)
    )
    payload = _build_pplx_payload(
        query=query,
        model_id=parsed.model,
        extras=extras,
    )
    return json.dumps(payload).encode()


def _resolve_pplx_extras(*, raw_extras: dict[str, Any]) -> dict[str, Any]:
    """Pull the Perplexity-specific extras block out of ``raw_extras``.

    The OpenAI inbound parser stashes the top-level ``pplx`` wire field
    in ``raw_extras["pplx"]`` (it's not in
    :data:`openai_inbound._ABSORBED_BODY_KEYS`). Returns an empty dict
    when the field is absent or not a dict.
    """
    raw = raw_extras.get("pplx")
    if isinstance(raw, dict):
        return cast(dict[str, Any], raw)
    return {}


def _ir_to_openai_messages(
    *, messages: list[ModelMessage]
) -> list[dict[str, Any]]:
    """Project IR messages back to OpenAI-format chat dicts.

    This is the inverse of the relevant subset of
    :func:`ccproxy.lightllm.openai_inbound.parse_openai_chat` — the
    Perplexity payload only reads ``role`` + ``content`` text via the
    flatten helpers, so we collapse multimodal parts to their text
    fragments and drop tool-call metadata. Image content (if any
    survives this far) is preserved as ``image_url`` blocks so the
    flatten helpers can drop them per the existing behavior.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            result.extend(_request_to_openai(msg=msg))
        elif isinstance(msg, ModelResponse):
            result.append(_response_to_openai(msg=msg))
    return result


def _request_to_openai(*, msg: ModelRequest) -> list[dict[str, Any]]:
    """Split a ``ModelRequest`` into one or more OpenAI-format dicts.

    A single ``ModelRequest`` may carry a mix of ``SystemPromptPart``,
    ``UserPromptPart``, and ``ToolReturnPart`` (the latter we omit —
    Perplexity has no tool-result message concept and the flatten
    helpers ignore unknown roles).
    """
    out: list[dict[str, Any]] = []
    for part in msg.parts:
        if isinstance(part, SystemPromptPart):
            out.append({"role": "system", "content": part.content})
        elif isinstance(part, UserPromptPart):
            out.append(
                {"role": "user", "content": _user_content_to_openai(content=part.content)}
            )
        elif isinstance(part, ToolReturnPart):
            out.append(
                {
                    "role": "tool",
                    "content": _coerce_tool_content(content=part.content),
                    "tool_call_id": part.tool_call_id,
                }
            )
    return out


def _response_to_openai(*, msg: ModelResponse) -> dict[str, Any]:
    """Project a ``ModelResponse`` into an assistant-role OpenAI dict.

    Tool calls are dropped — Perplexity flattens everything to text and
    the existing ``_flatten_messages`` helper only reads ``content``.
    Thinking parts are also dropped (Perplexity reasoning is server-side).
    """
    text_chunks: list[str] = []
    for part in msg.parts:
        if isinstance(part, TextPart):
            text_chunks.append(part.content)
    content = "".join(text_chunks)
    return {"role": "assistant", "content": content}


def _user_content_to_openai(
    *, content: Any,
) -> str | list[dict[str, Any]]:
    """Convert ``UserPromptPart.content`` back to the OpenAI wire shape.

    Plain strings pass through unchanged. Sequences become a list of
    ``{type: "text", text: ...}`` blocks for textual fragments. Any
    non-text content (images, audio, etc.) is emitted as the smallest
    OpenAI-compatible placeholder block so the flatten helpers' existing
    filter (which drops non-text parts) keeps working.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list | tuple):
        return str(content)

    blocks: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
            continue
        if isinstance(item, TextContent):
            blocks.append({"type": "text", "text": item.content})
            continue
        # Non-text user content (BinaryContent, ImageUrl, AudioUrl, etc.)
        # — emit a non-text block so the flatten helpers drop it. The
        # extract_pplx_files hook should have moved these to S3
        # attachments upstream; anything reaching here is residual.
        blocks.append({"type": "image_url", "image_url": {"url": ""}})
    return blocks


def _coerce_tool_content(*, content: Any) -> str:
    """Stringify a tool-return content payload for the OpenAI wire."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content)
