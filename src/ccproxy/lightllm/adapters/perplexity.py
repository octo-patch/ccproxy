"""Perplexity Pro UIAdapter (outbound-only).

Perplexity Pro has no pydantic-ai counterpart — its wire shape is not
chat-completions-shaped, it's a Perplexity-specific
``{params: {...28 fields...}, query_str: "..."}`` payload posted to
``POST https://www.perplexity.ai/rest/sse/perplexity_ask``. This module
renders an :class:`~ccproxy.lightllm.adapters.LLMRenderInput` (typically a
:class:`~ccproxy.pipeline.context.Context`) to Perplexity wire bytes by
projecting IR messages back to OpenAI-format dicts, then invoking the
existing ``_build_pplx_payload`` helper from :mod:`ccproxy.lightllm.pplx`.

Conversion strategy: walk the IR messages, project each one back to its
OpenAI-format dict equivalent (the inverse of OpenAI load), then hand
the result to the existing ``_flatten_messages`` / ``_flatten_last_user_turn``
/ ``_build_pplx_payload`` helpers. The Perplexity-specific ``params``
block (sources, search focus, attachments, thread continuation) is
sourced from ``req.raw_extras["pplx"]`` — the same top-level wire
field that the inbound hooks (``extract_pplx_files``,
``pplx_thread_inject``) write to.

Why this approach: the existing ``_build_pplx_payload`` is the source of
truth for the 28-field Perplexity production payload. Re-implementing it
against IR walks would invite drift; the conversion to OpenAI-format
dicts is lossless for the fields Perplexity actually consumes
(``role`` + ``content`` text — images are already stripped to S3
attachments upstream of the IR by the ``extract_pplx_files`` hook).

This is an OUTBOUND-ONLY adapter — :meth:`load_messages` raises
:class:`NotImplementedError`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.ui import UIAdapter, UIEventStream

from ccproxy.lightllm.pplx import (
    _build_pplx_payload,
    _flatten_last_user_turn,
    _flatten_messages,
)


@dataclass
class PerplexityAdapter(UIAdapter[Any, dict[str, Any], Any, AgentDepsT, OutputDataT]):
    """Outbound-only UIAdapter for Perplexity Pro ``perplexity_ask``.

    :meth:`load_messages` raises :class:`NotImplementedError` because ccproxy
    does not host a Perplexity-format listener. :meth:`render` projects IR
    messages back to OpenAI-format dicts and invokes the existing
    ``_build_pplx_payload`` helper from :mod:`ccproxy.lightllm.pplx` to
    produce the 28-field Perplexity wire body.

    :meth:`build_event_stream` raises :class:`NotImplementedError`;
    streaming intake lives in :mod:`ccproxy.lightllm.graph.perplexity_intake`.
    """

    @classmethod
    def load_messages(cls, *_args: Any, **_kwargs: Any) -> list[ModelMessage]:
        raise NotImplementedError(
            "ccproxy does not host a Perplexity-format listener; "
            "PerplexityAdapter is outbound-only."
        )

    def build_event_stream(
        self,
    ) -> UIEventStream[Any, Any, AgentDepsT, OutputDataT]:
        raise NotImplementedError(
            "Perplexity streaming intake lives in ccproxy.lightllm.graph.perplexity_intake."
        )

    @classmethod
    def render(cls, req: Any) -> bytes:
        """Render an :class:`LLMRenderInput` to Perplexity Pro wire bytes.

        Walks ``req.messages`` into OpenAI-format chat messages, then
        invokes the existing ``_build_pplx_payload`` helper with the
        appropriate query string (flattened full history for first turn,
        last user turn only for followup). The Perplexity ``pplx`` block
        (attachments, last_backend_uuid, read_write_token, etc.) is read
        from ``req.raw_extras["pplx"]``.

        Args:
            req: The :class:`LLMRenderInput` (typically a Context) to render.

        Returns:
            JSON-encoded Perplexity wire payload as bytes.

        Raises:
            ValueError: If the model is not in the Perplexity catalog.
        """
        messages_openai = _ir_to_openai_messages(messages=req.messages)
        extras = _resolve_pplx_extras(raw_extras=req.raw_extras)
        is_followup = bool(extras.get("last_backend_uuid") or extras.get("thread_uuid"))
        query = _flatten_last_user_turn(messages_openai) if is_followup else _flatten_messages(messages_openai)
        payload = _build_pplx_payload(
            query=query,
            model_id=req.model,
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


def _ir_to_openai_messages(*, messages: list[ModelMessage]) -> list[dict[str, Any]]:
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
    """Split a ``ModelRequest`` into one or more OpenAI-format dicts."""
    out: list[dict[str, Any]] = []
    for part in msg.parts:
        if isinstance(part, SystemPromptPart):
            out.append({"role": "system", "content": part.content})
        elif isinstance(part, UserPromptPart):
            out.append(
                {
                    "role": "user",
                    "content": _user_content_to_openai(content=part.content),
                }
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


def _user_content_to_openai(*, content: Any) -> str | list[dict[str, Any]]:
    """Convert ``UserPromptPart.content`` back to the OpenAI wire shape."""
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
