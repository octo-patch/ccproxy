"""Commitbee compatibility hook — strips markdown fencing instruction.

Detects commitbee requests by their system prompt signature and appends
an instruction to emit raw JSON without markdown code block wrapping.
Runs after the shape hook so the system prompt is already assembled.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from ccproxy.pipeline.hook import HookParams, hook

if TYPE_CHECKING:
    from ccproxy.pipeline.context import Context

logger = logging.getLogger(__name__)

_COMMITBEE_SIGNATURE = "You generate Conventional Commit messages from git diffs"
_RAW_JSON_INSTRUCTION = (
    "\n\nCRITICAL FORMATTING RULE: You MUST output ONLY the raw JSON object. "
    "Do NOT use ```json code fences. Do NOT use any markdown formatting. "
    "Your entire response must be parseable by JSON.parse() with zero preprocessing."
)


def _text_block(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return cast("dict[str, object]", value)


def _block_text(value: object) -> str:
    block = _text_block(value)
    if block is None:
        return ""
    text = block.get("text")
    return text if isinstance(text, str) else ""


def commitbee_compat_guard(ctx: Context) -> bool:
    """Only run for requests whose system prompt contains the commitbee signature.

    Routes like Anthropic's ``/api/v2/logs`` post a list-shaped body — short-
    circuit those before ``.get()`` raises.
    """
    raw_body = cast(object, ctx._body)
    if not isinstance(raw_body, dict):
        return False
    body = cast("dict[str, object]", raw_body)
    system = body.get("system")
    if isinstance(system, str):
        return _COMMITBEE_SIGNATURE in system
    if isinstance(system, list):
        return any(_COMMITBEE_SIGNATURE in _block_text(block) for block in system)
    return False


@hook(reads=["system"], writes=["system"])
def commitbee_compat(ctx: Context, _: HookParams) -> Context:
    """Append raw-JSON instruction to commitbee's system prompt."""
    raw_body = cast(object, ctx._body)
    if not isinstance(raw_body, dict):
        return ctx
    body = cast("dict[str, object]", raw_body)
    system = body.get("system")
    if isinstance(system, str):
        body["system"] = system + _RAW_JSON_INSTRUCTION
    elif isinstance(system, list):
        for block in reversed(system):
            text_block = _text_block(block)
            if text_block is None:
                continue
            text = text_block.get("text")
            if isinstance(text, str) and _COMMITBEE_SIGNATURE in text:
                text_block["text"] = text + _RAW_JSON_INSTRUCTION
                break
    logger.info("commitbee_compat: appended raw-JSON instruction")
    return ctx
