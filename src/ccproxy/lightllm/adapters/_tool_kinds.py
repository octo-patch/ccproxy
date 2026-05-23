"""Wire-format ``tool.type`` → :data:`ToolPartKind` mapping for typed promotion.

The intake FSMs feed each emitted :class:`ToolCallPart` through
:meth:`ModelResponsePartsManager._typed_call_part`, which promotes a base
``ToolCallPart`` to its typed subclass (e.g.
:class:`pydantic_ai.messages.ToolSearchCallPart`) when the matching
:class:`ToolDefinition` carries a ``tool_kind`` discriminator. The
listener-side ``_parse_tools`` functions in
:mod:`ccproxy.lightllm.adapters._anthropic_envelope` and
:mod:`ccproxy.lightllm.adapters._openai_envelope` consult these dicts to
populate ``tool_kind`` from the incoming wire-format ``type`` field.

Tools whose wire ``type`` is not in this map (e.g. user-defined Anthropic
``{"name": ..., "input_schema": ...}`` tools or OpenAI
``{"type": "function", ...}`` tools) get ``tool_kind=None`` — the typed
promotion path is a no-op for them.

Add new entries as ``pydantic_ai.messages.ToolPartKind`` gains values.
The current registered set is documented in
``pydantic_ai/messages.py`` under the ``ToolPartKind`` ``Literal`` alias.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.messages import ToolPartKind


# Anthropic server-side tools — wire ``type`` discriminator → ``ToolPartKind``.
# Versioned ``type`` strings (e.g. ``web_search_20250305``) are stable per
# Anthropic's release notes; add new dated variants here as they ship.
ANTHROPIC_TYPED_TOOLS: dict[str, ToolPartKind] = {
    "web_search_20250305": "tool-search",
}


# OpenAI typed tool wire shapes — ``type`` discriminator → ``ToolPartKind``.
# OpenAI Chat Completions tools are almost always ``{"type": "function", ...}``
# (user-defined); built-in server-side tools like ``web_search`` live in the
# Responses API and are not currently routed through ccproxy's Chat Completions
# listener. The dict is intentionally empty — extend when adding Responses API
# support or other typed OpenAI tools.
OPENAI_TYPED_TOOLS: dict[str, ToolPartKind] = {}
