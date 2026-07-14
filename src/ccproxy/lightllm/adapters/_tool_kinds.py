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

**Scope constraint** — pydantic-ai's :data:`ToolPartKind` is currently
``Literal['tool-search']``. The only registered narrowers (in
``pydantic_ai._tool_search``) are ``_TOOL_CALL_NARROWERS['tool-search']``
and ``_NATIVE_CALL_NARROWERS['tool-search']``. Mapping a non-search wire
``type`` to ``'tool-search'`` would mis-promote it; mapping to any other
string is a no-op (the narrower lookup returns ``None``). So today only
search-flavored server-side tools should appear in this map. When
pydantic-ai adds new kinds (e.g. ``'tool-browse'``, ``'tool-code'``),
extend with the corresponding wire types here.

**Promotion, not preservation** — this map only drives response-part
promotion; wire fidelity for typed tools is the ``raw_extras['tools']``
verbatim override in ``_parse_tools``. Note the bm25/regex tool-search
entries are mostly redundant for *native* traffic: native tool-search calls
arrive as ``server_tool_use`` blocks and are already typed by
``_map_server_tool_use_block``. They matter for the local/client-flavored
path, where a plain ``tool_use`` block needs name-keyed promotion — do not
conclude the entries are dead.

Currently shipped Anthropic dated tool variants per ``anthropic/types/``:

- ``web_search_20250305`` (mapped)
- ``web_search_20260209`` (mapped)
- ``tool_search_tool_bm25_20251119`` / ``tool_search_tool_regex_20251119`` (mapped)
- ``web_fetch_20250910`` / ``web_fetch_20260209`` / ``web_fetch_20260309`` — fetch, not search
- ``bash_20241022`` / ``bash_20250124`` — bash, no ToolPartKind yet
- ``code_execution_20250522`` / ``code_execution_20250825`` / ``code_execution_20260120`` — code, no ToolPartKind yet
- ``computer_20241022`` / ``computer_20250124`` / ``computer_20251124`` — computer-use, no ToolPartKind yet
- ``text_editor_20241022`` / ``text_editor_20250124`` / ``text_editor_20250429`` /
  ``text_editor_20250728`` — file editor, no ToolPartKind yet

Add new ``web_search_*`` dated variants as Anthropic ships them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.messages import ToolPartKind


# Anthropic server-side tools — wire ``type`` discriminator → ``ToolPartKind``.
# Only ``web_search_*`` variants map today; the other Anthropic server-side
# tool families (bash, code_execution, computer, text_editor, web_fetch) don't
# have ``ToolPartKind`` equivalents in pydantic-ai yet.
ANTHROPIC_TYPED_TOOLS: dict[str, ToolPartKind] = {
    "web_search_20250305": "tool-search",
    "web_search_20260209": "tool-search",
    "tool_search_tool_bm25_20251119": "tool-search",
    "tool_search_tool_regex_20251119": "tool-search",
}


# OpenAI typed tool wire shapes — ``type`` discriminator → ``ToolPartKind``.
# OpenAI Chat Completions tools are typed ``Literal["function"]`` only
# (verified against ``openai/types/chat/chat_completion_function_tool.py``);
# all server-side tools (``web_search_preview``, ``file_search``,
# ``code_interpreter``) live in the Responses API. ccproxy's listener
# currently routes ``/v1/chat/completions`` only, so this dict stays
# intentionally empty. Populate when ccproxy adds a Responses API listener.
OPENAI_TYPED_TOOLS: dict[str, ToolPartKind] = {}
