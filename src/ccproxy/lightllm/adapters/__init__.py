"""ccproxy/lightllm UIAdapter subclasses.

One adapter per listener wire format. Each subclass extends pydantic-ai's
:class:`pydantic_ai.ui.UIAdapter` and provides classmethod ``load_messages``
and ``dump_messages`` (plus ``dump_system`` for Anthropic) for wire ↔ IR
translation without instantiating the agent machinery.

Replaces the FSM-based ``ccproxy.lightllm.graph.*_load`` / ``*_dump``
modules with procedural code that uses ``MessagesBuilder`` and SDK
TypedDicts directly. The streaming intake / render FSMs in
:mod:`ccproxy.lightllm.graph` are unaffected — only the request-body
load/dump path moves here.
"""

from __future__ import annotations

from ccproxy.lightllm.adapters.anthropic import AnthropicAdapter
from ccproxy.lightllm.adapters.openai_chat import OpenAIChatAdapter

__all__ = [
    "AnthropicAdapter",
    "OpenAIChatAdapter",
]
