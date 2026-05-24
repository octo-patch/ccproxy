"""ccproxy/lightllm UIAdapter subclasses + render-input Protocol.

One adapter per listener wire format. Each subclass extends pydantic-ai's
:class:`pydantic_ai.ui.UIAdapter` and provides classmethod ``load_messages``
and ``dump_messages`` (plus ``dump_system`` for Anthropic) for wire ↔ IR
translation without instantiating the agent machinery. Google and
Perplexity are outbound-only — their :meth:`load_messages` raises
:class:`NotImplementedError`.

:class:`LLMRenderInput` is the Protocol the dispatchers and adapters
consume: any object exposing ``messages``, ``settings``, ``raw_extras``,
``function_tools``, ``model``, and ``stream`` properties satisfies it.
:class:`ccproxy.pipeline.context.Context` is the production
implementation; tests build minimal namespaces or dataclasses.

The streaming intake / render FSMs in :mod:`ccproxy.lightllm.graph` are
unaffected — only the request-body load/dump path lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ccproxy.lightllm.adapters.anthropic import AnthropicAdapter
from ccproxy.lightllm.adapters.google import GoogleAdapter
from ccproxy.lightllm.adapters.openai_chat import OpenAIChatAdapter
from ccproxy.lightllm.adapters.openai_responses import OpenAIResponsesAdapter
from ccproxy.lightllm.adapters.perplexity import PerplexityAdapter

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.settings import ModelSettings


@runtime_checkable
class LLMRenderInput(Protocol):
    """Protocol consumed by adapters and dispatchers when rendering to wire bytes.

    Any object exposing the six properties below satisfies the protocol.
    :class:`ccproxy.pipeline.context.Context` is the production
    implementation; tests build small namespaces.
    """

    @property
    def model(self) -> str: ...

    @property
    def messages(self) -> list[ModelMessage]: ...

    @property
    def request_parameters(self) -> ModelRequestParameters: ...

    @property
    def settings(self) -> ModelSettings: ...

    @property
    def stream(self) -> bool: ...

    @property
    def raw_extras(self) -> dict[str, Any]: ...


__all__ = [
    "AnthropicAdapter",
    "GoogleAdapter",
    "LLMRenderInput",
    "OpenAIChatAdapter",
    "OpenAIResponsesAdapter",
    "PerplexityAdapter",
]
