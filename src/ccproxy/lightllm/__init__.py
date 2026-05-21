"""lightllm — ccproxy's wire layer.

Historically a connector into LiteLLM's BaseConfig. Mid-refactor (see
``plans/reshape-wire-py-as-lexical-graham.md``): this package is the home
of the pydantic-ai-mediated wire translation layer that replaces the
LiteLLM-based one. The module name is preserved across the cut.
"""

from ccproxy.lightllm.dispatch import (
    MitmResponseShim,
    SseTransformer,
    make_sse_transformer,
    transform_to_openai,
    transform_to_provider,
)
from ccproxy.lightllm.parsed import ListenerFormat, ParsedRequest
from ccproxy.lightllm.registry import get_config

__all__ = [
    "ListenerFormat",
    "MitmResponseShim",
    "ParsedRequest",
    "SseTransformer",
    "get_config",
    "make_sse_transformer",
    "transform_to_openai",
    "transform_to_provider",
]
