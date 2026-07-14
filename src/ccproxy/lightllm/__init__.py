"""lightllm — ccproxy's wire layer.

Pydantic-ai-mediated wire translation between client listener formats
and upstream provider formats. The per-provider FSMs live in
:mod:`ccproxy.lightllm.graph`; the dispatchers re-exported here are the
public entry points for the rest of ccproxy.
"""

from ccproxy.lightllm.adapters import LLMRenderInput
from ccproxy.lightllm.cache_policy import (
    CacheBudgetReport,
    CachePolicy,
    apply_cache_policy,
)
from ccproxy.lightllm.graph import (
    UnsupportedUpstreamError,
    dispatch_dump,
    dispatch_dump_sync,
    dispatch_intake,
    dispatch_render,
)
from ccproxy.lightllm.parsed import InboundFormat
from ccproxy.lightllm.pplx import (
    LightLLMError,
    PerplexityError,
)

__all__ = [
    "CacheBudgetReport",
    "CachePolicy",
    "InboundFormat",
    "LLMRenderInput",
    "LightLLMError",
    "PerplexityError",
    "UnsupportedUpstreamError",
    "apply_cache_policy",
    "dispatch_dump",
    "dispatch_dump_sync",
    "dispatch_intake",
    "dispatch_render",
]
