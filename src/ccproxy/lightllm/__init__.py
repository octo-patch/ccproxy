"""lightllm — ccproxy's wire layer.

Pydantic-ai-mediated wire translation between client listener formats
and upstream provider formats. The per-provider FSMs live in
:mod:`ccproxy.lightllm.graph`; the dispatchers re-exported here are the
public entry points for the rest of ccproxy.
"""

from ccproxy.lightllm.graph import (
    UnsupportedUpstreamError,
    dispatch_dump,
    dispatch_dump_sync,
    dispatch_intake,
    dispatch_load,
    dispatch_render,
)
from ccproxy.lightllm.parsed import ListenerFormat, ParsedRequest
from ccproxy.lightllm.pplx import (
    LightllmException,
    PerplexityException,
    PerplexityProConfig,
)

__all__ = [
    "LightllmException",
    "ListenerFormat",
    "ParsedRequest",
    "PerplexityException",
    "PerplexityProConfig",
    "UnsupportedUpstreamError",
    "dispatch_dump",
    "dispatch_dump_sync",
    "dispatch_intake",
    "dispatch_load",
    "dispatch_render",
]
