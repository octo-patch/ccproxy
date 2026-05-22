"""Provider name → ccproxy-internal config resolution.

Only ccproxy-internal providers are registered here (currently just
Perplexity Pro). Standard providers route through the FSM dispatchers
in :mod:`ccproxy.lightllm.graph`.
"""

from __future__ import annotations

from collections.abc import Callable

from ccproxy.lightllm.pplx import PERPLEXITY_PROVIDER_NAME, PerplexityProConfig

_LOCAL_CONFIGS: dict[str, Callable[[], PerplexityProConfig]] = {
    PERPLEXITY_PROVIDER_NAME: PerplexityProConfig,
}
"""ccproxy-internal providers. Each entry is a zero-arg factory."""


def get_config(provider: str, model: str) -> PerplexityProConfig:
    """Resolve a ccproxy-internal provider name to its config instance."""
    del model  # accepted for call-site compatibility; unused
    factory = _LOCAL_CONFIGS.get(provider)
    if factory is None:
        valid = list(_LOCAL_CONFIGS)
        raise ValueError(f"Unknown provider {provider!r}. Valid providers: {valid}")
    return factory()
