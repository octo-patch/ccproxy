"""Tests for ccproxy.lightllm.registry — ccproxy-internal provider resolution."""

from __future__ import annotations

import pytest

from ccproxy.lightllm.pplx import PerplexityProConfig
from ccproxy.lightllm.registry import get_config


class TestGetConfig:
    def test_perplexity_pro(self) -> None:
        config = get_config("perplexity_pro", "perplexity/best")
        assert isinstance(config, PerplexityProConfig)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            get_config("nonexistent_provider_xyz", "some-model")
