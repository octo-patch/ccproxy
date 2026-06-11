"""Shared test fixtures and helpers."""

import pytest

from ccproxy.config import clear_config_instance
from ccproxy.flows.store import clear_flow_store
from ccproxy.hooks.gemini_cli import reset_cache as reset_gemini_project_cache
from ccproxy.lightllm.pplx_threads import clear_pplx_threads
from ccproxy.mcp.buffer import clear_buffer
from ccproxy.shaping.executor import clear_shape_hook_cache
from ccproxy.shaping.store import clear_store_instance
from ccproxy.transport.dispatch import reset_cache as reset_transport_cache


@pytest.fixture(autouse=True)
def cleanup():
    """Ensure clean state between tests."""
    yield
    clear_config_instance()
    clear_buffer()
    clear_flow_store()
    clear_store_instance()
    clear_shape_hook_cache()
    clear_pplx_threads()
    reset_transport_cache()
    reset_gemini_project_cache()
