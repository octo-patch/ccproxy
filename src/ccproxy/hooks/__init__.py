"""Pipeline hooks with dependency declarations.

Each hook uses the @hook decorator to declare reads/writes dependencies.
The HookDAG uses these to compute execution order via topological sort.
"""

from ccproxy.hooks.cache_breakpoints import cache_breakpoints
from ccproxy.hooks.extract_pplx_files import extract_pplx_files
from ccproxy.hooks.extract_session_id import extract_session_id
from ccproxy.hooks.gemini_cli import gemini_cli
from ccproxy.hooks.inject_auth import inject_auth
from ccproxy.hooks.inject_mcp_notifications import inject_mcp_notifications
from ccproxy.hooks.pplx_preflight import pplx_preflight
from ccproxy.hooks.pplx_stamp_headers import pplx_stamp_headers
from ccproxy.hooks.pplx_thread_inject import pplx_thread_inject

__all__ = [
    "cache_breakpoints",
    "extract_pplx_files",
    "extract_session_id",
    "gemini_cli",
    "inject_auth",
    "inject_mcp_notifications",
    "pplx_preflight",
    "pplx_stamp_headers",
    "pplx_thread_inject",
]
