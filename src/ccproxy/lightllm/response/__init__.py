"""Response-side wire layer.

Per-vendor sync intakes parse upstream SSE bytes into pydantic-ai
``ModelResponseStreamEvent`` IR. Per-listener-format sync renderers
emit listener wire bytes from IR events. ``SSEPipeline`` ties them
together behind a ``flow.response.stream`` callable.
"""

from __future__ import annotations
