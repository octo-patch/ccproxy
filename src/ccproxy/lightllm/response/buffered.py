"""Non-streaming response transforms: upstream JSON body → listener JSON body.

For flows where the client requested ``stream=false`` (or upstream
downgraded a streaming request to buffered), the inspector reads the
full response body once and calls these entry points to transform it.

The same intake + render abstractions used in :mod:`ccproxy.lightllm.response.pipeline`
are reused: ``feed_all → close`` produces all IR events from the buffered
body, then the render emits the listener-format response bytes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ccproxy.lightllm.response.intake import select_intake
from ccproxy.lightllm.response.render import select_render

if TYPE_CHECKING:
    from ccproxy.lightllm.parsed import ListenerFormat
    from pydantic_ai.models import ModelRequestParameters


def transform_buffered_response(
    *,
    upstream_provider: str,
    model: str,
    listener_format: ListenerFormat,
    request_params: ModelRequestParameters,
    upstream_body: bytes,
) -> bytes:
    """Transform a buffered upstream response body to listener-format bytes.

    Wraps the upstream body in synthetic SSE framing so the same sync
    intake/render abstractions used for streaming flows handle the
    one-shot buffered case. The intake emits all IR events at once;
    the render flushes them all then emits the listener terminator.
    """
    intake = select_intake(
        upstream_provider=upstream_provider,
        model=model,
        request_params=request_params,
    )
    render = select_render(listener_format)

    framed = _wrap_as_sse(upstream_body)
    out = bytearray()
    for event in intake.feed(framed):
        out.extend(render.render(event))
    for event in intake.close():
        out.extend(render.render(event))
    out.extend(render.close())
    return bytes(out)


def _wrap_as_sse(body: bytes) -> bytes:
    """Wrap a buffered JSON body as a single synthetic SSE frame.

    The vendor intakes are SSE-parsers; for the buffered case we wrap
    the response body in ``data: {body}\\n\\n`` so the same parser drains
    a single event. Sufficient for OpenAI (single ``ChatCompletion``
    JSON) and Google (single ``GenerateContentResponse``). Anthropic's
    buffered response is a ``BetaMessage`` JSON — different shape from
    ``BetaRawMessageStreamEvent`` — and should use pydantic-ai's
    ``_process_response`` instead; that path is out of scope for the
    first response-side cut.
    """
    return b"data: " + body.strip() + b"\n\n"
