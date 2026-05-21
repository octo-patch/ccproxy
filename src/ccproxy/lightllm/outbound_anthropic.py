"""Render a :class:`ParsedRequest` back to Anthropic Messages API wire bytes.

The strategy is to delegate the wire assembly to pydantic-ai's
``AnthropicModel._messages_create`` via a *capture* pattern: instantiate
``AnthropicModel`` with a stand-in ``AsyncAnthropic`` whose
``beta.messages.create`` short-circuits by raising :class:`CaptureSentinel`
carrying the kwargs that would have hit the SDK. We then serialize those
kwargs to JSON bytes, stripping the SDK-only sentinels
(``anthropic.omit`` / ``anthropic.NotGiven``) and the SDK control fields
that don't belong on the wire body (``extra_headers``, ``extra_body``,
``timeout``, ``betas``).

Cache-control fidelity is preserved via two channels:

* ``raw_extras['system']`` and ``raw_extras['tools']`` — populated by the
  inbound parser when system/tool ``cache_control`` is non-uniform — are
  copied verbatim onto the rendered body, overriding pydantic-ai's
  settings-driven version.
* All other ``raw_extras`` entries that aren't IR-internal keys (the
  ``cc:*`` / ``unknown_block:*`` markers) are stitched in if they don't
  collide with a key pydantic-ai already produced.

This mirrors :func:`ccproxy.lightllm.anthropic_inbound.parse_anthropic_messages`:
roundtripping ``render_anthropic(await parse_anthropic_messages(body))``
recovers the input body modulo field ordering and ``null``/missing
omission.
"""

from __future__ import annotations

import base64
import io
import json
from typing import TYPE_CHECKING, Any, cast

import anthropic
from anthropic import AsyncAnthropic
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

if TYPE_CHECKING:
    from ccproxy.lightllm.parsed import ParsedRequest


class CaptureSentinel(Exception):  # noqa: N818 - "Sentinel" is the established name.
    """Raised inside the capture client to short-circuit pydantic-ai's request flow.

    Carries the kwargs that ``AnthropicModel`` would have passed to
    ``client.beta.messages.create``. The kwargs include both wire-body
    fields (``messages``, ``system``, ``tools``, etc.) and SDK control
    fields (``extra_headers``, ``betas``, ``timeout``) which the renderer
    filters out before serializing.
    """

    def __init__(self, kwargs: dict[str, Any]) -> None:
        super().__init__("captured")
        self.kwargs = kwargs


# Top-level keys returned by ``messages.create`` that are SDK control
# parameters, not wire-body fields. ``betas`` becomes the ``anthropic-beta``
# HTTP header; the rest live on the SDK request object itself.
_SDK_CONTROL_FIELDS: frozenset[str] = frozenset(
    {
        "extra_headers",
        "extra_query",
        "extra_body",
        "timeout",
        "betas",
    }
)


def _is_omit(value: Any) -> bool:
    """True if ``value`` is one of anthropic-sdk's *not-given* sentinels."""
    return isinstance(value, anthropic.Omit | anthropic.NotGiven)


def _jsonable(value: Any) -> Any:
    """Convert SDK-internal carriers (``BytesIO``) to a JSON-serializable form.

    pydantic-ai's ``_map_binary_data`` wraps image/document bytes in
    ``io.BytesIO`` for the Anthropic SDK to consume. The wire body
    requires the same payload as a base64 string, which is what the SDK
    would produce on its own before sending — we replicate that step.
    """
    if isinstance(value, io.BytesIO):
        return base64.b64encode(value.getvalue()).decode("ascii")
    if isinstance(value, bytes | bytearray):
        return base64.b64encode(bytes(value)).decode("ascii")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


async def _capture_create_kwargs(parsed: ParsedRequest) -> dict[str, Any]:
    """Drive ``AnthropicModel.request`` against a capture client and return the kwargs."""
    fake_client = AsyncAnthropic(api_key="ccproxy-capture-sentinel")

    async def _capture(**kwargs: Any) -> Any:
        raise CaptureSentinel(kwargs)

    # The SDK's ``create`` overload signature can't be satisfied by a generic
    # capture stub — patch via ``setattr`` to bypass static-checker complaints
    # on both branches of the overload union.
    setattr(fake_client.beta.messages, "create", _capture)  # noqa: B010

    provider = AnthropicProvider(anthropic_client=fake_client)
    model = AnthropicModel(parsed.model, provider=provider)
    try:
        await model.request(parsed.messages, parsed.settings, parsed.request_parameters)
    except CaptureSentinel as captured:
        return captured.kwargs
    raise RuntimeError(
        "AnthropicModel.request did not invoke the capture client — "
        "pydantic-ai's request flow may have changed."
    )


def _strip_sentinels(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop SDK control fields and ``Omit`` / ``NotGiven`` placeholders."""
    body: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in _SDK_CONTROL_FIELDS:
            continue
        if value is None or _is_omit(value):
            continue
        body[key] = value
    return body


def _stitch_raw_extras(body: dict[str, Any], parsed: ParsedRequest) -> None:
    """Re-inject ``raw_extras`` entries onto the rendered body.

    * ``raw_extras['system']`` and ``raw_extras['tools']`` override the
      pydantic-ai-rendered versions (these are populated only when the
      inbound parser detected non-uniform ``cache_control`` that the IR's
      settings-level cache markers can't represent).
    * IR-internal markers (keys starting with ``cc:`` or ``unknown_block:``)
      are skipped — they're inbound-only bookkeeping.
    * Any other keys that don't collide with a key already on the body are
      copied verbatim, restoring fields like ``metadata`` that the inbound
      parser stashed for passthrough fidelity.
    """
    overrides = ("system", "tools")
    for key in overrides:
        if key in parsed.raw_extras:
            body[key] = parsed.raw_extras[key]

    for key, value in parsed.raw_extras.items():
        if key in overrides:
            continue
        if key.startswith(("cc:", "unknown_block:")):
            continue
        body.setdefault(key, value)


def _apply_settings_fields(body: dict[str, Any], parsed: ParsedRequest) -> None:
    """Restore body fields the IR carries on ``settings`` but pydantic-ai's outbound drops.

    ``top_k`` is the canonical example: the Anthropic wire accepts it, the
    inbound parser stashes it on ``settings``, but ``AnthropicModel._messages_create``
    omits it from the kwargs it hands to ``client.beta.messages.create``.
    """
    settings = cast("dict[str, Any]", parsed.settings)
    if "top_k" in settings and "top_k" not in body:
        body["top_k"] = settings["top_k"]


def _apply_stream_flag(body: dict[str, Any], parsed: ParsedRequest) -> None:
    """Honour the listener's ``stream`` request.

    pydantic-ai's non-streaming ``request()`` call always sets ``stream=False``
    on the kwargs. If the listener body had ``stream=true``, restore it.
    """
    if parsed.stream:
        body["stream"] = True


async def render_anthropic(parsed: ParsedRequest) -> bytes:
    """Render a :class:`ParsedRequest` to Anthropic Messages wire bytes.

    Returns the JSON-encoded request body — what the upstream
    ``POST /v1/messages`` endpoint expects. Headers and SDK-only fields
    are stripped; ``raw_extras`` overrides for ``system`` / ``tools`` and
    other top-level wire fields are re-applied; settings fields the
    pydantic-ai outbound drops (e.g. ``top_k``) are restored.
    """
    kwargs = await _capture_create_kwargs(parsed)
    body = _strip_sentinels(kwargs)
    _apply_settings_fields(body, parsed)
    _stitch_raw_extras(body, parsed)
    _apply_stream_flag(body, parsed)
    return json.dumps(body, separators=(",", ":"), default=_jsonable).encode()
