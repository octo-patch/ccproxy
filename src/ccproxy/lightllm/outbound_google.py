"""Outbound renderer: pydantic-ai IR → Google Gemini `generateContent` wire bytes.

Drives pydantic-ai's ``GoogleModel`` against a capture-only ``Provider`` whose
``client.aio.models.generate_content`` raises :class:`CaptureSentinel` after
recording the kwargs that pydantic-ai assembled. We then transform those
kwargs into the Google API JSON wire body (camelCase keys, base64-encoded
inline data, config fields hoisted to top level under ``generationConfig``)
and return the serialized bytes.

This is the OUTBOUND-only half of the wire layer for Gemini; ccproxy doesn't
accept Gemini-format inbound requests, so there is no matching inbound
parser in this module.

The kwargs captured at ``generate_content`` are ``model``, ``contents``,
``config`` — straight from ``GoogleModel._generate_content`` (see
``pydantic_ai/models/google.py:783``). The wire shaping below mirrors
``_GenerateContentParameters_to_mldev`` + ``_GenerateContentConfig_to_mldev``
in ``google.genai.models``: contents stay at the top level; the config dict
is split so that ``system_instruction``, ``tools``, ``tool_config``,
``safety_settings``, ``cached_content`` hoist to the top level, while the
remaining sampling/generation parameters live under ``generationConfig``.
"""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from typing import Any, cast

from pydantic.alias_generators import to_camel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.providers import Provider

from ccproxy.lightllm.parsed import ParsedRequest


class CaptureSentinel(Exception):  # noqa: N818 - "Sentinel" is the established name.
    """Raised by the fake Google client to short-circuit pydantic-ai's request flow."""

    def __init__(self, kwargs: dict[str, Any]) -> None:
        super().__init__("captured")
        self.kwargs = kwargs


class _CaptureGoogleModels:
    """Stand-in for ``client.aio.models``. ``generate_content`` records kwargs and raises."""

    async def generate_content(self, **kwargs: Any) -> Any:
        raise CaptureSentinel(kwargs)

    async def generate_content_stream(self, **kwargs: Any) -> Any:
        raise CaptureSentinel(kwargs)


class _CaptureGoogleAio:
    """Stand-in for ``client.aio``. Exposes a ``models`` namespace."""

    def __init__(self) -> None:
        self.models = _CaptureGoogleModels()


class _CaptureGoogleClient:
    """Fake ``google.genai.Client`` used only by ``GoogleModel`` for kwargs capture."""

    def __init__(self) -> None:
        self.aio = _CaptureGoogleAio()


class _CaptureGoogleProvider(Provider[Any]):
    """Provider stand-in that exposes a capture client with no network access."""

    def __init__(self) -> None:
        self._client = _CaptureGoogleClient()

    @property
    def name(self) -> str:
        return "google"

    @property
    def base_url(self) -> str:
        return "https://generativelanguage.googleapis.com"

    @property
    def client(self) -> Any:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> Any:
        return google_model_profile(model_name)


# Config keys hoisted to the top level of the wire body (camelCased).
_HOISTED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "system_instruction",
        "tools",
        "tool_config",
        "safety_settings",
        "cached_content",
    }
)

# Config keys we ignore entirely — they're transport- or SDK-internal,
# never appear on the upstream wire body.
_IGNORED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "http_options",
        "should_return_http_response",
    }
)

# Snake-case keys whose VALUE is user payload data — we still camelCase
# the key itself, but the value passes through verbatim. Otherwise we'd
# corrupt user-defined JSON Schema property names, tool arg structures,
# and tool response payloads.
_PASSTHROUGH_VALUE_KEYS: frozenset[str] = frozenset(
    {
        "args",
        "response",
        "parameters_json_schema",
        "response_json_schema",
        "response_schema",
        "vendor_metadata",
    }
)


async def render_google(parsed: ParsedRequest) -> bytes:
    """Render :class:`ParsedRequest` to Google Gemini ``generateContent`` wire bytes."""
    provider = _CaptureGoogleProvider()
    # ``GoogleModel`` calls ``check_allow_model_requests`` first; pydantic-ai's
    # default ``ALLOW_MODEL_REQUESTS = True`` is the path we want, so no override
    # is needed. ``request_parameters`` is consumed by ``prepare_request`` and
    # ``_build_content_and_config`` to derive the wire body.
    model = GoogleModel(parsed.model, provider=provider)

    settings_dict: dict[str, Any] = {**parsed.settings}
    request_parameters = parsed.request_parameters
    # ``GoogleModel.prepare_request`` mutates ``request_parameters.output_mode``
    # in some scenarios — pass a clone so a re-run of ``render_google`` on the
    # same ``ParsedRequest`` is idempotent.
    cloned_request_parameters = replace(request_parameters)

    kwargs: dict[str, Any] | None = None
    try:
        await model.request(
            parsed.messages,
            cast(Any, settings_dict),
            cloned_request_parameters,
        )
    except CaptureSentinel as exc:
        kwargs = exc.kwargs
    if kwargs is None:
        raise RuntimeError("GoogleModel.request did not hit the capture client")

    body = _kwargs_to_wire_body(kwargs)
    return json.dumps(body, separators=(",", ":")).encode()


def _kwargs_to_wire_body(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate captured ``generate_content`` kwargs into the Google API wire body."""
    body: dict[str, Any] = {}

    contents = kwargs.get("contents")
    if contents is not None:
        body["contents"] = [_camelize(c) for c in contents]

    config = kwargs.get("config") or {}
    if not isinstance(config, dict):
        # Mirror google-genai's behavior: dump pydantic model into a dict.
        config = dict(config)

    generation_config: dict[str, Any] = {}
    for key, value in config.items():
        if value is None or key in _IGNORED_CONFIG_KEYS:
            continue
        if key in _HOISTED_CONFIG_KEYS:
            body[to_camel(key)] = _camelize(value)
        else:
            generation_config[to_camel(key)] = _camelize(value)

    if generation_config:
        body["generationConfig"] = generation_config

    return body


def _camelize(value: Any) -> Any:
    """Recursively convert dict keys to camelCase and encode ``bytes`` as base64.

    Keys listed in :data:`_PASSTHROUGH_VALUE_KEYS` are still camelCased
    themselves but their values pass through verbatim — they hold user
    payload data (tool args, tool response, JSON Schemas) whose internal
    structure must not be rewritten.
    """
    if isinstance(value, dict):
        narrowed = cast("dict[str, Any]", value)
        result: dict[str, Any] = {}
        for k, v in narrowed.items():
            new_key = to_camel(k)
            if k in _PASSTHROUGH_VALUE_KEYS:
                # Bytes inside passthrough values still need base64 (binary
                # payloads shouldn't be serialized as raw bytes); other
                # values pass through unchanged.
                result[new_key] = _encode_bytes_only(v)
            else:
                result[new_key] = _camelize(v)
        return result
    if isinstance(value, list):
        narrowed_list = cast("list[Any]", value)
        return [_camelize(item) for item in narrowed_list]
    if isinstance(value, tuple):
        narrowed_tuple = cast("tuple[Any, ...]", value)
        return [_camelize(item) for item in narrowed_tuple]
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return value


def _encode_bytes_only(value: Any) -> Any:
    """Recursively encode ``bytes`` as base64 without rewriting dict keys."""
    if isinstance(value, dict):
        narrowed = cast("dict[str, Any]", value)
        return {k: _encode_bytes_only(v) for k, v in narrowed.items()}
    if isinstance(value, list):
        narrowed_list = cast("list[Any]", value)
        return [_encode_bytes_only(item) for item in narrowed_list]
    if isinstance(value, tuple):
        narrowed_tuple = cast("tuple[Any, ...]", value)
        return [_encode_bytes_only(item) for item in narrowed_tuple]
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return value
