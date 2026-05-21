"""Pydantic-AI IR → OpenAI Chat Completions wire bytes.

We render outbound by instantiating pydantic-ai's ``OpenAIChatModel`` with a
capture-only :class:`Provider` whose client raises :class:`CaptureSentinel`
on ``client.chat.completions.create(**kwargs)``. The captured kwargs are
exactly what pydantic-ai would have sent to the OpenAI SDK; we strip
``omit``/``NOT_GIVEN`` sentinels, JSON-serialize, and stitch the
inbound-parser's ``raw_extras`` back on for passthrough fidelity.

Pydantic-ai owns the per-vendor wire shape (system/developer message
routing, ``tool_calls[].function.arguments`` JSON-string serialization,
multimodal block layout, instruction inlining). This module just provides
the capture seam.
"""

from __future__ import annotations

from typing import Any, cast

import httpx
from openai import NOT_GIVEN, AsyncOpenAI, NotGiven, Omit
from pydantic_ai import ModelProfile
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers import Provider
from pydantic_core import to_json

from ccproxy.lightllm.parsed import ParsedRequest

# Keys our inbound parser stashes in ``raw_extras`` as IR-internal markers.
# We do NOT re-inject these as top-level wire fields — they're sidecars
# that the outbound assembler already accounts for via the IR.
_INTERNAL_RAW_EXTRA_PREFIXES = (
    "cc:",
    "unknown_block:",
    "refusal:",
    "file:",
    "image_detail:",
    "function_call:",
)


class CaptureSentinel(Exception):  # noqa: N818 — sentinel, not a real error class
    """Raised by the capture client to short-circuit pydantic-ai's request flow."""

    def __init__(self, kwargs: dict[str, Any]) -> None:
        super().__init__("captured")
        self.kwargs = kwargs


class _CaptureCompletions:
    """Stand-in for ``client.chat.completions``."""

    async def create(self, **kwargs: Any) -> Any:
        raise CaptureSentinel(kwargs)


class _CaptureChat:
    """Stand-in for ``client.chat``."""

    def __init__(self) -> None:
        self.completions = _CaptureCompletions()


class _CaptureOpenAIClient:
    """Stand-in for :class:`openai.AsyncOpenAI`.

    Mimics the minimal surface ``OpenAIChatModel._completions_create``
    touches: ``self.client.chat.completions.create(**kwargs)`` plus
    ``self.client.base_url`` (read by ``OpenAIChatModel.base_url``).
    """

    def __init__(self) -> None:
        self.chat = _CaptureChat()
        self.base_url = httpx.URL("https://api.openai.com/v1/")


class _CaptureOpenAIProvider(Provider[AsyncOpenAI]):
    """Stand-in for :class:`pydantic_ai.providers.openai.OpenAIProvider`.

    We declare the generic as ``AsyncOpenAI`` so pydantic-ai's type
    bookkeeping is happy, but at runtime ``self.client`` returns the
    duck-typed :class:`_CaptureOpenAIClient`. Pydantic-ai only ever calls
    ``client.chat.completions.create`` and reads ``client.base_url``;
    nothing else hits the wire.
    """

    def __init__(self) -> None:
        self._capture_client = _CaptureOpenAIClient()

    @property
    def name(self) -> str:
        return "openai"

    @property
    def base_url(self) -> str:
        return str(self._capture_client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return cast(AsyncOpenAI, self._capture_client)

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return openai_model_profile(model_name)


def _is_omit_or_not_given(value: Any) -> bool:
    """OpenAI uses two sentinel types for "field absent": ``Omit`` (typical) and ``NotGiven`` (``timeout``).

    Both must be stripped from the captured kwargs before serialization,
    otherwise we'd emit unserializable objects on the wire.
    """
    return isinstance(value, (Omit, NotGiven)) or value is NOT_GIVEN


def _scrub_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop ``Omit`` / ``NOT_GIVEN`` sentinels and ``None``-valued ``extra_body``."""
    scrubbed: dict[str, Any] = {}
    for key, value in kwargs.items():
        if _is_omit_or_not_given(value):
            continue
        if key == "extra_body" and value is None:
            continue
        scrubbed[key] = value
    return scrubbed


def _coerce_jsonable(value: Any) -> Any:
    """Lower pydantic models, TypedDicts, and other duck-typed records to JSON-safe primitives."""
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {str(k): _coerce_jsonable(v) for k, v in cast("dict[Any, Any]", value).items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(item) for item in cast("list[Any]", list(value))]
    return value


def _is_internal_raw_extra(key: str) -> bool:
    """Per-block / IR-internal markers — handled by the IR or skipped during passthrough."""
    return key.startswith(_INTERNAL_RAW_EXTRA_PREFIXES)


async def render_openai_chat(parsed: ParsedRequest) -> bytes:
    """Render a :class:`ParsedRequest` into OpenAI Chat Completions wire bytes."""
    provider = _CaptureOpenAIProvider()
    model = OpenAIChatModel(parsed.model, provider=provider)

    # ``ModelSettings`` is a TypedDict at runtime — preserve nominal typing
    # via spread (per CLAUDE.md: ``{**parsed.settings}`` not ``dict(...)``).
    settings_dict = {**parsed.settings}

    try:
        await model.request(parsed.messages, parsed.settings, parsed.request_parameters)
    except CaptureSentinel as exc:
        kwargs = exc.kwargs
    else:
        raise RuntimeError(
            "OpenAIChatModel.request did not hit the capture client — "
            "pydantic-ai's invocation surface may have changed."
        )

    body: dict[str, Any] = _scrub_kwargs(kwargs)

    # Stitch the inbound parser's raw_extras back on for passthrough fidelity.
    # Skip IR-internal markers (per-block image_detail, refusal, file, etc.)
    # and anything already present in the rendered body.
    for key, value in parsed.raw_extras.items():
        if _is_internal_raw_extra(key):
            continue
        if key in body:
            continue
        body[key] = value

    # tool_choice / response_format / parallel_tool_calls live in raw_extras
    # when the inbound parser couldn't fold them into IR fields. Force-override
    # the pydantic-ai-rendered value so the listener's intent wins.
    if "tool_choice" in parsed.raw_extras:
        body["tool_choice"] = parsed.raw_extras["tool_choice"]
    if "response_format" in parsed.raw_extras:
        body["response_format"] = parsed.raw_extras["response_format"]
    if "parallel_tool_calls" in settings_dict and "parallel_tool_calls" not in body:
        body["parallel_tool_calls"] = settings_dict["parallel_tool_calls"]

    if parsed.stream:
        body["stream"] = True

    # Drop ``extra_headers`` — that's a client-side concern, not wire data.
    body.pop("extra_headers", None)

    return _to_json_bytes(_coerce_jsonable(body))


def _to_json_bytes(value: Any) -> bytes:
    """Encode the rendered body using pydantic-core's serializer.

    Pydantic-core handles ``BaseModel``-shaped values and datetimes that
    plain ``json.dumps`` would reject, matching the encoding pydantic-ai
    itself would have used downstream.
    """
    return to_json(value)
