"""Context dataclass for pipeline execution.

Wraps a mitmproxy HTTPFlow (or bare http.Request for shapes) as a
first-class member. Content fields (messages, system, tools, settings,
raw_extras, request_parameters) are lazy-parsed into Pydantic AI typed
objects and flushed back via commit(). Header mutations are live — they
hit the flow immediately.

Context satisfies :class:`ccproxy.lightllm.adapters.LLMRenderInput` —
adapters and the outbound dispatcher accept Context directly via that
Protocol; there is no intermediate IR bundle.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import MISSING, dataclass, field, fields
from dataclasses import Field as DataclassField
from dataclasses import replace as _dataclass_replace
from typing import TYPE_CHECKING, Any, Self

from glom import assign as _glom_assign
from glom import delete as _glom_delete
from glom import glom as _glom_get
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic_ai.messages import ModelMessage, SystemPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from ccproxy.inspector.fingerprint import CapturedFingerprint
from ccproxy.lightllm.parsed import InboundFormat

if TYPE_CHECKING:
    from mitmproxy import http
    from mitmproxy.http import HTTPFlow


_EXTRAS_MISSING = object()
_METADATA_PREFIX = "ccproxy."
_METADATA_FIELD_KEY = "ccproxy_metadata_key"


class _ExtrasAccessor:
    """Typed glom-pathed accessor over ``Context._body``.

    Layer 3 of the three-layer access model — equivalent to raw
    ``glom(ctx._body, path)`` calls but typed and discoverable.

    Operates directly on ``ctx._body`` so mutations are visible to the
    rest of the pipeline immediately; ``commit()`` re-renders the IR on
    top later. Existing ``glom(ctx._body, ...)`` call sites stay
    valid — migration is opportunistic.

    Path strings are standard glom dot-paths
    (``"metadata.user_id"``, ``"pplx.attachments"``, etc.).
    """

    __slots__ = ("_ctx",)

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

    def get(self, path: str, default: Any = None) -> Any:
        """Read ``path`` from the body; returns ``default`` if missing."""
        return _glom_get(self._ctx._body, path, default=default)

    def set(self, path: str, value: Any) -> None:
        """Write ``value`` at ``path``, creating intermediate dicts as needed."""
        _glom_assign(self._ctx._body, path, value, missing=dict)

    def delete(self, path: str) -> None:
        """Delete ``path`` from the body; no-op if missing."""
        _glom_delete(self._ctx._body, path, ignore_missing=True)

    def has(self, path: str) -> bool:
        """True if ``path`` resolves to a value (including falsy values)."""
        return _glom_get(self._ctx._body, path, default=_EXTRAS_MISSING) is not _EXTRAS_MISSING


def metadata_field(
    *,
    key: str | None = None,
    default: Any = None,
    default_factory: Callable[[], Any] | None = None,
) -> Any:
    """Declare a typed ccproxy metadata field backed by ``flow.metadata``."""
    metadata = {_METADATA_FIELD_KEY: key}
    if default_factory is not None:
        return field(default_factory=default_factory, metadata=metadata)
    return field(default=default, metadata=metadata)


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), slots=False, eq=False)
class MetadataSection(MutableMapping[str, Any]):
    """Base for typed ccproxy metadata sections backed by ``flow.metadata``."""

    _source: MutableMapping[Any, Any] = field(repr=False, compare=False)
    _prefix: str = field(default="", repr=False, compare=False)
    _ready: bool = field(default=False, init=False, repr=False, compare=False)

    @classmethod
    def from_source(cls, source: MutableMapping[Any, Any], prefix: str = "") -> Self:
        values: dict[str, Any] = {}
        for field_ in fields(cls):
            item = cls._field_storage(field_)
            if item is None:
                continue
            _, storage_key = item(prefix)
            if storage_key in source:
                values[field_.name] = source[storage_key]
        instance = cls(_source={}, _prefix=prefix, **values)
        object.__setattr__(instance, "_source", source)
        return instance

    def __post_init__(self) -> None:
        object.__setattr__(self, "_ready", True)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name.startswith("_") or not getattr(self, "_ready", False):
            return
        storage_key = type(self)._storage_key_for_field(name, self._prefix)
        if storage_key is None:
            storage_key = self._storage_key(name, self._prefix)
        if value is None:
            self._source.pop(storage_key, None)
        else:
            self._source[storage_key] = value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        storage_key = self._storage_key(name, self._prefix)
        if storage_key in self._source:
            return self._source[storage_key]
        prefix = self._logical_key_for_prefix(name, self._prefix)
        return MetadataSection.from_source(self._source, prefix)

    @classmethod
    def _storage_key(cls, key: str, prefix: str = "") -> str:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if key.startswith(_METADATA_PREFIX):
            return key
        logical_key = cls._logical_key_for_prefix(key, prefix)
        return f"{_METADATA_PREFIX}{logical_key}"

    @staticmethod
    def _logical_key_for_prefix(key: str, prefix: str) -> str:
        if not prefix or key == prefix or key.startswith(f"{prefix}."):
            return key
        return f"{prefix}.{key}"

    @staticmethod
    def _relative_key_for_prefix(key: Any, prefix: str) -> str | None:
        if not isinstance(key, str) or not key.startswith(_METADATA_PREFIX):
            return None
        logical_key = key[len(_METADATA_PREFIX) :]
        if not prefix:
            return logical_key
        if logical_key == prefix:
            return ""
        prefix_dot = f"{prefix}."
        if logical_key.startswith(prefix_dot):
            return logical_key[len(prefix_dot) :]
        return None

    @classmethod
    def _field_storage(cls, field_: DataclassField[Any]) -> Callable[[str], tuple[str, str]] | None:
        if _METADATA_FIELD_KEY not in field_.metadata:
            return None
        logical_key = field_.metadata[_METADATA_FIELD_KEY] or field_.name
        if not isinstance(logical_key, str):
            raise TypeError(f"metadata key for {field_.name} must be a string")
        return lambda prefix: (
            cls._logical_key_for_prefix(logical_key, prefix),
            cls._storage_key(logical_key, prefix),
        )

    @classmethod
    def _storage_key_for_field(cls, field_name: str, prefix: str) -> str | None:
        for field_ in fields(cls):
            if field_.name != field_name:
                continue
            item = cls._field_storage(field_)
            return item(prefix)[1] if item is not None else None
        return None

    @classmethod
    def _field_name_for_storage_key(cls, storage_key: str, prefix: str) -> str | None:
        for field_ in fields(cls):
            item = cls._field_storage(field_)
            if item is not None and item(prefix)[1] == storage_key:
                return field_.name
        return None

    @staticmethod
    def _field_default(field_: DataclassField[Any]) -> Any:
        if field_.default_factory is not MISSING:  # type: ignore[comparison-overlap]
            return field_.default_factory()  # type: ignore[misc]
        if field_.default is not MISSING:
            return field_.default
        return None

    def __getitem__(self, key: str) -> Any:
        return self._source[self._storage_key(key, self._prefix)]

    def __setitem__(self, key: str, value: Any) -> None:
        storage_key = self._storage_key(key, self._prefix)
        self._source[storage_key] = value
        field_name = type(self)._field_name_for_storage_key(storage_key, self._prefix)
        if field_name is not None:
            object.__setattr__(self, field_name, value)

    def __delitem__(self, key: str) -> None:
        storage_key = self._storage_key(key, self._prefix)
        del self._source[storage_key]
        field_name = type(self)._field_name_for_storage_key(storage_key, self._prefix)
        if field_name is None:
            return
        for field_ in fields(type(self)):
            if field_.name == field_name:
                object.__setattr__(self, field_name, self._field_default(field_))
                return

    def __iter__(self) -> Iterator[str]:
        for key in self._source:
            logical = self._relative_key_for_prefix(key, self._prefix)
            if logical is not None:
                yield logical

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._storage_key(key, self._prefix) in self._source

    def __repr__(self) -> str:
        return repr(dict(self.items()))

    def _set_optional(self, key: str, value: Any | None) -> None:
        if value is None:
            self.pop(key, None)
        else:
            self[key] = value


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), slots=False, eq=False)
class PplxMetadata(MetadataSection):
    """Typed ``ccproxy.pplx.*`` metadata."""

    preflight: bool | None = metadata_field(default=None)
    resolved_via: str = metadata_field(default="")
    divergence: str = metadata_field(default="")
    captured_ids: dict[str, str] | None = metadata_field(default=None)


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), slots=False, eq=False)
class FingerprintMetadata(MetadataSection):
    """Typed ``ccproxy.fingerprint.*`` metadata."""

    client: dict[str, Any] | None = metadata_field(default=None)
    profile: dict[str, Any] | None = metadata_field(default=None)


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), slots=False, eq=False)
class CcproxyMetadata(MetadataSection):
    """Typed facade over ccproxy-owned mitmproxy flow metadata.

    Fields are declared once with :func:`metadata_field`; construction
    populates those fields from ``flow.metadata`` and assignment writes back
    to the corresponding ``ccproxy.*`` key. Mapping access stays available for
    dynamic keys.
    """

    record: Any | None = metadata_field(default=None)
    direction: str = metadata_field(default="")
    conversation_id: str = metadata_field(default="")
    system_prompt_sha: str = metadata_field(default="")
    sse_transformer: Any | None = metadata_field(default=None)
    otel_span: Any | None = metadata_field(default=None)
    otel_span_ended: bool = metadata_field(default=False)
    auth_provider: str = metadata_field(default="")
    auth_injected: bool = metadata_field(default=False)
    session_id: str = metadata_field(default="")
    inbound_format: str = metadata_field(default="unknown")
    request_parameters: ModelRequestParameters | None = metadata_field(key="parsed_request_parameters", default=None)
    hook_results: list[Any] = metadata_field(default_factory=list)
    transport_override: bool = metadata_field(default=False)
    fingerprint_profile: str = metadata_field(default="")
    retry_transport: str = metadata_field(default="")
    retry_profile: str = metadata_field(default="")
    legacy_client_fingerprint: dict[str, Any] | None = metadata_field(key="client_fingerprint", default=None)

    @property
    def pplx(self) -> PplxMetadata:
        return PplxMetadata.from_source(self._source, "pplx")

    @property
    def fingerprint(self) -> FingerprintMetadata:
        return FingerprintMetadata.from_source(self._source, "fingerprint")


def metadata_from_flow(flow: Any) -> CcproxyMetadata:
    """Return the ccproxy metadata facade for a mitmproxy flow."""
    return CcproxyMetadata.from_source(flow.metadata)


def _replace_system_parts(
    messages: list[ModelMessage],
    system_parts: list[SystemPromptPart],
) -> list[ModelMessage]:
    """Return ``messages`` with all ``SystemPromptPart``s replaced by ``system_parts``.

    System parts are stripped from every ``ModelRequest`` and the new
    parts are prepended to the first ``ModelRequest``. If no
    ``ModelRequest`` exists, one is created at the front.
    """
    # deferred: import inside function to avoid a top-level cycle if dataclasses change
    from pydantic_ai.messages import ModelRequest

    result: list[ModelMessage] = []
    placed = False
    for msg in messages:
        if isinstance(msg, ModelRequest):
            non_system = [p for p in msg.parts if not isinstance(p, SystemPromptPart)]
            if not placed:
                result.append(_dataclass_replace(msg, parts=[*system_parts, *non_system]))
                placed = True
            else:
                result.append(_dataclass_replace(msg, parts=non_system))
        else:
            result.append(msg)
    if not placed and system_parts:
        result.insert(0, ModelRequest(parts=list(system_parts)))
    return result


def _select_inbound_format(req: http.Request | None) -> InboundFormat:
    """Determine the listener-side wire format from path + headers.

    The choice is independent of upstream auth provider resolution
    (which happens later in the pipeline via ``inject_auth``) — wire
    format is dictated by what the client SENT, not what we route to.
    """
    if req is None:
        return InboundFormat.UNKNOWN
    path = (req.path or "").split("?", 1)[0]
    if path.startswith("/v1/messages") or req.headers.get("anthropic-version"):
        return InboundFormat.ANTHROPIC_MESSAGES
    if path.startswith("/v1/chat/completions") or path.startswith("/chat/completions"):
        return InboundFormat.OPENAI_CHAT
    if (
        path.startswith("/v1/responses")
        or path.startswith("/responses")
        or path.startswith("/backend-api/codex/responses")
    ):
        return InboundFormat.OPENAI_RESPONSES
    return InboundFormat.UNKNOWN


@dataclass
class Context:
    """Typed context for hook pipeline execution.

    The flow (or bare request) is the source of truth. Body fields are
    parsed once on first access and flushed back via :meth:`commit`.

    Satisfies :class:`ccproxy.lightllm.adapters.LLMRenderInput` —
    adapters consume Context directly for outbound wire rendering.
    """

    flow: HTTPFlow | None
    """Mitmproxy flow (None for shape-only contexts)."""

    _body: dict[str, Any] = field(default_factory=dict, repr=False)
    """Parsed JSON request body, flushed back via commit()."""

    _request: http.Request | None = field(default=None, repr=False)
    """Bare request for shape contexts (no flow)."""

    _local_flow_metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    """Flow-metadata backing store for request-only contexts."""

    _inbound_format: InboundFormat = field(default=InboundFormat.UNKNOWN, repr=False)
    """Listener-side wire format, pinned at construction. UNKNOWN for unmatched routes."""

    # Lazy-parsed IR cache. ``None`` = not yet parsed; ``parse_sync()`` populates.
    _cached_messages: list[ModelMessage] | None = field(default=None, repr=False)
    """Lazy-parsed typed messages, populated by parse_sync()."""

    _cached_system: list[SystemPromptPart] | None = field(default=None, repr=False)
    """Lazy-parsed typed system prompts, populated by parse_sync()."""

    _cached_request_parameters: ModelRequestParameters | None = field(default=None, repr=False)
    """Lazy-parsed tool / output config, populated by parse_sync()."""

    _cached_settings: ModelSettings | None = field(default=None, repr=False)
    """Lazy-parsed sampling settings, populated by parse_sync()."""

    _cached_raw_extras: dict[str, Any] | None = field(default=None, repr=False)
    """Lazy-parsed raw_extras (wire fields not absorbed into IR), populated by parse_sync()."""

    def invalidate_parsed(self) -> None:
        """Drop cached parse state so the next access re-parses from ``_body``."""
        self._cached_messages = None
        self._cached_system = None
        self._cached_request_parameters = None
        self._cached_settings = None
        self._cached_raw_extras = None

    def parse_sync(self) -> None:
        """Parse ``self._body`` via the listener-format-matched parser.

        Populates the five lazy-parsed slots in-place. Returns ``None``.
        Subsequent calls are no-ops until :meth:`invalidate_parsed` clears
        the cache.

        Sync because the new adapters in :mod:`ccproxy.lightllm.adapters`
        are pure (``json.loads`` + procedural dispatch), so there's no
        asyncio bridge to maintain.
        """
        if self._cached_messages is not None:
            return  # already parsed

        if self._inbound_format is InboundFormat.UNKNOWN:
            self._cached_messages = []
            self._cached_system = []
            self._cached_request_parameters = ModelRequestParameters()
            self._cached_settings = ModelSettings()
            self._cached_raw_extras = {}
            return

        from ccproxy.lightllm.adapters._envelope import parse_request_into_fields

        parse_request_into_fields(
            body=self._body,
            inbound_format=self._inbound_format,
            ctx=self,
        )

    @classmethod
    def from_flow(cls, flow: HTTPFlow) -> Context:
        """Build Context from a mitmproxy HTTPFlow."""
        try:
            body = json.loads(flow.request.content or b"{}")
        except (json.JSONDecodeError, TypeError):
            body = {}
        return cls(
            flow=flow,
            _body=body,
            _inbound_format=_select_inbound_format(flow.request),
        )

    @classmethod
    def from_request(cls, req: http.Request) -> Context:
        """Build Context from a bare http.Request (for shapes, no flow)."""
        try:
            body = json.loads(req.content or b"{}")
        except (json.JSONDecodeError, TypeError):
            body = {}
        return cls(
            flow=None,
            _body=body,
            _request=req,
            _inbound_format=_select_inbound_format(req),
        )

    @property
    def extras(self) -> _ExtrasAccessor:
        """Typed glom-pathed accessor over ``self._body``.

        Layer 3 of the three-layer access model. Equivalent to raw
        ``glom(ctx._body, path)`` calls but typed and discoverable.
        Existing call sites that use ``glom`` directly remain valid.
        """
        return _ExtrasAccessor(self)

    # --- LLMRenderInput Protocol properties ---

    @property
    def model(self) -> str:
        return str(self._body.get("model", ""))

    @model.setter
    def model(self, value: str) -> None:
        self._body["model"] = value

    @property
    def messages(self) -> list[ModelMessage]:
        self.parse_sync()
        assert self._cached_messages is not None
        return self._cached_messages

    @messages.setter
    def messages(self, value: list[ModelMessage]) -> None:
        self.parse_sync()
        self._cached_messages = value

    @property
    def request_parameters(self) -> ModelRequestParameters:
        self.parse_sync()
        assert self._cached_request_parameters is not None
        return self._cached_request_parameters

    @request_parameters.setter
    def request_parameters(self, value: ModelRequestParameters) -> None:
        self.parse_sync()
        self._cached_request_parameters = value

    @property
    def settings(self) -> ModelSettings:
        self.parse_sync()
        assert self._cached_settings is not None
        return self._cached_settings

    @settings.setter
    def settings(self, value: ModelSettings) -> None:
        self.parse_sync()
        self._cached_settings = value

    @property
    def raw_extras(self) -> dict[str, Any]:
        self.parse_sync()
        assert self._cached_raw_extras is not None
        return self._cached_raw_extras

    @raw_extras.setter
    def raw_extras(self, value: dict[str, Any]) -> None:
        self.parse_sync()
        self._cached_raw_extras = value

    @property
    def stream(self) -> bool:
        """Whether the request uses SSE streaming."""
        return bool(self._body.get("stream", False))

    @stream.setter
    def stream(self, value: bool) -> None:
        self._body["stream"] = value

    # --- Convenience accessors (not in LLMRenderInput) ---

    @property
    def system(self) -> list[SystemPromptPart]:
        """Top-level system prompts extracted from the message stream."""
        self.parse_sync()
        if self._cached_system is None:
            self._cached_system = [
                part
                for msg in (self._cached_messages or [])
                if hasattr(msg, "parts")
                for part in msg.parts
                if isinstance(part, SystemPromptPart)
            ]
        return self._cached_system

    @system.setter
    def system(self, value: list[SystemPromptPart]) -> None:
        self.parse_sync()
        self._cached_system = value

    @property
    def tools(self) -> list[ToolDefinition]:
        """Function tool definitions extracted from request_parameters."""
        return list(self.request_parameters.function_tools)

    @tools.setter
    def tools(self, value: list[ToolDefinition]) -> None:
        self.parse_sync()
        assert self._cached_request_parameters is not None
        self._cached_request_parameters = _dataclass_replace(
            self._cached_request_parameters, function_tools=list(value)
        )

    @property
    def tool_choice(self) -> Any:
        """Tool choice configuration from the request body."""
        return self._body.get("tool_choice")

    @tool_choice.setter
    def tool_choice(self, value: Any) -> None:
        self._body["tool_choice"] = value

    # --- ccproxy flow metadata ---

    @property
    def metadata(self) -> CcproxyMetadata:
        """ccproxy-owned metadata stored on ``flow.metadata``.

        Keys are presented without the ``ccproxy.`` prefix, so
        ``ctx.metadata["session_id"]`` is backed by
        ``flow.metadata["ccproxy.session_id"]``. Use ``ctx.extras`` for
        request-body paths such as ``metadata.user_id``.
        """
        return CcproxyMetadata.from_source(self.flow_metadata)

    @metadata.setter
    def metadata(self, value: dict[str, Any]) -> None:
        target = self.flow_metadata
        for key in list(target):
            if isinstance(key, str) and key.startswith(_METADATA_PREFIX):
                del target[key]
        for key, item in value.items():
            self.metadata[key] = item

    # --- Inspector metadata ---

    @property
    def flow_metadata(self) -> dict[str, Any]:
        """Mitmproxy flow metadata. Separate from request-body ``metadata``."""
        if self.flow is None:
            return self._local_flow_metadata
        return self.flow.metadata

    @property
    def client_fingerprint(self) -> CapturedFingerprint | None:
        metadata = self.metadata
        raw = metadata.fingerprint.client or metadata.legacy_client_fingerprint
        return CapturedFingerprint.from_dict(raw) if isinstance(raw, dict) else None

    @property
    def replay_fingerprint(self) -> CapturedFingerprint | None:
        raw = self.metadata.fingerprint.profile
        return CapturedFingerprint.from_dict(raw) if isinstance(raw, dict) else None

    # --- Headers (read/write flow.request.headers directly) ---

    @property
    def headers(self) -> dict[str, str]:
        """Snapshot of flow headers, lowercased keys."""
        req = self._resolve_request()
        if req is None:
            return {}
        return {k.lower(): v for k, v in req.headers.items()}  # type: ignore[no-untyped-call]

    def get_header(self, name: str, default: str = "") -> str:
        """Get header value (case-insensitive)."""
        req = self._resolve_request()
        if req is None:
            return default
        return req.headers.get(name, default)  # type: ignore[no-any-return]

    def set_header(self, name: str, value: str) -> None:
        """Set or remove a header on the flow."""
        req = self._resolve_request()
        if req is None:
            return
        if value == "":
            req.headers.pop(name, None)
        else:
            req.headers[name] = value

    @property
    def authorization(self) -> str:
        return self.get_header("authorization")

    @property
    def x_api_key(self) -> str:
        return self.get_header("x-api-key")

    @property
    def flow_id(self) -> str:
        if self.flow is not None:
            return self.flow.id
        return ""

    # --- Metadata convenience properties ---

    @property
    def auth_provider(self) -> str:
        return self.metadata.auth_provider

    @auth_provider.setter
    def auth_provider(self, value: str) -> None:
        self.metadata.auth_provider = value

    # --- Commit ---

    def _flush_parsed_to_body(self) -> None:
        """Re-render mutated typed properties back into ``self._body``.

        Invokes the listener-format outbound dispatcher to produce wire
        bytes from Context's typed state, then replaces ``self._body``
        with the result.

        UNKNOWN listener format is a no-op — there's no IR roundtrip path,
        and the typed-property getters return empty defaults so there's
        nothing to flush.
        """
        if self._inbound_format is InboundFormat.UNKNOWN:
            return

        # If the caller mutated ctx.system, rebuild messages so the first
        # ModelRequest carries the new system parts and any prior system
        # parts are stripped.
        if self._cached_system is not None:
            self._cached_messages = _replace_system_parts(
                list(self._cached_messages or []),
                self._cached_system,
            )

        # Pick the listener-side adapter and render bytes.
        from ccproxy.lightllm.adapters.anthropic import AnthropicAdapter
        from ccproxy.lightllm.adapters.openai_chat import OpenAIChatAdapter
        from ccproxy.lightllm.adapters.openai_responses import OpenAIResponsesAdapter

        if self._inbound_format is InboundFormat.ANTHROPIC_MESSAGES:
            rendered = AnthropicAdapter.render(self)
        elif self._inbound_format is InboundFormat.OPENAI_CHAT:
            rendered = OpenAIChatAdapter.render(self)
        elif self._inbound_format is InboundFormat.OPENAI_RESPONSES:
            rendered = OpenAIResponsesAdapter.render(self)
        else:
            raise ValueError(f"no outbound renderer for inbound_format={self._inbound_format}")

        self._body = json.loads(rendered)

    def commit(self) -> None:
        """Flush body mutations back to the underlying request content.

        If a typed property setter mutated the cached IR, re-render the
        IR back to listener-wire bytes via the matching outbound adapter
        and refresh ``self._body`` from that. Raw ``_body`` mutations (the
        shaping inner-DAG, ``extract_pplx_files``) are picked up directly.

        Strips empty ``metadata`` dicts injected by property access —
        upstream APIs reject unknown fields (e.g. Google: "Unknown name
        metadata").
        """
        if (
            self._cached_messages is not None
            or self._cached_system is not None
            or self._cached_request_parameters is not None
            or self._cached_settings is not None
            or self._cached_raw_extras is not None
        ):
            self._flush_parsed_to_body()
        body = self._body
        if "metadata" in body and isinstance(body["metadata"], dict) and not body["metadata"]:
            del body["metadata"]
        encoded = json.dumps(body).encode()

        if self.flow is not None:
            self.flow.request.content = encoded
        elif self._request is not None:
            self._request.content = encoded

    # --- Internal ---

    def _resolve_request(self) -> http.Request | None:
        """Return the underlying http.Request, from flow or direct."""
        if self.flow is not None:
            return self.flow.request  # type: ignore[return-value]
        return self._request
