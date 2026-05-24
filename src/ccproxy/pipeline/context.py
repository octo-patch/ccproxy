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
from dataclasses import dataclass, field
from dataclasses import replace as _dataclass_replace
from typing import TYPE_CHECKING, Any

from glom import assign as _glom_assign
from glom import delete as _glom_delete
from glom import glom as _glom_get
from pydantic_ai.messages import ModelMessage, SystemPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from ccproxy.lightllm.parsed import InboundFormat

if TYPE_CHECKING:
    from mitmproxy import http
    from mitmproxy.http import HTTPFlow


_EXTRAS_MISSING = object()


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

    The choice is independent of upstream OAuth provider resolution
    (which happens later in the pipeline via ``forward_oauth``) — wire
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

    # --- Body metadata ---

    @property
    def metadata(self) -> dict[str, Any]:
        return self._body.setdefault("metadata", {})  # type: ignore[no-any-return]

    @metadata.setter
    def metadata(self, value: dict[str, Any]) -> None:
        self._body["metadata"] = value

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
    def ccproxy_oauth_provider(self) -> str:
        return str(self.metadata.get("ccproxy_oauth_provider", ""))

    @ccproxy_oauth_provider.setter
    def ccproxy_oauth_provider(self, value: str) -> None:
        self.metadata["ccproxy_oauth_provider"] = value

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
