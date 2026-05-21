"""Context dataclass for pipeline execution.

Wraps a mitmproxy HTTPFlow (or bare http.Request for shapes) as a
first-class member. Content fields (messages, system, tools) are
lazy-parsed into Pydantic AI typed objects and flushed back via
commit(). Header mutations are live — they hit the flow immediately.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from dataclasses import replace as _dataclass_replace
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import ModelMessage, SystemPromptPart
from pydantic_ai.tools import ToolDefinition

from ccproxy.lightllm.parsed import ListenerFormat, ParsedRequest

if TYPE_CHECKING:
    from mitmproxy import http
    from mitmproxy.http import HTTPFlow


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


def _select_listener_format(req: http.Request | None) -> ListenerFormat:
    """Determine the listener-side wire format from path + headers.

    The choice is independent of upstream OAuth provider resolution
    (which happens later in the pipeline via ``forward_oauth``) — wire
    format is dictated by what the client SENT, not what we route to.
    """
    if req is None:
        return ListenerFormat.UNKNOWN
    path = (req.path or "").split("?", 1)[0]
    if path.startswith("/v1/messages") or req.headers.get("anthropic-version"):
        return ListenerFormat.ANTHROPIC_MESSAGES
    if path.startswith("/v1/chat/completions") or path.startswith("/chat/completions"):
        return ListenerFormat.OPENAI_CHAT
    return ListenerFormat.UNKNOWN


@dataclass
class Context:
    """Typed context for hook pipeline execution.

    The flow (or bare request) is the source of truth. Body fields are
    parsed once on first access and flushed back via commit().
    """

    flow: HTTPFlow | None
    """Mitmproxy flow (None for shape-only contexts)."""

    _body: dict[str, Any] = field(default_factory=dict, repr=False)
    """Parsed JSON request body, flushed back via commit()."""

    _request: http.Request | None = field(default=None, repr=False)
    """Bare request for shape contexts (no flow)."""

    _cached_messages: list[ModelMessage] | None = field(default=None, repr=False)
    """Lazy-parsed typed messages, populated on first access."""

    _cached_system: list[SystemPromptPart] | None = field(default=None, repr=False)
    """Lazy-parsed typed system prompts, populated on first access."""

    _cached_tools: list[ToolDefinition] | None = field(default=None, repr=False)
    """Lazy-parsed typed tool definitions, populated on first access."""

    _listener_format: ListenerFormat = field(default=ListenerFormat.UNKNOWN, repr=False)
    """Listener-side wire format, pinned at construction. UNKNOWN for unmatched routes."""

    _parsed: ParsedRequest | None = field(default=None, repr=False)
    """Lazy-parsed IR view of the request. Populated by per-listener parser on demand."""

    async def ensure_parsed(self) -> ParsedRequest:
        """Lazily parse ``self._body`` via the listener-format-matched inbound parser.

        Raises ``ValueError`` if the listener format is UNKNOWN — callers
        that need the IR view should branch on ``self._listener_format``
        first. Subsequent calls return the cached ``ParsedRequest`` even
        if ``_body`` has been mutated; call ``invalidate_parsed()`` to
        force a re-parse.
        """
        if self._parsed is not None:
            return self._parsed
        from ccproxy.lightllm.anthropic_inbound import parse_anthropic_messages
        from ccproxy.lightllm.openai_inbound import parse_openai_chat

        if self._listener_format is ListenerFormat.ANTHROPIC_MESSAGES:
            self._parsed = await parse_anthropic_messages(self._body)
        elif self._listener_format is ListenerFormat.OPENAI_CHAT:
            self._parsed = await parse_openai_chat(self._body)
        else:
            raise ValueError(f"no IR parser for listener_format={self._listener_format}")
        return self._parsed

    def invalidate_parsed(self) -> None:
        """Drop the cached ``ParsedRequest`` so the next ``ensure_parsed`` re-parses."""
        self._parsed = None

    def parse_sync(self) -> ParsedRequest:
        """Sync wrapper around :meth:`ensure_parsed`.

        Drives the async parser on a private event loop so sync callers
        (xepor route handlers, mitmproxy stream callbacks) can pull the
        IR view without contaminating the surrounding async runtime.
        Safe because the inbound parsers raise ``CaptureSentinel`` before
        any actual I/O, so the loop never blocks on the network.
        """
        if self._parsed is not None:
            return self._parsed
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.ensure_parsed())
        finally:
            loop.close()

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
            _listener_format=_select_listener_format(flow.request),
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
            _listener_format=_select_listener_format(req),
        )

    # --- Typed content properties ---

    @property
    def messages(self) -> list[ModelMessage]:
        if self._cached_messages is None:
            if self._listener_format is ListenerFormat.UNKNOWN:
                self._cached_messages = []
            else:
                self._cached_messages = self.parse_sync().messages
        return self._cached_messages

    @messages.setter
    def messages(self, value: list[ModelMessage]) -> None:
        self._cached_messages = value
        if self._parsed is not None:
            self._parsed = _dataclass_replace(self._parsed, messages=value)
        # _body re-serialization happens at commit() via the outbound renderer.

    @property
    def system(self) -> list[SystemPromptPart]:
        if self._cached_system is None:
            if self._listener_format is ListenerFormat.UNKNOWN:
                self._cached_system = []
            else:
                # SystemPromptParts live inside the ModelRequest parts of the IR.
                # Extract them so hooks that read ctx.system see the canonical view.
                self._cached_system = [
                    part
                    for msg in self.parse_sync().messages
                    if hasattr(msg, "parts")
                    for part in msg.parts
                    if isinstance(part, SystemPromptPart)
                ]
        return self._cached_system

    @system.setter
    def system(self, value: list[SystemPromptPart]) -> None:
        self._cached_system = value
        # No direct write-back to _body — commit() re-renders via outbound.

    @property
    def tools(self) -> list[ToolDefinition]:
        if self._cached_tools is None:
            if self._listener_format is ListenerFormat.UNKNOWN:
                self._cached_tools = []
            else:
                self._cached_tools = list(self.parse_sync().request_parameters.function_tools)
        return self._cached_tools

    @tools.setter
    def tools(self, value: list[ToolDefinition]) -> None:
        self._cached_tools = value
        # No direct write-back to _body — commit() re-renders via outbound.

    @property
    def model(self) -> str:
        return str(self._body.get("model", ""))

    @model.setter
    def model(self, value: str) -> None:
        self._body["model"] = value

    @property
    def stream(self) -> bool:
        """Whether the request uses SSE streaming."""
        return bool(self._body.get("stream", False))

    @stream.setter
    def stream(self, value: bool) -> None:
        self._body["stream"] = value

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

        Builds (or refreshes) ``self._parsed`` from the cached typed
        properties, then calls the listener-format outbound renderer to
        produce wire bytes, and replaces ``self._body`` with the result.

        UNKNOWN listener format is a no-op — there's no IR roundtrip
        path, and the typed-property getters return ``[]`` for that case
        so there's nothing to flush.
        """
        if self._listener_format is ListenerFormat.UNKNOWN:
            return

        from ccproxy.lightllm.outbound import render_outbound_sync

        # Ensure we have a base ParsedRequest to mutate.
        parsed = self.parse_sync()

        if self._cached_messages is not None or self._cached_system is not None:
            # System parts live INSIDE ModelRequest.parts in the IR — when the
            # caller mutated ``ctx.system``, rebuild messages so the first
            # ModelRequest carries the new system parts and any prior system
            # parts are stripped.
            messages = list(self._cached_messages if self._cached_messages is not None else parsed.messages)
            if self._cached_system is not None:
                messages = _replace_system_parts(messages, self._cached_system)
            parsed = _dataclass_replace(parsed, messages=messages)

        if self._cached_tools is not None:
            new_params = _dataclass_replace(parsed.request_parameters, function_tools=list(self._cached_tools))
            parsed = _dataclass_replace(parsed, request_parameters=new_params)

        self._parsed = parsed
        # ``provider`` here is the LISTENER format name — the outbound dispatcher
        # routes it to the matching renderer (anthropic/openai).
        listener_provider = "anthropic" if self._listener_format is ListenerFormat.ANTHROPIC_MESSAGES else "openai"
        rendered = render_outbound_sync(parsed, provider=listener_provider)
        self._body = json.loads(rendered)

    def commit(self) -> None:
        """Flush body mutations back to the underlying request content.

        If a typed property setter mutated ``self._parsed``, re-render the
        IR back to listener-wire bytes via the matching outbound renderer
        and refresh ``self._body`` from that. Raw ``_body`` mutations (the
        shaping inner-DAG, ``extract_pplx_files``) are picked up directly.

        Strips empty ``metadata`` dicts injected by property access —
        upstream APIs reject unknown fields (e.g. Google: "Unknown name
        metadata").
        """
        if self._cached_messages is not None or self._cached_system is not None or self._cached_tools is not None:
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
