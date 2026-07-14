"""Alloy PromptAst → pydantic-ai IR adapter (library-only, inbound-only).

Converts Alloy's provider-specialized prompt AST JSON (emitted by
``alloy/baml/render_prompt {projection: "ast"}`` via ``prompt_ast_to_json``)
directly into pydantic-ai's ``list[ModelMessage]`` IR — the same IR the
lightllm pipeline operates on. This is the **library lane** bridge: Alloy
renders typed prompts, a Python consumer (first: talkstream) converts them
here and rides pydantic-ai / lightllm for transport.

The transform is AST → IR **directly**. It never lowers PromptAst to a wire
format and re-parses through :meth:`AnthropicAdapter.load_messages` — both
ends already carry structure, so a wire round-trip would only lose fidelity.

**Library-only contract.** :class:`PromptAstAdapter` is NOT an
:class:`~ccproxy.lightllm.parsed.InboundFormat`, has no
:class:`~ccproxy.pipeline.context.Context` integration, and is not wired into
``dispatch_dump_sync`` / ``dispatch_intake`` / ``dispatch_render``. PromptAst
is not a listener wire format the proxy receives — the proxy pipeline is
untouched by this module. The single downstream integration is
:func:`parsed_request_from_alloy`, which builds a
:class:`~ccproxy.lightllm.parsed.ParsedRequest` a library consumer can hand
straight to :func:`ccproxy.lightllm.graph.dispatch_dump_sync`.

The AST arrives already provider-specialized by Alloy: roleless prompts are
role-wrapped, adjacent same-role messages merged, adjacent text runs
coalesced, and roles validated against the client's allowed-role list before
the JSON is produced. ``ctx.output_format`` (the SAP schema prose) is already
burned into message text and is intentionally not recovered as structure.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserContent,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from ccproxy.lightllm.parsed import ParsedRequest

if TYPE_CHECKING:
    from collections.abc import Callable

# pydantic-ai's CachePoint accepts only these two TTLs (Literal['5m', '1h']);
# any other TTL is stashed verbatim in raw_extras under the `cc:` convention.
_SUPPORTED_TTLS: frozenset[str] = frozenset({"5m", "1h"})

# Media kinds Alloy emits, plus the resolved kind a `generic` node dispatches to.
type _MediaKind = Literal["image", "audio", "video", "pdf"]

# Keys the pydantic-ai ModelSettings TypedDict models; everything else in an
# Alloy requestBody is carried through raw_extras['alloy_request_body'].
_MODEL_SETTINGS_KEYS: frozenset[str] = frozenset(ModelSettings.__annotations__)


class PromptAstAdapter:
    """Alloy PromptAst → pydantic-ai IR loader.

    A standalone class (not a :class:`pydantic_ai.ui.UIAdapter` subclass):
    PromptAst is inbound-only with no wire-format analogue, so the UIAdapter
    base contract (``build_run_input`` / ``build_event_stream`` /
    ``dump_messages``) would be pure stubs. The class exposes exactly one
    entry point, :meth:`load_messages`, matching the classmethod conventions
    of the wire adapters.
    """

    @classmethod
    def load_messages(
        cls,
        prompt_ast: Mapping[str, Any],
        *,
        raw_extras: dict[str, Any] | None = None,
    ) -> list[ModelMessage]:
        """Convert a provider-specialized PromptAst JSON tree to IR messages.

        Consecutive ``system``/``user`` messages accumulate as parts of one
        :class:`ModelRequest`; an ``assistant``/``model`` message closes the
        open request (if any) and appends a :class:`ModelResponse`. Message
        order and text bytes are preserved exactly.

        Args:
            prompt_ast: The root PromptAst node — ``vec``, ``message``, or
                ``simple``.
            raw_extras: Optional sink for anything the IR can't model
                (non-standard cache TTLs, non-cache metadata). Never dropped
                on the floor when provided.

        Returns:
            The conversation as pydantic-ai IR messages.

        Raises:
            ValueError: On an unknown role, an unrecognized node/content type,
                media in a system/assistant message, a local ``file`` media
                source, or base64/generic media missing ``mimeType``.
        """
        messages: list[ModelMessage] = []
        request_parts: list[ModelRequestPart] = []

        def flush_request() -> None:
            nonlocal request_parts
            if request_parts:
                messages.append(ModelRequest(parts=request_parts))
                request_parts = []

        for msg_index, node in enumerate(cls._root_nodes(prompt_ast)):
            ntype = node.get("type")

            if ntype == "simple":
                # Defensive: the specialized pipeline role-wraps roleless
                # prompts, but the encoder admits a bare `simple` at root.
                content = cls._load_user_content(node.get("content") or {}, msg_index=msg_index)
                request_parts.append(cls._user_prompt_part(node, content, msg_index=msg_index, raw_extras=raw_extras))
                continue

            if ntype != "message":
                raise ValueError(
                    f"PromptAst node {msg_index} has unexpected type {ntype!r}; expected 'message' or 'simple'"
                )

            role = node.get("role")
            if role == "system":
                text = cls._text_only_content(node.get("content") or {}, role="system", msg_index=msg_index)
                request_parts.append(SystemPromptPart(content=text))

                def emit_system_cache(ttl: Literal["5m", "1h"]) -> None:
                    # Mirror the Anthropic adapter: a sentinel UserPromptPart
                    # carries the system-level cache marker between turns.
                    request_parts.append(UserPromptPart(content=[CachePoint(ttl=ttl)]))

                cls._apply_metadata(
                    node.get("metadata"), msg_index=msg_index, raw_extras=raw_extras, emit_cache_point=emit_system_cache
                )

            elif role == "user":
                content = cls._load_user_content(node.get("content") or {}, msg_index=msg_index)
                request_parts.append(cls._user_prompt_part(node, content, msg_index=msg_index, raw_extras=raw_extras))

            elif role in ("assistant", "model"):
                flush_request()
                text = cls._text_only_content(node.get("content") or {}, role="assistant", msg_index=msg_index)
                messages.append(ModelResponse(parts=[TextPart(content=text)]))

                def emit_response_cache(ttl: Literal["5m", "1h"], *, i: int = msg_index) -> None:
                    # ModelResponse has no CachePoint slot — preserve verbatim.
                    if raw_extras is not None:
                        raw_extras[f"cc:promptast:msg:{i}"] = {"type": "ephemeral", "ttl": ttl}

                cls._apply_metadata(
                    node.get("metadata"),
                    msg_index=msg_index,
                    raw_extras=raw_extras,
                    emit_cache_point=emit_response_cache,
                )

            else:
                raise ValueError(
                    f"PromptAst message {msg_index} has unknown role {role!r}; Alloy validates roles against the "
                    "client's allowed-role list before emitting, so an unknown role here is a contract break"
                )

        flush_request()
        return messages

    # ── node walking ─────────────────────────────────────────────────────────

    @classmethod
    def _root_nodes(cls, prompt_ast: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        ptype = prompt_ast.get("type")
        if ptype == "vec":
            return list(prompt_ast.get("items") or [])
        if ptype in ("message", "simple"):
            return [prompt_ast]
        raise ValueError(f"PromptAst root has unexpected type {ptype!r}; expected 'vec', 'message', or 'simple'")

    @classmethod
    def _user_prompt_part(
        cls,
        node: Mapping[str, Any],
        content: str | list[UserContent],
        *,
        msg_index: int,
        raw_extras: dict[str, Any] | None,
    ) -> UserPromptPart:
        """Build a UserPromptPart, folding any cache marker into the content list."""
        parts: list[UserContent] = [content] if isinstance(content, str) else list(content)

        def emit_cache(ttl: Literal["5m", "1h"]) -> None:
            parts.append(CachePoint(ttl=ttl))

        cls._apply_metadata(
            node.get("metadata"), msg_index=msg_index, raw_extras=raw_extras, emit_cache_point=emit_cache
        )

        # No cache marker and a sole text run → keep the plain-string content shape.
        if isinstance(content, str) and len(parts) == 1:
            return UserPromptPart(content=content)
        return UserPromptPart(content=parts)

    # ── content mapping ──────────────────────────────────────────────────────

    @classmethod
    def _load_user_content(cls, content: Mapping[str, Any], *, msg_index: int) -> str | list[UserContent]:
        """Map a user/simple content node to IR content (``str`` or a list)."""
        ctype = content.get("type")
        if ctype == "text":
            return cast(str, content.get("text", ""))
        if ctype == "media":
            return [cls._load_media(content, msg_index=msg_index)]
        if ctype == "multiple":
            items: list[UserContent] = []
            cls._extend_multiple(items, content.get("items") or [], msg_index=msg_index)
            return items
        raise ValueError(f"PromptAst content node in message {msg_index} has unexpected type {ctype!r}")

    @classmethod
    def _extend_multiple(cls, out: list[UserContent], items: list[Mapping[str, Any]], *, msg_index: int) -> None:
        for item in items:
            itype = item.get("type")
            if itype == "text":
                out.append(item.get("text", ""))
            elif itype == "media":
                out.append(cls._load_media(item, msg_index=msg_index))
            elif itype == "multiple":
                cls._extend_multiple(out, item.get("items") or [], msg_index=msg_index)
            else:
                raise ValueError(f"PromptAst 'multiple' item in message {msg_index} has unexpected type {itype!r}")

    @classmethod
    def _text_only_content(cls, content: Mapping[str, Any], *, role: str, msg_index: int) -> str:
        """Extract text from a system/assistant content node; media fails loudly."""
        ctype = content.get("type")
        if ctype == "text":
            return cast(str, content.get("text", ""))
        if ctype == "media":
            raise ValueError(_no_media_slot(role, msg_index))
        if ctype == "multiple":
            texts: list[str] = []
            for item in content.get("items") or []:
                itype = item.get("type")
                if itype == "text":
                    texts.append(cast(str, item.get("text", "")))
                elif itype == "media":
                    raise ValueError(_no_media_slot(role, msg_index))
                else:
                    raise ValueError(
                        f"PromptAst 'multiple' item in {role} message {msg_index} has unexpected type {itype!r}"
                    )
            return "".join(texts)
        raise ValueError(f"PromptAst content node in {role} message {msg_index} has unexpected type {ctype!r}")

    @classmethod
    def _load_media(cls, node: Mapping[str, Any], *, msg_index: int) -> UserContent:
        """Map a ``media`` content node to a URL or binary :class:`UserContent`."""
        if "file" in node:
            raise ValueError(
                f"PromptAst media in message {msg_index} uses a local 'file' path, which does not cross the "
                "process boundary; re-render the prompt with URL or base64 media"
            )

        mime = node.get("mimeType")
        kind = node.get("kind")
        if kind == "generic":
            resolved = cls._resolve_generic_kind(mime, msg_index=msg_index)
        elif kind in ("image", "audio", "video", "pdf"):
            resolved = cast("_MediaKind", kind)
        else:
            raise ValueError(f"PromptAst media in message {msg_index} has unexpected kind {kind!r}")

        if "url" in node:
            return cls._url_content(resolved, url=node["url"], mime=mime)
        if "base64" in node:
            if not mime:
                raise ValueError(
                    f"PromptAst base64 media in message {msg_index} is missing 'mimeType'; no media_type fallback"
                )
            return BinaryContent(data=base64.b64decode(node["base64"]), media_type=mime)
        raise ValueError(f"PromptAst media in message {msg_index} has none of 'url', 'file', or 'base64'")

    @staticmethod
    def _url_content(kind: _MediaKind, *, url: str, mime: str | None) -> UserContent:
        if kind == "image":
            return ImageUrl(url=url, media_type=mime)
        if kind == "audio":
            return AudioUrl(url=url, media_type=mime)
        if kind == "video":
            return VideoUrl(url=url, media_type=mime)
        return DocumentUrl(url=url, media_type=mime)

    @staticmethod
    def _resolve_generic_kind(mime: str | None, *, msg_index: int) -> _MediaKind:
        if not mime:
            raise ValueError(
                f"PromptAst 'generic' media in message {msg_index} is missing 'mimeType'; cannot resolve the media kind"
            )
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("audio/"):
            return "audio"
        if mime.startswith("video/"):
            return "video"
        if mime == "application/pdf":
            return "pdf"
        raise ValueError(
            f"PromptAst 'generic' media in message {msg_index} has mimeType {mime!r} with no dispatchable prefix"
        )

    # ── metadata (VM-003 forward contract) ───────────────────────────────────

    @classmethod
    def _apply_metadata(
        cls,
        metadata: Mapping[str, Any] | None,
        *,
        msg_index: int,
        raw_extras: dict[str, Any] | None,
        emit_cache_point: Callable[[Literal["5m", "1h"]], None],
    ) -> None:
        """Map PromptAst message ``metadata`` to CachePoint / raw_extras.

        The forward contract for Alloy VM-003 (planned ``cache_control``
        population): a supported TTL becomes a :class:`CachePoint`, any other
        TTL is stashed under ``cc:promptast:msg:{i}`` (the ``cc:`` family is
        already stripped by the round-trip contract), and any non-cache
        metadata key stashes the whole object under ``promptast_meta:msg:{i}``
        so nothing is dropped silently.
        """
        if not metadata:
            return

        if any(key != "cache_control" for key in metadata) and raw_extras is not None:
            raw_extras[f"promptast_meta:msg:{msg_index}"] = dict(metadata)

        if "cache_control" in metadata:
            cc = metadata["cache_control"]
            ttl = cc.get("ttl", "5m") if isinstance(cc, Mapping) else "5m"
            if ttl in _SUPPORTED_TTLS:
                emit_cache_point(cast("Literal['5m', '1h']", ttl))
            elif raw_extras is not None:
                raw_extras[f"cc:promptast:msg:{msg_index}"] = dict(cc) if isinstance(cc, Mapping) else cc


def parsed_request_from_alloy(
    prompt_ast: Mapping[str, Any],
    client_view: Mapping[str, Any],
    *,
    stream: bool = False,
) -> ParsedRequest:
    """Build a :class:`ParsedRequest` envelope from Alloy PromptAst + client view.

    The library-lane entry point: a consumer renders a PromptAst and the
    credential-free client view from Alloy, calls this, and hands the result
    to :func:`ccproxy.lightllm.graph.dispatch_dump_sync` for whatever upstream
    provider it targets.

    Args:
        prompt_ast: Root PromptAst node (from ``prompt_ast_to_json``).
        client_view: Alloy ``client_view`` — needs ``model``; ``requestBody``
            supplies sampling settings.
        stream: Whether the downstream request should stream.

    Returns:
        A :class:`ParsedRequest` with messages, model, and settings populated;
        ``request_parameters`` stays empty (schema prose is message text by
        design — the SAP owns typing on the reply side).

    Raises:
        ValueError: When ``client_view`` has no ``model``, or the PromptAst is
            malformed (see :meth:`PromptAstAdapter.load_messages`).
    """
    raw_extras: dict[str, Any] = {}
    messages = PromptAstAdapter.load_messages(prompt_ast, raw_extras=raw_extras)

    model = client_view.get("model")
    if not model:
        raise ValueError("Alloy client_view is missing 'model'; cannot build a ParsedRequest")

    settings, leftover = _settings_from_request_body(client_view.get("requestBody") or {})
    if leftover:
        raw_extras["alloy_request_body"] = leftover

    return ParsedRequest(
        model=model,
        messages=messages,
        request_parameters=ModelRequestParameters(),
        settings=settings,
        stream=stream,
        raw_extras=raw_extras,
    )


def _no_media_slot(role: str, msg_index: int) -> str:
    return f"PromptAst media in a {role} message (index {msg_index}) has no IR slot; Alloy providers don't emit it"


def _settings_from_request_body(request_body: Mapping[str, Any]) -> tuple[ModelSettings, dict[str, Any]]:
    """Split an Alloy ``requestBody`` into ModelSettings-modeled keys + remainder."""
    settings: dict[str, Any] = {}
    leftover: dict[str, Any] = {}
    for key, value in request_body.items():
        if key in _MODEL_SETTINGS_KEYS:
            settings[key] = value
        else:
            leftover[key] = value
    return cast(ModelSettings, settings), leftover
