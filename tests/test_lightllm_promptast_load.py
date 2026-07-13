"""Tests for the Alloy PromptAst → pydantic-ai IR adapter (library lane).

Exercises :class:`PromptAstAdapter.load_messages` node/content/metadata
mapping with exact-structure assertions, plus the
:func:`parsed_request_from_alloy` envelope builder end-to-end through
``dispatch_dump_sync`` to Anthropic wire bytes.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    CachePoint,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from ccproxy.lightllm.adapters.promptast import (
    PromptAstAdapter,
    parsed_request_from_alloy,
)
from ccproxy.lightllm.graph import dispatch_dump_sync

# ---------------------------------------------------------------------------
# Fixtures — the worked example from the spec (live anthropic-client probe)
# ---------------------------------------------------------------------------

WORKED_EXAMPLE: dict[str, Any] = {
    "type": "vec",
    "items": [
        {
            "type": "message",
            "role": "system",
            "content": {
                "type": "text",
                "text": (
                    "You correct dictation transcripts.\nAnswer in JSON using this schema:\n"
                    "{\n  corrected: string,\n  confidence: float,\n}"
                ),
            },
            "metadata": {},
        },
        {
            "type": "message",
            "role": "user",
            "content": {
                "type": "text",
                "text": "Transcript: the quick brown facts jumped over\nHints: fox quartz",
            },
            "metadata": {},
        },
    ],
}

CLIENT_VIEW: dict[str, Any] = {
    "name": "Coprocessor",
    "provider": "anthropic",
    "model": "claude-haiku-4-5",
    "baseUrl": "http://example.invalid",
    "defaultRole": "user",
    "allowedRoles": ["system", "user", "assistant"],
    "requestBody": {"max_tokens": 512, "temperature": 0.25},
    "providerOptions": {"max_tokens": 512},
}


def _msg(role: str, text: str, **extra: Any) -> dict[str, Any]:
    node: dict[str, Any] = {"type": "message", "role": role, "content": {"type": "text", "text": text}}
    node.update(extra)
    return node


def _vec(*items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "vec", "items": list(items)}


# ---------------------------------------------------------------------------
# 1. Worked example
# ---------------------------------------------------------------------------


def test_worked_example_system_then_user() -> None:
    messages = PromptAstAdapter.load_messages(WORKED_EXAMPLE)
    assert len(messages) == 1
    request = messages[0]
    assert isinstance(request, ModelRequest)
    assert len(request.parts) == 2

    system_part, user_part = request.parts
    assert isinstance(system_part, SystemPromptPart)
    assert system_part.content == WORKED_EXAMPLE["items"][0]["content"]["text"]

    assert isinstance(user_part, UserPromptPart)
    assert user_part.content == WORKED_EXAMPLE["items"][1]["content"]["text"]


# ---------------------------------------------------------------------------
# 2. Multi-turn grouping
# ---------------------------------------------------------------------------


def test_multi_turn_grouping() -> None:
    ast = _vec(
        _msg("system", "sys"),
        _msg("user", "u1"),
        _msg("assistant", "a1"),
        _msg("user", "u2"),
    )
    messages = PromptAstAdapter.load_messages(ast)
    assert [type(m) for m in messages] == [ModelRequest, ModelResponse, ModelRequest]

    first = messages[0]
    assert isinstance(first, ModelRequest)
    assert isinstance(first.parts[0], SystemPromptPart)
    assert isinstance(first.parts[1], UserPromptPart)
    assert first.parts[0].content == "sys"
    assert first.parts[1].content == "u1"

    response = messages[1]
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == "a1"

    third = messages[2]
    assert isinstance(third, ModelRequest)
    assert isinstance(third.parts[0], UserPromptPart)
    assert third.parts[0].content == "u2"


# ---------------------------------------------------------------------------
# 3. Media variants
# ---------------------------------------------------------------------------


def test_media_image_url() -> None:
    ast = _vec(
        {
            "type": "message",
            "role": "user",
            "content": {"type": "media", "kind": "image", "url": "https://example.com/x.png"},
        }
    )
    messages = PromptAstAdapter.load_messages(ast)
    user_part = messages[0].parts[0]
    assert isinstance(user_part, UserPromptPart)
    assert isinstance(user_part.content, list)
    item = user_part.content[0]
    assert isinstance(item, ImageUrl)
    assert item.url == "https://example.com/x.png"


def test_media_image_base64_with_mimetype() -> None:
    ast = _vec(
        {
            "type": "message",
            "role": "user",
            "content": {"type": "media", "kind": "image", "mimeType": "image/png", "base64": "aGVsbG8="},
        }
    )
    messages = PromptAstAdapter.load_messages(ast)
    item = messages[0].parts[0].content[0]  # type: ignore[union-attr,index]
    assert isinstance(item, BinaryContent)
    assert item.media_type == "image/png"
    assert item.data == b"hello"


def test_media_base64_without_mimetype_raises() -> None:
    ast = _vec(
        {
            "type": "message",
            "role": "user",
            "content": {"type": "media", "kind": "image", "base64": "aGVsbG8="},
        }
    )
    with pytest.raises(ValueError, match="missing 'mimeType'"):
        PromptAstAdapter.load_messages(ast)


def test_media_file_source_raises() -> None:
    ast = _vec(
        {
            "type": "message",
            "role": "user",
            "content": {"type": "media", "kind": "image", "file": "/local/path.png"},
        }
    )
    with pytest.raises(ValueError, match="local 'file' path"):
        PromptAstAdapter.load_messages(ast)


def test_media_generic_pdf_becomes_document_url() -> None:
    ast = _vec(
        {
            "type": "message",
            "role": "user",
            "content": {
                "type": "media",
                "kind": "generic",
                "mimeType": "application/pdf",
                "url": "https://example.com/doc.pdf",
            },
        }
    )
    messages = PromptAstAdapter.load_messages(ast)
    item = messages[0].parts[0].content[0]  # type: ignore[union-attr,index]
    assert isinstance(item, DocumentUrl)
    assert item.url == "https://example.com/doc.pdf"


def test_media_generic_pdf_base64_becomes_binary() -> None:
    ast = _vec(
        {
            "type": "message",
            "role": "user",
            "content": {
                "type": "media",
                "kind": "generic",
                "mimeType": "application/pdf",
                "base64": "aGVsbG8=",
            },
        }
    )
    messages = PromptAstAdapter.load_messages(ast)
    item = messages[0].parts[0].content[0]  # type: ignore[union-attr,index]
    assert isinstance(item, BinaryContent)
    assert item.media_type == "application/pdf"
    assert item.data == b"hello"


# ---------------------------------------------------------------------------
# 4. Multiple content — ordered interleaving
# ---------------------------------------------------------------------------


def test_multiple_content_ordered() -> None:
    ast = _vec(
        {
            "type": "message",
            "role": "user",
            "content": {
                "type": "multiple",
                "items": [
                    {"type": "text", "text": "before"},
                    {"type": "media", "kind": "image", "url": "https://example.com/a.png"},
                    {"type": "text", "text": "after"},
                ],
            },
        }
    )
    messages = PromptAstAdapter.load_messages(ast)
    user_part = messages[0].parts[0]
    assert isinstance(user_part, UserPromptPart)
    assert isinstance(user_part.content, list)
    assert len(user_part.content) == 3
    assert user_part.content[0] == "before"
    assert isinstance(user_part.content[1], ImageUrl)
    assert user_part.content[1].url == "https://example.com/a.png"
    assert user_part.content[2] == "after"


# ---------------------------------------------------------------------------
# 5. Unknown role
# ---------------------------------------------------------------------------


def test_unknown_role_raises_naming_role() -> None:
    ast = _vec(_msg("moderator", "hi"))
    with pytest.raises(ValueError, match="moderator"):
        PromptAstAdapter.load_messages(ast)


# ---------------------------------------------------------------------------
# 6. Metadata → CachePoint / raw_extras
# ---------------------------------------------------------------------------


def test_metadata_cache_control_1h_emits_cache_point() -> None:
    ast = _vec(_msg("user", "ctx", metadata={"cache_control": {"ttl": "1h"}}))
    raw_extras: dict[str, Any] = {}
    messages = PromptAstAdapter.load_messages(ast, raw_extras=raw_extras)
    user_part = messages[0].parts[0]
    assert isinstance(user_part, UserPromptPart)
    assert isinstance(user_part.content, list)
    assert user_part.content[0] == "ctx"
    cache = user_part.content[1]
    assert isinstance(cache, CachePoint)
    assert cache.ttl == "1h"
    assert "cc:promptast:msg:0" not in raw_extras


def test_metadata_cache_control_unsupported_ttl_stashes() -> None:
    ast = _vec(_msg("user", "ctx", metadata={"cache_control": {"ttl": "30m"}}))
    raw_extras: dict[str, Any] = {}
    messages = PromptAstAdapter.load_messages(ast, raw_extras=raw_extras)
    user_part = messages[0].parts[0]
    assert isinstance(user_part, UserPromptPart)
    # Sole text run, no CachePoint → plain-string content preserved.
    assert user_part.content == "ctx"
    assert raw_extras["cc:promptast:msg:0"] == {"ttl": "30m"}


def test_metadata_unknown_key_stashes_verbatim() -> None:
    ast = _vec(_msg("user", "ctx", metadata={"trace_id": "abc123"}))
    raw_extras: dict[str, Any] = {}
    PromptAstAdapter.load_messages(ast, raw_extras=raw_extras)
    assert raw_extras["promptast_meta:msg:0"] == {"trace_id": "abc123"}


# ---------------------------------------------------------------------------
# 7. Integration — parsed_request_from_alloy → dispatch_dump_sync (anthropic)
# ---------------------------------------------------------------------------


def test_parsed_request_dumps_to_anthropic_body() -> None:
    req = parsed_request_from_alloy(WORKED_EXAMPLE, CLIENT_VIEW)
    assert req.model == "claude-haiku-4-5"
    assert req.settings.get("max_tokens") == 512  # type: ignore[attr-defined]
    assert req.settings.get("temperature") == 0.25  # type: ignore[attr-defined]

    wire = dispatch_dump_sync(req, provider_type="anthropic")
    body = json.loads(wire)

    assert body["model"] == "claude-haiku-4-5"
    assert body["max_tokens"] == 512
    assert body["temperature"] == 0.25

    ast_system = WORKED_EXAMPLE["items"][0]["content"]["text"]
    ast_user = WORKED_EXAMPLE["items"][1]["content"]["text"]

    # system round-trips as a plain string (single bare block).
    assert body["system"] == ast_system

    first_message = body["messages"][0]
    assert first_message["role"] == "user"
    first_block = first_message["content"][0]
    assert first_block["type"] == "text"
    assert first_block["text"] == ast_user


# ---------------------------------------------------------------------------
# 8. Bare simple root
# ---------------------------------------------------------------------------


def test_bare_simple_root() -> None:
    ast = {"type": "simple", "content": {"type": "text", "text": "just do it"}}
    messages = PromptAstAdapter.load_messages(ast)
    assert len(messages) == 1
    request = messages[0]
    assert isinstance(request, ModelRequest)
    assert len(request.parts) == 1
    assert isinstance(request.parts[0], UserPromptPart)
    assert request.parts[0].content == "just do it"
