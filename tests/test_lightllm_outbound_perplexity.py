"""Tests for the Perplexity Pro outbound renderer."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters

from ccproxy.lightllm.outbound_perplexity import render_perplexity_pro
from ccproxy.lightllm.parsed import ParsedRequest


def _make_parsed(
    *,
    model: str = "perplexity/best",
    messages: list[ModelMessage] | None = None,
    raw_extras: dict[str, Any] | None = None,
) -> ParsedRequest:
    """Build a minimal :class:`ParsedRequest` for tests."""
    return ParsedRequest(
        model=model,
        messages=messages or [],
        request_parameters=ModelRequestParameters(),
        settings={},
        stream=False,
        raw_extras=raw_extras or {},
    )


class TestSingleUserTextQuery:
    """Basic flow — one user message, no extras, first turn."""

    async def test_single_user_message_renders_first_turn_payload(self) -> None:
        parsed = _make_parsed(
            messages=[
                ModelRequest(parts=[UserPromptPart(content="what is quantum?")])
            ],
        )

        body = await render_perplexity_pro(parsed)

        payload = json.loads(body)
        assert payload["query_str"] == "what is quantum?"
        assert payload["params"]["dsl_query"] == "what is quantum?"
        assert payload["params"]["query_source"] == "home"
        assert payload["params"]["model_preference"] == "default"
        assert payload["params"]["version"] == "2.18"
        assert payload["params"]["use_schematized_api"] is True
        assert payload["params"]["send_back_text_in_streaming_api"] is False
        assert payload["params"]["time_from_first_type"] == 18361

    async def test_system_then_user_flattens_with_system_prefix(self) -> None:
        parsed = _make_parsed(
            messages=[
                ModelRequest(
                    parts=[
                        SystemPromptPart(content="be terse"),
                        UserPromptPart(content="what is quantum?"),
                    ]
                ),
            ],
        )

        body = await render_perplexity_pro(parsed)

        payload = json.loads(body)
        assert payload["query_str"].startswith("[System]: be terse")
        assert "what is quantum?" in payload["query_str"]
        assert payload["params"]["query_source"] == "home"

    async def test_multimodal_user_content_drops_image_block_in_flatten(self) -> None:
        parsed = _make_parsed(
            messages=[
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[
                                "what is in this image?",
                                ImageUrl(url="http://example.com/img.png"),
                            ]
                        )
                    ]
                ),
            ],
        )

        body = await render_perplexity_pro(parsed)

        payload = json.loads(body)
        assert payload["query_str"] == "what is in this image?"
        assert "image_url" not in payload["query_str"]
        assert "example.com" not in payload["query_str"]


class TestAttachmentsInRawExtras:
    """File upload chain output — extract_pplx_files hook output."""

    async def test_attachments_propagate_to_params(self) -> None:
        attachments = [
            "https://s3.example.com/upload/abc.png",
            "https://s3.example.com/upload/def.pdf",
        ]
        parsed = _make_parsed(
            messages=[
                ModelRequest(
                    parts=[UserPromptPart(content="describe these")]
                )
            ],
            raw_extras={"pplx": {"attachments": attachments}},
        )

        body = await render_perplexity_pro(parsed)

        payload = json.loads(body)
        assert payload["params"]["attachments"] == attachments

    async def test_empty_pplx_block_defaults_to_no_attachments(self) -> None:
        parsed = _make_parsed(
            messages=[
                ModelRequest(parts=[UserPromptPart(content="hi")])
            ],
            raw_extras={},
        )

        body = await render_perplexity_pro(parsed)

        payload = json.loads(body)
        assert payload["params"]["attachments"] == []


class TestThreadContinuation:
    """Followup-request shape — last_backend_uuid + read_write_token injected."""

    async def test_followup_uses_only_last_user_turn(self) -> None:
        parsed = _make_parsed(
            messages=[
                ModelRequest(parts=[UserPromptPart(content="Name a fruit")]),
                ModelResponse(parts=[TextPart(content="Apple")]),
                ModelRequest(parts=[UserPromptPart(content="Name a vegetable")]),
            ],
            raw_extras={
                "pplx": {
                    "last_backend_uuid": "backend-1",
                    "read_write_token": "rw-1",
                    "frontend_context_uuid": "ctx-stable",
                }
            },
        )

        body = await render_perplexity_pro(parsed)

        payload = json.loads(body)
        assert payload["query_str"] == "Name a vegetable"
        assert payload["params"]["dsl_query"] == "Name a vegetable"
        assert payload["params"]["query_source"] == "followup"
        assert payload["params"]["followup_source"] == "link"
        assert payload["params"]["last_backend_uuid"] == "backend-1"
        assert payload["params"]["read_write_token"] == "rw-1"  # noqa: S105
        assert payload["params"]["frontend_context_uuid"] == "ctx-stable"
        assert payload["params"]["time_from_first_type"] == 8758

    async def test_followup_with_thread_uuid_alias_triggers_followup_source(
        self,
    ) -> None:
        parsed = _make_parsed(
            messages=[
                ModelRequest(parts=[UserPromptPart(content="prior")]),
                ModelResponse(parts=[TextPart(content="r1")]),
                ModelRequest(parts=[UserPromptPart(content="next")]),
            ],
            raw_extras={"pplx": {"thread_uuid": "thread-abc"}},
        )

        body = await render_perplexity_pro(parsed)

        payload = json.loads(body)
        assert payload["query_str"] == "next"
        assert payload["params"]["query_source"] == "followup"


class TestModelSelection:
    """Different ``parsed.model`` values select different model_preferences."""

    @pytest.mark.parametrize(
        ("model_id", "expected_identifier", "expected_mode"),
        [
            ("perplexity/best", "default", "search"),
            ("perplexity/deep-research", "pplx_alpha", "research"),
            ("openai/gpt-5.4", "gpt54", "copilot"),
            ("anthropic/claude-opus-4.7", "claude47opus", "copilot"),
        ],
    )
    async def test_model_routes_to_expected_identifier_and_mode(
        self,
        model_id: str,
        expected_identifier: str,
        expected_mode: str,
    ) -> None:
        parsed = _make_parsed(
            model=model_id,
            messages=[
                ModelRequest(parts=[UserPromptPart(content="hi")])
            ],
        )

        body = await render_perplexity_pro(parsed)

        payload = json.loads(body)
        assert payload["params"]["model_preference"] == expected_identifier
        assert payload["params"]["mode"] == expected_mode

    async def test_unknown_model_raises_value_error(self) -> None:
        parsed = _make_parsed(
            model="not/a/real/model",
            messages=[ModelRequest(parts=[UserPromptPart(content="hi")])],
        )

        with pytest.raises(ValueError, match="Unknown Perplexity model"):
            await render_perplexity_pro(parsed)


class TestBinaryContentSurvivorPath:
    """Defensive: BinaryContent that wasn't stripped by extract_pplx_files."""

    async def test_residual_binary_image_drops_in_flatten(self) -> None:
        parsed = _make_parsed(
            messages=[
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[
                                "what is in this image?",
                                BinaryContent(
                                    data=b"\x89PNG\r\n\x1a\n",
                                    media_type="image/png",
                                ),
                            ]
                        )
                    ]
                )
            ],
        )

        body = await render_perplexity_pro(parsed)

        payload = json.loads(body)
        assert payload["query_str"] == "what is in this image?"
