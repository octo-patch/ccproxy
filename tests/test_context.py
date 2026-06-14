"""Unit tests for the flow-native Context dataclass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition

from ccproxy.pipeline.context import Context

_DEFAULT_BODY: dict[str, Any] = {"model": "test", "messages": [], "metadata": {}}


def _make_flow(body: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> MagicMock:
    flow = MagicMock()
    flow.id = "test-id"
    flow.metadata = {}
    flow.request.content = json.dumps(_DEFAULT_BODY if body is None else body).encode()
    flow.request.headers = dict(headers or {})
    return flow


class TestContextFromFlow:
    def test_parses_model_from_body(self) -> None:
        flow = _make_flow(body={"model": "claude-3", "messages": []})
        ctx = Context.from_flow(flow)
        assert ctx.model == "claude-3"

    def test_parses_messages_from_body(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        flow = _make_flow(body={"model": "m", "messages": msgs})
        ctx = Context.from_flow(flow)
        assert len(ctx.messages) == 1
        assert isinstance(ctx.messages[0], ModelRequest)
        part = ctx.messages[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert part.content == "hi"

    def test_body_metadata_remains_in_extras(self) -> None:
        flow = _make_flow(body={"model": "m", "messages": [], "metadata": {"key": "val"}})
        ctx = Context.from_flow(flow)
        assert ctx.extras.get("metadata.key") == "val"

    def test_parses_system_from_body(self) -> None:
        flow = _make_flow(body={"model": "m", "messages": [], "system": "Be helpful."})
        ctx = Context.from_flow(flow)
        assert len(ctx.system) == 1
        assert ctx.system[0].content == "Be helpful."

    def test_missing_body_fields_use_defaults(self) -> None:
        flow = _make_flow(body={"model": "", "messages": [], "metadata": {}})
        ctx = Context.from_flow(flow)
        assert ctx.model == ""
        assert ctx.messages == []
        assert ctx.flow_metadata == {}
        assert ctx.system == []

    def test_invalid_json_body_uses_empty_body(self) -> None:
        flow = MagicMock()
        flow.id = "test-id"
        flow.request.content = b"not-json"
        flow.request.headers = {}
        ctx = Context.from_flow(flow)
        assert ctx.model == ""
        assert ctx.messages == []

    def test_empty_body_uses_defaults(self) -> None:
        flow = MagicMock()
        flow.id = "test-id"
        flow.request.content = b""
        flow.request.headers = {}
        ctx = Context.from_flow(flow)
        assert ctx.model == ""

    def test_flow_id_from_flow(self) -> None:
        flow = _make_flow()
        flow.id = "unique-flow-id-123"
        ctx = Context.from_flow(flow)
        assert ctx.flow_id == "unique-flow-id-123"


class TestBodyProperties:
    def test_messages_setter_writes_to_body(self) -> None:
        ctx = Context.from_flow(_make_flow())
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="test")])]
        ctx.messages = messages
        ctx.commit()
        assert isinstance(ctx._body["messages"], list)
        assert ctx._body["messages"][0]["role"] == "user"

    def test_system_setter_writes_to_body(self) -> None:
        ctx = Context.from_flow(_make_flow())
        ctx.system = [SystemPromptPart(content="Be helpful.")]
        ctx.commit()
        system_body = ctx._body["system"]
        # Anthropic outbound emits system as either a string or a list of blocks.
        if isinstance(system_body, str):
            assert system_body == "Be helpful."
        else:
            assert any(block.get("text") == "Be helpful." for block in system_body)

    def test_system_empty_list(self) -> None:
        flow = _make_flow(body={"model": "m", "messages": []})
        ctx = Context.from_flow(flow)
        assert ctx.system == []

    def test_tools_getter_and_setter(self) -> None:
        ctx = Context.from_flow(
            _make_flow(
                body={
                    "model": "m",
                    "messages": [],
                    "tools": [
                        {"name": "read_file", "description": "Read", "input_schema": {"type": "object"}},
                    ],
                }
            )
        )
        assert len(ctx.tools) == 1
        assert ctx.tools[0].name == "read_file"

    def test_tools_setter_writes_to_body(self) -> None:
        ctx = Context.from_flow(_make_flow())
        ctx.tools = [ToolDefinition(name="test", description="Test tool", parameters_json_schema={"type": "object"})]
        ctx.commit()
        assert ctx._body["tools"][0]["name"] == "test"

    def test_metadata_writes_to_ccproxy_flow_namespace(self) -> None:
        ctx = Context.from_flow(_make_flow())
        ctx.metadata.auth_provider = "anthropic"
        assert ctx.metadata.auth_provider == "anthropic"
        assert ctx.flow_metadata["ccproxy.auth_provider"] == "anthropic"

    def test_metadata_mapping_writes_dynamic_keys(self) -> None:
        ctx = Context.from_flow(_make_flow())
        ctx.metadata["new_key"] = "new_val"
        assert ctx.flow_metadata["ccproxy.new_key"] == "new_val"

    def test_metadata_accepts_prefixed_keys(self) -> None:
        ctx = Context.from_flow(_make_flow())
        ctx.metadata["ccproxy.trace_id"] = "t123"
        assert ctx.metadata["trace_id"] == "t123"
        assert ctx.flow_metadata["ccproxy.trace_id"] == "t123"

    def test_nested_metadata_section_writes_dotted_keys(self) -> None:
        ctx = Context.from_flow(_make_flow())
        ctx.metadata.pplx.preflight = True
        assert ctx.flow_metadata["ccproxy.pplx.preflight"] is True
        assert ctx.metadata.pplx.preflight is True

    def test_nested_metadata_section_reads_existing_dotted_keys(self) -> None:
        flow = _make_flow()
        flow.metadata["ccproxy.fingerprint.client"] = {"ja3": "abc"}
        ctx = Context.from_flow(flow)
        assert ctx.metadata.fingerprint.client == {"ja3": "abc"}

    def test_nested_metadata_mapping_writes_dynamic_keys(self) -> None:
        ctx = Context.from_flow(_make_flow())
        ctx.metadata.pplx.source = "web"
        assert ctx.flow_metadata["ccproxy.pplx.source"] == "web"
        assert ctx.metadata.pplx.source == "web"

    def test_dynamic_metadata_sections_can_nest(self) -> None:
        ctx = Context.from_flow(_make_flow())
        ctx.metadata.custom.section.value = 3
        assert ctx.flow_metadata["ccproxy.custom.section.value"] == 3
        assert ctx.metadata.custom.section.value == 3


class TestHeaderMethods:
    def test_get_header_exact_key_match(self) -> None:
        ctx = Context.from_flow(_make_flow(headers={"authorization": "Bearer tok"}))
        assert ctx.get_header("authorization") == "Bearer tok"

    def test_get_header_returns_default_when_missing(self) -> None:
        ctx = Context.from_flow(_make_flow(headers={}))
        assert ctx.get_header("authorization") == ""
        assert ctx.get_header("x-missing", "fallback") == "fallback"

    def test_set_header_empty_string_removes(self) -> None:
        ctx = Context.from_flow(_make_flow(headers={"x-api-key": "old"}))
        ctx.set_header("x-api-key", "")
        assert ctx.get_header("x-api-key") == ""

    def test_convenience_header_properties(self) -> None:
        ctx = Context.from_flow(_make_flow(headers={"authorization": "Bearer xyz", "x-api-key": "sk-123"}))
        assert ctx.authorization == "Bearer xyz"
        assert ctx.x_api_key == "sk-123"

    def test_headers_snapshot_lowercased(self) -> None:
        ctx = Context.from_flow(_make_flow(headers={"X-Custom": "val", "Content-Type": "json"}))
        snap = ctx.headers
        assert snap["x-custom"] == "val"
        assert snap["content-type"] == "json"


class TestMetadataConvenienceProperties:
    def test_auth_provider_getter(self) -> None:
        flow = _make_flow(body={"model": "m", "messages": []})
        flow.metadata["ccproxy.auth_provider"] = "anthropic"
        ctx = Context.from_flow(flow)
        assert ctx.auth_provider == "anthropic"


class TestCommit:
    def test_commit_writes_body_to_flow(self) -> None:
        flow = _make_flow(body={"model": "original", "messages": []})
        ctx = Context.from_flow(flow)
        ctx.model = "updated"
        ctx.commit()
        written = json.loads(flow.request.content)
        assert written["model"] == "updated"

    def test_commit_preserves_untouched_empty_body(self) -> None:
        flow = MagicMock()
        flow.id = "test-id"
        flow.metadata = {}
        flow.request.content = b""
        flow.request.headers = {"Upgrade": "websocket"}

        ctx = Context.from_flow(flow)
        ctx.metadata.auth_provider = "anthropic"
        ctx.commit()

        assert flow.request.content == b""

    def test_commit_preserves_untouched_invalid_json_body(self) -> None:
        flow = MagicMock()
        flow.id = "test-id"
        flow.metadata = {}
        flow.request.content = b"not-json"
        flow.request.headers = {}

        ctx = Context.from_flow(flow)
        ctx.metadata.auth_provider = "anthropic"
        ctx.commit()

        assert flow.request.content == b"not-json"

    def test_commit_writes_mutated_empty_body(self) -> None:
        flow = MagicMock()
        flow.id = "test-id"
        flow.metadata = {}
        flow.request.content = b""
        flow.request.headers = {}

        ctx = Context.from_flow(flow)
        ctx.extras.set("metadata.user_id", "session-1")
        ctx.commit()

        assert json.loads(flow.request.content) == {"metadata": {"user_id": "session-1"}}

    def test_commit_keeps_ccproxy_metadata_out_of_body(self) -> None:
        flow = _make_flow()
        ctx = Context.from_flow(flow)
        ctx.metadata.conversation_id = "t123"
        ctx.commit()
        written = json.loads(flow.request.content)
        assert "metadata" not in written
        assert flow.metadata["ccproxy.conversation_id"] == "t123"

    def test_commit_includes_system_when_set(self) -> None:
        flow = _make_flow()
        ctx = Context.from_flow(flow)
        ctx.system = [SystemPromptPart(content="Be helpful.")]
        ctx.commit()
        written = json.loads(flow.request.content)
        assert written["system"] == "Be helpful."

    def test_commit_round_trips_messages(self) -> None:
        flow = _make_flow(
            body={
                "model": "m",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                ],
            }
        )
        ctx = Context.from_flow(flow)
        # Access typed messages (triggers parse)
        msgs = ctx.messages
        assert len(msgs) == 2
        # Commit (triggers serialize back)
        ctx.messages = msgs
        ctx.commit()
        written = json.loads(flow.request.content)
        assert len(written["messages"]) == 2
        assert written["messages"][0]["role"] == "user"
        assert written["messages"][1]["role"] == "assistant"

    def test_commit_after_reading_responses_ir_preserves_tools(self) -> None:
        body = {
            "model": "gpt-5.5",
            "instructions": "Be direct.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "shell_command",
                    "description": "Run a shell command.",
                    "strict": False,
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "stream": True,
        }
        flow = _make_flow(body=body)
        flow.request.path = "/backend-api/codex/responses"
        ctx = Context.from_flow(flow)

        assert ctx.messages
        ctx.commit()

        written = json.loads(flow.request.content)
        assert written["tools"] == body["tools"]

    def test_header_mutations_do_not_require_commit(self) -> None:
        flow = _make_flow(headers={"x-orig": "a"})
        ctx = Context.from_flow(flow)
        ctx.set_header("x-new", "b")
        assert flow.request.headers["x-new"] == "b"


class TestFromRequest:
    def test_from_request_wraps_bare_request(self) -> None:
        req = MagicMock()
        req.content = json.dumps({"model": "test", "messages": [{"role": "user", "content": "hi"}]}).encode()
        req.headers = {}
        ctx = Context.from_request(req)
        assert ctx.flow is None
        assert ctx.model == "test"
        assert len(ctx.messages) == 1

    def test_from_request_commit_writes_to_request(self) -> None:
        req = MagicMock()
        req.content = json.dumps({"model": "old", "messages": []}).encode()
        req.headers = {}
        ctx = Context.from_request(req)
        ctx.model = "new"
        ctx.commit()
        written = json.loads(req.content)
        assert written["model"] == "new"

    def test_flow_id_empty_for_request_context(self) -> None:
        req = MagicMock()
        req.content = b"{}"
        req.headers = {}
        ctx = Context.from_request(req)
        assert ctx.flow_id == ""


class TestParseSync:
    def test_parse_sync_populates_typed_fields(self) -> None:
        from ccproxy.lightllm.parsed import InboundFormat

        flow = _make_flow(
            body={"model": "claude-3", "messages": [{"role": "user", "content": "hi"}]},
            headers={"anthropic-version": "2023-06-01"},
        )
        flow.request.path = "/v1/messages"
        ctx = Context.from_flow(flow)
        assert ctx._inbound_format is InboundFormat.ANTHROPIC_MESSAGES

        ctx.parse_sync()
        assert ctx.model == "claude-3"
        assert len(ctx.messages) == 1

    def test_parse_sync_is_idempotent(self) -> None:
        flow = _make_flow(
            body={"model": "claude-3", "messages": [{"role": "user", "content": "hi"}]},
            headers={"anthropic-version": "2023-06-01"},
        )
        flow.request.path = "/v1/messages"
        ctx = Context.from_flow(flow)

        ctx.parse_sync()
        first = ctx.messages
        ctx.parse_sync()
        second = ctx.messages
        assert first is second

    def test_parse_sync_returns_empty_for_unknown_inbound_format(self) -> None:
        flow = _make_flow(body={"model": "?", "messages": []}, headers={})
        flow.request.path = "/unknown/path"
        ctx = Context.from_flow(flow)

        ctx.parse_sync()
        # UNKNOWN inbound format yields empty defaults instead of raising.
        assert ctx.messages == []


class TestContextExtras:
    """Typed glom-pathed accessor over ``ctx._body``."""

    def test_get_returns_value_for_existing_path(self) -> None:
        flow = _make_flow(body={"model": "m", "messages": [], "metadata": {"user_id": "u123"}})
        ctx = Context.from_flow(flow)
        assert ctx.extras.get("metadata.user_id") == "u123"

    def test_get_returns_default_for_missing_path(self) -> None:
        flow = _make_flow(body={"model": "m", "messages": []})
        ctx = Context.from_flow(flow)
        assert ctx.extras.get("metadata.user_id", default="fallback") == "fallback"
        assert ctx.extras.get("does.not.exist") is None

    def test_set_creates_nested_path(self) -> None:
        flow = _make_flow(body={"model": "m", "messages": []})
        ctx = Context.from_flow(flow)
        ctx.extras.set("pplx.attachments", ["s3://x", "s3://y"])
        assert ctx._body["pplx"]["attachments"] == ["s3://x", "s3://y"]

    def test_delete_removes_existing_path_and_noops_missing(self) -> None:
        flow = _make_flow(body={"model": "m", "messages": [], "tool_choice": "auto"})
        ctx = Context.from_flow(flow)
        ctx.extras.delete("tool_choice")
        assert "tool_choice" not in ctx._body
        # idempotent — second delete is a no-op
        ctx.extras.delete("tool_choice")
        assert "tool_choice" not in ctx._body

    def test_has_distinguishes_missing_from_falsy(self) -> None:
        flow = _make_flow(body={"model": "m", "messages": [], "x": 0, "y": None, "z": ""})
        ctx = Context.from_flow(flow)
        assert ctx.extras.has("x")  # 0 is a real value
        assert ctx.extras.has("y")  # None is a real value
        assert ctx.extras.has("z")  # empty string is a real value
        assert not ctx.extras.has("missing")


def test_raw_ccproxy_flow_metadata_access_stays_private_to_context_facade() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "ccproxy"
    allowed = (root / "pipeline" / "context.py").resolve()
    patterns = (
        "flow.metadata",
        "ctx.flow.metadata",
        "ctx.flow_metadata",
        'metadata["ccproxy',
        "metadata['ccproxy",
        'metadata.get("ccproxy',
        "metadata.get('ccproxy",
    )

    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path.resolve() == allowed:
            continue
        text = path.read_text()
        if any(pattern in text for pattern in patterns):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []
