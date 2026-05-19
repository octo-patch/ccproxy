"""Tests for the Perplexity step renderer dispatcher (`pplx_steps`)."""

from __future__ import annotations

import json

from ccproxy.lightllm.pplx_steps import (
    StepRenderResult,
    content_field_for,
    render_step,
)


def test_content_field_for_convention() -> None:
    assert content_field_for("MCP_TOOL_INPUT") == "mcp_tool_input_content"
    assert content_field_for("INITIAL_QUERY") == "initial_query_content"
    assert content_field_for("FINAL") == "final_content"
    assert content_field_for("WAT") == "wat_content"


def test_render_step_dispatches_by_step_type_to_specialized_renderer() -> None:
    step = {
        "step_type": "SEARCH_WEB",
        "uuid": "u1",
        "search_web_content": {"queries": ["quantum computing"]},
    }
    result = render_step(step)
    assert "Web search" in result.reasoning_text
    assert "quantum computing" in result.reasoning_text
    assert result.structured is not None
    assert result.structured["phase"] == "search"
    assert result.structured["step_uuid"] == "u1"


def test_render_step_unknown_step_type_falls_through_to_generic() -> None:
    step = {"step_type": "XYZ_NEW", "uuid": "u9", "xyz_new_content": {"summary": "hello"}}
    result = render_step(step)
    assert "[XYZ_NEW]" in result.reasoning_text
    assert "hello" in result.reasoning_text
    assert result.structured is not None
    assert "unmapped_step" in result.structured
    assert result.structured["unmapped_step"]["step_type"] == "XYZ_NEW"
    assert result.structured["unmapped_step"]["content"] == {"summary": "hello"}


def test_render_step_text_field_shape_uses_generic_content_key() -> None:
    # The text-field JSON channel uses `content` instead of typed `*_content`.
    step = {
        "step_type": "MCP_TOOL_INPUT",
        "uuid": "u2",
        "content": {
            "app": "GitHub",
            "tool_name": "get_me",
            "tool_args": {},
            "tool_input_summary": "Get me",
        },
    }
    result = render_step(step)
    assert "[GitHub]" in result.reasoning_text
    assert "get_me" in result.reasoning_text
    assert result.structured is not None
    assert "mcp_step" in result.structured


def test_render_initial_query_is_suppressed() -> None:
    step = {"step_type": "INITIAL_QUERY", "uuid": "u0", "initial_query_content": {"query": "..."}}
    result = render_step(step)
    assert result == StepRenderResult()


def test_render_final_is_suppressed() -> None:
    step = {"step_type": "FINAL", "uuid": "uf", "final_content": {"answer": "..."}}
    result = render_step(step)
    assert result == StepRenderResult()


def test_render_search_web_multiple_queries_joined() -> None:
    step = {
        "step_type": "SEARCH_WEB",
        "uuid": "u",
        "search_web_content": {"queries": ["a", "b", "c"]},
    }
    result = render_step(step)
    assert "a · b · c" in result.reasoning_text


def test_render_read_results_includes_url_sample() -> None:
    step = {
        "step_type": "READ_RESULTS",
        "uuid": "u",
        "read_results_content": {"urls": ["http://x/1", "http://x/2", "http://x/3", "http://x/4"]},
    }
    result = render_step(step)
    assert "Read 4 results" in result.reasoning_text
    assert "http://x/1" in result.reasoning_text
    assert "…" in result.reasoning_text


def test_render_mcp_tool_input_full_structured_and_text() -> None:
    step = {
        "step_type": "MCP_TOOL_INPUT",
        "uuid": "step-uuid-1",
        "mcp_tool_input_content": {
            "goal_id": "0",
            "tool_name": "list_pull_requests",
            "tool_args": {"author": "starbaser", "per_page": 5},
            "app": "GitHub",
            "tool_input_summary": "Listing recent PRs",
            "request_user_approval": {"uuid": "", "request_user_approval": False},
            "approval_result": None,
            "mcp_server_type": "MCP_SERVER_TYPE_REMOTE",
            "source_type": "github_mcp_direct",
            "authenticated": True,
            "logo_url": "https://example/icon.png",
        },
    }
    result = render_step(step)
    assert "[GitHub] list_pull_requests" in result.reasoning_text
    assert '"author":"starbaser"' in result.reasoning_text
    assert "Listing recent PRs" in result.reasoning_text
    assert result.structured is not None
    mcp = result.structured["mcp_step"]
    assert mcp["phase"] == "input"
    assert mcp["app"] == "GitHub"
    assert mcp["tool_name"] == "list_pull_requests"
    assert mcp["tool_args"] == {"author": "starbaser", "per_page": 5}
    assert mcp["goal_id"] == "0"
    assert mcp["needs_user_approval"] is False
    assert mcp["mcp_server_type"] == "MCP_SERVER_TYPE_REMOTE"
    assert mcp["source_type"] == "github_mcp_direct"
    assert mcp["authenticated"] is True


def test_render_mcp_tool_input_empty_args_renders_empty_braces() -> None:
    step = {
        "step_type": "MCP_TOOL_INPUT",
        "uuid": "u",
        "mcp_tool_input_content": {"app": "GitHub", "tool_name": "get_me", "tool_args": {}},
    }
    result = render_step(step)
    assert "get_me({})" in result.reasoning_text


def test_render_mcp_tool_input_needs_user_approval_propagated() -> None:
    step = {
        "step_type": "MCP_TOOL_INPUT",
        "uuid": "u",
        "mcp_tool_input_content": {
            "tool_name": "create_branch",
            "tool_args": {"name": "feat/x"},
            "app": "GitHub",
            "request_user_approval": {"request_user_approval": True},
        },
    }
    result = render_step(step)
    assert result.structured is not None
    assert result.structured["mcp_step"]["needs_user_approval"] is True


def test_render_mcp_tool_output_success_parses_json_content() -> None:
    raw_payload = {"login": "starbaser", "id": 207763516}
    step = {
        "step_type": "MCP_TOOL_OUTPUT",
        "uuid": "out-1",
        "mcp_tool_output_content": {
            "goal_id": "0",
            "status": "success",
            "content": json.dumps(raw_payload),
            "should_rerun_query": False,
            "app": "GitHub",
            "tool_name": "get_me",
        },
    }
    result = render_step(step)
    assert "get_me (success)" in result.reasoning_text
    assert result.structured is not None
    mcp = result.structured["mcp_step"]
    assert mcp["phase"] == "output"
    assert mcp["status"] == "success"
    assert mcp["content"] == raw_payload  # JSON-decoded
    assert mcp["should_rerun_query"] is False
    assert mcp["goal_id"] == "0"


def test_render_mcp_tool_output_non_json_content_falls_back_to_string() -> None:
    step = {
        "step_type": "MCP_TOOL_OUTPUT",
        "uuid": "out-2",
        "mcp_tool_output_content": {"status": "success", "content": "plain text result"},
    }
    result = render_step(step)
    assert result.structured is not None
    mcp = result.structured["mcp_step"]
    assert mcp["content"] == "plain text result"


def test_render_terminate_with_reason() -> None:
    step = {"step_type": "TERMINATE", "uuid": "u", "terminate_content": {"reason": "complete"}}
    result = render_step(step)
    assert "Done" in result.reasoning_text
    assert "complete" in result.reasoning_text


def test_render_browser_search() -> None:
    step = {"step_type": "BROWSER_SEARCH", "uuid": "u", "browser_search_content": {"query": "python"}}
    result = render_step(step)
    assert "Browser search" in result.reasoning_text
    assert "python" in result.reasoning_text


def test_render_url_navigate() -> None:
    step = {"step_type": "URL_NAVIGATE", "uuid": "u", "url_navigate_content": {"url": "https://example.com"}}
    result = render_step(step)
    assert "https://example.com" in result.reasoning_text


def test_render_generate_image() -> None:
    step = {
        "step_type": "GENERATE_IMAGE",
        "uuid": "u",
        "generate_image_content": {"prompt": "a sunset"},
    }
    result = render_step(step)
    assert "Generating image" in result.reasoning_text
    assert "a sunset" in result.reasoning_text


def test_render_generate_image_results() -> None:
    step = {
        "step_type": "GENERATE_IMAGE_RESULTS",
        "uuid": "u",
        "generate_image_results_content": {"image_results": [{"url": "x"}, {"url": "y"}]},
    }
    result = render_step(step)
    assert "2 images generated" in result.reasoning_text


def test_render_create_tasks() -> None:
    step = {
        "step_type": "CREATE_TASKS",
        "uuid": "u",
        "create_tasks_content": {"tasks": [{"title": "a"}, {"title": "b"}]},
    }
    result = render_step(step)
    assert "Creating 2 tasks" in result.reasoning_text


def test_render_code() -> None:
    step = {"step_type": "CODE", "uuid": "u", "code_content": {"language": "python"}}
    result = render_step(step)
    assert "Code execution" in result.reasoning_text
    assert "python" in result.reasoning_text


def test_render_clarifying_questions_non_raising() -> None:
    step = {
        "step_type": "CLARIFYING_QUESTIONS",
        "uuid": "u",
        "clarifying_questions_content": {"questions": ["q1", "q2"]},
    }
    result = render_step(step)
    assert "Clarifying questions" in result.reasoning_text
    assert result.structured is not None
    assert result.structured["questions"] == ["q1", "q2"]


def test_render_step_unknown_step_type_with_no_summary_renders_just_marker() -> None:
    step = {"step_type": "MYSTERIOUS_STEP", "uuid": "u", "mysterious_step_content": {}}
    result = render_step(step)
    assert result.reasoning_text.strip() == "[MYSTERIOUS_STEP]"
    assert result.structured is not None
    assert result.structured["unmapped_step"]["step_type"] == "MYSTERIOUS_STEP"
