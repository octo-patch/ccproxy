"""Tests for ccproxy.hooks.extract_pplx_files — multimodal extraction + upload chain."""
# ruff: noqa: S106, S107  # fake session-cookie literals

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ccproxy.auth.sources import FileAuthSource
from ccproxy.config import CCProxyConfig, PplxConfig, PplxUploadConfig, Provider, set_config_instance
from ccproxy.hooks.extract_pplx_files import (
    FileInfo,
    PerplexityFileError,
    _await_processing,
    _batch_create_upload_urls,
    _collect_parts,
    _decode_data_uri,
    _fetch_part,
    _fetch_url,
    _s3_upload,
    _strip_parts,
    _validate,
    extract_pplx_files,
    extract_pplx_files_guard,
)
from ccproxy.pipeline.context import Context

_PNG_BYTES = b"\x89PNG\r\n\x1a\nfakepng"
_PNG_DATA_URI = "data:image/png;base64," + base64.b64encode(_PNG_BYTES).decode()


def make_ctx(body: dict[str, Any], *, auth_provider: str = "perplexity_pro") -> Context:
    flow = MagicMock()
    flow.id = "test-id"
    flow.request.content = json.dumps(body).encode()
    flow.request.headers = {}
    flow.metadata = {"ccproxy.auth_provider": auth_provider}
    return Context.from_flow(flow)


def set_pplx_config(tmp_path: Path, *, upload: PplxUploadConfig | None = None, token: str = "cookie-token") -> None:
    token_file = tmp_path / "pplx-token"
    token_file.write_text(token)
    set_config_instance(
        CCProxyConfig(
            providers={
                "perplexity_pro": Provider(
                    auth=FileAuthSource(file=str(token_file)),
                    base_url="https://www.perplexity.ai",
                    path="/rest/sse/perplexity_ask",
                    type="perplexity_pro",
                ),
            },
            pplx=PplxConfig(upload=upload or PplxUploadConfig()),
        )
    )


def _file(name: str = "image.png", data: bytes = _PNG_BYTES, mimetype: str = "image/png") -> FileInfo:
    return FileInfo(filename=name, mimetype=mimetype, data=data, is_image=mimetype.startswith("image/"))


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


def test_guard_true_for_perplexity_sentinel() -> None:
    assert extract_pplx_files_guard(make_ctx({"messages": []})) is True


def test_guard_false_for_other_provider() -> None:
    assert extract_pplx_files_guard(make_ctx({"messages": []}, auth_provider="gemini")) is False


# ---------------------------------------------------------------------------
# _collect_parts / _strip_parts
# ---------------------------------------------------------------------------


def test_collect_parts_finds_non_text_parts_with_indices() -> None:
    messages = [
        {"role": "user", "content": "plain string"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": _PNG_DATA_URI}},
                "not-a-dict",
                {"type": "input_audio", "input_audio": {"data": "..."}},
            ],
        },
    ]

    parts = _collect_parts(messages)

    assert [(mi, pi, p["type"]) for mi, pi, p in parts] == [(1, 1, "image_url"), (1, 3, "input_audio")]


def test_strip_parts_removes_exactly_the_collected_parts() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "keep me"},
                {"type": "image_url", "image_url": {"url": _PNG_DATA_URI}},
            ],
        },
    ]
    parts = _collect_parts(messages)

    _strip_parts(messages, parts)

    assert messages[0]["content"] == [{"type": "text", "text": "keep me"}]


# ---------------------------------------------------------------------------
# _decode_data_uri
# ---------------------------------------------------------------------------


def test_decode_data_uri_base64_png() -> None:
    info = _decode_data_uri(_PNG_DATA_URI)

    assert info == FileInfo(filename="image.png", mimetype="image/png", data=_PNG_BYTES, is_image=True)


def test_decode_data_uri_urlencoded_text() -> None:
    info = _decode_data_uri("data:text/plain,hello%20world")

    assert info is not None
    assert info.data == b"hello world"
    assert info.mimetype == "text/plain"
    assert info.is_image is False


def test_decode_data_uri_rejects_invalid_shapes() -> None:
    assert _decode_data_uri("no-comma-here") is None
    assert _decode_data_uri("http://example.com,payload") is None
    assert _decode_data_uri("data:image/png;base64,!!!not-base64!!!") is None


# ---------------------------------------------------------------------------
# _fetch_part
# ---------------------------------------------------------------------------


def test_fetch_part_handles_dict_and_str_image_url() -> None:
    from_dict = _fetch_part({"type": "image_url", "image_url": {"url": _PNG_DATA_URI}})
    from_str = _fetch_part({"type": "image_url", "image_url": _PNG_DATA_URI})

    assert from_dict is not None
    assert from_str is not None
    assert from_dict.data == _PNG_BYTES
    assert from_str.data == _PNG_BYTES


def test_fetch_part_skips_unsupported_shapes() -> None:
    assert _fetch_part({"type": "input_audio", "input_audio": {}}) is None
    assert _fetch_part({"type": "image_url", "image_url": {}}) is None
    assert _fetch_part({"type": "image_url", "image_url": {"url": "ftp://host/file.png"}}) is None


# ---------------------------------------------------------------------------
# _fetch_url
# ---------------------------------------------------------------------------


def test_fetch_url_builds_fileinfo_from_response(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    response = httpx.Response(
        200,
        content=_PNG_BYTES,
        headers={"content-type": "image/png"},
        request=httpx.Request("GET", "https://example.com/pics/cat"),
    )

    with patch("ccproxy.hooks.extract_pplx_files.httpx.get", return_value=response):
        info = _fetch_url("https://example.com/pics/cat")

    assert info == FileInfo(filename="cat.png", mimetype="image/png", data=_PNG_BYTES, is_image=True)


def test_fetch_url_http_error_raises_structured_400(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)

    with (
        patch(
            "ccproxy.hooks.extract_pplx_files.httpx.get",
            side_effect=httpx.ConnectError("boom", request=httpx.Request("GET", "https://example.com/x.png")),
        ),
        pytest.raises(PerplexityFileError, match=r"Failed to fetch image_url"),
    ):
        _fetch_url("https://example.com/x.png")


# ---------------------------------------------------------------------------
# _validate
# ---------------------------------------------------------------------------


def test_validate_too_many_files(tmp_path: Path) -> None:
    set_pplx_config(tmp_path, upload=PplxUploadConfig(max_files=2))

    with pytest.raises(PerplexityFileError, match=r"Too many attachments: 3\. Maximum allowed is 2\."):
        _validate([_file("a.png"), _file("b.png"), _file("c.png")])


def test_validate_empty_file(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)

    with pytest.raises(PerplexityFileError, match=r"Attachment 'empty\.png' is empty\."):
        _validate([_file("empty.png", data=b"")])


def test_validate_oversized_file(tmp_path: Path) -> None:
    set_pplx_config(tmp_path, upload=PplxUploadConfig(max_file_size_bytes=4))

    with pytest.raises(PerplexityFileError, match=r"Attachment 'big\.png' exceeds 0\.0 MB limit"):
        _validate([_file("big.png", data=b"12345")])


# ---------------------------------------------------------------------------
# _batch_create_upload_urls
# ---------------------------------------------------------------------------

_BATCH_URL_REQ = httpx.Request("POST", "https://www.perplexity.ai/rest/uploads/batch_create_upload_urls")


def test_batch_create_upload_urls_maps_results(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    upstream = {
        "results": {
            "u1": {
                "s3_bucket_url": "https://s3/b",
                "s3_object_url": "https://s3/b/o1",
                "fields": {},
                "file_uuid": "f1",
            },
        },
        "rate_limited": False,
    }

    with patch(
        "ccproxy.hooks.extract_pplx_files.httpx.post",
        return_value=httpx.Response(200, json=upstream, request=_BATCH_URL_REQ),
    ):
        mapping = _batch_create_upload_urls([_file()], token="cookie-token")

    assert len(mapping) == 1
    result = next(iter(mapping.values()))
    assert result["s3_object_url"] == "https://s3/b/o1"
    assert result["file_uuid"] == "f1"


def test_batch_create_upload_urls_missing_results_raises(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)

    with (
        patch(
            "ccproxy.hooks.extract_pplx_files.httpx.post",
            return_value=httpx.Response(200, json={"ok": True}, request=_BATCH_URL_REQ),
        ),
        pytest.raises(PerplexityFileError, match=r"returned no results"),
    ):
        _batch_create_upload_urls([_file()], token="cookie-token")


def test_batch_create_upload_urls_rate_limited_raises_429(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    upstream = {"results": {"u1": {}}, "rate_limited": True}

    with (
        patch(
            "ccproxy.hooks.extract_pplx_files.httpx.post",
            return_value=httpx.Response(200, json=upstream, request=_BATCH_URL_REQ),
        ),
        pytest.raises(PerplexityFileError, match=r"rate-limited the upload batch"),
    ):
        _batch_create_upload_urls([_file()], token="cookie-token")


def test_batch_create_upload_urls_http_error_raises_502(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)

    with (
        patch(
            "ccproxy.hooks.extract_pplx_files.httpx.post",
            return_value=httpx.Response(500, json={}, request=_BATCH_URL_REQ),
        ),
        pytest.raises(PerplexityFileError, match=r"batch_create_upload_urls failed"),
    ):
        _batch_create_upload_urls([_file()], token="cookie-token")


# ---------------------------------------------------------------------------
# _s3_upload validation paths (network path is e2e territory)
# ---------------------------------------------------------------------------


def test_s3_upload_missing_urls_raises() -> None:
    with pytest.raises(PerplexityFileError, match=r"missing s3_bucket_url / s3_object_url"):
        _s3_upload(_file(), {"fields": {}})


def test_s3_upload_missing_fields_raises() -> None:
    with pytest.raises(PerplexityFileError, match=r"missing presigned fields"):
        _s3_upload(_file(), {"s3_bucket_url": "https://s3/b", "s3_object_url": "https://s3/b/o"})


# ---------------------------------------------------------------------------
# _await_processing
# ---------------------------------------------------------------------------


def test_await_processing_no_uuids_is_noop() -> None:
    # No HTTP patch installed — an outbound call would hit the network and fail loudly.
    _await_processing([], token="cookie-token")


def test_await_processing_swallows_http_errors(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)

    with patch(
        "ccproxy.hooks.extract_pplx_files.httpx.stream",
        side_effect=httpx.ConnectError("down", request=httpx.Request("POST", "https://www.perplexity.ai")),
    ):
        _await_processing(["f1"], token="cookie-token")


# ---------------------------------------------------------------------------
# Hook-level behavior
# ---------------------------------------------------------------------------


def test_hook_no_multimodal_parts_is_noop(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    body = {"messages": [{"role": "user", "content": "text only"}]}
    ctx = make_ctx(body)

    result = extract_pplx_files(ctx, {})

    assert result._body["messages"] == body["messages"]
    assert "pplx" not in result._body


def test_hook_no_token_strips_parts_without_upload(tmp_path: Path) -> None:
    set_config_instance(CCProxyConfig(providers={}))
    ctx = make_ctx(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": _PNG_DATA_URI}},
                    ],
                }
            ]
        }
    )

    result = extract_pplx_files(ctx, {})

    assert result._body["messages"][0]["content"] == [{"type": "text", "text": "describe"}]
    assert "pplx" not in result._body


def test_hook_unresolvable_parts_stripped_without_upload(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    ctx = make_ctx(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "listen"},
                        {"type": "input_audio", "input_audio": {"data": "zzz"}},
                    ],
                }
            ]
        }
    )

    result = extract_pplx_files(ctx, {})

    assert result._body["messages"][0]["content"] == [{"type": "text", "text": "listen"}]
    assert "pplx" not in result._body


def test_hook_uploads_data_uri_and_attaches_object_url(tmp_path: Path) -> None:
    set_pplx_config(tmp_path)
    ctx = make_ctx(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image_url", "image_url": {"url": _PNG_DATA_URI}},
                    ],
                }
            ]
        }
    )
    batch_response = httpx.Response(
        200,
        json={
            "results": {
                "u1": {
                    "s3_bucket_url": "https://s3/bucket",
                    "s3_object_url": "https://s3/bucket/object.png",
                    "fields": {"key": "object.png"},
                    "file_uuid": "f1",
                }
            },
            "rate_limited": False,
        },
        request=_BATCH_URL_REQ,
    )
    s3_session = MagicMock()
    s3_session.__enter__.return_value.post.return_value = MagicMock(status_code=204)

    with (
        patch("ccproxy.hooks.extract_pplx_files.httpx.post", return_value=batch_response),
        patch("ccproxy.hooks.extract_pplx_files.CurlSession", return_value=s3_session),
        patch("ccproxy.hooks.extract_pplx_files._await_processing") as mock_await,
    ):
        result = extract_pplx_files(ctx, {})

    assert result._body["pplx"]["attachments"] == ["https://s3/bucket/object.png"]
    assert result._body["messages"][0]["content"] == [{"type": "text", "text": "what is this?"}]
    mock_await.assert_called_once_with(["f1"], "cookie-token")
