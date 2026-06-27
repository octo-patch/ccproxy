"""Tests for OpenAI Conversations image generation + edit (CHATGPT-006).

Three layers, all network-free:

- ``image_parse`` — pure multipart / data-URL parsing + dimension probing.
- ``images`` pure helpers — pointer extraction, body building, output shape.
- ``images`` network helpers — driven by a real ``httpx.AsyncClient`` over an
  ``httpx.MockTransport`` so request shapes (and the no-Authorization presigned
  download) are asserted exactly without touching the network.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ccproxy.config import OpenAIConversationsConfig
from ccproxy.inspector.openai_conversations_addon import OpenAIConversationsAddon
from ccproxy.inspector.routes.images import _claim_image_flow, register_image_routes
from ccproxy.lightllm.openai.conversations_image_parse import (
    ParsedImageEdit,
    ParsedImageGen,
    RemoteImageURLError,
    guess_mime_from_name,
    mime_to_ext,
    parse_image_edit_request,
    parse_image_gen_request,
    probe_image_dimensions,
)
from ccproxy.lightllm.openai.conversations_images import (
    ImageGenerationError,
    ImagePointer,
    UploadedImage,
    attach_uploaded_image,
    build_image_conversation_body,
    build_images_response,
    download_image,
    extract_pointers_from_conversation,
    extract_pointers_from_sse,
    image_model_slug,
    poll_conversation_for_pointers,
    upload_image,
)

# One module-level loop reused across all async helper calls — the suite closes
# the ambient loop elsewhere, so a dedicated loop keeps these deterministic.
_loop = asyncio.new_event_loop()


def _run(coro: Any) -> Any:
    return _loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Fixture bytes
# ---------------------------------------------------------------------------


def _png(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big")


def _gif(width: int, height: int) -> bytes:
    return b"GIF89a" + width.to_bytes(2, "little") + height.to_bytes(2, "little") + b"\x00" * 4


def _jpeg(width: int, height: int) -> bytes:
    return (
        b"\xff\xd8"
        + b"\xff\xc0"
        + (0x11).to_bytes(2, "big")
        + b"\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x00" * 8
    )


def _multipart(*, boundary: str, image: bytes, filename: str, mime: str, prompt: str) -> bytes:
    b = boundary.encode()
    return (
        b"--" + b + b"\r\n"
        b'Content-Disposition: form-data; name="image"; filename="' + filename.encode() + b'"\r\n'
        b"Content-Type: " + mime.encode() + b"\r\n\r\n" + image + b"\r\n"
        b"--" + b + b"\r\n"
        b'Content-Disposition: form-data; name="prompt"\r\n\r\n' + prompt.encode() + b"\r\n"
        b"--" + b + b"--\r\n"
    )


# ---------------------------------------------------------------------------
# parse_image_gen_request
# ---------------------------------------------------------------------------


class TestParseImageGenRequest:
    def test_basic(self) -> None:
        parsed = parse_image_gen_request(
            json.dumps({"prompt": "a red cube", "model": "gpt-image-1", "n": 2, "size": "1024x1024"}).encode()
        )
        assert parsed == ParsedImageGen(prompt="a red cube", model="gpt-image-1", n=2, size="1024x1024")

    def test_missing_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty 'prompt'"):
            parse_image_gen_request(json.dumps({"model": "gpt-image-1"}).encode())

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_image_gen_request(b"not json{{")


# ---------------------------------------------------------------------------
# parse_image_edit_request
# ---------------------------------------------------------------------------


class TestParseImageEditMultipart:
    def test_binary_integrity_and_fields(self) -> None:
        png = _png(2, 3) + b"\x00\xff\xfe\r\n\x89tail"  # bytes that include CR/LF
        body = _multipart(boundary="X1Y2", image=png, filename="cat.png", mime="image/png", prompt="make it blue")
        parsed = parse_image_edit_request(body=body, content_type="multipart/form-data; boundary=X1Y2")
        assert parsed.image_bytes == png
        assert parsed.prompt == "make it blue"
        assert parsed.filename == "cat.png"
        assert parsed.mime_type == "image/png"
        assert parsed.size_bytes == len(png)
        assert (parsed.width, parsed.height) == (2, 3)

    def test_missing_content_type_header_uses_filename(self) -> None:
        body = (
            b"--B\r\n"
            b'Content-Disposition: form-data; name="image"; filename="pic.gif"\r\n\r\n' + _gif(4, 5) + b"\r\n--B--\r\n"
        )
        parsed = parse_image_edit_request(body=body, content_type="multipart/form-data; boundary=B")
        assert parsed.mime_type == "image/gif"
        assert (parsed.width, parsed.height) == (4, 5)


class TestParseImageEditJson:
    def test_data_url(self) -> None:
        png = _png(7, 9)
        data_url = "data:image/png;base64," + base64.b64encode(png).decode()
        body = json.dumps({"image": data_url, "prompt": "sharpen"}).encode()
        parsed = parse_image_edit_request(body=body, content_type="application/json")
        assert parsed.image_bytes == png
        assert parsed.mime_type == "image/png"
        assert parsed.filename == "image.png"
        assert parsed.prompt == "sharpen"
        assert (parsed.width, parsed.height) == (7, 9)

    def test_remote_http_url_rejected(self) -> None:
        body = json.dumps({"image": "https://example.com/cat.png", "prompt": "x"}).encode()
        with pytest.raises(RemoteImageURLError, match="remote http"):
            parse_image_edit_request(body=body, content_type="application/json")

    def test_images_array_image_url(self) -> None:
        gif = _gif(3, 3)
        data_url = "data:image/gif;base64," + base64.b64encode(gif).decode()
        body = json.dumps({"images": [{"image_url": data_url}], "prompt": "p"}).encode()
        parsed = parse_image_edit_request(body=body, content_type="application/json")
        assert parsed.image_bytes == gif
        assert parsed.mime_type == "image/gif"


# ---------------------------------------------------------------------------
# probe_image_dimensions + mime helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimCase:
    name: str
    data: bytes
    expected: tuple[int, int]


_DIM_CASES: list[DimCase] = [
    DimCase(name="png", data=_png(12, 34), expected=(12, 34)),
    DimCase(name="gif", data=_gif(40, 50), expected=(40, 50)),
    DimCase(name="jpeg", data=_jpeg(60, 70), expected=(60, 70)),
    DimCase(name="unknown_fallback", data=b"\x00\x01\x02\x03nonsense", expected=(1024, 1024)),
    DimCase(name="truncated_png_fallback", data=b"\x89PNG\r\n\x1a\n\x00", expected=(1024, 1024)),
]


@pytest.mark.parametrize("case", [pytest.param(c, id=c.name) for c in _DIM_CASES])
def test_probe_image_dimensions(case: DimCase) -> None:
    assert probe_image_dimensions(case.data) == case.expected


def test_guess_mime_from_name() -> None:
    assert guess_mime_from_name("a.png") == "image/png"
    assert guess_mime_from_name("a.JPG") == "image/jpeg"
    assert guess_mime_from_name("noext") == "image/png"


def test_mime_to_ext() -> None:
    assert mime_to_ext("image/png") == "png"
    assert mime_to_ext("image/jpeg") == "jpg"
    assert mime_to_ext("application/octet-stream") == "bin"


# ---------------------------------------------------------------------------
# image_model_slug + body building
# ---------------------------------------------------------------------------


def test_image_model_slug() -> None:
    assert image_model_slug("gpt-image-1") == "auto"
    assert image_model_slug("dall-e-3") == "auto"
    assert image_model_slug("") == "auto"
    assert image_model_slug("gpt-5-5-pro") == "gpt-5-5-pro"


class TestBuildImageConversationBody:
    def test_generation_forces_hint_and_saved(self) -> None:
        body = build_image_conversation_body(prompt="a red cube", model="gpt-image-1")
        assert body["model"] == "auto"
        assert body["system_hints"] == ["picture_v2"]
        assert body["history_and_training_disabled"] is False
        message = body["messages"][0]
        assert message["content"] == {"content_type": "text", "parts": ["a red cube"]}
        assert message["metadata"]["system_hints"] == ["picture_v2"]

    def test_prompt_is_verbatim(self) -> None:
        body = build_image_conversation_body(prompt="exactly this", model="gpt-5-5-pro")
        assert body["model"] == "gpt-5-5-pro"
        assert body["messages"][0]["content"]["parts"] == ["exactly this"]

    def test_edit_attaches_uploaded_image(self) -> None:
        uploaded = UploadedImage(
            file_id="file_up", size_bytes=123, width=2, height=3, filename="c.png", mime_type="image/png"
        )
        body = build_image_conversation_body(prompt="make it bluer", model="", uploaded_image=uploaded)
        content = body["messages"][0]["content"]
        assert content["content_type"] == "multimodal_text"
        assert content["parts"][0] == {
            "content_type": "image_asset_pointer",
            "asset_pointer": "sediment://file_up",
            "size_bytes": 123,
            "width": 2,
            "height": 3,
        }
        assert content["parts"][1] == "make it bluer"
        attachment = body["messages"][0]["metadata"]["attachments"][0]
        assert attachment["id"] == "file_up"
        assert attachment["mime_type"] == "image/png"
        assert attachment["source"] == "library"
        assert attachment["is_big_paste"] is False


def test_attach_uploaded_image_noop_without_messages() -> None:
    body: dict[str, Any] = {"messages": []}
    attach_uploaded_image(
        body,
        prompt="x",
        uploaded=UploadedImage(file_id="f", size_bytes=1, width=1, height=1, filename="a.png", mime_type="image/png"),
    )
    assert body == {"messages": []}


# ---------------------------------------------------------------------------
# pointer extraction + output
# ---------------------------------------------------------------------------


def test_extract_pointers_from_sse_both_schemes_and_dedup() -> None:
    sse = (
        b'data: {"v": {"message": {"content": {"content_type": "multimodal_text", '
        b'"parts": [{"content_type": "image_asset_pointer", "asset_pointer": "sediment://file_gen1"}]}}}}\n\n'
        b'data: {"p": "/message/content/parts/0", "o": "append", "v": "see file-service://file_old2 now"}\n\n'
        b'data: {"p": "/x", "o": "append", "v": "sediment://file_gen1"}\n\n'
        b"data: [DONE]\n\n"
    )
    assert extract_pointers_from_sse(sse) == [
        ImagePointer(file_id="file_gen1", is_sediment=True),
        ImagePointer(file_id="file_old2", is_sediment=False),
    ]


def test_extract_pointers_from_sse_empty() -> None:
    assert extract_pointers_from_sse(b"data: {}\n\ndata: [DONE]\n\n") == []


def test_extract_pointers_from_conversation_strict_filter() -> None:
    conversation = {
        "mapping": {
            "skip": {"message": {"author": {"role": "assistant"}, "content": {"content_type": "text"}}},
            "hit": {
                "message": {
                    "author": {"role": "tool"},
                    "metadata": {"async_task_type": "image_gen"},
                    "content": {
                        "content_type": "multimodal_text",
                        "parts": [{"asset_pointer": "sediment://file_x"}],
                    },
                }
            },
        }
    }
    assert extract_pointers_from_conversation(conversation) == [ImagePointer(file_id="file_x", is_sediment=True)]


def test_extract_pointers_from_conversation_fallback_scan() -> None:
    conversation = {"note": "result at file-service://file_y somewhere"}
    assert extract_pointers_from_conversation(conversation) == [ImagePointer(file_id="file_y", is_sediment=False)]


def test_build_images_response() -> None:
    response = build_images_response([b"\x01\x02", b"\x03"])
    assert isinstance(response["created"], int)
    assert len(response["data"]) == 2
    assert response["data"][0]["b64_json"] == base64.b64encode(b"\x01\x02").decode("ascii")
    assert response["data"][0]["revised_prompt"] is None
    assert response["data"][1]["b64_json"] == base64.b64encode(b"\x03").decode("ascii")


# ---------------------------------------------------------------------------
# Injected-client network helpers (httpx.MockTransport — no real network)
# ---------------------------------------------------------------------------


def _client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _edit(image_bytes: bytes = b"IMGBYTES") -> ParsedImageEdit:
    return ParsedImageEdit(
        prompt="x",
        image_bytes=image_bytes,
        filename="c.png",
        mime_type="image/png",
        width=2,
        height=3,
        size_bytes=len(image_bytes),
    )


class TestUploadImage:
    def test_three_step_success(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            path = request.url.path
            if request.method == "POST" and path == "/backend-api/files":
                return httpx.Response(200, json={"upload_url": "https://blob.test/up?sig=abc", "file_id": "file_abc"})
            if request.method == "PUT" and request.url.host == "blob.test":
                return httpx.Response(200)
            if request.method == "POST" and path == "/backend-api/files/process_upload_stream":
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(599)

        client = _client(handler)
        uploaded = _run(
            upload_image(
                client=client,
                base_url="https://chatgpt.com",
                access_token="tok",  # noqa: S106
                device_id="dev",
                parsed=_edit(),
                timeout=5.0,
            )
        )
        _run(client.aclose())

        assert uploaded == UploadedImage(
            file_id="file_abc", size_bytes=8, width=2, height=3, filename="c.png", mime_type="image/png"
        )
        create = next(r for r in captured if r.url.path == "/backend-api/files")
        create_body = json.loads(create.content)
        assert create_body["use_case"] == "multimodal"
        assert create_body["file_size"] == 8
        assert create_body["store_in_library"] is True
        assert create.headers["authorization"] == "Bearer tok"
        assert create.headers["oai-device-id"] == "dev"
        put = next(r for r in captured if r.method == "PUT")
        assert put.headers["x-ms-blob-type"] == "BlockBlob"
        assert put.content == b"IMGBYTES"
        assert "authorization" not in put.headers
        activate = next(r for r in captured if r.url.path == "/backend-api/files/process_upload_stream")
        assert json.loads(activate.content)["file_id"] == "file_abc"

    def test_activation_falls_back_on_404(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(f"{request.method} {request.url.path}")
            path = request.url.path
            if request.method == "POST" and path == "/backend-api/files":
                return httpx.Response(200, json={"upload_url": "https://blob.test/up", "file_id": "file_z"})
            if request.method == "PUT":
                return httpx.Response(201)
            if path == "/backend-api/files/process_upload_stream":
                return httpx.Response(404)
            if path == "/backend-api/files/file_z/uploaded":
                return httpx.Response(200)
            return httpx.Response(599)

        client = _client(handler)
        uploaded = _run(
            upload_image(
                client=client,
                base_url="https://chatgpt.com",
                access_token="tok",  # noqa: S106
                device_id="dev",
                parsed=_edit(),
                timeout=5.0,
            )
        )
        _run(client.aclose())
        assert uploaded.file_id == "file_z"
        assert "POST /backend-api/files/file_z/uploaded" in seen


class TestPollConversation:
    def test_returns_on_pointer(self) -> None:
        responses = iter(
            [
                httpx.Response(200, json={"mapping": {}}),
                httpx.Response(
                    200,
                    json={
                        "mapping": {
                            "n": {
                                "message": {
                                    "author": {"role": "tool"},
                                    "metadata": {"async_task_type": "image_gen"},
                                    "content": {
                                        "content_type": "multimodal_text",
                                        "parts": [{"asset_pointer": "sediment://file_a"}],
                                    },
                                }
                            }
                        }
                    },
                ),
            ]
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return next(responses)

        client = _client(handler)
        pointers = _run(
            poll_conversation_for_pointers(
                client=client,
                base_url="https://chatgpt.com",
                conversation_id="c1",
                access_token="t",  # noqa: S106
                device_id="d",
                interval_seconds=0,
                max_attempts=5,
                timeout=5.0,
            )
        )
        _run(client.aclose())
        assert pointers == [ImagePointer(file_id="file_a", is_sediment=True)]

    def test_async_status_4_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"async_status": 4, "mapping": {}})

        client = _client(handler)
        with pytest.raises(ImageGenerationError, match="without assets"):
            _run(
                poll_conversation_for_pointers(
                    client=client,
                    base_url="https://chatgpt.com",
                    conversation_id="c1",
                    access_token="t",  # noqa: S106
                    device_id="d",
                    interval_seconds=0,
                    max_attempts=3,
                    timeout=5.0,
                )
            )
        _run(client.aclose())

    def test_timeout_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"mapping": {}})

        client = _client(handler)
        with pytest.raises(ImageGenerationError, match="timed out"):
            _run(
                poll_conversation_for_pointers(
                    client=client,
                    base_url="https://chatgpt.com",
                    conversation_id="c1",
                    access_token="t",  # noqa: S106
                    device_id="d",
                    interval_seconds=0,
                    max_attempts=2,
                    timeout=5.0,
                )
            )
        _run(client.aclose())


class TestDownloadImage:
    def test_sediment_omits_auth_on_presigned(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            if request.url.host == "chatgpt.com":
                assert request.url.path == "/backend-api/conversation/c1/attachment/file_a/download"
                return httpx.Response(200, json={"download_url": "https://dl.test/i?sig=z"})
            return httpx.Response(200, content=b"PNGDATA")

        client = _client(handler)
        data = _run(
            download_image(
                client=client,
                base_url="https://chatgpt.com",
                pointer=ImagePointer(file_id="file_a", is_sediment=True),
                conversation_id="c1",
                access_token="tok",  # noqa: S106
                device_id="dev",
                timeout=5.0,
            )
        )
        _run(client.aclose())
        assert data == b"PNGDATA"
        meta = next(r for r in captured if r.url.host == "chatgpt.com")
        assert meta.headers["authorization"] == "Bearer tok"
        presigned = next(r for r in captured if r.url.host == "dl.test")
        assert "authorization" not in presigned.headers

    def test_file_service_routing(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            if request.url.host == "chatgpt.com":
                return httpx.Response(200, json={"download_url": "https://dl.test/j?sig=q"})
            return httpx.Response(200, content=b"JPEGDATA")

        client = _client(handler)
        data = _run(
            download_image(
                client=client,
                base_url="https://chatgpt.com",
                pointer=ImagePointer(file_id="file_b", is_sediment=False),
                conversation_id="c2",
                access_token="tok",  # noqa: S106
                device_id="dev",
                timeout=5.0,
            )
        )
        _run(client.aclose())
        assert data == b"JPEGDATA"
        meta = next(r for r in captured if r.url.host == "chatgpt.com")
        assert meta.url.path == "/backend-api/files/download/file_b"
        assert meta.url.params.get("conversation_id") == "c2"
        assert meta.url.params.get("inline") == "false"


# ---------------------------------------------------------------------------
# Addon wiring (image branches on OpenAIConversationsAddon)
# ---------------------------------------------------------------------------


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.type = "openai_conversations"
    provider.host = "chatgpt.com"
    provider.path = "/backend-api/f/conversation"
    provider.fingerprint_profile = "chrome136"
    return provider


def _config(provider: MagicMock) -> MagicMock:
    cfg = MagicMock()
    cfg.providers = {"openai_conversations": provider}
    cfg.lightllm.openai_conversations = OpenAIConversationsConfig()
    return cfg


def _make_image_flow(
    *,
    operation: str,
    request_content: bytes,
    request_headers: dict[str, str] | None = None,
    response_content: bytes = b"",
    response_headers: dict[str, str] | None = None,
    status_code: int = 200,
) -> MagicMock:
    flow = MagicMock()
    flow.metadata = {
        "ccproxy.auth_provider": "openai_conversations",
        "ccproxy.oaic_image_operation": operation,
        "ccproxy.direction": "inbound",
    }
    flow.request.method = "POST"
    flow.request.path = "/v1/images/generations" if operation == "generation" else "/v1/images/edits"
    flow.request.content = request_content
    flow.request.headers = dict(request_headers or {})
    flow.response = MagicMock()
    flow.response.status_code = status_code
    flow.response.content = response_content
    flow.response.headers = dict(response_headers or {})
    return flow


_ADDON_NS = "ccproxy.inspector.openai_conversations_addon"


class TestAddonResponseHeaders:
    def test_buffers_image_event_stream(self) -> None:
        flow = _make_image_flow(
            operation="generation",
            request_content=b"{}",
            response_headers={"content-type": "text/event-stream"},
        )
        addon = OpenAIConversationsAddon()
        with patch(f"{_ADDON_NS}.get_config", return_value=_config(_provider())):
            _run(addon.responseheaders(flow))
        assert flow.response.stream is False


class TestAddonRenderImageRequest:
    def test_generation_rewrites_flow(self) -> None:
        body = json.dumps({"prompt": "a red cube", "model": "gpt-image-1"}).encode()
        flow = _make_image_flow(
            operation="generation",
            request_content=body,
            request_headers={"authorization": "Bearer tok", "content-type": "application/json"},
        )
        addon = OpenAIConversationsAddon()
        _run(
            addon._render_image_request(
                flow,
                client=MagicMock(),
                provider=_provider(),
                device_id="dev",
                oaic_cfg=OpenAIConversationsConfig(),
            )
        )
        assert flow.request.path == "/backend-api/f/conversation"
        assert flow.request.host == "chatgpt.com"
        rendered = json.loads(flow.request.content)
        assert rendered["model"] == "auto"
        assert rendered["system_hints"] == ["picture_v2"]
        assert rendered["history_and_training_disabled"] is False
        assert rendered["messages"][0]["content"]["parts"] == ["a red cube"]

    def test_edit_uploads_and_attaches(self) -> None:
        png = _png(2, 3)
        data_url = "data:image/png;base64," + base64.b64encode(png).decode()
        body = json.dumps({"image": data_url, "prompt": "bluer", "model": "gpt-image-1"}).encode()
        flow = _make_image_flow(
            operation="edit",
            request_content=body,
            request_headers={"authorization": "Bearer tok", "content-type": "application/json"},
        )

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if request.method == "POST" and path == "/backend-api/files":
                return httpx.Response(200, json={"upload_url": "https://blob.test/up", "file_id": "file_up"})
            if request.method == "PUT":
                return httpx.Response(200)
            if path == "/backend-api/files/process_upload_stream":
                return httpx.Response(200)
            return httpx.Response(599)

        addon = OpenAIConversationsAddon()
        _run(
            addon._render_image_request(
                flow,
                client=_client(handler),
                provider=_provider(),
                device_id="dev",
                oaic_cfg=OpenAIConversationsConfig(),
            )
        )
        rendered = json.loads(flow.request.content)
        content = rendered["messages"][0]["content"]
        assert content["content_type"] == "multimodal_text"
        assert content["parts"][0]["asset_pointer"] == "sediment://file_up"
        assert content["parts"][1] == "bluer"
        assert rendered["messages"][0]["metadata"]["attachments"][0]["id"] == "file_up"

    def test_remote_url_sets_400_and_clears_flag(self) -> None:
        body = json.dumps({"image": "https://example.com/c.png", "prompt": "x"}).encode()
        flow = _make_image_flow(
            operation="edit",
            request_content=body,
            request_headers={"authorization": "Bearer tok", "content-type": "application/json"},
        )
        addon = OpenAIConversationsAddon()
        _run(
            addon._render_image_request(
                flow,
                client=MagicMock(),
                provider=_provider(),
                device_id="dev",
                oaic_cfg=OpenAIConversationsConfig(),
            )
        )
        assert flow.response.status_code == 400
        error = json.loads(flow.response.content)
        assert "remote" in error["error"]["message"]
        assert flow.metadata["ccproxy.oaic_image_operation"] == ""


class TestAddonHandleImageResponse:
    def test_immediate_pointer_builds_images_response(self) -> None:
        sse = (
            b'data: {"conversation_id": "conv-1", "message": {"id": "m1", "author": {"role": "assistant"}}}\n\n'
            b'data: {"v": {"message": {"content": {"content_type": "multimodal_text", '
            b'"parts": [{"asset_pointer": "sediment://file_a"}]}}}}\n\n'
            b"data: [DONE]\n\n"
        )
        flow = _make_image_flow(
            operation="generation",
            request_content=b"{}",
            request_headers={"authorization": "Bearer tok", "oai-device-id": "dev"},
            response_content=sse,
            response_headers={"content-type": "text/event-stream"},
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "chatgpt.com" and "attachment/file_a/download" in request.url.path:
                return httpx.Response(200, json={"download_url": "https://dl.test/i?sig=z"})
            if request.url.host == "dl.test":
                return httpx.Response(200, content=b"PNGDATA")
            return httpx.Response(599)

        provider = _provider()
        addon = OpenAIConversationsAddon()
        with (
            patch(f"{_ADDON_NS}.get_config", return_value=_config(provider)),
            patch(f"{_ADDON_NS}.transport.get_client", new=AsyncMock(return_value=_client(handler))),
        ):
            _run(addon._handle_image_response(flow))

        assert flow.response.status_code == 200
        assert flow.response.headers["content-type"] == "application/json"
        payload = json.loads(flow.response.content)
        assert payload["data"][0]["b64_json"] == base64.b64encode(b"PNGDATA").decode("ascii")

    def test_upstream_error_yields_image_error(self) -> None:
        flow = _make_image_flow(
            operation="generation",
            request_content=b"{}",
            response_content=b"upstream boom",
            status_code=403,
        )
        provider = _provider()
        addon = OpenAIConversationsAddon()
        with patch(f"{_ADDON_NS}.get_config", return_value=_config(provider)):
            _run(addon._handle_image_response(flow))
        assert flow.response.status_code == 502
        error = json.loads(flow.response.content)
        assert "403" in error["error"]["message"]


# ---------------------------------------------------------------------------
# Route claims (/v1/images/{generations,edits})
# ---------------------------------------------------------------------------


class TestImageRoutes:
    def test_claim_sets_operation_for_inbound(self) -> None:
        flow = MagicMock()
        flow.metadata = {"ccproxy.direction": "inbound"}
        flow.request.path = "/v1/images/generations"
        _claim_image_flow(flow, "generation")
        assert flow.metadata["ccproxy.oaic_image_operation"] == "generation"

    def test_claim_skips_non_inbound(self) -> None:
        flow = MagicMock()
        flow.metadata = {"ccproxy.direction": "outbound"}
        _claim_image_flow(flow, "edit")
        assert "ccproxy.oaic_image_operation" not in flow.metadata

    def test_registered_routes_claim_both_paths(self) -> None:
        from ccproxy.inspector.router import InspectorRouter, RouteType

        router = InspectorRouter(name="test_images", request_passthrough=True, response_passthrough=True)
        register_image_routes(router)

        for path, expected in (("/v1/images/generations", "generation"), ("/v1/images/edits", "edit")):
            handler, _params = router.find_handler("chatgpt.com", path, RouteType.REQUEST)
            assert handler is not None
            flow = MagicMock()
            flow.metadata = {"ccproxy.direction": "inbound"}
            flow.request.path = path
            handler(flow)  # literal route → no path params
            assert flow.metadata["ccproxy.oaic_image_operation"] == expected


# ---------------------------------------------------------------------------
# image_parse error paths + edge coverage
# ---------------------------------------------------------------------------


def test_multipart_extra_fields() -> None:
    body = (
        b"--BB\r\n"
        b'Content-Disposition: form-data; name="image"; filename="a.png"\r\n'
        b"Content-Type: image/png\r\n\r\n" + _png(1, 1) + b"\r\n"
        b"--BB\r\n"
        b'Content-Disposition: form-data; name="model"\r\n\r\ngpt-image-1\r\n'
        b"--BB\r\n"
        b'Content-Disposition: form-data; name="n"\r\n\r\n2\r\n'
        b"--BB\r\n"
        b'Content-Disposition: form-data; name="size"\r\n\r\n512x512\r\n'
        b"--BB\r\n"
        b'Content-Disposition: form-data; name="prompt"\r\n\r\ngo\r\n'
        b"--BB--\r\n"
    )
    parsed = parse_image_edit_request(body=body, content_type="multipart/form-data; boundary=BB")
    assert parsed.model == "gpt-image-1"
    assert parsed.n == 2
    assert parsed.size == "512x512"
    assert parsed.prompt == "go"


def test_multipart_missing_image_raises() -> None:
    body = b'--B\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nhi\r\n--B--\r\n'
    with pytest.raises(ValueError, match="missing image part"):
        parse_image_edit_request(body=body, content_type="multipart/form-data; boundary=B")


def test_json_missing_image_ref_raises() -> None:
    with pytest.raises(ValueError, match="missing 'image'"):
        parse_image_edit_request(body=json.dumps({"prompt": "x"}).encode(), content_type="application/json")


def test_json_non_data_url_raises() -> None:
    body = json.dumps({"image": "ftp://host/y.png", "prompt": "p"}).encode()
    with pytest.raises(ValueError, match="data: URL"):
        parse_image_edit_request(body=body, content_type="application/json")


def test_json_bad_base64_raises() -> None:
    body = json.dumps({"image": "data:image/png;base64,AB", "prompt": "p"}).encode()
    with pytest.raises(ValueError, match="base64"):
        parse_image_edit_request(body=body, content_type="application/json")


def test_probe_jpeg_skips_app0_segment() -> None:
    app0 = b"\xff\xe0" + (0x10).to_bytes(2, "big") + b"JFIF\x00" + b"\x00" * 9
    jpeg = (
        b"\xff\xd8"
        + app0
        + b"\xff\xc0"
        + (0x11).to_bytes(2, "big")
        + b"\x08"
        + (70).to_bytes(2, "big")
        + (60).to_bytes(2, "big")
        + b"\x00" * 8
    )
    assert probe_image_dimensions(jpeg) == (60, 70)
