"""OpenAI Conversations image generation + edit primitives (CHATGPT-006).

Splits cleanly into three layers so the ``OpenAIConversationsAddon`` image
branch can orchestrate them and tests can exercise each in isolation:

1. **Pure body building** — :func:`build_image_conversation_body` wraps the
   text adapter's :func:`build_conversation_body`, forcing the ``picture_v2``
   generation hint (top-level *and* in the first message's metadata, the
   aurora double-placement) and a saved (non-temporary) conversation. The
   user's prompt text is passed through VERBATIM — no instruction prefix is
   injected (operator decision). :func:`attach_uploaded_image` converts the
   first user message into a ``multimodal_text`` carrying a ``sediment://``
   ``image_asset_pointer`` for the edit path.
2. **Pure extraction / output** — :func:`extract_pointers_from_sse` and
   :func:`extract_pointers_from_conversation` recover ``file-service://`` /
   ``sediment://`` pointers; :func:`build_images_response` packages downloaded
   bytes into an OpenAI ``images.response`` envelope.
3. **Injected-client network** — :func:`upload_image`,
   :func:`poll_conversation_for_pointers`, :func:`download_image` accept an
   ``httpx.AsyncClient`` (the addon passes the shared browser-fingerprinted
   curl-cffi client) so the cookie jar + Cloudflare clearance are reused across
   every side trip. The final presigned download OMITS ``Authorization`` (a
   Bearer there triggers a 403 — gproxy ``image.rs`` comment).

Ported from the MIT-licensed gproxy chatgpt channel (``image.rs`` /
``image_edit.rs`` / ``channel.rs``), cross-checked against aurora and pro-cli.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from ccproxy.lightllm.adapters.openai_conversations import build_conversation_body
from ccproxy.openai_conversations.image_parse import ParsedImageEdit

__all__ = [
    "ImageGenerationError",
    "ImagePointer",
    "UploadedImage",
    "attach_uploaded_image",
    "build_image_conversation_body",
    "build_images_response",
    "download_image",
    "extract_pointers_from_conversation",
    "extract_pointers_from_sse",
    "image_model_slug",
    "poll_conversation_for_pointers",
    "upload_image",
]

_IMAGE_HINT = "picture_v2"
"""ChatGPT system hint that routes a conversation turn to image generation."""

_DOWNLOAD_ACCEPT = "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
_POINTER_RE = re.compile(r"(?:sediment|file-service)://[A-Za-z0-9_.\-]+")
_SEDIMENT_SCHEME = "sediment://"
_FILE_SERVICE_SCHEME = "file-service://"


class ImageGenerationError(RuntimeError):
    """Raised when an image generation / edit side trip fails terminally.

    The addon converts this into an OpenAI-shape error response so the client
    never receives a raw ChatGPT SSE body.
    """


@dataclass(frozen=True)
class ImagePointer:
    """A resolved image asset pointer with its scheme tag preserved."""

    file_id: str
    """Bare file id (the ``scheme://`` prefix stripped)."""

    is_sediment: bool
    """``True`` for ``sediment://`` (generated / uploaded), ``False`` for
    ``file-service://`` (older flows). Selects the download metadata route."""


@dataclass(frozen=True)
class UploadedImage:
    """Result of a 3-step ChatGPT file upload (image edit source image)."""

    file_id: str
    size_bytes: int
    width: int
    height: int
    filename: str
    mime_type: str


# ---------------------------------------------------------------------------
# Pure body building
# ---------------------------------------------------------------------------


def image_model_slug(model: str) -> str:
    """Map an OpenAI image model slug to a chatgpt.com conversation model.

    ``gpt-image-*`` / ``dall-e-*`` (and the empty default) become ``"auto"``
    so chatgpt.com routes its own image pipeline (aurora ``imageModelSlug``);
    any other slug passes through unchanged.
    """
    m = (model or "").strip()
    if not m or m.startswith("dall-e") or m.startswith("gpt-image"):
        return "auto"
    return m


def build_image_conversation_body(
    *,
    prompt: str,
    model: str,
    conversation_id: str = "",
    parent_message_id: str = "",
    is_continuation: bool = False,
    uploaded_image: UploadedImage | None = None,
) -> dict[str, Any]:
    """Build the ``/backend-api/f/conversation`` body for an image request.

    Forces ``system_hints: ["picture_v2"]`` (top-level + first-message
    metadata), a saved conversation (``temporary_chat=False``), and the
    aurora-mapped model slug. The prompt is passed through verbatim.

    Args:
        prompt: The user's image prompt (used verbatim).
        model: The client-requested model slug (mapped via
            :func:`image_model_slug`).
        conversation_id: ChatGPT conversation id to continue (continuation only).
        parent_message_id: Parent message id (continuation only).
        is_continuation: Whether this continues an existing conversation.
        uploaded_image: When set (image edit), attaches the uploaded source
            image to the user message as a ``sediment://`` pointer.

    Returns:
        Dict ready for ``json.dumps``.
    """
    messages_ir: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=prompt)])]
    body = build_conversation_body(
        messages_ir=messages_ir,
        model=image_model_slug(model),
        system_hints=[_IMAGE_HINT],
        temporary_chat=False,
        conversation_id=conversation_id,
        parent_message_id=parent_message_id,
        is_continuation=is_continuation,
    )

    # aurora double-placement: also stamp the hint into the first message metadata.
    messages = body.get("messages")
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        metadata = messages[0].setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["system_hints"] = [_IMAGE_HINT]

    if uploaded_image is not None:
        attach_uploaded_image(body, prompt=prompt, uploaded=uploaded_image)
    return body


def attach_uploaded_image(body: dict[str, Any], *, prompt: str, uploaded: UploadedImage) -> None:
    """Convert the first user message into a ``multimodal_text`` with an image.

    Mutates ``body`` in place: the message content becomes
    ``[image_asset_pointer, prompt_text]`` and an ``attachments`` entry is added
    to the message metadata (gproxy ``attach_uploaded_image``).
    """
    messages = body.get("messages")
    if not (isinstance(messages, list) and messages and isinstance(messages[0], dict)):
        return
    message = messages[0]
    asset = {
        "content_type": "image_asset_pointer",
        "asset_pointer": f"{_SEDIMENT_SCHEME}{uploaded.file_id}",
        "size_bytes": uploaded.size_bytes,
        "width": uploaded.width,
        "height": uploaded.height,
    }
    message["content"] = {"content_type": "multimodal_text", "parts": [asset, prompt]}
    metadata = message.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["attachments"] = [
            {
                "id": uploaded.file_id,
                "size": uploaded.size_bytes,
                "name": uploaded.filename,
                "mime_type": uploaded.mime_type,
                "width": uploaded.width,
                "height": uploaded.height,
                "source": "library",
                "is_big_paste": False,
            }
        ]


# ---------------------------------------------------------------------------
# Pure extraction / output
# ---------------------------------------------------------------------------


def extract_pointers_from_sse(raw_body: bytes) -> list[ImagePointer]:
    """Scan a raw ``/f/conversation`` SSE body for image asset pointers.

    A single regex pass over the decoded body recovers both ``asset_pointer``
    field values and inline ``file-service://`` / ``sediment://`` references
    that appear in text deltas. Pointers are de-duplicated by file id in order
    of first appearance.
    """
    text = raw_body.decode("utf-8", errors="replace")
    return _dedup_pointers(_POINTER_RE.findall(text))


def extract_pointers_from_conversation(conversation: dict[str, Any]) -> list[ImagePointer]:
    """Recover image pointers from a polled conversation object.

    Walks ``mapping`` for ``tool`` messages whose metadata marks an
    ``image_gen`` async task and whose content is ``multimodal_text`` (gproxy's
    precise filter). Falls back to a generic pointer scan when the strict walk
    finds nothing (pro-cli / aurora style).
    """
    raws: list[str] = []
    mapping = conversation.get("mapping")
    if isinstance(mapping, dict):
        for node in mapping.values():
            if not isinstance(node, dict):
                continue
            message = node.get("message")
            if not isinstance(message, dict):
                continue
            author = message.get("author")
            if not isinstance(author, dict) or author.get("role") != "tool":
                continue
            metadata = message.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("async_task_type") != "image_gen":
                continue
            content = message.get("content")
            if not isinstance(content, dict) or content.get("content_type") != "multimodal_text":
                continue
            for part in content.get("parts") or []:
                if isinstance(part, dict):
                    pointer = part.get("asset_pointer")
                    if isinstance(pointer, str):
                        raws.append(pointer)

    pointers = _dedup_pointers(raws)
    if pointers:
        return pointers
    return _dedup_pointers(_POINTER_RE.findall(json.dumps(conversation)))


def build_images_response(images: list[bytes]) -> dict[str, Any]:
    """Package downloaded image bytes into an OpenAI ``images.response``.

    ``revised_prompt`` is ``null`` (OpenAI spec shape); chatgpt.com does not
    return a structured revised prompt.
    """
    return {
        "created": int(time.time()),
        "data": [{"b64_json": base64.b64encode(image).decode("ascii"), "revised_prompt": None} for image in images],
    }


def _normalize_pointer(raw: str) -> ImagePointer | None:
    if raw.startswith(_SEDIMENT_SCHEME):
        return ImagePointer(file_id=raw[len(_SEDIMENT_SCHEME) :], is_sediment=True)
    if raw.startswith(_FILE_SERVICE_SCHEME):
        return ImagePointer(file_id=raw[len(_FILE_SERVICE_SCHEME) :], is_sediment=False)
    return None


def _dedup_pointers(raws: list[str]) -> list[ImagePointer]:
    seen: set[str] = set()
    result: list[ImagePointer] = []
    for raw in raws:
        pointer = _normalize_pointer(raw)
        if pointer is None or pointer.file_id in seen:
            continue
        seen.add(pointer.file_id)
        result.append(pointer)
    return result


# ---------------------------------------------------------------------------
# Injected-client network helpers
# ---------------------------------------------------------------------------


def _auth_headers(*, access_token: str, device_id: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    if access_token:
        headers["authorization"] = f"Bearer {access_token}"
    if device_id:
        headers["oai-device-id"] = device_id
    return headers


async def upload_image(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    access_token: str,
    device_id: str,
    parsed: ParsedImageEdit,
    timeout: float,
) -> UploadedImage:
    """Execute the 3-step ChatGPT file upload for an image edit source image.

    1. ``POST /backend-api/files`` → ``{upload_url, file_id}``.
    2. ``PUT <upload_url>`` (Azure blob) with the raw bytes.
    3. ``POST /backend-api/files/process_upload_stream`` to activate; falls back
       to ``POST /backend-api/files/{file_id}/uploaded`` on 404/405 (aurora API
       variant — divergence hedge).

    Raises:
        ImageGenerationError: Any step fails or returns no upload target.
    """
    json_headers = {**_auth_headers(access_token=access_token, device_id=device_id), "content-type": "application/json"}

    create = await client.post(
        f"{base_url}/backend-api/files",
        content=json.dumps(
            {
                "file_name": parsed.filename,
                "file_size": parsed.size_bytes,
                "use_case": "multimodal",
                "timezone_offset_min": -480,
                "reset_rate_limits": False,
                "store_in_library": True,
                "library_persistence_mode": "opportunistic",
            }
        ).encode(),
        headers=json_headers,
        timeout=timeout,
    )
    if create.status_code not in (200, 201):
        raise ImageGenerationError(f"file upload create failed: HTTP {create.status_code}")
    create_data = create.json()
    upload_url = str(create_data.get("upload_url") or "")
    file_id = str(create_data.get("file_id") or "")
    if not upload_url or not file_id:
        raise ImageGenerationError("file upload create returned no upload_url/file_id")

    put = await client.put(
        upload_url,
        content=parsed.image_bytes,
        headers={"content-type": parsed.mime_type, "x-ms-blob-type": "BlockBlob"},
        timeout=timeout,
    )
    if put.status_code not in (200, 201):
        raise ImageGenerationError(f"blob upload PUT failed: HTTP {put.status_code}")

    activate = await client.post(
        f"{base_url}/backend-api/files/process_upload_stream",
        content=json.dumps(
            {
                "file_id": file_id,
                "use_case": "multimodal",
                "index_for_retrieval": False,
                "file_name": parsed.filename,
                "library_persistence_mode": "opportunistic",
                "metadata": {"store_in_library": True},
            }
        ).encode(),
        headers=json_headers,
        timeout=timeout,
    )
    if activate.status_code in (404, 405):
        activate = await client.post(
            f"{base_url}/backend-api/files/{file_id}/uploaded",
            content=b"{}",
            headers=json_headers,
            timeout=timeout,
        )
    if activate.status_code not in (200, 201):
        raise ImageGenerationError(f"file activation failed: HTTP {activate.status_code}")

    return UploadedImage(
        file_id=file_id,
        size_bytes=parsed.size_bytes,
        width=parsed.width,
        height=parsed.height,
        filename=parsed.filename,
        mime_type=parsed.mime_type,
    )


async def poll_conversation_for_pointers(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    conversation_id: str,
    access_token: str,
    device_id: str,
    interval_seconds: float,
    max_attempts: int,
    timeout: float,
) -> list[ImagePointer]:
    """Poll ``GET /backend-api/conversation/{id}`` until image pointers appear.

    Raises:
        ImageGenerationError: ``async_status == 4`` (finished without assets) or
            the deadline (``interval_seconds`` times ``max_attempts``) elapses.
    """
    headers = {**_auth_headers(access_token=access_token, device_id=device_id), "accept": "application/json"}
    url = f"{base_url}/backend-api/conversation/{conversation_id}"

    for attempt in range(max_attempts):
        resp = await client.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            conversation = resp.json()
            if isinstance(conversation, dict):
                if conversation.get("async_status") == 4:
                    raise ImageGenerationError("image generation finished without assets (async_status=4)")
                pointers = extract_pointers_from_conversation(conversation)
                if pointers:
                    return pointers
        if attempt < max_attempts - 1:
            await asyncio.sleep(interval_seconds)

    raise ImageGenerationError(f"image generation timed out after {max_attempts} polls")


async def download_image(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    pointer: ImagePointer,
    conversation_id: str,
    access_token: str,
    device_id: str,
    timeout: float,
) -> bytes:
    """Two-step image download: authenticated metadata then presigned bytes.

    The presigned ``download_url`` GET OMITS ``Authorization`` — the URL carries
    its own ``sig=`` and a Bearer there yields a 403 (gproxy ``image.rs``).

    Raises:
        ImageGenerationError: The metadata URL cannot be resolved or the
            presigned download fails.
    """
    metadata_headers = {
        **_auth_headers(access_token=access_token, device_id=device_id),
        "accept": "application/json",
    }
    download_url = await _fetch_download_url(
        client=client,
        pointer=pointer,
        conversation_id=conversation_id,
        base_url=base_url,
        headers=metadata_headers,
        timeout=timeout,
    )

    resp = await client.get(
        download_url,
        headers={"accept": _DOWNLOAD_ACCEPT, "referer": f"{base_url}/"},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise ImageGenerationError(f"presigned image download failed: HTTP {resp.status_code}")
    return resp.content


def _metadata_urls(*, base_url: str, pointer: ImagePointer, conversation_id: str) -> list[str]:
    """Candidate metadata endpoints for a pointer (primary then fallback)."""
    file_id = pointer.file_id
    if pointer.is_sediment:
        return [
            f"{base_url}/backend-api/conversation/{conversation_id}/attachment/{file_id}/download",
            f"{base_url}/backend-api/files/download/{file_id}",
        ]
    return [f"{base_url}/backend-api/files/download/{file_id}?conversation_id={conversation_id}&inline=false"]


async def _fetch_download_url(
    *,
    client: httpx.AsyncClient,
    pointer: ImagePointer,
    conversation_id: str,
    base_url: str,
    headers: dict[str, str],
    timeout: float,
) -> str:
    last_status = 0
    for url in _metadata_urls(base_url=base_url, pointer=pointer, conversation_id=conversation_id):
        resp = await client.get(url, headers=headers, timeout=timeout)
        last_status = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                download_url = data.get("download_url") or data.get("url")
                if isinstance(download_url, str) and download_url:
                    return download_url
        if resp.status_code != 404:
            break
    raise ImageGenerationError(f"could not resolve image download_url (last HTTP {last_status})")
