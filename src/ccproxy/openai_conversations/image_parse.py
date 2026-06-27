"""Pure parsing for OpenAI image-generation and image-edit requests.

Ported from the MIT-licensed gproxy chatgpt channel
(``sdk/gproxy-channel/src/channels/chatgpt/image_edit.rs``). No network, no
mutation of inputs — every function returns a frozen dataclass so the
``OpenAIConversationsAddon`` image branch and its tests can exercise parsing
in isolation.

Two request shapes are recognised:

* ``POST /v1/images/generations`` — JSON ``{prompt, model, n, size}``.
* ``POST /v1/images/edits`` — either ``multipart/form-data`` (the OpenAI SDK
  default) or a JSON body carrying a ``data:`` URL. Remote ``http(s)://`` image
  URLs are rejected (out of scope for CHATGPT-006).

Image dimensions are probed from the raw bytes (PNG / JPEG / GIF headers) with
a ``(1024, 1024)`` fallback, mirroring gproxy ``probe_png_dimensions``.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
from dataclasses import dataclass

__all__ = [
    "ParsedImageEdit",
    "ParsedImageGen",
    "RemoteImageURLError",
    "guess_mime_from_name",
    "mime_to_ext",
    "parse_image_edit_request",
    "parse_image_gen_request",
    "probe_image_dimensions",
]

_DEFAULT_DIMENSIONS: tuple[int, int] = (1024, 1024)
_IMAGE_FIELD_NAMES: frozenset[str] = frozenset({"image", "image[]", "image[0]"})

_MIME_BY_EXT: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}
_EXT_BY_MIME: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}


class RemoteImageURLError(ValueError):
    """Raised when an image-edit body references a remote ``http(s)://`` URL.

    Remote image ingestion is explicitly out of scope (issue CHATGPT-006); the
    caller must supply raw bytes (multipart) or a ``data:`` URL.
    """


@dataclass(frozen=True)
class ParsedImageGen:
    """Normalised ``/v1/images/generations`` request."""

    prompt: str
    model: str = ""
    n: int = 1
    size: str = ""


@dataclass(frozen=True)
class ParsedImageEdit:
    """Normalised ``/v1/images/edits`` request with decoded image bytes."""

    prompt: str
    image_bytes: bytes
    filename: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    model: str = ""
    n: int = 1
    size: str = ""


def parse_image_gen_request(body: bytes) -> ParsedImageGen:
    """Parse a JSON ``/v1/images/generations`` body.

    Args:
        body: Raw request bytes (OpenAI Images API JSON).

    Returns:
        The normalised :class:`ParsedImageGen`.

    Raises:
        ValueError: ``body`` is not valid JSON or carries no ``prompt``.
    """
    try:
        data = json.loads(body or b"{}")
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"image generation body is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("image generation body must be a JSON object")

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("image generation body missing a non-empty 'prompt'")

    return ParsedImageGen(
        prompt=prompt,
        model=str(data.get("model") or ""),
        n=int(data.get("n") or 1),
        size=str(data.get("size") or ""),
    )


def parse_image_edit_request(*, body: bytes, content_type: str) -> ParsedImageEdit:
    """Parse a ``/v1/images/edits`` body (multipart or JSON data-URL).

    Args:
        body: Raw request bytes.
        content_type: The request ``Content-Type`` header value.

    Returns:
        The normalised :class:`ParsedImageEdit` with decoded ``image_bytes``
        and probed dimensions.

    Raises:
        RemoteImageURLError: The image reference is a remote ``http(s)://`` URL.
        ValueError: The body is malformed or carries no image part.
    """
    if _is_multipart(body=body, content_type=content_type):
        prompt, image_bytes, filename, mime_type, model, n, size = _parse_multipart(body)
    else:
        prompt, image_bytes, filename, mime_type, model, n, size = _parse_json_data_url(body)

    if not image_bytes:
        raise ValueError("image edit body carried no image bytes")

    width, height = probe_image_dimensions(image_bytes)
    return ParsedImageEdit(
        prompt=prompt,
        image_bytes=image_bytes,
        filename=filename,
        mime_type=mime_type,
        width=width,
        height=height,
        size_bytes=len(image_bytes),
        model=model,
        n=n,
        size=size,
    )


def guess_mime_from_name(name: str) -> str:
    """Return a MIME type for ``name`` by extension, defaulting to PNG."""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return _MIME_BY_EXT.get(ext, "image/png")


def mime_to_ext(mime_type: str) -> str:
    """Return a file extension for ``mime_type``, defaulting to ``bin``."""
    return _EXT_BY_MIME.get(mime_type.split(";")[0].strip().lower(), "bin")


def probe_image_dimensions(data: bytes) -> tuple[int, int]:
    """Best-effort ``(width, height)`` from PNG / JPEG / GIF headers.

    Falls back to ``(1024, 1024)`` when the format is unknown or truncated.
    chatgpt.com re-reads the real dimensions server-side; sending plausible
    values mirrors browser behaviour for the ``image_asset_pointer`` part.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        if width and height:
            return width, height
    elif data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        width = int.from_bytes(data[6:8], "little")
        height = int.from_bytes(data[8:10], "little")
        if width and height:
            return width, height
    elif data[:2] == b"\xff\xd8":
        probed = _probe_jpeg(data)
        if probed is not None:
            return probed
    return _DEFAULT_DIMENSIONS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_multipart(*, body: bytes, content_type: str) -> bool:
    """Detect a multipart body (gproxy ``is_multipart`` + content-type check)."""
    if "multipart/form-data" in content_type.lower():
        return True
    return body.startswith(b"--") and any(b in (0x0D, 0x0A) for b in body[:256])


def _parse_multipart(body: bytes) -> tuple[str, bytes, str, str, str, int, str]:
    """Parse a ``multipart/form-data`` edit body into its fields.

    Returns ``(prompt, image_bytes, filename, mime_type, model, n, size)``.
    Faithful port of gproxy ``parse_multipart`` — operates on bytes throughout
    so binary image data is never corrupted by text decoding.
    """
    newline = body.find(b"\n")
    if newline == -1:
        raise ValueError("multipart: missing first newline")
    first_line = body[:newline].rstrip(b"\r")
    if not first_line.startswith(b"--"):
        raise ValueError("multipart: first line does not start with --")
    boundary = first_line[2:]
    if not boundary:
        raise ValueError("multipart: empty boundary")

    separator = b"\r\n--" + boundary
    rest = body[newline + 1 :]

    image_bytes: bytes | None = None
    filename: str | None = None
    mime_type: str | None = None
    prompt = ""
    model = ""
    n = 1
    size = ""

    while True:
        end = rest.find(separator)
        if end == -1:
            raise ValueError("multipart: trailing boundary not found")
        part = rest[:end]
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            raise ValueError("multipart: part header/body separator missing")
        name, file_name, content_type = _parse_part_headers(part[:header_end])
        part_body = part[header_end + 4 :]

        if name in _IMAGE_FIELD_NAMES:
            image_bytes = part_body
            if file_name:
                filename = file_name
            if content_type:
                mime_type = content_type
        elif name == "prompt":
            prompt = part_body.decode("utf-8", errors="replace")
        elif name == "model":
            model = part_body.decode("utf-8", errors="replace").strip()
        elif name == "n":
            with contextlib.suppress(ValueError):
                n = int(part_body.decode("utf-8", errors="replace").strip())
        elif name == "size":
            size = part_body.decode("utf-8", errors="replace").strip()

        after = rest[end + len(separator) :]
        if after.startswith(b"--"):
            break
        rest = after[2:] if after.startswith(b"\r\n") else after

    if image_bytes is None:
        raise ValueError("multipart: missing image part")
    resolved_name = filename or "image.png"
    resolved_mime = mime_type or guess_mime_from_name(resolved_name)
    return prompt, image_bytes, resolved_name, resolved_mime, model, n, size


def _parse_part_headers(raw: bytes) -> tuple[str, str, str]:
    """Extract ``(name, filename, content_type)`` from a multipart part header."""
    name = ""
    filename = ""
    content_type = ""
    for raw_line in raw.split(b"\n"):
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r").strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("content-disposition:"):
            for token in line.split(";"):
                token = token.strip()
                if token.startswith("name="):
                    name = _trim_quotes(token[len("name=") :])
                elif token.startswith("filename="):
                    filename = _trim_quotes(token[len("filename=") :])
        elif lowered.startswith("content-type:"):
            content_type = line[len("content-type:") :].strip()
    return name, filename, content_type


def _parse_json_data_url(body: bytes) -> tuple[str, bytes, str, str, str, int, str]:
    """Parse a JSON edit body carrying a ``data:`` image URL.

    Returns ``(prompt, image_bytes, filename, mime_type, model, n, size)``.

    Raises:
        RemoteImageURLError: The image reference is ``http(s)://``.
        ValueError: The body is malformed or the reference is not a data URL.
    """
    try:
        data = json.loads(body or b"{}")
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"image edit body is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("image edit body must be a JSON object")

    image_ref = _extract_image_ref(data)
    if image_ref is None:
        raise ValueError("image edit body missing 'image' / 'image_url'")
    if image_ref.startswith(("http://", "https://")):
        raise RemoteImageURLError("remote http(s) image URLs are not supported; supply image bytes or a data: URL")
    if not image_ref.startswith("data:"):
        raise ValueError("image edit 'image' must be a data: URL or multipart upload")

    mime_type, image_bytes = _decode_data_url(image_ref)
    filename = f"image.{mime_to_ext(mime_type)}"
    prompt = str(data.get("prompt") or "")
    model = str(data.get("model") or "")
    n = int(data.get("n") or 1)
    size = str(data.get("size") or "")
    return prompt, image_bytes, filename, mime_type, model, n, size


def _extract_image_ref(data: dict[str, object]) -> str | None:
    """Find the image reference string in a JSON edit body."""
    direct = data.get("image")
    if isinstance(direct, str):
        return direct
    images = data.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict):
            url = first.get("image_url")
            if isinstance(url, str):
                return url
    url = data.get("image_url")
    if isinstance(url, str):
        return url
    return None


def _decode_data_url(url: str) -> tuple[str, bytes]:
    """Decode ``data:<mime>;base64,<payload>`` into ``(mime, bytes)``."""
    if "," not in url:
        raise ValueError("malformed data URL: missing payload separator")
    header, payload = url.split(",", 1)
    meta = header[len("data:") :]
    mime_type = meta.split(";", 1)[0].strip() or "image/png"
    try:
        image_bytes = base64.b64decode(payload, validate=False)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"malformed data URL base64 payload: {exc}") from exc
    return mime_type, image_bytes


def _probe_jpeg(data: bytes) -> tuple[int, int] | None:
    """Walk JPEG markers for the first SOF segment's ``(width, height)``."""
    i = 2
    n = len(data)
    _sof_markers = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in _sof_markers:
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            if width and height:
                return width, height
            return None
        if marker == 0x01 or 0xD0 <= marker <= 0xD9:
            i += 2
            continue
        seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
        if seg_len < 2:
            return None
        i += 2 + seg_len
    return None


def _trim_quotes(value: str) -> str:
    return value.strip().strip('"')  # type: ignore[arg-type]
