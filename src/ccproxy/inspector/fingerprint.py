"""TLS ClientHello fingerprint parsing and curl-cffi replay specs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from curl_cffi.const import CurlHttpVersion, CurlOpt

GREASE_VALUES: frozenset[int] = frozenset(
    {
        0x0A0A,
        0x1A1A,
        0x2A2A,
        0x3A3A,
        0x4A4A,
        0x5A5A,
        0x6A6A,
        0x7A7A,
        0x8A8A,
        0x9A9A,
        0xAAAA,
        0xBABA,
        0xCACA,
        0xDADA,
        0xEAEA,
        0xFAFA,
    }
)

CLIENT_FINGERPRINT_METADATA = "ccproxy.fingerprint.client"
REPLAY_FINGERPRINT_METADATA = "ccproxy.fingerprint.profile"
LEGACY_CLIENT_FINGERPRINT_METADATA = "ccproxy.client_fingerprint"

_TLS_VERSION_LABELS = {
    0x0304: "13",
    0x0303: "12",
    0x0302: "11",
    0x0301: "10",
    0x0300: "s3",
}

_HTTP_VERSION_VALUES = {
    "v1_0": CurlHttpVersion.V1_0,
    "v1_1": CurlHttpVersion.V1_1,
    "v2": CurlHttpVersion.V2_0,
}

_SIGNATURE_ALGORITHM_NAMES = {
    "0201": "rsa_pkcs1_sha1",
    "0203": "ecdsa_sha1",
    "0401": "rsa_pkcs1_sha256",
    "0403": "ecdsa_secp256r1_sha256",
    "0501": "rsa_pkcs1_sha384",
    "0503": "ecdsa_secp384r1_sha384",
    "0601": "rsa_pkcs1_sha512",
    "0603": "ecdsa_secp521r1_sha512",
    "0804": "rsa_pss_rsae_sha256",
    "0805": "rsa_pss_rsae_sha384",
    "0806": "rsa_pss_rsae_sha512",
    "0807": "ed25519",
    "0808": "ed448",
    "0809": "rsa_pss_pss_sha256",
    "080a": "rsa_pss_pss_sha384",
    "080b": "rsa_pss_pss_sha512",
}


@dataclass(frozen=True)
class CapturedFingerprint:
    """Shape-captured TLS profile used to replay a native client fingerprint."""

    schema_version: int
    source: str
    captured_at: str
    sni: str | None
    alpn_protocols: tuple[str, ...]
    legacy_version: int
    supported_versions: tuple[str, ...]
    cipher_suites: tuple[str, ...]
    extensions: tuple[str, ...]
    supported_groups: tuple[str, ...]
    ec_point_formats: tuple[str, ...]
    signature_algorithms: tuple[str, ...]
    signature_algorithm_names: tuple[str, ...]
    ja3: str
    ja3_full: str
    ja4: str
    ja4_r: str
    http_version: str
    provider: str | None = None
    user_agent: str | None = None
    runtime_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "captured_at": self.captured_at,
            "sni": self.sni,
            "alpn_protocols": list(self.alpn_protocols),
            "legacy_version": self.legacy_version,
            "supported_versions": list(self.supported_versions),
            "cipher_suites": list(self.cipher_suites),
            "extensions": list(self.extensions),
            "supported_groups": list(self.supported_groups),
            "ec_point_formats": list(self.ec_point_formats),
            "signature_algorithms": list(self.signature_algorithms),
            "signature_algorithm_names": list(self.signature_algorithm_names),
            "ja3": self.ja3,
            "ja3_full": self.ja3_full,
            "ja4": self.ja4,
            "ja4_r": self.ja4_r,
            "http_version": self.http_version,
            "provider": self.provider,
            "user_agent": self.user_agent,
            "runtime_version": self.runtime_version,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CapturedFingerprint:
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            source=str(raw.get("source", "")),
            captured_at=str(raw.get("captured_at", "")),
            sni=raw.get("sni"),
            alpn_protocols=tuple(str(x) for x in raw.get("alpn_protocols", [])),
            legacy_version=int(raw.get("legacy_version", 0)),
            supported_versions=tuple(str(x) for x in raw.get("supported_versions", [])),
            cipher_suites=tuple(str(x) for x in raw.get("cipher_suites", [])),
            extensions=tuple(str(x) for x in raw.get("extensions", [])),
            supported_groups=tuple(str(x) for x in raw.get("supported_groups", [])),
            ec_point_formats=tuple(str(x) for x in raw.get("ec_point_formats", [])),
            signature_algorithms=tuple(str(x) for x in raw.get("signature_algorithms", [])),
            signature_algorithm_names=tuple(str(x) for x in raw.get("signature_algorithm_names", [])),
            ja3=str(raw.get("ja3", "")),
            ja3_full=str(raw.get("ja3_full", "")),
            ja4=str(raw.get("ja4", "")),
            ja4_r=str(raw.get("ja4_r", "")),
            http_version=str(raw.get("http_version", "v1_1")),
            provider=raw.get("provider"),
            user_agent=raw.get("user_agent"),
            runtime_version=raw.get("runtime_version"),
        )

    @property
    def transport_cache_key(self) -> str:
        doc = {
            "http_version": self.http_version,
            "ja3_full": self.ja3_full,
            "signature_algorithm_names": list(self.signature_algorithm_names),
        }
        return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]

    def transport_kwargs(self) -> dict[str, Any]:
        # libcurl decodes Content-Encoding (gzip/br/zstd/deflate) so callers and
        # the sidecar receive decoded bytes. The Accept-Encoding request header
        # still goes out on the wire via CURLOPT_ACCEPT_ENCODING, preserving the
        # browser fingerprint.
        curl_options: dict[CurlOpt, Any] = {}
        if self.signature_algorithm_names:
            curl_options[CurlOpt.SSL_SIG_HASH_ALGS] = ",".join(self.signature_algorithm_names)
        return {
            "ja3": self.ja3_full,
            "http_version": _HTTP_VERSION_VALUES.get(self.http_version, CurlHttpVersion.V1_1),
            "curl_options": curl_options,
        }

    def with_request_context(self, *, provider: str, user_agent: str, runtime_version: str) -> CapturedFingerprint:
        raw = self.to_dict()
        raw.update(
            {
                "provider": provider,
                "user_agent": user_agent or None,
                "runtime_version": runtime_version or None,
            }
        )
        return CapturedFingerprint.from_dict(raw)


def parse_client_hello_bytes(raw: bytes, *, source: str = "mitmproxy_tls_clienthello") -> CapturedFingerprint:
    """Parse a TLS ClientHello into JA3/JA4 material.

    ``raw`` may be a bare ClientHello body, a Handshake record, or a TLS record.
    mitmproxy's ``ClientHello.raw_bytes(wrap_in_record=False)`` returns the
    bare body, which is the normal runtime input.
    """
    body = _unwrap_client_hello(raw)
    if len(body) < 42:
        raise ValueError("ClientHello too short")

    offset = 0
    legacy_version = _read_u16(body, offset)
    offset += 2 + 32

    session_len = body[offset]
    offset += 1 + session_len

    cipher_len = _read_u16(body, offset)
    offset += 2
    cipher_bytes = body[offset : offset + cipher_len]
    offset += cipher_len
    ciphers = _u16_list(cipher_bytes)

    compression_len = body[offset]
    offset += 1 + compression_len

    extensions: list[tuple[int, bytes]] = []
    if offset + 2 <= len(body):
        extensions_len = _read_u16(body, offset)
        offset += 2
        end = min(offset + extensions_len, len(body))
        while offset + 4 <= end:
            ext_type = _read_u16(body, offset)
            ext_len = _read_u16(body, offset + 2)
            offset += 4
            ext_body = body[offset : offset + ext_len]
            offset += ext_len
            extensions.append((ext_type, ext_body))

    ext_map = dict(extensions)
    supported_groups = _parse_u16_vector(ext_map.get(10, b""), width_bytes=2)
    ec_point_formats = _parse_u8_vector(ext_map.get(11, b""))
    signature_algorithms = _parse_u16_vector(ext_map.get(13, b""), width_bytes=2)
    supported_versions = _parse_u16_vector(ext_map.get(43, b""), width_bytes=1)
    alpn_protocols = _parse_alpn(ext_map.get(16, b""))
    sni = _parse_sni(ext_map.get(0, b""))

    ja3_full = _ja3_full(
        legacy_version=legacy_version,
        ciphers=ciphers,
        extensions=extensions,
        supported_groups=supported_groups,
        ec_point_formats=ec_point_formats,
    )
    ja3 = hashlib.md5(ja3_full.encode(), usedforsecurity=False).hexdigest()

    ja4, ja4_r = _ja4(
        legacy_version=legacy_version,
        ciphers=ciphers,
        extensions=extensions,
        supported_versions=supported_versions,
        alpn_protocols=alpn_protocols,
        signature_algorithms=signature_algorithms,
        has_sni=sni is not None,
    )
    sig_hex = tuple(_hex4(v) for v in signature_algorithms if not _is_grease(v))
    sig_names = tuple(_SIGNATURE_ALGORITHM_NAMES[v] for v in sig_hex if v in _SIGNATURE_ALGORITHM_NAMES)
    first_alpn = alpn_protocols[0] if alpn_protocols else ""
    http_version = "v2" if first_alpn == "h2" else "v1_1"

    return CapturedFingerprint(
        schema_version=1,
        source=source,
        captured_at=datetime.now(UTC).isoformat(),
        sni=sni,
        alpn_protocols=tuple(alpn_protocols),
        legacy_version=legacy_version,
        supported_versions=tuple(_hex4(v) for v in supported_versions if not _is_grease(v)),
        cipher_suites=tuple(_hex4(v) for v in ciphers if not _is_grease(v)),
        extensions=tuple(_hex4(v) for v, _ in extensions if not _is_grease(v)),
        supported_groups=tuple(_hex4(v) for v in supported_groups if not _is_grease(v)),
        ec_point_formats=tuple(f"{v:02x}" for v in ec_point_formats),
        signature_algorithms=sig_hex,
        signature_algorithm_names=sig_names,
        ja3=ja3,
        ja3_full=ja3_full,
        ja4=ja4,
        ja4_r=ja4_r,
        http_version=http_version,
    )


def _unwrap_client_hello(raw: bytes) -> bytes:
    if len(raw) >= 9 and raw[0] == 0x16:
        pos = 5
        if raw[pos] == 0x01:
            size = int.from_bytes(raw[pos + 1 : pos + 4], "big")
            return raw[pos + 4 : pos + 4 + size]
    if len(raw) >= 4 and raw[0] == 0x01:
        size = int.from_bytes(raw[1:4], "big")
        return raw[4 : 4 + size]
    return raw


def _read_u16(buf: bytes, offset: int) -> int:
    return int.from_bytes(buf[offset : offset + 2], "big")


def _u16_list(buf: bytes) -> list[int]:
    return [_read_u16(buf, i) for i in range(0, len(buf) - 1, 2)]


def _parse_u16_vector(buf: bytes, *, width_bytes: int) -> list[int]:
    if len(buf) < width_bytes:
        return []
    size = int.from_bytes(buf[:width_bytes], "big")
    data = buf[width_bytes : width_bytes + size]
    return _u16_list(data)


def _parse_u8_vector(buf: bytes) -> list[int]:
    if not buf:
        return []
    size = buf[0]
    return list(buf[1 : 1 + size])


def _parse_alpn(buf: bytes) -> list[str]:
    if len(buf) < 2:
        return []
    size = _read_u16(buf, 0)
    data = buf[2 : 2 + size]
    out: list[str] = []
    offset = 0
    while offset < len(data):
        item_len = data[offset]
        offset += 1
        item = data[offset : offset + item_len]
        offset += item_len
        out.append(item.decode("ascii", errors="replace"))
    return out


def _parse_sni(buf: bytes) -> str | None:
    if len(buf) < 5:
        return None
    list_len = _read_u16(buf, 0)
    offset = 2
    end = min(2 + list_len, len(buf))
    while offset + 3 <= end:
        name_type = buf[offset]
        name_len = _read_u16(buf, offset + 1)
        offset += 3
        name = buf[offset : offset + name_len]
        offset += name_len
        if name_type == 0:
            return name.decode("ascii", errors="replace")
    return None


def _is_grease(value: int) -> bool:
    return value in GREASE_VALUES


def _decimal_segment(values: list[int]) -> str:
    return "-".join(str(v) for v in values if not _is_grease(v))


def _ja3_full(
    *,
    legacy_version: int,
    ciphers: list[int],
    extensions: list[tuple[int, bytes]],
    supported_groups: list[int],
    ec_point_formats: list[int],
) -> str:
    return ",".join(
        [
            str(legacy_version),
            _decimal_segment(ciphers),
            _decimal_segment([ext_type for ext_type, _ in extensions]),
            _decimal_segment(supported_groups),
            _decimal_segment(ec_point_formats),
        ]
    )


def _hex4(value: int) -> str:
    return f"{value:04x}"


def _sha12(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def _alpn_code(values: list[str]) -> str:
    if not values:
        return "00"
    value = values[0]
    if not value:
        return "00"
    first = value[0]
    last = value[-1]
    if first.isalnum() and last.isalnum():
        return f"{first}{last}"
    raw = value.encode()
    return f"{raw[0]:02x}"[0] + f"{raw[-1]:02x}"[-1]


def _version_code(legacy_version: int, supported_versions: list[int]) -> str:
    candidates = [v for v in supported_versions if not _is_grease(v)]
    version = max(candidates) if candidates else legacy_version
    return _TLS_VERSION_LABELS.get(version, "00")


def _ja4(
    *,
    legacy_version: int,
    ciphers: list[int],
    extensions: list[tuple[int, bytes]],
    supported_versions: list[int],
    alpn_protocols: list[str],
    signature_algorithms: list[int],
    has_sni: bool,
) -> tuple[str, str]:
    clean_ciphers = sorted(_hex4(v) for v in ciphers if not _is_grease(v))
    clean_extensions = [_hex4(v) for v, _ in extensions if not _is_grease(v)]
    sorted_extensions = sorted(v for v in clean_extensions if v not in {"0000", "0010"})
    sigs = [_hex4(v) for v in signature_algorithms if not _is_grease(v)]

    cipher_count = min(len(clean_ciphers), 99)
    ext_count = min(len(clean_extensions), 99)
    prefix = (
        f"t{_version_code(legacy_version, supported_versions)}"
        f"{'d' if has_sni else 'i'}"
        f"{cipher_count:02d}"
        f"{ext_count:02d}"
        f"{_alpn_code(alpn_protocols)}"
    )

    cipher_str = ",".join(clean_ciphers)
    ext_str = ",".join(sorted_extensions)
    ext_sig_str = f"{ext_str}_{','.join(sigs)}" if sigs else ext_str

    cipher_hash = _sha12(cipher_str) if cipher_str else "000000000000"
    ext_hash = _sha12(ext_sig_str) if ext_str else "000000000000"
    return f"{prefix}_{cipher_hash}_{ext_hash}", f"{prefix}_{cipher_str}_{ext_sig_str}"
