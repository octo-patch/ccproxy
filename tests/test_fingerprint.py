"""Tests for ccproxy.inspector.fingerprint — TLS ClientHello parsing and JA3/JA4."""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import pytest
from curl_cffi.const import CurlHttpVersion, CurlOpt

from ccproxy.inspector.fingerprint import (
    CapturedFingerprint,
    _alpn_code,
    _decimal_segment,
    _hex4,
    _is_grease,
    _ja3_full,
    _parse_alpn,
    _parse_sni,
    _parse_u8_vector,
    _parse_u16_vector,
    _read_u16,
    _sha12,
    _u16_list,
    _unwrap_client_hello,
    _version_code,
    parse_client_hello_bytes,
)

# ---------------------------------------------------------------------------
# ClientHello byte-builder helpers
# ---------------------------------------------------------------------------


def _ext(ext_type: int, body: bytes) -> bytes:
    """Build one TLS extension wire encoding."""
    return struct.pack(">HH", ext_type, len(body)) + body


def _sni_ext(hostname: str) -> bytes:
    name = hostname.encode("ascii")
    server_name = struct.pack(">BH", 0, len(name)) + name
    return struct.pack(">H", len(server_name)) + server_name


def _alpn_ext(protocols: list[str]) -> bytes:
    inner = b""
    for p in protocols:
        pb = p.encode("ascii")
        inner += struct.pack("B", len(pb)) + pb
    return struct.pack(">H", len(inner)) + inner


def _supported_groups_ext(groups: list[int]) -> bytes:
    groups_bytes = struct.pack(">" + "H" * len(groups), *groups)
    return struct.pack(">H", len(groups_bytes)) + groups_bytes


def _sig_algs_ext(algs: list[int]) -> bytes:
    algs_bytes = struct.pack(">" + "H" * len(algs), *algs)
    return struct.pack(">H", len(algs_bytes)) + algs_bytes


def _ec_point_formats_ext(formats: list[int]) -> bytes:
    return struct.pack("B", len(formats)) + bytes(formats)


def _supported_versions_ext(versions: list[int]) -> bytes:
    vers_bytes = struct.pack(">" + "H" * len(versions), *versions)
    return struct.pack("B", len(vers_bytes)) + vers_bytes


def _build_client_hello(
    *,
    legacy_version: int = 0x0303,
    cipher_suites: list[int] | None = None,
    extensions_map: dict[int, bytes] | None = None,
) -> bytes:
    """Build a minimal bare ClientHello body suitable for parse_client_hello_bytes.

    Uses 2 cipher suites by default so the minimum 42-byte threshold is always met
    (2+32+1+2+4+1+1 = 43 bytes with 2 ciphers, no extensions).
    """
    if cipher_suites is None:
        cipher_suites = [0x1301, 0x1302]
    if extensions_map is None:
        extensions_map = {}

    random_bytes = b"\xab" * 32
    ciphers_bytes = struct.pack(">" + "H" * len(cipher_suites), *cipher_suites)
    exts_bytes = b""
    for etype, ebody in extensions_map.items():
        exts_bytes += _ext(etype, ebody)
    ext_block = struct.pack(">H", len(exts_bytes)) + exts_bytes if exts_bytes else b""

    return (
        struct.pack(">H", legacy_version)
        + random_bytes
        + struct.pack("B", 0)  # session_id len 0
        + struct.pack(">H", len(ciphers_bytes))
        + ciphers_bytes
        + struct.pack("B", 1)
        + struct.pack("B", 0)  # compression: 1 method, null
        + ext_block
    )


def _make_captured_fingerprint(**overrides: Any) -> CapturedFingerprint:
    defaults: dict[str, Any] = {
        "schema_version": 1,
        "source": "test",
        "captured_at": "2024-01-01T00:00:00+00:00",
        "sni": "api.anthropic.com",
        "alpn_protocols": ("h2", "http/1.1"),
        "legacy_version": 0x0303,
        "supported_versions": ("0304",),
        "cipher_suites": ("1301", "1302"),
        "extensions": ("0000", "0010"),
        "supported_groups": ("001d", "0017"),
        "ec_point_formats": ("00",),
        "signature_algorithms": ("0403", "0804"),
        "signature_algorithm_names": ("ecdsa_secp256r1_sha256", "rsa_pss_rsae_sha256"),
        "ja3": "abc123",
        "ja3_full": "771,4865-4866,0-16,29-23,0",
        "ja4": "t13d0202h2_abc_def",
        "ja4_r": "t13d0202h2_...",
        "http_version": "v2",
    }
    defaults.update(overrides)
    return CapturedFingerprint(**defaults)


# ---------------------------------------------------------------------------
# _unwrap_client_hello
# ---------------------------------------------------------------------------


class TestUnwrapClientHello:
    def test_bare_body_returned_unchanged(self) -> None:
        body = _build_client_hello()
        assert _unwrap_client_hello(body) == body

    def test_handshake_record_unwrapped(self) -> None:
        body = _build_client_hello()
        size_bytes = len(body).to_bytes(3, "big")
        handshake = b"\x01" + size_bytes + body
        assert _unwrap_client_hello(handshake) == body

    def test_tls_record_unwrapped(self) -> None:
        body = _build_client_hello()
        size_bytes = len(body).to_bytes(3, "big")
        handshake = b"\x01" + size_bytes + body
        tls_record = b"\x16\x03\x01" + struct.pack(">H", len(handshake)) + handshake
        assert _unwrap_client_hello(tls_record) == body

    def test_all_wrappers_produce_same_parse_result(self) -> None:
        body = _build_client_hello(
            cipher_suites=[0x1301, 0x0035],
            extensions_map={0: _sni_ext("test.com")},
        )
        size_bytes = len(body).to_bytes(3, "big")
        handshake = b"\x01" + size_bytes + body
        tls_record = b"\x16\x03\x01" + struct.pack(">H", len(handshake)) + handshake

        fp_bare = parse_client_hello_bytes(body)
        fp_hs = parse_client_hello_bytes(handshake)
        fp_tls = parse_client_hello_bytes(tls_record)

        assert fp_bare.ja3 == fp_hs.ja3 == fp_tls.ja3
        assert fp_bare.sni == fp_hs.sni == fp_tls.sni == "test.com"


# ---------------------------------------------------------------------------
# Low-level byte helpers
# ---------------------------------------------------------------------------


class TestReadU16:
    def test_reads_big_endian_u16(self) -> None:
        assert _read_u16(b"\x03\x03", 0) == 0x0303
        assert _read_u16(b"\x00\x13\x01", 1) == 0x1301


class TestU16List:
    def test_empty_bytes_returns_empty(self) -> None:
        assert _u16_list(b"") == []

    def test_single_value(self) -> None:
        assert _u16_list(struct.pack(">H", 0x1301)) == [0x1301]

    def test_multiple_values(self) -> None:
        buf = struct.pack(">HHH", 0x1301, 0x1302, 0x002F)
        assert _u16_list(buf) == [0x1301, 0x1302, 0x002F]


class TestParseU16Vector:
    def test_empty_buf_returns_empty(self) -> None:
        assert _parse_u16_vector(b"", width_bytes=2) == []

    def test_width_bytes_2(self) -> None:
        values = [0x001D, 0x0017]
        inner = struct.pack(">HH", *values)
        buf = struct.pack(">H", len(inner)) + inner
        assert _parse_u16_vector(buf, width_bytes=2) == values

    def test_width_bytes_1(self) -> None:
        values = [0x0304, 0x0303]
        inner = struct.pack(">HH", *values)
        buf = struct.pack("B", len(inner)) + inner
        assert _parse_u16_vector(buf, width_bytes=1) == values

    def test_buf_shorter_than_width_returns_empty(self) -> None:
        assert _parse_u16_vector(b"\x00", width_bytes=2) == []


class TestParseU8Vector:
    def test_empty_returns_empty(self) -> None:
        assert _parse_u8_vector(b"") == []

    def test_parses_list(self) -> None:
        buf = struct.pack("B", 2) + struct.pack("BB", 0x00, 0x01)
        assert _parse_u8_vector(buf) == [0x00, 0x01]


class TestParseAlpn:
    def test_empty_buf_returns_empty(self) -> None:
        assert _parse_alpn(b"") == []

    def test_single_protocol(self) -> None:
        ext = _alpn_ext(["h2"])
        assert _parse_alpn(ext) == ["h2"]

    def test_multiple_protocols(self) -> None:
        ext = _alpn_ext(["h2", "http/1.1"])
        assert _parse_alpn(ext) == ["h2", "http/1.1"]


class TestParseSni:
    def test_empty_buf_returns_none(self) -> None:
        assert _parse_sni(b"") is None

    def test_too_short_returns_none(self) -> None:
        assert _parse_sni(b"\x00\x01") is None

    def test_parses_hostname(self) -> None:
        sni_body = _sni_ext("example.com")
        assert _parse_sni(sni_body) == "example.com"

    def test_no_host_name_type_returns_none(self) -> None:
        # name_type=1 (not 0) — not a dns hostname
        name = b"example.com"
        server_name = struct.pack(">BH", 1, len(name)) + name
        buf = struct.pack(">H", len(server_name)) + server_name
        assert _parse_sni(buf) is None


class TestIsGrease:
    def test_grease_values_return_true(self) -> None:
        for v in [0x0A0A, 0x1A1A, 0xFAFA]:
            assert _is_grease(v) is True

    def test_non_grease_values_return_false(self) -> None:
        for v in [0x1301, 0x0303, 0x0035]:
            assert _is_grease(v) is False


class TestDecimalSegment:
    def test_excludes_grease(self) -> None:
        result = _decimal_segment([0x0A0A, 0x1301, 0x0035])
        assert result == f"{0x1301}-{0x0035}"

    def test_empty_list(self) -> None:
        assert _decimal_segment([]) == ""

    def test_all_grease_returns_empty(self) -> None:
        assert _decimal_segment([0x0A0A, 0x1A1A]) == ""


class TestJa3Full:
    def test_builds_expected_string(self) -> None:
        result = _ja3_full(
            legacy_version=0x0303,
            ciphers=[0x1301, 0x1302],
            extensions=[(0, b""), (10, b"")],
            supported_groups=[0x001D],
            ec_point_formats=[0x00],
        )
        assert result == "771,4865-4866,0-10,29,0"

    def test_grease_excluded_from_all_segments(self) -> None:
        result = _ja3_full(
            legacy_version=0x0303,
            ciphers=[0x0A0A, 0x1301],
            extensions=[(0x0A0A, b""), (10, b"")],
            supported_groups=[0x0A0A, 0x001D],
            ec_point_formats=[0x00],
        )
        assert "2506" not in result  # 0x0A0A == 2570
        assert "4865" in result
        assert "29" in result


class TestHex4:
    def test_pads_to_4_hex_digits(self) -> None:
        assert _hex4(0x0001) == "0001"
        assert _hex4(0x1301) == "1301"
        assert _hex4(0xFFFF) == "ffff"


class TestSha12:
    def test_returns_12_char_hex(self) -> None:
        result = _sha12("test")
        assert len(result) == 12
        expected = hashlib.sha256(b"test").hexdigest()[:12]
        assert result == expected


class TestAlpnCode:
    def test_empty_list_returns_00(self) -> None:
        assert _alpn_code([]) == "00"

    def test_empty_string_returns_00(self) -> None:
        assert _alpn_code([""]) == "00"

    def test_h2_returns_h2(self) -> None:
        assert _alpn_code(["h2"]) == "h2"

    def test_http_1_1_returns_h1(self) -> None:
        assert _alpn_code(["http/1.1"]) == "h1"

    def test_uses_first_protocol(self) -> None:
        result = _alpn_code(["h2", "http/1.1"])
        assert result == "h2"

    def test_non_alphanumeric_boundary_uses_hex_encoding(self) -> None:
        # Protocol string starting and ending with '/' (non-alnum) → hex path
        result = _alpn_code(["/h2/"])
        assert result == "2f"


class TestVersionCode:
    def test_tls13_from_supported_versions(self) -> None:
        assert _version_code(0x0303, [0x0304]) == "13"

    def test_tls12_fallback_when_no_supported_versions(self) -> None:
        assert _version_code(0x0303, []) == "12"

    def test_grease_filtered_from_supported_versions(self) -> None:
        assert _version_code(0x0303, [0x0A0A, 0x0304]) == "13"

    def test_legacy_version_used_when_all_grease(self) -> None:
        assert _version_code(0x0303, [0x0A0A]) == "12"

    def test_unknown_version_returns_00(self) -> None:
        assert _version_code(0x0101, []) == "00"


# ---------------------------------------------------------------------------
# parse_client_hello_bytes — integration
# ---------------------------------------------------------------------------


class TestParseClientHelloBytes:
    def test_too_short_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="ClientHello too short"):
            parse_client_hello_bytes(b"\x00" * 10)

    def test_minimal_hello_parses(self) -> None:
        body = _build_client_hello(cipher_suites=[0x1301, 0x1302])
        fp = parse_client_hello_bytes(body)
        assert fp.legacy_version == 0x0303
        assert fp.cipher_suites == ("1301", "1302")
        assert fp.sni is None
        assert fp.http_version == "v1_1"
        assert len(fp.ja3) == 32

    def test_sni_captured(self) -> None:
        body = _build_client_hello(
            extensions_map={0: _sni_ext("api.anthropic.com")},
        )
        fp = parse_client_hello_bytes(body)
        assert fp.sni == "api.anthropic.com"

    def test_alpn_h2_sets_http_version_v2(self) -> None:
        body = _build_client_hello(
            extensions_map={16: _alpn_ext(["h2", "http/1.1"])},
        )
        fp = parse_client_hello_bytes(body)
        assert fp.http_version == "v2"
        assert fp.alpn_protocols == ("h2", "http/1.1")

    def test_grease_filtered_from_ciphers(self) -> None:
        body = _build_client_hello(cipher_suites=[0x0A0A, 0x1301, 0x0035])
        fp = parse_client_hello_bytes(body)
        assert "0a0a" not in fp.cipher_suites
        assert "1301" in fp.cipher_suites
        assert "0035" in fp.cipher_suites

    def test_grease_filtered_from_extensions(self) -> None:
        body = _build_client_hello(
            extensions_map={
                0x0A0A: b"",
                10: _supported_groups_ext([0x001D]),
            }
        )
        fp = parse_client_hello_bytes(body)
        assert "0a0a" not in fp.extensions
        assert "000a" in fp.extensions

    def test_signature_algorithms_extracted(self) -> None:
        body = _build_client_hello(
            extensions_map={
                13: _sig_algs_ext([0x0403, 0x0804]),
            }
        )
        fp = parse_client_hello_bytes(body)
        assert fp.signature_algorithms == ("0403", "0804")
        assert "ecdsa_secp256r1_sha256" in fp.signature_algorithm_names
        assert "rsa_pss_rsae_sha256" in fp.signature_algorithm_names

    def test_unknown_signature_algorithms_excluded_from_names(self) -> None:
        body = _build_client_hello(
            extensions_map={
                13: _sig_algs_ext([0xFFFF]),  # unknown sig alg
            }
        )
        fp = parse_client_hello_bytes(body)
        assert fp.signature_algorithms == ("ffff",)
        assert fp.signature_algorithm_names == ()

    def test_supported_groups_extracted(self) -> None:
        body = _build_client_hello(
            extensions_map={
                10: _supported_groups_ext([0x001D, 0x0017]),
            }
        )
        fp = parse_client_hello_bytes(body)
        assert fp.supported_groups == ("001d", "0017")

    def test_supported_versions_extracted(self) -> None:
        body = _build_client_hello(
            extensions_map={
                43: _supported_versions_ext([0x0304]),
            }
        )
        fp = parse_client_hello_bytes(body)
        assert fp.supported_versions == ("0304",)

    def test_ec_point_formats_extracted(self) -> None:
        body = _build_client_hello(
            extensions_map={
                11: _ec_point_formats_ext([0x00]),
            }
        )
        fp = parse_client_hello_bytes(body)
        assert fp.ec_point_formats == ("00",)

    def test_source_parameter_stored(self) -> None:
        body = _build_client_hello()
        fp = parse_client_hello_bytes(body, source="custom_source")
        assert fp.source == "custom_source"

    def test_ja3_is_md5_of_ja3_full(self) -> None:
        body = _build_client_hello(cipher_suites=[0x1301, 0x1302])
        fp = parse_client_hello_bytes(body)
        expected_ja3 = hashlib.md5(fp.ja3_full.encode(), usedforsecurity=False).hexdigest()
        assert fp.ja3 == expected_ja3

    def test_full_hello_with_all_extensions(self) -> None:
        body = _build_client_hello(
            legacy_version=0x0303,
            cipher_suites=[0x1301, 0x1302, 0x0035],
            extensions_map={
                0: _sni_ext("api.example.com"),
                10: _supported_groups_ext([0x001D, 0x0017]),
                11: _ec_point_formats_ext([0x00]),
                13: _sig_algs_ext([0x0403, 0x0804]),
                16: _alpn_ext(["h2", "http/1.1"]),
                43: _supported_versions_ext([0x0304]),
            },
        )
        fp = parse_client_hello_bytes(body)
        assert fp.sni == "api.example.com"
        assert fp.http_version == "v2"
        assert "1301" in fp.cipher_suites
        assert "1302" in fp.cipher_suites
        assert fp.supported_versions == ("0304",)
        assert "000a" in fp.extensions  # supported_groups ext type
        assert "000b" in fp.extensions  # ec_point_formats ext type
        assert "ecdsa_secp256r1_sha256" in fp.signature_algorithm_names
        assert len(fp.ja4) > 0
        assert len(fp.ja4_r) > 0


# ---------------------------------------------------------------------------
# CapturedFingerprint dataclass
# ---------------------------------------------------------------------------


class TestCapturedFingerprintRoundTrip:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        fp = _make_captured_fingerprint(
            provider="anthropic",
            user_agent="claude-cli/1.0",
            runtime_version="3.14",
        )
        restored = CapturedFingerprint.from_dict(fp.to_dict())
        assert fp == restored

    def test_from_dict_defaults_on_missing_keys(self) -> None:
        fp = CapturedFingerprint.from_dict({})
        assert fp.schema_version == 1
        assert fp.source == ""
        assert fp.http_version == "v1_1"
        assert fp.sni is None
        assert fp.provider is None

    def test_to_dict_lists_not_tuples(self) -> None:
        fp = _make_captured_fingerprint()
        d = fp.to_dict()
        assert isinstance(d["alpn_protocols"], list)
        assert isinstance(d["cipher_suites"], list)
        assert isinstance(d["extensions"], list)

    def test_provider_user_agent_runtime_optional_none(self) -> None:
        fp = _make_captured_fingerprint()
        d = fp.to_dict()
        assert d["provider"] is None
        assert d["user_agent"] is None
        assert d["runtime_version"] is None


class TestCapturedFingerprintTransportCacheKey:
    def test_cache_key_is_16_hex_chars(self) -> None:
        fp = _make_captured_fingerprint()
        key = fp.transport_cache_key
        assert len(key) == 16
        int(key, 16)  # must be valid hex

    def test_cache_key_is_stable(self) -> None:
        fp = _make_captured_fingerprint()
        assert fp.transport_cache_key == fp.transport_cache_key

    def test_different_ja3_full_gives_different_key(self) -> None:
        fp1 = _make_captured_fingerprint(ja3_full="771,1,2,3,4")
        fp2 = _make_captured_fingerprint(ja3_full="771,9,9,9,9")
        assert fp1.transport_cache_key != fp2.transport_cache_key


class TestCapturedFingerprintTransportKwargs:
    def test_returns_expected_keys(self) -> None:
        fp = _make_captured_fingerprint()
        kwargs = fp.transport_kwargs()
        assert set(kwargs.keys()) == {"ja3", "http_version", "curl_options"}

    def test_http_version_v2_maps_to_curl_v2(self) -> None:
        fp = _make_captured_fingerprint(http_version="v2")
        assert fp.transport_kwargs()["http_version"] == CurlHttpVersion.V2_0

    def test_http_version_v1_0_maps_to_curl_v1_0(self) -> None:
        fp = _make_captured_fingerprint(http_version="v1_0")
        assert fp.transport_kwargs()["http_version"] == CurlHttpVersion.V1_0

    def test_http_version_v1_1_maps_to_curl_v1_1(self) -> None:
        fp = _make_captured_fingerprint(http_version="v1_1")
        assert fp.transport_kwargs()["http_version"] == CurlHttpVersion.V1_1

    def test_unknown_http_version_defaults_to_v1_1(self) -> None:
        fp = _make_captured_fingerprint(http_version="unknown_version")
        assert fp.transport_kwargs()["http_version"] == CurlHttpVersion.V1_1

    def test_curl_options_leaves_content_decoding_enabled(self) -> None:
        # libcurl decodes Content-Encoding so callers + the sidecar receive
        # plaintext; the transport must NOT disable HTTP_CONTENT_DECODING.
        fp = _make_captured_fingerprint()
        opts = fp.transport_kwargs()["curl_options"]
        assert CurlOpt.HTTP_CONTENT_DECODING not in opts

    def test_sig_algs_injected_into_curl_options_when_present(self) -> None:
        fp = _make_captured_fingerprint(
            signature_algorithm_names=("ecdsa_secp256r1_sha256", "rsa_pss_rsae_sha256"),
        )
        opts = fp.transport_kwargs()["curl_options"]
        assert CurlOpt.SSL_SIG_HASH_ALGS in opts
        assert opts[CurlOpt.SSL_SIG_HASH_ALGS] == "ecdsa_secp256r1_sha256,rsa_pss_rsae_sha256"

    def test_sig_algs_absent_when_empty(self) -> None:
        fp = _make_captured_fingerprint(signature_algorithm_names=())
        opts = fp.transport_kwargs()["curl_options"]
        assert CurlOpt.SSL_SIG_HASH_ALGS not in opts

    def test_ja3_field_is_ja3_full(self) -> None:
        fp = _make_captured_fingerprint(ja3_full="771,4865,0,29,0")
        assert fp.transport_kwargs()["ja3"] == "771,4865,0,29,0"


class TestCapturedFingerprintWithRequestContext:
    def test_updates_provider_user_agent_runtime(self) -> None:
        fp = _make_captured_fingerprint()
        updated = fp.with_request_context(
            provider="gemini",
            user_agent="gemini-cli/2.0",
            runtime_version="2.5",
        )
        assert updated.provider == "gemini"
        assert updated.user_agent == "gemini-cli/2.0"
        assert updated.runtime_version == "2.5"

    def test_other_fields_preserved(self) -> None:
        fp = _make_captured_fingerprint()
        updated = fp.with_request_context(
            provider="gemini",
            user_agent="ua",
            runtime_version="1",
        )
        assert updated.ja3 == fp.ja3
        assert updated.ja3_full == fp.ja3_full
        assert updated.cipher_suites == fp.cipher_suites
        assert updated.sni == fp.sni

    def test_empty_user_agent_stored_as_none(self) -> None:
        fp = _make_captured_fingerprint()
        updated = fp.with_request_context(
            provider="anthropic",
            user_agent="",
            runtime_version="1",
        )
        assert updated.user_agent is None

    def test_empty_runtime_version_stored_as_none(self) -> None:
        fp = _make_captured_fingerprint()
        updated = fp.with_request_context(
            provider="anthropic",
            user_agent="ua",
            runtime_version="",
        )
        assert updated.runtime_version is None
