"""Tests for ccproxy.inspector.transport_override_addon.TransportOverrideAddon.

Covers the engagement precedence: explicit ``Provider.fingerprint_profile``
wins, otherwise falls back to ``ShapeStore.pick_fingerprint(provider.type)``,
otherwise leaves the flow on mitmproxy's native transport.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from ccproxy.config import CCProxyConfig, Provider, set_config_instance
from ccproxy.flows.store import FlowRecord, InspectorMeta
from ccproxy.inspector.fingerprint import CapturedFingerprint
from ccproxy.inspector.transport_override_addon import TransportOverrideAddon
from ccproxy.transport.sidecar import IMPERSONATE_HEADER, TARGET_URL_HEADER

_SIDECAR_PORT = 19200
type ShapeFingerprintSetter = Callable[[CapturedFingerprint | None], None]


# ---------------------------------------------------------------------------
# Shape-store stub fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def shape_fingerprint(monkeypatch: pytest.MonkeyPatch) -> ShapeFingerprintSetter:
    """Stub ``ShapeStore.pick_fingerprint`` with a configurable return value.

    Returns a setter that takes the desired ``CapturedFingerprint | None`` to
    return from the next ``get_store().pick_fingerprint(...)`` call. Defaults
    to ``None`` (no shape fingerprint available) so tests that don't set it
    behave as if no shape exists.
    """
    state: dict[str, CapturedFingerprint | None] = {"value": None}
    fake_store = MagicMock()
    fake_store.pick_fingerprint = MagicMock(side_effect=lambda _provider: state["value"])
    monkeypatch.setattr("ccproxy.shaping.store.get_store", lambda: fake_store)

    def setter(value: CapturedFingerprint | None) -> None:
        state["value"] = value

    return setter


def _make_captured_fingerprint(provider: str = "anthropic") -> CapturedFingerprint:
    """Build a minimal valid CapturedFingerprint for fallback tests."""
    return CapturedFingerprint(
        schema_version=1,
        source="test",
        captured_at="2026-05-24T00:00:00Z",
        sni="api.anthropic.com",
        alpn_protocols=("http/1.1",),
        legacy_version=0x0303,
        supported_versions=("0x0304", "0x0303"),
        cipher_suites=("0x1301", "0x1302", "0x1303"),
        extensions=("0x0000", "0x0010"),
        supported_groups=("0x001d",),
        ec_point_formats=("0x00",),
        signature_algorithms=("0x0403", "0x0804"),
        signature_algorithm_names=("ecdsa_secp256r1_sha256", "rsa_pss_rsae_sha256"),
        ja3="769,4865-4866-4867,0-10,29,0",
        ja3_full="t13d1714h1_5b57614c22b0_43ade6aba3df",
        ja4="t13d1714h1",
        ja4_r="t13d1714h1_test",
        http_version="http/1.1",
        provider=provider,
    )


# ---------------------------------------------------------------------------
# Flow factory helper
# ---------------------------------------------------------------------------


def _make_flow(
    *,
    auth_provider: str | None = None,
    pretty_url: str = "https://api.anthropic.com/v1/messages",
    host: str = "api.anthropic.com",
    port: int = 443,
    scheme: str = "https",
    content: bytes = b'{"model": "claude-sonnet"}',
    method: str = "POST",
) -> MagicMock:
    """Build a minimal MagicMock that approximates a mitmproxy HTTPFlow.

    ``flow.metadata`` is a real dict so writes are observable.
    ``flow.request`` attributes are normal MagicMock attributes except for
    ``pretty_url``, ``headers``, ``content``, and ``method``, which are set
    explicitly.
    """
    flow = MagicMock()
    flow.id = "test-flow-id"
    flow.metadata = {}
    if auth_provider is not None:
        flow.metadata["ccproxy.auth_provider"] = auth_provider

    flow.request.pretty_url = pretty_url
    flow.request.host = host
    flow.request.port = port
    flow.request.scheme = scheme
    flow.request.headers = {}
    flow.request.content = content
    flow.request.method = method
    return flow


# ---------------------------------------------------------------------------
# Helper: install a minimal config with a named Provider
# ---------------------------------------------------------------------------


def _set_provider(name: str, *, fingerprint_profile: str | None) -> None:
    provider = Provider(
        host="api.anthropic.com",
        type="anthropic",
        fingerprint_profile=fingerprint_profile,
    )
    cfg = CCProxyConfig(providers={name: provider})
    set_config_instance(cfg)


# ---------------------------------------------------------------------------
# No-op paths
# ---------------------------------------------------------------------------


class TestNoopPaths:
    async def test_noop_when_auth_provider_absent(self) -> None:
        """Flow with no ccproxy.auth_provider metadata is left completely untouched."""
        flow = _make_flow(auth_provider=None)
        original_host = flow.request.host
        original_port = flow.request.port
        original_scheme = flow.request.scheme

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.request.host == original_host
        assert flow.request.port == original_port
        assert flow.request.scheme == original_scheme
        assert "ccproxy.transport_override" not in flow.metadata
        assert TARGET_URL_HEADER not in flow.request.headers
        assert IMPERSONATE_HEADER not in flow.request.headers

    async def test_noop_when_auth_provider_empty_string(self) -> None:
        """An empty string for auth_provider is falsy — treated as absent."""
        flow = _make_flow()
        flow.metadata["ccproxy.auth_provider"] = ""
        original_host = flow.request.host

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.request.host == original_host
        assert "ccproxy.transport_override" not in flow.metadata

    async def test_noop_when_provider_unknown_to_config(self) -> None:
        """auth_provider set to a name not in config.providers — untouched."""
        flow = _make_flow(auth_provider="doesnotexist")
        # Leave config empty (autouse cleanup already cleared it)
        original_host = flow.request.host

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.request.host == original_host
        assert "ccproxy.transport_override" not in flow.metadata

    async def test_noop_when_fingerprint_profile_is_none_and_no_shape(
        self, shape_fingerprint: ShapeFingerprintSetter
    ) -> None:
        """Provider exists, fingerprint_profile=None, no shape fingerprint — flow is untouched."""
        _set_provider("anthropic", fingerprint_profile=None)
        shape_fingerprint(None)
        flow = _make_flow(auth_provider="anthropic")
        original_host = flow.request.host
        original_port = flow.request.port

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.request.host == original_host
        assert flow.request.port == original_port
        assert "ccproxy.transport_override" not in flow.metadata

    async def test_noop_leaves_headers_clean_when_no_profile_and_no_shape(
        self, shape_fingerprint: ShapeFingerprintSetter
    ) -> None:
        _set_provider("anthropic", fingerprint_profile=None)
        shape_fingerprint(None)
        flow = _make_flow(auth_provider="anthropic")

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert TARGET_URL_HEADER not in flow.request.headers
        assert IMPERSONATE_HEADER not in flow.request.headers


# ---------------------------------------------------------------------------
# Rewrite path — fingerprint_profile set
# ---------------------------------------------------------------------------


class TestRewritePath:
    async def test_target_url_header_set_to_original_pretty_url(self) -> None:
        _set_provider("anthropic", fingerprint_profile="chrome131")
        pretty_url = "https://api.anthropic.com/v1/messages"
        flow = _make_flow(auth_provider="anthropic", pretty_url=pretty_url)

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.request.headers[TARGET_URL_HEADER] == pretty_url

    async def test_impersonate_header_set_to_profile(self) -> None:
        _set_provider("anthropic", fingerprint_profile="chrome131")
        flow = _make_flow(auth_provider="anthropic")

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.request.headers[IMPERSONATE_HEADER] == "chrome131"

    async def test_host_rewritten_to_loopback(self) -> None:
        _set_provider("anthropic", fingerprint_profile="chrome131")
        flow = _make_flow(auth_provider="anthropic")

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.request.host == "127.0.0.1"

    async def test_port_rewritten_to_sidecar_port(self) -> None:
        _set_provider("anthropic", fingerprint_profile="chrome131")
        flow = _make_flow(auth_provider="anthropic")

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.request.port == _SIDECAR_PORT

    async def test_scheme_rewritten_to_http(self) -> None:
        _set_provider("anthropic", fingerprint_profile="chrome131")
        flow = _make_flow(auth_provider="anthropic", scheme="https")

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.request.scheme == "http"

    async def test_host_header_set_to_loopback_with_port(self) -> None:
        _set_provider("anthropic", fingerprint_profile="chrome131")
        flow = _make_flow(auth_provider="anthropic")

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.request.headers["host"] == f"127.0.0.1:{_SIDECAR_PORT}"

    async def test_transport_override_flag_set_in_metadata(self) -> None:
        _set_provider("anthropic", fingerprint_profile="chrome131")
        flow = _make_flow(auth_provider="anthropic")

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.metadata["ccproxy.transport_override"] is True

    async def test_fingerprint_profile_recorded_in_metadata(self) -> None:
        _set_provider("anthropic", fingerprint_profile="chrome131")
        flow = _make_flow(auth_provider="anthropic")

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.metadata["ccproxy.fingerprint_profile"] == "chrome131"

    async def test_full_rewrite_state_snapshot(self) -> None:
        """Assert all rewritten fields in one go for the full happy path."""
        profile = "chrome131"
        pretty_url = "https://api.anthropic.com/v1/messages"
        _set_provider("myanthropic", fingerprint_profile=profile)
        flow = _make_flow(
            auth_provider="myanthropic",
            pretty_url=pretty_url,
            host="api.anthropic.com",
            port=443,
            scheme="https",
        )

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.request.headers[TARGET_URL_HEADER] == pretty_url
        assert flow.request.headers[IMPERSONATE_HEADER] == profile
        assert flow.request.host == "127.0.0.1"
        assert flow.request.port == _SIDECAR_PORT
        assert flow.request.scheme == "http"
        assert flow.request.headers["host"] == f"127.0.0.1:{_SIDECAR_PORT}"
        assert flow.metadata["ccproxy.transport_override"] is True
        assert flow.metadata["ccproxy.fingerprint_profile"] == profile


# ---------------------------------------------------------------------------
# Sidecar port propagated correctly
# ---------------------------------------------------------------------------


class TestSidecarPortPropagation:
    async def test_different_sidecar_ports_reflected(self) -> None:
        """Different sidecar_port values are written to flow.request.port independently."""
        _set_provider("anthropic", fingerprint_profile="chrome131")

        for port in (12345, 54321, 9999):
            flow = _make_flow(auth_provider="anthropic")
            addon = TransportOverrideAddon(sidecar_port=port)
            await addon.request(flow)
            assert flow.request.port == port
            assert flow.request.headers["host"] == f"127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# Forwarded-request snapshot capture (R4)
# ---------------------------------------------------------------------------


class TestForwardedRequestCapture:
    """TransportOverrideAddon populates FlowRecord.forwarded_request before rewriting."""

    async def test_snapshot_captured_when_record_present(self) -> None:
        """forwarded_request is populated when a FlowRecord is on the flow."""
        _set_provider("anthropic", fingerprint_profile="chrome131")
        flow = _make_flow(
            auth_provider="anthropic",
            pretty_url="https://api.anthropic.com/v1/messages",
            method="POST",
            content=b'{"model": "claude-sonnet"}',
        )
        record = FlowRecord(direction="inbound")
        flow.metadata[InspectorMeta.RECORD] = record

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert record.forwarded_request is not None

    async def test_snapshot_method_matches_original(self) -> None:
        """Snapshot preserves the original HTTP method."""
        _set_provider("anthropic", fingerprint_profile="chrome131")
        flow = _make_flow(auth_provider="anthropic", method="POST")
        record = FlowRecord(direction="inbound")
        flow.metadata[InspectorMeta.RECORD] = record

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert record.forwarded_request is not None
        assert record.forwarded_request.method == "POST"

    async def test_snapshot_url_is_original_pretty_url(self) -> None:
        """Snapshot URL is the real upstream URL, not the rewritten sidecar URL."""
        _set_provider("anthropic", fingerprint_profile="chrome131")
        original_url = "https://api.anthropic.com/v1/messages"
        flow = _make_flow(auth_provider="anthropic", pretty_url=original_url)
        record = FlowRecord(direction="inbound")
        flow.metadata[InspectorMeta.RECORD] = record

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert record.forwarded_request is not None
        assert record.forwarded_request.url == original_url
        assert "127.0.0.1" not in (record.forwarded_request.url or "")

    async def test_snapshot_taken_before_rewrite(self) -> None:
        """Snapshot URL is the original pretty_url, not the localhost sidecar URL."""
        _set_provider("anthropic", fingerprint_profile="chrome131")
        original_url = "https://api.openai.com/v1/chat/completions"
        flow = _make_flow(auth_provider="anthropic", pretty_url=original_url)
        record = FlowRecord(direction="inbound")
        flow.metadata[InspectorMeta.RECORD] = record

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert record.forwarded_request is not None
        assert record.forwarded_request.url == original_url
        assert f"127.0.0.1:{_SIDECAR_PORT}" not in (record.forwarded_request.url or "")

    async def test_snapshot_headers_are_pre_rewrite(self) -> None:
        """Snapshot headers contain original headers, not sidecar-injected ones."""
        _set_provider("anthropic", fingerprint_profile="chrome131")
        flow = _make_flow(auth_provider="anthropic")
        flow.request.headers = {"authorization": "Bearer tok", "content-type": "application/json"}
        record = FlowRecord(direction="inbound")
        flow.metadata[InspectorMeta.RECORD] = record

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert record.forwarded_request is not None
        # Pre-rewrite headers present
        assert record.forwarded_request.headers.get("authorization") == "Bearer tok"
        assert record.forwarded_request.headers.get("content-type") == "application/json"
        # Sidecar-injected headers must NOT appear in the snapshot
        assert "x-ccproxy-target-url" not in record.forwarded_request.headers
        assert "x-ccproxy-impersonate" not in record.forwarded_request.headers
        assert record.forwarded_request.headers.get("host") != f"127.0.0.1:{_SIDECAR_PORT}"

    async def test_snapshot_body_matches_original_content(self) -> None:
        """Snapshot body equals flow.request.content at capture time."""
        _set_provider("anthropic", fingerprint_profile="chrome131")
        original_body = b'{"messages": [{"role": "user", "content": "hello"}]}'
        flow = _make_flow(auth_provider="anthropic", content=original_body)
        record = FlowRecord(direction="inbound")
        flow.metadata[InspectorMeta.RECORD] = record

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert record.forwarded_request is not None
        assert record.forwarded_request.body == original_body

    async def test_no_record_on_flow_no_crash(self) -> None:
        """Missing FlowRecord — addon still rewrites normally without raising."""
        _set_provider("anthropic", fingerprint_profile="chrome131")
        flow = _make_flow(auth_provider="anthropic")
        # No InspectorMeta.RECORD in metadata

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        # Rewrite still happened
        assert flow.request.host == "127.0.0.1"
        assert flow.request.port == _SIDECAR_PORT
        assert flow.metadata.get("ccproxy.transport_override") is True

    async def test_no_fingerprint_profile_and_no_shape_leaves_forwarded_request_none(
        self, shape_fingerprint: ShapeFingerprintSetter
    ) -> None:
        """Provider with fingerprint_profile=None AND no shape — forwarded_request stays None."""
        _set_provider("anthropic", fingerprint_profile=None)
        shape_fingerprint(None)
        flow = _make_flow(auth_provider="anthropic")
        record = FlowRecord(direction="inbound")
        flow.metadata[InspectorMeta.RECORD] = record

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert record.forwarded_request is None


# ---------------------------------------------------------------------------
# Implicit shape-driven path — fingerprint_profile=None + shape has fingerprint
# ---------------------------------------------------------------------------


class TestShapeImplicitPath:
    """When Provider.fingerprint_profile is None and the shape carries a
    CapturedFingerprint, sidecar engages implicitly keyed by provider.type."""

    async def test_shape_fingerprint_engages_sidecar(self, shape_fingerprint: ShapeFingerprintSetter) -> None:
        _set_provider("anthropic", fingerprint_profile=None)
        shape_fingerprint(_make_captured_fingerprint())
        flow = _make_flow(auth_provider="anthropic")

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.request.host == "127.0.0.1"
        assert flow.request.port == _SIDECAR_PORT
        assert flow.request.scheme == "http"

    async def test_shape_fingerprint_uses_provider_type_as_impersonate_key(
        self, shape_fingerprint: ShapeFingerprintSetter
    ) -> None:
        """The IMPERSONATE_HEADER carries provider.type (= shape lookup key)."""
        provider = Provider(
            host="api.anthropic.com",
            type="anthropic",
            fingerprint_profile=None,
        )
        cfg = CCProxyConfig(providers={"some-alias": provider})
        set_config_instance(cfg)
        shape_fingerprint(_make_captured_fingerprint())
        flow = _make_flow(auth_provider="some-alias")

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.request.headers[IMPERSONATE_HEADER] == "anthropic"
        assert flow.metadata["ccproxy.fingerprint_profile"] == "anthropic"

    async def test_explicit_profile_wins_over_shape_fingerprint(
        self, shape_fingerprint: ShapeFingerprintSetter
    ) -> None:
        """Explicit Provider.fingerprint_profile takes precedence; shape is not consulted."""
        _set_provider("anthropic", fingerprint_profile="chrome131")
        shape_fingerprint(_make_captured_fingerprint())

        flow = _make_flow(auth_provider="anthropic")

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.request.headers[IMPERSONATE_HEADER] == "chrome131"
        assert flow.metadata["ccproxy.fingerprint_profile"] == "chrome131"

    async def test_target_url_preserved_in_implicit_path(self, shape_fingerprint: ShapeFingerprintSetter) -> None:
        _set_provider("anthropic", fingerprint_profile=None)
        shape_fingerprint(_make_captured_fingerprint())
        pretty_url = "https://api.anthropic.com/v1/messages"
        flow = _make_flow(auth_provider="anthropic", pretty_url=pretty_url)

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert flow.request.headers[TARGET_URL_HEADER] == pretty_url

    async def test_forwarded_request_captured_in_implicit_path(self, shape_fingerprint: ShapeFingerprintSetter) -> None:
        _set_provider("anthropic", fingerprint_profile=None)
        shape_fingerprint(_make_captured_fingerprint())
        flow = _make_flow(auth_provider="anthropic")
        record = FlowRecord(direction="inbound")
        flow.metadata[InspectorMeta.RECORD] = record

        addon = TransportOverrideAddon(sidecar_port=_SIDECAR_PORT)
        await addon.request(flow)

        assert record.forwarded_request is not None
        assert record.forwarded_request.url == "https://api.anthropic.com/v1/messages"


# ---------------------------------------------------------------------------
# Parametrized: different provider names + profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderRewriteCase:
    name: str
    """Descriptive name for the test scenario."""

    provider_name: str
    """Key in providers dict."""

    fingerprint_profile: str
    """Profile to configure and assert on."""

    sidecar_port: int
    """Port the addon was built with."""


PROVIDER_REWRITE_CASES: list[ProviderRewriteCase] = [
    ProviderRewriteCase(
        name="chrome131_anthropic",
        provider_name="myanthropic",
        fingerprint_profile="chrome131",
        sidecar_port=19200,
    ),
    ProviderRewriteCase(
        name="firefox133_openai",
        provider_name="myopenai",
        fingerprint_profile="firefox133",
        sidecar_port=19201,
    ),
    ProviderRewriteCase(
        name="safari260_custom",
        provider_name="mycustom",
        fingerprint_profile="safari260",
        sidecar_port=19202,
    ),
]


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.name) for c in PROVIDER_REWRITE_CASES],
)
async def test_provider_rewrite_profile_applied(case: ProviderRewriteCase) -> None:
    _set_provider(case.provider_name, fingerprint_profile=case.fingerprint_profile)
    flow = _make_flow(auth_provider=case.provider_name)

    addon = TransportOverrideAddon(sidecar_port=case.sidecar_port)
    await addon.request(flow)

    assert flow.request.headers[IMPERSONATE_HEADER] == case.fingerprint_profile
    assert flow.request.port == case.sidecar_port
    assert flow.metadata["ccproxy.fingerprint_profile"] == case.fingerprint_profile
