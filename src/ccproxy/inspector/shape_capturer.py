"""Shape capture addon.

Registers ``ccproxy.shape``: a mitmproxy command that saves the specified
flows as shape artifacts to the provider's shape store on disk.
"""

from __future__ import annotations

import json
import logging
import re

from mitmproxy import command, http

from ccproxy.config import get_config
from ccproxy.constants import SENSITIVE_PATTERNS
from ccproxy.inspector.fingerprint import CapturedFingerprint
from ccproxy.inspector.flow_lookup import find_http_flow
from ccproxy.pipeline.context import metadata_from_flow
from ccproxy.shaping.store import get_store

logger = logging.getLogger(__name__)


_STRIP_SHAPE_HEADERS = {
    *SENSITIVE_PATTERNS,
    "x-goog-api-key",
    "proxy-authorization",
    "content-length",
    "content-encoding",
    "host",
    "transfer-encoding",
    "connection",
    "x-client-request-id",
    "session-id",
    "thread-id",
    "chatgpt-account-id",
    "x-openai-fedramp",
    "x-codex-installation-id",
    "x-codex-turn-state",
    "x-codex-turn-metadata",
    "x-codex-parent-thread-id",
    "x-codex-window-id",
    "x-openai-memgen-request",
    "x-openai-subagent",
    "openai-organization",
    "openai-project",
    # Internal ccproxy correlation header — meaningful only to the running
    # process that observed the flow. No identity value persists into a
    # shape, so strip at capture time. Apply-time defense in depth lives
    # in EgressSanitizerAddon.
    "x-ccproxy-flow-id",
}


class ShapeCaptureAddon:
    """Addon exposing ``ccproxy.shape`` — save provider shape artifacts."""

    @command.command("ccproxy.shape")  # type: ignore[untyped-decorator]
    def save_shape_artifact(self, flow_ids: str, provider: str, mode: str = "patch") -> str:
        """Save the listed flows as shape artifacts.

        ``flow_ids`` is a comma-separated list of mitmproxy flow ids.
        ``provider`` is the target provider name (e.g. ``anthropic``).
        ``mode`` is ``patch`` (default) or ``mflow``.
        Returns a JSON summary of the save operation.
        """
        ids = [fid.strip() for fid in flow_ids.split(",") if fid.strip()]
        if not ids:
            raise ValueError("no flow ids provided")

        mode = mode.strip().lower()
        if mode not in {"patch", "mflow"}:
            raise ValueError("mode must be 'patch' or 'mflow'")
        if mode == "patch" and len(ids) != 1:
            raise ValueError("patch shape generation requires exactly one flow")

        store = get_store()
        saved = 0
        missing: list[str] = []
        patch_path: str | None = None
        fingerprint_saved = False
        fingerprint_missing: list[str] = []

        config = get_config()
        profile = config.shaping.providers.get(provider)

        for fid in ids:
            flow = self._find_http_flow(fid)
            if flow is None:
                logger.warning("ccproxy.shape: no flow with id %s, skipping", fid)
                missing.append(fid)
                continue
            if not _validate_flow(flow, profile):
                missing.append(fid)
                continue
            fingerprint = _fingerprint_from_flow(flow, provider)
            if fingerprint is None:
                fingerprint_missing.append(fid)
            clean = _prepare_local_shape_flow(flow)
            if fingerprint is not None:
                metadata_from_flow(clean).fingerprint.profile = fingerprint.to_dict()
            if mode == "patch":
                if fingerprint is not None:
                    store.write_fingerprint(provider, fingerprint)
                    fingerprint_saved = True
                result = store.write_patch(provider, clean)
                patch_path = str(result.path)
                saved += 1 if result.changed else 0
                continue
            store.add(provider, clean)
            saved += 1

        summary: dict[str, object] = {
            "status": "ok" if saved else "empty",
            "provider": provider,
            "mode": mode,
            "missing": missing,
        }
        if fingerprint_saved or (mode == "mflow" and not fingerprint_missing):
            summary["fingerprint"] = "embedded"
        if fingerprint_missing:
            summary["fingerprint_missing"] = fingerprint_missing
        if mode == "patch":
            summary["patches_written"] = saved
            if patch_path is not None:
                summary["patch"] = patch_path
            if patch_path is not None and not saved:
                summary["status"] = "unchanged"
        else:
            summary["flows_saved"] = saved

        logger.info(
            "Saved %d shape artifact(s) under provider %s (%d missing)",
            saved,
            provider,
            len(missing),
        )
        return json.dumps(summary)

    @staticmethod
    def _find_http_flow(flow_id: str) -> http.HTTPFlow | None:
        return find_http_flow(flow_id)


def _validate_flow(
    flow: http.HTTPFlow,
    profile: object | None,
) -> bool:
    """Check that a flow is a valid API request suitable for shaping."""
    from ccproxy.config import ProviderShapingConfig

    if flow.request.method != "POST":
        logger.warning(
            "ccproxy.shape: flow %s is %s not POST, skipping",
            flow.id,
            flow.request.method,
        )
        return False
    ct = flow.request.headers.get("content-type", "")
    if not ct.startswith("application/json"):
        logger.warning(
            "ccproxy.shape: flow %s content-type %r not JSON, skipping",
            flow.id,
            ct,
        )
        return False
    if (
        isinstance(profile, ProviderShapingConfig)
        and profile.capture.path_pattern
        and not re.search(profile.capture.path_pattern, flow.request.path)
    ):
        logger.warning(
            "ccproxy.shape: flow %s path %s doesn't match %s, skipping",
            flow.id,
            flow.request.path,
            profile.capture.path_pattern,
        )
        return False
    return True


def _prepare_local_shape_flow(flow: http.HTTPFlow) -> http.HTTPFlow:
    """Deep-copy a captured flow for local shape storage.

    This is not the public packaged-default scrub. It removes response-side
    state plus auth, transport, and ccproxy-internal request headers; package
    preparation runs the apply-time shaping hooks against a canonical request
    and then audits for public-distribution PII separately.
    """
    clone: http.HTTPFlow = flow.copy()  # type: ignore[no-untyped-call]
    clone.response = None
    clone.websocket = None
    clone.error = None
    clone.comment = ""
    for name in _STRIP_SHAPE_HEADERS:
        clone.request.headers.pop(name, None)
    return clone


def _fingerprint_from_flow(flow: http.HTTPFlow, provider: str) -> CapturedFingerprint | None:
    metadata = metadata_from_flow(flow)
    raw = metadata.fingerprint.client or metadata.legacy_client_fingerprint
    if not isinstance(raw, dict):
        return None
    fingerprint = CapturedFingerprint.from_dict(raw)
    return fingerprint.with_request_context(
        provider=provider,
        user_agent=flow.request.headers.get("user-agent", ""),
        runtime_version=flow.request.headers.get("x-stainless-runtime-version", ""),
    )
