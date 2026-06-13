"""Tests for WireGuard keylog writer."""

import json
from pathlib import Path

from ccproxy.inspector.wg_keylog import write_wg_keylog


class TestWriteWgKeylog:
    def test_writes_both_keys(self, tmp_path: Path) -> None:
        conf = tmp_path / "wg.conf"
        conf.write_text(json.dumps({"server_key": "srvABC123==", "client_key": "cltXYZ789=="}))
        out = tmp_path / "wg.keylog"

        assert write_wg_keylog(conf, out) is True

        content = out.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "LOCAL_STATIC_PRIVATE_KEY = srvABC123=="
        assert lines[1] == "LOCAL_STATIC_PRIVATE_KEY = cltXYZ789=="

    def test_writes_only_server_key_when_client_absent(self, tmp_path: Path) -> None:
        conf = tmp_path / "wg.conf"
        conf.write_text(json.dumps({"server_key": "srvABC123=="}))
        out = tmp_path / "wg.keylog"

        assert write_wg_keylog(conf, out) is True

        content = out.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1
        assert lines[0] == "LOCAL_STATIC_PRIVATE_KEY = srvABC123=="

    def test_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        conf = tmp_path / "nonexistent.conf"
        out = tmp_path / "wg.keylog"
        assert write_wg_keylog(conf, out) is False

    def test_returns_false_on_invalid_json(self, tmp_path: Path) -> None:
        conf = tmp_path / "wg.conf"
        conf.write_text("not valid json {{{")
        out = tmp_path / "wg.keylog"
        assert write_wg_keylog(conf, out) is False

    def test_returns_false_when_server_key_missing(self, tmp_path: Path) -> None:
        conf = tmp_path / "wg.conf"
        conf.write_text(json.dumps({"client_key": "cltXYZ789=="}))
        out = tmp_path / "wg.keylog"
        assert write_wg_keylog(conf, out) is False
