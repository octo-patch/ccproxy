"""Regression: legacy auth-source YAML formats still resolve after source extraction.

The split moved CredentialSource/AnyAuthSource out of config.py and into a
discriminated union under ccproxy.auth.sources. parse_auth_source must
continue to accept:

1. Bare command strings (most common form in user configs).
2. Dicts with only ``command`` or ``file`` keys (no ``type`` discriminator).
3. The new discriminated forms (``type: command|file|anthropic_oauth|google_oauth|codex_oauth``).
"""

from __future__ import annotations

import pytest

from ccproxy.auth.sources import (
    AnthropicAuthSource,
    CodexAuthSource,
    CommandAuthSource,
    FileAuthSource,
    GoogleAuthSource,
    parse_auth_source,
)


def test_bare_string_resolves_as_command_source() -> None:
    """Legacy ``providers.foo.auth: "echo bar"`` still maps to a CommandAuthSource."""
    source = parse_auth_source("echo bar")
    assert isinstance(source, CommandAuthSource)
    assert source.command == "echo bar"
    assert source.type == "command"


def test_dict_with_command_only_resolves_as_command_source() -> None:
    source = parse_auth_source({"command": "echo tok"})
    assert isinstance(source, CommandAuthSource)
    assert source.command == "echo tok"


def test_dict_with_file_only_resolves_as_file_source() -> None:
    source = parse_auth_source({"file": "/etc/example/token"})
    assert isinstance(source, FileAuthSource)
    assert source.file == "/etc/example/token"


def test_obsolete_auth_source_fields_remain_ignored() -> None:
    source = parse_auth_source({"command": "echo tok", "user_agent": "Test/1.0", "destinations": ["api.test.com"]})
    assert isinstance(source, CommandAuthSource)
    assert source.command == "echo tok"


def test_explicit_type_command_dispatches_correctly() -> None:
    source = parse_auth_source({"type": "command", "command": "echo x"})
    assert isinstance(source, CommandAuthSource)


def test_explicit_type_anthropic_oauth_dispatches_correctly() -> None:
    source = parse_auth_source(
        {
            "type": "anthropic_oauth",
            "file_path": "~/.config/ccproxy/oauth/anthropic.json",
        }
    )
    assert isinstance(source, AnthropicAuthSource)
    assert source.client_id == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"


def test_explicit_type_google_oauth_dispatches_correctly() -> None:
    source = parse_auth_source(
        {
            "type": "google_oauth",
            "client_id": "test.apps.googleusercontent.com",
            "client_secret": "GOCSPX-test",
        }
    )
    assert isinstance(source, GoogleAuthSource)
    assert source.endpoint == "https://oauth2.googleapis.com/token"


def test_explicit_type_codex_oauth_dispatches_correctly() -> None:
    source = parse_auth_source({"type": "codex_oauth"})
    assert isinstance(source, CodexAuthSource)
    assert source.file_path == "~/.codex/auth.json"


def test_unknown_type_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Cannot infer AuthSource type"):
        parse_auth_source({"unrecognized": "x"})


def test_already_typed_passthrough() -> None:
    typed = CommandAuthSource(command="echo y")
    result = parse_auth_source(typed)
    assert result is typed
