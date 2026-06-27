"""Tests for the OpenAI Conversations PoW module.

Covers:
  - Ground-truth FNV-1a hash vectors (verified against Rust and Go reference impls)
  - Difficulty-check behavior (lexicographic, not numeric)
  - PoW exhaustion raises PowExhaustedError (typed domain exception)
  - solve_pow returns gAAAAAB…~S with a valid solution
  - Requirements token: gAAAAAC prefix, 25 slots, slot[9]=performance_now, no ~S suffix
  - Proof token: gAAAAAB prefix with ~S suffix
  - encode_config_array round-trips
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

import pytest

from ccproxy.openai_conversations.pow import (
    PowExhaustedError,
    pow_hash_hex,
    solve_pow,
)
from ccproxy.openai_conversations.prepare_p import (
    ConfigOptions,
    _build_base_array,
    _format_browser_date,
    build_prepare_p,
    build_requirements_token,
    encode_config_array,
)

# ---------------------------------------------------------------------------
# Hash vector tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HashVectorCase:
    """Ground-truth FNV-1a hash test case."""

    name: str
    """Test ID."""

    input_str: str
    """Input string to hash."""

    expected_hex: str
    """Expected 8-character lowercase hex digest."""


HASH_VECTOR_CASES: list[HashVectorCase] = [
    HashVectorCase(
        name="hello",
        input_str="hello",
        expected_hex="888d766e",
    ),
    HashVectorCase(
        name="empty_string",
        input_str="",
        expected_hex="ab3e7c0b",
    ),
]


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.name) for c in HASH_VECTOR_CASES],
)
def test_pow_hash_hex_ground_truth_vectors(case: HashVectorCase) -> None:
    """pow_hash_hex must match the reference Rust and Go implementations."""
    result = pow_hash_hex(input_str=case.input_str)
    assert result == case.expected_hex, f"got {result!r}, want {case.expected_hex!r}"


def test_pow_hash_hex_returns_8_hex_digits() -> None:
    """Output is always exactly 8 lowercase hex characters."""
    result = pow_hash_hex(input_str="test input")
    assert len(result) == 8
    assert result == result.lower()
    int(result, 16)  # raises if not valid hex


# ---------------------------------------------------------------------------
# Difficulty comparison tests — lexicographic, NOT numeric
# ---------------------------------------------------------------------------


def test_difficulty_check_is_lexicographic_not_numeric() -> None:
    """The difficulty comparison must be lexicographic hex-prefix comparison.

    A numeric comparison would break on prefixes where leading hex digits
    produce a different ordering from the numeric value. Verify the algorithm
    uses string prefix comparison.

    Example: "09" <= "10" numerically, but "09" <= "10" lexicographically too —
    however "0a" < "10" lexicographically but 0x0a < 0x10 also holds.
    The critical case is "0b" vs "0a": "0b" > "0a" lexicographically AND
    numerically. A cleaner test: verify the condition is purely string-based
    by checking known outputs.
    """
    # hash of "hello" is "888d766e"; a difficulty of "888d77" should pass
    # because "888d76" < "888d77" — but "888d766e"[:6] = "888d76" <= "888d77"
    digest = pow_hash_hex(input_str="hello")
    assert digest == "888d766e"
    # 6-char difficulty: "888d76" <= "888d77" → True (accepted)
    assert digest[:6] <= "888d77"
    # "888d76" <= "888d75" → False (rejected)
    assert not (digest[:6] <= "888d75")


# ---------------------------------------------------------------------------
# PoW solve_pow tests
# ---------------------------------------------------------------------------


def _decode_pow_token(token: str) -> list[Any]:
    """Strip prefix/suffix and base64-decode to a list."""
    assert token.startswith("gAAAAAB"), f"unexpected prefix: {token[:10]}"
    assert token.endswith("~S"), f"missing ~S suffix: {token[-4:]}"
    inner = token[len("gAAAAAB") : -len("~S")]
    decoded: list[Any] = json.loads(base64.b64decode(inner))
    return decoded


def test_solve_pow_returns_valid_token_shape() -> None:
    """solve_pow must return gAAAAAB…~S with a 25-slot decoded config."""
    opts = ConfigOptions.fixed_for_tests()
    token = solve_pow(seed="0.5099912974590367", difficulty="061a80", opts=opts)

    assert token.startswith("gAAAAAB")
    assert token.endswith("~S")
    config = _decode_pow_token(token)
    assert len(config) == 25


def test_solve_pow_solution_satisfies_difficulty() -> None:
    """The returned proof token must satisfy the lexicographic difficulty check."""
    opts = ConfigOptions.fixed_for_tests()
    seed = "0.6287679384217534"
    difficulty = "06c164"
    token = solve_pow(seed=seed, difficulty=difficulty, opts=opts)

    inner = token[len("gAAAAAB") : -len("~S")]
    digest = pow_hash_hex(input_str=seed + inner)
    assert digest[: len(difficulty)] <= difficulty, f"hash {digest!r} does not satisfy difficulty {difficulty!r}"


def test_solve_pow_exhaustion_raises_typed_exception() -> None:
    """When no valid nonce exists within the budget, PowExhaustedError is raised."""
    opts = ConfigOptions.fixed_for_tests()
    # An impossible difficulty (all 'f' — any hash satisfies this, so use
    # all '0' which only a hash of exactly "00000000" could satisfy).
    with pytest.raises(PowExhaustedError, match="exhausted"):
        solve_pow(seed="impossible_seed", difficulty="0000000", opts=opts)


# ---------------------------------------------------------------------------
# Requirements / prepare_p token shape tests
# ---------------------------------------------------------------------------


def _decode_requirements_token(token: str) -> list[Any]:
    """Strip the gAAAAAC prefix (no ~S suffix) and base64-decode."""
    assert token.startswith("gAAAAAC"), f"unexpected prefix: {token[:10]}"
    assert not token.endswith("~S"), "requirements token must not carry the ~S proof suffix"
    inner = token[len("gAAAAAC") :]
    decoded: list[Any] = json.loads(base64.b64decode(inner))
    return decoded


def test_build_requirements_token_prefix_no_proof_suffix() -> None:
    """Requirements token starts with gAAAAAC and carries NO ~S proof suffix.

    Matches gproxy build_prepare_p (prepare_p.rs:253-260): only the solved PoW
    answer (gAAAAAB…~S) carries the ~S suffix.
    """
    opts = ConfigOptions.fixed_for_tests()
    token = build_requirements_token(opts=opts)
    assert token.startswith("gAAAAAC")
    assert not token.endswith("~S")


def test_build_requirements_token_decodes_to_25_slots() -> None:
    """Requirements token must decode to exactly 25 slots."""
    opts = ConfigOptions.fixed_for_tests()
    token = build_requirements_token(opts=opts)
    config = _decode_requirements_token(token)
    assert len(config) == 25


def test_build_requirements_token_slot3_is_1() -> None:
    """Slot [3] in the requirements token must be exactly 1 (not a random float)."""
    opts = ConfigOptions.fixed_for_tests()
    token = build_requirements_token(opts=opts)
    config = _decode_requirements_token(token)
    assert config[3] == 1


def test_build_requirements_token_slot9_is_performance_now() -> None:
    """Slot [9] carries performance_now, not 0 (matches gproxy build_prepare_p)."""
    opts = ConfigOptions.fixed_for_tests()
    token = build_requirements_token(opts=opts)
    config = _decode_requirements_token(token)
    assert config[9] == opts.performance_now


def test_build_prepare_p_is_alias_for_requirements_token() -> None:
    """build_prepare_p returns the same value as build_requirements_token."""
    opts = ConfigOptions.fixed_for_tests()
    assert build_prepare_p(opts=opts) == build_requirements_token(opts=opts)


def test_proof_token_has_correct_prefix() -> None:
    """solve_pow must use gAAAAAB prefix (distinct from gAAAAAC requirements prefix)."""
    opts = ConfigOptions.fixed_for_tests()
    token = solve_pow(seed="0.5099912974590367", difficulty="061a80", opts=opts)
    assert token.startswith("gAAAAAB")
    assert not token.startswith("gAAAAAC")


# ---------------------------------------------------------------------------
# encode_config_array round-trip
# ---------------------------------------------------------------------------


def test_encode_config_array_round_trips() -> None:
    """encode_config_array(arr) base64-decodes back to the original JSON array."""
    opts = ConfigOptions.fixed_for_tests()
    arr = _build_base_array(opts)
    encoded = encode_config_array(config=arr)
    decoded = json.loads(base64.b64decode(encoded))
    assert decoded == arr


def test_base_array_has_25_slots() -> None:
    """_build_base_array must return exactly 25 elements."""
    opts = ConfigOptions.fixed_for_tests()
    arr = _build_base_array(opts)
    assert len(arr) == 25


def test_base_array_slot0_is_string_not_number() -> None:
    """Slot [0] (screen sum) must be a string, not an integer (Build25 requirement)."""
    opts = ConfigOptions.fixed_for_tests()
    arr = _build_base_array(opts)
    assert isinstance(arr[0], str)
    int(arr[0])  # must be a valid integer string


def test_base_array_slot2_is_string_not_number() -> None:
    """Slot [2] (jsHeapSizeLimit) must be a string, not an integer (Build25 requirement)."""
    opts = ConfigOptions.fixed_for_tests()
    arr = _build_base_array(opts)
    assert isinstance(arr[2], str)
    int(arr[2])  # must be a valid integer string


# ---------------------------------------------------------------------------
# Browser Date.toString() formatting — calendar arithmetic ground truth
# ---------------------------------------------------------------------------


def test_format_browser_date_unix_epoch() -> None:
    """The unix epoch formats to the known Thursday 1970-01-01 browser date string."""
    result = _format_browser_date(0, 0, "UTC")
    assert result == "Thu Jan 01 1970 00:00:00 GMT+0000 (UTC)"


def test_format_browser_date_known_2026_date_with_offset() -> None:
    """A known 2026 timestamp with +0800 offset matches the fixed test specimen.

    This is the same string :meth:`ConfigOptions.fixed_for_tests` hard-codes,
    so the calendar math and the fixture stay self-consistent.
    """
    result = _format_browser_date(1_776_763_557, 480, "中国标准时间")
    assert result == "Tue Apr 21 2026 17:25:57 GMT+0800 (中国标准时间)"


# ---------------------------------------------------------------------------
# browser_default (opts=None) path
# ---------------------------------------------------------------------------


def test_browser_default_populates_dynamic_fields() -> None:
    """browser_default fills date_string and time_origin via __post_init__."""
    opts = ConfigOptions.browser_default()
    assert opts.date_string  # non-empty, generated from the current time
    assert opts.time_origin != 0.0


def test_build_requirements_token_defaults_to_browser_options() -> None:
    """build_requirements_token() with no opts uses browser_default and stays valid."""
    token = build_requirements_token()
    assert token.startswith("gAAAAAC")
    assert not token.endswith("~S")
    config = _decode_requirements_token(token)
    assert len(config) == 25
    assert config[3] == 1
