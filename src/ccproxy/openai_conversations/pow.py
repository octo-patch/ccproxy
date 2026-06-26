"""Proof-of-work solver for the OpenAI Conversations sentinel challenge.

Ported from the MIT-licensed gproxy chatgpt channel:
  Copyright (c) 2026 LeenHawk
  https://github.com/LeenHawk/gproxy  (MIT License)

The hash is a 32-bit FNV-1a function with a murmur3 avalanche mixer applied
to the string ``seed + base64(json(config))``. Matches ``powHashHex`` in the
deobfuscated chatgpt.com bundle.

Cross-checked against aurora-develop/aurora ``internal/prooftoken/prooftoken.go``
(FNV1aHash + SolveProofOfWork) and basketikun/chatgpt2api ``utils/pow.py``.
"""

from __future__ import annotations

import time

from ccproxy.openai_conversations.prepare_p import (
    ConfigOptions,
    _build_base_array,
    encode_config_array,
)

_MAX_ATTEMPTS = 500_000

# Fallback marker prefix emitted by the reference SDK when PoW is exhausted.
# Kept here only as a live-probe reference; ccproxy raises PowExhaustedError
# instead of emitting this token.
ERROR_PREFIX = "wQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D"


class PowExhaustedError(Exception):
    """Raised when PoW search exhausts the attempt budget without a solution."""


def pow_hash_hex(input_str: str) -> str:
    """FNV-1a 32-bit hash with murmur3 avalanche finalizer, returned as 8 hex digits.

    Matches ``powHashHex`` in the deobfuscated chatgpt.com bundle and
    ``FNV1aHash`` in aurora prooftoken.go.

    Ground-truth vectors (verified against the Rust and Go reference impls):
        pow_hash_hex("hello") == "888d766e"
        pow_hash_hex("")      == "ab3e7c0b"
    """
    h = 2_166_136_261  # FNV offset basis
    for byte in input_str.encode("utf-8"):
        h ^= byte
        h = (h * 16_777_619) & 0xFFFF_FFFF  # FNV prime, 32-bit wrap
    # murmur3 avalanche finalizer
    h ^= h >> 16
    h = (h * 2_246_822_507) & 0xFFFF_FFFF
    h ^= h >> 13
    h = (h * 3_266_489_909) & 0xFFFF_FFFF
    h ^= h >> 16
    return f"{h:08x}"


def solve_pow(seed: str, difficulty: str, opts: ConfigOptions | None = None) -> str:
    """Solve the PoW challenge for the given seed and hex difficulty string.

    Varies slot [3] (nonce / attempt index) and slot [9] (elapsed ms) on each
    iteration. Accepts the nonce when ``hash[:len(difficulty)] <= difficulty``
    (lexicographic hex-prefix comparison — NOT numeric).

    Returns ``gAAAAAB<base64(config)>~S`` on success.

    Raises:
        PowExhaustedError: When the attempt budget (500 000) is exhausted
            without finding a valid nonce.
    """
    if opts is None:
        opts = ConfigOptions.browser_default()

    base_array = _build_base_array(opts)
    dlen = len(difficulty)
    start_ms = time.monotonic() * 1000

    for attempt in range(_MAX_ATTEMPTS):
        cfg = list(base_array)
        cfg[3] = attempt
        elapsed = int(time.monotonic() * 1000 - start_ms)
        cfg[9] = elapsed
        payload = encode_config_array(cfg)
        digest = pow_hash_hex(seed + payload)
        if digest[:dlen] <= difficulty:
            return f"gAAAAAB{payload}~S"

    raise PowExhaustedError(f"PoW exhausted after {_MAX_ATTEMPTS} attempts (seed={seed!r}, difficulty={difficulty!r})")
