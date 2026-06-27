"""OpenAI Conversations provider helpers for ccproxy.

Ported from the MIT-licensed gproxy chatgpt channel:
  Copyright (c) 2026 LeenHawk
  https://github.com/LeenHawk/gproxy  (MIT License)

Cross-checked against the gproxy chatgpt channel for the two-call
sentinel chat-requirements (prepare→PoW→finalize) protocol and Build25
fingerprint layout.

Public surface:
    pow.py       — FNV-1a PoW hash and solver
    prepare_p.py — 25-slot config builder for the p field
    sentinel.py  — chat-requirements prepare/finalize body builders + expiry helpers
    credentials.py — credential state load/update for the JSON state file
"""

from ccproxy.openai_conversations.credentials import (
    OpenAIConversationsCredentialState,
    load_credential_state,
    update_sentinel_fields,
)
from ccproxy.openai_conversations.pow import (
    PowExhaustedError,
    pow_hash_hex,
    solve_pow,
)
from ccproxy.openai_conversations.prepare_p import (
    ConfigOptions,
    build_prepare_p,
    build_requirements_token,
    encode_config,
)
from ccproxy.openai_conversations.sentinel import (
    SentinelResult,
    build_finalize_body,
    build_prepare_body,
    decode_jwt_exp_ms,
    is_expired,
)

__all__ = [
    "ConfigOptions",
    "OpenAIConversationsCredentialState",
    "PowExhaustedError",
    "SentinelResult",
    "build_finalize_body",
    "build_prepare_body",
    "build_prepare_p",
    "build_requirements_token",
    "decode_jwt_exp_ms",
    "encode_config",
    "is_expired",
    "load_credential_state",
    "pow_hash_hex",
    "solve_pow",
    "update_sentinel_fields",
]
