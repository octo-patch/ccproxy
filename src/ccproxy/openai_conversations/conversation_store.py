"""In-memory L1 TTL store for OpenAI Conversations multi-turn threading.

ccproxy is stateless per request and the OpenAI ``/v1/chat/completions`` wire
carries no thread id, so cross-turn continuation is keyed on the
conversation-prefix hash (``FlowRecord.conversation_id`` — SHA12 of the first
user text, stamped by ``InspectorAddon``). The ``OpenAIConversationsAddon``
captures the ChatGPT ``conversation_id`` and the last assistant message id from
each completed SSE response into this store; the next-turn
``openai_conversations_thread_inject`` hook reads them back to continue the same
saved ChatGPT conversation.

In-memory only; no disk persistence; survives no ccproxy restarts. Modeled on
:class:`ccproxy.lightllm.pplx_threads.PerplexityThreadStore`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

__all__ = [
    "ConversationStore",
    "ConversationThread",
    "clear_conversation_store",
    "get_conversation_store",
]


_FALLBACK_TTL_SECONDS: float = 3600.0
"""Used when ``get_config()`` is unavailable (early startup, tests without a
config instance). Production reads
``CCProxyConfig.lightllm.openai_conversations.ttl_seconds``."""


@dataclass(frozen=True)
class ConversationThread:
    """ChatGPT conversation identifiers captured from a completed response.

    Keyed by the ccproxy ``conversation_id`` (SHA12 of the first user text) and
    reused on the next turn to continue the same saved ChatGPT conversation.
    """

    conversation_id: str
    """ChatGPT server conversation id (the saved conversation to continue)."""

    parent_message_id: str
    """Last assistant message id — the parent of the next user turn."""

    model: str
    """Model slug last used on this conversation."""

    last_used: float
    """Monotonic timestamp of the most recent access, for TTL eviction."""


def _get_ttl_seconds() -> float:
    """Lazy-read the active TTL from
    ``CCProxyConfig.lightllm.openai_conversations.ttl_seconds``.

    Falls back to ``_FALLBACK_TTL_SECONDS`` when the config singleton is not yet
    initialized (early startup or tests that bypass config loading), so YAML
    changes take effect on the next eviction pass with no singleton to flush.
    """
    try:
        from ccproxy.config import get_config

        return float(get_config().lightllm.openai_conversations.ttl_seconds)
    except Exception:
        return _FALLBACK_TTL_SECONDS


class ConversationStore:
    """Thread-safe TTL store keyed by the ccproxy conversation_id (SHA12).

    TTL is lazy-bound to ``OpenAIConversationsConfig.ttl_seconds`` via
    :func:`_get_ttl_seconds` at every eviction pass. A constructor override
    (``ttl_seconds=...``) freezes the TTL for the instance — used by tests that
    need deterministic eviction. Production uses the singleton from
    :func:`get_conversation_store`, which omits the override.
    """

    def __init__(self, ttl_seconds: float | None = None) -> None:
        self._ttl_override = ttl_seconds
        self._store: dict[str, ConversationThread] = {}
        self._lock = threading.Lock()

    @property
    def ttl(self) -> float:
        """Current TTL — override if set on the instance, else config-lazy."""
        if self._ttl_override is not None:
            return self._ttl_override
        return _get_ttl_seconds()

    def get(self, key: str) -> ConversationThread | None:
        """Return the cached thread for ``key`` or ``None``.

        Bumps the entry's ``last_used`` timestamp on hit and lazy-evicts any
        expired entries during the lookup pass.
        """
        with self._lock:
            self._evict_expired_locked()
            cached = self._store.get(key)
            if cached is None:
                return None
            refreshed = ConversationThread(
                conversation_id=cached.conversation_id,
                parent_message_id=cached.parent_message_id,
                model=cached.model,
                last_used=time.monotonic(),
            )
            self._store[key] = refreshed
            return refreshed

    def save(
        self,
        key: str,
        *,
        conversation_id: str,
        parent_message_id: str,
        model: str = "",
    ) -> None:
        """Insert or overwrite the thread state for ``key``.

        Called by ``OpenAIConversationsAddon`` after each completed SSE stream.
        An eviction sweep runs at the end so the store stays bounded.
        """
        with self._lock:
            self._store[key] = ConversationThread(
                conversation_id=conversation_id,
                parent_message_id=parent_message_id,
                model=model,
                last_used=time.monotonic(),
            )
            self._evict_expired_locked()

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def _evict_expired_locked(self) -> None:
        now = time.monotonic()
        ttl = self.ttl
        expired = [k for k, v in self._store.items() if now - v.last_used > ttl]
        for k in expired:
            del self._store[k]


_store_instance: ConversationStore | None = None
_store_lock = threading.Lock()


def get_conversation_store() -> ConversationStore:
    """Return the process-wide ``ConversationStore`` singleton."""
    global _store_instance
    with _store_lock:
        if _store_instance is None:
            _store_instance = ConversationStore()
        return _store_instance


def clear_conversation_store() -> None:
    """Reset the singleton. Called from the test cleanup fixture."""
    global _store_instance
    with _store_lock:
        _store_instance = None
