"""Resolve OpenAI Conversations thread continuation state into the request body.

Runs as an INBOUND hook before the transform router renders the
``/backend-api/f/conversation`` wire body. On a cache hit the hook writes
``ctx._body["openai_conversations"]`` so that
:class:`~ccproxy.lightllm.adapters.openai_conversations.OpenAIConversationsAdapter`
can continue an existing ChatGPT server-side conversation instead of starting a
new one.

On a cache miss the key is absent from ``ctx._body``; the adapter defaults to a
fresh conversation (full history flatten, ``parent_message_id:
"client-created-root"``).

Mirror of ``pplx_thread_inject`` / ``pplx_thread_inject_guard`` — same store
access pattern, same body-key convention.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ccproxy.openai_conversations.conversation_store import get_conversation_store
from ccproxy.pipeline.hook import hook

if TYPE_CHECKING:
    from ccproxy.pipeline.context import Context

logger = logging.getLogger(__name__)

__all__ = ["openai_conversations_thread_inject", "openai_conversations_thread_inject_guard"]

_PROVIDER_NAME = "openai_conversations"


def openai_conversations_thread_inject_guard(ctx: Context) -> bool:
    """Run only when inject_auth resolved the openai_conversations sentinel."""
    return ctx.metadata.auth_provider == _PROVIDER_NAME


@hook(
    reads=[],
    writes=["openai_conversations"],
)
def openai_conversations_thread_inject(ctx: Context, _: dict[str, Any]) -> Context:
    """Read the L1 ConversationStore and inject threading state into the body.

    On a store hit for ``ctx.metadata.conversation_id``, writes:
    ``ctx._body["openai_conversations"] = {conversation_id, parent_message_id,
    is_continuation: True}`` so the adapter emits only the new user turn.

    On a miss, writes ``{"is_continuation": False}`` to signal a fresh
    conversation. The adapter defaults to the fresh-conversation path when the
    key is absent, so this write is a belt-and-suspenders guarantee.
    """
    body = ctx._body if isinstance(ctx._body, dict) else {}

    conv_id = ctx.metadata.conversation_id
    if not isinstance(conv_id, str) or not conv_id:
        return ctx

    store = get_conversation_store()
    cached = store.get(conv_id)

    if cached is not None:
        body["openai_conversations"] = {
            "conversation_id": cached.conversation_id,
            "parent_message_id": cached.parent_message_id,
            "is_continuation": True,
        }
        ctx._body = body
        logger.info(
            "openai_conversations_thread_inject: continuing conv=%s chatgpt_id=%s",
            conv_id[:8],
            cached.conversation_id[:8] if cached.conversation_id else "",
        )
    else:
        body["openai_conversations"] = {"is_continuation": False}
        ctx._body = body
        logger.debug(
            "openai_conversations_thread_inject: no cached thread for conv=%s, fresh start",
            conv_id[:8],
        )

    return ctx
