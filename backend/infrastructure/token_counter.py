# -*- coding: utf-8 -*-
"""
Token counting & context window management.

Uses tiktoken when available (OpenAI-compatible models), falls back to
a character-based heuristic. Provides utilities to:
  - Count tokens in a message list
  - Trim message history to fit within a model's context window
  - Reserve space for the system prompt + completion tokens
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---- Model context window sizes (prompt tokens) ----
# Common models and their max context lengths
_MODEL_CONTEXT_SIZES: Dict[str, int] = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-3.5-turbo": 16385,
    "gemini-1.5-flash": 1048576,
    "gemini-1.5-pro": 2097152,
    "gemini-2.0-flash": 1048576,
    "gemini-2.5-flash": 1048576,
    "gemini-2.5-pro": 1048576,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "deepseek-chat": 65536,
    "deepseek-reasoner": 65536,
    "qwen-max": 32768,
    "qwen-plus": 131072,
}

# Default context window if model is unknown
_DEFAULT_CONTEXT_SIZE = 128000

# Tokens to reserve for the completion (output)
_DEFAULT_COMPLETION_RESERVE = 4096

# ---- tiktoken loader (lazy) ----

_tiktoken = None
_tiktoken_encoding = None


def _get_tiktoken():
    """Lazily load tiktoken; return None if not installed."""
    global _tiktoken, _tiktoken_encoding
    if _tiktoken is None:
        try:
            import tiktoken as _tk
            _tiktoken = _tk
            _tiktoken_encoding = _tk.get_encoding("cl100k_base")
        except ImportError:
            logger.debug("[token_counter] tiktoken not installed, using heuristic")
            _tiktoken = False  # sentinel: tried and failed
    return _tiktoken if _tiktoken is not False else None


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in *text*.

    Uses tiktoken when available; otherwise falls back to a heuristic
    of ~4 chars per token (reasonable for English + code).
    """
    enc = _get_tiktoken()
    if enc is not None and _tiktoken_encoding is not None:
        return len(_tiktoken_encoding.encode(text))
    # Heuristic: ~4 characters per token
    return max(1, len(text) // 4)


def count_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """
    Estimate total tokens for a list of chat messages.

    Follows the OpenAI token counting convention:
      - Each message: ~4 tokens overhead (role, separators)
      - Each message content: count_tokens(content)
      - +2 tokens for message framing
      - +3 tokens for priming (assistant reply start)
    """
    total = 3  # priming
    for msg in messages:
        total += 4  # message overhead
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            # Multimodal content (list of parts)
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", "")
                    total += count_tokens(text)
        # Tool calls in assistant messages
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                total += count_tokens(fn.get("arguments", ""))
                total += count_tokens(fn.get("name", ""))
    return total


def get_model_context_size(model: str) -> int:
    """Get the context window size for a model name."""
    # Try exact match first
    if model in _MODEL_CONTEXT_SIZES:
        return _MODEL_CONTEXT_SIZES[model]
    # Try prefix match (e.g. "gpt-4o-2024-05-13" matches "gpt-4o")
    for prefix, size in _MODEL_CONTEXT_SIZES.items():
        if model.startswith(prefix):
            return size
    return _DEFAULT_CONTEXT_SIZE


def trim_messages(
    messages: List[Dict[str, Any]],
    *,
    model: str = "",
    max_context: Optional[int] = None,
    completion_reserve: int = _DEFAULT_COMPLETION_RESERVE,
    keep_system: bool = True,
) -> List[Dict[str, Any]]:
    """
    Trim a message list to fit within the model's context window.

    Strategy:
      1. Always keep the system message (if keep_system=True)
      2. Keep the most recent messages
      3. Drop older messages from the middle until we fit

    Args:
        messages: The full message list.
        model: Model name (used to look up context window size).
        max_context: Override context window size (ignores model lookup).
        completion_reserve: Tokens to reserve for the completion.
        keep_system: Whether to always keep the first system message.

    Returns:
        A (possibly trimmed) copy of the message list.
    """
    if not messages:
        return messages

    context_size = max_context or get_model_context_size(model)
    budget = context_size - completion_reserve
    if budget <= 0:
        logger.warning(f"[token_counter] Budget <= 0 (context={context_size}, reserve={completion_reserve})")
        budget = context_size // 2

    # Separate system message
    system_msg = None
    body = list(messages)
    if keep_system and messages and messages[0].get("role") == "system":
        system_msg = messages[0]
        body = messages[1:]

    # Check if everything fits
    all_msgs = [system_msg] + body if system_msg else body
    total = count_message_tokens(all_msgs)
    if total <= budget:
        return messages

    # Need to trim — keep system + most recent messages
    system_tokens = count_message_tokens([system_msg]) if system_msg else 0
    remaining_budget = budget - system_tokens

    # Walk from newest to oldest, accumulating until budget exceeded
    kept = []
    used = 0
    for msg in reversed(body):
        msg_tokens = count_message_tokens([msg])
        if used + msg_tokens > remaining_budget:
            break
        kept.append(msg)
        used += msg_tokens

    kept.reverse()

    result = ([system_msg] + kept) if system_msg else kept

    trimmed_count = len(messages) - len(result)
    if trimmed_count > 0:
        logger.info(
            f"[token_counter] Trimmed {trimmed_count} messages "
            f"({total} tokens -> {count_message_tokens(result)} tokens, "
            f"budget={budget})"
        )

    return result
