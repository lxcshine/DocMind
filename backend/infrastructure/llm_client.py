# -*- coding: utf-8 -*-
"""
Unified LLM Client — single source of truth for all LLM calls.

Features:
  - Tenacity retry with exponential backoff (429/500/502/503)
  - Separate first-token timeout and total timeout for streaming
  - Fallback model when primary fails
  - Automatic token accounting via tracing integration
  - Singleton registry so every module shares the same client instances
"""

import asyncio
import logging
from functools import partial
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from openai import AsyncOpenAI, OpenAI

from config.settings import settings
from infrastructure.tracing import get_tracer
from infrastructure.token_counter import count_message_tokens, trim_messages

logger = logging.getLogger(__name__)

# ---- Retry strategy ----

_RETRIABLE_STATUS_CODES = {429, 500, 502, 503}
_MAX_RETRIES = 3
_RETRY_MULTIPLIER = 1.0  # seconds
_RETRY_MAX_WAIT = 30.0  # seconds


def _should_retry(exc: Exception) -> bool:
    """Decide whether to retry based on exception type."""
    # OpenAI API errors
    status = getattr(exc, "status_code", None)
    if status in _RETRIABLE_STATUS_CODES:
        return True
    # Network-level errors
    if isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
        return True
    # httpx transport errors
    exc_name = type(exc).__name__
    if "ConnectError" in exc_name or "ReadTimeout" in exc_name:
        return True
    return False


async def _retry_with_backoff(fn, *args, **kwargs):
    """
    Call *fn* with retry + exponential backoff.

    Only retries on transient errors (429, 5xx, network). Other errors
    propagate immediately.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            if not _should_retry(exc):
                raise
            last_exc = exc
            wait = min(_RETRY_MULTIPLIER * (2 ** (attempt - 1)), _RETRY_MAX_WAIT)
            logger.warning(
                f"[LLM Retry] attempt {attempt}/{_MAX_RETRIES} failed "
                f"({type(exc).__name__}: {exc}). Retrying in {wait:.1f}s..."
            )
            await asyncio.sleep(wait)
    # All retries exhausted
    raise last_exc  # type: ignore[misc]


# ---- Token extraction helper ----

def _extract_usage(response) -> Dict[str, int]:
    """Safely extract token usage from an OpenAI response object."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


# ===== Sync Client =====

class SyncLLMClient:
    """
    Synchronous LLM client with retry and fallback.

    Used by code paths that cannot be async (e.g. agentic_retrieve's
    tool-call loop which is inherently sequential).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        fallback_model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.fallback_model = fallback_model
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        response_format: Optional[Dict] = None,
        span=None,
    ) -> Any:
        """
        Synchronous chat completion with retry + fallback.

        Automatically trims messages to fit the model's context window.
        If the primary model fails after all retries, tries fallback_model
        once before giving up.
        """
        use_model = model or self.model
        messages = trim_messages(messages, model=use_model, completion_reserve=max_tokens)
        kwargs: Dict[str, Any] = dict(
            model=use_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format

        tracer = get_tracer()
        span = span or tracer.current_span()

        # Try primary model
        try:
            response = self._client.chat.completions.create(**kwargs)
            if span:
                usage = _extract_usage(response)
                span.set_tokens(
                    prompt=usage.get("prompt_tokens", 0),
                    completion=usage.get("completion_tokens", 0),
                )
            return response
        except Exception as exc:
            if not self.fallback_model or use_model == self.fallback_model:
                raise
            logger.warning(
                f"[LLM Fallback] Primary model {use_model} failed: {exc}. "
                f"Trying fallback {self.fallback_model}"
            )

        # Try fallback model
        kwargs["model"] = self.fallback_model
        try:
            response = self._client.chat.completions.create(**kwargs)
            if span:
                usage = _extract_usage(response)
                span.set_tokens(
                    prompt=usage.get("prompt_tokens", 0),
                    completion=usage.get("completion_tokens", 0),
                )
            return response
        except Exception:
            raise


# ===== Async Client =====

class AsyncLLMClient:
    """
    Async LLM client with retry, fallback, and streaming support.

    All LLM calls in async FastAPI handlers should go through this class.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        fallback_model: Optional[str] = None,
        timeout: float = 60.0,
        first_token_timeout: float = 15.0,
    ):
        self.model = model
        self.fallback_model = fallback_model
        self.first_token_timeout = first_token_timeout
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        response_format: Optional[Dict] = None,
        span=None,
    ) -> Any:
        """
        Async chat completion with retry + fallback.

        Automatically trims messages to fit the model's context window.
        Retries transient errors up to 3 times with exponential backoff.
        Falls back to fallback_model if primary fails after all retries.
        """
        use_model = model or self.model
        messages = trim_messages(messages, model=use_model, completion_reserve=max_tokens)
        kwargs: Dict[str, Any] = dict(
            model=use_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format

        tracer = get_tracer()
        span = span or tracer.current_span()

        async def _call():
            return await self._client.chat.completions.create(**kwargs)

        # Try primary model with retry
        try:
            response = await _retry_with_backoff(_call)
            if span:
                usage = _extract_usage(response)
                span.set_tokens(
                    prompt=usage.get("prompt_tokens", 0),
                    completion=usage.get("completion_tokens", 0),
                )
            return response
        except Exception as exc:
            if not self.fallback_model or use_model == self.fallback_model:
                raise
            logger.warning(
                f"[LLM Fallback] Primary model {use_model} failed: {exc}. "
                f"Trying fallback {self.fallback_model}"
            )

        # Try fallback model
        kwargs["model"] = self.fallback_model

        async def _call_fallback():
            return await self._client.chat.completions.create(**kwargs)

        try:
            response = await _retry_with_backoff(_call_fallback)
            if span:
                usage = _extract_usage(response)
                span.set_tokens(
                    prompt=usage.get("prompt_tokens", 0),
                    completion=usage.get("completion_tokens", 0),
                )
            return response
        except Exception:
            raise

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        first_token_timeout: Optional[float] = None,
        span=None,
    ) -> AsyncIterator:
        """
        Async streaming chat completion with first-token timeout.

        Automatically trims messages to fit the model's context window.
        Yields chunks as they arrive. If the first chunk doesn't arrive
        within first_token_timeout, the request is cancelled and the
        fallback model is tried.
        """
        use_model = model or self.model
        messages = trim_messages(messages, model=use_model, completion_reserve=max_tokens)
        ft_timeout = first_token_timeout or self.first_token_timeout

        kwargs: Dict[str, Any] = dict(
            model=use_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        tracer = get_tracer()
        span = span or tracer.current_span()
        prompt_tokens = 0
        completion_tokens = 0

        async def _stream_with_timeout(mdl: str):
            nonlocal prompt_tokens, completion_tokens
            kwargs["model"] = mdl
            stream = await self._client.chat.completions.create(**kwargs)

            first_chunk_received = False
            try:
                async for chunk in stream:
                    if not chunk.choices:
                        # Usage chunk at the end (some models send it)
                        if hasattr(chunk, "usage") and chunk.usage:
                            prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                            completion_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
                        continue

                    if not first_chunk_received:
                        first_chunk_received = True
                    yield chunk
            except Exception as exc:
                if not first_chunk_received and _should_retry(exc):
                    logger.warning(f"[LLM Stream] First chunk error, will retry: {exc}")
                    raise
                raise

        # Try primary
        try:
            async for chunk in _stream_with_timeout(use_model):
                yield chunk
            if span and (prompt_tokens or completion_tokens):
                span.set_tokens(prompt=prompt_tokens, completion=completion_tokens)
            return
        except Exception as exc:
            if not self.fallback_model or use_model == self.fallback_model:
                raise
            logger.warning(
                f"[LLM Stream Fallback] {use_model} failed: {exc}. "
                f"Trying {self.fallback_model}"
            )

        # Try fallback
        try:
            async for chunk in _stream_with_timeout(self.fallback_model):
                yield chunk
            if span and (prompt_tokens or completion_tokens):
                span.set_tokens(prompt=prompt_tokens, completion=completion_tokens)
        except Exception:
            raise


# ===== Client Registry (singleton) =====

_registry: Dict[str, Union[SyncLLMClient, AsyncLLMClient]] = {}


def get_async_llm(name: str = "default") -> AsyncLLMClient:
    """Get or create a named async LLM client."""
    key = f"async:{name}"
    if key not in _registry:
        fallback = os_get_fallback_model()
        _registry[key] = AsyncLLMClient(
            api_key=settings.GEMINI_API_KEY,
            base_url=settings.GEMINI_BASE_URL,
            model=settings.GEMINI_MODEL,
            fallback_model=fallback,
            timeout=float(os_get("LLM_TIMEOUT", "60")),
            first_token_timeout=float(os_get("LLM_FIRST_TOKEN_TIMEOUT", "15")),
        )
    return _registry[key]


def get_sync_llm(name: str = "default") -> SyncLLMClient:
    """Get or create a named sync LLM client."""
    key = f"sync:{name}"
    if key not in _registry:
        fallback = os_get_fallback_model()
        _registry[key] = SyncLLMClient(
            api_key=settings.GEMINI_API_KEY,
            base_url=settings.GEMINI_BASE_URL,
            model=settings.GEMINI_MODEL,
            fallback_model=fallback,
            timeout=float(os_get("LLM_TIMEOUT", "60")),
        )
    return _registry[key]


def get_vision_llm() -> SyncLLMClient:
    """Get or create the vision model client."""
    key = "sync:vision"
    if key not in _registry:
        if not settings.VISION_API_KEY:
            return get_sync_llm("default")
        _registry[key] = SyncLLMClient(
            api_key=settings.VISION_API_KEY,
            base_url=settings.VISION_BASE_URL,
            model=settings.VISION_MODEL,
            timeout=30.0,
        )
    return _registry[key]


# ---- Helpers ----

import os


def os_get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def os_get_fallback_model() -> Optional[str]:
    return os.getenv("LLM_FALLBACK_MODEL") or None
