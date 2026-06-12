# -*- coding: utf-8 -*-
"""
DocMind Chat API

Refactored to use LightRAG multi-mode query:
  - local/global/hybrid/mix/bypass modes
  - Vector recall + graph traversal
  - SSE streaming support
  - Standardized response format
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json
import asyncio

from config.settings import settings
from core.chat_history import ChatHistoryStore
from core.memory import init_memory_service, get_memory_service
from core.context_manager import get_context_manager
from core.adaptive_retrieval import get_adaptive_retrieval
from infrastructure.response import success_response
from infrastructure.llm_client import get_async_llm
from infrastructure.tracing import get_tracer

logger = logging.getLogger(__name__)

router = APIRouter()

vision_client = None
if settings.VISION_API_KEY:
    from infrastructure.llm_client import get_vision_llm
    vision_client = get_vision_llm()
    logger.info(f"Vision model client initialized: {settings.VISION_MODEL}")
else:
    logger.info("Vision model not configured, will use text-only mode")

history_store = ChatHistoryStore(
    host=settings.MYSQL_HOST,
    port=settings.MYSQL_PORT,
    user=settings.MYSQL_USER,
    password=settings.MYSQL_PASSWORD,
    database=settings.MYSQL_DATABASE,
)

memory_service = init_memory_service(
    db_path=str(settings.BASE_DIR / "memory_db.json"),
    api_key=settings.GEMINI_API_KEY,
    base_url=settings.GEMINI_BASE_URL,
    model=settings.GEMINI_MODEL,
)

active_streams: Dict[str, bool] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    session_title: Optional[str] = None
    query_mode: str = "mix"
    """LightRAG query mode: local, global, hybrid, naive, mix, bypass"""
    chat_mode: str = "kb"
    """Chat mode: 'kb' for knowledge base retrieval, 'direct' for direct LLM chat"""


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]


DIRECT_CHAT_SYSTEM_PROMPT = None  # loaded from PROMPTS registry below


def _get_direct_chat_system_prompt() -> str:
    """Get the direct chat system prompt from the centralized registry."""
    global DIRECT_CHAT_SYSTEM_PROMPT
    if DIRECT_CHAT_SYSTEM_PROMPT is None:
        from core.prompt_templates import PROMPTS
        DIRECT_CHAT_SYSTEM_PROMPT = PROMPTS.get("DIRECT_CHAT_SYSTEM", (
            "You are a helpful research assistant. Always respond in the same language as the user's question. "
            "If the user asks in Chinese, you MUST answer in Chinese. "
            "Provide clear, well-structured answers with proper formatting. "
            "Mathematical formulas MUST use LaTeX: $inline$ for inline math, $$display$$ for display math. "
            "When comparing multiple items, use a properly formatted Markdown table."
        ))
    return DIRECT_CHAT_SYSTEM_PROMPT


def _get_rag():
    """Get the global RAGAnything instance."""
    from core.raganything import _rag_instance
    return _rag_instance


def _get_memory_context(query: str) -> str:
    """Retrieve relevant long-term memory context for the query."""
    if not memory_service:
        return ""
    try:
        return memory_service.retrieve_context(query, top_k=5)
    except Exception as e:
        logger.warning(f"Memory retrieval failed (non-critical): {e}")
        return ""


def _has_lightrag() -> bool:
    """Synchronously check if LightRAG instance is initialized and ready."""
    rag = _get_rag()
    if rag is None:
        logger.info("[HasDocs] RAG instance is None")
        return False
    if not hasattr(rag, 'lightrag') or rag.lightrag is None:
        logger.info("[HasDocs] LightRAG instance not initialized (no documents processed yet)")
        return False
    return True


async def _has_documents() -> bool:
    """Check if there are any documents in the knowledge base."""
    if not _has_lightrag():
        return False

    rag = _get_rag()
    try:
        # Method 1: Check lightrag.doc_status
        if hasattr(rag.lightrag, 'doc_status'):
            doc_status = rag.lightrag.doc_status
            all_docs = await doc_status.get_all()
            count = len(all_docs) if all_docs else 0
            logger.info(f"[HasDocs] Found {count} documents via doc_status")
            return count > 0

        # Method 2: Check working directory for graphml files
        working_dir = getattr(rag.lightrag, 'working_dir', None)
        if working_dir:
            from pathlib import Path
            graphml_files = list(Path(working_dir).glob("*.graphml"))
            logger.info(f"[HasDocs] Found {len(graphml_files)} graphml files in {working_dir}")
            if graphml_files:
                return True

        logger.warning("[HasDocs] No documents found via any method")
        return False

    except Exception as e:
        logger.error(f"[HasDocs] Error checking documents: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the knowledge base using LightRAG multi-mode query."""
    tracer = get_tracer()
    with tracer.span("chat.request", mode=request.chat_mode, session_id=request.session_id or "") as span:
        try:
            history = []
            if request.session_id:
                history = await history_store.async_get_session_messages(request.session_id)

            if request.chat_mode == "direct":
                logger.info("[Chat] Direct mode - skipping KB")
                answer = await _direct_chat(request, history, span)
            else:
                has_docs = await _has_documents()
                logger.info(f"[Chat] KB mode - has_docs={has_docs}")
                if has_docs:
                    answer = await _rag_query_with_fallback(request, history, span)
                else:
                    logger.info("[Chat] No docs in KB, using direct chat")
                    answer = await _direct_chat(request, history, span)

            span.set_output(answer[:200] if answer else "")

            # Save conversation history (async)
            if request.session_id:
                import uuid
                await history_store.async_save_session(request.session_id, request.session_id)
                await history_store.async_save_message(str(uuid.uuid4()), request.session_id, "user", request.message)
                await history_store.async_save_message(str(uuid.uuid4()), request.session_id, "assistant", answer)

            # Extract memories asynchronously
            if request.session_id and memory_service:
                try:
                    memory_service.extract_from_turn(request.session_id, request.message, answer)
                except Exception as e:
                    logger.warning(f"Memory extraction failed (non-critical): {e}")

            return ChatResponse(response=answer, sources=[])

        except Exception as e:
            logger.error(f"Chat API error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


async def _rag_query_with_fallback(request: ChatRequest, history: list, parent_span=None) -> str:
    """Query using LightRAG, fallback to direct LLM if no relevant results found.

    Key fix: RAG query is the CLEAN user question only (no history mangled in).
    History context is handled by ContextManager when building LLM messages.
    """
    tracer = get_tracer()
    rag = _get_rag()
    if rag is None or not hasattr(rag, 'lightrag') or rag.lightrag is None:
        logger.warning("[RAG] No LightRAG instance, falling back to direct LLM")
        return await _direct_chat(request, history, parent_span)

    mode = request.query_mode or "mix"

    with tracer.start_span("rag.query", mode=mode) as rag_span:
        if parent_span:
            parent_span.children.append(rag_span)
        try:
            # RAG query = just the user's current question (no history concatenated)
            query = request.message
            system_prompt = _build_kb_system_prompt(request.message)

            logger.info(f"[RAG Query] Mode: {mode}, Query: {query[:200]}")

            result = await rag.aquery(
                query,
                mode=mode,
                system_prompt=system_prompt,
            )

            logger.info(f"[RAG Query] Result length: {len(result) if result else 0}")

            # Check if result indicates no relevant information
            if not result or "couldn't find relevant information" in result.lower() or "no relevant" in result.lower():
                logger.info("[RAG Query] No relevant results, falling back to direct LLM")
                rag_span.finish(status="ok", error="no_relevant_results")
                return await _direct_chat(request, history, parent_span)

            rag_span.set_output(result[:200] if result else "")
            rag_span.finish()
            return result

        except ValueError as e:
            logger.error(f"[RAG Query] LightRAG not ready: {e}")
            rag_span.finish(status="error", error=str(e))
            return await _direct_chat(request, history, parent_span)
        except Exception as e:
            logger.error(f"[RAG Query] Failed, falling back to direct LLM: {e}")
            import traceback
            logger.error(traceback.format_exc())
            rag_span.finish(status="error", error=str(e))
            return await _direct_chat(request, history, parent_span)


def _build_kb_system_prompt(message: str) -> str:
    """Build system prompt for knowledge base mode, with detailed response support."""
    from core.prompt_templates import PROMPTS

    base_prompt = PROMPTS.get("KB_CHAT_SYSTEM_BASE", "")

    is_detailed = any(keyword in message.lower() for keyword in ["详细", "detailed", "详细地", "详细说明", "详细解释"])
    if is_detailed:
        base_prompt += PROMPTS.get("KB_CHAT_SYSTEM_DETAILED_SUFFIX", "")

    return base_prompt


async def _direct_chat(request: ChatRequest, history: list, parent_span=None) -> str:
    """Direct chat without knowledge base — uses ContextManager for token-budget-aware context."""
    tracer = get_tracer()
    llm = get_async_llm()
    ctx = get_context_manager(model=llm.model)

    # Retrieve long-term memory context
    memory_context = _get_memory_context(request.message)

    # Build messages with token budget awareness
    messages, meta = ctx.build_messages(
        system_prompt=_get_direct_chat_system_prompt(),
        history=history,
        current_message=request.message,
        memory_context=memory_context,
        model=llm.model,
        session_id=request.session_id or "",
    )

    with tracer.span("llm.direct_chat", model=llm.model, **meta) as span:
        if parent_span:
            parent_span.children.append(span)
        try:
            response = await llm.chat(
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
                span=span,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Direct chat failed: {e}")
            raise


@router.post("/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    """SSE streaming chat with LightRAG."""
    tracer = get_tracer()
    stream_id = f"{request.session_id}_{id(request)}"
    active_streams[stream_id] = True

    async def generate():
        with tracer.span("chat.stream", mode=request.chat_mode, session_id=request.session_id or "") as span:
            try:
                # Load conversation history (async)
                history = []
                if request.session_id:
                    history = await history_store.async_get_session_messages(request.session_id)

                logger.info(f"[Stream] chat_mode={request.chat_mode}, session_id={request.session_id}")
                logger.info(f"[Stream] history count={len(history)}")

                full_response = ""

                if request.chat_mode == "direct":
                    logger.info("[Stream] Using DIRECT chat mode")
                    yield f"{json.dumps({'type': 'status', 'content': 'Generating response...'})}\n\n"
                    async for chunk in _stream_direct_chat_generator(history, request.message, stream_id, http_request, span, session_id=request.session_id or ""):
                        full_response += _extract_chunk_content(chunk)
                        yield chunk
                else:
                    logger.info("[Stream] Using KB chat mode")
                    has_docs = await _has_documents()
                    logger.info(f"[Stream] has_docs={has_docs}")
                    if has_docs:
                        yield f"{json.dumps({'type': 'status', 'content': 'Searching knowledge base...'})}\n\n"
                        async for chunk in _stream_rag_with_fallback(history, request.message, request.query_mode, stream_id, http_request, span, session_id=request.session_id or ""):
                            full_response += _extract_chunk_content(chunk)
                            yield chunk
                    else:
                        logger.info("[Stream] No docs in KB, using direct chat")
                        yield f"{json.dumps({'type': 'status', 'content': 'No documents in KB, using direct chat...'})}\n\n"
                        async for chunk in _stream_direct_chat_generator(history, request.message, stream_id, http_request, span, session_id=request.session_id or ""):
                            full_response += _extract_chunk_content(chunk)
                            yield chunk

                span.set_output(full_response[:200] if full_response else "")

                # Save to history (async)
                if request.session_id and full_response:
                    import uuid
                    await history_store.async_save_session(request.session_id, request.session_id)
                    await history_store.async_save_message(str(uuid.uuid4()), request.session_id, "user", request.message)
                    await history_store.async_save_message(str(uuid.uuid4()), request.session_id, "assistant", full_response)

                yield f"{json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"{json.dumps({'type': 'error', 'data': str(e)})}\n\n"
                yield f"{json.dumps({'type': 'done'})}\n\n"
            finally:
                active_streams.pop(stream_id, None)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _extract_chunk_content(chunk: str) -> str:
    """Extract content from a JSON chunk string."""
    try:
        parsed = json.loads(chunk)
        if parsed.get("type") == "chunk":
            return parsed.get("data", "")
    except Exception:
        pass
    return ""


async def _stream_rag_with_fallback(history, message, query_mode, stream_id, http_request, parent_span=None, session_id: str = ""):
    """Stream RAG response with true token-by-token streaming.

    Two-phase approach:
      Phase 1: Retrieve context via LightRAG (non-streaming, fast)
      Phase 2: Stream the final answer token-by-token via LLM (true streaming)

    Key fix: RAG query is the CLEAN user question only (no history mangled in).
    History context is handled by ContextManager in _stream_rag_answer.
    """
    tracer = get_tracer()
    rag = _get_rag()
    if rag is None or not hasattr(rag, 'lightrag') or rag.lightrag is None:
        logger.warning("[Stream RAG] No LightRAG instance, falling back to direct LLM")
        async for chunk in _stream_direct_chat_generator(history, message, stream_id, http_request, parent_span, session_id=session_id):
            yield chunk
        return

    with tracer.start_span("rag.stream_query", mode=query_mode or "mix") as rag_span:
        if parent_span:
            parent_span.children.append(rag_span)
        try:
            # RAG query = just the user's current question (no history concatenated)
            query = message
            system_prompt = _build_kb_system_prompt(message)

            logger.info(f"[Stream RAG] Mode: {query_mode or 'mix'}, Query: {query[:200]}")

            # Phase 1: Retrieve context (non-streaming)
            result = await rag.aquery(
                query,
                mode=query_mode or "mix",
                system_prompt=system_prompt,
            )

            logger.info(f"[Stream RAG] Result length: {len(result) if result else 0}")

            # Check if result indicates no relevant information
            if not result or "couldn't find relevant information" in result.lower() or "no relevant" in result.lower():
                logger.info("[Stream RAG] No relevant results, falling back to direct LLM")
                rag_span.finish(status="ok", error="no_relevant_results")
                async for chunk in _stream_direct_chat_generator(history, message, stream_id, http_request, parent_span, session_id=session_id):
                    yield chunk
                return

            rag_span.set_output(result[:200] if result else "")
            rag_span.finish()

            # Phase 2: Stream the RAG result through LLM for token-by-token output
            # Pass history so ContextManager can inject conversation context
            async for chunk in _stream_rag_answer(result, message, history, stream_id, http_request, parent_span, session_id=session_id):
                yield chunk

        except ValueError as e:
            logger.error(f"[Stream RAG] LightRAG not ready: {e}")
            rag_span.finish(status="error", error=str(e))
            async for chunk in _stream_direct_chat_generator(history, message, stream_id, http_request, parent_span, session_id=session_id):
                yield chunk
        except Exception as e:
            logger.error(f"[Stream RAG] Failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            rag_span.finish(status="error", error=str(e))
            yield f"{json.dumps({'type': 'status', 'content': 'RAG failed, using LLM...'})}\n\n"
            async for chunk in _stream_direct_chat_generator(history, message, stream_id, http_request, parent_span, session_id=session_id):
                yield chunk


async def _stream_rag_answer(rag_result: str, original_query: str, history: list, stream_id: str, http_request, parent_span=None, session_id: str = ""):
    """Stream RAG result token-by-token through the LLM for true streaming output.

    Uses ContextManager to build messages with conversation history and memory context,
    so the LLM is aware of the conversation context when presenting the RAG answer.
    """
    tracer = get_tracer()
    llm = get_async_llm()

    # If the RAG result is short enough, just chunk it directly (avoid extra LLM call)
    if len(rag_result) <= 200:
        for i in range(0, len(rag_result), 50):
            if not active_streams.get(stream_id, False):
                break
            if http_request and await http_request.is_disconnected():
                break
            chunk_text = rag_result[i:i+50]
            yield f"{json.dumps({'type': 'chunk', 'data': chunk_text})}\n\n"
        return

    # For longer results, stream through LLM for true token-by-token delivery
    # Use ContextManager to build context-aware messages with Lost-in-the-Middle mitigation
    from core.prompt_templates import PROMPTS
    ctx = get_context_manager(model=llm.model)

    # Retrieve memory context
    memory_context = _get_memory_context(original_query)

    # Build the RAG-specific system prompt
    rag_system = PROMPTS.get("RAG_STREAM_PASS_THROUGH", "")

    # Use ContextManager to build messages with history + memory + RAG context
    # Pass rag_result as rag_context for sandwich reordering
    presentation_message = f"User question: {original_query}\n\nPlease present the following RAG answer clearly, maintaining all key information:"
    messages, meta = ctx.build_messages(
        system_prompt=rag_system,
        history=history,
        current_message=presentation_message,
        memory_context=memory_context,
        model=llm.model,
        rag_context=rag_result,
        session_id=session_id,
    )

    with tracer.span("llm.rag_stream", model=llm.model, **meta) as span:
        if parent_span:
            parent_span.children.append(span)
        try:
            async for chunk in llm.chat_stream(
                messages=messages,
                temperature=0.1,
                max_tokens=4096,
                span=span,
            ):
                if not active_streams.get(stream_id, False):
                    break
                if http_request and await http_request.is_disconnected():
                    break
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield f"{json.dumps({'type': 'chunk', 'data': delta.content})}\n\n"
        except Exception as e:
            # If streaming fails, fall back to chunking the raw result
            logger.warning(f"RAG stream-through-LLM failed, chunking raw result: {e}")
            for i in range(0, len(rag_result), 50):
                if not active_streams.get(stream_id, False):
                    break
                if http_request and await http_request.is_disconnected():
                    break
                chunk_text = rag_result[i:i+50]
                yield f"{json.dumps({'type': 'chunk', 'data': chunk_text})}\n\n"


async def _stream_direct_chat_generator(history, message, stream_id, http_request, parent_span=None, session_id: str = ""):
    """Generator for direct LLM chat response — true token-by-token streaming with ContextManager."""
    tracer = get_tracer()
    llm = get_async_llm()
    ctx = get_context_manager(model=llm.model)

    # Retrieve long-term memory context
    memory_context = _get_memory_context(message)

    # Build messages with token budget awareness
    messages, meta = ctx.build_messages(
        system_prompt=_get_direct_chat_system_prompt(),
        history=history,
        current_message=message,
        memory_context=memory_context,
        model=llm.model,
        session_id=session_id,
    )

    with tracer.span("llm.stream_chat", model=llm.model, **meta) as span:
        if parent_span:
            parent_span.children.append(span)
        try:
            async for chunk in llm.chat_stream(
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
                span=span,
            ):
                if not active_streams.get(stream_id, False):
                    break
                if http_request and await http_request.is_disconnected():
                    break
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield f"{json.dumps({'type': 'chunk', 'data': delta.content})}\n\n"
        except Exception as e:
            logger.error(f"Stream direct chat failed: {e}")
            raise


@router.get("/query_modes")
async def get_query_modes():
    """Get available query modes."""
    return {
        "modes": [
            {"id": "local", "name": "Local", "description": "Focus on specific entities and their immediate context"},
            {"id": "global", "name": "Global", "description": "Summarize across all related information"},
            {"id": "hybrid", "name": "Hybrid", "description": "Combine local specificity and global context"},
            {"id": "naive", "name": "Naive", "description": "Basic vector similarity search"},
            {"id": "mix", "name": "Mix (Recommended)", "description": "Best balance of all retrieval strategies"},
            {"id": "bypass", "name": "Bypass", "description": "Skip retrieval, use LLM direct"},
        ]
    }


@router.get("/sessions")
async def list_sessions():
    """List all chat sessions."""
    sessions = await history_store.async_list_sessions()
    return success_response(data={"sessions": sessions, "total": len(sessions)})


@router.get("/sessions/{session_id}")
async def get_session_messages(session_id: str):
    """Get messages for a specific session."""
    messages = await history_store.async_get_session_messages(session_id)
    return success_response(data={"messages": messages, "total": len(messages)})


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session."""
    await history_store.async_delete_session(session_id)
    return success_response(message="Session deleted successfully")
