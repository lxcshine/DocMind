# -*- coding: utf-8 -*-
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json
import base64
import os
import re
import asyncio
from openai import OpenAI, AsyncOpenAI

from config.settings import settings
from core.tree_index import tree_index_store
from core.agentic_retrieve import AgenticRetriever
from core.chat_history import ChatHistoryStore
from core.memory import init_memory_service, get_memory_service

logger = logging.getLogger(__name__)

router = APIRouter()

llm_client = OpenAI(
    api_key=settings.GEMINI_API_KEY,
    base_url=settings.GEMINI_BASE_URL,
    timeout=60.0,
)

async_llm_client = AsyncOpenAI(
    api_key=settings.GEMINI_API_KEY,
    base_url=settings.GEMINI_BASE_URL,
    timeout=60.0,
)

vision_client = None
if settings.VISION_API_KEY:
    vision_client = OpenAI(
        api_key=settings.VISION_API_KEY,
        base_url=settings.VISION_BASE_URL,
        timeout=30.0,
    )
    logger.info(f"Vision model client initialized: {settings.VISION_MODEL}")
else:
    logger.info("Vision model not configured, will use text-only mode")

async_vision_client = None
if settings.VISION_API_KEY:
    async_vision_client = AsyncOpenAI(
        api_key=settings.VISION_API_KEY,
        base_url=settings.VISION_BASE_URL,
        timeout=30.0,
    )

agentic_retriever = AgenticRetriever(
    api_key=settings.GEMINI_API_KEY,
    base_url=settings.GEMINI_BASE_URL,
    model=settings.GEMINI_MODEL,
)

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
    top_k: int = 5


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]


DETAILED_KEYWORDS = [
    "\u8be6\u7ec6", "detailed", "elaborate", "comprehensive",
    "\u6df1\u5165", "thorough", "\u5168\u9762", "extensive",
    "\u8be6\u5c3d", "\u4ed4\u7ec6"
]

DIRECT_CHAT_SYSTEM_PROMPT = (
    "You are a helpful research assistant. Always respond in the same language as the user's question. "
    "If the user asks in Chinese, you MUST answer in Chinese. "
    "Provide clear, well-structured answers with proper formatting. "
    "Mathematical formulas MUST use LaTeX: $inline$ for inline, $$display$$ for display math. "
    "When comparing multiple items, use a properly formatted Markdown table."
)

STREAM_ANSWER_SYSTEM_PROMPT = (
    "You are a helpful research assistant. Always respond in the same language as the user's question. "
    "If the user asks in Chinese, you MUST answer in Chinese. "
    "Answer the user's question based on the provided context from the knowledge base. "
    "Mathematical formulas MUST use LaTeX: $inline$ for inline, $$display$$ for display math. "
    "When comparing multiple items, use a properly formatted Markdown table with aligned columns. "
    "Provide clear, well-structured answers with proper formatting."
)

VISION_SYSTEM_PROMPT = (
    "You are a helpful research assistant. Always respond in the same language as the user's question. "
    "If the user asks in Chinese, you MUST answer in Chinese. "
    "You are given images of PDF pages from the user's knowledge base as visual context. "
    "Look at the images carefully and use the information in them to answer the question. "
    "When the pages contain mathematical formulas, tables, or charts, read them directly from the images "
    "and present them in proper LaTeX format (use $$ for display math and $ for inline math). "
    "When comparing multiple items, use a properly formatted Markdown table with aligned columns. "
    "Provide clear, well-structured answers with proper formatting."
)


def is_detailed_request(message: str) -> bool:
    return any(kw in message.lower() for kw in DETAILED_KEYWORDS)


def _normalize_sources(agent_sources: List[Dict]) -> List[Dict]:
    return [
        {
            "content": s.get("section", "")[:300],
            "metadata": {
                "doc_title": s.get("doc_name", ""),
                "chunk_id": s.get("doc_id", ""),
            },
            "doc_title": s.get("doc_name", ""),
            "section_title": s.get("section", ""),
            "pages": s.get("pages", ""),
        }
        for s in agent_sources
    ]


def _collect_page_images(sources: List[Dict]) -> List[str]:
    image_paths = []
    seen = set()
    for s in sources:
        pages = s.get("pages", "")
        doc_id = s.get("doc_id", "") or s.get("metadata", {}).get("chunk_id", "")
        if doc_id and pages:
            paths = tree_index_store.get_page_images(doc_id, pages)
            for p in paths:
                if p not in seen:
                    image_paths.append(p)
                    seen.add(p)
    return image_paths


def _needs_vision(context: str) -> bool:
    if not context or not context.strip():
        return False

    formula_patterns = [
        r'\$\$', r'\$[^$]+\$', r'\\frac\{', r'\\sum', r'\\int',
        r'\\prod', r'\\sqrt\{', r'\\alpha', r'\\beta', r'\\gamma',
        r'\\delta', r'\\epsilon', r'\\theta', r'\\lambda', r'\\mu',
        r'\\sigma', r'\\omega', r'\\phi', r'\\pi', r'\\infty',
        r'\\partial', r'\\nabla', r'\\cdot', r'\\times', r'\\pm',
        r'\\leq', r'\\geq', r'\\neq', r'\\approx', r'\\equiv',
        r'\\begin\{', r'\\end\{', r'\\text\{', r'\\mathbb\{', r'\\mathbf\{',
        r'\\mathcal\{', r'\\left', r'\\right', r'\\lim', r'\\to',
        r'\\rightarrow', r'\\Rightarrow', r'\\subset', r'\\forall',
        r'\\exists', r'\\in', r'\\notin', r'\\cup', r'\\cap',
    ]

    table_patterns = [
        r'\|[^|]+\|[^|]+\|', r'<table[^>]*>', r'<tr[^>]*>', r'<td[^>]*>',
    ]

    for pattern in formula_patterns:
        if re.search(pattern, context):
            return True

    for pattern in table_patterns:
        if re.search(pattern, context):
            return True

    return False


async def _generate_vision_answer_async(
    question: str,
    image_paths: List[str],
    history: List[Dict],
    detailed: bool = False,
) -> Optional[str]:
    if async_vision_client is None:
        return None

    system_prompt = VISION_SYSTEM_PROMPT
    if detailed:
        system_prompt += (
            "\n\nIMPORTANT: The user has requested a DETAILED answer. "
            "You MUST provide a comprehensive, thorough response."
        )

    messages = [{"role": "system", "content": system_prompt}]

    recent_history = history[-20:] if len(history) > 20 else history
    for h in recent_history:
        role = h.get("role", "user")
        content = h.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    user_content = [{"type": "text", "text": f"Question: {question}\n\nPlease look at the provided PDF page images and answer the question based on what you see."}]

    valid_images = [p for p in image_paths[:10] if os.path.exists(p)]
    if not valid_images:
        logger.warning("No valid page images found for vision model")
        return None

    for img_path in valid_images:
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
            })

    messages.append({"role": "user", "content": user_content})

    try:
        response = await async_vision_client.chat.completions.create(
            model=settings.VISION_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=4096 if detailed else 2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Vision model failed ({settings.VISION_MODEL}): {e}")
        error_msg = str(e).lower()
        if "insufficient" in error_msg or "balance" in error_msg or "\u4f59\u989d" in str(e):
            logger.error("Vision API key has insufficient balance. Please recharge.")
            return "vision_balance_error"
        if "image" in error_msg or "not supported" in error_msg or "invalid" in error_msg:
            logger.error(
                f"VISION_MODEL '{settings.VISION_MODEL}' may not support vision. "
                f"Use a vision-capable model like 'gpt-4o' or 'gpt-4o-mini'."
            )
        if "timed out" in error_msg or "connect" in error_msg:
            logger.error(f"VISION_BASE_URL '{settings.VISION_BASE_URL}' is unreachable.")
        return None


def _generate_vision_answer(
    question: str,
    image_paths: List[str],
    history: List[Dict],
    detailed: bool = False,
) -> Optional[str]:
    if vision_client is None:
        return None

    system_prompt = VISION_SYSTEM_PROMPT
    if detailed:
        system_prompt += (
            "\n\nIMPORTANT: The user has requested a DETAILED answer. "
            "You MUST provide a comprehensive, thorough response."
        )

    messages = [{"role": "system", "content": system_prompt}]

    recent_history = history[-20:] if len(history) > 20 else history
    for h in recent_history:
        role = h.get("role", "user")
        content = h.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    user_content = [{"type": "text", "text": f"Question: {question}\n\nPlease look at the provided PDF page images and answer the question based on what you see."}]

    valid_images = [p for p in image_paths[:10] if os.path.exists(p)]
    if not valid_images:
        logger.warning("No valid page images found for vision model")
        return None

    for img_path in valid_images:
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
            })

    messages.append({"role": "user", "content": user_content})

    try:
        response = vision_client.chat.completions.create(
            model=settings.VISION_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=4096 if detailed else 2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Vision model failed ({settings.VISION_MODEL}): {e}")
        return None


def _build_direct_chat_messages(
    message: str,
    history: List[Dict],
    memory_context: str = "",
    max_history: int = 20,
) -> List[Dict]:
    system_prompt = DIRECT_CHAT_SYSTEM_PROMPT

    detailed = is_detailed_request(message)
    if detailed:
        system_prompt += (
            "\n\nIMPORTANT: The user has requested a DETAILED answer. "
            "You MUST provide a comprehensive, thorough response."
        )

    if memory_context:
        system_prompt += (
            "\n\n**User Profile & Past Context (from Memory):**\n"
            f"{memory_context}\n"
            "Use this information to personalize your response."
        )

    messages = [{"role": "system", "content": system_prompt}]
    recent_history = history[-max_history:] if len(history) > max_history else history
    for h in recent_history:
        role = h.get("role", "user")
        content = h.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})
    return messages


def _build_context_answer_messages(
    context: str,
    question: str,
    history: List[Dict],
    max_history: int = 10,
) -> List[Dict]:
    messages = [{"role": "system", "content": STREAM_ANSWER_SYSTEM_PROMPT}]
    recent_history = history[-max_history:] if len(history) > max_history else history
    for h in recent_history:
        role = h.get("role", "user")
        content = h.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({
        "role": "user",
        "content": (
            f"Context from knowledge base:\n\n{context}\n\n"
            f"User question: {question}\n\n"
            f"Please answer based on the context above."
        ),
    })
    return messages


async def _extract_memories_async(
    session_id: str,
    user_message: str,
    assistant_response: str,
) -> None:
    try:
        entries = memory_service.extract_memories(
            session_id, user_message, assistant_response
        )
        if entries:
            memory_service.add_entries(entries)
            logger.info(f"Saved {len(entries)} memory entries from chat")
    except Exception as e:
        logger.warning(f"Memory extraction skipped: {e}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        docs = tree_index_store.list_documents()
        doc_ids = [d["doc_id"] for d in docs]

        history = []
        if request.session_id:
            history = history_store.get_session_messages(request.session_id)

        if doc_ids:
            result = agentic_retriever.answer(
                query=request.message,
                doc_ids=doc_ids,
            )
            answer = result.get("answer", "")
            agent_sources = result.get("sources", [])
            sources = _normalize_sources(agent_sources)

            if vision_client is not None and _needs_vision(answer):
                image_paths = _collect_page_images(agent_sources)
                if image_paths:
                    logger.info(f"Vision mode: {len(image_paths)} page images (formulas/tables detected)")
                    detailed = is_detailed_request(request.message)
                    vision_answer = _generate_vision_answer(
                        question=request.message,
                        image_paths=image_paths,
                        history=history,
                        detailed=detailed,
                    )
                    if vision_answer:
                        answer = vision_answer
                    else:
                        logger.info("Vision model unavailable, keeping text answer")
                else:
                    logger.info("No page images found for formula/table pages, keeping text answer")
            else:
                logger.info("Text-only mode (no formulas/tables detected)")

            if not answer.strip():
                memory_context = memory_service.retrieve_context(request.message, top_k=5)
                messages = _build_direct_chat_messages(
                    request.message, history, memory_context
                )
                detailed = is_detailed_request(request.message)
                response = llm_client.chat.completions.create(
                    model=settings.GEMINI_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4096 if detailed else 2048,
                )
                answer = response.choices[0].message.content
                sources = []
        else:
            memory_context = memory_service.retrieve_context(request.message, top_k=5)
            messages = _build_direct_chat_messages(
                request.message, history, memory_context
            )
            detailed = is_detailed_request(request.message)
            response = llm_client.chat.completions.create(
                model=settings.GEMINI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=4096 if detailed else 2048,
            )
            answer = response.choices[0].message.content
            sources = []

        if request.session_id:
            history_store.save_session(
                request.session_id, request.session_title or request.message[:50])
            history_store.save_message(
                f"u_{request.session_id}_{len(history)}",
                request.session_id, "user", request.message
            )
            history_store.save_message(
                f"a_{request.session_id}_{len(history) + 1}",
                request.session_id, "assistant", answer, sources
            )

            asyncio.create_task(_extract_memories_async(
                request.session_id, request.message, answer
            ))

        return ChatResponse(response=answer, sources=sources)

    except Exception as e:
        logger.error(f"Chat API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest, req: Request):
    try:
        docs = tree_index_store.list_documents()
        doc_ids = [d["doc_id"] for d in docs]

        history = []
        if request.session_id:
            history = history_store.get_session_messages(request.session_id)

        session_id = request.session_id or "default"
        active_streams[session_id] = True

        if request.session_id:
            session_title = request.session_title or request.message[:50]
            history_store.save_session(session_id, session_title)
            history_store.save_message(
                f"u_{session_id}_{len(history)}",
                session_id, "user", request.message
            )

        thinking_steps = [
            "\u6b63\u5728\u7406\u89e3\u95ee\u9898...",
            "\u6b63\u5728\u68c0\u7d22\u77e5\u8bc6\u5e93...",
            "\u6b63\u5728\u5206\u6790\u76f8\u5173\u6587\u732e...",
            "\u6b63\u5728\u7ec4\u7ec7\u56de\u7b54...",
        ]

        async def generate():
            answer = ""
            sources: List[Dict] = []
            agent_sources: List[Dict] = []
            context = ""

            def _cancelled():
                return not active_streams.get(session_id, False)

            async def _client_lost():
                if _cancelled():
                    return True
                return await req.is_disconnected()

            async def _run_retrieve():
                import time
                t0 = time.time()
                try:
                    result = await asyncio.to_thread(
                        agentic_retriever.retrieve,
                        query=request.message,
                        doc_ids=doc_ids,
                    )
                except Exception as e:
                    logger.error(f"Retrieve in thread failed: {e}")
                    return {"context": "", "sources": []}
                logger.info(
                    f"Retrieval done in {time.time() - t0:.1f}s: "
                    f"{len(result.get('sources', []))} sources, "
                    f"{len(result.get('context', ''))} chars"
                )
                return result

            async def _run_vision_answer(img_paths, is_detailed):
                VISION_TIMEOUT = 30.0
                try:
                    return await asyncio.wait_for(
                        _generate_vision_answer_async(
                            question=request.message,
                            image_paths=img_paths,
                            history=history,
                            detailed=is_detailed,
                        ),
                        timeout=VISION_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Vision answer timed out after {VISION_TIMEOUT}s")
                    return None
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Vision answer failed: {e}")
                    return None

            async def _stream_text_answer(messages, is_detailed):
                nonlocal answer
                try:
                    stream = await async_llm_client.chat.completions.create(
                        model=settings.GEMINI_MODEL,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=4096 if is_detailed else 2048,
                        stream=True,
                    )
                    async for chunk in stream:
                        if _cancelled():
                            break
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            answer += content
                            yield json.dumps({"type": "chunk", "data": content}, ensure_ascii=False) + "\n"
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"LLM streaming error: {e}")
                    err = str(e)
                    if "1113" in err or "\u4f59\u989d" in err:
                        err = "\u26a0\ufe0f GLM API \u4f59\u989d\u4e0d\u8db3\uff0c\u8bf7\u524d\u5f80 open.bigmodel.cn \u5145\u503c\u3002\u5982\u5df2\u5145\u503c\uff0c\u8bf7\u68c0\u67e5 API Key \u914d\u7f6e\u3002"
                    elif "429" in err:
                        err = "\u8be5\u6a21\u578b\u5f53\u524d\u8bbf\u95ee\u91cf\u8fc7\u5927\uff0c\u8bf7\u60a8\u7a0d\u540e\u518d\u8bd5"
                    elif "401" in err or "403" in err:
                        err = "API\u5bc6\u94a5\u65e0\u6548\uff0c\u8bf7\u68c0\u67e5\u914d\u7f6e"
                    yield json.dumps({"type": "error", "data": err}, ensure_ascii=False) + "\n"

            if doc_ids:
                # Step 1: Understand question
                if await _client_lost():
                    yield json.dumps({"type": "stopped"}, ensure_ascii=False) + "\n"
                    return
                yield json.dumps({"type": "thinking", "data": thinking_steps[0]}, ensure_ascii=False) + "\n"
                await asyncio.sleep(0.3)

                # --- Step 2: retrieve knowledge base (running in thread, heartbeat polling) ---
                if await _client_lost():
                    yield json.dumps({"type": "stopped"}, ensure_ascii=False) + "\n"
                    return
                yield json.dumps({"type": "thinking", "data": thinking_steps[1]}, ensure_ascii=False) + "\n"

                retrieve_task = asyncio.create_task(_run_retrieve())
                RETRIEVE_TIMEOUT = 60.0
                HEARTBEAT_INTERVAL = 0.5

                while not retrieve_task.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(retrieve_task),
                            timeout=HEARTBEAT_INTERVAL,
                        )
                    except asyncio.TimeoutError:
                        if _cancelled():
                            retrieve_task.cancel()
                            yield json.dumps({"type": "stopped"}, ensure_ascii=False) + "\n"
                            return
                        continue

                try:
                    result = await asyncio.wait_for(
                        retrieve_task,
                        timeout=RETRIEVE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.error("Retrieval timed out after 60s")
                    retrieve_task.cancel()
                    yield json.dumps({"type": "error", "data": "\u68c0\u7d22\u8d85\u65f6\uff0c\u8bf7\u7a0d\u540e\u91cd\u8bd5"}, ensure_ascii=False) + "\n"
                    return

                context = result.get("context", "")
                agent_sources = result.get("sources", [])
                sources = _normalize_sources(agent_sources)

                if await _client_lost():
                    yield json.dumps({"type": "stopped"}, ensure_ascii=False) + "\n"
                    return

                # --- Step 3: 鍒嗘瀽鏂囩尞 ---
                yield json.dumps({"type": "thinking", "data": thinking_steps[2]}, ensure_ascii=False) + "\n"
                await asyncio.sleep(0.3)

                if await _client_lost():
                    yield json.dumps({"type": "stopped"}, ensure_ascii=False) + "\n"
                    return

                # --- Step 4: 缁勭粐鍥炵瓟 ---
                yield json.dumps({"type": "thinking", "data": thinking_steps[3]}, ensure_ascii=False) + "\n"
                await asyncio.sleep(0.3)

                if await _client_lost():
                    yield json.dumps({"type": "stopped"}, ensure_ascii=False) + "\n"
                    return

                yield json.dumps({"type": "sources", "data": sources}, ensure_ascii=False) + "\n"
                yield json.dumps({"type": "thinking_done"}, ensure_ascii=False) + "\n"

                # --- Answer generation ---
                detailed = is_detailed_request(request.message)
                use_vision = False

                if vision_client is not None and _needs_vision(context):
                    image_paths = _collect_page_images(agent_sources)
                    if image_paths:
                        logger.info(f"[Stream] Vision mode: {len(image_paths)} page images (formulas/tables detected)")
                        use_vision = True
                        vision_answer = await _run_vision_answer(image_paths, detailed)
                        if vision_answer and vision_answer != "vision_balance_error":
                            answer = vision_answer
                            for i in range(0, len(answer), 20):
                                if _cancelled():
                                    yield json.dumps({"type": "stopped"}, ensure_ascii=False) + "\n"
                                    return
                                yield json.dumps({"type": "chunk", "data": answer[i:i + 20]}, ensure_ascii=False) + "\n"
                        else:
                            logger.info("[Stream] Vision unavailable, falling back to text")
                            use_vision = False
                    else:
                        logger.info("[Stream] No page images for formula/table pages")
                else:
                    logger.info("[Stream] Text-only mode (no formulas/tables detected)")

                if not use_vision and context.strip():
                    messages = _build_context_answer_messages(context, request.message, history)
                    async for chunk_line in _stream_text_answer(messages, detailed):
                        if _cancelled():
                            yield json.dumps({"type": "stopped"}, ensure_ascii=False) + "\n"
                            return
                        yield chunk_line

                if not use_vision and not context.strip():
                    yield json.dumps({"type": "chunk", "data": "\u62b1\u6b49\uff0c\u672a\u5728\u77e5\u8bc6\u5e93\u4e2d\u627e\u5230\u76f8\u5173\u4fe1\u606f\u3002"}, ensure_ascii=False) + "\n"
            else:
                # No documents: direct chat
                for step in thinking_steps:
                    if await _client_lost():
                        yield json.dumps({"type": "stopped"}, ensure_ascii=False) + "\n"
                        return
                    yield json.dumps({"type": "thinking", "data": step}, ensure_ascii=False) + "\n"
                    await asyncio.sleep(0.3)

                yield json.dumps({"type": "sources", "data": []}, ensure_ascii=False) + "\n"
                yield json.dumps({"type": "thinking_done"}, ensure_ascii=False) + "\n"

                if await _client_lost():
                    yield json.dumps({"type": "stopped"}, ensure_ascii=False) + "\n"
                    return

                memory_context = memory_service.retrieve_context(request.message, top_k=5)
                messages = _build_direct_chat_messages(
                    request.message, history, memory_context
                )
                detailed = is_detailed_request(request.message)

                async for chunk_line in _stream_text_answer(messages, detailed):
                    if _cancelled():
                        yield json.dumps({"type": "stopped"}, ensure_ascii=False) + "\n"
                        return
                    yield chunk_line

            # --- Finalize ---
            if await _client_lost():
                yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
                active_streams.pop(session_id, None)
                return

            if answer.strip():
                history_store.save_message(
                    f"a_{session_id}_{len(history) + 1}",
                    session_id, "assistant", answer, sources)

                if request.session_id:
                    asyncio.create_task(_extract_memories_async(
                        request.session_id, request.message, answer
                    ))

            yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
            active_streams.pop(session_id, None)

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"Chat stream API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_chat(request: ChatRequest):
    session_id = request.session_id or "default"
    if session_id in active_streams:
        active_streams[session_id] = False
        return {"status": "stopped"}
    return {"status": "not_active"}


@router.get("/sessions")
async def list_sessions():
    sessions = history_store.list_sessions()
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    messages = history_store.get_session_messages(session_id)
    return {"session_id": session_id, "messages": messages}


@router.post("/sessions")
async def create_session():
    import time
    import random
    import string
    session_id = f"chat_{int(time.time())}_{''.join(random.choices(string.ascii_lowercase + string.digits, k=6))}"
    history_store.save_session(session_id, "New Conversation")
    return {"session_id": session_id, "title": "New Conversation"}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    history_store.delete_session(session_id)
    return {"status": "deleted"}
