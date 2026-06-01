"""
DocMind Backend - Memory API

Memory configuration, entry management, and knowledge extraction.
Inspired by RAGFlow's Memory module.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from core.memory import (
    get_memory_service,
    init_memory_service,
    MemoryEntry,
    MEMORY_TYPES,
    MEMORY_TYPE_LABELS,
    MEMORY_TYPE_COLORS,
)
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

memory_service = init_memory_service(
    db_path=str(settings.BASE_DIR / "memory_db.json"),
    api_key=settings.GEMINI_API_KEY,
    base_url=settings.GEMINI_BASE_URL,
    model=settings.GEMINI_MODEL,
)


class MemoryConfigRequest(BaseModel):
    active_types: Optional[List[str]] = None
    max_entries: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    forgetting_policy: Optional[str] = None
    system_prompt: Optional[str] = None


class MemoryEntryResponse(BaseModel):
    id: str
    type: str
    type_label: str
    type_color: str
    content: str
    keywords: List[str]
    source_session_id: str
    enabled: bool
    created_at: str
    updated_at: str


class PaginatedEntriesResponse(BaseModel):
    entries: List[MemoryEntryResponse]
    total: int
    page: int
    page_size: int


def _format_entry(entry: MemoryEntry) -> MemoryEntryResponse:
    return MemoryEntryResponse(
        id=entry.id,
        type=entry.type,
        type_label=MEMORY_TYPE_LABELS.get(entry.type, entry.type),
        type_color=MEMORY_TYPE_COLORS.get(entry.type, "default"),
        content=entry.content,
        keywords=entry.keywords,
        source_session_id=entry.source_session_id,
        enabled=entry.enabled,
        created_at=entry.created_at[:19].replace("T", " ") if entry.created_at else "",
        updated_at=entry.updated_at[:19].replace("T", " ") if entry.updated_at else "",
    )


# ====== Configuration ======


@router.get("/config")
async def get_memory_config():
    return memory_service.get_config()


@router.put("/config")
async def update_memory_config(request: MemoryConfigRequest):
    kwargs = {k: v for k, v in request.model_dump().items() if v is not None}
    return memory_service.update_config(**kwargs)


@router.get("/types")
async def get_memory_types():
    return {
        "types": [
            {
                "name": name,
                "label": MEMORY_TYPE_LABELS.get(name, name),
                "color": MEMORY_TYPE_COLORS.get(name, "default"),
            }
            for name in MEMORY_TYPES
        ]
    }


# ====== Entries ======


@router.get("/entries")
async def list_entries(
    memory_type: Optional[str] = None,
    enabled_only: Optional[bool] = None,
    page: int = 1,
    page_size: int = 50,
):
    result = memory_service.list_entries(
        memory_type=memory_type,
        enabled_only=enabled_only,
        page=page,
        page_size=page_size,
    )
    formatted = [_format_entry(e) for e in result["entries"]]
    return {
        "entries": formatted,
        "total": result["total"],
        "page": page,
        "page_size": page_size,
    }


@router.get("/entries/{entry_id}")
async def get_entry(entry_id: str):
    entry = memory_service.get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Memory entry not found")
    return _format_entry(entry)


@router.put("/entries/{entry_id}/enable")
async def enable_entry(entry_id: str):
    success = memory_service.enable_entry(entry_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory entry not found")
    return {"status": "enabled", "id": entry_id}


@router.put("/entries/{entry_id}/disable")
async def disable_entry(entry_id: str):
    success = memory_service.disable_entry(entry_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory entry not found")
    return {"status": "disabled", "id": entry_id}


@router.delete("/entries/{entry_id}")
async def forget_entry(entry_id: str):
    success = memory_service.forget_entry(entry_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory entry not found")
    return {"status": "forgotten", "id": entry_id}


@router.post("/extract")
async def extract_memories(request: dict):
    session_id = request.get("session_id", "manual_extract")
    user_message = request.get("user_message", "")
    assistant_response = request.get("assistant_response", "")
    if not user_message or not assistant_response:
        raise HTTPException(status_code=400, detail="user_message and assistant_response required")
    entries = memory_service.extract_memories(session_id, user_message, assistant_response)
    memory_service.add_entries(entries)
    return {"extracted": len(entries), "entries": [_format_entry(e) for e in entries]}


@router.delete("/clear")
async def clear_all_entries():
    count = memory_service.clear_all()
    return {"status": "cleared", "count": count}


# ====== Retrieval (for agent integration) ======


@router.post("/retrieve")
async def retrieve_memories(request: dict):
    query = request.get("query", "")
    top_k = request.get("top_k", 5)
    memory_type = request.get("memory_type")
    entries = memory_service.retrieve(query, top_k=top_k, memory_type=memory_type)
    return {"entries": [_format_entry(e) for e in entries], "query": query}


@router.get("/retrieve-context")
async def retrieve_context(query: str, top_k: int = 5):
    context = memory_service.retrieve_context(query, top_k=top_k)
    return {"context": context, "query": query}