# -*- coding: utf-8 -*-
"""
DocMind Memory Service
Memory extraction, storage, and retrieval with LLM-driven knowledge extraction.
Inspired by RAGFlow's Memory module: Raw / Semantic / Episodic / Procedural memory types.

Changes from previous version:
  - MemoryIndex uses real embeddings via OpenAI instead of random vectors
  - Persistence uses SQLite (state_db) instead of JSON files
  - LLM calls go through the unified client with retry/fallback
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from infrastructure.llm_client import get_sync_llm
from infrastructure.tracing import get_tracer

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

MEMORY_TYPES = {
    "raw": 1,
    "semantic": 2,
    "episodic": 4,
    "procedural": 8,
}

MEMORY_TYPE_LABELS = {
    "raw": "Raw Conversation",
    "semantic": "Semantic Knowledge",
    "episodic": "Episodic Memory",
    "procedural": "Procedural Memory",
}

MEMORY_TYPE_COLORS = {
    "raw": "blue",
    "semantic": "green",
    "episodic": "orange",
    "procedural": "purple",
}

DEFAULT_MEMORY_TYPES = ["raw", "semantic", "episodic", "procedural"]
DEFAULT_MAX_ENTRIES = 200
DEFAULT_MAX_TOKENS = 50000
DEFAULT_TEMPERATURE = 0.3

# ---------------------------------------------------------------------------
# Prompt Templates (adapted from RAGFlow's PromptAssembler)
# ---------------------------------------------------------------------------

MEMORY_EXTRACTION_SYSTEM = """**Memory Extraction Specialist**
You are an expert at analyzing conversations to extract structured memory.

{type_instructions}

**OUTPUT REQUIREMENTS:**
1. Output MUST be valid JSON
2. Each extracted item MUST have: "type", "content", "keywords"
3. Only extract memory types specified above
4. Maximum {max_items} items per type
5. "keywords": a list of 1-5 key search terms for retrieving this memory later
6. If there is nothing meaningful to extract for a type, return an empty array for that type

**REQUIRED OUTPUT FORMAT (JSON):**
```json
{{{output_format}}}
```"""

TYPE_INSTRUCTIONS = {
    "semantic": """
**EXTRACT SEMANTIC KNOWLEDGE:**
- Universal facts, definitions, concepts the user mentions or asks about
- User's domain expertise, research interests, academic background
- Time-invariant, generally true information

**Extraction Rules:**
- content: A clear factual statement (one sentence)
- keywords: 1-5 key search terms""",

    "episodic": """
**EXTRACT EPISODIC KNOWLEDGE:**
- Specific questions the user asked, topics they explored
- Their learning progress, milestones, research directions
- Time-bound, person-specific, contextual experiences

**Extraction Rules:**
- content: A narrative event description (one sentence)
- keywords: 1-5 key search terms""",

    "procedural": """
**EXTRACT PROCEDURAL KNOWLEDGE:**
- User's preferred formats (LaTeX, markdown, code style)
- Recurring workflows, habits, interaction patterns
- Goal-oriented preferences and constraints

**Extraction Rules:**
- content: An actionable preference or habit statement (one sentence)
- keywords: 1-5 key search terms""",
}

MEMORY_EXTRACTION_USER = """Analyze the following conversation and extract structured memories.

Conversation:
{conversation}

Extract memories according to the types specified. Focus on what is meaningful and reusable for future conversations."""


# ============================================================================
# Data Model
# ============================================================================

@dataclass
class MemoryEntry:
    id: str
    type: str  # "raw", "semantic", "episodic", "procedural"
    content: str
    keywords: List[str] = field(default_factory=list)
    source_session_id: str = ""
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""


@dataclass
class MemoryConfig:
    active_types: List[str] = field(default_factory=lambda: DEFAULT_MEMORY_TYPES.copy())
    max_entries: int = DEFAULT_MAX_ENTRIES
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    forgetting_policy: str = "FIFO"
    system_prompt: str = ""


# ============================================================================
# Memory Index — real embedding-based retrieval
# ============================================================================

class MemoryIndex:
    """
    Embedding-based memory index using the project's configured embedding model.

    Falls back to BM25-style keyword matching if the embedding API is unavailable.
    """

    def __init__(self, embed_fn=None):
        """
        Args:
            embed_fn: callable(text: str) -> List[float]
                      If None, will lazily initialize from settings.
        """
        self._entries: Dict[str, np.ndarray] = {}
        self._texts: Dict[str, str] = {}  # entry_id → "content keywords" for BM25 fallback
        self._lock = threading.Lock()
        self._embed_fn = embed_fn
        self._embed_dim: int = 0

    def _get_embed_fn(self):
        """Lazily initialize the embedding function."""
        if self._embed_fn is not None:
            return self._embed_fn

        try:
            from config.settings import settings
            from openai import OpenAI

            client = OpenAI(
                api_key=settings.GEMINI_API_KEY,
                base_url=settings.GEMINI_BASE_URL,
                timeout=30.0,
            )
            embed_model = getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-small")

            def _embed(text: str) -> List[float]:
                resp = client.embeddings.create(
                    model=embed_model,
                    input=text,
                )
                return resp.data[0].embedding

            self._embed_fn = _embed
            return self._embed_fn
        except Exception as e:
            logger.warning(f"Embedding function init failed, using BM25 fallback: {e}")
            self._embed_fn = False
            return None

    def add(self, entry_id: str, content: str, keywords: List[str]):
        """Add an entry with its content and keywords."""
        text_for_search = f"{content} {' '.join(keywords)}"
        with self._lock:
            self._texts[entry_id] = text_for_search

        embed_fn = self._get_embed_fn()
        if embed_fn:
            try:
                vec = np.array(embed_fn(text_for_search), dtype=np.float32)
                vec /= np.linalg.norm(vec) + 1e-8
                self._embed_dim = len(vec)
                with self._lock:
                    self._entries[entry_id] = vec
                return
            except Exception as e:
                logger.warning(f"Embedding failed for {entry_id}, using keyword fallback: {e}")

        # BM25-style: store as None, search will use keyword matching
        with self._lock:
            self._entries[entry_id] = None

    def remove(self, entry_id: str):
        with self._lock:
            self._entries.pop(entry_id, None)
            self._texts.pop(entry_id, None)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        embed_fn = self._get_embed_fn()

        # Try embedding-based search
        if embed_fn:
            try:
                query_vec = np.array(embed_fn(query), dtype=np.float32)
                query_vec /= np.linalg.norm(query_vec) + 1e-8
                with self._lock:
                    if self._entries:
                        scores = []
                        for eid, vec in self._entries.items():
                            if vec is not None:
                                sim = float(np.dot(query_vec, vec))
                                scores.append((eid, sim))
                        if scores:
                            scores.sort(key=lambda x: x[1], reverse=True)
                            return scores[:top_k]
            except Exception as e:
                logger.warning(f"Embedding search failed, falling back to BM25: {e}")

        # BM25-style keyword fallback
        return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Simple BM25-like scoring using term overlap."""
        query_terms = set(self._tokenize(query))
        if not query_terms:
            return []
        with self._lock:
            scores = []
            for eid, text in self._texts.items():
                doc_terms = set(self._tokenize(text))
                overlap = len(query_terms & doc_terms)
                if overlap > 0:
                    # Simple TF-based score with length normalization
                    score = overlap / (len(doc_terms) + 1)
                    scores.append((eid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [w.lower().strip(",.!?;:()[]{}") for w in text.split() if len(w) > 2]


# ============================================================================
# Memory Service — SQLite-backed persistence
# ============================================================================

class MemoryService:
    def __init__(
        self,
        db_path: str = "",
        api_key: str = "",
        base_url: str = "",
        model: str = "",
    ):
        self._lock = threading.RLock()
        self._config: MemoryConfig = MemoryConfig()
        self._entries: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._index = MemoryIndex()
        self._llm_client = None
        self._model = model
        if api_key:
            self._llm_client = get_sync_llm()
        self._use_sqlite = False
        self._sqlite_db = None
        self._init_sqlite(db_path)
        self._load()

    # ---- SQLite Persistence ------------------------------------------------

    def _init_sqlite(self, db_path: str):
        """Initialize SQLite storage using the shared state_db infrastructure."""
        try:
            from infrastructure.state_db import get_state_db
            self._sqlite_db = get_state_db()
            self._ensure_memory_tables()
            self._use_sqlite = True
            logger.info("Memory service using SQLite persistence")
        except Exception as e:
            logger.warning(f"SQLite init failed for memory, using in-memory only: {e}")
            self._use_sqlite = False

    def _ensure_memory_tables(self):
        """Create memory_entries table if it doesn't exist."""
        if not self._sqlite_db:
            return
        self._sqlite_db.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                keywords TEXT NOT NULL DEFAULT '[]',
                source_session_id TEXT DEFAULT '',
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self._sqlite_db.execute("""
            CREATE TABLE IF NOT EXISTS memory_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

    def _load(self):
        if self._use_sqlite and self._sqlite_db:
            self._load_from_sqlite()
        # If no SQLite, entries stay empty (in-memory only)

    def _load_from_sqlite(self):
        try:
            # Load config
            rows = self._sqlite_db.query_all("SELECT key, value FROM memory_config")
            cfg = {}
            for row in rows:
                try:
                    cfg[row["key"]] = json.loads(row["value"])
                except (json.JSONDecodeError, TypeError):
                    cfg[row["key"]] = row["value"]
            if cfg:
                self._config = MemoryConfig(
                    active_types=cfg.get("active_types", DEFAULT_MEMORY_TYPES),
                    max_entries=cfg.get("max_entries", DEFAULT_MAX_ENTRIES),
                    max_tokens=cfg.get("max_tokens", DEFAULT_MAX_TOKENS),
                    temperature=cfg.get("temperature", DEFAULT_TEMPERATURE),
                    forgetting_policy=cfg.get("forgetting_policy", "FIFO"),
                    system_prompt=cfg.get("system_prompt", ""),
                )

            # Load entries
            rows = self._sqlite_db.query_all(
                "SELECT * FROM memory_entries ORDER BY created_at ASC"
            )
            for row in rows:
                keywords = json.loads(row.get("keywords", "[]"))
                entry = MemoryEntry(
                    id=row["id"],
                    type=row["type"],
                    content=row["content"],
                    keywords=keywords,
                    source_session_id=row.get("source_session_id", ""),
                    enabled=bool(row.get("enabled", 1)),
                    created_at=row.get("created_at", ""),
                    updated_at=row.get("updated_at", ""),
                )
                self._entries[entry.id] = entry
                if entry.enabled:
                    self._index.add(entry.id, entry.content, entry.keywords)
            logger.info(f"Loaded {len(self._entries)} memory entries from SQLite")
        except Exception as exc:
            logger.error(f"Failed to load memory from SQLite: {exc}")

    def _save_entry_to_sqlite(self, entry: MemoryEntry):
        if not self._use_sqlite or not self._sqlite_db:
            return
        try:
            self._sqlite_db.execute("""
                INSERT INTO memory_entries (id, type, content, keywords, source_session_id, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    type=excluded.type, content=excluded.content, keywords=excluded.keywords,
                    source_session_id=excluded.source_session_id, enabled=excluded.enabled,
                    updated_at=excluded.updated_at
            """, (
                entry.id, entry.type, entry.content,
                json.dumps(entry.keywords, ensure_ascii=False),
                entry.source_session_id,
                1 if entry.enabled else 0,
                entry.created_at, entry.updated_at,
            ))
        except Exception as exc:
            logger.error(f"Failed to save entry to SQLite: {exc}")

    def _delete_entry_from_sqlite(self, entry_id: str):
        if not self._use_sqlite or not self._sqlite_db:
            return
        try:
            self._sqlite_db.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
        except Exception as exc:
            logger.error(f"Failed to delete entry from SQLite: {exc}")

    def _save_config_to_sqlite(self):
        if not self._use_sqlite or not self._sqlite_db:
            return
        try:
            cfg = asdict(self._config)
            for k, v in cfg.items():
                self._sqlite_db.execute("""
                    INSERT INTO memory_config (key, value) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """, (k, json.dumps(v, ensure_ascii=False)))
        except Exception as exc:
            logger.error(f"Failed to save config to SQLite: {exc}")

    def _clear_sqlite(self):
        if not self._use_sqlite or not self._sqlite_db:
            return
        try:
            self._sqlite_db.execute("DELETE FROM memory_entries")
        except Exception as exc:
            logger.error(f"Failed to clear SQLite memory: {exc}")

    # ---- Config ------------------------------------------------------------

    def get_config(self) -> dict:
        with self._lock:
            return asdict(self._config)

    def update_config(self, **kwargs) -> dict:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._config, k) and v is not None:
                    setattr(self._config, k, v)
            self._save_config_to_sqlite()
            return asdict(self._config)

    # ---- LLM Knowledge Extraction -----------------------------------------

    def _build_extraction_prompt(self, types: List[str]) -> str:
        type_instructions = "\n".join(
            TYPE_INSTRUCTIONS[t] for t in types if t in TYPE_INSTRUCTIONS
        )
        if not type_instructions:
            return ""
        output_parts = []
        for t in types:
            if t in TYPE_INSTRUCTIONS:
                output_parts.append(
                    f'"{t}": [{{"content": "...", "keywords": ["kw1", "kw2"]}}]'
                )
        output_format = ",\n".join(output_parts)
        return MEMORY_EXTRACTION_SYSTEM.format(
            type_instructions=type_instructions,
            max_items=5,
            output_format=output_format,
        )

    def extract_memories(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
    ) -> List[MemoryEntry]:
        extract_types = [t for t in self._config.active_types if t != "raw"]
        if not extract_types or not self._llm_client:
            return []

        system_prompt = self._build_extraction_prompt(extract_types)
        if not system_prompt:
            return []

        conversation = f"User: {user_message}\nAssistant: {assistant_response}"
        user_prompt = MEMORY_EXTRACTION_USER.format(conversation=conversation)

        try:
            tracer = get_tracer()
            with tracer.span("memory.extract", session_id=session_id) as span:
                response = self._llm_client.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self._config.temperature,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                    span=span,
                )
            raw = response.choices[0].message.content
            extracted = json.loads(raw)

            entries: List[MemoryEntry] = []
            now = datetime.now(timezone.utc).isoformat()
            for mtype in extract_types:
                items = extracted.get(mtype, [])
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content", "").strip()
                    if not content:
                        continue
                    keywords = item.get("keywords", [])
                    if isinstance(keywords, str):
                        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
                    eid = f"{mtype}_{session_id}_{len(entries)}"
                    entry = MemoryEntry(
                        id=eid,
                        type=mtype,
                        content=content,
                        keywords=keywords,
                        source_session_id=session_id,
                        enabled=True,
                        created_at=now,
                        updated_at=now,
                    )
                    entries.append(entry)

            return entries
        except Exception as exc:
            logger.error(f"Memory extraction failed: {exc}")
            return []

    def extract_from_turn(self, session_id: str, user_message: str, assistant_response: str):
        """Convenience: extract and store memories from a conversation turn."""
        entries = self.extract_memories(session_id, user_message, assistant_response)
        if entries:
            self.add_entries(entries)

    # ---- CRUD for Entries -------------------------------------------------

    def add_entry(self, entry: MemoryEntry) -> None:
        with self._lock:
            self._entries[entry.id] = entry
            self._entries.move_to_end(entry.id)
            if entry.enabled:
                self._index.add(entry.id, entry.content, entry.keywords)
            self._save_entry_to_sqlite(entry)
            self._apply_forgetting_policy()

    def add_entries(self, entries: List[MemoryEntry]) -> None:
        with self._lock:
            for entry in entries:
                self._entries[entry.id] = entry
                self._entries.move_to_end(entry.id)
                if entry.enabled:
                    self._index.add(entry.id, entry.content, entry.keywords)
                self._save_entry_to_sqlite(entry)
            self._apply_forgetting_policy()

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        with self._lock:
            return self._entries.get(entry_id)

    def list_entries(
        self,
        memory_type: Optional[str] = None,
        enabled_only: bool = False,
        page: int = 1,
        page_size: int = 50,
    ) -> dict:
        with self._lock:
            entries = list(self._entries.values())
            if memory_type and memory_type != "all":
                entries = [e for e in entries if e.type == memory_type]
            if enabled_only:
                entries = [e for e in entries if e.enabled]
            total = len(entries)
            start = (page - 1) * page_size
            end = start + page_size
            return {"entries": entries[start:end], "total": total}

    def enable_entry(self, entry_id: str) -> bool:
        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return False
            entry.enabled = True
            entry.updated_at = datetime.now(timezone.utc).isoformat()
            self._index.add(entry.id, entry.content, entry.keywords)
            self._save_entry_to_sqlite(entry)
            return True

    def disable_entry(self, entry_id: str) -> bool:
        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return False
            entry.enabled = False
            entry.updated_at = datetime.now(timezone.utc).isoformat()
            self._index.remove(entry.id)
            self._save_entry_to_sqlite(entry)
            return True

    def forget_entry(self, entry_id: str) -> bool:
        with self._lock:
            if entry_id not in self._entries:
                return False
            del self._entries[entry_id]
            self._index.remove(entry_id)
            self._delete_entry_from_sqlite(entry_id)
            return True

    def clear_all(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._index = MemoryIndex(self._index._embed_fn)
            self._clear_sqlite()
            return count

    # ---- Retrieval ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[MemoryEntry]:
        with self._lock:
            scored = self._index.search(query, top_k=top_k * 2)
            results = []
            for eid, score in scored:
                entry = self._entries.get(eid)
                if entry and entry.enabled:
                    if memory_type and memory_type != "all" and entry.type != memory_type:
                        continue
                    results.append((entry, score))
            results.sort(key=lambda x: x[1], reverse=True)
            return [e for e, _ in results[:top_k]]

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        entries = self.retrieve(query, top_k=top_k)
        if not entries:
            return ""
        lines = []
        for e in entries:
            label = MEMORY_TYPE_LABELS.get(e.type, e.type)
            lines.append(f"[{label}] {e.content}")
        return "\n".join(lines)

    # ---- Forgetting Policy ------------------------------------------------

    def _apply_forgetting_policy(self):
        if self._config.forgetting_policy != "FIFO":
            return
        excess = len(self._entries) - self._config.max_entries
        for _ in range(excess):
            oldest = next(iter(self._entries))
            self._index.remove(oldest)
            self._delete_entry_from_sqlite(oldest)
            del self._entries[oldest]


# ============================================================================
# Global singleton
# ============================================================================

_memory_service: Optional[MemoryService] = None


def init_memory_service(
    db_path: str = "",
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> MemoryService:
    global _memory_service
    _memory_service = MemoryService(
        db_path=db_path,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    return _memory_service


def get_memory_service() -> Optional[MemoryService]:
    return _memory_service
