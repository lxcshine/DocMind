"""
DocMind Memory Service
Memory extraction, storage, and retrieval with LLM-driven knowledge extraction.
Inspired by RAGFlow's Memory module: Raw / Semantic / Episodic / Procedural memory types.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

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
# Memory Index -- lightweight local embedding for retrieval
# ============================================================================

class MemoryIndex:
    def __init__(self):
        self._entries: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def add(self, entry_id: str, keywords: List[str]):
        vec = self._keywords_to_vec(keywords)
        with self._lock:
            self._entries[entry_id] = vec

    def remove(self, entry_id: str):
        with self._lock:
            self._entries.pop(entry_id, None)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_vec = self._keywords_to_vec(self._tokenize(query))
        with self._lock:
            if not self._entries:
                return []
            scores = []
            for eid, vec in self._entries.items():
                sim = self._cosine_sim(query_vec, vec)
                scores.append((eid, sim))
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [w.lower().strip(",.!?;:()[]{}") for w in text.split() if len(w) > 2]

    @staticmethod
    def _keywords_to_vec(keywords: List[str]) -> np.ndarray:
        if not keywords:
            return np.zeros(128, dtype=np.float32)
        np.random.seed(abs(hash(" ".join(sorted(keywords)))) % (2 ** 31))
        vec = np.random.randn(128).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        return vec

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ============================================================================
# Memory Service
# ============================================================================

class MemoryService:
    def __init__(
        self,
        db_path: str = "./memory_db.json",
        api_key: str = "",
        base_url: str = "",
        model: str = "",
    ):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._config: MemoryConfig = MemoryConfig()
        self._entries: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._index = MemoryIndex()
        self._llm_client: Optional[OpenAI] = None
        if api_key:
            self._llm_client = OpenAI(api_key=api_key, base_url=base_url)
            self._model = model
        self._load()

    # ---- Persistence -------------------------------------------------------

    def _load(self):
        if not self.db_path.exists():
            return
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = data.get("config", {})
            self._config = MemoryConfig(
                active_types=cfg.get("active_types", DEFAULT_MEMORY_TYPES),
                max_entries=cfg.get("max_entries", DEFAULT_MAX_ENTRIES),
                max_tokens=cfg.get("max_tokens", DEFAULT_MAX_TOKENS),
                temperature=cfg.get("temperature", DEFAULT_TEMPERATURE),
                forgetting_policy=cfg.get("forgetting_policy", "FIFO"),
                system_prompt=cfg.get("system_prompt", ""),
            )
            for e in data.get("entries", []):
                entry = MemoryEntry(**e)
                self._entries[entry.id] = entry
                if entry.enabled:
                    self._index.add(entry.id, entry.keywords)
            logger.info(f"Loaded {len(self._entries)} memory entries")
        except Exception as exc:
            logger.error(f"Failed to load memory: {exc}")

    def _save(self):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                data = {
                    "config": asdict(self._config),
                    "entries": [asdict(e) for e in self._entries.values()],
                }
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"Failed to save memory: {exc}")

    # ---- Config ------------------------------------------------------------

    def get_config(self) -> dict:
        with self._lock:
            return asdict(self._config)

    def update_config(self, **kwargs) -> dict:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._config, k) and v is not None:
                    setattr(self._config, k, v)
            self._save()
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
            response = self._llm_client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._config.temperature,
                max_tokens=2048,
                response_format={"type": "json_object"},
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

    # ---- CRUD for Entries -------------------------------------------------

    def add_entry(self, entry: MemoryEntry) -> None:
        with self._lock:
            self._entries[entry.id] = entry
            self._entries.move_to_end(entry.id)
            if entry.enabled:
                self._index.add(entry.id, entry.keywords)
            self._apply_forgetting_policy()
            self._save()

    def add_entries(self, entries: List[MemoryEntry]) -> None:
        with self._lock:
            for entry in entries:
                self._entries[entry.id] = entry
                self._entries.move_to_end(entry.id)
                if entry.enabled:
                    self._index.add(entry.id, entry.keywords)
            self._apply_forgetting_policy()
            self._save()

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
            self._index.add(entry.id, entry.keywords)
            self._save()
            return True

    def disable_entry(self, entry_id: str) -> bool:
        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return False
            entry.enabled = False
            entry.updated_at = datetime.now(timezone.utc).isoformat()
            self._index.remove(entry.id)
            self._save()
            return True

    def forget_entry(self, entry_id: str) -> bool:
        with self._lock:
            if entry_id not in self._entries:
                return False
            del self._entries[entry_id]
            self._index.remove(entry_id)
            self._save()
            return True

    def clear_all(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._index = MemoryIndex()
            self._save()
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
            del self._entries[oldest]


# ============================================================================
# Global singleton
# ============================================================================

_memory_service: Optional[MemoryService] = None


def init_memory_service(
    db_path: str = "./memory_db.json",
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
