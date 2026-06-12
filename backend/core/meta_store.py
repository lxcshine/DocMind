# -*- coding: utf-8 -*-
"""
Document Metadata Store

Persisted to SQLite (WAL mode) instead of a JSON file. The previous
implementation lost writes on crashes, raced under concurrent uploads,
and required a full file rewrite for every update. With SQLite:

  * atomic per-row updates
  * status / file_type / created_at indexed for fast listing
  * safe under multi-worker uvicorn (WAL) and thread safe (StateDB lock)

Public API is preserved so the rest of the code (documents API,
batch processor) is unchanged.
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from infrastructure.state_db import get_state_db

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """UTC timestamp with 'Z' suffix to avoid local-time ambiguity."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class DocumentMetadataStore:
    """SQLite-backed document metadata store."""

    UPSERT_SQL = """
    INSERT INTO documents (
        doc_id, filename, file_type, file_size, sections_count,
        status, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(doc_id) DO UPDATE SET
        filename        = excluded.filename,
        file_type       = excluded.file_type,
        file_size       = excluded.file_size,
        sections_count  = excluded.sections_count,
        status          = excluded.status,
        updated_at      = excluded.updated_at
    """

    def __init__(self, db_path: Optional[str] = None):
        # db_path is kept for backwards compatibility (callers pass one in
        # from config); when None we rely on the global StateDB singleton.
        self._db_path = db_path
        self._lock = threading.Lock()

    def _row_to_dict(self, row) -> Dict:
        return dict(row) if row else None

    def add_document(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        file_size: int,
        sections_count: int = 0,
        status: str = "uploaded",
    ) -> Dict:
        now = _now_iso()
        with self._lock:
            get_state_db().execute(
                self.UPSERT_SQL,
                (
                    doc_id, filename, file_type, file_size,
                    sections_count, status, now, now,
                ),
            )
        return {
            "doc_id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "file_size": file_size,
            "sections_count": sections_count,
            "status": status,
            "created_at": now,
            "updated_at": now,
        }

    def update_document(self, doc_id: str, **kwargs) -> Optional[Dict]:
        if not kwargs:
            return self.get_document(doc_id)

        with self._lock:
            current = self.get_document(doc_id)
            if not current:
                return None
            current.update(kwargs)
            current["updated_at"] = _now_iso()
            get_state_db().execute(
                self.UPSERT_SQL,
                (
                    current["doc_id"],
                    current["filename"],
                    current["file_type"],
                    current["file_size"],
                    current["sections_count"],
                    current["status"],
                    current["created_at"],
                    current["updated_at"],
                ),
            )
            return current

    def get_document(self, doc_id: str) -> Optional[Dict]:
        row = get_state_db().query_one(
            "SELECT * FROM documents WHERE doc_id = ?",
            (doc_id,),
        )
        return self._row_to_dict(row)

    def list_documents(self) -> List[Dict]:
        rows = get_state_db().query_all(
            "SELECT * FROM documents ORDER BY created_at DESC"
        )
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> bool:
        db = get_state_db()
        if db.query_one("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)) is None:
            return False
        db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        return True

    def get_total_sections(self) -> int:
        row = get_state_db().query_one(
            "SELECT COALESCE(SUM(sections_count), 0) AS total FROM documents"
        )
        return int(row["total"]) if row else 0
