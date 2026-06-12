# -*- coding: utf-8 -*-
"""
Document Processing Progress Tracker

Persisted to SQLite (WAL mode) instead of an in-process dict so progress
survives restarts and is shared across uvicorn workers.

The public surface is intentionally kept identical to the previous
in-memory implementation so callers (documents API, batch processor, ...)
do not need to change.
"""

import logging
import threading
from datetime import datetime
from typing import Dict, Optional

from infrastructure.state_db import get_state_db

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """UTC timestamp with explicit 'Z' suffix to remove local-time ambiguity."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class ProcessingProgress:
    """Track processing progress for a document."""

    def __init__(self, doc_id: str, filename: str):
        self.doc_id = doc_id
        self.filename = filename
        self.status = "pending"  # pending, uploading, uploaded, processing, completed, failed
        self.progress = 0  # 0-100
        self.current_step = ""
        self.total_steps = 0
        self.completed_steps = 0
        self.sections_count = 0
        self.error = None
        self.created_at = _now_iso()
        self.updated_at = _now_iso()

    def update(self, status: str = None, progress: int = None,
               current_step: str = None, sections_count: int = None,
               error: str = None):
        if status:
            self.status = status
        if progress is not None:
            self.progress = max(0, min(100, progress))
        if current_step:
            self.current_step = current_step
        if sections_count is not None:
            self.sections_count = sections_count
        if error:
            self.error = error
            self.status = "failed"
        self.updated_at = _now_iso()

    def set_step(self, step: int, total: int, description: str):
        """Update current step and calculate progress."""
        self.current_step = description
        self.total_steps = total
        self.completed_steps = step
        self.progress = int((step / total) * 100) if total > 0 else 0
        self.updated_at = _now_iso()

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "status": self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "sections_count": self.sections_count,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def _row_values(self):
        return (
            self.doc_id, self.filename, self.status, self.progress,
            self.current_step, self.total_steps, self.completed_steps,
            self.sections_count, self.error, self.created_at, self.updated_at,
        )


class ProgressTracker:
    """SQLite-backed progress tracker; safe across processes & restarts."""

    UPSERT_SQL = """
    INSERT INTO progress (
        doc_id, filename, status, progress, current_step,
        total_steps, completed_steps, sections_count, error,
        created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(doc_id) DO UPDATE SET
        filename        = excluded.filename,
        status          = excluded.status,
        progress        = excluded.progress,
        current_step    = excluded.current_step,
        total_steps     = excluded.total_steps,
        completed_steps = excluded.completed_steps,
        sections_count  = excluded.sections_count,
        error           = excluded.error,
        updated_at      = excluded.updated_at
    """

    def __init__(self):
        # Local in-process lock to coalesce concurrent writes within the
        # same worker; StateDB takes a process-wide lock for the actual
        # commit.
        self._lock = threading.Lock()

    def _load(self, row: Dict) -> ProcessingProgress:
        p = ProcessingProgress(row["doc_id"], row["filename"])
        p.status = row["status"]
        p.progress = row["progress"]
        p.current_step = row["current_step"]
        p.total_steps = row["total_steps"]
        p.completed_steps = row["completed_steps"]
        p.sections_count = row["sections_count"]
        p.error = row["error"]
        p.created_at = row["created_at"]
        p.updated_at = row["updated_at"]
        return p

    def _save(self, progress: ProcessingProgress) -> None:
        get_state_db().execute(self.UPSERT_SQL, progress._row_values())

    def create(self, doc_id: str, filename: str) -> ProcessingProgress:
        progress = ProcessingProgress(doc_id, filename)
        with self._lock:
            self._save(progress)
        return progress

    def get(self, doc_id: str) -> Optional[ProcessingProgress]:
        row = get_state_db().query_one(
            "SELECT * FROM progress WHERE doc_id = ?",
            (doc_id,),
        )
        return self._load(dict(row)) if row else None

    def update(self, doc_id: str, **kwargs) -> Optional[ProcessingProgress]:
        with self._lock:
            progress = self.get(doc_id)
            if not progress:
                return None
            progress.update(**kwargs)
            self._save(progress)
            return progress

    def set_step(self, doc_id: str, step: int, total: int, description: str):
        with self._lock:
            progress = self.get(doc_id)
            if progress:
                progress.set_step(step, total, description)
                self._save(progress)

    def list_all(self) -> Dict[str, Dict]:
        rows = get_state_db().query_all(
            "SELECT * FROM progress ORDER BY updated_at DESC"
        )
        return {row["doc_id"]: self._load(dict(row)).to_dict() for row in rows}

    def delete(self, doc_id: str) -> bool:
        db = get_state_db()
        if db.query_one("SELECT 1 FROM progress WHERE doc_id = ?", (doc_id,)) is None:
            return False
        db.execute("DELETE FROM progress WHERE doc_id = ?", (doc_id,))
        return True


progress_tracker = ProgressTracker()
