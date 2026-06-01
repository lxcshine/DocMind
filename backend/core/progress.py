"""
ResearchFlow Backend - Processing Progress Tracker

Tracks document processing progress in real-time.
"""

import threading
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


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
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
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
        self.updated_at = datetime.now().isoformat()
    
    def set_step(self, step: int, total: int, description: str):
        """Update current step and calculate progress."""
        self.current_step = description
        self.total_steps = total
        self.completed_steps = step
        self.progress = int((step / total) * 100) if total > 0 else 0
        self.updated_at = datetime.now().isoformat()
    
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


class ProgressTracker:
    """Thread-safe progress tracker for all documents."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._progresses: Dict[str, ProcessingProgress] = {}
    
    def create(self, doc_id: str, filename: str) -> ProcessingProgress:
        with self._lock:
            progress = ProcessingProgress(doc_id, filename)
            self._progresses[doc_id] = progress
            return progress
    
    def get(self, doc_id: str) -> Optional[ProcessingProgress]:
        with self._lock:
            return self._progresses.get(doc_id)
    
    def update(self, doc_id: str, **kwargs) -> Optional[ProcessingProgress]:
        with self._lock:
            progress = self._progresses.get(doc_id)
            if progress:
                progress.update(**kwargs)
                return progress
            return None
    
    def set_step(self, doc_id: str, step: int, total: int, description: str):
        with self._lock:
            progress = self._progresses.get(doc_id)
            if progress:
                progress.set_step(step, total, description)
    
    def list_all(self) -> Dict[str, Dict]:
        with self._lock:
            return {doc_id: p.to_dict() for doc_id, p in self._progresses.items()}
    
    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id in self._progresses:
                del self._progresses[doc_id]
                return True
            return False


progress_tracker = ProgressTracker()
