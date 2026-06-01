"""
ResearchFlow Backend - Document Metadata Store

Tracks uploaded documents and their processing status.
"""

import json
import logging
import threading
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentMetadataStore:
    """Thread-safe document metadata store backed by JSON file."""
    
    def __init__(self, db_path: str = "./documents_meta.json"):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._documents: Dict[str, Dict] = {}
        self._load()
    
    def _load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self._documents = json.load(f)
                logger.info(f"Loaded {len(self._documents)} document records")
            except Exception as e:
                logger.error(f"Failed to load document metadata: {e}")
                self._documents = {}
    
    def _save(self):
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self._documents, f, ensure_ascii=False, separators=(',', ':'))
        except Exception as e:
            logger.error(f"Failed to save document metadata: {e}")
    
    def add_document(self, doc_id: str, filename: str, file_type: str, file_size: int, sections_count: int = 0, status: str = "uploaded") -> Dict:
        with self._lock:
            now = datetime.now().isoformat()
            doc = {
                "doc_id": doc_id,
                "filename": filename,
                "file_type": file_type,
                "file_size": file_size,
                "sections_count": sections_count,
                "status": status,
                "created_at": now,
                "updated_at": now,
            }
            self._documents[doc_id] = doc
            self._save()
            return doc
    
    def update_document(self, doc_id: str, **kwargs) -> Optional[Dict]:
        with self._lock:
            if doc_id not in self._documents:
                return None
            self._documents[doc_id].update(kwargs)
            self._documents[doc_id]["updated_at"] = datetime.now().isoformat()
            self._save()
            return self._documents[doc_id]
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        with self._lock:
            return self._documents.get(doc_id)
    
    def list_documents(self) -> List[Dict]:
        with self._lock:
            return list(self._documents.values())
    
    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id in self._documents:
                del self._documents[doc_id]
                self._save()
                return True
            return False
    
    def get_total_sections(self) -> int:
        with self._lock:
            return sum(doc.get("sections_count", 0) for doc in self._documents.values())
