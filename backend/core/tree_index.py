"""
DocMind Backend - Tree Index Store

PageIndex-style hierarchical tree index for documents.
Replaces ChromaDB vector storage. No embeddings, no vectors.

Each document is stored as a JSON file containing:
  - doc_id, doc_name, doc_description, type, path
  - page_count / line_count
  - structure: hierarchical tree with title, start_index, end_index, summary, text, nodes
  - pages: per-page text content (for PDF)
"""

import os
import json
import logging
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

META_INDEX = "_meta.json"


class TreeIndexStore:
    """
    PageIndex-style hierarchical document index.
    
    Documents are stored as JSON files -- no vectors, no embeddings.
    """
    
    def __init__(self, index_dir: str = None):
        if index_dir is None:
            from config.settings import settings
            self.index_dir = Path(settings.TREE_INDEX_DIR)
        else:
            self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._meta: Dict[str, Dict] = {}
        self._load_meta()
        logger.info(f"TreeIndexStore initialized at {self.index_dir}")

    def _load_meta(self):
        meta_path = self.index_dir / META_INDEX
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._meta = json.load(f)
                logger.info(f"Loaded {len(self._meta)} document index entries")
            except Exception as e:
                logger.error(f"Failed to load meta index: {e}")
                self._meta = {}

    def _save_meta(self):
        meta_path = self.index_dir / META_INDEX
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self._meta, f, ensure_ascii=False, separators=(',', ':'))
        except Exception as e:
            logger.error(f"Failed to save meta index: {e}")

    def _save_meta_locked(self):
        """Internal version, caller must hold self._lock"""
        meta_path = self.index_dir / META_INDEX
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self._meta, f, ensure_ascii=False, separators=(',', ':'))
        except Exception as e:
            logger.error(f"Failed to save meta index: {e}")

    def _make_meta_entry(self, doc: Dict) -> Dict:
        entry = {
            "doc_id": doc.get("doc_id", ""),
            "doc_name": doc.get("doc_name", ""),
            "doc_description": doc.get("doc_description", ""),
            "type": doc.get("type", "pdf"),
            "path": doc.get("path", ""),
            "status": doc.get("status", "completed"),
            "page_count": doc.get("page_count", 0),
            "created_at": doc.get("created_at", datetime.now().isoformat()),
            "updated_at": doc.get("updated_at", datetime.now().isoformat()),
        }
        return entry

    def save_document(self, doc: Dict) -> str:
        doc_id = doc.get("doc_id", "")
        if not doc_id:
            raise ValueError("doc_id is required")

        now = datetime.now().isoformat()
        doc.setdefault("created_at", now)
        doc["updated_at"] = now
        doc["status"] = "completed"

        pages = doc.pop("pages", [])
        doc["page_count"] = len(pages)

        pages_dir = self.index_dir / f"{doc_id}_pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        for p in pages:
            page_num = p.get("page", 0)
            page_path = pages_dir / f"{page_num}.txt"
            with open(page_path, "w", encoding="utf-8") as pf:
                pf.write(p.get("content", ""))

        doc_path = self.index_dir / f"{doc_id}.json"
        with open(doc_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, separators=(',', ':'))

        with self._lock:
            self._meta[doc_id] = self._make_meta_entry(doc)
            self._save_meta_locked()

        logger.info(f"Saved document tree index: {doc_id} ({doc.get('doc_name', '')}), {len(pages)} pages in {pages_dir}")
        return doc_id

    def load_document(self, doc_id: str) -> Optional[Dict]:
        doc_path = self.index_dir / f"{doc_id}.json"
        if not doc_path.exists():
            return None
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load document {doc_id}: {e}")
            return None

    def get_document_meta(self, doc_id: str) -> Optional[Dict]:
        """Return lightweight document metadata (no structure/pages)."""
        return self._meta.get(doc_id)

    def get_document_info(self, doc_id: str) -> str:
        """Return document metadata as JSON string (for agent tool)."""
        meta = self._meta.get(doc_id)
        if not meta:
            return json.dumps({"error": f"Document {doc_id} not found"})
        return json.dumps({
            "doc_id": doc_id,
            "doc_name": meta.get("doc_name", ""),
            "doc_description": meta.get("doc_description", ""),
            "type": meta.get("type", ""),
            "page_count": meta.get("page_count", 0),
            "status": meta.get("status", ""),
        }, ensure_ascii=False)

    def get_document_structure(self, doc_id: str) -> str:
        """Return tree structure JSON without text (saves tokens)."""
        doc = self.load_document(doc_id)
        if not doc:
            return json.dumps({"error": f"Document {doc_id} not found"})

        structure = doc.get("structure", [])
        structure_no_text = self._remove_fields(structure, fields=["text"])
        return json.dumps(structure_no_text, ensure_ascii=False)

    def get_page_content(self, doc_id: str, pages: str) -> str:
        """
        Retrieve page content for a document.

        pages format: '5-7', '3,8', or '12'
        """
        pages_dir = self.index_dir / f"{doc_id}_pages"
        if not pages_dir.exists():
            return json.dumps({"error": f"Document {doc_id} has no page content"})

        try:
            page_nums = self._parse_pages(pages)
        except (ValueError, AttributeError) as e:
            return json.dumps({"error": f"Invalid pages format: {pages!r}. Error: {e}"})

        result = []
        for pn in page_nums:
            page_path = pages_dir / f"{pn}.txt"
            if page_path.exists():
                try:
                    with open(page_path, "r", encoding="utf-8") as pf:
                        result.append({"page": pn, "content": pf.read()})
                except Exception as e:
                    logger.error(f"Failed to read page {pn}: {e}")
            else:
                result.append({"page": pn, "content": f"[Page {pn} not found]"})

        return json.dumps(result, ensure_ascii=False)

    def get_page_images(self, doc_id: str, pages: str) -> List[str]:
        """
        Get image file paths for specific pages of a document.

        pages format: '5-7', '3,8', or '12'
        """
        images_dir = self.index_dir / f"{doc_id}_images"
        if not images_dir.exists():
            return []

        try:
            page_nums = self._parse_pages(pages)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid pages format in get_page_images: {pages!r}")
            return []

        image_paths = []
        for pn in page_nums:
            for ext in (".jpg", ".png", ".jpeg"):
                img_path = images_dir / f"{pn}{ext}"
                if img_path.exists():
                    image_paths.append(str(img_path))
                    break

        return image_paths

    def delete_document(self, doc_id: str) -> bool:
        import shutil
        doc_path = self.index_dir / f"{doc_id}.json"
        pages_dir = self.index_dir / f"{doc_id}_pages"
        images_dir = self.index_dir / f"{doc_id}_images"
        with self._lock:
            if doc_path.exists():
                os.remove(doc_path)
            if pages_dir.exists():
                shutil.rmtree(pages_dir)
            if images_dir.exists():
                shutil.rmtree(images_dir)
            if doc_id in self._meta:
                del self._meta[doc_id]
                self._save_meta_locked()
        logger.info(f"Deleted document tree index: {doc_id}")
        return True

    def list_documents(self) -> List[Dict]:
        return list(self._meta.values())

    def get_stats(self) -> Dict:
        return {
            "total_documents": len(self._meta),
            "index_dir": str(self.index_dir),
        }

    @staticmethod
    def _parse_pages(pages: str) -> List[int]:
        result = []
        for part in pages.split(","):
            part = part.strip()
            if "-" in part:
                start, end = int(part.split("-", 1)[0].strip()), int(part.split("-", 1)[1].strip())
                if start > end:
                    raise ValueError(f"Invalid range '{part}': start must be <= end")
                result.extend(range(start, end + 1))
            else:
                result.append(int(part))
        return sorted(set(result))

    @staticmethod
    def _remove_fields(data, fields):
        if isinstance(data, dict):
            return {
                k: TreeIndexStore._remove_fields(v, fields)
                for k, v in data.items()
                if k not in fields
            }
        elif isinstance(data, list):
            return [TreeIndexStore._remove_fields(item, fields) for item in data]
        return data


tree_index_store = TreeIndexStore()
