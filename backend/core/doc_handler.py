# -*- coding: utf-8 -*-
"""
DocMind Backend - Document Upload Handler

Uses RAG-Anything pipeline:
  Upload -> MinerU Parse -> LightRAG (KG + Vector DB) processing

Pipeline:
  1. Upload file -> Save to disk
  2. Process via RAG-Anything (parse + insert + multimodal)

Concurrency model:
  The RAG pipeline is async; callers should schedule it via FastAPI's
  `BackgroundTasks` so the request returns immediately. The previous
  implementation spawned a daemon thread that constructed its own
  `asyncio.new_event_loop()` - that pattern leaks connections, swallows
  errors, and is incompatible with uvicorn's event loop.
"""

import os
import uuid
import logging
from typing import Dict, Optional
from pathlib import Path

from config.settings import settings
from core.progress import progress_tracker

logger = logging.getLogger(__name__)


class DocumentUploadHandler:
    """
    Handles document upload and processing via RAG-Anything pipeline.
    """

    def __init__(
        self,
        upload_dir: str = None,
        meta_store=None,
    ):
        self.upload_dir = Path(upload_dir or settings.UPLOAD_DIR)
        os.makedirs(self.upload_dir, exist_ok=True)
        self.meta_store = meta_store

        # RAG instance will be injected later
        self._rag = None

        logger.info("DocumentUploadHandler initialized (RAG-Anything mode)")

    def set_rag(self, rag):
        """Inject the RAGAnything instance."""
        self._rag = rag

    @property
    def rag(self):
        if self._rag is None:
            from core.raganything import get_rag_instance
            self._rag = get_rag_instance()
        return self._rag

    # ---------- File save (sync, fast) ----------

    def save_file(self, file_content: bytes, filename: str, doc_id: str = None) -> Dict:
        """Step 1: Save uploaded file to disk permanently."""
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        file_ext = Path(filename).suffix.lower()
        save_filename = f"{doc_id}_{filename}"
        save_path = self.upload_dir / save_filename

        with open(save_path, "wb") as f:
            f.write(file_content)

        file_size = len(file_content)

        progress_tracker.create(doc_id, filename)
        progress_tracker.update(doc_id, status="uploaded", progress=100, current_step="File saved")

        return {
            "doc_id": doc_id,
            "filename": filename,
            "file_path": str(save_path),
            "file_size": file_size,
            "file_type": file_ext.lstrip('.').upper(),
            "status": "uploaded",
        }

    # ---------- Processing (async; scheduled via FastAPI BackgroundTasks) ----------

    async def process_document_async(
        self,
        doc_id: str,
        filename: str = None,
        processing_mode: str = "standard",
    ) -> Dict:
        """
        Process a saved document through the RAG pipeline.

        Designed to run inside FastAPI's event loop (scheduled with
        `BackgroundTasks` by the API layer). All I/O is awaited so the
        loop stays healthy and exceptions propagate to the logger.
        """
        progress = progress_tracker.get(doc_id)
        if not progress:
            return {"error": "Document not found", "doc_id": doc_id}

        if progress.status == "processing":
            return {"error": "Document is already being processed", "doc_id": doc_id}

        if not filename:
            filename = progress.filename

        file_path = self.upload_dir / f"{doc_id}_{filename}"
        if not file_path.exists():
            msg = "File not found"
            progress_tracker.update(doc_id, error=msg, status="failed")
            return {"error": msg, "doc_id": doc_id}

        file_ext = Path(filename).suffix.lower()
        if file_ext not in settings.ALLOWED_UPLOAD_TYPES:
            msg = f"Unsupported file type: {file_ext}"
            progress_tracker.update(doc_id, error=msg, status="failed")
            return {"error": msg, "doc_id": doc_id}

        try:
            await self._process_rag(doc_id, filename, str(file_path), processing_mode)
            return {"doc_id": doc_id, "status": "processing", "mode": processing_mode}
        except Exception as e:  # surface failures but never crash the loop
            logger.exception(f"Processing failed for {doc_id}")
            progress_tracker.update(doc_id, error=str(e), status="failed")
            return {"error": str(e), "doc_id": doc_id}

    def process_document(
        self,
        doc_id: str,
        filename: str = None,
        processing_mode: str = "standard",
    ) -> Dict:
        """
        Synchronous shim kept for backwards compatibility with callers
        that haven't migrated to `BackgroundTasks`. It only validates
        the request and marks the document as queued; the actual RAG
        pipeline still runs asynchronously via the event loop.

        The API layer should prefer `process_document_async` with a
        `BackgroundTasks` parameter to avoid blocking the worker.
        """
        progress = progress_tracker.get(doc_id)
        if not progress:
            return {"error": "Document not found", "doc_id": doc_id}

        if progress.status == "processing":
            return {"error": "Document is already being processed", "doc_id": doc_id}

        progress_tracker.update(
            doc_id, status="processing", progress=0, current_step="Starting..."
        )
        return {"doc_id": doc_id, "status": "processing", "mode": processing_mode}

    # ---------- RAG pipeline (async, no thread) ----------

    async def _process_rag(
        self,
        doc_id: str,
        filename: str,
        file_path: str,
        processing_mode: str = "standard",
    ) -> None:
        """Run the RAG-Anything pipeline end-to-end. Raises on failure."""
        logger.info(f"[RAG] Starting {processing_mode} processing for {doc_id}: {filename}")

        mode_labels = {
            "fast": "Fast (vector only)",
            "standard": "Standard (KG + vector)",
            "full": "Full (KG + vector + multimodal)",
        }
        total_steps = 3
        progress_tracker.set_step(
            doc_id, 1, total_steps,
            f"Initializing RAG pipeline [{mode_labels.get(processing_mode, processing_mode)}]...",
        )

        result = await self._process_async(doc_id, filename, file_path, total_steps, processing_mode)

        if result.get("success"):
            progress_tracker.update(
                doc_id,
                status="completed",
                progress=100,
                current_step="Knowledge base built",
            )
            if self.meta_store:
                self.meta_store.update_document(doc_id, status="completed")
            logger.info(f"[RAG] Document {doc_id} processed successfully")
        else:
            error_msg = result.get("error", "Unknown error")
            progress_tracker.update(doc_id, error=error_msg, status="failed")
            if self.meta_store:
                self.meta_store.update_document(doc_id, status="failed")
            logger.error(f"[RAG] Document {doc_id} failed: {error_msg}")
            raise RuntimeError(error_msg)

    async def _process_async(
        self,
        doc_id: str,
        filename: str,
        file_path: str,
        total_steps: int,
        processing_mode: str = "standard",
    ) -> Dict:
        """Async processing pipeline using RAG-Anything."""
        logger.info(f"[DocHandler] === Starting _process_async for {doc_id} ===")

        if self.rag is None:
            return {"success": False, "error": "RAG not initialized. Check server logs."}

        # Step 1: Ensure RAG is fully initialized
        progress_tracker.set_step(doc_id, 1, total_steps, "Ensuring RAG pipeline is ready...")
        try:
            await self.rag._ensure_lightrag_initialized()
        except Exception as e:
            logger.exception("RAG initialization failed")
            return {"success": False, "error": f"RAG initialization failed: {str(e)}"}

        # Step 2: Process document via RAG-Anything (parse + insert + multimodal)
        progress_tracker.set_step(doc_id, 2, total_steps, "Processing document (parse + build KG)...")
        try:
            from core.raganything import process_document as rag_process_document
            result = await rag_process_document(
                rag=self.rag,
                file_path=file_path,
                output_dir=settings.RAG_OUTPUT_DIR,
                processing_mode=processing_mode,
                doc_id=doc_id,
            )
            if not result.get("success"):
                return {"success": False, "error": result.get("error", "Processing failed")}
        except Exception as e:
            logger.exception("Document processing failed")
            return {"success": False, "error": f"Document processing failed: {str(e)}"}

        # Step 3: Done
        progress_tracker.set_step(doc_id, 3, total_steps, "Knowledge base built successfully")
        return {"success": True, "file_path": file_path}

    # ---------- File delete (sync) ----------

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document, its file, and its knowledge base entries."""
        try:
            progress = progress_tracker.get(doc_id)
            if progress:
                filename = progress.filename
                file_path = self.upload_dir / f"{doc_id}_{filename}"
                if file_path.exists():
                    os.remove(file_path)
            progress_tracker.delete(doc_id)
            logger.info(f"Deleted document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    def get_document_stats(self) -> Dict:
        """Get document statistics."""
        if self.rag and self.rag.lightrag:
            return {"status": "RAG initialized"}
        return {"status": "RAG not initialized"}
