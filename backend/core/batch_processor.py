# -*- coding: utf-8 -*-
"""
Batch Document Processor

Industrial-grade batch processing with:
- asyncio.Semaphore for controlled concurrency (prevents OOM / API rate limits)
- Per-document progress tracking
- Batch-level aggregation
- Graceful error handling (one failure doesn't kill the batch)
- Retry logic for transient failures
"""

import os
import uuid
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from config.settings import settings
from core.progress import progress_tracker

logger = logging.getLogger(__name__)


class BatchStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # some succeeded, some failed


class DocBatchStatus(str, Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # unsupported file type


@dataclass
class BatchDocItem:
    """Single document within a batch."""
    doc_id: str
    filename: str
    file_path: str
    file_size: int
    status: DocBatchStatus = DocBatchStatus.PENDING
    progress: int = 0
    current_step: str = ""
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "status": self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
        }


@dataclass
class BatchJob:
    """A batch processing job containing multiple documents."""
    batch_id: str
    directory: str
    status: BatchStatus = BatchStatus.PENDING
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    documents: List[BatchDocItem] = field(default_factory=list)
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    _semaphore: Optional[asyncio.Semaphore] = field(default=None, repr=False)
    _task: Optional[asyncio.Task] = field(default=None, repr=False)
    _processing_mode: str = field(default="fast", repr=False)

    def to_dict(self) -> Dict:
        return {
            "batch_id": self.batch_id,
            "directory": self.directory,
            "status": self.status,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "progress_percent": int((self.completed + self.failed + self.skipped) / self.total * 100) if self.total > 0 else 0,
            "documents": [d.to_dict() for d in self.documents],
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


# ===== Configuration =====

# Max concurrent document processing tasks
MAX_CONCURRENT_PROCESSING = int(os.getenv("BATCH_MAX_CONCURRENT", "3"))

# Max retries per document
MAX_RETRIES = int(os.getenv("BATCH_MAX_RETRIES", "2"))

# Supported file extensions for batch import
BATCH_SUPPORTED_EXTENSIONS = {
    ".pdf", ".md", ".txt", ".doc", ".docx",
    ".ppt", ".pptx", ".xls", ".xlsx",
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
}


class BatchProcessor:
    """
    Manages batch document processing with controlled concurrency.

    Architecture:
    - User submits a directory path
    - System scans for supported files
    - Creates a BatchJob with individual BatchDocItems
    - Processes documents using asyncio.Semaphore for concurrency control
    - Each document goes through: save -> process (RAG pipeline)
    - Progress is tracked per-document and aggregated at batch level
    """

    def __init__(self, max_concurrent: int = MAX_CONCURRENT_PROCESSING):
        self._batches: Dict[str, BatchJob] = {}
        self._max_concurrent = max_concurrent
        self._upload_handler = None
        self._meta_store = None
        logger.info(f"BatchProcessor initialized (max_concurrent={max_concurrent})")

    def set_handlers(self, upload_handler, meta_store):
        """Inject document upload handler and meta store."""
        self._upload_handler = upload_handler
        self._meta_store = meta_store

    def _ensure_handlers(self):
        """Ensure handlers are available."""
        if self._upload_handler is None:
            from api.documents import upload_handler, meta_store
            self._upload_handler = upload_handler
            self._meta_store = meta_store

    def scan_directory(self, directory: str) -> List[Dict]:
        """
        Scan a directory for supported document files.

        Returns list of file info dicts.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        files = []
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in BATCH_SUPPORTED_EXTENSIONS:
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "file_size": stat.st_size,
                    "extension": file_path.suffix.lower(),
                })

        return files

    def create_batch(self, directory: str) -> BatchJob:
        """
        Create a new batch job from a directory.

        Scans the directory, creates BatchDocItems, but does NOT start processing.
        """
        self._ensure_handlers()

        files = self.scan_directory(directory)
        if not files:
            raise ValueError(f"No supported documents found in: {directory}")

        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()

        documents = []
        for f in files:
            doc_id = str(uuid.uuid4())
            documents.append(BatchDocItem(
                doc_id=doc_id,
                filename=f["filename"],
                file_path=f["file_path"],
                file_size=f["file_size"],
                status=DocBatchStatus.PENDING,
            ))

        batch = BatchJob(
            batch_id=batch_id,
            directory=directory,
            total=len(documents),
            documents=documents,
            created_at=now,
        )

        self._batches[batch_id] = batch
        logger.info(
            f"Batch created: {batch_id} with {len(documents)} documents from {directory}"
        )

        return batch

    def start_batch(self, batch_id: str) -> BatchJob:
        """
        Start processing a batch job in the background.

        Returns the batch job immediately; processing runs as an asyncio Task.
        """
        batch = self._batches.get(batch_id)
        if not batch:
            raise ValueError(f"Batch not found: {batch_id}")

        if batch.status == BatchStatus.RUNNING:
            raise ValueError(f"Batch is already running: {batch_id}")

        batch.status = BatchStatus.RUNNING
        batch.started_at = datetime.now().isoformat()

        # Create semaphore for concurrency control
        batch._semaphore = asyncio.Semaphore(self._max_concurrent)

        # Launch as background task
        batch._task = asyncio.create_task(self._run_batch(batch))

        logger.info(f"Batch started: {batch_id} (max_concurrent={self._max_concurrent})")
        return batch

    async def _run_batch(self, batch: BatchJob):
        """
        Execute all documents in a batch with controlled concurrency.

        Uses asyncio.Semaphore to limit parallel processing.
        One document failure does not affect others.
        """
        logger.info(f"[Batch {batch.batch_id}] Starting processing of {batch.total} documents")

        # Ensure RAG is initialized before processing
        self._ensure_handlers()
        try:
            from core.raganything import get_rag_instance
            rag = get_rag_instance()
            if rag:
                self._upload_handler.set_rag(rag)
        except Exception as e:
            logger.warning(f"[Batch {batch.batch_id}] RAG injection warning: {e}")

        # Create tasks for all documents
        tasks = []
        for doc_item in batch.documents:
            task = asyncio.create_task(
                self._process_single_doc(batch, doc_item)
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate final status
        batch.completed = sum(1 for d in batch.documents if d.status == DocBatchStatus.COMPLETED)
        batch.failed = sum(1 for d in batch.documents if d.status == DocBatchStatus.FAILED)
        batch.skipped = sum(1 for d in batch.documents if d.status == DocBatchStatus.SKIPPED)
        batch.finished_at = datetime.now().isoformat()

        if batch.completed == batch.total:
            batch.status = BatchStatus.COMPLETED
        elif batch.completed > 0:
            batch.status = BatchStatus.PARTIAL
        else:
            batch.status = BatchStatus.FAILED

        logger.info(
            f"[Batch {batch.batch_id}] Finished: "
            f"{batch.completed} completed, {batch.failed} failed, {batch.skipped} skipped"
        )

    async def _process_single_doc(self, batch: BatchJob, doc_item: BatchDocItem):
        """
        Process a single document within a batch.

        Uses the batch's semaphore to control concurrency.
        Implements retry logic for transient failures.
        """
        async with batch._semaphore:
            doc_item.status = DocBatchStatus.UPLOADING
            doc_item.started_at = datetime.now().isoformat()
            doc_item.current_step = "Reading file..."

            # Read file content
            try:
                file_path = Path(doc_item.file_path)
                if not file_path.exists():
                    doc_item.status = DocBatchStatus.FAILED
                    doc_item.error = "File not found"
                    return

                content = file_path.read_bytes()
                doc_item.file_size = len(content)

            except Exception as e:
                doc_item.status = DocBatchStatus.FAILED
                doc_item.error = f"Failed to read file: {str(e)}"
                logger.error(f"[Batch {batch.batch_id}] Read failed for {doc_item.filename}: {e}")
                return

            # Save file via upload handler
            doc_item.current_step = "Saving file..."
            saved_path = None
            try:
                result = self._upload_handler.save_file(
                    file_content=content,
                    filename=doc_item.filename,
                    doc_id=doc_item.doc_id,
                )
                saved_path = result.get("file_path", "")

                # Register in meta store
                if self._meta_store:
                    self._meta_store.add_document(
                        doc_id=doc_item.doc_id,
                        filename=doc_item.filename,
                        file_type=Path(doc_item.filename).suffix.lstrip('.').upper(),
                        file_size=doc_item.file_size,
                        sections_count=0,
                        status="uploaded",
                    )

                doc_item.status = DocBatchStatus.QUEUED
                doc_item.progress = 10
                doc_item.current_step = "File saved, queuing for processing..."

            except Exception as e:
                doc_item.status = DocBatchStatus.FAILED
                doc_item.error = f"Save failed: {str(e)}"
                logger.error(f"[Batch {batch.batch_id}] Save failed for {doc_item.filename}: {e}")
                return

            # Process document (with retry)
            # In batch mode, we call the async processing directly instead of
            # going through upload_handler.process_document() which spawns a thread.
            # This avoids thread-within-asyncio issues.
            doc_item.status = DocBatchStatus.PROCESSING
            doc_item.current_step = "Processing via RAG pipeline..."

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    # Update meta store status
                    if self._meta_store:
                        self._meta_store.update_document(doc_item.doc_id, status="processing")

                    # Direct async processing (no thread spawning)
                    completed = await self._process_doc_direct(
                        doc_id=doc_item.doc_id,
                        filename=doc_item.filename,
                        file_path=saved_path,
                        processing_mode=batch._processing_mode,
                        batch=batch,
                        doc_item=doc_item,
                    )

                    if completed:
                        doc_item.status = DocBatchStatus.COMPLETED
                        doc_item.progress = 100
                        doc_item.current_step = "Completed"
                        doc_item.completed_at = datetime.now().isoformat()

                        if self._meta_store:
                            self._meta_store.update_document(doc_item.doc_id, status="completed")

                        logger.info(f"[Batch {batch.batch_id}] Completed: {doc_item.filename}")
                        return
                    else:
                        raise Exception("Processing failed")

                except Exception as e:
                    doc_item.retry_count = attempt
                    logger.warning(
                        f"[Batch {batch.batch_id}] Attempt {attempt}/{MAX_RETRIES} "
                        f"failed for {doc_item.filename}: {e}"
                    )
                    if attempt < MAX_RETRIES:
                        doc_item.current_step = f"Retrying (attempt {attempt + 1})..."
                        await asyncio.sleep(5 * attempt)  # Exponential backoff
                    else:
                        doc_item.status = DocBatchStatus.FAILED
                        doc_item.error = str(e)
                        doc_item.completed_at = datetime.now().isoformat()

                        if self._meta_store:
                            self._meta_store.update_document(doc_item.doc_id, status="failed")

                        logger.error(
                            f"[Batch {batch.batch_id}] FAILED after {MAX_RETRIES} retries: "
                            f"{doc_item.filename}: {e}"
                        )

    async def _process_doc_direct(
        self,
        doc_id: str,
        filename: str,
        file_path: str,
        processing_mode: str,
        batch: BatchJob,
        doc_item: BatchDocItem,
    ) -> bool:
        """
        Process a document directly in the current async context.
        Avoids the thread-spawning overhead of upload_handler.process_document().
        """
        from core.raganything import process_document
        from config.settings import settings

        try:
            # Initialize progress tracking
            progress_tracker.update(doc_id, progress=0, current_step="Starting document processing...")

            # Get or create RAG instance
            rag = self._upload_handler.rag
            if rag is None:
                doc_item.error = "RAG not initialized"
                return False

            # Ensure RAG is fully initialized
            progress_tracker.update(doc_id, progress=5, current_step="Ensuring RAG pipeline is ready...")
            await rag._ensure_lightrag_initialized()

            # Process document
            progress_tracker.update(doc_id, progress=10, current_step=f"Processing document [{processing_mode} mode]...")
            result = await process_document(
                rag=rag,
                file_path=file_path,
                output_dir=settings.RAG_OUTPUT_DIR,
                processing_mode=processing_mode,
            )

            # Update progress from result
            if result.get("success"):
                progress_tracker.update(doc_id, progress=100, current_step="Completed")
                return True
            else:
                error = result.get("error", "Unknown error")
                progress_tracker.update(doc_id, progress=0, current_step=f"Failed: {error}")
                doc_item.error = error
                return False

        except Exception as e:
            logger.error(f"[Batch] Direct processing failed for {filename}: {e}")
            progress_tracker.update(doc_id, progress=0, current_step=f"Failed: {str(e)}")
            doc_item.error = str(e)
            return False

    def get_batch(self, batch_id: str) -> Optional[BatchJob]:
        """Get a batch job by ID."""
        return self._batches.get(batch_id)

    def list_batches(self) -> List[Dict]:
        """List all batch jobs."""
        return [b.to_dict() for b in self._batches.values()]

    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a running batch job."""
        batch = self._batches.get(batch_id)
        if not batch or batch.status != BatchStatus.RUNNING:
            return False

        if batch._task and not batch._task.done():
            batch._task.cancel()

        # Mark remaining docs as skipped
        for doc in batch.documents:
            if doc.status in (DocBatchStatus.PENDING, DocBatchStatus.QUEUED):
                doc.status = DocBatchStatus.SKIPPED

        batch.status = BatchStatus.PARTIAL
        batch.finished_at = datetime.now().isoformat()
        logger.info(f"[Batch {batch_id}] Cancelled")
        return True


# ===== Global Instance =====
batch_processor = BatchProcessor()
