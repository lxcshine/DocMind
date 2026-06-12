# -*- coding: utf-8 -*-
"""
Document Management API

Handles document upload, processing, progress tracking, and deletion.
Uses standardized response format and proper error handling.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import uuid
import shutil
from pathlib import Path
import logging

from config.settings import settings
from core.doc_handler import DocumentUploadHandler
from core.meta_store import DocumentMetadataStore
from core.progress import progress_tracker
from core.batch_processor import batch_processor, BATCH_SUPPORTED_EXTENSIONS
from infrastructure.response import success_response, paginated_response
from infrastructure.validation import validate_upload_file, sanitize_filename

logger = logging.getLogger(__name__)

router = APIRouter()

meta_store = DocumentMetadataStore(
    db_path=str(settings.BASE_DIR / "documents_meta.json")
)

upload_handler = DocumentUploadHandler(
    upload_dir=str(settings.UPLOAD_DIR),
    meta_store=meta_store,
)


def _inject_rag():
    """Inject RAG instance into upload_handler."""
    try:
        from core.raganything import get_rag_instance
        rag = get_rag_instance()
        if rag:
            upload_handler.set_rag(rag)
            logger.info("RAG instance injected into DocumentUploadHandler")
            return True
    except Exception as e:
        logger.warning(f"Failed to inject RAG: {e}")
    return False


async def _schedule_process_document(
    background_tasks: BackgroundTasks,
    doc_id: str,
    filename: str,
    mode: str,
) -> None:
    """
    Background task that runs the RAG pipeline inside the event loop.

    Replacing the previous `threading.Thread + asyncio.new_event_loop`
    pattern: this coroutine reuses the worker's loop, so DB / HTTP
    pools stay healthy and exceptions are logged with the right
    correlation id.
    """
    _inject_rag()
    try:
        await upload_handler.process_document_async(
            doc_id=doc_id, filename=filename, processing_mode=mode,
        )
    except Exception as e:  # never let BackgroundTasks swallow silently
        logger.exception(f"Background processing failed for {doc_id}")
        try:
            meta_store.update_document(doc_id, status="failed")
        except Exception:
            pass


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Step 1: Upload and save file permanently.
    
    This only saves the file to disk. User must click "Add to KB" to process it.
    """
    # Validate file
    await validate_upload_file(file)
    
    filename = sanitize_filename(file.filename or "unknown")
    file_ext = Path(filename).suffix.lower()
    
    doc_id = str(uuid.uuid4())
    
    try:
        content = await file.read()
        file_size = len(content)
        logger.info(f"[Upload] Read {file_size} bytes for {filename}")
        
        if file_size == 0:
            # File was consumed by validate_upload_file and seek(0) may have failed.
            # Try reading from the underlying file object.
            try:
                await file.seek(0)
                content = await file.read()
                file_size = len(content)
                logger.info(f"[Upload] Retry read: {file_size} bytes for {filename}")
            except Exception as e:
                logger.warning(f"[Upload] Retry read failed: {e}")
        
        result = upload_handler.save_file(
            file_content=content,
            filename=filename,
            doc_id=doc_id,
        )
        
        meta_store.add_document(
            doc_id=doc_id,
            filename=filename,
            file_type=file_ext.lstrip('.').upper(),
            file_size=file_size,
            sections_count=0,
            status="uploaded",
        )
        
        logger.info(f"Document uploaded: {filename} ({doc_id[:8]}...)")
        
        return success_response(
            data={
                "doc_id": doc_id,
                "filename": filename,
                "file_size": file_size,
                "file_type": result["file_type"],
                "status": "uploaded",
            },
            message=f"File '{filename}' saved successfully. Click 'Add to KB' to process it.",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/{doc_id}/process")
async def process_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    mode: str = "standard",
):
    """
    Step 2: Add document to knowledge base.

    Args:
        doc_id: Document ID
        mode: Processing mode:
            - "fast": Vector-only, no KG extraction (~1-2 min)
            - "standard": KG + vector (~3-8 min)
            - "full": KG + vector + multimodal (~10-30 min)

    The RAG pipeline runs as a FastAPI background task on the worker's
    event loop (no more daemon thread + new event loop), so the request
    returns immediately and the caller polls `/progress` for status.
    """
    if mode not in ("fast", "standard", "full"):
        raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}'. Use: fast, standard, full")

    doc = meta_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.get("status") == "completed":
        return success_response(
            data={"status": "already_processed", "sections_count": doc.get("sections_count", 0)},
            message="Document already processed",
        )

    if doc.get("status") == "processing":
        return success_response(
            data={"status": "already_processing"},
            message="Document is already being processed",
        )

    # Mark as processing and schedule the actual work on the event loop.
    meta_store.update_document(doc_id, status="processing")
    progress_tracker.update(
        doc_id, status="processing", progress=0, current_step="Queued for processing",
    )
    background_tasks.add_task(
        _schedule_process_document,
        background_tasks, doc_id, doc["filename"], mode,
    )

    logger.info(
        f"Processing scheduled for document: {doc['filename']} "
        f"({doc_id[:8]}...) mode={mode}"
    )

    return success_response(
        data={"doc_id": doc_id, "status": "processing", "mode": mode},
        message=(
            f"Document processing started in {mode} mode. "
            f"Check progress endpoint for status."
        ),
    )


@router.get("/{doc_id}/progress")
async def get_progress(doc_id: str):
    """Get processing progress for a document."""
    progress = progress_tracker.get(doc_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return success_response(data=progress.to_dict())


@router.get("/progress")
async def get_all_progress():
    """Get processing progress for all documents."""
    return success_response(data=progress_tracker.list_all())


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document, its file, and its chunks."""
    success = upload_handler.delete_document(doc_id)
    meta_store.delete_document(doc_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    logger.info(f"Document deleted: {doc_id[:8]}...")
    return success_response(data={"doc_id": doc_id}, message="Document deleted successfully")


@router.get("/stats")
async def get_stats():
    """Get document statistics."""
    docs = meta_store.list_documents()
    completed = sum(1 for d in docs if d.get("status") == "completed")
    processing = sum(1 for d in docs if d.get("status") == "processing")
    failed = sum(1 for d in docs if d.get("status") == "failed")
    
    return success_response(
        data={
            "total_documents": len(docs),
            "completed": completed,
            "processing": processing,
            "failed": failed,
            "total_sections": meta_store.get_total_sections(),
        }
    )


@router.get("/list")
async def list_documents():
    """List all documents in the knowledge base."""
    documents = meta_store.list_documents()
    
    formatted = []
    for doc in documents:
        progress = progress_tracker.get(doc["doc_id"])
        
        formatted.append({
            "key": doc["doc_id"],
            "name": doc["filename"],
            "type": doc["file_type"],
            "size": _format_size(doc["file_size"]),
            "status": doc["status"],
            "progress": progress.progress if progress else 0,
            "current_step": progress.current_step if progress else "",
            "completed_steps": progress.completed_steps if progress else 0,
            "total_steps": progress.total_steps if progress else 0,
            "sections": doc.get("sections_count", 0),
            "updatedAt": doc["updated_at"][:19].replace("T", " "),
        })
    
    return success_response(data={"documents": formatted, "total": len(formatted)})


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# ===== Batch Import Endpoints =====

@router.get("/batches")
async def batch_list_jobs():
    """List all batch jobs."""
    return success_response(data={"batches": batch_processor.list_batches()})


@router.post("/batch/scan")
async def batch_scan_directory(directory: str):
    """
    Scan a directory for supported documents.

    Returns the list of files found without starting processing.
    """
    directory = directory.strip().strip('"').strip("'")

    try:
        files = batch_processor.scan_directory(directory)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not files:
        return success_response(
            data={"files": [], "total": 0, "supported_extensions": sorted(BATCH_SUPPORTED_EXTENSIONS)},
            message="No supported documents found in the directory",
        )

    return success_response(
        data={
            "directory": directory,
            "files": files,
            "total": len(files),
            "supported_extensions": sorted(BATCH_SUPPORTED_EXTENSIONS),
        },
        message=f"Found {len(files)} supported documents",
    )


@router.post("/batch/start")
async def batch_start_processing(directory: str, mode: str = "fast"):
    """
    Create and start a batch processing job.

    Args:
        directory: Directory path on the server
        mode: Processing mode - "fast" (default), "standard", or "full"
    """
    if mode not in ("fast", "standard", "full"):
        raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}'. Use: fast, standard, full")

    directory = directory.strip().strip('"').strip("'")

    try:
        # Inject handlers
        batch_processor.set_handlers(upload_handler, meta_store)

        # Create batch (scans directory)
        batch = batch_processor.create_batch(directory)

        # Set processing mode on batch
        batch._processing_mode = mode

        # Start processing in background
        batch = batch_processor.start_batch(batch.batch_id)

        return success_response(
            data=batch.to_dict(),
            message=f"Batch processing started: {batch.total} documents in {mode} mode",
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch start failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@router.get("/batch/{batch_id}")
async def batch_get_status(batch_id: str):
    """Get batch job status and progress."""
    batch = batch_processor.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch job not found")

    return success_response(data=batch.to_dict())


@router.post("/batch/{batch_id}/cancel")
async def batch_cancel(batch_id: str):
    """Cancel a running batch job."""
    # Validate batch_id format to prevent route collision with /batch/scan, /batch/start
    if batch_id in ("scan", "start"):
        raise HTTPException(status_code=400, detail="Invalid batch ID")

    success = batch_processor.cancel_batch(batch_id)
    if not success:
        raise HTTPException(status_code=400, detail="Batch not found or not running")

    return success_response(data={"batch_id": batch_id}, message="Batch cancelled")
