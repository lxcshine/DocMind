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

logger = logging.getLogger(__name__)

router = APIRouter()

meta_store = DocumentMetadataStore(
    db_path=str(settings.BASE_DIR / "documents_meta.json")
)

upload_handler = DocumentUploadHandler(
    upload_dir=str(settings.UPLOAD_DIR),
    model=settings.GEMINI_MODEL,
    api_key=settings.GEMINI_API_KEY,
    base_url=settings.GEMINI_BASE_URL,
    meta_store=meta_store,
)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Step 1: Upload and save file permanently.
    
    This only saves the file to disk. User must click "Add to KB" to process it.
    """
    filename = file.filename or "unknown"
    file_ext = Path(filename).suffix.lower()
    
    if file_ext not in settings.ALLOWED_UPLOAD_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {settings.ALLOWED_UPLOAD_TYPES}"
        )
    
    doc_id = str(uuid.uuid4())
    
    try:
        content = await file.read()
        file_size = len(content)
        
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
        
        return JSONResponse(content={
            "doc_id": doc_id,
            "filename": filename,
            "file_size": file_size,
            "file_type": result["file_type"],
            "status": "uploaded",
            "message": f"File '{filename}' saved successfully. Click 'Add to KB' to process it.",
        })
    
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{doc_id}/process")
async def process_document(doc_id: str):
    """
    Step 2: Add document to knowledge base.
    
    This triggers the PageIndex pipeline: Parse -> Chunk -> Embed -> Store
    """
    doc = meta_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if doc.get("status") == "completed":
        return {"status": "already_processed", "sections_count": doc.get("sections_count", 0)}
    
    if doc.get("status") == "processing":
        return {"status": "already_processing"}
    
    try:
        meta_store.update_document(doc_id, status="processing")
        
        result = upload_handler.process_document(
            doc_id=doc_id,
            filename=doc["filename"],
        )
        
        if "error" in result:
            meta_store.update_document(doc_id, status="failed")
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "doc_id": doc_id,
            "status": "processing",
            "message": "Document processing started. Check progress endpoint for status.",
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        meta_store.update_document(doc_id, status="failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{doc_id}/progress")
async def get_progress(doc_id: str):
    """Get processing progress for a document."""
    progress = progress_tracker.get(doc_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return progress.to_dict()


@router.get("/progress")
async def get_all_progress():
    """Get processing progress for all documents."""
    return progress_tracker.list_all()


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document, its file, and its chunks."""
    success = upload_handler.delete_document(doc_id)
    meta_store.delete_document(doc_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"status": "deleted", "doc_id": doc_id}


@router.get("/stats")
async def get_stats():
    """Get document statistics."""
    return {
        "total_documents": len(meta_store.list_documents()),
        "total_sections": meta_store.get_total_sections(),
    }


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
            "sections": doc.get("sections_count", 0),
            "updatedAt": doc["updated_at"][:19].replace("T", " "),
        })
    
    return {"documents": formatted, "total": len(formatted)}


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
