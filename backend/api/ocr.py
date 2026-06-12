"""
OCR API - Intelligent Document OCR with LLM-enhanced accuracy

Features:
- Multi-format support: PDF, images (PNG/JPG/TIFF/BMP), PPT/PPTX
- Real-time SSE streaming progress (upload -> convert -> OCR -> LLM)
- Tesseract OCR engine with Chinese + English language support
- LLM intelligent post-processing: error correction, table formatting, structure preservation
"""

# -*- coding: utf-8 -*-

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging
import json
import uuid
import os
import asyncio
from pathlib import Path

from config.settings import settings
from core.ocr_handler import ocr_processor
from core.progress import progress_tracker
from infrastructure.state_db import get_state_db

logger = logging.getLogger(__name__)

router = APIRouter()


# ===== OCR Result Persistence (SQLite) =====

def _save_ocr_result(doc_id: str, result: Dict) -> None:
    """Persist OCR result to SQLite so it survives server restarts."""
    from datetime import datetime
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    get_state_db().execute(
        """
        INSERT INTO ocr_results (doc_id, result_json, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            result_json = excluded.result_json,
            updated_at  = excluded.updated_at
        """,
        (doc_id, json.dumps(result, ensure_ascii=False), now, now),
    )


def _load_ocr_result(doc_id: str) -> Optional[Dict]:
    """Load OCR result from SQLite. Returns None if not found."""
    row = get_state_db().query_one(
        "SELECT result_json FROM ocr_results WHERE doc_id = ?",
        (doc_id,),
    )
    if row:
        return json.loads(row["result_json"])
    return None


class OCRProgressResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    progress: int
    current_step: str
    total_pages: Optional[int] = None
    error: Optional[str] = None


@router.post("/upload")
async def ocr_upload_document(file: UploadFile = File(...)):
    """
    Step 1: Upload document for OCR processing.

    Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP, PPT, PPTX
    Returns doc_id for subsequent processing steps.
    """
    filename = file.filename or "unknown"
    file_ext = Path(filename).suffix.lower()

    if file_ext not in settings.OCR_ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {settings.OCR_ALLOWED_TYPES}"
        )

    content = await file.read()
    file_size = len(content)
    max_size = settings.OCR_MAX_FILE_SIZE_MB * 1024 * 1024

    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_size / 1024 / 1024:.1f}MB. Max: {settings.OCR_MAX_FILE_SIZE_MB}MB"
        )

    doc_id = str(uuid.uuid4())
    save_filename = f"{doc_id}_{filename}"
    save_path = settings.UPLOAD_DIR / "ocr" / save_filename

    os.makedirs(save_path.parent, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(content)

    progress = progress_tracker.create(doc_id, filename)
    progress.update(status="uploaded", progress=100, current_step="File uploaded")

    return JSONResponse(content={
        "doc_id": doc_id,
        "filename": filename,
        "file_size": file_size,
        "file_type": file_ext.lstrip('.').upper(),
        "status": "uploaded",
        "message": f"File uploaded. Ready for OCR processing.",
    })


@router.get("/progress/{doc_id}")
async def ocr_progress(doc_id: str):
    """
    Get OCR processing progress for a document.

    Returns progress percentage (0-100), current step description,
    and processing status.
    """
    progress = progress_tracker.get(doc_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Document not found")

    return JSONResponse(content=progress.to_dict())


async def _process_ocr_async(doc_id: str, file_path: str, filename: str, enable_llm: bool = False):
    """Run OCR processing directly on the event loop.

    The previous implementation used `run_in_executor + asyncio.run()` which
    created a nested event loop — that pattern leaks connections and can
    deadlock. Since `ocr_processor.process_document` is already async, we
    just await it directly.
    """
    return await ocr_processor.process_document(
        doc_id, file_path, filename, enable_llm=enable_llm
    )


async def _run_ocr_background(doc_id: str, file_path: str, filename: str):
    """Background task: run OCR (without LLM correction by default) and store result."""
    try:
        result = await _process_ocr_async(doc_id, file_path, filename, enable_llm=False)
        _save_ocr_result(doc_id, result)
        logger.info(f"[OCR] Background processing completed for {doc_id} (LLM skipped)")
    except Exception as e:
        logger.error(f"Background OCR failed for {doc_id}: {e}", exc_info=True)
        _save_ocr_result(doc_id, {"error": str(e), "doc_id": doc_id, "status": "failed"})


async def _run_llm_correction_background(doc_id: str, raw_text: str, filename: str):
    """Background task: run LLM correction on existing OCR result."""
    try:
        result = _load_ocr_result(doc_id) or {}
        corrected = await ocr_processor.intelligent_correct(raw_text, filename=filename)

        result["ocr_text"] = corrected
        result["corrected_char_count"] = len(corrected)
        result["llm_corrected"] = True
        _save_ocr_result(doc_id, result)

        progress_tracker.update(
            doc_id,
            status="completed",
            progress=100,
            current_step="LLM correction completed",
        )
        logger.info(f"[OCR] LLM correction completed for {doc_id}")
    except Exception as e:
        logger.error(f"LLM correction failed for {doc_id}: {e}", exc_info=True)
        progress_tracker.update(
            doc_id,
            status="failed",
            error=f"LLM correction failed: {str(e)}",
        )


@router.post("/process-poll")
async def ocr_process_poll(file: UploadFile = File(...)):
    """
    Simple OCR: upload file, get doc_id, poll /progress/{doc_id} for status,
    then get result from /result/{doc_id}.
    No SSE streaming - just reliable HTTP polling.
    """
    filename = file.filename or "unknown"
    file_ext = Path(filename).suffix.lower()

    if file_ext not in settings.OCR_ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}"
        )

    content = await file.read()
    file_size = len(content)
    max_size = settings.OCR_MAX_FILE_SIZE_MB * 1024 * 1024

    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_size / 1024 / 1024:.1f}MB"
        )

    doc_id = str(uuid.uuid4())
    save_filename = f"{doc_id}_{filename}"
    save_path = settings.UPLOAD_DIR / "ocr" / save_filename

    os.makedirs(save_path.parent, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(content)

    progress_tracker.create(doc_id, filename)
    progress_tracker.update(doc_id, status="uploaded", progress=0, current_step="Starting OCR...")

    logger.info(f"[OCR] Starting background processing for {filename} ({file_size} bytes, doc_id={doc_id})")

    asyncio.create_task(_run_ocr_background(doc_id, str(save_path), filename))

    return JSONResponse(content={
        "doc_id": doc_id,
        "filename": filename,
        "file_size": file_size,
        "status": "processing",
    })


@router.get("/result/{doc_id}")
async def ocr_get_result(doc_id: str):
    """Get the OCR result for a completed document."""
    result = _load_ocr_result(doc_id)
    progress = progress_tracker.get(doc_id)

    if result:
        return JSONResponse(content=result)

    if not progress:
        raise HTTPException(status_code=404, detail="Document not found")

    if progress.status == "failed":
        return JSONResponse(content={
            "doc_id": doc_id,
            "status": "failed",
            "error": progress.error,
        })

    return JSONResponse(content={
        "doc_id": doc_id,
        "status": progress.status,
        "progress": progress.progress,
        "current_step": progress.current_step,
    })


@router.post("/correct/{doc_id}")
async def ocr_correct_text(doc_id: str):
    """
    Run LLM intelligent correction on an existing OCR result.

    The original raw OCR text is preserved in raw_text.
    The corrected result is stored in ocr_text.
    Poll /progress/{doc_id} for status, then /result/{doc_id} for the updated result.
    """
    result = _load_ocr_result(doc_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found. Run OCR first.")

    if result.get("status") == "failed":
        raise HTTPException(status_code=400, detail="OCR processing failed, cannot correct.")

    raw_text = result.get("raw_text", "")
    if not raw_text or len(raw_text.strip()) < 10:
        raise HTTPException(status_code=400, detail="No text content to correct.")

    filename = result.get("filename", "unknown")

    progress_tracker.update(
        doc_id,
        status="processing",
        progress=75,
        current_step="LLM correction in progress...",
    )

    asyncio.create_task(_run_llm_correction_background(doc_id, raw_text, filename))

    return JSONResponse(content={
        "doc_id": doc_id,
        "status": "processing",
        "message": "LLM correction started. Poll /progress/ for status.",
    })


@router.get("/test-stream")
async def ocr_test_stream():
    """Simple SSE stream test endpoint to verify streaming works."""

    async def test_generator():
        for i in range(5):
            yield json.dumps({
                "type": "test",
                "data": {
                    "seq": i + 1,
                    "message": f"Test event {i + 1}/5",
                }
            }, ensure_ascii=False) + "\n"
            await asyncio.sleep(1)
        yield json.dumps({
            "type": "test",
            "data": {"seq": 6, "message": "Test complete"}
        }, ensure_ascii=False) + "\n"

    return StreamingResponse(
        test_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/process-stream")
async def ocr_process_stream(
        file: UploadFile = File(...),
):
    """
    Intelligent OCR processing with real-time SSE progress streaming.
    Uses queue-based streaming for reliable delivery on all platforms.
    """
    filename = file.filename or "unknown"
    file_ext = Path(filename).suffix.lower()

    if file_ext not in settings.OCR_ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}"
        )

    content = await file.read()
    file_size = len(content)
    max_size = settings.OCR_MAX_FILE_SIZE_MB * 1024 * 1024

    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_size / 1024 / 1024:.1f}MB"
        )

    doc_id = str(uuid.uuid4())
    save_filename = f"{doc_id}_{filename}"
    save_path = settings.UPLOAD_DIR / "ocr" / save_filename

    os.makedirs(save_path.parent, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(content)

    progress_tracker.create(doc_id, filename)
    progress_tracker.update(doc_id, status="uploaded", progress=0, current_step="File uploaded")

    logger.info(f"[OCR] Starting process-stream for {filename} ({file_size} bytes, doc_id={doc_id})")

    event_queue = asyncio.Queue()

    async def process_ocr():
        """Background task: runs OCR pipeline and pushes events to queue."""
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            try:
                event_queue.put_nowait(json.dumps({
                    "type": "upload_progress",
                    "data": {
                        "doc_id": doc_id,
                        "filename": filename,
                        "file_size": file_size,
                        "status": "uploaded",
                        "progress": 100,
                    }
                }, ensure_ascii=False) + "\n")
                logger.info(f"[OCR] upload_progress queued for {doc_id}")

                progress_tracker.update(
                    doc_id,
                    status="processing",
                    progress=0,
                    current_step="Starting OCR pipeline..."
                )

                event_queue.put_nowait(json.dumps({
                    "type": "status",
                    "data": "Starting intelligent OCR pipeline..."
                }, ensure_ascii=False) + "\n")
                logger.info(f"[OCR] status queued for {doc_id}")

                logger.info(f"[OCR] Starting document conversion for {doc_id}")
                doc_meta = await loop.run_in_executor(
                    executor,
                    ocr_processor.convert_to_images,
                    str(save_path),
                    doc_id,
                )
                total_pages = len(doc_meta)
                logger.info(f"[OCR] Document converted: {total_pages} pages for {doc_id}")

                event_queue.put_nowait(json.dumps({
                    "type": "convert_progress",
                    "data": {
                        "stage": "convert",
                        "total_pages": total_pages,
                        "progress_pct": 10,
                        "message": f"Document converted: {total_pages} pages detected"
                    }
                }, ensure_ascii=False) + "\n")

                progress_tracker.update(doc_id, progress=10, current_step=f"Converted: {total_pages} pages")

                processed_images = []
                for i, img in enumerate(doc_meta):
                    processed = await loop.run_in_executor(
                        executor,
                        ocr_processor.preprocess_image,
                        img,
                    )
                    processed_images.append(processed)

                    pct = 10 + int((i + 1) / total_pages * 15)
                    event_queue.put_nowait(json.dumps({
                        "type": "preprocess_progress",
                        "data": {
                            "stage": "preprocess",
                            "page": i + 1,
                            "total_pages": total_pages,
                            "progress_pct": pct,
                            "message": f"Preprocessing page {i + 1}/{total_pages}"
                        }
                    }, ensure_ascii=False) + "\n")

                    progress_tracker.update(
                        doc_id,
                        progress=pct,
                        current_step=f"Preprocessing page {i + 1}/{total_pages}"
                    )

                ocr_results = []
                logger.info(f"[OCR] Starting OCR extraction for {total_pages} pages, doc_id={doc_id}")
                for i, img in enumerate(processed_images):
                    text = await loop.run_in_executor(
                        executor,
                        ocr_processor.extract_text,
                        img,
                    )
                    text = text or ""
                    ocr_results.append({
                        "page": i + 1,
                        "text": text,
                        "char_count": len(text),
                    })

                    pct = 25 + int((i + 1) / total_pages * 40)
                    event_queue.put_nowait(json.dumps({
                        "type": "ocr_progress",
                        "data": {
                            "stage": "ocr",
                            "page": i + 1,
                            "total_pages": total_pages,
                            "progress_pct": pct,
                            "char_count": len(text),
                            "message": f"OCR extracting page {i + 1}/{total_pages}"
                        }
                    }, ensure_ascii=False) + "\n")

                    progress_tracker.update(
                        doc_id,
                        progress=pct,
                        current_step=f"OCR page {i + 1}/{total_pages}"
                    )

                combined_text = ""
                for result in ocr_results:
                    if result["text"]:
                        combined_text += f"\n--- Page {result['page']} ---\n{result['text']}\n"

                event_queue.put_nowait(json.dumps({
                    "type": "llm_progress",
                    "data": {
                        "stage": "llm_correction",
                        "progress_pct": 65,
                        "raw_chars": sum(r["char_count"] for r in ocr_results),
                        "message": "LLM intelligent correction - analyzing OCR text..."
                    }
                }, ensure_ascii=False) + "\n")

                progress_tracker.update(doc_id, progress=65, current_step="LLM intelligent correction...")

                event_queue.put_nowait(json.dumps({
                    "type": "llm_progress",
                    "data": {
                        "stage": "llm_correction",
                        "progress_pct": 75,
                        "message": "LLM: fixing OCR errors, detecting tables, preserving structure..."
                    }
                }, ensure_ascii=False) + "\n")

                progress_tracker.update(doc_id, progress=75, current_step="LLM formatting text...")

                logger.info(f"[OCR] Starting LLM intelligent correction for {doc_id}, text length={len(combined_text)}")
                corrected_text = await ocr_processor.intelligent_correct(
                    combined_text,
                    filename=filename,
                )

                event_queue.put_nowait(json.dumps({
                    "type": "llm_progress",
                    "data": {
                        "stage": "llm_correction",
                        "progress_pct": 95,
                        "corrected_chars": len(corrected_text),
                        "message": "LLM correction complete"
                    }
                }, ensure_ascii=False) + "\n")

                progress_tracker.update(
                    doc_id,
                    status="completed",
                    progress=100,
                    current_step="OCR completed with LLM enhancement"
                )

                event_queue.put_nowait(json.dumps({
                    "type": "result",
                    "data": {
                        "doc_id": doc_id,
                        "filename": filename,
                        "total_pages": total_pages,
                        "ocr_text": corrected_text,
                        "raw_char_count": sum(r["char_count"] for r in ocr_results),
                        "corrected_char_count": len(corrected_text),
                        "page_results": [
                            {
                                "page": r["page"],
                                "char_count": r["char_count"],
                                "has_content": len(r["text"].strip()) > 0,
                            }
                            for r in ocr_results
                        ],
                        "status": "completed",
                    }
                }, ensure_ascii=False) + "\n")

                # Persist result to SQLite so /result/{doc_id} works after SSE ends
                _save_ocr_result(doc_id, {
                    "doc_id": doc_id,
                    "filename": filename,
                    "total_pages": total_pages,
                    "raw_text": combined_text,
                    "ocr_text": corrected_text,
                    "raw_char_count": sum(r["char_count"] for r in ocr_results),
                    "corrected_char_count": len(corrected_text),
                    "llm_corrected": True,
                    "page_results": [
                        {
                            "page": r["page"],
                            "char_count": r["char_count"],
                            "has_content": len(r["text"].strip()) > 0,
                        }
                        for r in ocr_results
                    ],
                    "status": "completed",
                })

                logger.info(f"[OCR] result queued for {doc_id}")

            except Exception as e:
                logger.error(f"OCR processing failed for {doc_id}: {e}", exc_info=True)
                progress_tracker.update(doc_id, error=str(e))
                event_queue.put_nowait(json.dumps({
                    "type": "error",
                    "data": f"OCR processing failed: {str(e)}"
                }, ensure_ascii=False) + "\n")

            finally:
                event_queue.put_nowait(None)
                logger.info(f"[OCR] Background task finished for {doc_id}")

    async def stream_events():
        """Read events from queue and yield them as SSE chunks."""
        logger.info(f"[OCR] stream_events started for {doc_id}")
        while True:
            event = await event_queue.get()
            if event is None:
                logger.info(f"[OCR] stream_events received end marker for {doc_id}")
                break
            yield event

    import concurrent.futures
    task = asyncio.create_task(process_ocr())

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/process/{doc_id}")
async def ocr_process_document(doc_id: str):
    """
    Process an already-uploaded document through the OCR pipeline.

    Call /upload first, then this endpoint with the returned doc_id.
    """
    progress = progress_tracker.get(doc_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Document not found")

    filename = progress.filename
    file_path = settings.UPLOAD_DIR / "ocr" / f"{doc_id}_{filename}"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    try:
        result = await ocr_processor.process_document(
            doc_id=str(doc_id),
            file_path=str(file_path),
            filename=filename,
        )
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"OCR processing error for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{doc_id}")
async def ocr_delete_document(doc_id: str):
    """
    Delete OCR result and uploaded file.
    """
    progress = progress_tracker.get(doc_id)
    if progress:
        filename = progress.filename
        file_path = settings.UPLOAD_DIR / "ocr" / f"{doc_id}_{filename}"
        if file_path.exists():
            os.remove(file_path)

    progress_tracker.delete(doc_id)
    # Delete OCR result from SQLite
    get_state_db().execute("DELETE FROM ocr_results WHERE doc_id = ?", (doc_id,))

    return JSONResponse(content={
        "doc_id": doc_id,
        "status": "deleted",
        "message": "OCR document and data deleted"
    })
