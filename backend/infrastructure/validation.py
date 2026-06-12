# -*- coding: utf-8 -*-
"""
Request Validation and Rate Limiting

Provides:
- File upload validation (size, type) - now streams to disk so we never
  load a 100MB file into memory just to size-check it.
- Sliding-window rate limit persisted to SQLite (works across processes).
- Input sanitization
"""

import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import Request, HTTPException, UploadFile

from config.settings import settings
from infrastructure.state_db import get_state_db

logger = logging.getLogger(__name__)


# ===== File Upload Validation =====

# Cap at settings.MAX_UPLOAD_SIZE_MB so the upload limit is centralized.
MAX_FILE_SIZE = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {
    # Documents
    ".pdf", ".doc", ".docx", ".txt", ".md",
    # Spreadsheets
    ".xls", ".xlsx", ".csv",
    # Presentations
    ".ppt", ".pptx",
    # Images (for OCR)
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
}


async def validate_upload_file(file: UploadFile) -> None:
    """
    Validate uploaded file size and type.

    Size is checked from the request's Content-Length header when available
    (fast, no I/O). Otherwise, the file is streamed with a hard byte cap
    and the pointer is reset via seek(0) for the route handler.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Size check: the multipart part has its own Content-Length header
    # which FastAPI exposes via file.headers. Use it to avoid consuming
    # the file stream unnecessarily.
    part_content_length = None
    if hasattr(file, 'headers') and file.headers:
        part_content_length = file.headers.get("content-length")
    if part_content_length:
        try:
            length = int(part_content_length)
            if length > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File too large (> "
                        f"{settings.MAX_UPLOAD_SIZE_MB}MB). "
                        f"Maximum: {settings.MAX_UPLOAD_SIZE_MB}MB"
                    ),
                )
            # Header check passed — don't consume the file stream.
            return
        except ValueError:
            pass  # Malformed header; fall through to streaming check.

    # Fallback: stream-read with a hard byte cap, then seek(0).
    total = 0
    chunk_size = 1024 * 1024  # 1MB
    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                try:
                    await file.close()
                except Exception:
                    pass
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File too large (> "
                        f"{settings.MAX_UPLOAD_SIZE_MB}MB). "
                        f"Maximum: {settings.MAX_UPLOAD_SIZE_MB}MB"
                    ),
                )
    finally:
        try:
            await file.seek(0)
        except Exception:
            pass


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and special characters."""
    # Remove path separators
    filename = filename.replace("/", "").replace("\\", "")
    # Remove null bytes
    filename = filename.replace("\x00", "")
    # Limit length
    if len(filename) > 255:
        name, ext = Path(filename).stem[:200], Path(filename).suffix
        filename = name + ext
    return filename


# ===== Rate Limiting =====

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    burst_size: int = 10
    # Drop entries older than this (seconds) on cleanup. Keep short so the
    # bucket table doesn't grow unbounded.
    bucket_window_seconds: int = 120


class RateLimiter:
    """
    Sliding-window rate limiter backed by SQLite.

    The previous in-memory defaultdict only worked inside a single Python
    process. With SQLite + WAL we get correct counters across:
      * multiple uvicorn workers (multi-process)
      * restarts (counters persist for the active window)
      * threads within a process (single connection is locked for writes)

    Trade-off vs Redis: SQLite is single-machine only. For multi-host
    deployments swap `get_state_db()` for a Redis-backed implementation
    without changing this class's surface.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()

    def is_allowed(self, client_ip: str) -> bool:
        """Return True if the request is within the configured rate budget."""
        db = get_state_db()
        now = time.time()
        window = float(self.config.bucket_window_seconds)
        per_minute_cutoff = now - 60.0

        # Opportunistic cleanup - cheap because of the (client_ip, ts) index.
        db.execute(
            "DELETE FROM rate_limit_buckets WHERE ts < ?",
            (now - window,),
        )

        # Read counts under READ COMMITTED (sqlite default).
        rows = db.query_all(
            "SELECT ts FROM rate_limit_buckets "
            "WHERE client_ip = ? AND ts > ?",
            (client_ip, per_minute_cutoff),
        )
        count_last_minute = len(rows)

        if count_last_minute >= self.config.requests_per_minute:
            return False

        # Burst guard - requests in the last 1s.
        recent = sum(1 for r in rows if float(r["ts"]) > now - 1.0)
        if recent >= self.config.burst_size:
            return False

        # Record this request and commit.
        db.execute(
            "INSERT INTO rate_limit_buckets (client_ip, ts) VALUES (?, ?)",
            (client_ip, now),
        )
        return True

    def check(self, client_ip: str) -> None:
        """Check rate limit and raise HTTPException with retry metadata."""
        if not self.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later.",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.config.requests_per_minute),
                },
            )


# Global rate limiter instance (state lives in SQLite, not in this object).
rate_limiter = RateLimiter()


async def rate_limit_middleware(request: Request, call_next):
    """FastAPI middleware for rate limiting."""
    client_ip = request.client.host if request.client else "unknown"
    rate_limiter.check(client_ip)
    return await call_next(request)
