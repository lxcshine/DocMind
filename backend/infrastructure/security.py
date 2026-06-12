# -*- coding: utf-8 -*-
"""
Security Middleware

Provides:
- API key authentication middleware (X-API-Key header)
- Request body size limiting (skips multipart/form-data uploads so
  MAX_UPLOAD_SIZE_MB can be larger than the JSON body cap)
- Security headers injection
- IP whitelist/blacklist
"""

import hmac
import time
import logging
from typing import Optional, Set

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from config.settings import settings

logger = logging.getLogger(__name__)


# ===== Security Headers =====

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Cache-Control": "no-store",
}


async def security_headers_middleware(request: Request, call_next):
    """Inject security headers into all responses."""
    response = await call_next(request)

    # Only add security headers to API responses, not SSE streams
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value

    return response


# ===== API Key Authentication =====

class APIKeyAuth:
    """
    API key authentication dependency / middleware.

    Activated only when `settings.API_KEYS` is non-empty (i.e. in
    production). In development (empty list) the check is a no-op so
    local dev loops keep working.

    Keys are accepted via the `X-API-Key` header. Use `hmac.compare_digest`
    to avoid timing attacks.
    """

    def __init__(self, api_keys: Optional[list[str]] = None, public_paths: Optional[list[str]] = None):
        self.api_keys: set[str] = set(api_keys or [])
        self.public_paths: set[str] = set(public_paths or [])
        self.enabled = bool(self.api_keys)

    def add_key(self, key: str) -> None:
        self.api_keys.add(key)
        self.enabled = True

    def remove_key(self, key: str) -> None:
        self.api_keys.discard(key)
        self.enabled = bool(self.api_keys)

    def is_public(self, path: str) -> bool:
        if path in self.public_paths:
            return True
        # Allow docs asset paths and favicon.
        return path.startswith("/api/docs") or path.startswith("/api/redoc")

    def is_valid_key(self, provided: Optional[str]) -> bool:
        if not provided:
            return False
        # Constant-time comparison against every configured key.
        for key in self.api_keys:
            if hmac.compare_digest(provided, key):
                return True
        return False

    async def __call__(self, request: Request) -> None:
        """Validate API key from header. Raises 401 when invalid/missing."""
        if not self.enabled or self.is_public(request.url.path):
            return

        provided = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if not self.is_valid_key(provided):
            client_ip = request.client.host if request.client else "unknown"
            logger.warning(
                f"Unauthorized access attempt from {client_ip} "
                f"to {request.method} {request.url.path}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )


# Module-level singleton so middleware & dependencies share the same state.
api_key_auth = APIKeyAuth(api_keys=settings.API_KEYS, public_paths=settings.PUBLIC_PATHS)


async def api_key_middleware(request: Request, call_next):
    """Middleware variant of APIKeyAuth (works for any path, no Depends)."""
    if api_key_auth.enabled and not api_key_auth.is_public(request.url.path):
        provided = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if not api_key_auth.is_valid_key(provided):
            client_ip = request.client.host if request.client else "unknown"
            logger.warning(
                f"Unauthorized access attempt from {client_ip} "
                f"to {request.method} {request.url.path}"
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "success": False,
                    "code": "UNAUTHORIZED",
                    "message": "Invalid or missing API key",
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )
    return await call_next(request)


# ===== Request Body Size Limiter =====

async def body_size_limit_middleware(request: Request, call_next):
    """
    Reject requests whose body is larger than `settings.MAX_BODY_SIZE_MB`.

    Skips multipart/form-data requests (file uploads) so the JSON body cap
    can be set lower than `MAX_UPLOAD_SIZE_MB`. Per-file size is enforced
    separately by `validate_upload_file` while the file is being streamed.
    """
    content_type = request.headers.get("content-type", "")
    content_length = request.headers.get("content-length")

    # Uploads are bounded per-file inside the route handler, so don't
    # let the body middleware short-circuit them.
    is_multipart = content_type.startswith("multipart/form-data")

    if not is_multipart and content_length:
        max_bytes = settings.MAX_BODY_SIZE_MB * 1024 * 1024
        try:
            length = int(content_length)
        except ValueError:
            length = 0
        if length > max_bytes:
            client_ip = request.client.host if request.client else "unknown"
            logger.warning(
                f"Request body too large: {length} bytes from {client_ip} "
                f"(limit {max_bytes // (1024 * 1024)}MB)"
            )
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "success": False,
                    "code": "PAYLOAD_TOO_LARGE",
                    "message": (
                        f"Request body too large. Maximum: "
                        f"{settings.MAX_BODY_SIZE_MB}MB"
                    ),
                },
            )
    return await call_next(request)


# ===== IP Filter =====

class IPFilter:
    """IP address whitelist/blacklist filter."""

    def __init__(
        self,
        whitelist: Optional[Set[str]] = None,
        blacklist: Optional[Set[str]] = None,
    ):
        self.whitelist: Set[str] = whitelist or set()
        self.blacklist: Set[str] = blacklist or set()
        self.enabled = bool(self.whitelist or self.blacklist)

    def add_to_whitelist(self, ip: str) -> None:
        self.whitelist.add(ip)

    def add_to_blacklist(self, ip: str) -> None:
        self.blacklist.add(ip)

    async def __call__(self, request: Request) -> None:
        """Check if request IP is allowed."""
        if not self.enabled:
            return

        client_ip = request.client.host if request.client else "unknown"

        # Check blacklist first
        if client_ip in self.blacklist:
            logger.warning(f"Blocked request from blacklisted IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )

        # Check whitelist (if configured, only whitelisted IPs allowed)
        if self.whitelist and client_ip not in self.whitelist:
            logger.warning(f"Blocked request from non-whitelisted IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )


# ===== Request Timing =====

async def request_timing_middleware(request: Request, call_next):
    """Log request duration for monitoring."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    # Log slow requests
    if duration > 5.0:
        logger.warning(
            f"Slow request: {request.method} {request.url.path} took {duration:.2f}s"
        )

    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    return response
