# -*- coding: utf-8 -*-
"""
Unified Error Handling Middleware

Provides:
- Global exception handler for all uncaught exceptions
- Standardized error response format
- Request correlation ID injection
- Error logging with full traceback
"""

import uuid
import logging
import traceback
from contextvars import ContextVar

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

logger = logging.getLogger(__name__)

# Context variable for correlation ID
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Get the current request's correlation ID."""
    return correlation_id_var.get()


async def correlation_id_middleware(request: Request, call_next):
    """Middleware that injects a unique correlation ID into each request."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:12])
    correlation_id_var.set(correlation_id)

    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all uncaught exceptions."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:12])
    correlation_id_var.set(correlation_id)

    # Log the error with full traceback
    tb = traceback.format_exc()
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {exc}\n{tb}",
        extra={"correlation_id": correlation_id},
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "code": "INTERNAL_ERROR",
            "message": "An internal server error occurred. Please try again later.",
            "correlation_id": correlation_id,
        },
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with clear error messages."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:12])

    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"],
        })

    logger.warning(
        f"Validation error: {errors}",
        extra={"correlation_id": correlation_id},
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "errors": errors,
            "correlation_id": correlation_id,
        },
    )


async def http_exception_handler(request: Request, exc):
    """Handle FastAPI HTTPException with standardized format."""
    from fastapi import HTTPException

    if not isinstance(exc, HTTPException):
        return None

    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:12])

    # Log 5xx errors as errors, 4xx as warnings
    if exc.status_code >= 500:
        logger.error(
            f"HTTP {exc.status_code}: {exc.detail}",
            extra={"correlation_id": correlation_id},
        )
    else:
        logger.warning(
            f"HTTP {exc.status_code}: {exc.detail}",
            extra={"correlation_id": correlation_id},
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "code": _http_code_to_error_code(exc.status_code),
            "message": exc.detail,
            "correlation_id": correlation_id,
        },
    )


def _http_code_to_error_code(status_code: int) -> str:
    """Map HTTP status codes to error codes."""
    mapping = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMITED",
        500: "INTERNAL_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
    }
    return mapping.get(status_code, "UNKNOWN_ERROR")
