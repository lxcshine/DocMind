# -*- coding: utf-8 -*-
"""
Standardized API Response Format

Provides:
- Unified success/error response wrapper
- Pagination helper
- Consistent response structure across all endpoints
"""

from typing import Any, Optional, Generic, TypeVar, List
from pydantic import BaseModel

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Standardized API response wrapper."""
    success: bool = True
    code: str = "OK"
    message: str = "Success"
    data: Optional[T] = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated API response."""
    success: bool = True
    code: str = "OK"
    message: str = "Success"
    data: List[T]
    total: int = 0
    page: int = 1
    page_size: int = 20
    has_more: bool = False


class ErrorResponse(BaseModel):
    """Standardized error response."""
    success: bool = False
    code: str
    message: str
    correlation_id: Optional[str] = None
    errors: Optional[List[dict]] = None


def success_response(
    data: Any = None,
    message: str = "Success",
    code: str = "OK",
) -> dict:
    """Create a success response dict."""
    return {
        "success": True,
        "code": code,
        "message": message,
        "data": data,
    }


def error_response(
    message: str,
    code: str = "ERROR",
    correlation_id: Optional[str] = None,
    errors: Optional[List[dict]] = None,
) -> dict:
    """Create an error response dict."""
    return {
        "success": False,
        "code": code,
        "message": message,
        "correlation_id": correlation_id,
        "errors": errors,
    }


def paginated_response(
    data: List[Any],
    total: int,
    page: int = 1,
    page_size: int = 20,
) -> dict:
    """Create a paginated response dict."""
    return {
        "success": True,
        "code": "OK",
        "message": "Success",
        "data": data,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_more": (page * page_size) < total,
    }
