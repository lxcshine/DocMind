# -*- coding: utf-8 -*-
"""
DocMind API - Main Application Entry Point

Enterprise-grade FastAPI application with:
- Structured logging with correlation IDs
- Global error handling
- Rate limiting
- CORS configuration
- RAG-Anything pipeline integration
"""

import sys
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException

# Fix Playwright asyncio issue on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ===== Infrastructure Setup =====
from config.settings import settings
from infrastructure.logging import setup_logging
from infrastructure.errors import (
    correlation_id_middleware,
    global_exception_handler,
    validation_exception_handler,
    http_exception_handler,
)
from infrastructure.validation import rate_limit_middleware
from infrastructure.security import (
    security_headers_middleware,
    body_size_limit_middleware,
    request_timing_middleware,
    api_key_middleware,
    api_key_auth,
)

# Initialize logging before anything else
setup_logging(
    log_level=getattr(settings, "LOG_LEVEL", "INFO"),
    log_format=getattr(settings, "LOG_FORMAT", "plain"),
    log_file=str(settings.BASE_DIR / "logs" / "docmind.log") if getattr(settings, "LOG_FILE_ENABLED", False) else None,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("=" * 60)
    logger.info("DocMind API Starting")
    logger.info(f"Version: {settings.APP_VERSION}")
    logger.info(f"Architecture: RAG-Anything (LightRAG + MinerU)")
    logger.info(f"Environment: {getattr(settings, 'ENVIRONMENT', 'development')}")
    logger.info("=" * 60)

    # Initialize RAG pipeline
    try:
        from core.raganything import init_rag
        from api.documents import upload_handler

        rag = await init_rag()
        logger.info("RAG pipeline initialized successfully")
        app.state.rag = rag

        # Inject RAG into DocumentUploadHandler
        upload_handler.set_rag(rag)
        logger.info("RAG instance injected into DocumentUploadHandler")
    except Exception as e:
        logger.warning(f"RAG initialization failed (will retry on first use): {e}")
        app.state.rag = None

    # Initialize MCP session manager
    if getattr(settings, "MCP_ENABLED", True):
        try:
            from core.mcp_server import mcp_lifespan
            async with mcp_lifespan(app):
                logger.info("MCP Server session manager started")
                yield
        except Exception as e:
            logger.warning(f"MCP initialization failed: {e}")
            yield
    else:
        yield

    # Shutdown
    logger.info("Shutting down DocMind API...")
    if app.state.rag and hasattr(app.state.rag, 'lightrag') and app.state.rag.lightrag:
        try:
            await app.state.rag.lightrag.finalize_storages()
            logger.info("RAG storages finalized")
        except Exception as e:
            logger.warning(f"Error during RAG shutdown: {e}")
    logger.info("DocMind API stopped")


# ===== Application Factory =====
app = FastAPI(
    title="DocMind API",
    description="Enterprise Document Intelligence Platform - RAG-Anything Architecture",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs" if getattr(settings, "ENVIRONMENT", "development") == "development" else None,
    redoc_url="/api/redoc" if getattr(settings, "ENVIRONMENT", "development") == "development" else None,
)

# ===== Middleware Stack (order matters) =====

# 1. API key authentication (first, so unauthorised callers don't burn
#    rate-limit budget or get logged with internal paths).
app.middleware("http")(api_key_middleware)

# 2. Rate limiting (reject early before doing any real work)
app.middleware("http")(rate_limit_middleware)

# 3. Body size limit
app.middleware("http")(body_size_limit_middleware)

# 4. Correlation ID injection (for request tracing)
app.middleware("http")(correlation_id_middleware)

# 5. Request timing (for monitoring)
app.middleware("http")(request_timing_middleware)

# 6. Security headers
app.middleware("http")(security_headers_middleware)

# 7. CORS - configured per the spec: when credentials are enabled the
#    wildcard "*" origin is invalid. Default to an explicit list of
#    trusted dev origins; production deployments should set CORS_ORIGINS.
_cors_origins = settings.CORS_ORIGINS
if "*" in _cors_origins and not settings.CORS_ALLOW_WILDCARD:
    logger.warning(
        "CORS wildcard '*' is incompatible with allow_credentials=True; "
        "removing it. Set CORS_ALLOW_WILDCARD=true only for trusted "
        "single-tenant deployments, or configure explicit CORS_ORIGINS."
    )
    _cors_origins = [o for o in _cors_origins if o != "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization", "Content-Type", "X-API-Key",
        "X-Correlation-ID", "X-Requested-With",
    ],
    expose_headers=["X-Response-Time", "X-Correlation-ID"],
)

# ===== Exception Handlers =====
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)

# ===== Router Registration =====
from api import documents, chat, search, memory, ocr

app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(memory.router, prefix="/api/memory", tags=["memory"])
app.include_router(ocr.router, prefix="/api/ocr", tags=["ocr"])

# ===== MCP Server Mount =====
if getattr(settings, "MCP_ENABLED", True):
    try:
        from core.mcp_server import get_mcp_asgi_app
        app.mount("/mcp", get_mcp_asgi_app())
        logger.info("MCP Server mounted at /mcp (Streamable HTTP transport)")
    except Exception as e:
        logger.warning(f"MCP Server mount failed: {e}")


# ===== Health & Info Endpoints =====

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    rag_status = "initialized" if app.state.rag else "not_initialized"
    return {
        "success": True,
        "status": "ok",
        "version": settings.APP_VERSION,
        "architecture": "RAG-Anything (LightRAG + MinerU)",
        "rag_status": rag_status,
    }


@app.get("/api/info")
def app_info():
    """Application information endpoint."""
    return {
        "success": True,
        "name": "DocMind API",
        "version": settings.APP_VERSION,
        "description": "Enterprise Document Intelligence Platform",
        "architecture": "RAG-Anything (LightRAG + MinerU)",
        "features": [
            "PDF/Document parsing via MinerU",
            "Knowledge graph construction via LightRAG",
            "Vector similarity search",
            "Hybrid retrieval (vector + graph)",
            "Multi-modal processing (images, tables, equations)",
            "SSE streaming chat",
            "Adaptive context retrieval (RF-Mem)",
        ],
    }
