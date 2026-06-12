# -*- coding: utf-8 -*-
"""
Structured Logging System

Provides:
- JSON-formatted structured logging
- Request correlation IDs for tracing
- Log level configuration via environment
- File + console dual output
"""

import logging
import sys
import json
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from config.settings import settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if available
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = record.correlation_id

        # Add exception info
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key in ["request_id", "user_id", "doc_id", "session_id"]:
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        correlation = ""
        if hasattr(record, "correlation_id"):
            correlation = f" [{record.correlation_id[:8]}]"

        if record.exc_info and record.exc_info[0] is not None:
            exc = self.formatException(record.exc_info)
            return f"{timestamp} {record.levelname:<7} {record.name}{correlation} | {record.getMessage()}\n{exc}"

        return f"{timestamp} {record.levelname:<7} {record.name}{correlation} | {record.getMessage()}"


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "plain",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure application-wide logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: "plain" for development, "json" for production
        log_file: Optional log file path
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    formatter = JSONFormatter() if log_format == "json" else PlainFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logging.info(f"Logging initialized: level={log_level}, format={log_format}")
