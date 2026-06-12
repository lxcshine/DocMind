# -*- coding: utf-8 -*-
"""
SQLite-backed Persistent State Layer

Replaces the in-memory `defaultdict` / JSON file storage that was used for:
  - document metadata
  - processing progress
  - rate limit counters

Why SQLite (and not Redis): the deployment may not have Redis available.
SQLite ships with the Python stdlib, supports concurrent readers, ACID
writes, and is safe to share between processes when WAL mode is enabled.
This gives us durability + multi-process safety without adding infra.

Design notes:
  - One connection per thread/process (sqlite3 connections are not shareable
    across threads by default; we use `check_same_thread=False` and guard
    with a per-instance lock for write operations).
  - All tables are created on first connect; idempotent.
  - Writes go through a context manager that commits on success.
"""

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class StateDB:
    """
    Process-wide SQLite store used for progress, metadata, and rate limits.

    Use `with_state_db()` to acquire a short-lived connection (auto-commit),
    or `execute()`/`query()` for one-shot operations.
    """

    SCHEMA = [
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id        TEXT PRIMARY KEY,
            filename      TEXT NOT NULL,
            file_type     TEXT NOT NULL,
            file_size     INTEGER NOT NULL,
            sections_count INTEGER NOT NULL DEFAULT 0,
            status        TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_documents_status
        ON documents(status)
        """,
        """
        CREATE TABLE IF NOT EXISTS progress (
            doc_id          TEXT PRIMARY KEY,
            filename        TEXT NOT NULL,
            status          TEXT NOT NULL,
            progress        INTEGER NOT NULL DEFAULT 0,
            current_step    TEXT NOT NULL DEFAULT '',
            total_steps     INTEGER NOT NULL DEFAULT 0,
            completed_steps INTEGER NOT NULL DEFAULT 0,
            sections_count  INTEGER NOT NULL DEFAULT 0,
            error           TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS rate_limit_buckets (
            client_ip     TEXT NOT NULL,
            ts            REAL NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_rate_limit_ip_ts
        ON rate_limit_buckets(client_ip, ts)
        """,
        """
        CREATE TABLE IF NOT EXISTS ocr_results (
            doc_id          TEXT PRIMARY KEY,
            result_json     TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        )
        """,
    ]

    def __init__(self, db_path: Union[Path, str]):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=10.0,
            isolation_level=None,  # autocommit; we manage txns explicitly
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                for stmt in self.SCHEMA:
                    conn.execute(stmt)
            finally:
                conn.close()
        logger.info(f"StateDB initialized at {self.db_path}")

    @contextmanager
    def transaction(self):
        """Yield a connection inside a transaction (commit on success)."""
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self._lock:
            with self.transaction() as conn:
                conn.execute(sql, tuple(params))

    def executemany(self, sql: str, seq_of_params: Iterable[Iterable[Any]]) -> None:
        with self._lock:
            with self.transaction() as conn:
                conn.executemany(sql, [tuple(p) for p in seq_of_params])

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        conn = self._connect()
        try:
            cur = conn.execute(sql, tuple(params))
            return cur.fetchone()
        finally:
            conn.close()

    def query_all(self, sql: str, params: Iterable[Any] = ()) -> List[sqlite3.Row]:
        conn = self._connect()
        try:
            cur = conn.execute(sql, tuple(params))
            return cur.fetchall()
        finally:
            conn.close()


# Module-level singleton (resolved lazily so tests can override).
_db_instance: Optional[StateDB] = None
_db_lock = threading.Lock()


def get_state_db() -> StateDB:
    """Return the process-wide StateDB instance (creates on first call)."""
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                from config.settings import settings
                _db_instance = StateDB(settings.STATE_DB_PATH)
    return _db_instance


def row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}


def json_dumps(value: Any) -> Optional[str]:
    return None if value is None else json.dumps(value, ensure_ascii=False)


def json_loads(value: Optional[str]) -> Any:
    return None if value is None else json.loads(value)
