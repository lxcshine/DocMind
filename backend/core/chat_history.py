# -*- coding: utf-8 -*-
"""
Chat History Store — async with aiomysql connection pool.

Replaces the old pymysql sync-per-call pattern that blocked the event loop.
Features:
  - aiomysql async connection pool (minsize=2, maxsize=10)
  - Auto-reconnect with pool recycling
  - All methods are async coroutines
  - Backward-compatible sync wrappers for call-sites not yet migrated
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy import — aiomysql may not be installed in all environments
_aiomysql = None


def _get_aiomysql():
    global _aiomysql
    if _aiomysql is None:
        try:
            import aiomysql as _m
            _aiomysql = _m
        except ImportError:
            logger.warning("aiomysql not installed, chat history will use sync fallback")
            _aiomysql = False
    return _aiomysql if _aiomysql is not False else None


class ChatHistoryStore:
    """
    Async chat history store backed by MySQL via aiomysql connection pool.

    Provides both async methods (preferred) and sync wrappers (backward compat).
    """

    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._pool = None
        self._available = False
        self._sync_store = None  # Fallback sync store
        self._init_lock = asyncio.Lock()

    async def _ensure_pool(self):
        """Lazily create the connection pool on first use."""
        if self._pool is not None:
            return True

        am = _get_aiomysql()
        if am is None:
            # Fallback to sync pymysql
            if self._sync_store is None:
                self._sync_store = _SyncFallback(
                    self.host, self.port, self.user, self.password, self.database
                )
            return False

        async with self._init_lock:
            if self._pool is not None:
                return True
            try:
                self._pool = await am.create_pool(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    db=self.database,
                    charset="utf8mb4",
                    minsize=2,
                    maxsize=10,
                    autocommit=True,
                    pool_recycle=1800,  # Recycle connections after 30 min
                )
                await self._init_db()
                self._available = True
                logger.info("Chat history DB pool initialized (aiomysql)")
                return True
            except Exception as e:
                logger.warning(f"Chat history DB pool init failed: {e}. Falling back to sync.")
                if self._sync_store is None:
                    self._sync_store = _SyncFallback(
                        self.host, self.port, self.user, self.password, self.database
                    )
                return False

    async def _init_db(self):
        """Create tables if they don't exist."""
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id VARCHAR(64) PRIMARY KEY,
                        title VARCHAR(255) NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                    )
                """)
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id VARCHAR(64) PRIMARY KEY,
                        session_id VARCHAR(64) NOT NULL,
                        role VARCHAR(16) NOT NULL,
                        content TEXT NOT NULL,
                        sources JSON,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
                        INDEX idx_session (session_id),
                        INDEX idx_session_created (session_id, created_at)
                    )
                """)

    # ---- Async API (preferred) ----

    async def async_save_session(self, session_id: str, title: str):
        if not await self._ensure_pool():
            self._sync_store.save_session(session_id, title)
            return
        try:
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        "INSERT IGNORE INTO chat_sessions (id, title) VALUES (%s, %s)",
                        (session_id, title),
                    )
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    async def async_save_message(
        self,
        message_id: str,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List] = None,
    ):
        if not await self._ensure_pool():
            self._sync_store.save_message(message_id, session_id, role, content, sources)
            return
        try:
            sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
            now = datetime.now()
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        "INSERT INTO chat_messages (id, session_id, role, content, sources, created_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (message_id, session_id, role, content, sources_json, now),
                    )
        except Exception as e:
            logger.error(f"Failed to save message: {e}")

    async def async_get_session_messages(self, session_id: str) -> List[Dict]:
        if not await self._ensure_pool():
            return self._sync_store.get_session_messages(session_id)
        try:
            am = _get_aiomysql()
            async with self._pool.acquire() as conn:
                async with conn.cursor(am.DictCursor) as cursor:
                    await cursor.execute(
                        "SELECT * FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC, id ASC",
                        (session_id,),
                    )
                    rows = await cursor.fetchall()
            result = []
            for row in rows:
                msg = {
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": int(row["created_at"].timestamp() * 1000) if row["created_at"] else 0,
                    "sources": row["sources"] if row["sources"] else [],
                }
                result.append(msg)
            return result
        except Exception as e:
            logger.error(f"Failed to get session messages: {e}")
            return []

    async def async_list_sessions(self) -> List[Dict]:
        if not await self._ensure_pool():
            return self._sync_store.list_sessions()
        try:
            am = _get_aiomysql()
            async with self._pool.acquire() as conn:
                async with conn.cursor(am.DictCursor) as cursor:
                    await cursor.execute(
                        "SELECT * FROM chat_sessions ORDER BY updated_at DESC"
                    )
                    rows = await cursor.fetchall()
            result = []
            for row in rows:
                if row["id"].startswith("search_"):
                    continue
                updated = row.get("updated_at", "")
                if updated and hasattr(updated, "isoformat"):
                    updated = updated.isoformat()
                elif updated:
                    updated = str(updated)
                result.append({
                    "id": row["id"],
                    "title": row["title"],
                    "updated_at": updated,
                })
            return result
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    async def async_delete_session(self, session_id: str):
        if not await self._ensure_pool():
            self._sync_store.delete_session(session_id)
            return
        try:
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        "DELETE FROM chat_sessions WHERE id = %s", (session_id,)
                    )
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")

    # ---- Sync wrappers (backward compatibility) ----
    # These run the async methods in an event loop. They will block the
    # calling thread, but at least the DB I/O itself is non-blocking.

    def save_session(self, session_id: str, title: str):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.async_save_session(session_id, title))
        except RuntimeError:
            # No running loop — run synchronously
            asyncio.run(self.async_save_session(session_id, title))

    def save_message(self, message_id: str, session_id: str, role: str, content: str, sources: Optional[List] = None):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.async_save_message(message_id, session_id, role, content, sources))
        except RuntimeError:
            asyncio.run(self.async_save_message(message_id, session_id, role, content, sources))

    def get_session_messages(self, session_id: str) -> List[Dict]:
        try:
            loop = asyncio.get_running_loop()
            # Need result synchronously — run in thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self.async_get_session_messages(session_id))
                return future.result(timeout=5.0)
        except RuntimeError:
            return asyncio.run(self.async_get_session_messages(session_id))

    def list_sessions(self) -> List[Dict]:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self.async_list_sessions())
                return future.result(timeout=5.0)
        except RuntimeError:
            return asyncio.run(self.async_list_sessions())

    def delete_session(self, session_id: str):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.async_delete_session(session_id))
        except RuntimeError:
            asyncio.run(self.async_delete_session(session_id))

    async def close(self):
        """Gracefully close the pool."""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None


class _SyncFallback:
    """
    Minimal sync pymysql fallback when aiomysql is not available.
    Reuses the old per-call connection pattern.
    """

    def __init__(self, host, port, user, password, database):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._available = False
        self._init_db()

    def _get_connection(self):
        import pymysql
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )

    def _init_db(self):
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id VARCHAR(64) PRIMARY KEY,
                        title VARCHAR(255) NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id VARCHAR(64) PRIMARY KEY,
                        session_id VARCHAR(64) NOT NULL,
                        role VARCHAR(16) NOT NULL,
                        content TEXT NOT NULL,
                        sources JSON,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
                        INDEX idx_session (session_id),
                        INDEX idx_session_created (session_id, created_at)
                    )
                """)
                conn.commit()
            conn.close()
            self._available = True
        except Exception as e:
            logger.warning(f"Sync fallback DB init failed: {e}")
            self._available = False

    def save_session(self, session_id, title):
        if not self._available:
            return
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT IGNORE INTO chat_sessions (id, title) VALUES (%s, %s)",
                    (session_id, title),
                )
                conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def save_message(self, message_id, session_id, role, content, sources=None):
        if not self._available:
            return
        try:
            conn = self._get_connection()
            sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
            now = datetime.now()
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO chat_messages (id, session_id, role, content, sources, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (message_id, session_id, role, content, sources_json, now),
                )
                conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save message: {e}")

    def get_session_messages(self, session_id):
        if not self._available:
            return []
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC, id ASC",
                    (session_id,),
                )
                rows = cursor.fetchall()
            conn.close()
            result = []
            for row in rows:
                msg = {
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": int(row["created_at"].timestamp() * 1000) if row["created_at"] else 0,
                    "sources": row["sources"] if row["sources"] else [],
                }
                result.append(msg)
            return result
        except Exception as e:
            logger.error(f"Failed to get session messages: {e}")
            return []

    def list_sessions(self):
        if not self._available:
            return []
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM chat_sessions ORDER BY updated_at DESC")
                rows = cursor.fetchall()
            conn.close()
            result = []
            for row in rows:
                if row["id"].startswith("search_"):
                    continue
                updated = row.get("updated_at", "")
                if updated and hasattr(updated, "isoformat"):
                    updated = updated.isoformat()
                elif updated:
                    updated = str(updated)
                result.append({"id": row["id"], "title": row["title"], "updated_at": updated})
            return result
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    def delete_session(self, session_id):
        if not self._available:
            return
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
                conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
