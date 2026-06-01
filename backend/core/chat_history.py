import pymysql
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ChatHistoryStore:
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._available = False
        self._init_db()

    def _get_connection(self):
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset='utf8mb4',
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
                        INDEX idx_session (session_id)
                    )
                """)
                conn.commit()
            conn.close()
            logger.info("Chat history tables initialized")
            self._available = True
        except Exception as e:
            logger.warning(f"Chat history DB unavailable: {e}. History will not be persisted.")
            self._available = False

    def save_session(self, session_id: str, title: str):
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

    def save_message(self, message_id: str, session_id: str, role: str, content: str, sources: Optional[List] = None):
        if not self._available:
            return
        try:
            conn = self._get_connection()
            import json
            sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO chat_messages (id, session_id, role, content, sources) VALUES (%s, %s, %s, %s, %s)",
                    (message_id, session_id, role, content, sources_json),
                )
                conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save message: {e}")

    def update_message(self, message_id: str, content: str, sources: Optional[List] = None):
        if not self._available:
            return
        try:
            conn = self._get_connection()
            import json
            sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE chat_messages SET content = %s, sources = %s WHERE id = %s",
                    (content, sources_json, message_id),
                )
                conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update message: {e}")

    def get_session_messages(self, session_id: str) -> List[Dict]:
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC",
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

    def list_sessions(self) -> List[Dict]:
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM chat_sessions ORDER BY updated_at DESC")
                rows = cursor.fetchall()
            conn.close()
            result = []
            for row in rows:
                # Filter out search sessions - only return chat sessions
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

    def delete_session(self, session_id: str):
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
                conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
