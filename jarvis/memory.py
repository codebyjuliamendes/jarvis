from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


class Memory:
    """Histórico de conversa por chat, persistido em SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    chat_id INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY (chat_id, key)
                )
                """
            )
            self._conn.commit()

    def add(self, chat_id: int, role: str, content: str) -> None:
        if not content:
            return
        created = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (int(chat_id), role, content, created),
            )
            self._conn.commit()

    def history(self, chat_id: int, limit: int = 40) -> List[Dict[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
                (int(chat_id), int(limit)),
            ).fetchall()
        rows = list(reversed(rows))
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def clear(self, chat_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE chat_id = ?", (int(chat_id),))
            self._conn.commit()

    def get_pref(self, chat_id: int, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM preferences WHERE chat_id = ? AND key = ?",
                (int(chat_id), str(key)),
            ).fetchone()
        return row["value"] if row is not None else default

    def set_pref(self, chat_id: int, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO preferences (chat_id, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT(chat_id, key) DO UPDATE SET value = excluded.value",
                (int(chat_id), str(key), str(value)),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
