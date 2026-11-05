from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Settings
from .telegram_api import send_message

logger = logging.getLogger(__name__)


class ReminderService:
    """Lembretes persistentes enviados pelo Telegram via APScheduler."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db_path = Path(settings.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._scheduler: Optional[Any] = None
        self._started = False
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    sent_at TEXT
                )
                """
            )
            self._conn.commit()

    def start(self) -> None:
        if self._started:
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError as exc:
            raise RuntimeError(
                "APScheduler não instalado. Rode ./setup.sh novamente."
            ) from exc
        self._scheduler = BackgroundScheduler()
        self._scheduler.start()
        self._started = True
        self._schedule_pending()

    @property
    def is_started(self) -> bool:
        return self._started

    def add(self, chat_id: int, when: datetime, text: str) -> int:
        if not self._started or self._scheduler is None:
            raise RuntimeError("serviço de lembretes não iniciado")
        if when.tzinfo is None:
            when = when.astimezone()
        now = datetime.now().astimezone()
        if when <= now:
            raise ValueError("esse horário já passou")
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO reminders (chat_id, text, due_at, status, created_at) "
                "VALUES (?, ?, ?, 'pending', ?)",
                (int(chat_id), text, when.isoformat(), now.isoformat()),
            )
            self._conn.commit()
            reminder_id = int(cursor.lastrowid or 0)
        self._schedule(reminder_id, int(chat_id), when, text)
        return reminder_id

    def list_pending(self, chat_id: int) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, chat_id, text, due_at, status FROM reminders "
                "WHERE chat_id = ? AND status = 'pending' ORDER BY due_at ASC",
                (int(chat_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def cancel(self, reminder_id: int, chat_id: int) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE reminders SET status = 'cancelled' "
                "WHERE id = ? AND chat_id = ? AND status = 'pending'",
                (int(reminder_id), int(chat_id)),
            )
            self._conn.commit()
            changed = cursor.rowcount > 0
        if changed and self._scheduler is not None:
            try:
                self._scheduler.remove_job("reminder-{}".format(int(reminder_id)))
            except Exception:  # noqa: BLE001
                pass
        return changed

    def _schedule(self, reminder_id: int, chat_id: int, when: datetime, text: str) -> None:
        if self._scheduler is None:
            return
        self._scheduler.add_job(
            self._fire,
            trigger="date",
            run_date=when,
            args=[reminder_id, chat_id, text],
            id="reminder-{}".format(reminder_id),
            replace_existing=True,
            misfire_grace_time=3600,
        )

    def _fire(self, reminder_id: int, chat_id: int, text: str) -> None:
        ok, detail = send_message(
            self.settings.telegram_token, chat_id, "Lembrete: " + text
        )
        status = "sent" if ok else "failed"
        with self._lock:
            self._conn.execute(
                "UPDATE reminders SET status = ?, sent_at = ? WHERE id = ?",
                (status, datetime.now().astimezone().isoformat(), int(reminder_id)),
            )
            self._conn.commit()
        if ok:
            logger.info("Lembrete %s enviado para o chat %s", reminder_id, chat_id)
        else:
            logger.warning("Falha ao enviar lembrete %s: %s", reminder_id, detail)

    def _schedule_pending(self) -> None:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, chat_id, text, due_at FROM reminders "
                "WHERE status = 'pending' ORDER BY due_at ASC"
            ).fetchall()
        now = datetime.now().astimezone()
        for row in rows:
            try:
                due = datetime.fromisoformat(row["due_at"])
            except (TypeError, ValueError):
                continue
            if due.tzinfo is None:
                due = due.astimezone()
            if due <= now:
                self._fire(int(row["id"]), int(row["chat_id"]), row["text"])
            else:
                self._schedule(int(row["id"]), int(row["chat_id"]), due, row["text"])

    def close(self) -> None:
        if self._scheduler is not None:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass
            self._scheduler = None
        self._started = False
        with self._lock:
            self._conn.close()
