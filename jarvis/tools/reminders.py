from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from ..reminders import ReminderService
from .base import ToolRegistry

_RELATIVE_RE = re.compile(
    r"^(\d+)\s*(s|seg|segundos?|min|m|minutos?|h|horas?|d|dias?)$", re.IGNORECASE
)


def _timezone(name: Optional[str]):
    if not name:
        return None
    try:
        import pytz

        return pytz.timezone(name)
    except Exception:  # noqa: BLE001
        return None


def parse_when(value: str, timezone_name: Optional[str] = None) -> datetime:
    """Interpreta datas/horas em formatos comuns (pt-BR) e relativos."""
    text = str(value or "").strip()
    if not text:
        raise ValueError("informe quando (ex.: '10m', 'amanhã 14:00', '2026-11-20 18:30')")

    tz = _timezone(timezone_name)
    now = datetime.now(tz) if tz is not None else datetime.now().astimezone()
    lowered = text.lower()

    match = _RELATIVE_RE.match(lowered)
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith("s"):
            delta = timedelta(seconds=amount)
        elif unit.startswith("min") or unit == "m":
            delta = timedelta(minutes=amount)
        elif unit.startswith("h"):
            delta = timedelta(hours=amount)
        else:
            delta = timedelta(days=amount)
        return now + delta

    for prefix, day_offset in (("hoje", 0), ("amanhã", 1), ("amanha", 1)):
        if lowered.startswith(prefix):
            rest = text[len(prefix) :].strip()
            try:
                parsed_time = datetime.strptime(rest, "%H:%M").time()
            except ValueError:
                raise ValueError(
                    "use 'hoje 14:30' ou 'amanhã 09:00' (formato HH:MM)"
                ) from None
            candidate = now.replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=0,
                microsecond=0,
            ) + timedelta(days=day_offset)
            return candidate

    try:
        parsed_time = datetime.strptime(text, "%H:%M").time()
        candidate = now.replace(
            hour=parsed_time.hour,
            minute=parsed_time.minute,
            second=0,
            microsecond=0,
        )
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate
    except ValueError:
        pass

    for fmt in (
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y",
    ):
        try:
            parsed = datetime.strptime(text, fmt)
            if tz is not None:
                return tz.localize(parsed)
            return parsed.astimezone()
        except ValueError:
            continue

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(
            "não entendi a data/hora. Use '10m', 'hoje 18:00', "
            "'amanhã 09:30', '2026-11-20 18:30' ou ISO 8601."
        ) from None
    if parsed.tzinfo is None:
        if tz is not None:
            return tz.localize(parsed)
        return parsed.astimezone()
    return parsed


class ReminderTools:
    def __init__(self, service: ReminderService, registry: ToolRegistry) -> None:
        self.service = service
        self.registry = registry
        self._register()

    def _register(self) -> None:
        self.registry.register(
            "reminder",
            "Cria, lista e cancela lembretes que chegam no Telegram. "
            "Aceita '10m', '2h', 'hoje 18:00', 'amanhã 09:30', '2026-11-20 18:30'.",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "list", "delete"],
                    },
                    "when": {
                        "type": "string",
                        "description": "Quando (add): '10m', 'hoje 18:00', etc.",
                    },
                    "text": {"type": "string", "description": "O que lembrar (add)."},
                    "id": {"type": "integer", "description": "ID do lembrete (delete)."},
                },
                "required": ["action"],
            },
            self._reminder,
        )

    def _reminder(self, args: Dict[str, Any]) -> str:
        action = str(args.get("action") or "").strip().lower()
        chat_id = args.get("_chat_id")
        if not chat_id:
            return "Não consegui identificar o chat para o lembrete."

        if action == "add":
            when_raw = str(args.get("when") or "").strip()
            text = str(args.get("text") or "").strip()
            if not when_raw or not text:
                return "Informe 'when' e 'text' para criar o lembrete."
            try:
                when = parse_when(when_raw, self.service.settings.timezone)
            except ValueError as exc:
                return "Erro: {}".format(exc)
            try:
                reminder_id = self.service.add(int(chat_id), when, text)
            except (ValueError, RuntimeError) as exc:
                return "Erro: {}".format(exc)
            return "Lembrete #{} agendado para {}: {}".format(
                reminder_id, when.strftime("%d/%m/%Y %H:%M"), text
            )

        if action == "list":
            rows = self.service.list_pending(int(chat_id))
            if not rows:
                return "Nenhum lembrete pendente."
            lines = []
            for row in rows:
                try:
                    due = datetime.fromisoformat(row["due_at"])
                    when_text = due.strftime("%d/%m/%Y %H:%M")
                except (TypeError, ValueError):
                    when_text = str(row.get("due_at") or "-")
                lines.append(
                    "#{} - {} - {}".format(row["id"], when_text, row["text"])
                )
            return "\n".join(lines)

        if action == "delete":
            try:
                reminder_id = int(args.get("id") or 0)
            except (TypeError, ValueError):
                reminder_id = 0
            if not reminder_id:
                return "Informe o 'id' do lembrete."
            if self.service.cancel(reminder_id, int(chat_id)):
                return "Lembrete #{} cancelado.".format(reminder_id)
            return "Não encontrei esse lembrete pendente."

        return "Ação desconhecida. Use: add, list, delete."
