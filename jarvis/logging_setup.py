"""Configuração de logs: texto (padrão, formato de sempre) ou JSON por linha.

Em JSON cada linha traz ``request_id`` (do ``contextvar`` da requisição) e os
campos extras passados como ``extra={"fields": {...}}``.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .observability import current_request_id

TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_FORMATS = ("text", "json")


class RequestIdFilter(logging.Filter):
    """Coloca ``record.request_id`` em todo registro (``None`` fora de requisição)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = current_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """Uma linha JSON por evento, com chaves estáveis para ``jq``/``grep``."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or current_request_id(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                if key not in payload:
                    payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(log_format: str = "text", level: str = "INFO") -> logging.Handler:
    """Configura o logger raiz. Formato desconhecido cai em ``text`` (seguro)."""
    fmt = (log_format or "text").strip().lower()
    if fmt not in LOG_FORMATS:
        fmt = "text"
    numeric: Optional[int] = logging.getLevelName(str(level or "INFO").strip().upper())
    if not isinstance(numeric, int):
        numeric = logging.INFO

    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(TEXT_FORMAT))

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(numeric)
    return handler
