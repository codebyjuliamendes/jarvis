"""Persistência e resumo das métricas de requisição (comando /metrics).

- ``MetricsStore``: SQLite próprio (``data/metrics.sqlite3``), uma transação por
  requisição.
- ``summarize`` e ``format_summary``: funções puras, sem banco nem Telegram.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .metrics_utils import mean, percentile, ratio
from .observability import RequestTrace

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics (
    request_id        TEXT PRIMARY KEY,
    ts                REAL NOT NULL,
    created_at        TEXT NOT NULL,
    source            TEXT NOT NULL,
    chat_id           INTEGER,
    model             TEXT NOT NULL,
    mode              TEXT NOT NULL,
    latency_s         REAL NOT NULL,
    llm_calls         INTEGER NOT NULL,
    llm_s             REAL NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    eval_s            REAL,
    tokens_per_s      REAL,
    iterations        INTEGER NOT NULL,
    tool_calls        INTEGER NOT NULL,
    outcome           TEXT NOT NULL,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(ts);
CREATE TABLE IF NOT EXISTS metric_tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT NOT NULL REFERENCES metrics(request_id),
    seq         INTEGER NOT NULL,
    tool        TEXT NOT NULL,
    duration_s  REAL NOT NULL,
    ok          INTEGER NOT NULL,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_request ON metric_tool_calls(request_id);
"""

_METRIC_COLUMNS = (
    "request_id",
    "ts",
    "created_at",
    "source",
    "chat_id",
    "model",
    "mode",
    "latency_s",
    "llm_calls",
    "llm_s",
    "prompt_tokens",
    "completion_tokens",
    "eval_s",
    "tokens_per_s",
    "iterations",
    "tool_calls",
    "outcome",
    "error",
)


class MetricsStore:
    """Tabelas ``metrics`` e ``metric_tool_calls`` num arquivo SQLite próprio."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # Telemetria: WAL + synchronous=NORMAL troca o fsync a cada commit
            # (~13 ms medidos no Windows) por < 1 ms. Numa queda de energia
            # pode-se perder as últimas métricas, nunca corromper o banco.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.execute("PRAGMA user_version = {}".format(SCHEMA_VERSION))
            self._conn.commit()

    # -------------------------------------------------------------- escrita

    def record(self, trace: RequestTrace) -> None:
        """Grava uma requisição numa única transação (1 INSERT + 1 executemany)."""
        row = trace.as_row()
        row["created_at"] = (_EPOCH + timedelta(seconds=row["ts"])).isoformat()
        values = tuple(row[column] for column in _METRIC_COLUMNS)
        tools = [
            (trace.request_id, seq, call.tool, call.duration_s, 1 if call.ok else 0, call.error)
            for seq, call in enumerate(trace.tool_calls, start=1)
        ]
        with self._lock:
            with self._conn:  # commit no fim, rollback se der erro
                self._conn.execute(
                    "INSERT INTO metrics ({}) VALUES ({})".format(
                        ", ".join(_METRIC_COLUMNS), ", ".join("?" * len(_METRIC_COLUMNS))
                    ),
                    values,
                )
                if tools:
                    self._conn.executemany(
                        "INSERT INTO metric_tool_calls "
                        "(request_id, seq, tool, duration_s, ok, error) VALUES (?, ?, ?, ?, ?, ?)",
                        tools,
                    )

    # -------------------------------------------------------------- leitura

    def fetch_since(self, since_ts: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Requisições com ``ts >= since_ts`` e as chamadas de ferramenta delas."""
        with self._lock:
            requests = [
                dict(row)
                for row in self._conn.execute(
                    "SELECT * FROM metrics WHERE ts >= ? ORDER BY ts", (float(since_ts),)
                ).fetchall()
            ]
            tools = [
                dict(row)
                for row in self._conn.execute(
                    "SELECT t.* FROM metric_tool_calls t "
                    "JOIN metrics m ON m.request_id = t.request_id "
                    "WHERE m.ts >= ? ORDER BY t.id",
                    (float(since_ts),),
                ).fetchall()
            ]
        return requests, tools

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ------------------------------------------------------------------ resumo


def summarize(
    requests: Sequence[Mapping[str, Any]],
    tool_calls: Sequence[Mapping[str, Any]],
    top: int = 5,
) -> Dict[str, Any]:
    """Resumo de uma janela de tempo. Função pura (recebe linhas já filtradas).

    Taxa de erro = (``error`` + ``max_iterations``) / total. Vazão ponderada:
    soma de tokens gerados ÷ soma do tempo de geração (como nas avaliações).
    """
    total = len(requests)
    errors = sum(1 for row in requests if row.get("outcome") == "error")
    incomplete = sum(1 for row in requests if row.get("outcome") == "max_iterations")
    latencies = [float(row["latency_s"]) for row in requests if row.get("latency_s") is not None]

    completion = [row.get("completion_tokens") for row in requests]
    eval_s = [row.get("eval_s") for row in requests]
    tok_total = 0.0
    eval_total = 0.0
    for tokens, seconds in zip(completion, eval_s):
        if tokens is None or seconds is None or seconds <= 0:
            continue
        tok_total += float(tokens)
        eval_total += float(seconds)

    per_tool: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    for call in tool_calls:
        name = str(call.get("tool"))
        stats = per_tool.setdefault(name, {"tool": name, "calls": 0, "failures": 0, "durations": []})
        stats["calls"] += 1
        if not call.get("ok"):
            stats["failures"] += 1
        if call.get("duration_s") is not None:
            stats["durations"].append(float(call["duration_s"]))

    tools: List[Dict[str, Any]] = []
    for stats in per_tool.values():
        tools.append(
            {
                "tool": stats["tool"],
                "calls": stats["calls"],
                "failures": stats["failures"],
                "failure_rate": ratio(stats["failures"], stats["calls"]),
                "p50_s": percentile(stats["durations"], 50),
            }
        )
    most_used = sorted(tools, key=lambda item: (-item["calls"], item["tool"]))[:top]
    most_failing = sorted(
        (item for item in tools if item["failures"] > 0),
        key=lambda item: (-item["failures"], -(item["failure_rate"] or 0.0), item["tool"]),
    )[:top]

    return {
        "requests": total,
        "errors": errors,
        "incomplete": incomplete,
        "error_rate": ratio(errors + incomplete, total),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_max_s": max(latencies) if latencies else None,
        "iterations_mean": mean(row["iterations"] for row in requests if row.get("iterations") is not None),
        "prompt_tokens_mean": mean(
            row["prompt_tokens"] for row in requests if row.get("prompt_tokens") is not None
        ),
        "completion_tokens_mean": mean(
            row["completion_tokens"] for row in requests if row.get("completion_tokens") is not None
        ),
        "tokens_per_s": (tok_total / eval_total) if eval_total > 0 else None,
        "tool_calls": len(tool_calls),
        "tool_failures": sum(item["failures"] for item in tools),
        "most_used_tools": most_used,
        "most_failing_tools": most_failing,
        "models": sorted({str(row.get("model")) for row in requests if row.get("model")}),
    }


def _fmt_s(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if value < 1:
        return "{:.0f} ms".format(value * 1000)
    return "{:.1f} s".format(value)


def _fmt_pct(value: Optional[float]) -> str:
    return "-" if value is None else "{:.0f}%".format(value * 100)


def _fmt_num(value: Optional[float], digits: int = 0) -> str:
    if value is None:
        return "-"
    return "{:.{}f}".format(value, digits)


def format_window(label: str, summary: Mapping[str, Any]) -> str:
    """Bloco de texto de uma janela (ex.: "Últimas 24 h")."""
    lines = ["== {} ==".format(label)]
    total = summary.get("requests") or 0
    if not total:
        lines.append("Nenhuma requisição registrada.")
        return "\n".join(lines)

    lines.append(
        "Requisições: {} | erro: {} | sem concluir: {} | taxa de erro: {}".format(
            total,
            summary.get("errors", 0),
            summary.get("incomplete", 0),
            _fmt_pct(summary.get("error_rate")),
        )
    )
    lines.append(
        "Latência: p50 {} | p95 {} | máx {}".format(
            _fmt_s(summary.get("latency_p50_s")),
            _fmt_s(summary.get("latency_p95_s")),
            _fmt_s(summary.get("latency_max_s")),
        )
    )
    lines.append(
        "Por requisição: {} iterações | {} tokens de prompt | {} de resposta".format(
            _fmt_num(summary.get("iterations_mean"), 1),
            _fmt_num(summary.get("prompt_tokens_mean")),
            _fmt_num(summary.get("completion_tokens_mean")),
        )
    )
    lines.append(
        "Vazão do modelo: {} tokens/s".format(_fmt_num(summary.get("tokens_per_s"), 1))
    )
    models = summary.get("models") or []
    if models:
        lines.append("Modelos: {}".format(", ".join(models)))

    used = summary.get("most_used_tools") or []
    lines.append(
        "Ferramentas: {} chamada(s), {} falha(s)".format(
            summary.get("tool_calls", 0), summary.get("tool_failures", 0)
        )
    )
    if used:
        lines.append("Mais usadas:")
        for item in used:
            lines.append(
                "  {:<16} {:>3}x  p50 {:>7}  falhas {}".format(
                    item["tool"], item["calls"], _fmt_s(item.get("p50_s")), item["failures"]
                )
            )
    failing = summary.get("most_failing_tools") or []
    if failing:
        lines.append("Mais falham:")
        for item in failing:
            lines.append(
                "  {:<16} {}/{} ({})".format(
                    item["tool"], item["failures"], item["calls"], _fmt_pct(item.get("failure_rate"))
                )
            )
    return "\n".join(lines)


def format_summary(windows: Iterable[Tuple[str, Mapping[str, Any]]]) -> str:
    """Texto completo do /metrics a partir de pares (rótulo, resumo)."""
    return "\n\n".join(format_window(label, summary) for label, summary in windows)


WINDOWS = (("Últimas 24 h", 24 * 3600.0), ("Últimos 7 dias", 7 * 24 * 3600.0))


def metrics_report(store: MetricsStore, now: float) -> str:
    """Consulta as janelas padrão (24 h e 7 dias) e devolve o texto do /metrics."""
    longest = max(seconds for _, seconds in WINDOWS)
    requests, tools = store.fetch_since(now - longest)
    windows = []
    for label, seconds in WINDOWS:
        since = now - seconds
        kept = [row for row in requests if float(row["ts"]) >= since]
        ids = {row["request_id"] for row in kept}
        kept_tools = [call for call in tools if call["request_id"] in ids]
        windows.append((label, summarize(kept, kept_tools)))
    return format_summary(windows)

