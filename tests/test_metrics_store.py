"""Gravação, agregação e formatação das métricas (sem Ollama, sem Telegram)."""
from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List

import pytest

from jarvis.metrics_store import (
    MetricsStore,
    format_summary,
    format_window,
    metrics_report,
    summarize,
)
from jarvis.observability import LlmCall, RequestTrace, ToolCall


def _trace(rid: str, **kwargs: Any) -> RequestTrace:
    trace = RequestTrace(request_id=rid, model=kwargs.pop("model", "qwen2.5:3b"))
    for key, value in kwargs.items():
        setattr(trace, key, value)
    return trace


def _req(rid: str, latency: float, outcome: str = "ok", **extra: Any) -> Dict[str, Any]:
    row = {
        "request_id": rid,
        "ts": 1000.0,
        "model": "qwen2.5:3b",
        "latency_s": latency,
        "outcome": outcome,
        "iterations": 2,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "eval_s": 1.0,
    }
    row.update(extra)
    return row


def _tool(rid: str, name: str, ok: bool = True, duration: float = 0.1) -> Dict[str, Any]:
    return {"request_id": rid, "tool": name, "ok": 1 if ok else 0, "duration_s": duration}


# ------------------------------------------------------------------ gravação


def test_store_grava_requisicao_e_ferramentas(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    trace = _trace("abc123", source="demo", chat_id=7, iterations=3, mode="native")
    trace.llm_calls = [
        LlmCall(duration_s=1.0, prompt_tokens=300, completion_tokens=40, eval_ns=2e9),
        LlmCall(duration_s=0.5, prompt_tokens=350, completion_tokens=20, eval_ns=1e9),
    ]
    trace.tool_calls = [
        ToolCall("read_file", 0.01, True),
        ToolCall("web_fetch", 0.20, False, "Erro: timeout"),
    ]
    trace.set_outcome("ok")
    trace.finish()
    store.record(trace)

    requests, tools = store.fetch_since(0)
    store.close()
    assert len(requests) == 1
    row = requests[0]
    assert row["request_id"] == "abc123"
    assert row["source"] == "demo" and row["chat_id"] == 7
    assert row["prompt_tokens"] == 650
    assert row["completion_tokens"] == 60
    assert row["eval_s"] == pytest.approx(3.0)
    assert row["tokens_per_s"] == pytest.approx(20.0)  # 60 tokens / 3 s, ponderado
    assert row["llm_calls"] == 2 and row["llm_s"] == pytest.approx(1.5)
    assert row["iterations"] == 3 and row["tool_calls"] == 2
    assert row["outcome"] == "ok" and row["error"] is None
    assert row["created_at"].endswith("+00:00")
    assert [(t["seq"], t["tool"], t["ok"]) for t in tools] == [
        (1, "read_file", 1),
        (2, "web_fetch", 0),
    ]
    assert tools[1]["error"] == "Erro: timeout"


def test_store_sem_tokens_grava_nulos(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    trace = _trace("semtok")
    trace.llm_calls = [LlmCall(duration_s=0.2, error="OllamaUnavailable: x")]
    trace.set_outcome("error", "OllamaUnavailable: x")
    trace.finish()
    store.record(trace)
    (row,), _ = store.fetch_since(0)
    store.close()
    assert row["prompt_tokens"] is None and row["tokens_per_s"] is None
    assert row["outcome"] == "error" and row["error"].startswith("OllamaUnavailable")


def test_store_filtra_por_janela_de_tempo(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    old = _trace("velha", ts=100.0)
    old.tool_calls = [ToolCall("git", 0.1, True)]
    new = _trace("nova", ts=5000.0)
    new.tool_calls = [ToolCall("list_dir", 0.1, True)]
    for trace in (old, new):
        trace.finish()
        store.record(trace)
    requests, tools = store.fetch_since(1000.0)
    store.close()
    assert [r["request_id"] for r in requests] == ["nova"]
    assert [t["tool"] for t in tools] == ["list_dir"]


def test_store_cria_esquema_versionado(tmp_path):
    path = tmp_path / "m.sqlite3"
    MetricsStore(path).close()
    MetricsStore(path).close()  # reabrir não quebra (CREATE IF NOT EXISTS)
    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert {"metrics", "metric_tool_calls"} <= tables


def test_store_id_repetido_levanta_e_nao_grava_meia_transacao(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    first = _trace("dup")
    first.finish()
    store.record(first)
    second = _trace("dup")
    second.tool_calls = [ToolCall("git", 0.1, True)]
    second.finish()
    with pytest.raises(sqlite3.IntegrityError):
        store.record(second)
    requests, tools = store.fetch_since(0)
    store.close()
    assert len(requests) == 1 and tools == []


# ------------------------------------------------------------------ agregação


def test_summarize_percentis_e_taxa_de_erro():
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    requests = [_req("r{}".format(i), lat) for i, lat in enumerate(latencies)]
    requests[0]["outcome"] = "error"
    requests[1]["outcome"] = "max_iterations"
    summary = summarize(requests, [])
    assert summary["requests"] == 10
    assert summary["errors"] == 1 and summary["incomplete"] == 1
    assert summary["error_rate"] == pytest.approx(0.2)
    assert summary["latency_p50_s"] == pytest.approx(5.5)
    assert summary["latency_p95_s"] == pytest.approx(9.55)  # interpolação linear (NumPy)
    assert summary["latency_max_s"] == 10.0
    assert summary["iterations_mean"] == pytest.approx(2.0)
    assert summary["tokens_per_s"] == pytest.approx(20.0)


def test_summarize_vazao_ponderada_ignora_requisicao_sem_tokens():
    requests = [
        _req("a", 1.0, completion_tokens=100, eval_s=1.0),
        _req("b", 1.0, completion_tokens=10, eval_s=1.0),
        _req("c", 1.0, completion_tokens=None, eval_s=None, prompt_tokens=None),
    ]
    summary = summarize(requests, [])
    assert summary["tokens_per_s"] == pytest.approx(55.0)  # 110 / 2, não média de 100 e 10
    assert summary["prompt_tokens_mean"] == pytest.approx(100.0)


def test_summarize_ranking_de_ferramentas():
    tools = (
        [_tool("r1", "read_file")] * 4
        + [_tool("r1", "web_fetch", ok=False), _tool("r2", "web_fetch", ok=False), _tool("r2", "web_fetch")]
        + [_tool("r3", "git", ok=False)]
        + [_tool("r3", "list_dir", duration=0.3), _tool("r3", "list_dir", duration=0.1)]
    )
    summary = summarize([_req("r1", 1.0)], tools)
    used = [(t["tool"], t["calls"]) for t in summary["most_used_tools"]]
    assert used[0] == ("read_file", 4)
    assert used[1] == ("web_fetch", 3)
    assert ("list_dir", 2) in used
    failing = [(t["tool"], t["failures"], t["calls"]) for t in summary["most_failing_tools"]]
    assert failing == [("web_fetch", 2, 3), ("git", 1, 1)]
    assert summary["most_failing_tools"][1]["failure_rate"] == pytest.approx(1.0)
    list_dir = [t for t in summary["most_used_tools"] if t["tool"] == "list_dir"][0]
    assert list_dir["p50_s"] == pytest.approx(0.2)
    assert summary["tool_calls"] == 10 and summary["tool_failures"] == 3


def test_summarize_limita_top():
    tools = [_tool("r", "t{}".format(i)) for i in range(8)]
    assert len(summarize([], tools, top=5)["most_used_tools"]) == 5


def test_summarize_janela_vazia():
    summary = summarize([], [])
    assert summary["requests"] == 0
    assert summary["error_rate"] is None
    assert summary["latency_p50_s"] is None and summary["tokens_per_s"] is None
    assert summary["most_used_tools"] == [] and summary["most_failing_tools"] == []


# ------------------------------------------------------------------ formatação


def test_format_window_com_dados():
    requests = [_req("a", 0.5), _req("b", 2.0, outcome="error"), _req("c", 12.0)]
    tools = [_tool("a", "read_file"), _tool("b", "web_fetch", ok=False)]
    text = format_window("Últimas 24 h", summarize(requests, tools))
    assert text.startswith("== Últimas 24 h ==")
    assert "Requisições: 3 | erro: 1 | sem concluir: 0 | taxa de erro: 33%" in text
    assert "p50 2.0 s" in text and "máx 12.0 s" in text
    assert "Vazão do modelo: 20.0 tokens/s" in text
    assert "Mais usadas:" in text and "read_file" in text
    assert "Mais falham:" in text and "web_fetch        1/1 (100%)" in text
    assert "Modelos: qwen2.5:3b" in text


def test_format_window_milissegundos_e_sem_falhas():
    text = format_window("x", summarize([_req("a", 0.25)], [_tool("a", "git")]))
    assert "p50 250 ms" in text
    assert "Mais falham:" not in text


def test_format_window_vazia():
    assert format_window("Últimos 7 dias", summarize([], [])) == (
        "== Últimos 7 dias ==\nNenhuma requisição registrada."
    )


def test_format_summary_junta_janelas():
    text = format_summary([("A", summarize([], [])), ("B", summarize([_req("x", 1.0)], []))])
    assert text.index("== A ==") < text.index("== B ==")
    assert "\n\n== B ==" in text


def test_metrics_report_separa_24h_de_7_dias(tmp_path):
    now = 10_000_000.0
    store = MetricsStore(tmp_path / "m.sqlite3")
    recent = _trace("recente", ts=now - 3600)
    recent.tool_calls = [ToolCall("read_file", 0.1, True)]
    week = _trace("semana", ts=now - 3 * 86400)
    week.tool_calls = [ToolCall("git", 0.1, False, "Erro: x")]
    week.set_outcome("error", "x")
    ancient = _trace("antiga", ts=now - 30 * 86400)
    for trace in (recent, week, ancient):
        trace.finish()
        store.record(trace)
    text = metrics_report(store, now)
    store.close()
    day, seven = text.split("\n\n")
    assert "Requisições: 1 |" in day and "git" not in day
    assert "Requisições: 2 |" in seven and "taxa de erro: 50%" in seven
    assert "git" in seven


def test_metrics_report_com_relogio_real(tmp_path):
    store = MetricsStore(tmp_path / "m.sqlite3")
    trace = _trace("agora")
    trace.finish()
    store.record(trace)
    text = metrics_report(store, time.time())
    store.close()
    assert text.count("Requisições: 1 |") == 2
