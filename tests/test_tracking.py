"""evals/tracking.py: MLflow opcional, params e métricas dos runs (sem Ollama)."""
from __future__ import annotations

import builtins
import sys

from evals import tracking


def _tool_result():
    return {
        "model": "qwen2.5:3b", "mode": "native", "effective_mode": "native",
        "dataset": "evals/data/tool_selection.jsonl", "dataset_sha256": "abc123",
        "n_tools": 25, "seed": 42, "num_ctx": 8192, "warmup_s": 1.5,
        "meta": {"git_commit": "deadbee", "ollama": {"version": "0.40"}, "gpu": "RTX", "platform": "x"},
        "metrics": {
            "n": 10, "tool_accuracy": 0.8, "args_accuracy": 0.5, "args_accuracy_given_tool": 0.7,
            "invalid_json_rate": 0.1, "leaked_calls": 1, "unknown_tool_calls": 0, "errors": 0,
            "latency_s": {"mean": 1.0, "p50": 0.9, "p95": 2.0},
            "tokens_per_s": 70.0, "mean_prompt_tokens": 4000.0,
            "by_category": {"direto": {"accuracy": 0.9}, "ambiguo": {"accuracy": None}},
        },
    }


def test_tool_params_e_metricas():
    result = _tool_result()
    params = tracking.tool_params(result)
    assert params["model"] == "qwen2.5:3b" and params["n_examples"] == 10
    assert params["dataset_sha256"] == "abc123" and params["git_commit"] == "deadbee"
    metrics = tracking.tool_metrics(result)
    assert metrics["tool_accuracy"] == 0.8 and metrics["latency_p95_s"] == 2.0
    assert metrics["accuracy_direto"] == 0.9
    assert "accuracy_ambiguo" not in metrics  # None não vira métrica
    assert all(isinstance(value, float) for value in metrics.values())


def test_rag_params_e_metricas():
    result = {
        "embed_model": "nomic-embed-text", "k": 5, "dataset": "d", "dataset_sha256": "x",
        "fixtures_sha256": "y", "corpus": {"documents": 13, "chunks": 25, "index_s": 2.0},
        "meta": {"git_commit": "c"},
        "metrics": {"n": 36, "recall_at_1": 0.4, "recall_at_5": 0.7, "mrr": 0.5,
                    "doc_recall_at_1": 0.6, "search_latency_s": {"p50": 0.07, "p95": 0.1}},
    }
    assert tracking.rag_params(result)["n_chunks"] == 25
    metrics = tracking.rag_metrics(result)
    assert metrics["recall_at_5"] == 0.7 and metrics["mrr"] == 0.5


def test_sem_mlflow_instalado_so_avisa(monkeypatch, capsys):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("sem mlflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.setenv("JARVIS_MLFLOW_ENABLED", "true")
    monkeypatch.setattr(tracking, "_warned", False)
    assert tracking.log_run("x", {"a": 1}, {"m": 1.0}) is None
    with tracking.parent_run("p", {}) as run:
        assert run is None
    assert "MLflow não instalado" in capsys.readouterr().out


def test_desligado_por_variavel(monkeypatch):
    monkeypatch.setenv("JARVIS_MLFLOW_ENABLED", "false")
    assert tracking.enabled() is False
    assert tracking.log_run("x", {}, {}) is None


def test_tracking_uri_padrao_e_configuravel(monkeypatch):
    monkeypatch.delenv("JARVIS_MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(tracking, "_load_env", lambda: None)
    assert tracking.tracking_uri().startswith("sqlite:///")
    assert tracking.tracking_uri().endswith("evals/mlflow.db")
    monkeypatch.setenv("JARVIS_MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    assert tracking.tracking_uri() == "http://127.0.0.1:5000"
