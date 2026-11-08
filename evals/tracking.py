"""Registro opcional dos experimentos no MLflow.

Cada execução de eval_tools / eval_rag / compare_models vira um run com
parâmetros (modelo, modo, nº de exemplos, hash do dataset, commit git...),
métricas e, como artefatos, o JSON de resultados e a tabela Markdown.

O MLflow é OPCIONAL: se não estiver instalado (requirements-evals.txt), ou se
JARVIS_MLFLOW_ENABLED=false, as avaliações rodam normalmente e só avisam.

Configuração (.env ou ambiente):
- JARVIS_MLFLOW_TRACKING_URI: padrão sqlite em evals/mlflow.db;
- JARVIS_MLFLOW_EXPERIMENT: padrão "jarvis-evals";
- JARVIS_MLFLOW_ENABLED: padrão true.
"""
from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from evals import common

DEFAULT_DB = common.EVALS_DIR / "mlflow.db"
DEFAULT_ARTIFACTS = common.EVALS_DIR / "mlartifacts"
DEFAULT_EXPERIMENT = "jarvis-evals"

_warned = False


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(common.BASE_DIR / ".env", override=False)


def _warn(message: str) -> None:
    global _warned
    if not _warned:
        print("[mlflow] " + message)
        _warned = True


def enabled() -> bool:
    _load_env()
    return (os.getenv("JARVIS_MLFLOW_ENABLED") or "true").strip().lower() not in {
        "0", "false", "no", "nao", "não", "off",
    }


def tracking_uri() -> str:
    _load_env()
    uri = (os.getenv("JARVIS_MLFLOW_TRACKING_URI") or "").strip()
    return uri or "sqlite:///" + DEFAULT_DB.as_posix()


def _mlflow():
    """Importa o MLflow já configurado, ou devolve None (com aviso)."""
    if not enabled():
        return None
    try:
        logging.getLogger("alembic").setLevel(logging.WARNING)
        logging.getLogger("mlflow").setLevel(logging.WARNING)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import mlflow
    except ImportError:
        _warn(
            "MLflow não instalado; seguindo sem registrar o experimento. "
            "Instale com: uv pip install --python .venv/Scripts/python.exe -r requirements-evals.txt"
        )
        return None
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    name = (os.getenv("JARVIS_MLFLOW_EXPERIMENT") or DEFAULT_EXPERIMENT).strip()
    if mlflow.get_experiment_by_name(name) is None:
        artifact_location = None
        if uri.startswith("sqlite:"):
            # Sem isto os artefatos iriam para ./mlruns relativo ao cwd.
            DEFAULT_ARTIFACTS.mkdir(parents=True, exist_ok=True)
            artifact_location = DEFAULT_ARTIFACTS.as_uri()
        mlflow.create_experiment(name, artifact_location=artifact_location)
    mlflow.set_experiment(name)
    return mlflow


def _clean_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _clean_params(params: Dict[str, Any]) -> Dict[str, str]:
    return {key: ("" if value is None else str(value))[:500] for key, value in params.items()}


# ---------------------------------------------------- params e métricas


def tool_params(result: Dict[str, Any]) -> Dict[str, Any]:
    meta = result["meta"]
    return {
        "eval": "tool_selection",
        "model": result["model"],
        "mode": result["mode"],
        "effective_mode": result["effective_mode"],
        "n_examples": result["metrics"]["n"],
        "dataset": result["dataset"],
        "dataset_sha256": result["dataset_sha256"],
        "n_tools": result["n_tools"],
        "seed": result["seed"],
        "num_ctx": result["num_ctx"],
        "git_commit": meta.get("git_commit"),
        "ollama_version": (meta.get("ollama") or {}).get("version"),
        "gpu": meta.get("gpu"),
        "platform": meta.get("platform"),
    }


def tool_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    m = result["metrics"]
    lat = m["latency_s"]
    metrics: Dict[str, Any] = {
        "tool_accuracy": m["tool_accuracy"],
        "args_accuracy": m["args_accuracy"],
        "args_accuracy_given_tool": m["args_accuracy_given_tool"],
        "invalid_json_rate": m["invalid_json_rate"],
        "leaked_calls": m["leaked_calls"],
        "unknown_tool_calls": m["unknown_tool_calls"],
        "errors": m["errors"],
        "latency_mean_s": lat["mean"],
        "latency_p50_s": lat["p50"],
        "latency_p95_s": lat["p95"],
        "tokens_per_s": m["tokens_per_s"],
        "mean_prompt_tokens": m["mean_prompt_tokens"],
        "warmup_s": result.get("warmup_s"),
    }
    for category, info in m["by_category"].items():
        metrics["accuracy_" + category] = info["accuracy"]
    return _clean_metrics(metrics)


def rag_params(result: Dict[str, Any]) -> Dict[str, Any]:
    meta = result["meta"]
    return {
        "eval": "rag_retrieval",
        "embed_model": result["embed_model"],
        "k": result["k"],
        "n_examples": result["metrics"]["n"],
        "dataset": result["dataset"],
        "dataset_sha256": result["dataset_sha256"],
        "fixtures_sha256": result["fixtures_sha256"],
        "n_documents": result["corpus"]["documents"],
        "n_chunks": result["corpus"]["chunks"],
        "git_commit": meta.get("git_commit"),
        "ollama_version": (meta.get("ollama") or {}).get("version"),
        "gpu": meta.get("gpu"),
    }


def rag_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    m = result["metrics"]
    return _clean_metrics(
        {
            "recall_at_1": m["recall_at_1"],
            "recall_at_5": m["recall_at_5"],
            "mrr": m["mrr"],
            "doc_recall_at_1": m["doc_recall_at_1"],
            "search_latency_p50_s": m["search_latency_s"]["p50"],
            "search_latency_p95_s": m["search_latency_s"]["p95"],
            "index_s": result["corpus"]["index_s"],
        }
    )


# --------------------------------------------------------------- runs


@contextmanager
def parent_run(run_name: str, params: Dict[str, Any]) -> Iterator[Optional[Any]]:
    """Run "pai" (compare_models); os runs de cada modelo ficam aninhados nele."""
    mlflow = _mlflow()
    if mlflow is None:
        yield None
        return
    try:
        run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(_clean_params(params))
    except Exception as exc:  # noqa: BLE001 - tracking nunca derruba a avaliação
        _warn("falha ao abrir o run no MLflow ({}); seguindo sem registrar.".format(exc))
        yield None
        return
    try:
        yield run
    finally:
        mlflow.end_run()


def log_run(
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts: Optional[List[Path]] = None,
    tags: Optional[Dict[str, str]] = None,
    nested: bool = False,
) -> Optional[str]:
    """Registra um run completo. Devolve o run_id, ou None se o MLflow estiver fora."""
    mlflow = _mlflow()
    if mlflow is None:
        return None
    try:
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            mlflow.log_params(_clean_params(params))
            mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)
            for path in artifacts or []:
                if path and Path(path).exists():
                    mlflow.log_artifact(str(path))
            return run.info.run_id
    except Exception as exc:  # noqa: BLE001
        _warn("falha ao registrar no MLflow ({}); resultados locais foram salvos.".format(exc))
        return None


def log_artifacts_to_active(paths: List[Path], metrics: Optional[Dict[str, float]] = None) -> None:
    mlflow = _mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    try:
        if metrics:
            mlflow.log_metrics(metrics)
        for path in paths:
            if path and Path(path).exists():
                mlflow.log_artifact(str(path))
    except Exception as exc:  # noqa: BLE001
        _warn("falha ao enviar artefatos ao MLflow ({}).".format(exc))
