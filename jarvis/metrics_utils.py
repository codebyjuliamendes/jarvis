"""Métricas pequenas e reutilizáveis (avaliações e observabilidade).

Sem dependências externas: só a biblioteca padrão.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

NS_PER_SECOND = 1e9


def percentile(values: Iterable[float], pct: float) -> Optional[float]:
    """Percentil por interpolação linear entre ordens (método padrão do NumPy).

    ``pct`` vai de 0 a 100. Devolve ``None`` para lista vazia.
    """
    data = sorted(float(value) for value in values)
    if not data:
        return None
    if not 0.0 <= pct <= 100.0:
        raise ValueError("pct deve estar entre 0 e 100")
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * pct / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return data[lower]
    fraction = position - lower
    return data[lower] + (data[upper] - data[lower]) * fraction


def latency_summary(seconds: Sequence[float]) -> Dict[str, Optional[float]]:
    """Resumo de latências em segundos: n, média, p50, p95, mínimo e máximo."""
    data = [float(value) for value in seconds]
    if not data:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "n": len(data),
        "mean": sum(data) / len(data),
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "min": min(data),
        "max": max(data),
    }


def tokens_per_second(response: Mapping[str, Any]) -> Optional[float]:
    """Tokens gerados por segundo de UMA resposta do Ollama.

    Usa ``eval_count`` (tokens gerados) e ``eval_duration`` (nanossegundos de
    geração). Não inclui o processamento do prompt. ``None`` se faltar dado.
    """
    count = response.get("eval_count")
    duration = response.get("eval_duration")
    if not isinstance(count, (int, float)) or not isinstance(duration, (int, float)):
        return None
    if duration <= 0:
        return None
    return float(count) / (float(duration) / NS_PER_SECOND)


def aggregate_tokens_per_second(responses: Iterable[Mapping[str, Any]]) -> Optional[float]:
    """Vazão agregada: soma de ``eval_count`` ÷ soma de ``eval_duration`` (em s).

    Pondera pelo tamanho de cada resposta, ao contrário da média simples das
    taxas individuais (que superestima respostas curtíssimas).
    """
    total_tokens = 0.0
    total_ns = 0.0
    for response in responses:
        count = response.get("eval_count")
        duration = response.get("eval_duration")
        if not isinstance(count, (int, float)) or not isinstance(duration, (int, float)):
            continue
        if duration <= 0:
            continue
        total_tokens += float(count)
        total_ns += float(duration)
    if total_ns <= 0:
        return None
    return total_tokens / (total_ns / NS_PER_SECOND)


def ollama_timings(response: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Extrai as métricas de uma resposta do Ollama, com durações em segundos."""

    def seconds(key: str) -> Optional[float]:
        value = response.get(key)
        if isinstance(value, (int, float)):
            return float(value) / NS_PER_SECOND
        return None

    def count(key: str) -> Optional[float]:
        value = response.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return {
        "total_s": seconds("total_duration"),
        "load_s": seconds("load_duration"),
        "prompt_eval_count": count("prompt_eval_count"),
        "prompt_eval_s": seconds("prompt_eval_duration"),
        "eval_count": count("eval_count"),
        "eval_s": seconds("eval_duration"),
        "tokens_per_s": tokens_per_second(response),
    }


def ratio(numerator: float, denominator: float) -> Optional[float]:
    """Divisão segura: ``None`` quando o denominador é zero."""
    if not denominator:
        return None
    return float(numerator) / float(denominator)


def mean(values: Iterable[float]) -> Optional[float]:
    data: List[float] = [float(value) for value in values]
    if not data:
        return None
    return sum(data) / len(data)
