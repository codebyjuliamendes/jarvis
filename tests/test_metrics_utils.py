"""jarvis/metrics_utils.py e o chat_full do cliente Ollama (com transporte simulado)."""
from __future__ import annotations

import asyncio
import json

import httpx
import numpy as np
import pytest

from jarvis import llm as llm_module
from jarvis.llm import OllamaClient, OllamaError
from jarvis.metrics_utils import (
    aggregate_tokens_per_second,
    latency_summary,
    mean,
    ollama_timings,
    percentile,
    ratio,
    tokens_per_second,
)


@pytest.mark.parametrize("pct", [0, 10, 25, 50, 90, 95, 99, 100])
def test_percentile_igual_ao_numpy(pct):
    values = [3.2, 0.5, 9.1, 4.4, 4.4, 7.0, 1.1, 12.5, 2.0, 6.3, 5.5]
    assert percentile(values, pct) == pytest.approx(float(np.percentile(values, pct)))


def test_percentile_casos_de_borda():
    assert percentile([], 50) is None
    assert percentile([7.0], 95) == 7.0
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_latency_summary():
    summary = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["n"] == 4
    assert summary["mean"] == 2.5
    assert summary["p50"] == 2.5
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["min"] == 1.0 and summary["max"] == 4.0
    assert latency_summary([])["p50"] is None


def test_tokens_per_second():
    assert tokens_per_second({"eval_count": 50, "eval_duration": 500_000_000}) == 100.0
    assert tokens_per_second({"eval_count": 50}) is None
    assert tokens_per_second({"eval_count": 50, "eval_duration": 0}) is None


def test_aggregate_tokens_per_second_pondera_por_tamanho():
    responses = [
        {"eval_count": 10, "eval_duration": 1_000_000_000},  # 10 tok/s
        {"eval_count": 90, "eval_duration": 1_000_000_000},  # 90 tok/s
        {"sem": "metricas"},
    ]
    assert aggregate_tokens_per_second(responses) == 50.0
    assert aggregate_tokens_per_second([]) is None


def test_ollama_timings():
    timings = ollama_timings(
        {
            "total_duration": 2_000_000_000,
            "load_duration": 100_000_000,
            "prompt_eval_count": 1200,
            "prompt_eval_duration": 400_000_000,
            "eval_count": 30,
            "eval_duration": 600_000_000,
        }
    )
    assert timings["total_s"] == 2.0
    assert timings["prompt_eval_count"] == 1200
    assert timings["tokens_per_s"] == 50.0


def test_ratio_e_mean():
    assert ratio(1, 4) == 0.25
    assert ratio(1, 0) is None
    assert mean([1, 2, 3]) == 2.0
    assert mean([]) is None


# --------------------------------------------- OllamaClient com transporte falso


def _patch_transport(monkeypatch, handler):
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(llm_module.httpx, "AsyncClient", factory)


def test_chat_full_devolve_metricas_e_chat_so_a_mensagem(monkeypatch):
    seen = {}

    def handler(request):
        seen["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "oi"},
                "eval_count": 5,
                "eval_duration": 100_000_000,
            },
        )

    _patch_transport(monkeypatch, handler)
    client = OllamaClient("http://ollama", "m", extra_options={"seed": 42})
    data = asyncio.run(client.chat_full([{"role": "user", "content": "x"}]))
    assert data["eval_count"] == 5
    assert seen["payload"]["options"] == {"temperature": 0.2, "num_ctx": 8192, "seed": 42}
    message = asyncio.run(client.chat([{"role": "user", "content": "x"}]))
    assert message == {"role": "assistant", "content": "oi"}


def test_chat_sem_extra_options_mantem_payload_original(monkeypatch):
    seen = {}

    def handler(request):
        seen["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    _patch_transport(monkeypatch, handler)
    client = OllamaClient("http://ollama", "m", num_ctx=4096)
    tools = [{"type": "function", "function": {"name": "t"}}]
    asyncio.run(client.chat([{"role": "user", "content": "x"}], tools=tools))
    assert seen["payload"]["options"] == {"temperature": 0.2, "num_ctx": 4096}
    assert seen["payload"]["tools"] == tools
    assert seen["payload"]["stream"] is False


def test_chat_sem_mensagem_levanta_erro(monkeypatch):
    _patch_transport(monkeypatch, lambda request: httpx.Response(200, json={"done": True}))
    client = OllamaClient("http://ollama", "m")
    with pytest.raises(OllamaError, match="nenhuma mensagem"):
        asyncio.run(client.chat([{"role": "user", "content": "x"}]))


def test_chat_detecta_modelo_sem_tools(monkeypatch):
    _patch_transport(
        monkeypatch,
        lambda request: httpx.Response(400, text='{"error":"m does not support tools"}'),
    )
    client = OllamaClient("http://ollama", "m")
    with pytest.raises(llm_module.ToolsNotSupported):
        asyncio.run(client.chat([{"role": "user", "content": "x"}], tools=[{"x": 1}]))
    assert client.tools_supported is False
