"""Comando /metrics: respeita a lista de usuários autorizados e devolve o resumo.

O ``JarvisBot`` é construído de verdade (sem rede: o PTB só conecta no
``run_polling``); o ``Update`` é um objeto mínimo que registra as respostas.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import List

import pytest

from jarvis.metrics_store import MetricsStore
from jarvis.observability import RequestTrace, ToolCall
from jarvis.telegram_bot import HELP_TEXT, JarvisBot


class FakeMessage:
    def __init__(self) -> None:
        self.replies: List[str] = []

    async def reply_text(self, text: str, parse_mode=None) -> None:
        self.replies.append(text)


def _update(user_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=user_id),
        message=FakeMessage(),
    )


@pytest.fixture
def store(tmp_path):
    db = MetricsStore(tmp_path / "metrics.sqlite3")
    trace = RequestTrace(request_id="r1", model="qwen2.5:3b")
    trace.tool_calls = [ToolCall("read_file", 0.05, True)]
    trace.finish()
    db.record(trace)
    yield db
    db.close()


def _metrics_cmd(settings, metrics, update) -> None:
    # O bot é criado dentro do laço: o PTB cria uma asyncio.Queue no construtor.
    async def go():
        bot = JarvisBot(settings, SimpleNamespace(metrics=metrics), memory=None)
        await bot._cmd_metrics(update, None)

    asyncio.run(go())


def test_metrics_nega_usuario_nao_autorizado(settings, store):
    update = _update(999)  # settings autoriza só o id 1
    _metrics_cmd(settings, store, update)
    assert update.message.replies == ["Acesso negado. Seu ID: 999"]


def test_metrics_responde_resumo_para_autorizado(settings, store):
    update = _update(1)
    _metrics_cmd(settings, store, update)
    (reply,) = update.message.replies
    assert reply.startswith("```\n== Últimas 24 h ==")
    assert "Requisições: 1 |" in reply and "read_file" in reply
    assert "== Últimos 7 dias ==" in reply and reply.endswith("\n```")


def test_metrics_desativadas(settings):
    update = _update(1)
    _metrics_cmd(settings, None, update)
    assert "JARVIS_METRICS_ENABLED" in update.message.replies[0]


def test_ajuda_lista_o_comando():
    assert "/metrics" in HELP_TEXT
