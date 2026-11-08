"""Instrumentação do agente, do cliente Ollama, do registro de ferramentas e dos logs.

Nenhum teste fala com o Ollama: o ``OllamaClient`` é real, mas o POST HTTP
(``_post_chat``) é substituído por respostas roteirizadas no formato do Ollama.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, List, Optional

import pytest

from jarvis import observability
from jarvis.agent import Agent
from jarvis.llm import OllamaClient, OllamaUnavailable
from jarvis.logging_setup import JsonFormatter, RequestIdFilter, setup_logging
from jarvis.memory import Memory
from jarvis.metrics_store import MetricsStore
from jarvis.tools.base import ToolRegistry
from jarvis.utils import run_blocking


def _ollama(message: Dict[str, Any], prompt: int = 200, completion: int = 30, eval_ns: float = 1e9) -> Dict[str, Any]:
    return {
        "model": "fake:latest",
        "message": dict({"role": "assistant", "content": ""}, **message),
        "prompt_eval_count": prompt,
        "eval_count": completion,
        "eval_duration": eval_ns,
        "total_duration": eval_ns * 2,
    }


class ScriptedOllama(OllamaClient):
    """OllamaClient de verdade, com o HTTP trocado por um roteiro."""

    def __init__(self, script: List[Any]) -> None:
        super().__init__(url="http://127.0.0.1:9", model="fake:latest")
        self.script = list(script)
        self.payloads: List[Dict[str, Any]] = []

    async def _post_chat(self, payload, tools):  # type: ignore[override]
        self.payloads.append(payload)
        item = self.script.pop(0) if self.script else _ollama({"content": "fim"})
        if isinstance(item, BaseException):
            raise item
        return item


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    params = {"type": "object", "properties": {"texto": {"type": "string"}}}

    def eco(args: Dict[str, Any]) -> str:
        return "eco: {}".format(args.get("texto"))

    def falha_texto(args: Dict[str, Any]) -> str:
        return "Erro: arquivo não encontrado"

    def explode(args: Dict[str, Any]) -> str:
        raise RuntimeError("bum")

    registry.register("eco", "Repete o texto.", params, eco)
    registry.register("falha_texto", "Devolve erro em texto.", params, falha_texto)
    registry.register("explode", "Levanta exceção.", params, explode)
    return registry


def _call(name: str, texto: str = "x") -> Dict[str, Any]:
    return {"function": {"name": name, "arguments": {"texto": texto}}}


class BoomStore:
    """Gravador que sempre falha: a resposta ao usuário não pode mudar."""

    def __init__(self) -> None:
        self.calls = 0

    def record(self, trace) -> None:
        self.calls += 1
        raise RuntimeError("disco cheio")


def _run(settings, script, metrics=None, **kwargs):
    memory = Memory(settings.db_path)
    llm = ScriptedOllama(script)
    agent = Agent(settings, _registry(), memory, llm=llm, metrics=metrics)
    try:
        reply = asyncio.run(agent.respond(kwargs.pop("chat_id", 1), "pedido", remember=False, **kwargs))
    finally:
        memory.close()
    return reply, llm


@pytest.fixture
def store(tmp_path):
    db = MetricsStore(tmp_path / "metrics.sqlite3")
    yield db
    db.close()


# ------------------------------------------------------------------ agente


def test_agente_grava_tokens_iteracoes_e_ferramentas(settings, store):
    script = [
        _ollama({"tool_calls": [_call("eco"), _call("falha_texto")]}, prompt=300, completion=20, eval_ns=1e9),
        _ollama({"tool_calls": [_call("explode"), _call("nao_existe")]}, prompt=400, completion=10, eval_ns=0.5e9),
        _ollama({"content": "Pronto."}, prompt=500, completion=30, eval_ns=1.5e9),
    ]
    reply, _ = _run(settings, script, metrics=store, chat_id=42)
    assert reply == "Pronto."

    (row,), tools = store.fetch_since(0)
    assert row["outcome"] == "ok" and row["error"] is None
    assert row["source"] == "chat" and row["chat_id"] == 42
    assert row["model"] == "fake:latest" and row["mode"] == "native"
    assert row["iterations"] == 3 and row["llm_calls"] == 3
    assert row["prompt_tokens"] == 1200 and row["completion_tokens"] == 60
    assert row["tokens_per_s"] == pytest.approx(20.0)  # 60 / 3 s
    assert row["latency_s"] >= row["llm_s"] >= 0
    assert [(t["tool"], t["ok"]) for t in tools] == [
        ("eco", 1),
        ("falha_texto", 0),
        ("explode", 0),
        ("nao_existe", 0),
    ]
    assert tools[1]["error"].startswith("Erro: arquivo")
    assert tools[2]["error"] == "RuntimeError: bum"
    assert tools[3]["error"] == "ferramenta desconhecida"


def test_agente_registra_erro_do_ollama(settings, store):
    reply, _ = _run(settings, [OllamaUnavailable("Ollama fora do ar")], metrics=store)
    assert reply == "Ollama fora do ar"  # mesma resposta de antes da instrumentação
    (row,), _ = store.fetch_since(0)
    assert row["outcome"] == "error"
    assert row["error"] == "OllamaUnavailable: Ollama fora do ar"
    assert row["llm_calls"] == 1 and row["prompt_tokens"] is None


def test_agente_registra_limite_de_iteracoes(settings, store):
    script = [_ollama({"tool_calls": [_call("eco")]})] * 5
    reply, _ = _run(settings, script, metrics=store, max_iterations=2)
    assert "2 passos" in reply
    (row,), tools = store.fetch_since(0)
    assert row["outcome"] == "max_iterations" and row["iterations"] == 2
    assert len(tools) == 2


def test_agente_modo_texto_e_source(settings, store):
    script = [
        _ollama({"content": '{"tool": "eco", "args": {"texto": "a"}}'}),
        _ollama({"content": "Feito."}),
    ]
    memory = Memory(settings.db_path)
    llm = ScriptedOllama(script)
    llm.tools_supported = False
    agent = Agent(settings, _registry(), memory, llm=llm, metrics=store)
    try:
        reply = asyncio.run(agent.respond(1, "x", remember=False, source="briefing"))
    finally:
        memory.close()
    assert reply == "Feito."
    (row,), tools = store.fetch_since(0)
    assert row["mode"] == "text" and row["source"] == "briefing"
    assert [t["tool"] for t in tools] == ["eco"]


def test_excecao_no_gravador_nao_quebra_o_agente(settings, caplog):
    script = [_ollama({"tool_calls": [_call("eco")]}), _ollama({"content": "Resposta final."})]
    baseline, _ = _run(settings, list(script))
    boom = BoomStore()
    with caplog.at_level(logging.WARNING, logger="jarvis.agent"):
        reply, _ = _run(settings, list(script), metrics=boom)
    assert reply == baseline == "Resposta final."
    assert boom.calls == 1
    assert any("Falha ao gravar métricas" in r.getMessage() for r in caplog.records)


def test_excecao_inesperada_e_gravada_e_relancada(settings, store):
    class Quebra(ScriptedOllama):
        async def _post_chat(self, payload, tools):  # type: ignore[override]
            raise ValueError("inesperado")

    memory = Memory(settings.db_path)
    agent = Agent(settings, _registry(), memory, llm=Quebra([]), metrics=store)
    try:
        with pytest.raises(ValueError):
            asyncio.run(agent.respond(1, "x", remember=False))
    finally:
        memory.close()
    (row,), _ = store.fetch_since(0)
    assert row["outcome"] == "error" and row["error"] == "ValueError: inesperado"


def test_sem_gravador_o_comportamento_e_o_mesmo(settings):
    script = [_ollama({"tool_calls": [_call("eco", "oi")]}), _ollama({"content": "ok"})]
    reply, llm = _run(settings, script)
    assert reply == "ok"
    assert llm.payloads[1]["messages"][-1] == {"role": "tool", "content": "eco: oi", "tool_name": "eco"}


def test_requisicoes_simultaneas_nao_misturam_metricas(settings, store):
    memory = Memory(settings.db_path)

    class Lento(ScriptedOllama):
        async def _post_chat(self, payload, tools):  # type: ignore[override]
            await asyncio.sleep(0.01)
            return await super()._post_chat(payload, tools)

    def agent_for(script):
        return Agent(settings, _registry(), memory, llm=Lento(script), metrics=store)

    a = agent_for([_ollama({"tool_calls": [_call("eco")]}), _ollama({"content": "A"})])
    b = agent_for([_ollama({"content": "B"}, prompt=7)])

    async def both():
        return await asyncio.gather(
            a.respond(1, "a", remember=False), b.respond(2, "b", remember=False)
        )

    try:
        assert asyncio.run(both()) == ["A", "B"]
    finally:
        memory.close()
    requests, tools = store.fetch_since(0)
    by_chat = {row["chat_id"]: row for row in requests}
    assert by_chat[1]["llm_calls"] == 2 and by_chat[1]["tool_calls"] == 1
    assert by_chat[2]["llm_calls"] == 1 and by_chat[2]["prompt_tokens"] == 7
    assert {t["request_id"] for t in tools} == {by_chat[1]["request_id"]}


def test_escopo_externo_define_o_id_da_metrica(settings, store):
    memory = Memory(settings.db_path)
    agent = Agent(settings, _registry(), memory, llm=ScriptedOllama([]), metrics=store)

    async def run():
        with observability.request_id_scope("feedbeef0001"):
            return await agent.respond(1, "x", remember=False)

    try:
        asyncio.run(run())
    finally:
        memory.close()
    (row,), _ = store.fetch_since(0)
    assert row["request_id"] == "feedbeef0001"
    assert observability.current_request_id() is None  # escopo foi desfeito


# ------------------------------------------------------------------ cliente e ganchos


def test_chat_full_alimenta_o_rastro_e_chat_continua_igual():
    llm = ScriptedOllama([_ollama({"content": "oi"}, prompt=11, completion=5, eval_ns=0.25e9)])

    async def run():
        with observability.trace_scope(model="m") as trace:
            message = await llm.chat([{"role": "user", "content": "x"}])
            return message, trace

    message, trace = asyncio.run(run())
    assert message == {"role": "assistant", "content": "oi"}
    assert trace.prompt_tokens() == 11 and trace.completion_tokens() == 5
    assert trace.tokens_per_s() == pytest.approx(20.0)


def test_ganchos_sem_requisicao_ativa_nao_fazem_nada():
    assert observability.current_trace() is None
    observability.record_llm_response({"eval_count": 1}, 0.1)
    observability.record_llm_error(RuntimeError("x"), 0.1)
    observability.record_tool_call("eco", 0.1, True)


def test_gancho_com_dado_estranho_nao_levanta():
    async def run():
        with observability.trace_scope(model="m") as trace:
            observability.record_llm_response(None, 0.1)  # type: ignore[arg-type]
            observability.record_llm_response({"eval_count": "muitos"}, "rápido")  # type: ignore[arg-type]
            return trace

    trace = asyncio.run(run())
    assert trace.completion_tokens() is None


def test_run_blocking_propaga_request_id_para_a_thread():
    seen: List[Optional[str]] = []

    def work():
        seen.append(observability.current_request_id())
        return threading.get_ident()

    async def run():
        with observability.request_id_scope("abc000000001"):
            return await run_blocking(work)

    thread_id = asyncio.run(run())
    assert seen == ["abc000000001"]
    assert thread_id != threading.get_ident()


def test_erro_curto_trunca():
    text = observability.short_error(RuntimeError("x" * 1000))
    assert text.startswith("RuntimeError: ") and len(text) == observability.ERROR_MAX_CHARS


# ------------------------------------------------------------------ logs


def _record(msg: str = "olá %s", args=("mundo",), **extra: Any) -> logging.LogRecord:
    record = logging.LogRecord("jarvis.teste", logging.INFO, __file__, 1, msg, args, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_linha_com_request_id_e_campos():
    record = _record(fields={"latency_s": 1.5, "tool": "eco", "msg": "não sobrescreve"})
    with observability.request_id_scope("0123456789ab"):
        RequestIdFilter().filter(record)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "olá mundo"
    assert payload["request_id"] == "0123456789ab"
    assert payload["level"] == "INFO" and payload["logger"] == "jarvis.teste"
    assert payload["latency_s"] == 1.5 and payload["tool"] == "eco"
    assert payload["ts"].endswith("+00:00")


def test_json_formatter_fora_de_requisicao_e_excecao():
    try:
        raise ValueError("ops")
    except ValueError:
        import sys

        record = _record("falhou", (), exc_info=sys.exc_info())
    payload = json.loads(JsonFormatter().format(record))
    assert payload["request_id"] is None
    assert "ValueError: ops" in payload["exc"]


def test_setup_logging_formato_invalido_cai_em_texto():
    root = logging.getLogger()
    before = (list(root.handlers), root.level)
    try:
        handler = setup_logging("xml", "nivel-que-nao-existe")
        assert not isinstance(handler.formatter, JsonFormatter)
        assert root.level == logging.INFO
        handler = setup_logging("JSON", "debug")
        assert isinstance(handler.formatter, JsonFormatter)
        assert root.level == logging.DEBUG
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in before[0]:
            root.addHandler(h)
        root.setLevel(before[1])


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: List[str] = []
        self.addFilter(RequestIdFilter())
        self.setFormatter(JsonFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


def test_logs_da_requisicao_saem_com_o_mesmo_id_da_metrica(settings, store):
    handler = _ListHandler()
    loggers = [logging.getLogger("jarvis.agent"), logging.getLogger("jarvis.observability")]
    old_levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)
    try:
        _run(settings, [_ollama({"tool_calls": [_call("eco")]}), _ollama({"content": "ok"})], metrics=store)
    finally:
        for lg, level in zip(loggers, old_levels):
            lg.removeHandler(handler)
            lg.setLevel(level)
    (row,), _ = store.fetch_since(0)
    lines = [json.loads(line) for line in handler.lines]
    assert lines, "nenhum log capturado"
    assert {line["request_id"] for line in lines} == {row["request_id"]}
    tool_line = [line for line in lines if line["msg"].startswith("ferramenta executada")][0]
    assert tool_line["tool"] == "eco" and tool_line["ok"] is True
    done = [line for line in lines if line["msg"].startswith("requisição concluída")][0]
    assert done["outcome"] == "ok" and done["iterations"] == 2 and done["tool_calls"] == 1
    assert done["prompt_tokens"] == 400 and done["model"] == "fake:latest"
