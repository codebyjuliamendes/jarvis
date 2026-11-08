"""Parsing das chamadas de ferramenta do agente (nativo e fallback JSON).

Escritos ANTES de extrair o parsing para um módulo reutilizável: fixam o
comportamento atual para a refatoração não mudá-lo.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from jarvis.agent import Agent
from jarvis.llm import ToolsNotSupported
from jarvis.memory import Memory
from jarvis.tools.base import ToolRegistry

parse_tool_call = Agent._parse_tool_call
extract_text_tool = Agent._extract_text_tool


# ---------------------------------------------------------------- nativo


def test_parse_tool_call_formato_ollama_com_dict():
    call = {"function": {"name": "read_file", "arguments": {"path": "a.txt"}}}
    assert parse_tool_call(call) == ("read_file", {"path": "a.txt"})


def test_parse_tool_call_arguments_como_string_json():
    call = {"function": {"name": "git", "arguments": '{"args": ["status"]}'}}
    assert parse_tool_call(call) == ("git", {"args": ["status"]})


def test_parse_tool_call_arguments_string_invalida_vira_dict_vazio():
    call = {"function": {"name": "git", "arguments": "{status"}}
    assert parse_tool_call(call) == ("git", {})


def test_parse_tool_call_arguments_nao_dict_vira_vazio():
    call = {"function": {"name": "git", "arguments": ["status"]}}
    assert parse_tool_call(call) == ("git", {})
    call = {"function": {"name": "git", "arguments": "[1, 2]"}}
    assert parse_tool_call(call) == ("git", {})


def test_parse_tool_call_sem_wrapper_function():
    call = {"name": "list_dir", "arguments": {"path": "."}}
    assert parse_tool_call(call) == ("list_dir", {"path": "."})


def test_parse_tool_call_sem_arguments():
    assert parse_tool_call({"function": {"name": "list_dir"}}) == ("list_dir", {})


def test_parse_tool_call_entrada_invalida():
    assert parse_tool_call("lixo") == ("", {})  # type: ignore[arg-type]
    assert parse_tool_call({}) == ("", {})


# ---------------------------------------------------------- fallback JSON


def test_extract_text_tool_json_puro():
    content = '{"tool": "write_file", "args": {"path": "x.txt", "content": "oi"}}'
    assert extract_text_tool(content) == ("write_file", {"path": "x.txt", "content": "oi"})


def test_extract_text_tool_bloco_markdown():
    content = 'Vou ler:\n```json\n{"tool": "read_file", "args": {"path": "a.md"}}\n```'
    assert extract_text_tool(content) == ("read_file", {"path": "a.md"})


def test_extract_text_tool_bloco_sem_rotulo_json():
    content = '```\n{"tool": "list_dir", "args": {}}\n```'
    assert extract_text_tool(content) == ("list_dir", {})


def test_extract_text_tool_texto_em_volta():
    content = 'Claro! {"tool": "web_search", "args": {"query": "clima"}} pronto.'
    assert extract_text_tool(content) == ("web_search", {"query": "clima"})


def test_extract_text_tool_aceita_name_e_arguments():
    content = '{"name": "git", "arguments": {"args": ["log"]}}'
    assert extract_text_tool(content) == ("git", {"args": ["log"]})


def test_extract_text_tool_parameters_nao_e_reconhecido():
    # Comportamento atual (registrado, não corrigido aqui): a chave
    # "parameters" não é lida, então os argumentos se perdem.
    content = '{"name": "git", "parameters": {"args": ["log"]}}'
    assert extract_text_tool(content) == ("git", {})


def test_extract_text_tool_sem_args_vira_dict_vazio():
    assert extract_text_tool('{"tool": "list_dir"}') == ("list_dir", {})


@pytest.mark.parametrize(
    "content",
    [
        "",
        "Olá! Tudo bem?",
        '{"tool": "x", "args": [1, 2]}',  # args não é dict
        '{"tool": 123, "args": {}}',  # nome não é string
        '{"foo": "bar"}',  # sem tool/name
        '{"tool": "read_file", "args": {"path": "a"}',  # JSON quebrado
        "[1, 2, 3]",
    ],
)
def test_extract_text_tool_rejeita(content):
    assert extract_text_tool(content) is None


def test_extract_text_tool_dois_objetos_cai_no_bloco_cercado():
    # Com dois objetos soltos o regex guloso pega tudo e o JSON falha;
    # um bloco cercado é tentado primeiro e resolve.
    solto = '{"tool": "a", "args": {}} e {"tool": "b", "args": {}}'
    assert extract_text_tool(solto) is None
    cercado = '```json\n{"tool": "a", "args": {}}\n```\n{"tool": "b", "args": {}}'
    assert extract_text_tool(cercado) == ("a", {})


# ------------------------------------------------- laço do agente com mocks


class ScriptedLLM:
    def __init__(self, responses: List[Any], tools_supported: bool = True) -> None:
        self.responses = list(responses)
        self.tools_supported = tools_supported
        self.calls: List[Any] = []

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.calls.append((list(messages), tools))
        if not self.responses:
            return {"content": "fim"}
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            if isinstance(item, ToolsNotSupported):
                self.tools_supported = False
            raise item
        return item


def _registry(executed: List[Dict[str, Any]]) -> ToolRegistry:
    registry = ToolRegistry()

    def handler(args: Dict[str, Any]) -> str:
        executed.append(args)
        return "eco: {}".format(args.get("texto"))

    registry.register(
        "eco",
        "Repete o texto.",
        {"type": "object", "properties": {"texto": {"type": "string"}}, "required": ["texto"]},
        handler,
    )
    return registry


def _agent(settings, llm, executed):
    memory = Memory(settings.db_path)
    return Agent(settings, _registry(executed), memory, llm=llm), memory


def test_agente_modo_nativo_executa_tool_calls(settings):
    executed: List[Dict[str, Any]] = []
    llm = ScriptedLLM(
        [
            {"tool_calls": [{"function": {"name": "eco", "arguments": {"texto": "oi"}}}]},
            {"content": "Feito."},
        ]
    )
    agent, memory = _agent(settings, llm, executed)
    try:
        reply = asyncio.run(agent.respond(1, "repete oi", remember=False))
    finally:
        memory.close()
    assert reply == "Feito."
    assert executed and executed[0]["texto"] == "oi"
    assert executed[0]["_chat_id"] == 1  # contexto injetado
    assert llm.calls[0][1] is not None  # schemas enviados no modo nativo
    assert llm.calls[1][0][-1]["role"] == "tool"


def test_agente_modo_texto_executa_json(settings):
    executed: List[Dict[str, Any]] = []
    llm = ScriptedLLM(
        [{"content": '{"tool": "eco", "args": {"texto": "abc"}}'}, {"content": "Pronto."}],
        tools_supported=False,
    )
    agent, memory = _agent(settings, llm, executed)
    try:
        reply = asyncio.run(agent.respond(2, "repete abc", remember=False))
    finally:
        memory.close()
    assert reply == "Pronto."
    assert executed[0]["texto"] == "abc"
    assert llm.calls[0][1] is None  # sem schemas no modo texto
    assert "MODO TEXTO" in llm.calls[0][0][0]["content"]


def test_agente_cai_para_modo_texto_quando_tools_nao_suportado(settings):
    executed: List[Dict[str, Any]] = []
    llm = ScriptedLLM(
        [
            ToolsNotSupported("sem tools"),
            {"content": '{"tool": "eco", "args": {"texto": "z"}}'},
            {"content": "Ok."},
        ]
    )
    agent, memory = _agent(settings, llm, executed)
    try:
        reply = asyncio.run(agent.respond(3, "repete z", remember=False))
    finally:
        memory.close()
    assert reply == "Ok."
    assert executed[0]["texto"] == "z"
    assert "MODO TEXTO" in llm.calls[1][0][0]["content"]


def test_agente_modo_nativo_nao_executa_json_no_texto(settings):
    # No modo nativo, JSON no content é resposta final, não chamada.
    executed: List[Dict[str, Any]] = []
    content = '{"tool": "eco", "args": {"texto": "x"}}'
    llm = ScriptedLLM([{"content": content}])
    agent, memory = _agent(settings, llm, executed)
    try:
        reply = asyncio.run(agent.respond(4, "x", remember=False))
    finally:
        memory.close()
    assert reply == content
    assert executed == []


def test_agente_para_no_limite_de_iteracoes(settings):
    executed: List[Dict[str, Any]] = []
    call = {"tool_calls": [{"function": {"name": "eco", "arguments": {"texto": "loop"}}}]}
    llm = ScriptedLLM([call] * 10)
    agent, memory = _agent(settings, llm, executed)
    try:
        reply = asyncio.run(agent.respond(5, "loop", remember=False, max_iterations=3))
    finally:
        memory.close()
    assert "3 passos" in reply
    assert len(executed) == 3
