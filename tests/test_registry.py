"""Registro de ferramentas (ToolRegistry e build_registry)."""
from __future__ import annotations

import asyncio
import json

import pytest

from jarvis.rag import RagIndex
from jarvis.reminders import ReminderService
from jarvis.tools import build_registry
from jarvis.tools.base import ToolRegistry

EXPECTED_TOOLS = {
    "run_shell", "read_file", "write_file", "replace_in_file", "list_dir",
    "search_files", "find_files", "git", "github", "obsidian", "notion",
    "spreadsheet", "google", "microsoft", "stripe", "web_search", "web_fetch",
    "send_file", "read_pdf", "read_docx", "read_image", "speak",
    "knowledge_search", "knowledge_index", "reminder",
}


def _schema(props=None, required=None):
    return {"type": "object", "properties": props or {}, "required": required or []}


def test_register_names_e_schemas():
    registry = ToolRegistry()
    registry.register("b_tool", "B", _schema(), lambda args: "b")
    registry.register("a_tool", "A", _schema({"x": {"type": "string"}}, ["x"]), lambda a: "a")
    assert registry.names() == ["a_tool", "b_tool"]
    schemas = registry.schemas()
    assert {s["function"]["name"] for s in schemas} == {"a_tool", "b_tool"}
    for schema in schemas:
        assert schema["type"] == "function"
        assert set(schema["function"]) == {"name", "description", "parameters"}
    only = registry.schemas(only=["a_tool"])
    assert [s["function"]["name"] for s in only] == ["a_tool"]
    assert registry.schemas(only=[]) == []


def test_prompts_do_modo_texto():
    registry = ToolRegistry()
    registry.register("a_tool", "Faz A.", _schema({"x": {"type": "string"}}), lambda a: "a")
    registry.register("b_tool", "Faz B.", _schema(), lambda a: "b")
    description = registry.describe_for_prompt(only=["a_tool"])
    assert "a_tool: Faz A." in description and "b_tool" not in description
    prompt = registry.text_protocol_prompt()
    assert "MODO TEXTO" in prompt
    assert '{"tool": "nome_da_ferramenta"' in prompt
    assert "a_tool" in prompt and "b_tool" in prompt


def test_execute_ferramenta_desconhecida():
    registry = ToolRegistry()
    registry.register("a_tool", "A", _schema(), lambda a: "a")
    result = asyncio.run(registry.execute("nao_existe", {}))
    assert "ferramenta desconhecida" in result and "a_tool" in result


def test_execute_injeta_contexto_sem_sobrescrever_args():
    seen = {}

    def handler(args):
        seen.update(args)
        return "ok"

    registry = ToolRegistry()
    registry.register("a_tool", "A", _schema(), handler)
    asyncio.run(registry.execute("a_tool", {"x": 1, "_chat_id": 99}, context={"chat_id": 7}))
    assert seen == {"x": 1, "_chat_id": 99}
    seen.clear()
    asyncio.run(registry.execute("a_tool", {"x": 2}, context={"chat_id": 7}))
    assert seen == {"x": 2, "_chat_id": 7}


def test_execute_args_nao_dict_e_excecao_do_handler():
    def boom(args):
        raise RuntimeError("falhou feio")

    registry = ToolRegistry()
    registry.register("boom", "B", _schema(), boom)
    result = asyncio.run(registry.execute("boom", "lixo"))  # type: ignore[arg-type]
    assert result == "Erro ao executar 'boom': falhou feio"


def test_execute_trunca_saida_longa():
    registry = ToolRegistry()
    registry.register("big", "B", _schema(), lambda a: "x" * 20000)
    result = asyncio.run(registry.execute("big", {}))
    assert len(result) < 13000 and "truncada" in result


def test_auditoria_mascara_segredos_e_conteudo(tmp_path):
    audit = tmp_path / "audit.log"
    registry = ToolRegistry(audit_path=audit)
    registry.register("a_tool", "A", _schema(), lambda a: "resultado")
    asyncio.run(
        registry.execute(
            "a_tool",
            {"api_key": "sk-123", "content": "texto secreto", "path": "a.txt"},
            context={"chat_id": 5},
        )
    )
    record = json.loads(audit.read_text(encoding="utf-8").strip())
    assert record["tool"] == "a_tool" and record["chat_id"] == 5
    assert record["args"]["api_key"] == "***"
    assert record["args"]["content"] == "<13 chars>"
    assert record["args"]["path"] == "a.txt"
    assert "_chat_id" not in record["args"]
    assert "sk-123" not in audit.read_text(encoding="utf-8")


@pytest.fixture
def full_registry(settings):
    rag = RagIndex(settings)
    reminders = ReminderService(settings)
    try:
        yield build_registry(settings, reminders, rag)
    finally:
        rag.close()
        reminders.close()


def test_build_registry_tem_todas_as_ferramentas(full_registry):
    assert set(full_registry.names()) == EXPECTED_TOOLS


def test_build_registry_sem_rag_e_lembretes(settings):
    names = set(build_registry(settings).names())
    assert names == EXPECTED_TOOLS - {"knowledge_search", "knowledge_index", "reminder"}


def test_schemas_validos(full_registry):
    for schema in full_registry.schemas():
        function = schema["function"]
        params = function["parameters"]
        assert function["description"].strip(), function["name"]
        assert params["type"] == "object", function["name"]
        props = params.get("properties", {})
        assert set(params.get("required", [])) <= set(props), function["name"]
        for prop in props.values():
            assert "type" in prop
            if "enum" in prop:
                assert prop["enum"] and len(set(prop["enum"])) == len(prop["enum"])
    json.dumps(full_registry.schemas())  # serializável para o Ollama
