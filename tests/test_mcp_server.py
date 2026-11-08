"""Servidor MCP pelo protocolo, com o cliente do SDK oficial.

Precisa do pacote `mcp` (Python >= 3.10, requirements-mcp.txt); sem ele os testes são
pulados — é o caso do .venv em Python 3.8. No CI rodam no job "mcp" (Python 3.12).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")

from mcp import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

from jarvis.mcp_server import READ_ONLY_TOOLS, build_server  # noqa: E402
from tests.mcp_support import build_bridge, make_workspace  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent


def _text(result) -> str:
    return "\n".join(block.text for block in result.content if getattr(block, "type", "") == "text")


@pytest.fixture
def server_bridge(settings):
    bridge, rag, metrics = build_bridge(settings)
    yield build_server(bridge), bridge
    rag.close()
    metrics.close()


def run(coro):
    return asyncio.run(coro)


def test_lista_so_as_ferramentas_de_leitura(server_bridge):
    server, _ = server_bridge

    async def scenario():
        async with Client(server) as client:
            return await client.list_tools()

    listed = run(scenario())
    tools = {tool.name: tool for tool in listed.tools}
    assert set(tools) == READ_ONLY_TOOLS
    for blocked in ("run_shell", "write_file", "google", "send_file", "knowledge_index"):
        assert blocked not in tools
    read_file = tools["read_file"]
    assert read_file.input_schema["required"] == ["path"]  # mesmo JSON Schema do registro
    assert "path" in read_file.input_schema["properties"]
    assert read_file.annotations.read_only_hint is True
    assert read_file.annotations.destructive_hint is False
    assert tools["web_fetch"].annotations.open_world_hint is True


def test_chama_read_file_e_knowledge_search(server_bridge):
    server, _ = server_bridge

    async def scenario():
        async with Client(server) as client:
            first = await client.call_tool("read_file", {"path": "leia.txt"})
            second = await client.call_tool("knowledge_search", {"query": "quando sai o Falcon"})
            return first, second

    first, second = run(scenario())
    assert first.is_error is False and "linha dois" in _text(first)
    assert second.is_error is False
    assert "24 de novembro" in _text(second) and "falcon.md" in _text(second)


def test_caminho_fora_da_raiz_vira_is_error(server_bridge, tmp_path_factory):
    server, _ = server_bridge
    fora = tmp_path_factory.mktemp("fora") / "segredo.txt"
    fora.write_text("conteúdo secreto", encoding="utf-8")

    async def scenario():
        async with Client(server) as client:
            return await client.call_tool("read_file", {"path": str(fora)})

    result = run(scenario())
    assert result.is_error is True
    assert "fora das pastas permitidas" in _text(result)
    assert "conteúdo secreto" not in _text(result)


def test_ferramenta_bloqueada_nao_executa(server_bridge, settings):
    server, bridge = server_bridge
    alvo = Path(settings.workspace) / "nao_deveria.txt"

    async def scenario():
        async with Client(server) as client:
            shell = await client.call_tool("run_shell", {"command": "echo oi > nao_deveria.txt"})
            write = await client.call_tool("write_file", {"path": str(alvo), "content": "x"})
            return shell, write

    shell, write = run(scenario())
    for result in (shell, write):
        assert result.is_error is True
        assert "não está exposta" in _text(result)
    assert not alvo.exists()
    rows, _ = bridge.metrics.fetch_since(0)
    assert rows == []  # nem chegou ao registro, então nem virou métrica


def test_duas_chaves_expoem_ferramenta_que_escreve(settings):
    bridge, rag, metrics = build_bridge(settings, ["read_file", "write_file"], allow_unsafe=True)
    server = build_server(bridge)
    alvo = Path(settings.workspace) / "saida.txt"
    fora = Path(settings.workspace).parent / "fora_da_raiz_mcp.txt"

    async def scenario():
        async with Client(server) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            ok = await client.call_tool("write_file", {"path": str(alvo), "content": "oi"})
            bad = await client.call_tool("write_file", {"path": str(fora), "content": "x"})
            return names, ok, bad

    try:
        names, ok, bad = run(scenario())
    finally:
        rag.close()
        metrics.close()
    assert names == {"read_file", "write_file"}
    assert ok.is_error is False and alvo.read_text(encoding="utf-8") == "oi"
    # mesmo liberada, a escrita continua presa às pastas permitidas (security.py)
    assert bad.is_error is True and "fora das pastas permitidas" in _text(bad)
    assert not fora.exists()


def test_integracao_stdio(tmp_path):
    """Sobe `python -m jarvis.mcp_server` como processo e fala com ele por stdio."""
    make_workspace(tmp_path)
    data = tmp_path / "data"
    env = {
        "PYTHONPATH": str(BASE_DIR),
        "PYTHONUTF8": "1",
        "JARVIS_ALLOWED_ROOTS": str(tmp_path),
        "JARVIS_WORKSPACE": str(tmp_path),
        "JARVIS_DB_PATH": str(data / "jarvis.sqlite3"),
        "JARVIS_RAG_ROOTS": str(tmp_path),
        "JARVIS_RAG_MODE": "bm25",
        "JARVIS_METRICS_DB_PATH": str(data / "metrics.sqlite3"),
        "JARVIS_MCP_TOOLS": "",
        "JARVIS_MCP_ALLOW_UNSAFE": "",
        "OLLAMA_URL": "http://127.0.0.1:9",
        "JARVIS_LOG_LEVEL": "WARNING",
    }
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "jarvis.mcp_server"], env=env, cwd=str(tmp_path)
    )

    async def scenario():
        async with Client(params) as client:
            names = sorted(tool.name for tool in (await client.list_tools()).tools)
            read = await client.call_tool("read_file", {"path": str(tmp_path / "leia.txt")})
            blocked = await client.call_tool("run_shell", {"command": "echo oi"})
            return names, read, blocked

    names, read, blocked = run(scenario())
    assert names == sorted(READ_ONLY_TOOLS)
    assert read.is_error is False and "linha um" in _text(read)
    assert blocked.is_error is True
    # a chamada virou métrica com source="mcp" no banco configurado pelo ambiente
    import sqlite3

    conn = sqlite3.connect(str(data / "metrics.sqlite3"))
    try:
        sources = [row[0] for row in conn.execute("SELECT source FROM metrics")]
    finally:
        conn.close()
    assert sources == ["mcp"]
    audit = [json.loads(line) for line in (data / "audit.log").read_text(encoding="utf-8").splitlines()]
    assert [record["tool"] for record in audit] == ["read_file"]
    assert os.path.exists(str(data / "rag.sqlite3"))
