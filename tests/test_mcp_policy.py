"""Allowlist e ponte do servidor MCP — sem o SDK `mcp` (rodam também em Python 3.8)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from jarvis.mcp_server import (
    DEFAULT_TOOLS,
    OPEN_WORLD_TOOLS,
    READ_ONLY_TOOLS,
    TELEGRAM_ONLY_TOOLS,
    resolve_exposed_tools,
)
from tests.mcp_support import NOTE, build_bridge
from tests.test_registry import EXPECTED_TOOLS

UNSAFE = sorted(EXPECTED_TOOLS - READ_ONLY_TOOLS - TELEGRAM_ONLY_TOOLS)


def test_conjunto_de_leitura_e_subconjunto_das_ferramentas():
    assert READ_ONLY_TOOLS <= EXPECTED_TOOLS
    assert TELEGRAM_ONLY_TOOLS <= EXPECTED_TOOLS
    # nenhuma ferramenta que escreve/executa/envia entrou por engano no padrão
    for name in ("run_shell", "write_file", "replace_in_file", "git", "github", "google",
                 "microsoft", "stripe", "notion", "obsidian", "spreadsheet", "knowledge_index",
                 "reminder", "send_file", "speak"):
        assert name not in READ_ONLY_TOOLS


def test_padrao_expoe_so_leitura():
    policy = resolve_exposed_tools(EXPECTED_TOOLS)
    assert set(policy.tools) == DEFAULT_TOOLS
    assert policy.refused == {}
    assert list(policy.tools) == sorted(policy.tools)


def test_padrao_nao_sai_para_a_internet():
    # read_file + web_fetch juntos permitiriam exfiltrar arquivo por URL (prompt injection)
    assert OPEN_WORLD_TOOLS == {"web_search", "web_fetch"}
    assert DEFAULT_TOOLS == READ_ONLY_TOOLS - OPEN_WORLD_TOOLS
    policy = resolve_exposed_tools(EXPECTED_TOOLS)
    for name in OPEN_WORLD_TOOLS:
        assert name not in policy.tools
    assert "read_file" in policy.tools


def test_web_liberavel_pela_lista_sem_allow_unsafe():
    policy = resolve_exposed_tools(EXPECTED_TOOLS, ["leitura", "web_search", "web_fetch"])
    assert set(policy.tools) == READ_ONLY_TOOLS
    assert policy.refused == {}
    assert policy.allow_unsafe is False
    assert "SAI PARA A INTERNET" in policy.describe()


def test_padrao_ignora_leitura_ausente_sem_reclamar():
    # RAG desligado: knowledge_search não existe no registro e não é erro de configuração
    policy = resolve_exposed_tools(EXPECTED_TOOLS - {"knowledge_search"})
    assert "knowledge_search" not in policy.tools
    assert policy.refused == {}


def test_lista_explicita_restringe():
    policy = resolve_exposed_tools(EXPECTED_TOOLS, ["read_file", " knowledge_search ", ""])
    assert policy.tools == ("knowledge_search", "read_file")


def test_atalho_leitura_mais_repetida():
    policy = resolve_exposed_tools(EXPECTED_TOOLS, ["LEITURA", "read_file"])
    assert set(policy.tools) == DEFAULT_TOOLS


@pytest.mark.parametrize("name", UNSAFE)
def test_perigosa_com_uma_chave_so_fica_de_fora(name):
    policy = resolve_exposed_tools(EXPECTED_TOOLS, ["read_file", name])
    assert name not in policy.tools
    assert "JARVIS_MCP_ALLOW_UNSAFE" in policy.refused[name]
    # a segunda chave sozinha (sem o nome na lista) também não libera nada
    assert name not in resolve_exposed_tools(EXPECTED_TOOLS, None, allow_unsafe=True).tools


@pytest.mark.parametrize("name", UNSAFE)
def test_perigosa_com_as_duas_chaves_entra(name):
    policy = resolve_exposed_tools(EXPECTED_TOOLS, [name], allow_unsafe=True)
    assert policy.tools == (name,)
    assert "ESCREVE" in policy.describe()


@pytest.mark.parametrize("name", sorted(TELEGRAM_ONLY_TOOLS))
def test_ferramentas_do_telegram_nunca(name):
    policy = resolve_exposed_tools(EXPECTED_TOOLS, [name], allow_unsafe=True)
    assert policy.tools == ()
    assert "Telegram" in policy.refused[name]


def test_nome_desconhecido_e_ignorado():
    policy = resolve_exposed_tools(EXPECTED_TOOLS, ["read_file", "apaga_tudo"])
    assert policy.tools == ("read_file",)
    assert "apaga_tudo" in policy.refused
    assert "apaga_tudo" in policy.describe()


def test_settings_le_as_variaveis(monkeypatch):
    from jarvis.config import Settings

    monkeypatch.setenv("JARVIS_MCP_TOOLS", "read_file; write_file")
    monkeypatch.setenv("JARVIS_MCP_ALLOW_UNSAFE", "true")
    settings = Settings.load()
    assert settings.mcp_tools == ["read_file", "write_file"]
    assert settings.mcp_allow_unsafe is True
    monkeypatch.setenv("JARVIS_MCP_TOOLS", "")
    monkeypatch.setenv("JARVIS_MCP_ALLOW_UNSAFE", "")
    settings = Settings.load()
    assert settings.mcp_tools == [] and settings.mcp_allow_unsafe is False


# ------------------------------------------------------------------ ponte


@pytest.fixture
def bridge(settings):
    bridge, rag, metrics = build_bridge(settings)
    yield bridge
    rag.close()
    metrics.close()


def test_specs_vem_do_registro(bridge):
    specs = {spec["name"]: spec for spec in bridge.tool_specs()}
    assert set(specs) == DEFAULT_TOOLS
    read_file = specs["read_file"]
    assert read_file["input_schema"]["required"] == ["path"]
    assert read_file["read_only"] is True and read_file["open_world"] is False


@pytest.fixture
def web_bridge(settings):
    bridge, rag, metrics = build_bridge(settings, ["leitura", "web_search", "web_fetch"])
    yield bridge
    rag.close()
    metrics.close()


def test_specs_web_marcadas_open_world(web_bridge):
    specs = {spec["name"]: spec for spec in web_bridge.tool_specs()}
    assert set(specs) == READ_ONLY_TOOLS
    assert specs["web_fetch"]["open_world"] is True
    assert specs["web_search"]["open_world"] is True
    assert specs["web_fetch"]["read_only"] is True


def test_web_fetch_fora_do_padrao_nao_executa(bridge):
    text, is_error = asyncio.run(bridge.call("web_fetch", {"url": "https://example.com/?q=segredo"}))
    assert is_error and "não está exposta" in text


def test_chamada_de_leitura(bridge):
    text, is_error = asyncio.run(bridge.call("read_file", {"path": "leia.txt"}))
    assert not is_error
    assert "linha dois" in text


def test_busca_no_rag(bridge):
    text, is_error = asyncio.run(bridge.call("knowledge_search", {"query": "Falcon adiado"}))
    assert not is_error
    assert "24 de novembro" in text and "falcon.md" in text
    assert NOTE[:20] in text


def test_ferramenta_nao_exposta_nao_executa(bridge, settings, monkeypatch):
    called = []
    original = bridge.registry.execute

    async def spy(name, args, context=None):
        called.append(name)
        return await original(name, args, context)

    monkeypatch.setattr(bridge.registry, "execute", spy)
    alvo = Path(settings.workspace) / "criado.txt"
    text, is_error = asyncio.run(bridge.call("write_file", {"path": str(alvo), "content": "x"}))
    assert is_error and "não está exposta" in text
    assert called == []  # recusada antes do registro
    assert not alvo.exists()


def test_caminho_fora_da_raiz_recusado_pelo_security(bridge, tmp_path_factory):
    fora = tmp_path_factory.mktemp("fora") / "segredo.txt"
    fora.write_text("não pode sair", encoding="utf-8")
    text, is_error = asyncio.run(bridge.call("read_file", {"path": str(fora)}))
    assert is_error
    assert "fora das pastas permitidas" in text  # mensagem do security.resolve_path
    assert "não pode sair" not in text


def test_credencial_dentro_da_raiz_recusada(bridge, settings):
    env = Path(settings.workspace) / ".env"
    env.write_text("TOKEN=123", encoding="utf-8")
    text, is_error = asyncio.run(bridge.call("read_file", {"path": str(env)}))
    assert is_error and "credenciais" in text and "123" not in text


def test_ssrf_recusado(web_bridge):
    text, is_error = asyncio.run(web_bridge.call("web_fetch", {"url": "http://127.0.0.1:11434/api/tags"}))
    assert is_error and text.startswith("Bloqueado")


def test_metricas_e_auditoria_com_source_mcp(bridge, settings, tmp_path_factory):
    fora = tmp_path_factory.mktemp("fora2") / "x.txt"
    asyncio.run(bridge.call("read_file", {"path": "leia.txt"}))
    asyncio.run(bridge.call("read_file", {"path": str(fora)}))
    rows, tools = bridge.metrics.fetch_since(0)
    assert [row["source"] for row in rows] == ["mcp", "mcp"]
    assert sorted(row["outcome"] for row in rows) == ["error", "ok"]
    assert all(row["model"] == "" and row["llm_calls"] == 0 for row in rows)
    assert [call["tool"] for call in tools] == ["read_file", "read_file"]
    audit = Path(settings.db_path).parent / "audit.log"
    records = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert records and all(record.get("source") == "mcp" for record in records)
    assert all("_source" not in record["args"] for record in records)
