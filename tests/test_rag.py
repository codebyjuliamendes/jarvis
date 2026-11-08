"""RagIndex: modos de busca, sqlite-vec com queda para NumPy, migração e prefixos.

Nenhum teste fala com o Ollama: os embeddings são um saco de palavras determinístico.
"""
from __future__ import annotations

import builtins
import json
import logging
import re
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import List

import pytest

from jarvis import rag as rag_module
from jarvis.rag import RagIndex, fts_query, rrf_fuse, use_prefixes

try:
    import sqlite_vec  # noqa: F401

    HAS_SQLITE_VEC = True
except ImportError:  # pragma: no cover - depende do ambiente
    HAS_SQLITE_VEC = False

BACKENDS = ["numpy"] + (["sqlite-vec"] if HAS_SQLITE_VEC else [])


class FakeEmbedRag(RagIndex):
    """Embeddings por hash de palavras (64 dimensões), registrando o que foi embedado."""

    def __init__(self, settings, fail_embed: bool = False) -> None:
        self.embedded: List[str] = []
        self.fail_embed = fail_embed
        super().__init__(settings)

    def _embed(self, text: str) -> List[float]:
        if self.fail_embed:
            raise RuntimeError("Ollama fora")
        self.embedded.append(text)
        vector = [0.0] * 64
        for word in re.findall(r"\w+", text.lower()):
            vector[sum(ord(char) for char in word) % 64] += 1.0
        return vector

    def _embed_batch(self, texts):
        return [self._embed(text) for text in texts]


DOCS = {
    "falcon.md": "Reunião do projeto Falcon: o lançamento foi adiado para 24 de novembro.",
    "receitas.md": "Bolo de cenoura da vovó: três cenouras, quatro ovos e cobertura de chocolate.",
    "roteador.txt": "Para resetar o roteador, segure o botão traseiro por dez segundos.",
}


def make_corpus(root: Path) -> Path:
    corpus = root / "corpus"
    corpus.mkdir()
    for name, text in DOCS.items():
        (corpus / name).write_text(text, encoding="utf-8")
    return corpus


def make_rag(settings, tmp_path, **overrides):
    fail = overrides.pop("fail_embed", False)
    return FakeEmbedRag(replace(settings, **overrides), fail_embed=fail)


# ------------------------------------------------------------ funções puras


def test_fts_query_tira_stopwords_acentos_e_repetidos():
    assert fts_query("Qual é o valor do aluguel do apartamento?") == (
        '"valor" OR "aluguel" OR "apartamento"'
    )
    assert fts_query("Reunião   reuniao") == '"reuniao"'
    assert fts_query("R$ 2.350,00") == '"2" OR "350" OR "00"'
    assert fts_query("o que é?") == ""
    assert fts_query('aspas "perigosas"') == '"aspas" OR "perigosas"'


def test_rrf_soma_inversos_das_posicoes():
    fused = rrf_fuse([[1, 2, 3], [3, 1]], k=60)
    scores = dict(fused)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62)
    assert [item for item, _ in fused] == [1, 3, 2]


def test_rrf_empate_respeita_a_primeira_lista():
    assert [item for item, _ in rrf_fuse([[7, 8], [8, 7]], k=60)] == [7, 8]


def test_use_prefixes():
    assert use_prefixes("auto", "nomic-embed-text") is True
    assert use_prefixes("auto", "mxbai-embed-large") is False
    assert use_prefixes("true", "qualquer") is True
    assert use_prefixes("false", "nomic-embed-text") is False


# ------------------------------------------------------------- busca real


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_busca_acha_o_documento_certo(settings, tmp_path, backend, mode):
    rag = make_rag(settings, tmp_path, rag_vector_backend=backend, rag_mode=mode)
    try:
        assert rag.vector_backend == backend
        assert rag.fts_enabled
        report = rag.index(paths=[make_corpus(tmp_path)], force=True)
        assert "Arquivos indexados: 3" in report
        results = rag.search("como resetar o roteador?", k=2)
        assert results[0]["name"] == "roteador.txt"
        assert results[0]["mode"] == mode
        assert results[0]["chunk"] == 0 and results[0]["chunks_total"] == 1
        assert set(results[0]) >= {"path", "name", "chunk", "chunks_total", "score", "text"}
    finally:
        rag.close()


@pytest.mark.skipif(not HAS_SQLITE_VEC, reason="sqlite-vec não instalado")
def test_sqlite_vec_e_numpy_dao_o_mesmo_ranking(settings, tmp_path):
    corpus = make_corpus(tmp_path)
    rankings = []
    for backend in ["numpy", "sqlite-vec"]:
        rag = make_rag(
            settings, tmp_path, rag_vector_backend=backend,
            db_path=tmp_path / backend / "x.sqlite3",
        )
        try:
            rag.index(paths=[corpus], force=True)
            results = rag.search("bolo de cenoura com chocolate", k=3, mode="vector")
            rankings.append([(r["name"], round(float(r["score"]), 4)) for r in results])
        finally:
            rag.close()
    assert rankings[0] == rankings[1]


def test_sem_sqlite_vec_cai_para_numpy_com_aviso(settings, tmp_path, monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sqlite_vec":
            raise ImportError("sem wheel")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with caplog.at_level(logging.WARNING, logger="jarvis.rag"):
        rag = make_rag(settings, tmp_path, rag_vector_backend="sqlite-vec")
    try:
        assert rag.vector_backend == "numpy"
        assert "sqlite-vec indisponível" in caplog.text
        rag.index(paths=[make_corpus(tmp_path)], force=True)
        assert rag.search("roteador", k=1, mode="vector")[0]["name"] == "roteador.txt"
    finally:
        rag.close()


def test_hibrido_sem_ollama_responde_com_bm25(settings, tmp_path, caplog):
    rag = make_rag(settings, tmp_path, rag_mode="hybrid")
    try:
        rag.index(paths=[make_corpus(tmp_path)], force=True)
        rag.fail_embed = True
        with caplog.at_level(logging.WARNING, logger="jarvis.rag"):
            results = rag.search("lançamento do Falcon", k=1)
        assert results[0]["name"] == "falcon.md"
        assert results[0]["mode"] == "bm25"
        assert rag.search("lançamento do Falcon", k=1, mode="vector") == []
    finally:
        rag.close()


def test_modo_invalido_vira_o_padrao(settings, tmp_path):
    rag = make_rag(settings, tmp_path, rag_mode="semantico")
    try:
        assert rag.mode == rag_module.DEFAULT_MODE == "bm25"
    finally:
        rag.close()


# ---------------------------------------------------------------- prefixos


def test_prefixos_no_documento_e_na_consulta(settings, tmp_path):
    rag = make_rag(settings, tmp_path, rag_prefixes="true")
    try:
        rag.index(paths=[make_corpus(tmp_path)], force=True)
        assert rag.embedded and all(t.startswith("search_document: ") for t in rag.embedded)
        rag.embedded.clear()
        rag.search("roteador", k=1, mode="vector")
        assert rag.embedded == ["search_query: roteador"]
        # o texto guardado (e citado) não leva o prefixo
        assert not rag.search("roteador", k=1)[0]["text"].startswith("search_")
    finally:
        rag.close()


def test_trocar_prefixo_reembeda_sem_force(settings, tmp_path):
    corpus = make_corpus(tmp_path)
    rag = make_rag(settings, tmp_path, rag_prefixes="false")
    try:
        rag.index(paths=[corpus])
        assert "Ignorados: 3" in rag.index(paths=[corpus])
    finally:
        rag.close()
    rag = make_rag(settings, tmp_path, rag_prefixes="true")
    try:
        assert "Arquivos indexados: 3" in rag.index(paths=[corpus])
        assert "Ignorados: 3" in rag.index(paths=[corpus])
    finally:
        rag.close()


# ---------------------------------------------------------------- migração


def _legacy_db(path: Path) -> None:
    """Índice no formato anterior ao subprojeto 4 (sem FTS, vec nem embed_sig)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    db.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE "
        "NOT NULL, mtime REAL NOT NULL, size INTEGER NOT NULL, indexed_at TEXT NOT NULL)"
    )
    db.execute(
        "CREATE TABLE chunks (id INTEGER PRIMARY KEY AUTOINCREMENT, document_id INTEGER "
        "NOT NULL, chunk_index INTEGER NOT NULL, text TEXT NOT NULL, embedding TEXT NOT NULL)"
    )
    for doc_id, (name, text) in enumerate(DOCS.items(), start=1):
        db.execute(
            "INSERT INTO documents VALUES (?, ?, 0, 0, '2026-01-01')",
            (doc_id, "/antigo/" + name),
        )
        vector = FakeEmbedRag._embed(type("X", (), {"fail_embed": False, "embedded": []})(), text)
        db.execute(
            "INSERT INTO chunks (document_id, chunk_index, text, embedding) VALUES (?, 0, ?, ?)",
            (doc_id, text, json.dumps(vector)),
        )
    db.commit()
    db.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_indice_antigo_migra_sem_reindexar(settings, tmp_path, backend):
    _legacy_db(Path(settings.db_path).parent / "rag.sqlite3")
    rag = make_rag(settings, tmp_path, rag_vector_backend=backend)
    try:
        assert rag.embedded == []  # nada foi re-embedado
        for mode in ["bm25", "vector", "hybrid"]:
            assert rag.search("resetar roteador", k=1, mode=mode)[0]["name"] == "roteador.txt"
        columns = {r[1] for r in rag._db.execute("PRAGMA table_info(documents)")}
        assert "embed_sig" in columns
    finally:
        rag.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_apagar_documento_limpa_indices_derivados(settings, tmp_path, backend):
    corpus = make_corpus(tmp_path)
    rag = make_rag(settings, tmp_path, rag_vector_backend=backend)
    try:
        rag.index(paths=[corpus], force=True)
        rag._delete_document(str((corpus / "roteador.txt").resolve()))
        assert rag.stats() == {"documents": 2, "chunks": 2}
        assert rag._db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 2
        if backend == "sqlite-vec":
            assert rag._db.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 2
        names = {r["name"] for r in rag.search("roteador botão", k=5, mode="hybrid")}
        assert "roteador.txt" not in names
    finally:
        rag.close()


@pytest.mark.skipif(not HAS_SQLITE_VEC, reason="sqlite-vec não instalado")
def test_indice_vec_divergente_e_reconstruido(settings, tmp_path):
    corpus = make_corpus(tmp_path)
    rag = make_rag(settings, tmp_path, rag_vector_backend="sqlite-vec")
    rag.index(paths=[corpus], force=True)
    rag.close()
    # Um processo sem sqlite-vec apaga um documento: chunks_vec fica com ids órfãos.
    rag = make_rag(settings, tmp_path, rag_vector_backend="numpy")
    rag._delete_document(str((corpus / "receitas.md").resolve()))
    rag.close()
    rag = make_rag(settings, tmp_path, rag_vector_backend="sqlite-vec")
    try:
        assert rag._db.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 2
        assert rag.search("bolo de cenoura", k=3, mode="vector")
        assert "receitas.md" not in {r["name"] for r in rag.search("bolo", k=3, mode="vector")}
    finally:
        rag.close()


def test_sem_fts5_bm25_vira_vector(settings, tmp_path, monkeypatch):
    rag = make_rag(settings, tmp_path)
    try:
        rag.index(paths=[make_corpus(tmp_path)], force=True)
        rag.fts_enabled = False  # simula SQLite compilado sem FTS5
        results = rag.search("roteador", k=1, mode="bm25")
        assert results[0]["mode"] == "vector"
        assert rag.search("roteador", k=1, mode="hybrid")[0]["mode"] == "vector"
    finally:
        rag.close()


def test_describe_menciona_a_configuracao(settings, tmp_path):
    rag = make_rag(settings, tmp_path, rag_mode="hybrid", rag_rrf_k=30)
    try:
        text = rag.describe()
        assert "modo=hybrid" in text and "rrf_k=30" in text
        assert rag_module.HYBRID_CANDIDATES == 20
    finally:
        rag.close()


# ------------------------------------------------- knowledge_search (citação)


def test_knowledge_search_devolve_arquivo_e_trecho_para_citar(settings, tmp_path):
    import asyncio

    from jarvis.agent import Agent
    from jarvis.memory import Memory
    from jarvis.reminders import ReminderService
    from jarvis.tools import build_registry

    rag = make_rag(settings, tmp_path, rag_mode="hybrid")
    reminders = ReminderService(settings)
    try:
        corpus = make_corpus(tmp_path)
        rag.index(paths=[corpus], force=True)
        registry = build_registry(settings, reminders, rag)
        output = asyncio.run(
            registry.execute("knowledge_search", {"query": "lançamento do Falcon", "k": 2})
        )
        lines = output.splitlines()
        assert lines[0] == "Resultados da memória local:"
        assert "[1] falcon.md, trecho 1 de 1" in lines
        assert "caminho: {}".format((corpus / "falcon.md").resolve()) in lines
        assert any("adiado para 24 de novembro" in line for line in lines)
        # instrução no fim, com o 1º resultado como exemplo
        assert "(fonte: falcon.md, trecho 1)" in lines[-1]
        assert "score=" not in output  # a escala muda com o modo; não ajuda o modelo

        memory = Memory(settings.db_path)
        try:
            prompt = Agent(settings, registry, memory)._system_prompt()
        finally:
            memory.close()
        assert "(fonte: arquivo, trecho N)" in prompt
    finally:
        rag.close()
        reminders.close()


def test_format_results_sem_total_e_texto_longo():
    from jarvis.tools.knowledge import format_results

    text = format_results(
        [{"path": "/x/notas.md", "chunk": 2, "score": 0.5, "text": "linha 1\nlinha 2" + "a" * 2000}],
        max_chars=20,
    )
    assert "[1] notas.md, trecho 3\ncaminho: /x/notas.md\nlinha 1 linha 2aaaaa..." in text


def _output(*items):
    from jarvis.tools.knowledge import format_results

    return format_results(
        [{"path": "/x/" + name, "chunk": chunk - 1, "chunks_total": 9, "text": "t"} for name, chunk in items]
    )


def test_parse_sources_le_a_saida_da_ferramenta():
    from jarvis.tools.knowledge import parse_sources

    out = _output(("a.md", 2), ("b.pdf", 1), ("a.md", 5), ("c.txt", 1))
    assert parse_sources(out) == [("a.md", 2), ("b.pdf", 1), ("a.md", 5)]
    assert parse_sources("Nada encontrado na memória local.") == []
    assert parse_sources("Erro: [1] x.md, trecho 1") == []


def test_ensure_citation_anexa_so_quando_o_modelo_nao_citou():
    from jarvis.tools.knowledge import ensure_citation

    out = _output(("a.md", 2), ("b.pdf", 1), ("a.md", 5), ("c.txt", 1))
    assert ensure_citation("O aluguel é R$ 2.350.", [out]) == (
        "O aluguel é R$ 2.350.\n\nFontes consultadas: a.md (trechos 2, 5); b.pdf (trecho 1)"
    )
    cited = "O aluguel é R$ 2.350 (fonte: a.md, trecho 2)."
    assert ensure_citation(cited, [out]) == cited
    assert ensure_citation("Nada achei.", ["Nada encontrado na memória local."]) == "Nada achei."
    assert ensure_citation("", [out]) == ""


def test_agente_anexa_fontes_quando_o_modelo_nao_cita(settings, tmp_path):
    import asyncio

    from jarvis.agent import Agent
    from jarvis.memory import Memory
    from jarvis.tools import build_registry
    from tests.test_agent_parsing import ScriptedLLM

    rag = make_rag(settings, tmp_path, rag_mode="bm25")
    memory = Memory(settings.db_path)
    try:
        rag.index(paths=[make_corpus(tmp_path)], force=True)
        call = {"function": {"name": "knowledge_search", "arguments": {"query": "roteador resetar"}}}
        llm = ScriptedLLM([{"tool_calls": [call]}, {"content": "Segure o botão por 10 s."}])
        agent = Agent(settings, build_registry(settings, rag=rag), memory, llm=llm)
        reply = asyncio.run(agent.respond(1, "como reseto o roteador?", remember=False))
        assert reply == "Segure o botão por 10 s.\n\nFontes consultadas: roteador.txt (trecho 1)"

        llm = ScriptedLLM(
            [{"tool_calls": [call]}, {"content": "Segure 10 s (fonte: roteador.txt, trecho 1)."}]
        )
        agent = Agent(settings, build_registry(settings, rag=rag), memory, llm=llm)
        reply = asyncio.run(agent.respond(1, "como reseto o roteador?", remember=False))
        assert reply == "Segure 10 s (fonte: roteador.txt, trecho 1)."

        llm = ScriptedLLM([{"content": "Oi!"}])  # sem RAG: nada muda
        agent = Agent(settings, build_registry(settings, rag=rag), memory, llm=llm)
        assert asyncio.run(agent.respond(1, "oi", remember=False)) == "Oi!"
    finally:
        memory.close()
        rag.close()
