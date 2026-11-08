"""Apoio dos testes do servidor MCP: registro real com RAG BM25 num diretório temporário."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence

from jarvis.mcp_server import McpBridge, resolve_exposed_tools
from jarvis.metrics_store import MetricsStore
from jarvis.rag import RagIndex
from jarvis.tools import build_registry

NOTE = "Projeto Falcon: o lançamento foi adiado para 24 de novembro por causa do fornecedor."


class FakeEmbedRag(RagIndex):
    """Embeddings por hash de palavras: indexar não fala com o Ollama (a busca é BM25)."""

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * 32
        for word in re.findall(r"\w+", text.lower()):
            vector[sum(ord(char) for char in word) % 32] += 1.0
        return vector

    def _embed_batch(self, texts):
        return [self._embed(text) for text in texts]


def make_workspace(root: Path) -> Path:
    """Cria arquivos de exemplo dentro da raiz permitida."""
    notes = root / "notas"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "falcon.md").write_text(NOTE + "\n", encoding="utf-8")
    (root / "leia.txt").write_text("linha um\nlinha dois\n", encoding="utf-8")
    return notes


def build_bridge(
    settings,
    requested: Optional[Sequence[str]] = None,
    allow_unsafe: bool = False,
    with_metrics: bool = True,
):
    """Monta registro + RAG indexado + métricas + ponte. Devolve (ponte, rag, métricas)."""
    notes = make_workspace(Path(settings.workspace))
    rag = FakeEmbedRag(settings)
    rag.index(paths=[notes], force=True)
    metrics = MetricsStore(settings.resolved_metrics_db_path()) if with_metrics else None
    registry = build_registry(settings, reminders=None, rag=rag)
    policy = resolve_exposed_tools(registry.names(), requested, allow_unsafe=allow_unsafe)
    return McpBridge(registry, policy, metrics=metrics), rag, metrics
