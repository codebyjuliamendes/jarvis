from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import security
from ..config import Settings
from ..rag import RagIndex
from ..security import SecurityError
from .base import ToolRegistry


CITATION_HINT = (
    "Ao responder, cite as fontes que usou no formato (fonte: arquivo, trecho N). "
    "Não cite o que não está abaixo."
)


def format_results(results: List[Dict[str, Any]], max_chars: int = 1200) -> str:
    """Resultados do RAG para o modelo: arquivo e trecho de cada um, para ele citar."""
    lines = ["Resultados da memória local. " + CITATION_HINT]
    for index, item in enumerate(results, start=1):
        text = str(item["text"]).strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        path = str(item["path"])
        name = str(item.get("name") or Path(path).name)
        chunk = int(item.get("chunk") or 0) + 1  # para humanos, começa em 1
        total = item.get("chunks_total")
        position = "trecho {} de {}".format(chunk, total) if total else "trecho {}".format(chunk)
        lines.append(
            "\n[{}] {}, {}\ncaminho: {}\n{}".format(index, name, position, path, text)
        )
    return "\n".join(lines)


class KnowledgeTools:
    def __init__(
        self, settings: Settings, rag: RagIndex, registry: ToolRegistry
    ) -> None:
        self.settings = settings
        self.rag = rag
        self.registry = registry
        self._register()

    def _register(self) -> None:
        self.registry.register(
            "knowledge_search",
            "Busca na memória local indexada (notas, PDFs, DOCX, planilhas, textos). "
            "Use antes de responder perguntas sobre conteúdo antigo da usuária ou dos arquivos dela.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "O que procurar."},
                    "k": {
                        "type": "integer",
                        "description": "Número de trechos (padrão 5, máximo 12).",
                    },
                },
                "required": ["query"],
            },
            self._search,
        )
        self.registry.register(
            "knowledge_index",
            "Indexa arquivos/pastas na memória local para busca semântica. "
            "Sem 'path', reindexa as pastas padrão (JARVIS_RAG_ROOTS).",
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Arquivo ou pasta (opcional).",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Reindexar mesmo sem mudanças (padrão false).",
                    },
                    "max_files": {
                        "type": "integer",
                        "description": "Máximo de arquivos nesta rodada (padrão 400).",
                    },
                },
                "required": [],
            },
            self._index,
        )

    def _search(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "Informe o campo 'query'."
        try:
            k = int(args.get("k") or 5)
        except (TypeError, ValueError):
            k = 5
        k = max(1, min(k, 12))
        results = self.rag.search(query, k=k)
        if not results:
            return (
                "Nada encontrado na memória local. "
                "Use knowledge_index para indexar arquivos primeiro."
            )
        return format_results(results)

    def _index(self, args: Dict[str, Any]) -> str:
        raw = args.get("path")
        paths: Optional[list] = None
        if raw:
            try:
                path = security.resolve_path(
                    str(raw),
                    roots=self.settings.allowed_roots,
                    base=self.settings.workspace,
                    must_exist=True,
                )
            except SecurityError as exc:
                return "Erro: {}".format(exc)
            paths = [Path(path)]
        try:
            max_files = int(args.get("max_files") or 400)
        except (TypeError, ValueError):
            max_files = 400
        max_files = max(1, min(max_files, 2000))
        return self.rag.index(
            paths=paths,
            force=bool(args.get("force")),
            max_files=max_files,
        )
