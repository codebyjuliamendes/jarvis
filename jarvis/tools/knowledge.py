from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import security
from ..config import Settings
from ..rag import RagIndex
from ..security import SecurityError
from .base import ToolRegistry

RESULTS_HEADER = "Resultados da memória local:"
CITATION_HINT = (
    "Ao responder, use só o que está acima e termine citando as fontes que usou, "
    "assim: (fonte: {name}, trecho {chunk}). Não cite o que não está acima."
)
_SOURCE_LINE = re.compile(r"^\[\d+\] (?P<name>.+?), trecho (?P<chunk>\d+)(?: de \d+)?$")
_MODEL_CITED = re.compile(r"\bfonte\s*:", re.IGNORECASE)


def format_results(results: List[Dict[str, Any]], max_chars: int = 1200) -> str:
    """Resultados do RAG para o modelo: arquivo e trecho de cada um, para ele citar.

    A instrução de citar vem no FIM, com o primeiro resultado como exemplo: modelos
    pequenos seguem melhor o que está mais perto da resposta.
    """
    lines = [RESULTS_HEADER]
    first = None
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
        if first is None:
            first = (name, chunk)
    if first is not None:
        lines.append("\n" + CITATION_HINT.format(name=first[0], chunk=first[1]))
    return "\n".join(lines)


def parse_sources(output: str, limit: int = 3) -> List[Tuple[str, int]]:
    """(arquivo, trecho) dos primeiros resultados de uma saída do knowledge_search."""
    if not output.startswith(RESULTS_HEADER):
        return []
    sources: List[Tuple[str, int]] = []
    for line in output.splitlines():
        match = _SOURCE_LINE.match(line.strip())
        if match:
            sources.append((match.group("name"), int(match.group("chunk"))))
            if len(sources) >= limit:
                break
    return sources


def ensure_citation(reply: str, outputs: Sequence[str], limit: int = 3) -> str:
    """Se a resposta usou o RAG e o modelo não citou, anexa as fontes consultadas.

    "Consultadas" e não "usadas": são os primeiros resultados da busca, e o agente não
    sabe quais o modelo aproveitou. Se o modelo já citou "(fonte: ...)", nada muda.
    """
    if not reply or _MODEL_CITED.search(reply):
        return reply
    grouped: Dict[str, List[int]] = {}
    for output in outputs:
        for name, chunk in parse_sources(output, limit):
            chunks = grouped.setdefault(name, [])
            if chunk not in chunks and sum(len(v) for v in grouped.values()) < limit:
                chunks.append(chunk)
    grouped = {name: chunks for name, chunks in grouped.items() if chunks}
    if not grouped:
        return reply
    parts = [
        "{} ({} {})".format(
            name, "trecho" if len(chunks) == 1 else "trechos", ", ".join(str(c) for c in chunks)
        )
        for name, chunks in grouped.items()
    ]
    return reply.rstrip() + "\n\nFontes consultadas: " + "; ".join(parts)


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
