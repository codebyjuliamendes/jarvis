#!/usr/bin/env python3
"""Sobreposição lexical entre as perguntas do RAG e o trecho certo (sem Ollama).

Para cada pergunta: tokens de conteúdo (os mesmos do ``fts_query`` do BM25 — minúsculas, sem
acento, sem as stopwords do ``jarvis/rag.py`` e sem tokens de uma letra) e a fração deles que
aparece no trecho que contém a resposta. Mede o quanto um conjunto favorece a busca por
palavras: 100% = toda palavra da pergunta está no trecho; 0% = nenhuma.

    .venv/Scripts/python.exe evals/lexical_overlap.py
    .venv/Scripts/python.exe evals/lexical_overlap.py evals/data/rag_paraphrase.jsonl --show
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import common  # noqa: E402
from jarvis.documents import extract_text  # noqa: E402
from jarvis.rag import _strip_accents, chunk_text, fts_query  # noqa: E402

DEFAULT_DATASETS = [common.DATA_DIR / "rag_retrieval.jsonl", common.DATA_DIR / "rag_paraphrase.jsonl"]


def content_tokens(text: str) -> List[str]:
    """Os tokens que o BM25 usa (mesma função do bot, sem as aspas do FTS5)."""
    query = fts_query(text)
    return [token.strip('"') for token in query.split(" OR ")] if query else []


def word_set(text: str) -> set:
    return {_strip_accents(token) for token in re.findall(r"\w+", text.lower())}


def load_chunks(fixtures: Path = common.FIXTURES_DIR) -> Dict[str, List[str]]:
    return {
        path.name: chunk_text(extract_text(path))
        for path in sorted(fixtures.iterdir())
        if path.is_file()
    }


def answer_chunk(chunks: Dict[str, List[str]], question: Dict[str, Any]) -> Optional[str]:
    """O trecho que o eval aceita como acerto (arquivo certo e contém o snippet)."""
    snippet = common.normalize(question["expected_snippet"])
    for chunk in chunks.get(question["expected_doc"], []):
        if snippet in common.normalize(chunk):
            return chunk
    return None


def overlap_records(questions: List[Dict[str, Any]], chunks: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    records = []
    for question in questions:
        chunk = answer_chunk(chunks, question)
        if chunk is None:
            raise ValueError("{}: snippet não encontrado em {}".format(
                question["id"], question["expected_doc"]))
        tokens = content_tokens(question["question"])
        words = word_set(chunk)
        shared = [token for token in tokens if token in words]
        records.append({
            "id": question["id"],
            "tokens": tokens,
            "shared": shared,
            "overlap": len(shared) / len(tokens) if tokens else 0.0,
        })
    return records


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    overlaps = [r["overlap"] for r in records]
    return {
        "n": len(records),
        "mean_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
        "zero_overlap": sum(1 for o in overlaps if o == 0),
        "full_overlap": sum(1 for o in overlaps if o == 1),
        "mean_tokens": sum(len(r["tokens"]) for r in records) / len(records) if records else None,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", nargs="*", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--show", action="store_true", help="lista os tokens de cada pergunta")
    args = parser.parse_args(argv)
    chunks = load_chunks()
    for path in args.datasets:
        if not path.exists():
            print("{}: não existe".format(path))
            continue
        records = overlap_records(common.load_jsonl(path), chunks)
        summary = summarize(records)
        print("{}: {} perguntas · sobreposição média {:.1f}% · {} com 0% · {} com 100% · "
              "{:.1f} tokens de conteúdo por pergunta".format(
                  path.name, summary["n"], summary["mean_overlap"] * 100, summary["zero_overlap"],
                  summary["full_overlap"], summary["mean_tokens"]))
        if args.show:
            for record in records:
                print("  {:<10} {:>5.0f}%  {}  (em comum: {})".format(
                    record["id"], record["overlap"] * 100, " ".join(record["tokens"]),
                    " ".join(record["shared"]) or "—"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
