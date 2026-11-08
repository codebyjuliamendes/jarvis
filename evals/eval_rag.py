#!/usr/bin/env python3
"""Avaliação de recuperação do RAG do Jarvis.

Copia evals/fixtures/ para uma pasta temporária, indexa com o RagIndex REAL
(embeddings pelo Ollama) e mede recall@1, recall@5 e MRR sobre as perguntas de
evals/data/rag_retrieval.jsonl.

    .venv/Scripts/python.exe evals/eval_rag.py --embed-model nomic-embed-text
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import common  # noqa: E402
from jarvis.metrics_utils import latency_summary, mean, ratio  # noqa: E402
from jarvis.rag import RagIndex  # noqa: E402

DATA_PATH = common.DATA_DIR / "rag_retrieval.jsonl"
TOP_K = 5


def is_hit(result: Dict[str, Any], question: Dict[str, Any]) -> bool:
    """Acerto = arquivo esperado E trecho contém o snippet (espaços/caixa/acentos normalizados)."""
    if Path(str(result["path"])).name != question["expected_doc"]:
        return False
    return common.normalize(question["expected_snippet"]) in common.normalize(result["text"])


def first_hit_rank(results: List[Dict[str, Any]], question: Dict[str, Any]) -> Optional[int]:
    for rank, result in enumerate(results, start=1):
        if is_hit(result, question):
            return rank
    return None


def compute_metrics(records: List[Dict[str, Any]], k: int = TOP_K) -> Dict[str, Any]:
    total = len(records)
    ranks = [r["rank"] for r in records]
    hits_at_1 = sum(1 for rank in ranks if rank == 1)
    hits_at_k = sum(1 for rank in ranks if rank is not None and rank <= k)
    doc_hits_at_1 = sum(1 for r in records if r["top_docs"][:1] == [r["expected_doc"]])
    return {
        "n": total,
        "recall_at_1": ratio(hits_at_1, total),
        "hits_at_1": hits_at_1,
        "recall_at_5": ratio(hits_at_k, total),
        "hits_at_5": hits_at_k,
        "mrr": mean(1.0 / rank if rank else 0.0 for rank in ranks),
        "doc_recall_at_1": ratio(doc_hits_at_1, total),
        "search_latency_s": latency_summary([r["latency_s"] for r in records]),
    }


def run_eval(
    embed_model: str = common.DEFAULT_EMBED_MODEL,
    questions: Optional[List[Dict[str, Any]]] = None,
    ollama_url: str = common.DEFAULT_OLLAMA_URL,
    k: int = TOP_K,
    verbose: bool = False,
) -> Dict[str, Any]:
    questions = questions if questions is not None else common.load_jsonl(DATA_PATH)
    available = {p.name for p in common.FIXTURES_DIR.iterdir() if p.is_file()}
    for question in questions:
        if question["expected_doc"] not in available:
            raise ValueError("{}: documento inexistente {}".format(question["id"], question["expected_doc"]))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        corpus = root / "corpus"
        shutil.copytree(str(common.FIXTURES_DIR), str(corpus))
        settings = common.make_eval_settings(root, ollama_url=ollama_url, embed_model=embed_model)
        rag = RagIndex(settings)
        try:
            started = time.perf_counter()
            report = rag.index(paths=[corpus], force=True)
            index_s = time.perf_counter() - started
            stats = rag.stats()
            if "Erros: 0" not in report or stats["chunks"] == 0:
                raise RuntimeError("indexação falhou:\n" + report)

            records: List[Dict[str, Any]] = []
            for question in questions:
                started = time.perf_counter()
                results = rag.search(question["question"], k=k)
                latency = time.perf_counter() - started
                if not results:
                    raise RuntimeError("busca vazia para {} (Ollama fora?)".format(question["id"]))
                rank = first_hit_rank(results, question)
                record = {
                    "id": question["id"],
                    "question": question["question"],
                    "expected_doc": question["expected_doc"],
                    "rank": rank,
                    "latency_s": latency,
                    "top_docs": [Path(str(r["path"])).name for r in results],
                    "top": [
                        {
                            "doc": Path(str(r["path"])).name,
                            "chunk": r["chunk"],
                            "score": round(float(r["score"]), 4),
                            "hit": is_hit(r, question),
                        }
                        for r in results
                    ],
                }
                records.append(record)
                if verbose:
                    print("{:<8} rank={} top1={}".format(question["id"], rank, record["top_docs"][0]))
        finally:
            rag.close()

    return {
        "kind": "rag_retrieval",
        "embed_model": embed_model,
        "k": k,
        "dataset": DATA_PATH.relative_to(common.BASE_DIR).as_posix(),
        "dataset_sha256": common.file_sha256(DATA_PATH),
        "fixtures_sha256": common.dir_sha256(common.FIXTURES_DIR),
        "corpus": {"documents": stats["documents"], "chunks": stats["chunks"], "index_s": index_s},
        "meta": common.run_metadata(ollama_url),
        "metrics": compute_metrics(records, k),
        "records": records,
    }


def print_summary(result: Dict[str, Any]) -> None:
    m = result["metrics"]
    corpus = result["corpus"]
    print("\n== Recuperação (RAG): {} ==".format(result["embed_model"]))
    print("corpus: {} documentos, {} trechos (indexado em {:.1f}s)".format(
        corpus["documents"], corpus["chunks"], corpus["index_s"]))
    print("perguntas: {}".format(m["n"]))
    print("recall@1: {}".format(common.fmt_pct(m["recall_at_1"], m["hits_at_1"], m["n"])))
    print("recall@5: {}".format(common.fmt_pct(m["recall_at_5"], m["hits_at_5"], m["n"])))
    print("MRR:      {}".format(common.fmt_num(m["mrr"], 3)))
    print("latência da busca p50/p95: {} / {} s".format(
        common.fmt_num(m["search_latency_s"]["p50"], 3), common.fmt_num(m["search_latency_s"]["p95"], 3)))


def result_filename(result: Dict[str, Any]) -> str:
    stamp = result["meta"]["timestamp"][:19].replace(":", "").replace("-", "")
    return "{}_rag_{}.json".format(stamp, common.slug(result["embed_model"]))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--embed-model", default=common.DEFAULT_EMBED_MODEL)
    parser.add_argument("--ollama-url", default=common.DEFAULT_OLLAMA_URL)
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not common.model_available(args.ollama_url, args.embed_model):
        print("Modelo de embeddings '{}' não encontrado no Ollama.".format(args.embed_model))
        return 2
    result = run_eval(args.embed_model, None, args.ollama_url, args.k, args.verbose)
    print_summary(result)
    saved: List[Path] = []
    if not args.no_save:
        path = common.save_json(result, result_filename(result))
        saved.append(path)
        print("\nresultado salvo em {}".format(path.relative_to(common.BASE_DIR)))
    if not args.no_mlflow:
        log_to_mlflow(result, saved)
    return 0


def log_to_mlflow(result: Dict[str, Any], saved: List[Path], nested: bool = False) -> Optional[str]:
    """Um run por modelo de embeddings: params, métricas, JSON e tabela Markdown."""
    from evals import report, tracking

    with tempfile.TemporaryDirectory() as tmp:
        files = [p.name for p in saved]
        json_path = saved[0] if saved else common.save_json(result, result_filename(result), Path(tmp))
        md_path = Path(tmp) / "tabela.md"
        md_path.write_text(
            report.render_markdown(report.single_summary(result, "eval_rag.py", files)),
            encoding="utf-8",
        )
        run_id = tracking.log_run(
            run_name="rag {}".format(result["embed_model"]),
            params=tracking.rag_params(result),
            metrics=tracking.rag_metrics(result),
            artifacts=[json_path, md_path],
            tags={"eval": "rag_retrieval"},
            nested=nested,
        )
    if run_id:
        print("MLflow: run {} registrado em {}".format(run_id[:8], tracking.tracking_uri()))
    return run_id


if __name__ == "__main__":
    sys.exit(main())
