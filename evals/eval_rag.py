#!/usr/bin/env python3
"""Avaliação de recuperação do RAG do Jarvis.

Copia evals/fixtures/ para uma pasta temporária, indexa com o RagIndex REAL
(embeddings pelo Ollama) e mede recall@1, recall@5 e MRR sobre as perguntas de
evals/data/rag_retrieval.jsonl, no modo de busca escolhido.

    .venv/Scripts/python.exe evals/eval_rag.py --embed-model nomic-embed-text
    .venv/Scripts/python.exe evals/eval_rag.py --mode hybrid --prefixes on
    .venv/Scripts/python.exe evals/eval_rag.py --grid      # todas as combinações
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import common  # noqa: E402
from jarvis.config import Settings  # noqa: E402
from jarvis.metrics_utils import latency_summary, mean, ratio  # noqa: E402
from jarvis.rag import RAG_MODES, VECTOR_BACKENDS, RagIndex, use_prefixes  # noqa: E402

DATA_PATH = common.DATA_DIR / "rag_retrieval.jsonl"
TOP_K = 5
GRID_MD = common.RESULTS_DIR / "RAG.md"


def _default(name: str) -> Any:
    return Settings.__dataclass_fields__[name].default  # type: ignore[attr-defined]


# Padrões do bot (jarvis/config.py): sem flags, o eval mede o que o Jarvis usa.
DEFAULT_MODE = _default("rag_mode")
DEFAULT_RRF_K = _default("rag_rrf_k")
DEFAULT_PREFIXES = use_prefixes(_default("rag_prefixes"), common.DEFAULT_EMBED_MODEL)

# Grade principal: cada linha muda UMA coisa em relação a uma anterior.
GRID = [
    {"mode": "vector", "vector_backend": "numpy", "prefixes": False},  # linha de base
    {"mode": "vector", "vector_backend": "sqlite-vec", "prefixes": False},
    {"mode": "vector", "vector_backend": "sqlite-vec", "prefixes": True},
    {"mode": "bm25", "vector_backend": "sqlite-vec", "prefixes": False},
    {"mode": "hybrid", "vector_backend": "sqlite-vec", "prefixes": False},
    {"mode": "hybrid", "vector_backend": "sqlite-vec", "prefixes": True},
]
# Sensibilidade do k do RRF: só informativa, não usada para escolher o padrão.
RRF_K_SENSITIVITY = [10, 30, 100]


def split_of(question_id: str) -> str:
    """Divisão fixa (decidida antes de medir): ids ímpares = dev, pares = teste."""
    match = re.search(r"(\d+)$", question_id)
    return "dev" if match and int(match.group(1)) % 2 == 1 else "test"


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


def compute_split_metrics(records: List[Dict[str, Any]], k: int = TOP_K) -> Dict[str, Any]:
    return {
        name: compute_metrics([r for r in records if split_of(r["id"]) == name], k)
        for name in ("dev", "test")
    }


def run_eval(
    embed_model: str = common.DEFAULT_EMBED_MODEL,
    questions: Optional[List[Dict[str, Any]]] = None,
    ollama_url: str = common.DEFAULT_OLLAMA_URL,
    k: int = TOP_K,
    verbose: bool = False,
    mode: str = DEFAULT_MODE,
    prefixes: bool = DEFAULT_PREFIXES,
    rrf_k: int = DEFAULT_RRF_K,
    vector_backend: str = "auto",
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
        settings = replace(
            common.make_eval_settings(root, ollama_url=ollama_url, embed_model=embed_model),
            rag_mode=mode,
            rag_prefixes="true" if prefixes else "false",
            rag_rrf_k=rrf_k,
            rag_vector_backend=vector_backend,
        )
        rag = RagIndex(settings)
        config = {
            "mode": rag.mode,
            "prefixes": rag.prefixes,
            "rrf_k": rag.rrf_k,
            "vector_backend": rag.vector_backend,
            "vector_backend_requested": vector_backend,
            "fts5": rag.fts_enabled,
        }
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
                if not results and rag.mode != "bm25":
                    # BM25 pode não achar nada (nenhuma palavra em comum): conta como erro.
                    raise RuntimeError("busca vazia para {} (Ollama fora?)".format(question["id"]))
                if any(r.get("mode") != rag.mode for r in results):
                    raise RuntimeError("{}: a busca caiu para outro modo ({})".format(
                        question["id"], results[0].get("mode")))
                rank = first_hit_rank(results, question)
                record = {
                    "id": question["id"],
                    "split": split_of(question["id"]),
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
                    print("{:<8} rank={} top1={}".format(
                        question["id"], rank, (record["top_docs"] or ["-"])[0]))
        finally:
            rag.close()

    return {
        "kind": "rag_retrieval",
        "embed_model": embed_model,
        "k": k,
        "config": config,
        "dataset": DATA_PATH.relative_to(common.BASE_DIR).as_posix(),
        "dataset_sha256": common.file_sha256(DATA_PATH),
        "fixtures_sha256": common.dir_sha256(common.FIXTURES_DIR),
        "corpus": {"documents": stats["documents"], "chunks": stats["chunks"], "index_s": index_s},
        "meta": common.run_metadata(ollama_url),
        "metrics": compute_metrics(records, k),
        "metrics_by_split": compute_split_metrics(records, k),
        "records": records,
    }


def config_label(result: Dict[str, Any]) -> str:
    """Rótulo curto da configuração: 'hybrid + prefixos (RRF k=60)'."""
    config = result.get("config") or {}
    mode = config.get("mode", "vector")
    parts = [mode]
    if mode == "vector":
        parts.append(config.get("vector_backend", "numpy"))
    if mode != "bm25":
        parts.append("+ prefixos" if config.get("prefixes") else "sem prefixos")
    if mode == "hybrid":
        parts.append("(RRF k={})".format(config.get("rrf_k")))
    return " ".join(parts)


def print_summary(result: Dict[str, Any]) -> None:
    m = result["metrics"]
    corpus = result["corpus"]
    print("\n== Recuperação (RAG): {} · {} ==".format(result["embed_model"], config_label(result)))
    print("corpus: {} documentos, {} trechos (indexado em {:.1f}s)".format(
        corpus["documents"], corpus["chunks"], corpus["index_s"]))
    print("perguntas: {}".format(m["n"]))
    print("recall@1: {}".format(common.fmt_pct(m["recall_at_1"], m["hits_at_1"], m["n"])))
    print("recall@5: {}".format(common.fmt_pct(m["recall_at_5"], m["hits_at_5"], m["n"])))
    print("MRR:      {}".format(common.fmt_num(m["mrr"], 3)))
    splits = result.get("metrics_by_split") or {}
    if splits:
        print("MRR dev/teste: {} / {}".format(
            common.fmt_num(splits["dev"]["mrr"], 3), common.fmt_num(splits["test"]["mrr"], 3)))
    print("latência da busca média/p95: {} / {} s".format(
        common.fmt_num(m["search_latency_s"]["mean"], 3), common.fmt_num(m["search_latency_s"]["p95"], 3)))


def result_filename(result: Dict[str, Any]) -> str:
    stamp = result["meta"]["timestamp"][:19].replace(":", "").replace("-", "")
    name = "{}_rag_{}".format(stamp, common.slug(result["embed_model"]))
    config = result.get("config")
    if config:
        name += "_" + config["mode"]
        if config["mode"] == "vector":
            name += "-" + config["vector_backend"]
        if config["mode"] != "bm25" and config.get("prefixes"):
            name += "-prefixos"
        if config["mode"] == "hybrid":
            name += "-k{}".format(config["rrf_k"])
    return name + ".json"


# ----------------------------------------------------------------- grade

GRID_HEADER = ["Configuração", "recall@1", "recall@5", "MRR", "MRR dev (18)", "MRR teste (18)",
               "Latência média (ms)", "Latência p95 (ms)"]


def _ms(value: Optional[float]) -> str:
    return common.fmt_num(value * 1000 if value is not None else None, 1)


def grid_row(result: Dict[str, Any]) -> List[str]:
    m = result["metrics"]
    splits = result.get("metrics_by_split") or {}
    return [
        config_label(result),
        common.fmt_pct(m["recall_at_1"], m["hits_at_1"], m["n"]),
        common.fmt_pct(m["recall_at_5"], m["hits_at_5"], m["n"]),
        common.fmt_num(m["mrr"], 3),
        common.fmt_num(splits.get("dev", {}).get("mrr"), 3),
        common.fmt_num(splits.get("test", {}).get("mrr"), 3),
        _ms(m["search_latency_s"]["mean"]),
        _ms(m["search_latency_s"]["p95"]),
    ]


def render_grid(main: List[Dict[str, Any]], sensitivity: List[Dict[str, Any]], files: List[str]) -> str:
    from evals import report

    first = main[0]
    meta = first["meta"]
    parts = [
        "# RAG: comparação de configurações",
        "",
        "Gerado por `evals/eval_rag.py --grid` em {} (commit `{}`, Ollama {}).".format(
            meta["timestamp"], meta["git_commit"], meta["ollama"].get("version")),
        "Máquina: {} · GPU: {}. Embeddings `{}`; corpus {} documentos / {} trechos; "
        "dataset `{}` ({} perguntas, sha256 `{}`).".format(
            meta["platform"], meta.get("gpu") or "—", first["embed_model"],
            first["corpus"]["documents"], first["corpus"]["chunks"], first["dataset"],
            first["metrics"]["n"], first["dataset_sha256"]),
        "",
        "Divisão fixa: ids ímpares = dev (18), pares = teste (18). O padrão do bot é escolhido "
        "pelo MRR no dev; o teste só confirma. Latência = `RagIndex.search` inteiro (inclui o "
        "embedding da pergunta no Ollama, exceto no bm25).",
        "",
        report.md_table(GRID_HEADER, [grid_row(r) for r in main]),
    ]
    if sensitivity:
        parts += [
            "",
            "## Sensibilidade do k do RRF (informativo, não usado para escolher)",
            "",
            report.md_table(GRID_HEADER, [grid_row(r) for r in sensitivity]),
        ]
    parts += ["", "Arquivos brutos (com cada pergunta): " + ", ".join(
        "`{}`".format(name) for name in files), ""]
    return "\n".join(parts)


def run_grid(args: argparse.Namespace) -> int:
    from evals import tracking

    configs = [dict(c, rrf_k=args.rrf_k) for c in GRID]
    sensitivity_configs: List[Dict[str, Any]] = []
    if not args.no_sensitivity:
        sensitivity_configs = [
            {"mode": "hybrid", "vector_backend": "sqlite-vec", "prefixes": prefixes, "rrf_k": k}
            for prefixes in (False, True)
            for k in RRF_K_SENSITIVITY
        ]
    main_results: List[Dict[str, Any]] = []
    sensitivity: List[Dict[str, Any]] = []
    files: List[str] = []
    parent = nullcontext() if args.no_mlflow else tracking.parent_run(
        "rag grid {}".format(args.embed_model),
        {"eval": "rag_grid", "embed_model": args.embed_model,
         "n_configs": len(configs) + len(sensitivity_configs)},
    )
    with parent as run:
        for target, bucket in ((configs, main_results), (sensitivity_configs, sensitivity)):
            for config in target:
                result = run_eval(args.embed_model, None, args.ollama_url, args.k, args.verbose, **config)
                print_summary(result)
                saved: List[Path] = []
                if not args.no_save:
                    path = common.save_json(result, result_filename(result))
                    saved.append(path)
                    files.append(path.name)
                if not args.no_mlflow:
                    log_to_mlflow(result, saved, nested=run is not None)
                bucket.append(result)
        markdown = render_grid(main_results, sensitivity, files)
        if not args.no_save:
            GRID_MD.write_text(markdown, encoding="utf-8")
            print("\ntabela salva em {}".format(GRID_MD.relative_to(common.BASE_DIR)))
            if run is not None:
                tracking.log_artifacts_to_active([GRID_MD])
    print("\n" + markdown)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--embed-model", default=common.DEFAULT_EMBED_MODEL)
    parser.add_argument("--ollama-url", default=common.DEFAULT_OLLAMA_URL)
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--mode", choices=RAG_MODES, default=DEFAULT_MODE)
    parser.add_argument("--prefixes", choices=["on", "off"],
                        default="on" if DEFAULT_PREFIXES else "off",
                        help="prefixos search_document:/search_query: do nomic-embed-text")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--vector-backend", choices=VECTOR_BACKENDS, default="auto")
    parser.add_argument("--grid", action="store_true",
                        help="roda todas as combinações e gera evals/results/RAG.md")
    parser.add_argument("--no-sensitivity", action="store_true",
                        help="com --grid, pula a sensibilidade do k do RRF")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not common.model_available(args.ollama_url, args.embed_model):
        print("Modelo de embeddings '{}' não encontrado no Ollama.".format(args.embed_model))
        return 2
    if args.grid:
        return run_grid(args)
    result = run_eval(
        args.embed_model, None, args.ollama_url, args.k, args.verbose,
        mode=args.mode, prefixes=args.prefixes == "on", rrf_k=args.rrf_k,
        vector_backend=args.vector_backend,
    )
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
    """Um run por configuração: params, métricas, JSON e tabela Markdown."""
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
            run_name="rag {} {}".format(result["embed_model"], config_label(result)),
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
