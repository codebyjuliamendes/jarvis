#!/usr/bin/env python3
"""Compara modelos do Ollama nas avaliações do Jarvis.

Roda a seleção de ferramentas para cada modelo (um por vez, descarregando da
GPU entre um e outro) e a recuperação do RAG para cada modelo de embeddings.
Grava um JSON por execução em evals/results/ e a tabela em
evals/results/RESULTS.md.

    .venv/Scripts/python.exe evals/compare_models.py qwen2.5:1.5b qwen2.5:3b llama3.2:3b
    .venv/Scripts/python.exe evals/compare_models.py qwen2.5:3b --modes native,json --limit 10
    .venv/Scripts/python.exe evals/compare_models.py qwen2.5:1.5b qwen2.5:3b llama3.2:3b \
        --repeats 3 --skip-rag --markdown evals/results/REPEATS.md
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import common, eval_rag, eval_tools, stats, tracking  # noqa: E402
from evals.report import render_markdown, render_repeats_markdown  # noqa: E402

RESULTS_MD = common.RESULTS_DIR / "RESULTS.md"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="+", help="modelos de chat do Ollama")
    parser.add_argument("--modes", default="native", help="native, json ou native,json")
    parser.add_argument("--embed-models", default=common.DEFAULT_EMBED_MODEL)
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ollama-url", default=common.DEFAULT_OLLAMA_URL)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--markdown", default=str(RESULTS_MD))
    parser.add_argument("--repeats", type=int, default=1,
                        help="rodadas idênticas por modelo; com N > 1 reporta média, desvio, "
                             "IC 95%% e McNemar entre os modelos")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        if mode not in eval_tools.MODES:
            parser.error("modo inválido: " + mode)
    if args.repeats < 1:
        parser.error("--repeats deve ser >= 1")
    cases = common.load_jsonl(eval_tools.DATA_PATH)
    if args.limit:
        cases = cases[: args.limit]
    if args.repeats > 1:
        return run_repeats(args, modes, cases)

    tool_results: List[Dict[str, Any]] = []
    rag_results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    files: List[str] = []

    parent_params = {
        "eval": "comparison",
        "models": ",".join(args.models),
        "modes": ",".join(modes),
        "embed_models": "" if args.skip_rag else args.embed_models,
        "n_examples_tools": len(cases),
        "dataset_tools_sha256": common.file_sha256(eval_tools.DATA_PATH),
        "dataset_rag_sha256": common.file_sha256(eval_rag.DATA_PATH),
        "git_commit": common.git_commit(),
    }
    run_ctx = (
        contextlib.nullcontext(None)
        if args.no_mlflow
        else tracking.parent_run("compare " + " ".join(args.models), parent_params)
    )
    with run_ctx as parent:
        nested = parent is not None
        for model in args.models:
            if not common.model_available(args.ollama_url, model):
                failures.append({"target": model, "error": "modelo não encontrado no Ollama"})
                print("!! {}: modelo não encontrado no Ollama".format(model))
                continue
            for mode in modes:
                print("\n>> {} ({})".format(model, mode), flush=True)
                try:
                    result = asyncio.run(
                        eval_tools.run_eval(model, mode, cases, args.ollama_url, verbose=args.verbose)
                    )
                except Exception:  # noqa: BLE001 - registra e segue para o próximo modelo
                    error = traceback.format_exc()
                    failures.append({"target": "{} ({})".format(model, mode), "error": error})
                    print(error)
                    continue
                eval_tools.print_summary(result)
                path = common.save_json(result, eval_tools.result_filename(result))
                files.append(path.name)
                tool_results.append(result)
                if not args.no_mlflow:
                    eval_tools.log_to_mlflow(result, [path], nested=nested)
            common.unload_model(args.ollama_url, model)

        if not args.skip_rag:
            for embed_model in [m.strip() for m in args.embed_models.split(",") if m.strip()]:
                print("\n>> RAG ({})".format(embed_model), flush=True)
                try:
                    result = eval_rag.run_eval(embed_model, None, args.ollama_url)
                except Exception:  # noqa: BLE001
                    error = traceback.format_exc()
                    failures.append({"target": "rag " + embed_model, "error": error})
                    print(error)
                    continue
                eval_rag.print_summary(result)
                path = common.save_json(result, eval_rag.result_filename(result))
                files.append(path.name)
                rag_results.append(result)
                if not args.no_mlflow:
                    eval_rag.log_to_mlflow(result, [path], nested=nested)
                common.unload_model(args.ollama_url, embed_model)

        summary = {
            "kind": "comparison",
            "meta": common.run_metadata(args.ollama_url),
            "models": args.models,
            "modes": modes,
            "limit": args.limit,
            "tool_results": tool_results,
            "rag_results": rag_results,
            "failures": failures,
            "files": files,
        }
        markdown = render_markdown(summary)
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(md_path), "w", encoding="utf-8", newline="\n") as handle:
            handle.write(markdown)
        stamp = summary["meta"]["timestamp"][:19].replace(":", "").replace("-", "")
        compact = dict(summary)
        compact["tool_results"] = [{k: v for k, v in r.items() if k != "records"} for r in tool_results]
        compact["rag_results"] = [{k: v for k, v in r.items() if k != "records"} for r in rag_results]
        summary_path = common.save_json(compact, "{}_compare.json".format(stamp))
        if parent is not None:
            tracking.log_artifacts_to_active(
                [md_path, summary_path], metrics={"n_failures": float(len(failures))}
            )

    print("\n" + markdown)
    print("tabela: {}\nresumo: {}".format(md_path, summary_path.relative_to(common.BASE_DIR)))
    if parent is not None:
        print("MLflow: {} (abra com: mlflow ui --backend-store-uri {})".format(
            tracking.tracking_uri(), tracking.tracking_uri()))
    return 1 if failures and not tool_results else 0


def pairwise_comparisons(repeated_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """McNemar exato e diferença pareada para cada par de modelos no mesmo modo."""
    comparisons: List[Dict[str, Any]] = []
    for mode in sorted({r["mode"] for r in repeated_results}):
        same_mode = [r for r in repeated_results if r["mode"] == mode]
        for a, b in itertools.combinations(same_mode, 2):
            comp = stats.paired_comparison(eval_tools.runs_outcomes(a), eval_tools.runs_outcomes(b))
            comp.update({"model_a": a["model"], "model_b": b["model"], "mode": mode})
            comparisons.append(comp)
    alpha = 0.05 / len(comparisons) if comparisons else 0.05
    for comp in comparisons:
        comp["alpha_bonferroni"] = alpha
    return comparisons


def run_repeats(args: argparse.Namespace, modes: List[str], cases: List[Dict[str, Any]]) -> int:
    """Seleção de ferramentas com N rodadas por (modelo, modo); sem RAG (é determinístico)."""
    repeated_results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    files: List[str] = []
    parent_params = {
        "eval": "comparison_repeated",
        "models": ",".join(args.models),
        "modes": ",".join(modes),
        "repeats": args.repeats,
        "n_examples_tools": len(cases),
        "dataset_tools_sha256": common.file_sha256(eval_tools.DATA_PATH),
        "git_commit": common.git_commit(),
    }
    run_ctx = (
        contextlib.nullcontext(None)
        if args.no_mlflow
        else tracking.parent_run(
            "compare x{} {}".format(args.repeats, " ".join(args.models)), parent_params)
    )
    with run_ctx as parent:
        nested = parent is not None
        for model in args.models:
            if not common.model_available(args.ollama_url, model):
                failures.append({"target": model, "error": "modelo não encontrado no Ollama"})
                print("!! {}: modelo não encontrado no Ollama".format(model))
                continue
            for mode in modes:
                print("\n>> {} ({}) x{}".format(model, mode, args.repeats), flush=True)
                try:
                    repeated = asyncio.run(eval_tools.run_repeated(
                        model, mode, cases, args.repeats, args.ollama_url, verbose=args.verbose))
                except Exception:  # noqa: BLE001 - registra e segue para o próximo modelo
                    error = traceback.format_exc()
                    failures.append({"target": "{} ({})".format(model, mode), "error": error})
                    print(error)
                    continue
                eval_tools.print_repeated_summary(repeated)
                path = common.save_json(repeated, eval_tools.repeated_filename(repeated))
                files.append(path.name)
                repeated_results.append(repeated)
                if not args.no_mlflow:
                    eval_tools.log_repeated_to_mlflow(repeated, [path], nested=nested)
            common.unload_model(args.ollama_url, model)

        comparisons = pairwise_comparisons(repeated_results)
        summary = {
            "kind": "comparison_repeated",
            "meta": common.run_metadata(args.ollama_url),
            "models": args.models,
            "modes": modes,
            "repeats": args.repeats,
            "limit": args.limit,
            "repeated_results": repeated_results,
            "comparisons": comparisons,
            "failures": failures,
            "files": files,
        }
        markdown = render_repeats_markdown(summary)
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(md_path), "w", encoding="utf-8", newline="\n") as handle:
            handle.write(markdown)
        stamp = summary["meta"]["timestamp"][:19].replace(":", "").replace("-", "")
        compact = dict(summary)
        compact["repeated_results"] = [
            {k: v for k, v in r.items() if k != "runs"} for r in repeated_results
        ]
        summary_path = common.save_json(compact, "{}_compare_x{}.json".format(stamp, args.repeats))
        if parent is not None:
            metrics = {"n_failures": float(len(failures))}
            for comp in comparisons:
                key = "mcnemar_p_{}_vs_{}_{}".format(
                    common.slug(comp["model_a"]), common.slug(comp["model_b"]), comp["mode"])
                metrics[key] = comp["mcnemar_p"]
            tracking.log_artifacts_to_active([md_path, summary_path], metrics=metrics)

    print("\n" + markdown)
    print("tabela: {}\nresumo: {}".format(md_path, summary_path.relative_to(common.BASE_DIR)))
    return 1 if failures and not repeated_results else 0


if __name__ == "__main__":
    sys.exit(main())
