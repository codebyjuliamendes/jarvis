#!/usr/bin/env python3
"""Compara modelos do Ollama nas avaliações do Jarvis.

Roda a seleção de ferramentas para cada modelo (um por vez, descarregando da
GPU entre um e outro) e a recuperação do RAG para cada modelo de embeddings.
Grava um JSON por execução em evals/results/ e a tabela em
evals/results/RESULTS.md.

    .venv/Scripts/python.exe evals/compare_models.py qwen2.5:1.5b qwen2.5:3b llama3.2:3b
    .venv/Scripts/python.exe evals/compare_models.py qwen2.5:3b --modes native,json --limit 10
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import common, eval_rag, eval_tools, tracking  # noqa: E402
from evals.report import render_markdown  # noqa: E402

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
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        if mode not in eval_tools.MODES:
            parser.error("modo inválido: " + mode)
    cases = common.load_jsonl(eval_tools.DATA_PATH)
    if args.limit:
        cases = cases[: args.limit]

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


if __name__ == "__main__":
    sys.exit(main())
