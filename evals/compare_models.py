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
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import common, eval_rag, eval_tools  # noqa: E402

RESULTS_MD = common.RESULTS_DIR / "RESULTS.md"


def tool_row(result: Dict[str, Any]) -> List[str]:
    m = result["metrics"]
    lat = m["latency_s"]
    return [
        "`{}`".format(result["model"]),
        result["effective_mode"],
        common.fmt_pct(m["tool_accuracy"], m["tool_hits"], m["n"]),
        common.fmt_pct(m["args_accuracy"], m["args_hits"], m["args_n"]),
        common.fmt_pct(m["invalid_json_rate"], m["invalid_json"], m["n"]),
        common.fmt_num(lat["p50"]),
        common.fmt_num(lat["p95"]),
        common.fmt_num(m["tokens_per_s"], 1),
    ]


def category_row(result: Dict[str, Any]) -> List[str]:
    cats = result["metrics"]["by_category"]

    def cell(name: str) -> str:
        info = cats.get(name)
        if not info:
            return "—"
        return common.fmt_pct(info["accuracy"], info["hits"], info["n"])

    m = result["metrics"]
    return [
        "`{}`".format(result["model"]),
        result["effective_mode"],
        cell("direto"),
        cell("ambiguo"),
        cell("sem_ferramenta"),
        common.fmt_pct(m["args_accuracy_given_tool"], m["args_hits_given_tool"], m["args_n_given_tool"]),
        str(m["leaked_calls"]),
        str(m["unknown_tool_calls"]),
    ]


def rag_row(result: Dict[str, Any]) -> List[str]:
    m = result["metrics"]
    return [
        "`{}`".format(result["embed_model"]),
        "{} docs / {} trechos".format(result["corpus"]["documents"], result["corpus"]["chunks"]),
        common.fmt_pct(m["recall_at_1"], m["hits_at_1"], m["n"]),
        common.fmt_pct(m["recall_at_5"], m["hits_at_5"], m["n"]),
        common.fmt_num(m["mrr"], 3),
        common.fmt_num(m["search_latency_s"]["p50"], 3),
    ]


def md_table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(summary: Dict[str, Any]) -> str:
    tools = summary["tool_results"]
    rags = summary["rag_results"]
    parts = [
        "# Resultados das avaliações",
        "",
        "Gerado por `evals/compare_models.py` em {} (commit `{}`, Ollama {}).".format(
            summary["meta"]["timestamp"], summary["meta"]["git_commit"],
            summary["meta"]["ollama"].get("version"),
        ),
        "Máquina: {} · GPU: {}. Um modelo por vez, seed {}, `num_ctx` {}.".format(
            summary["meta"]["platform"], summary["meta"].get("gpu") or "—", common.DEFAULT_SEED,
            tools[0]["num_ctx"] if tools else "—",
        ),
        "",
        "## Seleção de ferramentas",
        "",
        "Dataset `{}` ({} casos, sha256 `{}`), {} ferramentas reais.".format(
            tools[0]["dataset"], tools[0]["metrics"]["n"], tools[0]["dataset_sha256"],
            tools[0]["n_tools"],
        ) if tools else "Nenhum resultado.",
        "",
    ]
    if tools:
        parts.append(md_table(
            ["Modelo", "Modo", "Acurácia ferramenta", "Acurácia args", "JSON inválido",
             "Latência p50 (s)", "Latência p95 (s)", "tokens/s"],
            [tool_row(r) for r in tools],
        ))
        parts += ["", "### Por categoria", ""]
        parts.append(md_table(
            ["Modelo", "Modo", "Direto", "Ambíguo", "Sem ferramenta",
             "Args (dada a ferramenta certa)", "Chamadas vazadas no texto", "Ferramenta inexistente"],
            [category_row(r) for r in tools],
        ))
    parts += ["", "## Recuperação (RAG)", ""]
    if rags:
        parts.append("Dataset `{}` ({} perguntas, sha256 `{}`). O resultado depende só do modelo de "
                     "embeddings, não do modelo de chat.".format(
                         rags[0]["dataset"], rags[0]["metrics"]["n"], rags[0]["dataset_sha256"]))
        parts.append("")
        parts.append(md_table(
            ["Embeddings", "Corpus", "recall@1", "recall@5", "MRR", "Latência da busca p50 (s)"],
            [rag_row(r) for r in rags],
        ))
    else:
        parts.append("Não executado.")
    if summary["failures"]:
        parts += ["", "## Falhas", ""]
        parts.extend("- `{}`: {}".format(f["target"], f["error"].splitlines()[-1]) for f in summary["failures"])
    parts += ["", "Arquivos brutos (com cada caso): " + ", ".join(
        "`{}`".format(name) for name in summary["files"]), ""]
    return "\n".join(parts)


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
    print("\n" + markdown)
    print("tabela: {}\nresumo: {}".format(md_path, summary_path.relative_to(common.BASE_DIR)))
    return 1 if failures and not tool_results else 0


if __name__ == "__main__":
    sys.exit(main())
