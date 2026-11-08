"""Tabelas Markdown dos resultados (usadas por todas as avaliações)."""
from __future__ import annotations

from typing import Any, Dict, List

from evals import common


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
        "Gerado por `evals/{}` em {} (commit `{}`, Ollama {}).".format(
            summary.get("script", "compare_models.py"),
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


def single_summary(result: Dict[str, Any], script: str, files: List[str]) -> Dict[str, Any]:
    """Resumo no mesmo formato do compare_models, para uma avaliação isolada."""
    is_rag = result.get("kind") == "rag_retrieval"
    return {
        "script": script,
        "meta": result["meta"],
        "tool_results": [] if is_rag else [result],
        "rag_results": [result] if is_rag else [],
        "failures": [],
        "files": files,
    }
