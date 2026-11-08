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


def _rag_config_suffix(result: Dict[str, Any]) -> str:
    if not result.get("config"):
        return ""
    from evals.eval_rag import config_label

    return " · " + config_label(result)


def rag_row(result: Dict[str, Any]) -> List[str]:
    m = result["metrics"]
    return [
        "`{}`{}".format(result["embed_model"], _rag_config_suffix(result)),
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
        parts.extend(
            "- `{}`: {}".format(f["target"], f["error"].splitlines()[-1]) for f in summary["failures"]
        )
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


# ------------------------------------------------------------ repetições


def _pct1(value: float) -> str:
    return "{:.1f}%".format(value * 100).replace(".", ",")


def _ci(interval: List[float]) -> str:
    return "[{}; {}]".format(
        "{:.1f}".format(interval[0] * 100).replace(".", ","), _pct1(interval[1]))


def _pp(value: Any) -> str:
    if value is None:
        return "—"
    return "{:.1f}".format(value * 100).replace(".", ",") + " p.p."


def _p(value: float) -> str:
    return ("{:.3f}".format(value) if value >= 0.001 else "< 0,001").replace(".", ",")


def repeated_row(repeated: Dict[str, Any]) -> List[str]:
    tool = repeated["stats"]["tool"]
    args = repeated["stats"].get("args")
    return [
        "`{}`".format(repeated["model"]),
        repeated["effective_mode"],
        " / ".join(str(h) for h in tool["hits_per_run"]) + " (de {})".format(tool["n_cases"]),
        _pct1(tool["mean"]),
        _pp(tool["stdev"]),
        _ci(tool["ci95_bootstrap"]),
        _ci(tool["ci95_wilson"]),
        str(tool["unstable_cases"]),
        "{} ± {}".format(_pct1(args["mean"]), _pp(args["stdev"])) if args else "—",
    ]


def comparison_row(comp: Dict[str, Any]) -> List[str]:
    alpha = comp.get("alpha_bonferroni", 0.05)
    significant = comp["mcnemar_p"] < alpha
    return [
        "`{}` × `{}`".format(comp["model_a"], comp["model_b"]),
        comp["mode"],
        "{} / {}".format(comp["only_a"], comp["only_b"]),
        _p(comp["mcnemar_p"]),
        ", ".join(_p(p) for p in comp["mcnemar_p_per_run"]),
        "{:+.1f}".format(comp["diff_mean"] * 100).replace(".", ",") + " p.p.",
        "[{:+.1f}; {:+.1f}]".format(
            comp["diff_ci95_bootstrap"][0] * 100, comp["diff_ci95_bootstrap"][1] * 100
        ).replace(".", ",") + " p.p.",
        "sim" if significant else "não",
    ]


def render_repeats_markdown(summary: Dict[str, Any]) -> str:
    reps = summary["repeated_results"]
    comps = summary.get("comparisons") or []
    first = reps[0] if reps else None
    parts = [
        "# Seleção de ferramentas com repetições",
        "",
        "Gerado por `evals/compare_models.py --repeats {}` em {} (commit `{}`, Ollama {}).".format(
            summary["repeats"], summary["meta"]["timestamp"], summary["meta"]["git_commit"],
            summary["meta"]["ollama"].get("version"),
        ),
        "Máquina: {} · GPU: {}. Um modelo por vez, seed {} igual em todas as rodadas (o que varia "
        "é o não determinismo da GPU), temperatura 0,2, `num_ctx` {}.".format(
            summary["meta"]["platform"], summary["meta"].get("gpu") or "—", common.DEFAULT_SEED,
            first["num_ctx"] if first else "—",
        ),
        "",
    ]
    if not reps:
        parts.append("Nenhum resultado.")
        return "\n".join(parts) + "\n"
    boot = first["stats"]["tool"]["bootstrap"]
    parts += [
        "Dataset `{}` ({} casos, sha256 `{}`), {} ferramentas reais.".format(
            first["dataset"], first["stats"]["tool"]["n_cases"], first["dataset_sha256"],
            first["n_tools"]),
        "",
        "IC 95% principal: bootstrap percentil sobre os casos ({} reamostragens, seed {}; cada caso "
        "entra com a fração de rodadas em que acertou). Wilson: conferência, sobre a média de "
        "acertos com n = nº de casos. Desvio: amostral, entre as rodadas. Ver `evals/stats.py`."
        .format(boot["resamples"], boot["seed"]),
        "",
        md_table(
            ["Modelo", "Modo", "Acertos por rodada", "Média", "Desvio entre rodadas",
             "IC 95% bootstrap", "IC 95% Wilson", "Casos instáveis", "Args (média ± desvio)"],
            [repeated_row(r) for r in reps],
        ),
    ]
    if comps:
        alpha = comps[0].get("alpha_bonferroni", 0.05)
        parts += [
            "",
            "## Comparações pareadas (mesmos casos)",
            "",
            "McNemar exato bilateral sobre o acerto por maioria das rodadas; \"só A / só B\" são os "
            "casos discordantes. Também rodada a rodada (rodada i × rodada i), como sensibilidade. "
            "Diferença = média por caso da fração de acertos de A menos a de B, com IC bootstrap "
            "pareado. Com {} comparações, o limiar corrigido por Bonferroni é p < {}."
            .format(len(comps), "{:.4f}".format(alpha).replace(".", ",")),
            "",
            md_table(
                ["Comparação", "Modo", "Só A / só B", "p (maioria)", "p por rodada",
                 "Diferença A − B", "IC 95% da diferença", "Significativa?"],
                [comparison_row(c) for c in comps],
            ),
        ]
    if summary.get("failures"):
        parts += ["", "## Falhas", ""]
        parts.extend(
            "- `{}`: {}".format(f["target"], f["error"].splitlines()[-1]) for f in summary["failures"]
        )
    parts += ["", "Arquivos brutos (cada rodada, cada caso): " + ", ".join(
        "`{}`".format(name) for name in summary["files"]), ""]
    return "\n".join(parts)
