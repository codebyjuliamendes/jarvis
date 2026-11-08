#!/usr/bin/env python3
"""Avaliação de seleção de ferramentas do Jarvis.

Para cada pedido de evals/data/tool_selection.jsonl faz UMA chamada ao Ollama
com o system prompt e os schemas REAIS do agente e registra a ferramenta
escolhida, sem executá-la.

    .venv/Scripts/python.exe evals/eval_tools.py --model qwen2.5:3b --mode native
    .venv/Scripts/python.exe evals/eval_tools.py --model qwen2.5:3b --mode json --limit 10
    .venv/Scripts/python.exe evals/eval_tools.py --model qwen2.5:3b --repeats 3   # média, desvio, IC 95%
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import common, stats  # noqa: E402
from jarvis.agent import Agent  # noqa: E402
from jarvis.llm import OllamaClient, OllamaError, ToolsNotSupported  # noqa: E402
from jarvis.memory import Memory  # noqa: E402
from jarvis.metrics_utils import (  # noqa: E402
    aggregate_tokens_per_second,
    latency_summary,
    ollama_timings,
    ratio,
)
from jarvis.rag import RagIndex  # noqa: E402
from jarvis.reminders import ReminderService  # noqa: E402
from jarvis.tool_parsing import extract_text_tool, parse_tool_call  # noqa: E402
from jarvis.tools import build_registry  # noqa: E402

DATA_PATH = common.DATA_DIR / "tool_selection.jsonl"
MODES = ("native", "json")


# ------------------------------------------------------------------- decisão


def decide(
    message: Dict[str, Any], mode: str
) -> Tuple[Optional[str], Dict[str, Any], Dict[str, Any]]:
    """Interpreta a resposta do modelo como o agente faria.

    Devolve (ferramenta ou None, args, flags). Flags:
    - invalid_json: o modelo tentou chamar uma ferramenta e o parser do agente
      não conseguiu extrair ferramenta + args;
    - leaked: no modo nativo, JSON de ferramenta no texto em vez de tool_calls
      (o agente NÃO executa isso, então conta como "nenhuma ferramenta");
    - n_calls: quantas chamadas vieram (o agente executaria todas; avaliamos a 1ª).
    """
    flags: Dict[str, Any] = {"invalid_json": False, "leaked": False, "n_calls": 0}
    content = (message.get("content") or "").strip()
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        flags["n_calls"] = len(tool_calls)
        first = tool_calls[0]
        name, args = parse_tool_call(first)
        function = first.get("function") if isinstance(first, dict) else None
        raw_args = function.get("arguments") if isinstance(function, dict) else None
        if isinstance(raw_args, str):
            try:
                json.loads(raw_args)
            except ValueError:
                flags["invalid_json"] = True
        if not name:
            flags["invalid_json"] = True
            return None, {}, flags
        return name, args, flags

    parsed = extract_text_tool(content)
    if mode == "json":
        if parsed is None:
            if "{" in content:
                flags["invalid_json"] = True
            return None, {}, flags
        flags["n_calls"] = 1
        return parsed[0], parsed[1], flags

    # modo nativo sem tool_calls: resposta em texto (o agente devolve como final)
    if parsed is not None:
        flags["leaked"] = True
    return None, {}, flags


# ------------------------------------------------------------------ métricas


def score_case(case: Dict[str, Any], predicted: Optional[str], args: Dict[str, Any]) -> Dict[str, Any]:
    expected = case.get("expected_tool")
    acceptable = [expected] + list(case.get("acceptable_tools") or [])
    tool_ok = predicted in acceptable
    expected_args = case.get("expected_args")
    args_ok: Optional[bool] = None
    args_failed: List[str] = []
    if expected_args:
        if predicted == expected:
            args_ok, args_failed = common.match_args(args, expected_args)
        else:
            args_ok = False
            args_failed = ["<ferramenta errada>"]
    return {"tool_ok": tool_ok, "args_ok": args_ok, "args_failed": args_failed}


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    tool_hits = sum(1 for r in records if r["tool_ok"])
    with_args = [r for r in records if r["args_ok"] is not None]
    args_hits = sum(1 for r in with_args if r["args_ok"])
    with_args_tool_ok = [r for r in with_args if r["predicted"] == r["expected_tool"]]
    args_hits_cond = sum(1 for r in with_args_tool_ok if r["args_ok"])
    invalid = sum(1 for r in records if r["invalid_json"])
    leaked = sum(1 for r in records if r["leaked"])
    errors = sum(1 for r in records if r["error"])
    unknown = sum(1 for r in records if r["unknown_tool"])
    latencies = [r["latency_s"] for r in records if r["latency_s"] is not None and not r["error"]]
    responses = [r["timings"] for r in records if r.get("timings")]

    by_category: Dict[str, Dict[str, Any]] = {}
    for category in sorted({r["category"] for r in records}):
        subset = [r for r in records if r["category"] == category]
        hits = sum(1 for r in subset if r["tool_ok"])
        by_category[category] = {"n": len(subset), "hits": hits, "accuracy": ratio(hits, len(subset))}

    confusion = Counter(
        "{} -> {}".format(r["expected_tool"], r["predicted"]) for r in records if not r["tool_ok"]
    )
    return {
        "n": total,
        "tool_accuracy": ratio(tool_hits, total),
        "tool_hits": tool_hits,
        "args_accuracy": ratio(args_hits, len(with_args)),
        "args_hits": args_hits,
        "args_n": len(with_args),
        "args_accuracy_given_tool": ratio(args_hits_cond, len(with_args_tool_ok)),
        "args_hits_given_tool": args_hits_cond,
        "args_n_given_tool": len(with_args_tool_ok),
        "invalid_json_rate": ratio(invalid, total),
        "invalid_json": invalid,
        "leaked_calls": leaked,
        "unknown_tool_calls": unknown,
        "errors": errors,
        "latency_s": latency_summary(latencies),
        "tokens_per_s": aggregate_tokens_per_second(
            {"eval_count": t["eval_count"], "eval_duration": (t["eval_s"] or 0) * 1e9}
            for t in responses
            if t.get("eval_count") is not None and t.get("eval_s")
        ),
        "mean_prompt_tokens": ratio(
            sum(t["prompt_eval_count"] or 0 for t in responses), len(responses)
        ),
        "by_category": by_category,
        "errors_by_pair": dict(confusion.most_common()),
    }


# ------------------------------------------------------------------- execução


async def run_eval(
    model: str,
    mode: str = "native",
    cases: Optional[List[Dict[str, Any]]] = None,
    ollama_url: str = common.DEFAULT_OLLAMA_URL,
    seed: int = common.DEFAULT_SEED,
    verbose: bool = False,
) -> Dict[str, Any]:
    if mode not in MODES:
        raise ValueError("modo deve ser native ou json")
    cases = cases if cases is not None else common.load_jsonl(DATA_PATH)

    with tempfile.TemporaryDirectory() as tmp:
        settings = common.make_eval_settings(Path(tmp), model=model, ollama_url=ollama_url)
        rag = RagIndex(settings)
        reminders = ReminderService(settings)  # não é iniciado: só registra a ferramenta
        memory = Memory(settings.db_path)
        try:
            registry = build_registry(settings, reminders, rag)
            known = set(registry.names())
            for case in cases:
                for name in [case.get("expected_tool")] + list(case.get("acceptable_tools") or []):
                    if name is not None and name not in known:
                        raise ValueError("{}: ferramenta inexistente '{}'".format(case["id"], name))

            client = OllamaClient(
                url=settings.ollama_url,
                model=model,
                timeout=settings.ollama_timeout,
                num_ctx=settings.ollama_num_ctx,
                extra_options={"seed": seed},
            )
            agent = Agent(settings, registry, memory, llm=client)
            effective_mode = mode

            def build_messages(prompt: str, text_mode: bool) -> List[Dict[str, Any]]:
                return [
                    {"role": "system", "content": agent._system_prompt(text_mode=text_mode)},
                    {"role": "user", "content": prompt},
                ]

            async def call(prompt: str) -> Tuple[Dict[str, Any], float]:
                nonlocal effective_mode
                text_mode = effective_mode == "json"
                tools = None if text_mode else registry.schemas()
                started = time.perf_counter()
                try:
                    data = await client.chat_full(build_messages(prompt, text_mode), tools=tools)
                except ToolsNotSupported:
                    # Igual ao agente: sem function calling nativo, cai para JSON.
                    effective_mode = "json"
                    started = time.perf_counter()
                    data = await client.chat_full(build_messages(prompt, True), tools=None)
                return data, time.perf_counter() - started

            # Aquecimento: carrega o modelo na GPU; fora das métricas.
            warmup_started = time.perf_counter()
            await call("oi")
            warmup_s = time.perf_counter() - warmup_started

            records: List[Dict[str, Any]] = []
            for index, case in enumerate(cases, start=1):
                record: Dict[str, Any] = {
                    "id": case["id"],
                    "category": case.get("category", ""),
                    "prompt": case["prompt"],
                    "expected_tool": case.get("expected_tool"),
                    "predicted": None,
                    "args": {},
                    "error": None,
                    "latency_s": None,
                    "timings": None,
                    "invalid_json": False,
                    "leaked": False,
                    "n_calls": 0,
                    "unknown_tool": False,
                    "content": "",
                }
                try:
                    data, latency = await call(case["prompt"])
                    message = data.get("message") or {}
                    predicted, args, flags = decide(message, effective_mode)
                    record.update(flags)
                    record.update(
                        {
                            "predicted": predicted,
                            "args": args,
                            "latency_s": latency,
                            "timings": ollama_timings(data),
                            "content": (message.get("content") or "")[:500],
                            "unknown_tool": predicted is not None and predicted not in known,
                            "mode": effective_mode,
                        }
                    )
                except OllamaError as exc:
                    record["error"] = str(exc)[:500]
                    predicted, args = None, {}
                record.update(score_case(case, record["predicted"], record["args"]))
                records.append(record)
                if verbose:
                    mark = "ok " if record["tool_ok"] else "ERR"
                    print(
                        "[{:>3}/{}] {} {:<8} esperado={} previsto={} {:.2f}s".format(
                            index, len(cases), mark, case["id"], case.get("expected_tool"),
                            record["predicted"], record["latency_s"] or 0.0,
                        ),
                        flush=True,
                    )
        finally:
            memory.close()
            rag.close()
            reminders.close()

    return {
        "kind": "tool_selection",
        "model": model,
        "mode": mode,
        "effective_mode": effective_mode,
        "dataset": str(DATA_PATH.relative_to(common.BASE_DIR).as_posix()),
        "dataset_sha256": common.file_sha256(DATA_PATH),
        "n_tools": len(known),
        "seed": seed,
        "num_ctx": settings.ollama_num_ctx,
        "warmup_s": warmup_s,
        "meta": common.run_metadata(ollama_url),
        "metrics": compute_metrics(records),
        "records": records,
    }


# ---------------------------------------------------------------- repetições


def case_outcomes(result: Dict[str, Any], field: str = "tool_ok") -> Dict[str, bool]:
    """{id do caso: acertou} de uma rodada. Com ``args_ok``, só os casos que têm args esperados."""
    return {
        record["id"]: bool(record[field])
        for record in result["records"]
        if record.get(field) is not None
    }


def aggregate_repeats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Junta N rodadas da mesma configuração num resultado com estatística (ver evals/stats.py)."""
    if not results:
        raise ValueError("nenhuma rodada")
    first = results[0]
    for other in results[1:]:
        if (other["model"], other["mode"], other["dataset_sha256"]) != (
            first["model"], first["mode"], first["dataset_sha256"]
        ):
            raise ValueError("rodadas de configurações diferentes")
    tool_runs = [case_outcomes(r, "tool_ok") for r in results]
    args_runs = [case_outcomes(r, "args_ok") for r in results]
    return {
        "kind": "tool_selection_repeated",
        "model": first["model"],
        "mode": first["mode"],
        "effective_mode": first["effective_mode"],
        "repeats": len(results),
        "dataset": first["dataset"],
        "dataset_sha256": first["dataset_sha256"],
        "n_tools": first["n_tools"],
        "seed": first["seed"],
        "num_ctx": first["num_ctx"],
        "meta": first["meta"],
        "stats": {
            "tool": stats.accuracy_summary(tool_runs),
            "args": stats.accuracy_summary(args_runs) if args_runs[0] else None,
            "invalid_json_per_run": [r["metrics"]["invalid_json"] for r in results],
            "latency_p50_per_run": [r["metrics"]["latency_s"]["p50"] for r in results],
            "tokens_per_s_per_run": [r["metrics"]["tokens_per_s"] for r in results],
        },
        "runs": [
            {k: v for k, v in r.items() if k not in ("meta", "dataset", "n_tools", "num_ctx")}
            for r in results
        ],
    }


def runs_outcomes(repeated: Dict[str, Any], field: str = "tool_ok") -> List[Dict[str, bool]]:
    return [case_outcomes(run, field) for run in repeated["runs"]]


async def run_repeated(
    model: str,
    mode: str = "native",
    cases: Optional[List[Dict[str, Any]]] = None,
    repeats: int = 3,
    ollama_url: str = common.DEFAULT_OLLAMA_URL,
    seed: int = common.DEFAULT_SEED,
    verbose: bool = False,
) -> Dict[str, Any]:
    """N rodadas idênticas (mesmo seed: o que varia é o não determinismo da GPU)."""
    if repeats < 1:
        raise ValueError("repeats deve ser >= 1")
    results = []
    for index in range(repeats):
        print("   rodada {}/{}".format(index + 1, repeats), flush=True)
        results.append(await run_eval(model, mode, cases, ollama_url, seed, verbose))
    return aggregate_repeats(results)


def fmt_pp(value: Optional[float]) -> str:
    """Desvio em pontos percentuais."""
    if value is None:
        return "—"
    return "{:.1f}".format(value * 100).replace(".", ",") + " p.p."


def fmt_ci(interval: List[float]) -> str:
    return "[{}; {}]".format(
        "{:.1f}".format(interval[0] * 100).replace(".", ","),
        "{:.1f}%".format(interval[1] * 100).replace(".", ","),
    )


def print_repeated_summary(repeated: Dict[str, Any]) -> None:
    tool = repeated["stats"]["tool"]
    print("\n== {} ({}) · {} rodadas ==".format(
        repeated["model"], repeated["effective_mode"], repeated["repeats"]))
    print("acertos por rodada: {} de {}".format(
        ", ".join(str(h) for h in tool["hits_per_run"]), tool["n_cases"]))
    print("acurácia média: {} · desvio entre rodadas: {}".format(
        common.fmt_pct(tool["mean"]), fmt_pp(tool["stdev"])))
    print("IC 95% bootstrap (casos): {} · Wilson: {}".format(
        fmt_ci(tool["ci95_bootstrap"]), fmt_ci(tool["ci95_wilson"])))
    print("casos instáveis (mudaram entre rodadas): {}".format(tool["unstable_cases"]))


def repeated_filename(repeated: Dict[str, Any]) -> str:
    stamp = repeated["meta"]["timestamp"][:19].replace(":", "").replace("-", "")
    return "{}_tools_{}_{}_x{}.json".format(
        stamp, common.slug(repeated["model"]), repeated["mode"], repeated["repeats"])


def log_repeated_to_mlflow(
    repeated: Dict[str, Any], saved: List[Path], nested: bool = False
) -> Optional[str]:
    """Um run por (modelo, modo) com a média, o desvio, os ICs e cada rodada como métrica."""
    from evals import tracking

    run_id = tracking.log_run(
        run_name="tools {} {} x{}".format(
            repeated["model"], repeated["effective_mode"], repeated["repeats"]),
        params=tracking.repeated_tool_params(repeated),
        metrics=tracking.repeated_tool_metrics(repeated),
        artifacts=list(saved),
        tags={"eval": "tool_selection", "repeats": str(repeated["repeats"])},
        nested=nested,
    )
    if run_id:
        print("MLflow: run {} registrado em {}".format(run_id[:8], tracking.tracking_uri()))
    return run_id


def print_summary(result: Dict[str, Any]) -> None:
    m = result["metrics"]
    lat = m["latency_s"]
    print("\n== Seleção de ferramentas: {} ({}) ==".format(result["model"], result["effective_mode"]))
    print("casos: {}".format(m["n"]))
    print("acurácia de ferramenta: {}".format(common.fmt_pct(m["tool_accuracy"], m["tool_hits"], m["n"])))
    print("acurácia de args:       {}".format(
        common.fmt_pct(m["args_accuracy"], m["args_hits"], m["args_n"])))
    print("  (dado ferramenta certa: {})".format(
        common.fmt_pct(m["args_accuracy_given_tool"], m["args_hits_given_tool"], m["args_n_given_tool"])))
    print("JSON inválido:          {}".format(
        common.fmt_pct(m["invalid_json_rate"], m["invalid_json"], m["n"])))
    print("chamadas vazadas no texto: {} · ferramenta inexistente: {} · erros: {}".format(
        m["leaked_calls"], m["unknown_tool_calls"], m["errors"]))
    print("latência média/p50/p95: {} / {} / {} s".format(
        common.fmt_num(lat["mean"]), common.fmt_num(lat["p50"]), common.fmt_num(lat["p95"])))
    print("tokens/s: {}".format(common.fmt_num(m["tokens_per_s"], 1)))
    for category, info in m["by_category"].items():
        print("  {:<15} {}".format(category, common.fmt_pct(info["accuracy"], info["hits"], info["n"])))


def result_filename(result: Dict[str, Any]) -> str:
    stamp = result["meta"]["timestamp"][:19].replace(":", "").replace("-", "")
    return "{}_tools_{}_{}.json".format(stamp, common.slug(result["model"]), result["mode"])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=MODES, default="native")
    parser.add_argument("--limit", type=int, default=0, help="usar só os N primeiros casos")
    parser.add_argument("--ollama-url", default=common.DEFAULT_OLLAMA_URL)
    parser.add_argument("--seed", type=int, default=common.DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=1,
                        help="rodadas idênticas; com N > 1 reporta média, desvio e IC 95%%")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not common.model_available(args.ollama_url, args.model):
        print("Modelo '{}' não encontrado no Ollama em {}.".format(args.model, args.ollama_url))
        return 2
    if args.repeats < 1:
        parser.error("--repeats deve ser >= 1")
    cases = common.load_jsonl(DATA_PATH)
    if args.limit:
        cases = cases[: args.limit]
    if args.repeats > 1:
        repeated = asyncio.run(run_repeated(
            args.model, args.mode, cases, args.repeats, args.ollama_url, args.seed, args.verbose))
        for run in repeated["runs"]:
            print_summary(run)
        print_repeated_summary(repeated)
        saved_repeated: List[Path] = []
        if not args.no_save:
            path = common.save_json(repeated, repeated_filename(repeated))
            saved_repeated.append(path)
            print("\nresultado salvo em {}".format(path.relative_to(common.BASE_DIR)))
        if not args.no_mlflow:
            log_repeated_to_mlflow(repeated, saved_repeated)
        return 0
    result = asyncio.run(
        run_eval(args.model, args.mode, cases, args.ollama_url, args.seed, args.verbose)
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
    """Um run por (modelo, modo): params, métricas, JSON e tabela Markdown."""
    from evals import report, tracking

    with tempfile.TemporaryDirectory() as tmp:
        files = [p.name for p in saved]
        json_path = saved[0] if saved else common.save_json(result, result_filename(result), Path(tmp))
        md_path = Path(tmp) / "tabela.md"
        md_path.write_text(
            report.render_markdown(report.single_summary(result, "eval_tools.py", files)),
            encoding="utf-8",
        )
        run_id = tracking.log_run(
            run_name="tools {} {}".format(result["model"], result["effective_mode"]),
            params=tracking.tool_params(result),
            metrics=tracking.tool_metrics(result),
            artifacts=[json_path, md_path],
            tags={"eval": "tool_selection"},
            nested=nested,
        )
    if run_id:
        print("MLflow: run {} registrado em {}".format(run_id[:8], tracking.tracking_uri()))
    return run_id


if __name__ == "__main__":
    sys.exit(main())
