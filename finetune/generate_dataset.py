#!/usr/bin/env python3
"""Gera o dataset sintético de tool calling (pt-BR) para o LoRA do Jarvis.

    .venv/Scripts/python.exe finetune/generate_dataset.py               # padrão: seed 42, 2.200 pedidos
    .venv/Scripts/python.exe finetune/generate_dataset.py --total 3000  # maior
    .venv/Scripts/python.exe finetune/generate_dataset.py --paraphrase-frac 0.1 --paraphrase-model qwen2.5:3b

O que ele faz, nesta ordem:

1. monta o registro REAL de ferramentas (``build_registry``) e o system prompt
   REAL do ``Agent`` — falha se alguma ferramenta do registro não tiver modelos
   de frase em ``templates.py`` (ou se algum modelo citar ferramenta inexistente);
2. gera exemplos por modelos de frase (determinístico com ``--seed``), com uma
   fração de casos **sem ferramenta** e de **continuação** (resultado da
   ferramenta -> resposta final);
3. valida cada chamada contra o JSON Schema da ferramenta;
4. opcionalmente parafraseia uma fração dos pedidos com um modelo do Ollama
   (``--paraphrase-frac``), descartando paráfrases que percam alguma entidade;
5. remove duplicatas e **vazamentos** contra ``evals/data/tool_selection.jsonl``
   (ver ``leakage.py``), confere de novo e falha se sobrar algum;
6. grava ``finetune/data/{train,val}.jsonl``, ``tools.json``, ``stats.json`` e
   ``leakage_report.json``.

Cada linha do JSONL é uma conversa no formato de mensagens do Ollama/OpenAI; os
25 schemas ficam uma vez só em ``tools.json`` (são os mesmos em todo exemplo).
``chat_format.render_example`` transforma a conversa no texto exato que o
Ollama monta para o ``qwen2.5``.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
from collections import Counter, OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from finetune import leakage, templates  # noqa: E402
from finetune.chat_format import render_example  # noqa: E402

DATA_DIR = BASE_DIR / "finetune" / "data"
EVAL_PATH = BASE_DIR / "evals" / "data" / "tool_selection.jsonl"
WS_TOKEN = "@@WORKSPACE@@"
DATE_RE = re.compile(r"Data/hora: \d{2}/\d{2}/\d{4} \d{2}:\d{2}")

WORKSPACES = [
    "/home/julia", "/home/julia/projetos", "/home/julia/jarvis", "/Users/julia/dev",
    "C:\\Users\\julia\\projetos", "C:\\Users\\julia\\AppData\\Local\\Temp\\tmp{rand}",
]


# --------------------------------------------------------------- registro real


def load_real_context() -> Tuple[str, List[Dict[str, Any]]]:
    """System prompt real (com marcadores) e os schemas reais das ferramentas."""
    from evals import common
    from jarvis.agent import Agent
    from jarvis.memory import Memory
    from jarvis.rag import RagIndex
    from jarvis.reminders import ReminderService
    from jarvis.tools import build_registry

    with tempfile.TemporaryDirectory() as tmp:
        settings = common.make_eval_settings(Path(tmp))
        rag = RagIndex(settings)
        reminders = ReminderService(settings)
        memory = Memory(settings.db_path)
        try:
            registry = build_registry(settings, reminders, rag)
            schemas = registry.schemas()
            settings.workspace = Path(WS_TOKEN)
            settings.allowed_roots = [Path(WS_TOKEN)]
            agent = Agent(settings, registry, memory, llm=object())
            prompt = agent._system_prompt()
        finally:
            memory.close()
            rag.close()
            reminders.close()
    if not DATE_RE.search(prompt) or WS_TOKEN not in prompt:
        raise RuntimeError("o system prompt do Agent mudou de formato; ajuste DATE_RE/WS_TOKEN")
    return prompt, schemas


def system_prompt(template: str, now: datetime, workspace: str) -> str:
    text = DATE_RE.sub("Data/hora: " + now.strftime("%d/%m/%Y %H:%M"), template)
    return text.replace(WS_TOKEN, workspace)


# ------------------------------------------------------------ validação de args


def _type_ok(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    return True


def validate_args(args: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Erros de ``args`` contra o JSON Schema (subconjunto usado pelas ferramentas)."""
    errors: List[str] = []
    if not isinstance(args, dict):
        return ["argumentos não são um objeto"]
    properties = schema.get("properties") or {}
    for name in schema.get("required") or []:
        if name not in args:
            errors.append("falta o obrigatório '{}'".format(name))
    for name, value in args.items():
        prop = properties.get(name)
        if prop is None:
            errors.append("argumento desconhecido '{}'".format(name))
            continue
        kind = prop.get("type")
        if kind and not _type_ok(value, kind):
            errors.append("'{}' deveria ser {}".format(name, kind))
        if "enum" in prop and value not in prop["enum"]:
            errors.append("'{}'={!r} fora do enum".format(name, value))
        if kind == "array" and isinstance(value, list):
            item_kind = (prop.get("items") or {}).get("type")
            if item_kind and not all(_type_ok(v, item_kind) for v in value):
                errors.append("itens de '{}' deveriam ser {}".format(name, item_kind))
    return errors


def check_coverage(schemas: Sequence[Dict[str, Any]]) -> None:
    names = {s["function"]["name"] for s in schemas}
    covered = {i.tool for i in templates.TOOL_INTENTS}
    missing = sorted(names - covered)
    unknown = sorted(covered - names)
    if missing or unknown:
        raise SystemExit(
            "Modelos de frase desalinhados com o registro real. Sem modelos: {}; "
            "ferramentas que não existem mais: {}".format(missing or "-", unknown or "-")
        )
    for intent_id in templates.CONTINUATIONS:
        if intent_id not in {i.id for i in templates.TOOL_INTENTS}:
            raise SystemExit("continuação para intenção inexistente: " + intent_id)


# ------------------------------------------------------------------ geração


def split_phrasings(intents: Sequence[templates.Intent], val_frac: float, seed: int) -> Dict[str, str]:
    """Sorteia quais modelos de frase vão para o val (por intenção, fixo pela seed).

    Intenção com 3+ frases manda ``round(n * val_frac)`` (mínimo 1) para o val;
    com menos de 3, todas ficam no treino.
    """
    rng = random.Random(seed * 7919 + 1)
    assignment: Dict[str, str] = {}
    for intent in intents:
        ids = ["{}.{:02d}".format(intent.id, i) for i in range(len(intent.phrasings))]
        n_val = max(1, int(round(len(ids) * val_frac))) if len(ids) >= 3 else 0
        held = set(rng.sample(ids, n_val)) if n_val else set()
        for template_id in ids:
            assignment[template_id] = "val" if template_id in held else "train"
    return assignment


def _allocate(total: int, intents: Sequence[templates.Intent]) -> List[int]:
    """Distribui ``total`` exemplos entre intenções proporcionalmente ao peso."""
    weights = [i.weight for i in intents]
    raw = [total * w / sum(weights) for w in weights]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda k: -(raw[k] - counts[k]))
    for k in order[:remainder]:
        counts[k] += 1
    return counts


def _render_user(rng: random.Random, intent: templates.Intent, index: int, slots: Dict[str, Any]) -> str:
    phrasing = templates.vary_surface(rng, intent.phrasings[index])
    return phrasing.format(**slots)


def _tool_message(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": tool, "arguments": args}}]}


def generate(
    seed: int,
    total: int,
    no_tool_frac: float,
    continuation_frac: float,
    val_frac: float,
    sys_template: str,
    schemas: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """Devolve (exemplos, intenções que esgotaram antes da meta: id -> [pedidos, gerados])."""
    rng = random.Random(seed)
    schema_by_name = {s["function"]["name"]: s["function"]["parameters"] for s in schemas}
    splits = split_phrasings(templates.TOOL_INTENTS + templates.NO_TOOL_INTENTS, val_frac, seed)
    start = datetime(2026, 1, 1, 7, 0)

    def new_context() -> Tuple[templates.Context, str]:
        now = start + timedelta(days=rng.randint(0, 729), minutes=rng.randint(0, 16 * 60))
        workspace = templates.pick(rng, WORKSPACES).format(rand="".join(rng.choice("abcdefghij0123456789") for _ in range(8)))
        return templates.Context(now=now), system_prompt(sys_template, now, workspace)

    n_no_tool = int(round(total * no_tool_frac))
    n_cont = int(round(total * continuation_frac))
    n_tool = total - n_no_tool - n_cont

    examples: List[Dict[str, Any]] = []
    seen = set()
    shortfall: Dict[str, List[int]] = {}

    def produce(intent: templates.Intent, n: int, build: Any) -> None:
        """Gera até ``n`` exemplos únicos da intenção (pedido + resposta).

        Intenções com poucas entidades se esgotam antes de ``n``: tenta no máximo
        ``8 * n`` vezes e fica com os únicos que saíram.
        """
        made, attempts = 0, 0
        while made < n and attempts < 8 * n:
            attempts += 1
            example = build(intent)
            key = (leakage.normalize(leakage.user_text(example)),
                   json.dumps(example["messages"][2:], ensure_ascii=False, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            examples.append(example)
            made += 1
        if made < n:
            shortfall[intent.id + ("" if build is not build_continuation else ":continuação")] = [n, made]

    def build_tool(intent: templates.Intent) -> Dict[str, Any]:
        ctx, system = new_context()
        index = rng.randrange(len(intent.phrasings))
        slots, args = intent.make(rng, ctx)
        errors = validate_args(args, schema_by_name[intent.tool])
        if errors:
            raise SystemExit("{}: argumentos inválidos {}: {}".format(intent.id, args, errors))
        template_id = "{}.{:02d}".format(intent.id, index)
        return {
            "kind": "tool", "tool": intent.tool, "template": template_id, "split": splits[template_id],
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": _render_user(rng, intent, index, slots)},
                         _tool_message(intent.tool, args)],
        }

    def build_no_tool(intent: templates.Intent) -> Dict[str, Any]:
        ctx, system = new_context()
        index = rng.randrange(len(intent.phrasings))
        slots, answer = intent.make(rng, ctx)
        template_id = "{}.{:02d}".format(intent.id, index)
        return {
            "kind": "no_tool", "tool": None, "template": template_id, "split": splits[template_id],
            "tags": list(intent.tags),
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": _render_user(rng, intent, index, slots)},
                         {"role": "assistant", "content": answer}],
        }

    def build_continuation(intent: templates.Intent) -> Dict[str, Any]:
        """Pedido -> chamada -> resultado (inventado) -> resposta final em texto."""
        ctx, system = new_context()
        index = rng.randrange(len(intent.phrasings))
        slots, args = intent.make(rng, ctx)
        output, reply = templates.CONTINUATIONS[intent.id](rng, args)
        template_id = "{}.{:02d}".format(intent.id, index)
        return {
            "kind": "continuation", "tool": intent.tool, "template": template_id,
            "split": splits[template_id],
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": _render_user(rng, intent, index, slots)},
                         _tool_message(intent.tool, args),
                         {"role": "tool", "content": output},
                         {"role": "assistant", "content": reply}],
        }

    # com ferramenta: mesmo número por ferramenta, dividido entre as intenções pelo peso
    by_tool: "OrderedDict[str, List[templates.Intent]]" = OrderedDict()
    for intent in templates.TOOL_INTENTS:
        by_tool.setdefault(intent.tool, []).append(intent)
    per_tool = _allocate(n_tool, [templates.Intent(t, t, [""], lambda r, c: ({}, None)) for t in by_tool])
    for (tool, intents), count in zip(by_tool.items(), per_tool):
        for intent, n in zip(intents, _allocate(count, intents)):
            produce(intent, n, build_tool)
    for intent, n in zip(templates.NO_TOOL_INTENTS, _allocate(n_no_tool, templates.NO_TOOL_INTENTS)):
        produce(intent, n, build_no_tool)
    cont_intents = [i for i in templates.TOOL_INTENTS if i.id in templates.CONTINUATIONS]
    for intent, n in zip(cont_intents, _allocate(n_cont, cont_intents)):
        produce(intent, n, build_continuation)
    return examples, shortfall


# ----------------------------------------------------------------- paráfrase

PARAPHRASE_PROMPT = (
    "Reescreva o pedido abaixo em português do Brasil, do jeito que uma pessoa digitaria "
    "no Telegram, com outras palavras e a mesma intenção. Mantenha EXATAMENTE estes trechos, "
    "sem traduzir nem alterar: {keep}. Responda só com o pedido reescrito, sem aspas.\n\n"
    "Pedido: {text}"
)


def _must_keep(example: Dict[str, Any]) -> List[str]:
    """Valores de argumento que aparecem literalmente no pedido (têm de sobreviver)."""
    user = example["messages"][1]["content"]
    norm_user = leakage.normalize(user)
    keep: List[str] = []
    call = example["messages"][2].get("tool_calls") or []
    args = call[0]["function"]["arguments"] if call else {}
    for value in args.values():
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, bool):
                continue
            text = str(item)
            if leakage.normalize(text) and leakage.normalize(text) in norm_user:
                keep.append(text)
    return keep


def paraphrase(
    examples: List[Dict[str, Any]], frac: float, model: str, url: str, seed: int
) -> Dict[str, int]:
    """Reescreve ``frac`` dos pedidos com ferramenta via Ollama (no lugar)."""
    import httpx

    rng = random.Random(seed + 101)
    candidates = [e for e in examples if e["kind"] == "tool"]
    chosen = rng.sample(candidates, int(round(len(candidates) * frac)))
    stats = {"tentadas": len(chosen), "aceitas": 0, "perderam_entidade": 0, "iguais_ou_vazias": 0, "erros": 0}
    with httpx.Client(timeout=120.0) as client:
        for example in chosen:
            original = example["messages"][1]["content"]
            keep = _must_keep(example)
            prompt = PARAPHRASE_PROMPT.format(
                keep=", ".join("'{}'".format(k) for k in keep) or "(nada)", text=original)
            try:
                data = client.post(url.rstrip("/") + "/api/generate", json={
                    "model": model, "prompt": prompt, "stream": False,
                    "options": {"temperature": 0.7, "seed": rng.randint(0, 2 ** 31), "num_predict": 80},
                }).json()
            except Exception:  # noqa: BLE001 - paráfrase é opcional
                stats["erros"] += 1
                continue
            text = (data.get("response") or "").strip().strip("\"'").splitlines()[0:1]
            text = text[0].strip() if text else ""
            if not text or leakage.normalize(text) == leakage.normalize(original) or len(text) > 3 * len(original) + 40:
                stats["iguais_ou_vazias"] += 1
                continue
            if any(ord(ch) > 0x2FFF for ch in text):
                # o qwen2.5 às vezes escapa para caracteres chineses no meio da frase
                stats["iguais_ou_vazias"] += 1
                continue
            norm = leakage.normalize(text)
            if any(leakage.normalize(k) not in norm for k in keep):
                stats["perderam_entidade"] += 1
                continue
            example["messages"][1]["content"] = text
            example["paraphrased"] = True
            example["original_prompt"] = original
            stats["aceitas"] += 1
    return stats


# ------------------------------------------------------------------- saída


def dedupe(examples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen = set()
    kept = []
    for example in examples:
        target = json.dumps(example["messages"][2:], ensure_ascii=False, sort_keys=True)
        key = (leakage.normalize(leakage.user_text(example)), target)
        if key in seen:
            continue
        seen.add(key)
        kept.append(example)
    return kept, len(examples) - len(kept)


def composition(examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    kinds = Counter(e["kind"] for e in examples)
    tools = Counter(e["tool"] for e in examples if e["kind"] == "tool")
    return {
        "n": len(examples),
        "por_tipo": dict(kinds),
        "sem_ferramenta_pct": round(100.0 * kinds.get("no_tool", 0) / max(1, len(examples)), 1),
        "negativos_dificeis": sum(1 for e in examples if "negativo_dificil" in (e.get("tags") or [])),
        "parafraseados": sum(1 for e in examples if e.get("paraphrased")),
        "por_ferramenta": dict(sorted(tools.items())),
        "modelos_de_frase": len({e["template"] for e in examples}),
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(str(path), "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total", type=int, default=2200, help="exemplos pedidos antes dos filtros")
    parser.add_argument("--no-tool-frac", type=float, default=0.22)
    parser.add_argument("--continuation-frac", type=float, default=0.05)
    parser.add_argument("--val-frac", type=float, default=0.2, help="fração dos modelos de frase no val")
    parser.add_argument("--threshold", type=float, default=leakage.DEFAULT_THRESHOLD)
    parser.add_argument("--paraphrase-frac", type=float, default=0.0)
    parser.add_argument("--paraphrase-model", default="qwen2.5:3b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--out", default=str(DATA_DIR))
    args = parser.parse_args(argv)

    sys_template, schemas = load_real_context()
    check_coverage(schemas)
    examples, shortfall = generate(args.seed, args.total, args.no_tool_frac, args.continuation_frac,
                        args.val_frac, sys_template, schemas)
    paraphrase_stats = None
    if args.paraphrase_frac > 0:
        paraphrase_stats = paraphrase(examples, args.paraphrase_frac, args.paraphrase_model,
                                      args.ollama_url, args.seed)
    examples, n_dupes = dedupe(examples)

    index = leakage.EvalIndex.from_jsonl(EVAL_PATH)
    result = leakage.filter_examples(examples, index, args.threshold)
    clean = result["kept"]
    recheck = leakage.filter_examples(clean, index, args.threshold)
    if recheck["removed"]:
        raise SystemExit("vazamento depois do filtro: {} exemplos".format(len(recheck["removed"])))

    # ids estáveis, já sem os removidos
    for number, example in enumerate(clean, start=1):
        example_id = "ft-{:05d}".format(number)
        example.pop("id", None)
        ordered = OrderedDict([("id", example_id)])
        ordered.update(example)
        clean[number - 1] = ordered
    train = [e for e in clean if e["split"] == "train"]
    val = [e for e in clean if e["split"] == "val"]
    overlap = {e["template"] for e in train} & {e["template"] for e in val}
    if overlap:
        raise SystemExit("modelo de frase em treino e val: {}".format(sorted(overlap)[:5]))

    # tamanho em caracteres do texto renderizado (proxy de tokens; o notebook mede de verdade)
    lengths = sorted(len("".join(render_example(e, schemas))) for e in clean)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    with open(str(out / "tools.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(schemas, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    report = result["report"]
    report["eval_self_similarity"] = leakage.eval_self_similarity(index)
    report["sensitivity"] = {
        str(t): len(leakage.filter_examples(examples, index, t)["removed"]) for t in (0.3, 0.35, 0.4, 0.5)
    }
    with open(str(out / "leakage_report.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    stats = {
        "seed": args.seed, "pedidos": args.total, "gerados": len(examples) + n_dupes,
        "intencoes_esgotadas": shortfall, "duplicatas_pos_parafrase": n_dupes,
        "vazados_removidos": report["removed"], "limiar": args.threshold,
        "paráfrase": paraphrase_stats,
        "treino": composition(train), "val": composition(val), "tudo": composition(clean),
        "chars_renderizados": {"min": lengths[0], "p50": lengths[len(lengths) // 2], "max": lengths[-1]},
    }
    with open(str(out / "stats.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=1)
        handle.write("\n")

    print("pedidos: {} · gerados (únicos): {} · intenções esgotadas: {} · vazados removidos: {} (limiar {:.2f})".format(
        args.total, len(examples), len(shortfall), report["removed"], args.threshold))
    print("  por motivo: {}".format(report["removed_by_reason"]))
    print("  similaridade máxima dos mantidos: p50 {:.3f} · p95 {:.3f} · máx {:.3f}".format(
        report["kept_similarity"]["p50"], report["kept_similarity"]["p95"], report["kept_similarity"]["max"]))
    print("  sensibilidade (removidos por limiar): {}".format(report["sensitivity"]))
    if paraphrase_stats:
        print("paráfrase: {}".format(paraphrase_stats))
    for name, part in (("treino", train), ("val", val)):
        comp = composition(part)
        print("{}: {} exemplos · {} · sem ferramenta {}%".format(name, comp["n"], comp["por_tipo"], comp["sem_ferramenta_pct"]))
    print("texto renderizado (caracteres): p50 {} · máx {}".format(stats["chars_renderizados"]["p50"], lengths[-1]))
    print("arquivos em {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
