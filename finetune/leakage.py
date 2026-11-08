"""Checagem de vazamento entre o dataset de treino e o eval de ferramentas.

O conjunto ``evals/data/tool_selection.jsonl`` (78 casos) é só de **avaliação**:
nenhum pedido de treino pode ser cópia, paráfrase próxima ou carregar as
entidades específicas de um caso do eval. Três filtros, todos sobre o texto do
pedido da usuária:

1. **exato**: igualdade depois de :func:`normalize` (minúsculas, sem acento, sem
   pontuação, espaços colapsados);
2. **quase-duplicata**: similaridade de Jaccard entre os conjuntos de 4-gramas de
   caracteres do texto normalizado ≥ ``threshold`` (padrão
   :data:`DEFAULT_THRESHOLD`; a calibração está em ``finetune/README.md``);
3. **entidade**: o pedido contém uma entidade rara de algum caso do eval — nome
   de arquivo com extensão (e o nome sem a pasta), URL (e o domínio), e-mail ou
   ``dono/repo``.

Só biblioteca padrão. Com ~2 mil exemplos × 78 casos a comparação de todos com
todos leva menos de um segundo, então não precisa de MinHash/LSH.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set

NGRAM = 4
# Regra fixada antes de gerar o dataset: 1,5 × a maior similaridade entre dois casos
# DISTINTOS do próprio eval (0,234 com 4-gramas), arredondado -> 0,35. Ver README.
DEFAULT_THRESHOLD = 0.35

_URL_RE = re.compile(r"https?://[^\s'\"]+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_FILE_RE = re.compile(r"(?<![\w/.-])[\w./-]*[\w-]\.[a-z][a-z0-9]{1,4}(?![\w])")
_REPO_RE = re.compile(r"(?<![\w/.-])[a-z0-9_.-]+/[a-z0-9_.-]+(?![\w/])")


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(text: str) -> str:
    """Minúsculas, sem acento, só letras/dígitos e espaços simples."""
    text = _strip_accents(str(text).lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def char_ngrams(text: str, n: int = NGRAM) -> FrozenSet[str]:
    """4-gramas de caracteres do texto normalizado (com borda de espaço)."""
    padded = " " + normalize(text) + " "
    if len(padded) < n:
        return frozenset([padded])
    return frozenset(padded[i : i + n] for i in range(len(padded) - n + 1))


def jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / float(len(a | b))


def extract_entities(text: str) -> Set[str]:
    """Entidades "raras" de um pedido: arquivos, URLs/domínios, e-mails, dono/repo.

    Tudo em minúsculas e sem acento. Para arquivos dentro de pastas entra também
    o nome sem a pasta (``notas/x.txt`` -> ``x.txt``).
    """
    low = _strip_accents(str(text).lower())
    found: Set[str] = set()
    for url in _URL_RE.findall(low):
        url = url.rstrip(".,;:!?)")
        found.add(url)
        domain = url.split("://", 1)[1].split("/", 1)[0]
        found.add(domain)
        low = low.replace(url, " ")
    for email in _EMAIL_RE.findall(low):
        found.add(email)
        low = low.replace(email, " ")
    for name in _FILE_RE.findall(low):
        name = name.strip("./")
        if not name or name.replace(".", "").isdigit():
            continue
        found.add(name)
        if "/" in name:
            found.add(name.rsplit("/", 1)[1])
    for repo in _REPO_RE.findall(low):
        if "." in repo.rsplit("/", 1)[-1]:
            continue  # caminho de arquivo, já tratado acima
        found.add(repo)
    return found


def _contains_entity(text: str, entity: str) -> bool:
    low = _strip_accents(str(text).lower())
    pattern = r"(?<![\w]){}(?![\w])".format(re.escape(entity))
    return re.search(pattern, low) is not None


@dataclass
class EvalIndex:
    """Os casos do eval pré-processados para comparar rápido."""

    ids: List[str]
    prompts: List[str]
    normalized: List[str]
    grams: List[FrozenSet[str]]
    entities: Dict[str, List[str]]  # entidade -> ids dos casos que a contêm

    @classmethod
    def from_cases(cls, cases: Sequence[Dict[str, Any]]) -> EvalIndex:
        ids = [str(c.get("id")) for c in cases]
        prompts = [str(c["prompt"]) for c in cases]
        entities: Dict[str, List[str]] = {}
        for case_id, prompt in zip(ids, prompts):
            for entity in extract_entities(prompt):
                entities.setdefault(entity, []).append(case_id)
        return cls(
            ids=ids,
            prompts=prompts,
            normalized=[normalize(p) for p in prompts],
            grams=[char_ngrams(p) for p in prompts],
            entities=entities,
        )

    @classmethod
    def from_jsonl(cls, path: Path) -> EvalIndex:
        cases = []
        with open(str(path), encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    cases.append(json.loads(line))
        return cls.from_cases(cases)


@dataclass
class Verdict:
    leaked: bool
    reasons: List[str] = field(default_factory=list)
    best_similarity: float = 0.0
    best_eval_id: Optional[str] = None
    best_eval_prompt: Optional[str] = None
    entities: List[str] = field(default_factory=list)


def check_text(text: str, index: EvalIndex, threshold: float = DEFAULT_THRESHOLD) -> Verdict:
    """Compara um pedido de treino com todos os casos do eval."""
    norm = normalize(text)
    grams = char_ngrams(text)
    best, best_i = 0.0, -1
    exact = False
    for i, (eval_norm, eval_grams) in enumerate(zip(index.normalized, index.grams)):
        if norm == eval_norm:
            exact, best, best_i = True, 1.0, i
            break
        sim = jaccard(grams, eval_grams)
        if sim > best:
            best, best_i = sim, i
    entities = sorted(e for e in index.entities if _contains_entity(text, e))
    reasons: List[str] = []
    if exact:
        reasons.append("exato")
    elif best >= threshold:
        reasons.append("quase_duplicata")
    if entities:
        reasons.append("entidade")
    return Verdict(
        leaked=bool(reasons),
        reasons=reasons,
        best_similarity=best,
        best_eval_id=index.ids[best_i] if best_i >= 0 else None,
        best_eval_prompt=index.prompts[best_i] if best_i >= 0 else None,
        entities=entities,
    )


def user_text(example: Dict[str, Any]) -> str:
    """Todo texto escrito pela usuária no exemplo (mensagens ``user``)."""
    return "\n".join(
        str(m.get("content") or "") for m in example.get("messages", []) if m.get("role") == "user"
    )


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    data = sorted(values)
    pos = (len(data) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def filter_examples(
    examples: Iterable[Dict[str, Any]],
    index: EvalIndex,
    threshold: float = DEFAULT_THRESHOLD,
    max_listed: int = 200,
) -> Dict[str, Any]:
    """Separa exemplos limpos dos vazados e monta o relatório.

    Devolve ``{"kept": [...], "removed": [...], "report": {...}}``.
    """
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    counts = {"exato": 0, "quase_duplicata": 0, "entidade": 0}
    kept_sims: List[float] = []
    kept_pairs: List[Any] = []
    listed: List[Dict[str, Any]] = []
    for example in examples:
        text = user_text(example)
        verdict = check_text(text, index, threshold)
        if verdict.leaked:
            removed.append(example)
            for reason in verdict.reasons:
                counts[reason] += 1
            if len(listed) < max_listed:
                listed.append({
                    "id": example.get("id"),
                    "template": example.get("template"),
                    "prompt": text,
                    "reasons": verdict.reasons,
                    "similarity": round(verdict.best_similarity, 3),
                    "eval_id": verdict.best_eval_id,
                    "eval_prompt": verdict.best_eval_prompt,
                    "entities": verdict.entities,
                })
        else:
            kept.append(example)
            kept_sims.append(verdict.best_similarity)
            kept_pairs.append((verdict.best_similarity, text, verdict.best_eval_prompt))
    kept_pairs.sort(key=lambda item: -item[0])
    top_kept = [
        {"prompt": text, "similarity": round(sim, 3), "eval_prompt": eval_prompt}
        for sim, text, eval_prompt in kept_pairs[:10]
    ]
    report = {
        "threshold": threshold,
        "ngram": NGRAM,
        "eval_cases": len(index.ids),
        "eval_entities": sorted(index.entities),
        "checked": len(kept) + len(removed),
        "removed": len(removed),
        "kept": len(kept),
        "removed_by_reason": counts,
        "kept_similarity": {
            "p50": _percentile(kept_sims, 0.5),
            "p95": _percentile(kept_sims, 0.95),
            "max": max(kept_sims) if kept_sims else None,
        },
        "top_kept_similar": top_kept,
        "removed_examples": listed,
    }
    return {"kept": kept, "removed": removed, "report": report}


def eval_self_similarity(index: EvalIndex) -> Dict[str, Any]:
    """Similaridade entre casos **distintos** do próprio eval (para calibrar o limiar).

    Para cada caso, a maior similaridade com qualquer outro caso; devolve a
    distribuição e os pares mais parecidos.
    """
    best_per_case: List[float] = []
    pairs = []
    n = len(index.ids)
    for i in range(n):
        best = 0.0
        for j in range(n):
            if i == j:
                continue
            sim = jaccard(index.grams[i], index.grams[j])
            best = max(best, sim)
            if j > i:
                pairs.append((sim, index.ids[i], index.ids[j]))
        best_per_case.append(best)
    pairs.sort(reverse=True)
    return {
        "max_per_case_p50": _percentile(best_per_case, 0.5),
        "max_per_case_p95": _percentile(best_per_case, 0.95),
        "max": max(best_per_case) if best_per_case else None,
        "top_pairs": [
            {
                "similarity": round(s, 3),
                "a": index.prompts[index.ids.index(a)],
                "b": index.prompts[index.ids.index(b)],
            }
            for s, a, b in pairs[:5]
        ],
    }
