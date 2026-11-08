"""Estatística das avaliações, só com a biblioteca padrão (o `.venv` do bot é 3.8 e leve).

Por que estas escolhas (ver ``evals/README.md``, seção "Repetições e intervalo de confiança"):

- **Duas fontes de incerteza.** (1) O não determinismo do modelo: rodadas idênticas
  (mesmo seed, mesma temperatura) dão resultados diferentes na GPU — o desvio-padrão
  entre rodadas mede isso. (2) A amostra de casos: 78 pedidos são uma amostra de todos os
  pedidos possíveis. Esta costuma ser a maior das duas e é a que o intervalo de confiança
  precisa cobrir.
- **IC principal: bootstrap percentil sobre os casos.** Cada caso vira a fração de rodadas em
  que acertou (0, 1/3, 2/3, 1 com N=3); reamostramos os 78 casos com reposição (10.000 vezes,
  seed fixa) e tiramos os percentis 2,5 e 97,5 da média. Assim as rodadas de um mesmo caso
  ficam juntas (não são tratadas como 3×78 casos independentes, o que estreitaria o IC sem
  motivo) e as duas fontes entram.
- **IC de conferência: Wilson** sobre a média de acertos com n = nº de casos. Fórmula fechada,
  boa com proporções perto de 0 ou 1 e com n pequeno (melhor que a aproximação normal). Ignora
  a variação entre rodadas; serve para checar o bootstrap.
- **Comparação pareada: McNemar exato.** Os dois modelos respondem aos mesmos casos, então o
  teste certo olha só os casos discordantes (b = só A acerta, c = só B acerta) e testa
  b ~ Binomial(b + c, 1/2), bilateral. Com N rodadas, cada caso é "acerto" se acertou na
  maioria delas (N ímpar). Complementa com o IC bootstrap pareado da diferença de acurácia.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261108
Z_95 = 1.959963984540054


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def stdev(values: Sequence[float]) -> Optional[float]:
    """Desvio-padrão amostral (n - 1). None com menos de 2 valores."""
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


def wilson_interval(hits: float, n: int, z: float = Z_95) -> Tuple[float, float]:
    """Intervalo de Wilson para uma proporção (aceita ``hits`` fracionário: média de rodadas)."""
    if n <= 0:
        raise ValueError("n deve ser positivo")
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Percentil com interpolação linear (igual ao padrão do NumPy)."""
    if not sorted_values:
        raise ValueError("lista vazia")
    pos = (len(sorted_values) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return sorted_values[low]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)


def bootstrap_mean_ci(
    values: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = 0.95,
) -> Tuple[float, float]:
    """IC percentil da média por bootstrap (reamostra os valores com reposição)."""
    if not values:
        raise ValueError("lista vazia")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    alpha = (1 - level) / 2
    return _percentile(means, alpha), _percentile(means, 1 - alpha)


def per_case_rates(runs: Sequence[Dict[str, bool]]) -> Dict[str, float]:
    """Fração de rodadas em que cada caso acertou. ``runs`` = [{id: acertou}] por rodada."""
    if not runs:
        return {}
    ids = list(runs[0])
    for run in runs[1:]:
        if set(run) != set(ids):
            raise ValueError("as rodadas não têm os mesmos casos")
    return {case_id: sum(1 for run in runs if run[case_id]) / len(runs) for case_id in ids}


def majority(runs: Sequence[Dict[str, bool]]) -> Dict[str, bool]:
    """Acerto por maioria das rodadas (empate, só com N par, conta como erro)."""
    return {case_id: rate > 0.5 for case_id, rate in per_case_rates(runs).items()}


def accuracy_summary(runs: Sequence[Dict[str, bool]]) -> Dict[str, object]:
    """Resumo de N rodadas da mesma configuração: média, desvio, IC bootstrap e Wilson."""
    if not runs:
        raise ValueError("nenhuma rodada")
    n_cases = len(runs[0])
    hits = [sum(1 for ok in run.values() if ok) for run in runs]
    accuracies = [h / n_cases for h in hits]
    rates = list(per_case_rates(runs).values())
    boot_low, boot_high = bootstrap_mean_ci(rates)
    mean_hits = sum(hits) / len(hits)
    wilson_low, wilson_high = wilson_interval(mean_hits, n_cases)
    flips = sum(1 for rate in rates if 0 < rate < 1)
    return {
        "repeats": len(runs),
        "n_cases": n_cases,
        "hits_per_run": hits,
        "accuracy_per_run": accuracies,
        "mean": mean(accuracies),
        "stdev": stdev(accuracies),
        "mean_hits": mean_hits,
        "ci95_bootstrap": [boot_low, boot_high],
        "ci95_wilson": [wilson_low, wilson_high],
        "unstable_cases": flips,
        "majority_hits": sum(1 for ok in majority(runs).values() if ok),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "method": "percentil"},
    }


def mcnemar_exact(b: int, c: int) -> float:
    """p-valor bilateral do McNemar exato: b ~ Binomial(b + c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2 * tail)


def paired_comparison(
    runs_a: Sequence[Dict[str, bool]],
    runs_b: Sequence[Dict[str, bool]],
) -> Dict[str, object]:
    """McNemar exato (acerto por maioria) e IC bootstrap pareado da diferença A - B."""
    maj_a = majority(runs_a)
    maj_b = majority(runs_b)
    if set(maj_a) != set(maj_b):
        raise ValueError("os dois modelos não foram avaliados nos mesmos casos")
    ids = sorted(maj_a)
    b = sum(1 for i in ids if maj_a[i] and not maj_b[i])
    c = sum(1 for i in ids if maj_b[i] and not maj_a[i])
    rates_a = per_case_rates(runs_a)
    rates_b = per_case_rates(runs_b)
    diffs = [rates_a[i] - rates_b[i] for i in ids]
    low, high = bootstrap_mean_ci(diffs)
    # McNemar também rodada a rodada (rodada i de A contra rodada i de B), como sensibilidade.
    per_run: List[float] = []
    for run_a, run_b in zip(runs_a, runs_b):
        bb = sum(1 for i in ids if run_a[i] and not run_b[i])
        cc = sum(1 for i in ids if run_b[i] and not run_a[i])
        per_run.append(mcnemar_exact(bb, cc))
    return {
        "n_cases": len(ids),
        "only_a": b,
        "only_b": c,
        "both": sum(1 for i in ids if maj_a[i] and maj_b[i]),
        "neither": sum(1 for i in ids if not maj_a[i] and not maj_b[i]),
        "mcnemar_p": mcnemar_exact(b, c),
        "mcnemar_p_per_run": per_run,
        "diff_mean": mean(diffs),
        "diff_ci95_bootstrap": [low, high],
    }
