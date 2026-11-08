"""Estatística das avaliações (evals/stats.py) e agregação de rodadas repetidas, sem Ollama."""
from __future__ import annotations

import math

import pytest

from evals import compare_models, eval_tools, report, stats, tracking


def test_media_e_desvio_amostral():
    assert stats.mean([0.8, 0.82, 0.84]) == pytest.approx(0.82)
    assert stats.stdev([0.8, 0.82, 0.84]) == pytest.approx(0.02)
    assert stats.stdev([0.5]) is None
    assert stats.mean([]) is None


def test_wilson_valores_conhecidos():
    # 64/78: conferido à mão com a fórmula (centro deslocado para 0,5, não simétrico)
    low, high = stats.wilson_interval(64, 78)
    assert low == pytest.approx(0.72098, abs=1e-4)
    assert high == pytest.approx(0.88996, abs=1e-4)
    # extremos não saem de [0, 1] e não colapsam em zero de largura
    low, high = stats.wilson_interval(0, 10)
    assert low == 0.0 and 0.25 < high < 0.35
    low, high = stats.wilson_interval(10, 10)
    assert high == pytest.approx(1.0) and 0.65 < low < 0.75
    with pytest.raises(ValueError):
        stats.wilson_interval(1, 0)


def test_mcnemar_exato():
    # b=10, c=2: 2 * P(X <= 2), X ~ Bin(12, 1/2) = 2 * 79/4096
    assert stats.mcnemar_exact(10, 2) == pytest.approx(2 * 79 / 4096)
    assert stats.mcnemar_exact(2, 10) == stats.mcnemar_exact(10, 2)
    assert stats.mcnemar_exact(5, 5) == 1.0
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    # 6 a 0 não é significativo a 1% nem 5 a 0 a 5%
    assert stats.mcnemar_exact(0, 5) == pytest.approx(0.0625)


def test_bootstrap_deterministico_e_contem_a_media():
    values = [1.0] * 64 + [0.0] * 14
    first = stats.bootstrap_mean_ci(values)
    assert first == stats.bootstrap_mean_ci(values)  # seed fixa
    low, high = first
    assert low < 64 / 78 < high
    # perto do Wilson (n=78): diferença de poucos pontos
    w_low, w_high = stats.wilson_interval(64, 78)
    assert abs(low - w_low) < 0.03 and abs(high - w_high) < 0.03
    assert stats.bootstrap_mean_ci([0.5] * 10) == (0.5, 0.5)


def test_percentil_interpola_como_numpy():
    assert stats._percentile([0.0, 1.0, 2.0, 3.0], 0.5) == pytest.approx(1.5)
    assert stats._percentile([0.0, 10.0], 0.025) == pytest.approx(0.25)


def _runs(*patterns):
    """Cada padrão é uma string de 0/1, um caractere por caso."""
    return [{"c{}".format(i): ch == "1" for i, ch in enumerate(p)} for p in patterns]


def test_resumo_de_rodadas():
    runs = _runs("1110", "1100", "1110")
    summary = stats.accuracy_summary(runs)
    assert summary["hits_per_run"] == [3, 2, 3]
    assert summary["mean"] == pytest.approx(8 / 12)
    assert summary["stdev"] == pytest.approx(math.sqrt(((0.75 - 2 / 3) ** 2 * 2 + (0.5 - 2 / 3) ** 2) / 2))
    assert summary["unstable_cases"] == 1  # c2 acertou em 2 de 3
    assert summary["majority_hits"] == 3
    low, high = summary["ci95_bootstrap"]
    assert 0 <= low <= summary["mean"] <= high <= 1


def test_rodadas_com_casos_diferentes_falham():
    with pytest.raises(ValueError):
        stats.per_case_rates([{"a": True}, {"b": True}])


def test_comparacao_pareada():
    a = _runs("1111111111", "1111111111", "1111111110")
    b = _runs("1000000000", "1000000000", "1000000001")
    comp = stats.paired_comparison(a, b)
    assert comp["only_a"] == 9 and comp["only_b"] == 0  # maioria: c9 = A sim (2/3), B não (1/3)
    assert comp["both"] == 1 and comp["neither"] == 0
    assert comp["mcnemar_p"] == pytest.approx(2 / 512)
    assert len(comp["mcnemar_p_per_run"]) == 3
    assert comp["diff_mean"] > 0.7
    assert comp["diff_ci95_bootstrap"][0] > 0


def _fake_result(model, hits_pattern, args_pattern=None, mode="native"):
    records = []
    for index, ch in enumerate(hits_pattern):
        args_ok = None
        if args_pattern and args_pattern[index] != "-":
            args_ok = args_pattern[index] == "1"
        records.append({"id": "ts-{:03d}".format(index), "tool_ok": ch == "1", "args_ok": args_ok})
    hits = hits_pattern.count("1")
    return {
        "kind": "tool_selection",
        "model": model,
        "mode": mode,
        "effective_mode": mode,
        "dataset": "evals/data/tool_selection.jsonl",
        "dataset_sha256": "abc",
        "n_tools": 25,
        "seed": 42,
        "num_ctx": 8192,
        "warmup_s": 1.0,
        "meta": {"timestamp": "2026-11-08T04:00:00-03:00", "git_commit": "x", "platform": "p",
                 "gpu": "g", "ollama": {"version": "0.40.14"}},
        "metrics": {"n": len(hits_pattern), "tool_hits": hits, "invalid_json": 0,
                    "latency_s": {"p50": 0.5}, "tokens_per_s": 70.0},
        "records": records,
    }


def test_agrega_repeticoes_e_gera_tabela():
    runs_a = [_fake_result("qwen2.5:3b", p, "11-0") for p in ("1110", "1111", "1110")]
    runs_b = [_fake_result("llama3.2:3b", p, "10-0") for p in ("1000", "1100", "1000")]
    rep_a = eval_tools.aggregate_repeats(runs_a)
    rep_b = eval_tools.aggregate_repeats(runs_b)
    assert rep_a["repeats"] == 3 and rep_a["kind"] == "tool_selection_repeated"
    assert rep_a["stats"]["tool"]["hits_per_run"] == [3, 4, 3]
    assert rep_a["stats"]["args"]["n_cases"] == 3  # ts-002 não tem args esperados
    assert "meta" not in rep_a["runs"][0] and "records" in rep_a["runs"][0]
    assert eval_tools.repeated_filename(rep_a) == "20261108T040000_tools_qwen2.5-3b_native_x3.json"

    comps = compare_models.pairwise_comparisons([rep_a, rep_b])
    assert len(comps) == 1 and comps[0]["model_a"] == "qwen2.5:3b"
    assert comps[0]["alpha_bonferroni"] == 0.05

    metrics = tracking.repeated_tool_metrics(rep_a)
    assert metrics["tool_accuracy_run2"] == 1.0
    assert metrics["tool_accuracy_ci95_low"] <= metrics["tool_accuracy_mean"]
    params = tracking.repeated_tool_params(rep_a)
    assert params["repeats"] == 3 and params["n_examples"] == 4

    markdown = report.render_repeats_markdown({
        "repeats": 3, "meta": rep_a["meta"], "repeated_results": [rep_a, rep_b],
        "comparisons": comps, "failures": [], "files": ["a.json"],
    })
    assert "3 / 4 / 3 (de 4)" in markdown
    assert "McNemar" in markdown and "`qwen2.5:3b` × `llama3.2:3b`" in markdown
    assert "14,4 p.p." in markdown and "p,p" not in markdown


def test_agrega_recusa_configuracoes_diferentes():
    with pytest.raises(ValueError):
        eval_tools.aggregate_repeats([_fake_result("a", "1"), _fake_result("b", "1")])
    with pytest.raises(ValueError):
        eval_tools.aggregate_repeats([])


def test_compare_from_json_refaz_a_tabela_sem_ollama(tmp_path, monkeypatch):
    import json

    from evals import common

    reps = []
    runs = {"qwen2.5:3b": ("1110", "1111", "1110"), "llama3.2:3b": ("1000", "1100", "1000")}
    for model, patterns in runs.items():
        rep = eval_tools.aggregate_repeats([_fake_result(model, p, "11-0") for p in patterns])
        path = tmp_path / eval_tools.repeated_filename(rep).replace("040000", model[:4])
        path.write_text(json.dumps(rep), encoding="utf-8")
        reps.append(str(path))
    saved = []

    def fake_save(payload, name, out_dir=None):
        saved.append(name)
        return common.RESULTS_DIR / name  # não grava nada em results/

    monkeypatch.setattr(common, "save_json", fake_save)
    meta = json.loads(open(reps[0], encoding="utf-8").read())["meta"]
    monkeypatch.setattr(common, "run_metadata", lambda url: meta)
    md = tmp_path / "rep.md"
    base_args = ["x", "--no-mlflow", "--markdown", str(md)]
    code = compare_models.main(base_args + ["--repeats", "3", "--from-json"] + reps)
    assert code == 0
    text = md.read_text(encoding="utf-8")
    assert "refeita com `--from-json`" in text and "`qwen2.5:3b` × `llama3.2:3b`" in text
    assert saved and saved[0].endswith("_compare_x3.json")
    with pytest.raises(SystemExit):
        compare_models.main(base_args + ["--repeats", "5", "--from-json"] + reps)
