"""Lógica de pontuação das avaliações (sem Ollama) e validade dos datasets."""
from __future__ import annotations

import pytest

from evals import common, eval_rag, eval_tools
from jarvis.rag import RagIndex
from jarvis.reminders import ReminderService
from jarvis.tools import build_registry

# ---------------------------------------------------------------- matchers


@pytest.mark.parametrize(
    "actual,matcher,ok",
    [
        ("add", "add", True),
        ("ADD ", "add", True),
        ("café", "cafe", True),
        (3, 3, True),
        ("3", 3, True),
        (3.0, 3, True),
        (["status", "--short"], {"contains": "status"}, True),
        ("pip install requests", {"contains_all": ["pip", "requests"]}, True),
        ("pip install flask", {"contains_all": ["pip", "requests"]}, False),
        ("mail_search", {"any_of": ["mail_list", "mail_search"]}, True),
        ("mail_send", {"any_of": ["mail_list", "mail_search"]}, False),
        ("amanhã 09:00", {"regex": "(^|[^0-9])0?9(:|h)00"}, True),
        ("2026-11-09T09:00:00", {"regex": "(^|[^0-9])0?9(:|h)00|t09:00"}, True),
        ("19:00", {"regex": "(^|[^0-9])0?9(:|h)00"}, False),
        (None, "x", False),
        ("Bot Não Responde Áudio", {"contains": "nao responde audio"}, True),
    ],
)
def test_match_value(actual, matcher, ok):
    assert common.match_value(actual, matcher) is ok


def test_match_value_matcher_desconhecido():
    with pytest.raises(ValueError):
        common.match_value("x", {"igual": "x"})


def test_match_args():
    ok, failed = common.match_args({"action": "add", "when": "10m"}, {"action": "add", "text": "x"})
    assert not ok and failed == ["text"]
    assert common.match_args({}, None) == (True, [])


# ---------------------------------------------------------------- decisão


def test_decide_nativo():
    message = {"tool_calls": [{"function": {"name": "git", "arguments": {"args": ["status"]}}}]}
    assert eval_tools.decide(message, "native") == (
        "git", {"args": ["status"]}, {"invalid_json": False, "leaked": False, "n_calls": 1}
    )


def test_decide_nativo_arguments_invalidos():
    message = {"tool_calls": [{"function": {"name": "git", "arguments": "{quebrado"}}]}
    name, args, flags = eval_tools.decide(message, "native")
    assert name == "git" and args == {} and flags["invalid_json"]


def test_decide_nativo_json_no_texto_conta_como_vazamento():
    message = {"content": '{"name": "git", "arguments": {"args": ["log"]}}'}
    name, _, flags = eval_tools.decide(message, "native")
    assert name is None and flags["leaked"] and not flags["invalid_json"]


def test_decide_modo_json():
    ok = {"content": '{"tool": "reminder", "args": {"action": "list"}}'}
    assert eval_tools.decide(ok, "json")[0:2] == ("reminder", {"action": "list"})
    broken = {"content": '{"tool": "reminder", "args": {"action": "list"}'}
    name, _, flags = eval_tools.decide(broken, "json")
    assert name is None and flags["invalid_json"]
    text = {"content": "Olá! Como posso ajudar?"}
    name, _, flags = eval_tools.decide(text, "json")
    assert name is None and not flags["invalid_json"]


# ---------------------------------------------------------------- métricas


def _record(case, predicted, args=None, latency=1.0, **extra):
    record = {
        "id": case["id"], "category": case.get("category", "direto"),
        "expected_tool": case.get("expected_tool"), "predicted": predicted,
        "args": args or {}, "error": None, "latency_s": latency, "invalid_json": False,
        "leaked": False, "unknown_tool": False,
        "timings": {"eval_count": 10, "eval_s": 0.5, "prompt_eval_count": 100},
    }
    record.update(extra)
    record.update(eval_tools.score_case(case, predicted, record["args"]))
    return record


def test_score_e_compute_metrics():
    c1 = {"id": "a", "expected_tool": "git", "expected_args": {"args": {"contains": "status"}}}
    c2 = {"id": "b", "expected_tool": None, "category": "sem_ferramenta"}
    c3 = {"id": "c", "expected_tool": "knowledge_search", "acceptable_tools": ["obsidian"],
          "category": "ambiguo"}
    c4 = {"id": "d", "expected_tool": "reminder", "expected_args": {"action": "add"}}
    records = [
        _record(c1, "git", {"args": ["status"]}, latency=1.0),
        _record(c2, None, latency=2.0),
        _record(c3, "obsidian", latency=3.0),
        _record(c4, "google", latency=4.0, invalid_json=True),
    ]
    m = eval_tools.compute_metrics(records)
    assert m["tool_hits"] == 3 and m["tool_accuracy"] == 0.75
    assert m["args_n"] == 2 and m["args_hits"] == 1 and m["args_accuracy"] == 0.5
    assert m["args_n_given_tool"] == 1 and m["args_accuracy_given_tool"] == 1.0
    assert m["invalid_json"] == 1 and m["invalid_json_rate"] == 0.25
    assert m["latency_s"]["p50"] == 2.5
    assert m["tokens_per_s"] == pytest.approx(20.0)
    assert m["by_category"]["ambiguo"]["accuracy"] == 1.0
    assert m["errors_by_pair"] == {"reminder -> google": 1}


def test_rag_rank_e_metricas():
    question = {"expected_doc": "a.md", "expected_snippet": "R$ 2.350,00"}
    results = [
        {"path": "/x/b.md", "text": "R$ 2.350,00"},  # doc errado
        {"path": "/x/a.md", "text": "outro trecho"},  # snippet errado
        {"path": "/x/a.md", "text": "aluguel de\nR$  2.350,00 por mês"},
    ]
    assert eval_rag.first_hit_rank(results, question) == 3
    records = [
        {"rank": 1, "top_docs": ["a.md"], "expected_doc": "a.md", "latency_s": 0.1},
        {"rank": 3, "top_docs": ["b.md"], "expected_doc": "a.md", "latency_s": 0.2},
        {"rank": None, "top_docs": ["b.md"], "expected_doc": "a.md", "latency_s": 0.3},
        {"rank": 2, "top_docs": ["b.md"], "expected_doc": "a.md", "latency_s": 0.4},
    ]
    m = eval_rag.compute_metrics(records)
    assert m["recall_at_1"] == 0.25 and m["recall_at_5"] == 0.75
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)


# --------------------------------------------------------- datasets válidos


def test_dataset_de_ferramentas_cobre_todo_o_registro(settings):
    cases = common.load_jsonl(eval_tools.DATA_PATH)
    assert len(cases) >= 50
    rag, reminders = RagIndex(settings), ReminderService(settings)
    try:
        names = set(build_registry(settings, reminders, rag).names())
    finally:
        rag.close()
        reminders.close()
    expected = {c["expected_tool"] for c in cases if c["expected_tool"]}
    assert expected == names, "faltam casos para: {}".format(sorted(names - expected))
    assert any(c["expected_tool"] is None for c in cases)
    assert any(c.get("category") == "ambiguo" for c in cases)
    for case in cases:
        assert case["category"] in {"direto", "ambiguo", "sem_ferramenta"}
        for name in case.get("acceptable_tools") or []:
            assert name in names
        for matcher in (case.get("expected_args") or {}).values():
            common.match_value("x", matcher)  # matcher bem formado


def test_dataset_de_rag_aponta_para_trechos_existentes():
    from jarvis.documents import extract_text
    from jarvis.rag import chunk_text

    questions = common.load_jsonl(eval_rag.DATA_PATH)
    assert len(questions) >= 30
    chunks = {
        path.name: [common.normalize(c) for c in chunk_text(extract_text(path))]
        for path in common.FIXTURES_DIR.iterdir()
        if path.is_file()
    }
    for question in questions:
        snippet = common.normalize(question["expected_snippet"])
        assert any(snippet in chunk for chunk in chunks[question["expected_doc"]]), question["id"]


def test_rag_divisao_dev_teste_fixa():
    assert eval_rag.split_of("rag-001") == "dev"
    assert eval_rag.split_of("rag-036") == "test"
    questions = common.load_jsonl(eval_rag.DATA_PATH)
    splits = [eval_rag.split_of(q["id"]) for q in questions]
    assert splits.count("dev") == 18 and splits.count("test") == 18
    records = [
        {"id": "rag-001", "rank": 1, "top_docs": ["a.md"], "expected_doc": "a.md", "latency_s": 0.1},
        {"id": "rag-002", "rank": None, "top_docs": [], "expected_doc": "a.md", "latency_s": 0.1},
    ]
    by_split = eval_rag.compute_split_metrics(records)
    assert by_split["dev"]["mrr"] == 1.0 and by_split["test"]["mrr"] == 0.0


def test_rag_rotulo_e_nome_do_arquivo_por_configuracao():
    base = {"embed_model": "nomic-embed-text", "meta": {"timestamp": "2026-11-08T02:00:00-03:00"}}
    hybrid = dict(
        base, config={"mode": "hybrid", "prefixes": True, "rrf_k": 60, "vector_backend": "sqlite-vec"}
    )
    vector = dict(base, config={"mode": "vector", "prefixes": False, "rrf_k": 60, "vector_backend": "numpy"})
    bm25 = dict(base, config={"mode": "bm25", "prefixes": True, "rrf_k": 60, "vector_backend": "sqlite-vec"})
    assert eval_rag.config_label(hybrid) == "hybrid + prefixos (RRF k=60)"
    assert eval_rag.config_label(vector) == "vector numpy sem prefixos"
    assert eval_rag.config_label(bm25) == "bm25"
    assert eval_rag.result_filename(hybrid) == "20261108T020000_rag_nomic-embed-text_hybrid-prefixos-k60.json"
    assert eval_rag.result_filename(vector) == "20261108T020000_rag_nomic-embed-text_vector-numpy.json"
    assert eval_rag.result_filename(base) == "20261108T020000_rag_nomic-embed-text.json"  # formato antigo


def test_paraphrases_mesmos_trechos_e_baixa_sobreposicao():
    from evals import lexical_overlap

    chunks = lexical_overlap.load_chunks()
    original = common.load_jsonl(eval_rag.DATA_PATH)
    paraphrase = common.load_jsonl(eval_rag.PARAPHRASE_PATH)
    assert len(paraphrase) == len(original) == 36
    # mesmos documentos e trechos esperados, na mesma ordem (e a mesma paridade de id)
    for orig, para in zip(original, paraphrase):
        assert para["id"][-3:] == orig["id"][-3:]
        assert (para["expected_doc"], para["expected_snippet"]) == (
            orig["expected_doc"], orig["expected_snippet"])
        assert para["question"] != orig["question"]
    overlap_orig = lexical_overlap.summarize(lexical_overlap.overlap_records(original, chunks))
    overlap_para = lexical_overlap.summarize(lexical_overlap.overlap_records(paraphrase, chunks))
    assert overlap_orig["mean_overlap"] > 0.6
    assert overlap_para["mean_overlap"] < 0.15


def test_tokens_de_conteudo_sao_os_do_bm25():
    from evals import lexical_overlap

    assert lexical_overlap.content_tokens("Qual o valor do aluguel?") == ["valor", "aluguel"]
    assert lexical_overlap.content_tokens("o que é") == []


def test_rag_nome_do_arquivo_marca_as_parafrases():
    para = {"meta": {"timestamp": "2026-11-08T05:00:00-03:00"}, "embed_model": "nomic-embed-text",
            "dataset": "evals/data/rag_paraphrase.jsonl",
            "config": {"mode": "bm25", "prefixes": False, "rrf_k": 60, "vector_backend": "sqlite-vec"}}
    assert eval_rag.result_filename(para) == "20261108T050000_rag_nomic-embed-text_parafrases_bm25.json"
    assert set(eval_rag.DATASETS) == {"original", "paraphrase"}
