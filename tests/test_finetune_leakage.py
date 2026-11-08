"""finetune/leakage.py: o eval de ferramentas nunca pode vazar para o treino."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune import leakage

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_PATH = BASE_DIR / "evals" / "data" / "tool_selection.jsonl"
DATA_DIR = BASE_DIR / "finetune" / "data"

CASES = [
    {"id": "e1", "prompt": "me lembra de tomar remédio daqui a 10 minutos"},
    {"id": "e2", "prompt": "lê o PDF contrato_aluguel.pdf e me diz o valor do aluguel"},
    {"id": "e3", "prompt": "resume esse artigo pra mim: https://exemplo.com.br/blog/produtividade"},
    {"id": "e4", "prompt": "manda um e-mail para ana@empresa.com"},
    {"id": "e5", "prompt": "lista as issues abertas do repositório julia/jarvis no GitHub"},
    {"id": "e6", "prompt": "abre o arquivo notas/ideias.txt"},
]


@pytest.fixture(scope="module")
def index():
    return leakage.EvalIndex.from_cases(CASES)


def _example(text):
    return {"id": "x", "template": "t.00", "messages": [
        {"role": "system", "content": "S"}, {"role": "user", "content": text},
        {"role": "assistant", "content": "ok"}]}


def test_normalize():
    assert leakage.normalize("  Me LEMBRA, de tomar   remédio!! ") == "me lembra de tomar remedio"


def test_jaccard_limites():
    a = leakage.char_ngrams("abc def")
    assert leakage.jaccard(a, a) == 1.0
    assert leakage.jaccard(a, leakage.char_ngrams("xyz")) == 0.0


def test_duplicata_exata_depois_de_normalizar(index):
    verdict = leakage.check_text("Me lembra de tomar REMEDIO, daqui a 10 minutos!", index)
    assert verdict.leaked and verdict.reasons[0] == "exato" and verdict.best_eval_id == "e1"


def test_quase_duplicata_acima_do_limiar(index):
    verdict = leakage.check_text("me lembra de tomar o remédio daqui a 15 minutos", index)
    assert verdict.best_similarity >= leakage.DEFAULT_THRESHOLD
    assert "quase_duplicata" in verdict.reasons


def test_pedido_diferente_passa(index):
    verdict = leakage.check_text("quanto entrou no Stripe nos últimos 90 dias?", index)
    assert not verdict.leaked
    assert verdict.best_similarity < leakage.DEFAULT_THRESHOLD


@pytest.mark.parametrize("text,entity", [
    ("resume o contrato_aluguel.pdf rapidinho", "contrato_aluguel.pdf"),
    ("abre o ideias.txt", "ideias.txt"),  # nome sem a pasta também conta
    ("lê https://exemplo.com.br/outra-coisa", "exemplo.com.br"),  # domínio da URL
    ("escreve para ana@empresa.com", "ana@empresa.com"),
    ("prs abertos em julia/jarvis", "julia/jarvis"),
])
def test_entidade_do_eval_e_vazamento(index, text, entity):
    verdict = leakage.check_text(text, index)
    assert "entidade" in verdict.reasons
    assert entity in verdict.entities


def test_entidade_exige_fronteira(index):
    # "meu_contrato_aluguel.pdf" não é o arquivo do eval
    assert "entidade" not in leakage.check_text("abre meu_contrato_aluguel.pdf", index).reasons


def test_extract_entities():
    found = leakage.extract_entities("lê notas/x.txt, https://a.com/b e fala com c@d.org sobre org/repo")
    assert {"notas/x.txt", "x.txt", "https://a.com/b", "a.com", "c@d.org", "org/repo"} <= found


def test_filter_examples_relatorio(index):
    examples = [
        _example("me lembra de tomar remédio daqui a 10 minutos"),  # exato
        _example("me lembra de tomar o remédio daqui a 15 minutos"),  # quase
        _example("abre o contrato_aluguel.pdf"),  # entidade
        _example("qual a previsão do tempo em Recife?"),  # limpo
    ]
    result = leakage.filter_examples(examples, index)
    report = result["report"]
    assert report["removed"] == 3 and report["kept"] == 1
    assert report["removed_by_reason"]["exato"] == 1
    assert report["removed_by_reason"]["quase_duplicata"] >= 1  # o 3º também pode ser parecido
    assert report["removed_by_reason"]["entidade"] == 1
    assert len(report["removed_examples"]) == 3


def test_so_o_texto_da_usuaria_conta(index):
    example = _example("bom dia")
    example["messages"][2]["content"] = "me lembra de tomar remédio daqui a 10 minutos"
    assert leakage.filter_examples([example], index)["report"]["removed"] == 0


def test_limiar_acima_da_similaridade_entre_casos_do_eval():
    """Calibração: casos distintos do eval ficam bem abaixo do limiar.

    Se alguém trocar o eval ou o limiar e essa margem sumir, o teste avisa.
    """
    index = leakage.EvalIndex.from_jsonl(EVAL_PATH)
    stats = leakage.eval_self_similarity(index)
    assert stats["max"] < leakage.DEFAULT_THRESHOLD
    assert leakage.DEFAULT_THRESHOLD == pytest.approx(round(1.5 * stats["max"] / 0.05) * 0.05)


def _load(name):
    with open(str(DATA_DIR / name), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_dataset_versionado_nao_vaza_o_eval():
    index = leakage.EvalIndex.from_jsonl(EVAL_PATH)
    rows = _load("train.jsonl") + _load("val.jsonl")
    assert rows, "rode finetune/generate_dataset.py"
    result = leakage.filter_examples(rows, index)
    assert result["report"]["removed"] == 0, result["report"]["removed_examples"][:3]
