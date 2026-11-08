"""finetune/generate_dataset.py e o dataset versionado em finetune/data/."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from finetune import generate_dataset as gd
from finetune import templates
from finetune.chat_format import render_example

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "finetune" / "data"


@pytest.fixture(scope="module")
def real():
    return gd.load_real_context()


def _load(name):
    with open(str(DATA_DIR / name), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def committed():
    return _load("train.jsonl"), _load("val.jsonl")


# ------------------------------------------------------------- validação


SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["add", "list"]},
        "id": {"type": "integer"},
        "args": {"type": "array", "items": {"type": "string"}},
        "force": {"type": "boolean"},
    },
    "required": ["action"],
}


def test_validate_args_ok():
    assert gd.validate_args({"action": "add", "id": 3, "args": ["a"], "force": True}, SCHEMA) == []


@pytest.mark.parametrize("args,fragment", [
    ({}, "obrigatório"),
    ({"action": "apagar"}, "enum"),
    ({"action": "add", "id": "3"}, "integer"),
    ({"action": "add", "id": True}, "integer"),
    ({"action": "add", "args": [1]}, "itens"),
    ({"action": "add", "extra": 1}, "desconhecido"),
])
def test_validate_args_erros(args, fragment):
    errors = gd.validate_args(args, SCHEMA)
    assert errors and any(fragment in e for e in errors)


# --------------------------------------------------------------- geração


def test_toda_ferramenta_do_registro_tem_modelos(real):
    _, schemas = real
    gd.check_coverage(schemas)  # levanta SystemExit se desalinhado
    names = {s["function"]["name"] for s in schemas}
    assert names == {i.tool for i in templates.TOOL_INTENTS}


def test_system_prompt_real_com_data_e_pasta(real):
    sys_template, _ = real
    from datetime import datetime

    text = gd.system_prompt(sys_template, datetime(2026, 3, 4, 9, 5), "/home/julia")
    assert "Data/hora: 04/03/2026 09:05" in text
    assert "Pasta de trabalho: /home/julia" in text
    assert text.startswith("Você é o Jarvis")


def test_geracao_deterministica_e_valida(real):
    sys_template, schemas = real
    first, _ = gd.generate(7, 240, 0.22, 0.05, 0.2, sys_template, schemas)
    second, _ = gd.generate(7, 240, 0.22, 0.05, 0.2, sys_template, schemas)
    assert first == second
    other, _ = gd.generate(8, 240, 0.22, 0.05, 0.2, sys_template, schemas)
    assert first != other
    kinds = Counter(e["kind"] for e in first)
    assert kinds["no_tool"] >= 40 and kinds["continuation"] >= 8


def test_frases_do_val_nunca_no_treino():
    splits = gd.split_phrasings(templates.TOOL_INTENTS + templates.NO_TOOL_INTENTS, 0.2, 42)
    assert set(splits.values()) == {"train", "val"}
    for intent in templates.TOOL_INTENTS + templates.NO_TOOL_INTENTS:
        ids = ["{}.{:02d}".format(intent.id, i) for i in range(len(intent.phrasings))]
        assert any(splits[i] == "train" for i in ids), intent.id


def test_variacao_de_superficie_nao_mexe_nas_entidades():
    import random

    rng = random.Random(0)
    for _ in range(200):
        text = templates.vary_surface(rng, "lê o {file} pra mim").format(file="Músicas/ação.txt")
        assert "Músicas/ação.txt" in text


# ------------------------------------------------------- dataset versionado


def test_dataset_versionado_tem_tamanho_e_composicao(committed):
    train, val = committed
    rows = train + val
    assert 1000 <= len(rows) <= 3000
    kinds = Counter(r["kind"] for r in rows)
    assert 0.15 <= kinds["no_tool"] / len(rows) <= 0.30
    tools = Counter(r["tool"] for r in rows if r["kind"] == "tool")
    assert len(tools) == 25 and min(tools.values()) >= 40


def test_dataset_versionado_tools_json_igual_ao_registro(real):
    """Se um schema mudar no código, o dataset precisa ser regerado."""
    _, schemas = real
    with open(str(DATA_DIR / "tools.json"), encoding="utf-8") as handle:
        assert json.load(handle) == schemas


def test_dataset_versionado_chamadas_validas(committed):
    train, val = committed
    with open(str(DATA_DIR / "tools.json"), encoding="utf-8") as handle:
        schemas = {s["function"]["name"]: s["function"]["parameters"] for s in json.load(handle)}
    for row in train + val:
        for message in row["messages"]:
            for call in message.get("tool_calls") or []:
                function = call["function"]
                assert gd.validate_args(function["arguments"], schemas[function["name"]]) == [], row["id"]


def test_dataset_versionado_split_por_modelo_de_frase(committed):
    train, val = committed
    assert not ({r["template"] for r in train} & {r["template"] for r in val})
    assert all(r["split"] == "train" for r in train) and all(r["split"] == "val" for r in val)
    ids = [r["id"] for r in train + val]
    assert len(ids) == len(set(ids))


def test_dataset_versionado_renderiza(committed):
    train, _ = committed
    with open(str(DATA_DIR / "tools.json"), encoding="utf-8") as handle:
        schemas = json.load(handle)
    for row in train[:50]:
        prompt, completion = render_example(row, schemas)
        assert prompt.startswith("<|im_start|>system\nVocê é o Jarvis")
        assert prompt.endswith("<|im_start|>assistant\n")
        assert completion.endswith("<|im_end|>")
        if row["kind"] == "tool":
            assert completion.startswith("<tool_call>\n{\"name\": \"" + row["tool"] + "\"")


def test_dataset_versionado_e_leve():
    total = sum((DATA_DIR / n).stat().st_size for n in ("train.jsonl", "val.jsonl", "tools.json"))
    assert total < 5 * 1024 * 1024
