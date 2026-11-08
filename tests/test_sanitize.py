"""Sanitizador dos artefatos dos evals: nada da máquina (home, usuário) vai para o git."""
from __future__ import annotations

import json

import pytest

from evals import common, sanitize
from evals.sanitize import Sanitizer

# Usuário fictício, com acento e longo, como os nomes de conta do Windows.
USER = "MariaEduardaSoárezK"


@pytest.fixture
def san():
    return Sanitizer([USER])


@pytest.mark.parametrize(
    "raw,expected",
    [
        (r"C:\Users\MariaEduardaSoárezK\AppData\Local\Temp\tmpab\a.txt",
         r"<HOME>\AppData\Local\Temp\tmpab\a.txt"),
        # barras escapadas (o modelo devolve JSON dentro do texto)
        (r"C:\\Users\\MariaEduardaSoárezK\\Desktop\\x.pdf", r"<HOME>\\Desktop\\x.pdf"),
        (r"C:\\\\Users\\\\MariaEduardaSoárezK\\\\x", r"<HOME>\\\\x"),
        ("C:/Users/MariaEduardaSoárezK/AppData/Local/Temp/x/diario.md",
         "<HOME>/AppData/Local/Temp/x/diario.md"),
        # acento como escape \u00e1 (uma ou duas barras)
        (r"C:\\Users\\MariaEduardaSo\u00e1rezK\AppData", r"<HOME>\AppData"),
        (r"C:\\Users\\MariaEduardaSo\\u00e1rezK\\AppData", r"<HOME>\\AppData"),
        # nome truncado, sem acento ou reinventado pelo modelo
        (r"C:\Users\MariaEduardaSoarez\Desktop", r"<HOME>\Desktop"),
        ('"repo": "mariaeduardasoaresk"', '"repo": "<USER>"'),
        (r"from:MariaEduardaSo\u00e1rez is:unread", "from:<USER> is:unread"),
        # nome curto 8.3
        (r"C:\Users\MARIAE~1\AppData\Local\Temp", r"<HOME>\AppData\Local\Temp"),
        (r"C:\PROGRA~1\MARIAE~1\x", r"C:\PROGRA~1\<USER>\x"),
        # qualquer outra home do Windows também sai
        (r"D:\Users\outra.pessoa\notas.md", r"<HOME>\notas.md"),
        # POSIX: só a home do usuário desta máquina
        ("/home/MariaEduardaSoárezK/projeto", "<HOME>/projeto"),
        ("/Users/MariaEduardaSoárezK/projeto", "<HOME>/projeto"),
    ],
)
def test_sanitiza_variacoes(san, raw, expected):
    assert san.text(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "/home/julia/projeto",  # nome fictício usado nos datasets
        "Você está bem, Julia?",
        r"C:\Program Files\Git\bin\bash.exe",
        "evals/results/RESULTS.md",
        "o modelo Maria respondeu",
    ],
)
def test_nao_mexe_no_que_nao_e_da_maquina(san, text):
    assert san.text(text) == text


def test_nome_curto_so_casa_inteiro():
    san = Sanitizer(["ana"])
    assert san.text("/home/ana/x e banana e Ana") == "<HOME>/x e banana e <USER>"


def test_obj_percorre_dicts_listas_e_chaves(san):
    payload = {
        "records": [{"content": r"C:\Users\MariaEduardaSoárezK\a", "n": 3}],
        "C:/Users/MariaEduardaSoárezK": True,
        "nada": None,
    }
    assert san.obj(payload) == {
        "records": [{"content": r"<HOME>\a", "n": 3}],
        "<HOME>": True,
        "nada": None,
    }


def test_save_json_grava_sanitizado(tmp_path, monkeypatch, san):
    monkeypatch.setattr(sanitize, "_DEFAULT", san)
    path = common.save_json({"x": r"C:\Users\MariaEduardaSoárezK\Temp"}, "r.json", tmp_path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"x": r"<HOME>\Temp"}


def test_cli_check_e_limpeza(tmp_path, monkeypatch):
    monkeypatch.setattr(sanitize, "current_usernames", lambda: [USER])
    dirty = tmp_path / "a.json"
    dirty.write_text(json.dumps({"p": "/home/MariaEduardaSoárezK/x"}), encoding="utf-8")
    clean = tmp_path / "b.json"
    clean.write_text(json.dumps({"p": "ok"}, indent=2) + "\n", encoding="utf-8")

    assert sanitize.main(["--check", str(tmp_path)]) == 1
    assert "MariaEduarda" in dirty.read_text(encoding="utf-8")  # --check não grava
    assert sanitize.main([str(tmp_path)]) == 0
    assert json.loads(dirty.read_text(encoding="utf-8")) == {"p": "<HOME>/x"}
    assert sanitize.main(["--check", str(tmp_path)]) == 0
