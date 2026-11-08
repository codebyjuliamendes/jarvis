"""parse_when dos lembretes (jarvis/tools/reminders.py)."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import pytz

from jarvis.tools.reminders import parse_when

TZ = "America/Sao_Paulo"


def _now():
    return datetime.now(pytz.timezone(TZ))


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("30s", 30),
        ("45 seg", 45),
        ("10m", 600),
        ("10min", 600),
        ("5 minutos", 300),
        ("1 minuto", 60),
        ("2h", 7200),
        ("3 horas", 10800),
        ("1d", 86400),
        ("2 dias", 172800),
        ("10M", 600),
    ],
)
def test_relativos(text, seconds):
    before = _now()
    result = parse_when(text, TZ)
    delta = (result - before).total_seconds()
    assert seconds - 2 <= delta <= seconds + 5


def test_hoje_e_amanha():
    now = _now()
    today = parse_when("hoje 18:30", TZ)
    assert (today.hour, today.minute, today.second) == (18, 30, 0)
    assert today.date() == now.date()
    for text in ("amanhã 09:00", "amanha 09:00", "Amanhã 09:00"):
        tomorrow = parse_when(text, TZ)
        assert (tomorrow.hour, tomorrow.minute) == (9, 0)
        assert tomorrow.date() == (now + timedelta(days=1)).date()


def test_hoje_sem_hora_valida():
    with pytest.raises(ValueError, match="HH:MM"):
        parse_when("hoje de tarde", TZ)


def test_so_hora_vai_para_o_futuro():
    now = _now()
    result = parse_when("{:02d}:{:02d}".format(now.hour, now.minute), TZ)
    # mesmo minuto já passou (segundos zerados) -> amanhã
    assert result > now
    assert result - now <= timedelta(days=1)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2030-01-02 18:30", (2030, 1, 2, 18, 30)),
        ("02/01/2030 18:30", (2030, 1, 2, 18, 30)),
        ("2030-01-02 18:30:15", (2030, 1, 2, 18, 30)),
        ("02/01/2030", (2030, 1, 2, 0, 0)),
        ("2030-01-02", (2030, 1, 2, 0, 0)),
    ],
)
def test_datas_absolutas(text, expected):
    result = parse_when(text, TZ)
    assert (result.year, result.month, result.day, result.hour, result.minute) == expected
    assert result.utcoffset() is not None
    assert result.tzinfo.zone == TZ  # type: ignore[union-attr]


def test_iso_com_fuso():
    result = parse_when("2030-05-01T12:00:00Z", TZ)
    assert result.utcoffset() == timedelta(0)
    assert result.hour == 12


def test_iso_sem_fuso_recebe_o_fuso_configurado():
    result = parse_when("2030-05-01T12:00:00", TZ)
    assert result.tzinfo.zone == TZ  # type: ignore[union-attr]


def test_sem_fuso_configurado_usa_o_local():
    result = parse_when("2030-01-02 18:30", None)
    assert result.utcoffset() is not None


@pytest.mark.parametrize("text", ["", "   ", "quando der vontade", "semana que vem", "25:99"])
def test_invalidos(text):
    with pytest.raises(ValueError):
        parse_when(text, TZ)
