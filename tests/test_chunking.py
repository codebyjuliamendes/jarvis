"""chunk_text do RAG."""
from __future__ import annotations

from jarvis.rag import chunk_text


def test_texto_vazio():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_trechos_curtos_demais_sao_descartados():
    assert chunk_text("curto") == []
    assert chunk_text("x" * 19) == []
    assert chunk_text("x" * 20) == ["x" * 20]


def test_paragrafos_pequenos_sao_agrupados():
    text = "Primeiro parágrafo com texto.\n\nSegundo parágrafo com texto."
    assert chunk_text(text, size=200) == [
        "Primeiro parágrafo com texto.\n\nSegundo parágrafo com texto."
    ]


def test_paragrafos_que_estouram_o_tamanho_viram_trechos_separados():
    a = "a" * 60
    b = "b" * 60
    chunks = chunk_text(a + "\n\n" + b, size=100, overlap=10)
    assert chunks == [a, b]


def test_paragrafo_longo_e_quebrado_com_sobreposicao():
    text = "".join(chr(ord("a") + (i % 26)) for i in range(250))
    chunks = chunk_text(text, size=100, overlap=20)
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert chunks[0] == text[:100]
    assert chunks[1] == text[80:180]  # recomeça 20 caracteres antes
    assert chunks[1][:20] == chunks[0][-20:]
    joined = chunks[0] + "".join(chunk[20:] for chunk in chunks[1:])
    assert joined == text


def test_normaliza_crlf_e_nulos():
    text = "linha um com texto\r\n\r\nlinha dois com texto\x00fim"
    chunks = chunk_text(text, size=500)
    assert chunks == ["linha um com texto\n\nlinha dois com texto fim"]


def test_tamanho_padrao_respeitado():
    paragraph = "palavra " * 100  # ~800 caracteres
    text = "\n\n".join([paragraph] * 5)
    chunks = chunk_text(text)
    assert len(chunks) >= 3
    assert all(len(chunk) <= 1200 for chunk in chunks)
