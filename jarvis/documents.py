from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_OCR_LANG: Optional[str] = None


class ExtractionError(Exception):
    """Falha ao extrair texto de um documento."""


def _ocr_lang() -> str:
    global _OCR_LANG
    if _OCR_LANG is not None:
        return _OCR_LANG
    try:
        import pytesseract

        langs = set(pytesseract.get_languages(config=""))
    except Exception:  # noqa: BLE001 - OCR é opcional
        langs = set()
    _OCR_LANG = "por+eng" if "por" in langs else "eng"
    return _OCR_LANG


def ocr_image(path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ExtractionError(
            "OCR indisponível. Instale pytesseract e Pillow."
        ) from exc
    try:
        with Image.open(str(path)) as image:
            return pytesseract.image_to_string(image, lang=_ocr_lang()).strip()
    except Exception as exc:  # noqa: BLE001 - erro vira texto para o modelo
        raise ExtractionError("Falha no OCR de {}: {}".format(path, exc)) from exc


def _open_pdf(path: Path):
    try:
        import fitz
    except ImportError as exc:
        raise ExtractionError(
            "Leitura de PDF indisponível. Instale PyMuPDF."
        ) from exc
    return fitz.open(str(path))


def extract_pdf(
    path: Path,
    max_pages: int = 50,
    max_chars: int = 30000,
    ocr: bool = True,
    max_ocr_pages: int = 10,
) -> str:
    doc = _open_pdf(path)
    parts: List[str] = []
    total = 0
    ocr_used = 0
    try:
        for index, page in enumerate(doc):
            if index >= max_pages:
                parts.append("\n[... PDF truncado em {} páginas]".format(max_pages))
                break
            text = (page.get_text("text") or "").strip()
            if not text and ocr and ocr_used < max_ocr_pages:
                try:
                    import fitz
                    from PIL import Image

                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    image = Image.open(io.BytesIO(pix.tobytes("png")))
                    text = ocr_image_bytes(image)
                    ocr_used += 1
                except Exception as exc:  # noqa: BLE001 - segue sem OCR
                    logger.warning("OCR da página %s falhou em %s: %s", index + 1, path, exc)
            if text:
                block = "\n\n--- página {} ---\n{}".format(index + 1, text)
                parts.append(block)
                total += len(block)
                if total >= max_chars:
                    parts.append("\n[... texto truncado em {} caracteres]".format(max_chars))
                    break
    finally:
        doc.close()
    return "".join(parts).strip()


def ocr_image_bytes(image) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise ExtractionError("OCR indisponível. Instale pytesseract.") from exc
    return pytesseract.image_to_string(image, lang=_ocr_lang()).strip()


def extract_docx(path: Path, max_chars: int = 30000) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise ExtractionError(
            "Leitura de DOCX indisponível. Instale python-docx."
        ) from exc
    document = Document(str(path))
    parts: List[str] = []
    total = 0
    for paragraph in document.paragraphs:
        text = (paragraph.text or "").strip()
        if not text:
            continue
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    if total < max_chars:
        for table_index, table in enumerate(document.tables, start=1):
            rows: List[str] = []
            for row in table.rows:
                cells = [(cell.text or "").strip().replace("\n", " ") for cell in row.cells]
                rows.append(" | ".join(cells))
            if rows:
                block = "\n\n--- tabela {} ---\n{}".format(table_index, "\n".join(rows))
                parts.append(block)
                total += len(block)
                if total >= max_chars:
                    break
    return "\n\n".join(parts).strip()


def extract_xlsx(path: Path, max_rows: int = 2000, max_cols: int = 50) -> str:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise ExtractionError(
            "Leitura de XLSX indisponível. Instale openpyxl."
        ) from exc
    workbook = load_workbook(filename=str(path), read_only=True, data_only=True)
    parts: List[str] = []
    try:
        for sheet in workbook.worksheets:
            rows: List[str] = []
            for row_index, row in enumerate(sheet.iter_rows(values_only=True)):
                if row_index >= max_rows:
                    rows.append("[... planilha truncada]")
                    break
                values = ["" if value is None else str(value) for value in row[:max_cols]]
                if any(value.strip() for value in values):
                    rows.append("\t".join(values).rstrip())
            if rows:
                parts.append("--- aba {} ---\n{}".format(sheet.title, "\n".join(rows)))
    finally:
        workbook.close()
    return "\n\n".join(parts).strip()


def extract_csv(path: Path, max_rows: int = 2000) -> str:
    rows: List[str] = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.reader(handle)
                for index, row in enumerate(reader):
                    if index >= max_rows:
                        rows.append("[... CSV truncado]")
                        break
                    rows.append("\t".join(cell.strip() for cell in row))
            break
        except UnicodeDecodeError:
            rows = []
            continue
    return "\n".join(rows).strip()


def extract_plain_text(path: Path, max_chars: int = 30000) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = path.read_text(encoding=encoding, errors="strict")
            return text[:max_chars]
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise ExtractionError("Falha ao ler {}: {}".format(path, exc)) from exc
    raise ExtractionError("Não consegui decodificar o texto de {}".format(path))


def extract_text(path: Path, max_chars: int = 30000, ocr: bool = True) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, max_chars=max_chars, ocr=ocr)
    if suffix == ".docx":
        return extract_docx(path, max_chars=max_chars)
    if suffix in {".xlsx", ".xlsm"}:
        return extract_xlsx(path)
    if suffix in {".csv", ".tsv"}:
        return extract_csv(path)
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        if not ocr:
            raise ExtractionError("OCR desativado para imagens.")
        return ocr_image(path)
    return extract_plain_text(path, max_chars=max_chars)
