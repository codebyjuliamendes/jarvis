from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .. import security
from ..config import Settings
from ..documents import ExtractionError, extract_docx, extract_pdf, ocr_image
from ..security import SecurityError
from .base import ToolRegistry


class DocumentTools:
    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self._register()

    def _resolve(self, raw: str) -> Path:
        return security.resolve_path(
            raw,
            roots=self.settings.allowed_roots,
            base=self.settings.workspace,
            must_exist=True,
        )

    def _register(self) -> None:
        self.registry.register(
            "read_pdf",
            "Extrai o texto de um PDF (com OCR automático em páginas escaneadas). "
            "Use para provas, apostilas, contratos e relatórios.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Caminho do PDF."},
                    "max_pages": {
                        "type": "integer",
                        "description": "Máximo de páginas (padrão 50).",
                    },
                    "ocr": {
                        "type": "boolean",
                        "description": "Usar OCR em páginas sem texto (padrão true).",
                    },
                },
                "required": ["path"],
            },
            self._read_pdf,
        )
        self.registry.register(
            "read_docx",
            "Extrai o texto (incluindo tabelas) de um arquivo Word .docx.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Caminho do .docx."}
                },
                "required": ["path"],
            },
            self._read_docx,
        )
        self.registry.register(
            "read_image",
            "Faz OCR de uma imagem (print, foto de quadro, comprovante). "
            "Retorna o texto reconhecido.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Caminho da imagem."}
                },
                "required": ["path"],
            },
            self._read_image,
        )

    def _read_pdf(self, args: Dict[str, Any]) -> str:
        try:
            path = self._resolve(str(args.get("path", "")))
        except SecurityError as exc:
            return "Erro: {}".format(exc)
        if path.suffix.lower() != ".pdf":
            return "O arquivo não é um PDF: {}".format(path)
        try:
            max_pages = int(args.get("max_pages") or 50)
        except (TypeError, ValueError):
            max_pages = 50
        try:
            text = extract_pdf(
                path,
                max_pages=max(1, min(max_pages, 300)),
                ocr=bool(args.get("ocr", True)),
            )
        except ExtractionError as exc:
            return "Erro: {}".format(exc)
        if not text:
            return "Não encontrei texto no PDF (talvez seja só imagem e o OCR falhou)."
        return "PDF: {}\n\n{}".format(path, text)

    def _read_docx(self, args: Dict[str, Any]) -> str:
        try:
            path = self._resolve(str(args.get("path", "")))
        except SecurityError as exc:
            return "Erro: {}".format(exc)
        if path.suffix.lower() != ".docx":
            return "O arquivo não é um .docx: {}".format(path)
        try:
            text = extract_docx(path)
        except ExtractionError as exc:
            return "Erro: {}".format(exc)
        if not text:
            return "O .docx não tem texto extraível."
        return "DOCX: {}\n\n{}".format(path, text)

    def _read_image(self, args: Dict[str, Any]) -> str:
        try:
            path = self._resolve(str(args.get("path", "")))
        except SecurityError as exc:
            return "Erro: {}".format(exc)
        try:
            text = ocr_image(path)
        except ExtractionError as exc:
            return "Erro: {}".format(exc)
        if not text:
            return "Não reconheci texto na imagem."
        return "OCR: {}\n\n{}".format(path, text)
