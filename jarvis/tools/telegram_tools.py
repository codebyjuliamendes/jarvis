from __future__ import annotations

from typing import Any, Dict

from .. import security
from ..config import Settings
from ..security import SecurityError
from ..telegram_api import send_document
from ..utils import human_size
from .base import ToolRegistry


class TelegramTools:
    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self._register()

    def _register(self) -> None:
        self.registry.register(
            "send_file",
            "Envia um arquivo da máquina para a conversa atual do Telegram "
            "(relatórios, PDFs, planilhas, imagens, backups). Máximo ~50 MB.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Caminho do arquivo."},
                    "caption": {
                        "type": "string",
                        "description": "Legenda opcional do arquivo.",
                    },
                },
                "required": ["path"],
            },
            self._send_file,
        )

    def _send_file(self, args: Dict[str, Any]) -> str:
        chat_id = args.get("_chat_id")
        if not chat_id:
            return "Não consegui identificar o chat do Telegram para enviar o arquivo."
        raw = str(args.get("path") or "").strip()
        if not raw:
            return "Informe o campo 'path'."
        try:
            path = security.resolve_path(
                raw,
                roots=self.settings.allowed_roots,
                base=self.settings.workspace,
                must_exist=True,
            )
        except SecurityError as exc:
            return "Erro: {}".format(exc)
        if not path.is_file():
            return "Não é um arquivo: {}".format(path)

        try:
            size = path.stat().st_size
        except OSError as exc:
            return "Erro ao ler o arquivo: {}".format(exc)
        if size > 49 * 1024 * 1024:
            return "Arquivo grande demais para o Telegram ({}). Limite ~50 MB.".format(
                human_size(size)
            )

        ok, detail = send_document(
            self.settings.telegram_token,
            int(chat_id),
            path,
            caption=str(args.get("caption") or "") or None,
        )
        if ok:
            return "Arquivo enviado: {} ({}, id {})".format(
                path.name, human_size(size), detail
            )
        return "Erro ao enviar o arquivo: {}".format(detail)
