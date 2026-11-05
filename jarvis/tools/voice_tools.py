from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from ..config import Settings
from ..telegram_api import send_audio
from ..tts import TextToSpeech
from .base import ToolRegistry


class VoiceTools:
    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self.tts = TextToSpeech(settings)
        self._register()

    def _register(self) -> None:
        self.registry.register(
            "speak",
            "Gera um áudio de voz (edge-tts) com o texto informado e envia no Telegram. "
            "Use quando a usuária pedir 'em áudio', 'fala isso' ou 'manda um áudio'.",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Texto que será falado."},
                    "caption": {
                        "type": "string",
                        "description": "Legenda opcional do áudio.",
                    },
                },
                "required": ["text"],
            },
            self._speak,
        )

    def _speak(self, args: Dict[str, Any]) -> str:
        chat_id = args.get("_chat_id")
        if not chat_id:
            return "Não consegui identificar o chat do Telegram para enviar o áudio."
        text = str(args.get("text") or "").strip()
        if not text:
            return "Informe o campo 'text'."
        if not self.settings.tts_enabled:
            return "TTS desativado (JARVIS_TTS_ENABLED=false)."
        if not self.tts.available():
            return "edge-tts não instalado. Rode `pip install -r requirements.txt` no .venv."

        tts_dir = Path(self.settings.db_path).parent / "tts"
        path = tts_dir / "tts-{}.mp3".format(int(time.time() * 1000))
        try:
            self.tts.save_sync(text, path)
            ok, detail = send_audio(
                self.settings.telegram_token,
                int(chat_id),
                path,
                caption=str(args.get("caption") or "") or None,
            )
        except Exception as exc:  # noqa: BLE001
            return "Erro ao gerar/enviar áudio: {}".format(exc)
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if ok:
            return "Áudio enviado (edge-tts, voz {}).".format(self.settings.tts_voice)
        return "Erro ao enviar áudio: {}".format(detail)
