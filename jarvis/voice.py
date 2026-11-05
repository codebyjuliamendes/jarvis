from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

from .config import Settings

logger = logging.getLogger(__name__)


class VoiceTranscriber:
    """Transcrição local de áudio com faster-whisper (carregamento preguiçoso)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Optional[Any] = None
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def _load_model(self) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper não instalado. Rode ./scripts/setup_voice.sh"
            ) from exc
        with self._lock:
            if self._model is None:
                logger.info(
                    "Carregando modelo de voz %s (pode baixar na primeira vez)",
                    self.settings.whisper_model,
                )
                self._model = WhisperModel(
                    self.settings.whisper_model,
                    device="cpu",
                    compute_type="int8",
                )
        return self._model

    def transcribe(self, path: Path) -> str:
        model = self._load_model()
        segments, _info = model.transcribe(
            str(path),
            language=self.settings.whisper_language or None,
            vad_filter=True,
            beam_size=1,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
