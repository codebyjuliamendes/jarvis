from __future__ import annotations

import asyncio
from pathlib import Path

from .config import Settings


class TextToSpeech:
    """Síntese de voz com edge-tts (online)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def available() -> bool:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            return False
        return True

    async def save(
        self,
        text: str,
        path: Path,
        voice: str = "",
    ) -> Path:
        import edge_tts

        text = (text or "").strip()
        if not text:
            raise ValueError("texto vazio para TTS")
        text = text[: max(100, int(self.settings.tts_max_chars))]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        communicate = edge_tts.Communicate(
            text,
            voice or self.settings.tts_voice,
            rate=self.settings.tts_rate,
        )
        await communicate.save(str(path))
        return path

    def save_sync(self, text: str, path: Path, voice: str = "") -> Path:
        return asyncio.run(self.save(text, path, voice=voice))
