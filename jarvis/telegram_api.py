from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"


def _url(token: str, method: str) -> str:
    return "{}/bot{}/{}".format(API_BASE, token, method)


def _parse(response: httpx.Response) -> Tuple[bool, str]:
    try:
        data = response.json()
    except ValueError:
        return False, "resposta inválida do Telegram"
    if response.status_code == 200 and data.get("ok"):
        result = data.get("result") or {}
        if isinstance(result, dict):
            return True, str(result.get("message_id") or result.get("file_id") or "ok")
        return True, "ok"
    return False, str(data.get("description") or "HTTP {}".format(response.status_code))


def send_message(token: str, chat_id: int, text: str) -> Tuple[bool, str]:
    if not token:
        return False, "token do Telegram não configurado"
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                _url(token, "sendMessage"),
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                },
            )
    except httpx.HTTPError as exc:
        return False, "erro de rede: {}".format(exc)
    return _parse(response)


def send_document(
    token: str,
    chat_id: int,
    path: Path,
    caption: Optional[str] = None,
) -> Tuple[bool, str]:
    if not token:
        return False, "token do Telegram não configurado"
    data = {"chat_id": str(chat_id)}
    if caption:
        data["caption"] = caption[:1024]
    try:
        with path.open("rb") as handle:
            with httpx.Client(timeout=180.0) as client:
                response = client.post(
                    _url(token, "sendDocument"),
                    data=data,
                    files={
                        "document": (
                            path.name,
                            handle,
                            "application/octet-stream",
                        )
                    },
                )
    except (OSError, httpx.HTTPError) as exc:
        return False, "erro ao enviar: {}".format(exc)
    return _parse(response)


def send_audio(
    token: str,
    chat_id: int,
    path: Path,
    caption: Optional[str] = None,
    title: Optional[str] = None,
) -> Tuple[bool, str]:
    if not token:
        return False, "token do Telegram não configurado"
    data = {"chat_id": str(chat_id)}
    if caption:
        data["caption"] = caption[:1024]
    if title:
        data["title"] = title[:64]
    try:
        with path.open("rb") as handle:
            with httpx.Client(timeout=180.0) as client:
                response = client.post(
                    _url(token, "sendAudio"),
                    data=data,
                    files={
                        "audio": (
                            path.name,
                            handle,
                            "audio/mpeg",
                        )
                    },
                )
    except (OSError, httpx.HTTPError) as exc:
        return False, "erro ao enviar áudio: {}".format(exc)
    return _parse(response)
