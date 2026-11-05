#!/usr/bin/python3
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


def _write_env(token: str) -> None:
    lines: List[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    output: List[str] = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key == "TELEGRAM_BOT_TOKEN":
            output.append("TELEGRAM_BOT_TOKEN={}".format(token))
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.append("TELEGRAM_BOT_TOKEN={}".format(token))
    ENV_PATH.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    os.chmod(str(ENV_PATH), 0o600)


def _validate(token: str) -> Dict[str, object]:
    if ":" not in token or len(token) < 20:
        raise ValueError("formato inválido (esperado algo como 123456789:AAH...)")
    url = "https://api.telegram.org/bot{}/getMe".format(token)
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
            detail = body.get("description") or str(exc)
        except Exception:  # noqa: BLE001
            detail = str(exc)
        raise ValueError(detail) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("sem conexão com o Telegram: {}".format(exc.reason)) from exc
    if not data.get("ok"):
        raise ValueError(str(data.get("description") or "token inválido"))
    result = data.get("result") or {}
    if not isinstance(result, dict):
        raise RuntimeError("resposta inesperada do Telegram")
    return result


def _ask_token() -> str:
    dialog = Gtk.Dialog(title="Jarvis - Token do Telegram")
    dialog.set_default_size(560, -1)
    dialog.set_keep_above(True)
    dialog.add_buttons(
        "Cancelar", Gtk.ResponseType.CANCEL, "Salvar", Gtk.ResponseType.OK
    )
    dialog.set_default_response(Gtk.ResponseType.OK)

    box = dialog.get_content_area()
    box.set_spacing(10)
    box.set_border_width(14)

    title = Gtk.Label()
    title.set_markup("<b>Cole o token do @BotFather</b>")
    title.set_xalign(0)
    box.add(title)

    help_text = Gtk.Label()
    help_text.set_markup(
        "Formato: <tt>123456789:AAH...</tt>\n"
        "Copie só o token, sem espaços, aspas ou o nome do bot.\n"
        "Se precisar, peça de novo no Telegram: <tt>/token</tt>."
    )
    help_text.set_line_wrap(True)
    help_text.set_xalign(0)
    box.add(help_text)

    entry = Gtk.Entry()
    entry.set_visibility(False)
    entry.set_placeholder_text("123456789:AAH...")
    entry.set_activates_default(True)
    box.add(entry)

    dialog.show_all()
    try:
        dialog.get_window().raise_()
    except Exception:  # noqa: BLE001
        pass
    dialog.present()

    response = dialog.run()
    token = entry.get_text().strip()
    dialog.destroy()
    if response == Gtk.ResponseType.OK and token:
        return token
    return ""


def _message(kind: str, text: str) -> None:
    dialog = Gtk.MessageDialog(
        message_type=(
            Gtk.MessageType.ERROR if kind == "error" else Gtk.MessageType.INFO
        ),
        buttons=Gtk.ButtonsType.OK,
        text=text,
    )
    dialog.set_title("Jarvis")
    dialog.set_keep_above(True)
    dialog.run()
    dialog.destroy()


def main() -> int:
    attempts = 0
    while attempts < 3:
        token = _ask_token()
        if not token:
            return 1
        try:
            bot = _validate(token)
        except (ValueError, RuntimeError) as exc:
            attempts += 1
            if attempts >= 3:
                _message(
                    "error",
                    "Não consegui validar o token:\n{}\n\n"
                    "Confira no @BotFather com /token e tente de novo.".format(exc),
                )
                return 1
            _message(
                "error",
                "Token inválido:\n{}\n\n"
                "O formato é 123456789:AAH... Tente de novo.".format(exc),
            )
            continue

        _write_env(token)
        username = bot.get("username") or "?"
        _message(
            "info",
            "Bot validado: @{}\n\nToken salvo em {}.\n\n"
            "Agora rode ./run.sh, mande /start no Telegram e me avise.".format(
                username, ENV_PATH
            ),
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
