#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


def _zenity() -> Optional[str]:
    return shutil.which("zenity")


def _gui_ask() -> Optional[str]:
    zenity = _zenity()
    if not zenity:
        return None
    try:
        result = subprocess.run(
            [
                zenity,
                "--password",
                "--title",
                "Jarvis - Token do Telegram",
                "--text",
                "Cole o token do @BotFather (o campo fica oculto):",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _gui_message(kind: str, text: str) -> None:
    zenity = _zenity()
    if not zenity:
        return
    try:
        subprocess.run(
            [zenity, "--" + kind, "--title", "Jarvis", "--text", text],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _read_env() -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not ENV_PATH.exists():
        return values
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _write_env(updates: Dict[str, str]) -> None:
    lines: List[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    output: List[str] = []
    seen = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            output.append("{}={}".format(key, updates[key]))
            seen.add(key)
        else:
            output.append(line)
    for key, value in updates.items():
        if key not in seen:
            output.append("{}={}".format(key, value))
    ENV_PATH.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    os.chmod(str(ENV_PATH), 0o600)


def _validate(token: str) -> Dict[str, object]:
    if ":" not in token or len(token) < 20:
        raise ValueError("formato inválido (esperado algo como 123456:ABC...)")
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "httpx não instalado. Rode ./setup.sh primeiro."
        ) from exc
    try:
        response = httpx.get(
            "https://api.telegram.org/bot{}/getMe".format(token), timeout=20.0
        )
    except httpx.HTTPError as exc:
        raise RuntimeError("não consegui falar com o Telegram: {}".format(exc)) from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("resposta inválida do Telegram") from exc
    if response.status_code != 200 or not data.get("ok"):
        raise ValueError(str(data.get("description") or "token inválido"))
    result = data.get("result") or {}
    if not isinstance(result, dict):
        raise RuntimeError("resposta inesperada do Telegram")
    return result


def _parse_ids(raw_values: List[str]) -> List[int]:
    ids: List[int] = []
    for raw in raw_values:
        for item in raw.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                value = int(item)
            except ValueError as exc:
                raise ValueError("ID inválido: {}".format(item)) from exc
            if value not in ids:
                ids.append(value)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Configura o token do bot do Telegram no .env."
    )
    parser.add_argument(
        "--token",
        help="token do @BotFather (prefira omitir para digitar oculto)",
    )
    parser.add_argument(
        "--token-stdin",
        action="store_true",
        help="lê o token da entrada padrão",
    )
    parser.add_argument(
        "--allowed-id",
        action="append",
        default=[],
        help="IDs do Telegram autorizados (pode repetir ou separar por vírgula)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="apenas valida o token que já está no .env",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="abre uma caixinha gráfica (zenity) para colar o token",
    )
    args = parser.parse_args()

    current = _read_env()
    token = (args.token or "").strip()

    if args.check:
        token = (current.get("TELEGRAM_BOT_TOKEN") or "").strip()
        if not token:
            print("TELEGRAM_BOT_TOKEN não está preenchido no .env.", file=sys.stderr)
            return 1

    if args.token_stdin and not args.check:
        token = sys.stdin.readline().strip()
    if not token and not args.check:
        if args.gui:
            token = (_gui_ask() or "").strip()
            if not token:
                print("Cancelado. Nenhum token foi salvo.", file=sys.stderr)
                return 1
        else:
            token = getpass.getpass(
                "Cole o token do @BotFather (não aparece na tela): "
            ).strip()
    if not token:
        print("Token vazio.", file=sys.stderr)
        return 1

    try:
        bot = _validate(token)
        ids = _parse_ids(args.allowed_id)
    except (ValueError, RuntimeError) as exc:
        message = "Erro: {}".format(exc)
        print(message, file=sys.stderr)
        if args.gui:
            _gui_message("error", message)
        return 1

    updates: Dict[str, str] = {}
    if not args.check:
        updates["TELEGRAM_BOT_TOKEN"] = token
    if ids:
        updates["TELEGRAM_ALLOWED_USER_IDS"] = ",".join(str(value) for value in ids)
    if updates:
        _write_env(updates)

    username = bot.get("username") or "?"
    name = bot.get("first_name") or ""
    print("Bot validado: @{} {}".format(username, ("(" + str(name) + ")") if name else ""))
    if updates:
        print("Configuração salva em {}".format(ENV_PATH))
    if ids:
        print("IDs autorizados: {}".format(", ".join(str(value) for value in ids)))
    else:
        saved_ids = current.get("TELEGRAM_ALLOWED_USER_IDS", "").strip()
        print(
            "IDs autorizados: {}".format(saved_ids or "nenhum ainda")
        )
        print(
            "Próximo passo: rode ./run.sh, mande /start no Telegram e depois execute:\n"
            "  .venv/bin/python scripts/configure_telegram.py --allowed-id SEU_ID"
        )

    if args.gui:
        lines = ["Bot validado: @{}".format(username)]
        if updates:
            lines.append("Token salvo em {}".format(ENV_PATH))
        if ids:
            lines.append("IDs autorizados: {}".format(", ".join(str(value) for value in ids)))
        else:
            lines.append(
                "Agora rode ./run.sh, mande /start no Telegram\ne me avise que eu autorizo seu ID."
            )
        _gui_message("info", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
