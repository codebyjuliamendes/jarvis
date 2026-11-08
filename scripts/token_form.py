#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import parse_qs

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
RESULT_PATH = BASE_DIR / "data" / "token_form_result.json"
HOST = "127.0.0.1"
PORT = int(os.getenv("JARVIS_FORM_PORT", "8765"))
SECRET = os.getenv("JARVIS_FORM_SECRET") or secrets.token_urlsafe(18)


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
        raise ValueError(str(detail)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("sem conexão com o Telegram: {}".format(exc.reason)) from exc
    if not data.get("ok"):
        raise ValueError(str(data.get("description") or "token inválido"))
    result = data.get("result") or {}
    if not isinstance(result, dict):
        raise RuntimeError("resposta inesperada do Telegram")
    return result


def _page(title: str, body: str) -> bytes:
    document = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; min-height: 100vh; display: flex; align-items: center;
    justify-content: center; background: #101418; color: #e8eef5;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  .card {{
    width: min(560px, 92vw); background: #171d24; border: 1px solid #26303b;
    border-radius: 16px; padding: 28px; box-shadow: 0 18px 50px #0008;
  }}
  h1 {{ font-size: 1.35rem; margin: 0 0 8px; }}
  p {{ color: #a9b7c6; line-height: 1.5; margin: 10px 0; }}
  code {{ background: #0d1116; padding: 2px 6px; border-radius: 6px; color: #8fd3ff; }}
  label {{ display: block; margin: 20px 0 8px; font-weight: 600; }}
  input[type=password] {{
    width: 100%; padding: 14px; border-radius: 10px; border: 1px solid #344150;
    background: #0d1116; color: #fff; font-size: 1rem; letter-spacing: .02em;
  }}
  input[type=password]:focus {{ outline: 2px solid #2f81f7; border-color: #2f81f7; }}
  button {{
    width: 100%; margin-top: 16px; padding: 14px; border: 0; border-radius: 10px;
    background: #2f81f7; color: #fff; font-size: 1rem; font-weight: 700;
    cursor: pointer;
  }}
  button:hover {{ background: #4b93ff; }}
  .error {{
    margin: 16px 0; padding: 12px 14px; border-radius: 10px;
    background: #3a1d22; border: 1px solid #7d2d3a; color: #ffb3bd;
  }}
  .ok {{ color: #7ee2a8; }}
  .muted {{ font-size: .9rem; }}
</style>
</head>
<body>
  <main class="card">
    {body}
  </main>
</body>
</html>""".format(title=html.escape(title), body=body)
    return document.encode("utf-8")


def _form(error: str = "") -> bytes:
    body = """
    <h1>Conectar o Jarvis ao Telegram</h1>
    <p>Cole abaixo o token do seu bot, criado no <b>@BotFather</b>.</p>
    <p class="muted">Formato: <code>123456789:AAH...</code> — copie só o token,
    sem espaços, aspas ou o nome do bot.</p>
    {error}
    <form method="post" action="">
      <label for="token">Token do bot</label>
      <input id="token" name="token" type="password" autocomplete="off"
             spellcheck="false" autofocus placeholder="123456789:AAH...">
      <button type="submit">Salvar token</button>
    </form>
    <p class="muted">O token é gravado apenas no arquivo <code>.env</code> desta
    máquina, com permissão 600. Nada é enviado para outros servidores além do
    próprio Telegram para validar o bot.</p>
    """.format(error=('<div class="error">{}</div>'.format(html.escape(error)) if error else ""))
    return _page("Jarvis - Token do Telegram", body)


def _success(username: str, name: str) -> bytes:
    body = """
    <h1 class="ok">Token salvo!</h1>
    <p>Bot validado: <b>@{username}</b>{name}.</p>
    <p>Agora volte ao terminal e rode:</p>
    <p><code>cd ~/jarvis &amp;&amp; ./run.sh</code></p>
    <p>Depois mande <code>/start</code> para o seu bot no Telegram. Ele vai
    mostrar seu ID numérico; me avise que eu libero o acesso.</p>
    <p class="muted">Pode fechar esta aba.</p>
    """.format(
        username=html.escape(username),
        name=(" (" + html.escape(name) + ")") if name else "",
    )
    return _page("Jarvis - Token salvo", body)


class Handler(BaseHTTPRequestHandler):
    server_version = "JarvisTokenForm/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/" + SECRET + "/":
            self._send(404, _page("Jarvis", "<h1>404</h1><p>Página não encontrada.</p>"))
            return
        self._send(200, _form())

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/" + SECRET + "/":
            self._send(404, _page("Jarvis", "<h1>404</h1><p>Página não encontrada.</p>"))
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        if length <= 0 or length > 10000:
            self._send(413, _form("Requisição inválida. Tente de novo."))
            return
        body = self.rfile.read(length).decode("utf-8", "replace")
        token = (parse_qs(body).get("token") or [""])[0].strip()
        if not token:
            self._send(400, _form("Cole o token antes de salvar."))
            return

        try:
            bot = _validate(token)
        except (ValueError, RuntimeError) as exc:
            self._send(400, _form(str(exc)))
            return

        _write_env(token)
        username = str(bot.get("username") or "?")
        name = str(bot.get("first_name") or "")
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(
            json.dumps({"ok": True, "username": username, "name": name}),
            encoding="utf-8",
        )
        self._send(200, _success(username, name))

        def _stop() -> None:
            time.sleep(1.5)
            self.server.shutdown()

        threading.Thread(target=_stop, daemon=True).start()


def main() -> int:
    if not (BASE_DIR / ".env").exists():
        print("Arquivo .env não encontrado. Rode ./setup.sh primeiro.", file=sys.stderr)
        return 1
    url = "http://{}:{}/{}/".format(HOST, PORT, SECRET)
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as exc:
        print("Não consegui abrir a porta {}: {}".format(PORT, exc), file=sys.stderr)
        return 1

    print(url, flush=True)
    try:
        opener = subprocess.Popen(
            ["xdg-open", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        opener.wait(timeout=10)
    except Exception:  # noqa: BLE001
        print("Abra esta URL no navegador: {}".format(url), flush=True)

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
