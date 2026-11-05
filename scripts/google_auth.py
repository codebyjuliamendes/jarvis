#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from jarvis.config import Settings  # noqa: E402
from jarvis.tools.google import GoogleTools  # noqa: E402


def main() -> int:
    settings = Settings.load()
    credentials_path = settings.google_credentials
    token_path = settings.google_token

    if not credentials_path.exists():
        print("credentials.json não encontrado em: {}".format(credentials_path))
        print("")
        print("Como obter:")
        print("1. Acesse https://console.cloud.google.com/")
        print("2. Ative as APIs 'Google Calendar API' e 'Gmail API'.")
        print("3. Configure a tela de consentimento OAuth (tipo Externo, modo teste).")
        print("4. Crie credenciais > ID do cliente OAuth > Aplicativo para computador.")
        print("5. Baixe o JSON e salve como: {}".format(credentials_path))
        return 1

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("Dependências do Google não instaladas. Rode ./setup.sh primeiro.")
        return 1

    credentials = None
    if token_path.exists():
        try:
            credentials = Credentials.from_authorized_user_file(
                str(token_path), GoogleTools.SCOPES
            )
        except Exception:  # noqa: BLE001
            credentials = None

    if credentials and credentials.valid:
        print("Token existente ainda é válido: {}".format(token_path))
        return 0

    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
            token_path.write_text(credentials.to_json(), encoding="utf-8")
            print("Token renovado: {}".format(token_path))
            return 0
        except Exception as exc:  # noqa: BLE001
            print("Não consegui renovar o token ({}). Vou pedir autorização de novo.".format(exc))

    print("Abrindo o navegador para autorizar o Google...")
    print("Se você estiver em outra máquina, copie a URL que aparecer.")
    flow = InstalledAppFlow.from_client_secrets_file(
        str(credentials_path), GoogleTools.SCOPES
    )
    credentials = flow.run_local_server(port=0, open_browser=True)
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    print("Token salvo em: {}".format(token_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
