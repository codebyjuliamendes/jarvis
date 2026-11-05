#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from jarvis.config import Settings  # noqa: E402
from jarvis.graph_auth import (  # noqa: E402
    _build_app,
    save_cache,
    scopes,
    token_path,
)


def main() -> int:
    settings = Settings.load()
    if not settings.ms_graph_client_id:
        print(
            "Configure primeiro no .env:\n"
            "  MS_GRAPH_CLIENT_ID=<Application (client) ID>\n"
            "  MS_GRAPH_TENANT=<Directory (tenant) ID ou organizations>\n\n"
            "Veja as instruções no README (seção Microsoft 365).",
            file=sys.stderr,
        )
        return 1

    try:
        app, cache = _build_app(settings)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    flow = app.initiate_device_flow(scopes=scopes(settings))
    if "user_code" not in flow:
        print(
            "Falha ao iniciar o device code: {}".format(
                flow.get("error_description") or flow
            ),
            file=sys.stderr,
        )
        return 1

    print(flow.get("message") or "")
    print("\nAguardando a autorização no navegador...")
    result = app.acquire_token_by_device_flow(flow)
    save_cache(cache, token_path(settings))

    if result.get("access_token"):
        print("Autenticado com sucesso. Token salvo em {}".format(token_path(settings)))
        return 0

    print(
        "Falha na autenticação: {}".format(
            result.get("error_description") or result.get("error") or result
        ),
        file=sys.stderr,
    )
    print(
        "Se o erro for consentimento de admin, peça ao administrador do tenant "
        "ou remova escopos do Teams em MS_GRAPH_SCOPES e tente de novo.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
