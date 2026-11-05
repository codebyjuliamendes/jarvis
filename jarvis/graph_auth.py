from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .config import DEFAULT_MS_SCOPES, Settings

GRAPH_SCOPES = list(DEFAULT_MS_SCOPES)


def token_path(settings: Settings) -> Path:
    if settings.ms_graph_token is not None:
        return Path(settings.ms_graph_token)
    return Path(settings.db_path).parent / "ms_graph_token.json"


def scopes(settings: Settings) -> List[str]:
    return list(settings.ms_graph_scopes or GRAPH_SCOPES)


def _build_app(settings: Settings):
    try:
        import msal
    except ImportError as exc:
        raise RuntimeError(
            "MSAL não instalado. Rode `pip install -r requirements.txt` no .venv."
        ) from exc
    if not settings.ms_graph_client_id:
        raise RuntimeError(
            "Microsoft 365 não configurado. Defina MS_GRAPH_CLIENT_ID e "
            "MS_GRAPH_TENANT no .env e rode scripts/ms_graph_auth.py."
        )
    path = token_path(settings)
    cache = msal.SerializableTokenCache()
    if path.exists():
        try:
            cache.deserialize(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - cache corrompido é recriado
            pass
    authority = "https://login.microsoftonline.com/{}".format(
        settings.ms_graph_tenant or "organizations"
    )
    app = msal.PublicClientApplication(
        settings.ms_graph_client_id,
        authority=authority,
        token_cache=cache,
    )
    return app, cache


def save_cache(cache, path: Path) -> None:
    if not getattr(cache, "has_state_changed", False):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cache.serialize(), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def get_access_token(settings: Settings) -> Tuple[Optional[str], Optional[str]]:
    try:
        app, cache = _build_app(settings)
    except RuntimeError as exc:
        return None, str(exc)
    accounts = app.get_accounts()
    result = None
    if accounts:
        try:
            result = app.acquire_token_silent(scopes(settings), account=accounts[0])
        except Exception as exc:  # noqa: BLE001
            return None, "Erro ao renovar token da Microsoft: {}".format(exc)
    save_cache(cache, token_path(settings))
    if result and result.get("access_token"):
        return result["access_token"], None
    return None, (
        "Microsoft 365 não autenticado. Rode "
        "`python scripts/ms_graph_auth.py` uma vez para autorizar."
    )
