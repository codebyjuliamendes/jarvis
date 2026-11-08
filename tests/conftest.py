"""Configuração compartilhada dos testes. Nenhum teste fala com o Ollama."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from jarvis.config import Settings  # noqa: E402


def build_settings(root: Path) -> Settings:
    root = Path(root).resolve()
    vault = root / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    return Settings(
        telegram_token="teste",
        allowed_user_ids={1},
        ollama_url="http://127.0.0.1:9",  # porta descartável: nada deve chamar o Ollama
        ollama_model="fake:latest",
        ollama_timeout=5.0,
        ollama_num_ctx=2048,
        max_iterations=4,
        history_limit=10,
        db_path=root / "data" / "test.sqlite3",
        allowed_roots=[root],
        shell_enabled=True,
        shell_timeout=30,
        owner_name="Teste",
        timezone="America/Sao_Paulo",
        workspace=root,
        obsidian_vault=vault,
        notion_token=None,
        notion_default_page=None,
        stripe_api_key=None,
        github_token=None,
        github_default_repo=None,
        google_credentials=root / "credentials.json",
        google_token=root / "token.json",
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return build_settings(tmp_path)
