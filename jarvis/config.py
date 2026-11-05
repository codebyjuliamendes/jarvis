from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_MS_SCOPES = [
    "User.Read",
    "Mail.Read",
    "Mail.Send",
    "Calendars.ReadWrite",
    "Chat.Read",
    "Team.ReadBasic.All",
    "Channel.ReadBasic.All",
]


def _load_env() -> None:
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_file)


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "sim", "on", "y"}


def _as_int(value: Optional[str], default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_float(value: Optional[str], default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    raw = str(value).replace(";", ",")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _as_ids(value: Optional[str]) -> Set[int]:
    ids: Set[int] = set()
    for item in _as_list(value):
        try:
            ids.add(int(item))
        except ValueError:
            continue
    return ids


def _as_path(value: Optional[str], default: Optional[Path] = None) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return default
    return Path(str(value)).expanduser().resolve(strict=False)


@dataclass
class Settings:
    telegram_token: str
    allowed_user_ids: Set[int]
    ollama_url: str
    ollama_model: str
    ollama_timeout: float
    ollama_num_ctx: int
    max_iterations: int
    history_limit: int
    db_path: Path
    allowed_roots: List[Path]
    shell_enabled: bool
    shell_timeout: int
    owner_name: str
    timezone: str
    workspace: Path
    obsidian_vault: Optional[Path]
    notion_token: Optional[str]
    notion_default_page: Optional[str]
    stripe_api_key: Optional[str]
    github_token: Optional[str]
    github_default_repo: Optional[str]
    google_credentials: Path
    google_token: Path
    whisper_enabled: bool = True
    whisper_model: str = "small"
    whisper_language: str = "pt"
    briefing_enabled: bool = True
    briefing_hour: int = 6
    briefing_minute: int = 0
    briefing_max_iterations: int = 4
    rag_enabled: bool = True
    rag_model: str = "nomic-embed-text"
    rag_roots: List[Path] = field(default_factory=list)
    rag_extensions: str = ".md,.txt,.pdf,.docx,.csv,.xlsx"
    ms_graph_client_id: Optional[str] = None
    ms_graph_tenant: str = "organizations"
    ms_graph_token: Optional[Path] = None
    ms_graph_scopes: List[str] = field(default_factory=list)
    tts_enabled: bool = True
    tts_auto: bool = False
    tts_voice: str = "pt-BR-FranciscaNeural"
    tts_rate: str = "+0%"
    tts_max_chars: int = 3000

    @classmethod
    def load(cls) -> "Settings":
        _load_env()

        home = Path.home()
        workspace = _as_path(os.getenv("JARVIS_WORKSPACE"), home) or home
        workspace = workspace.resolve(strict=False)

        roots: List[Path] = []
        for raw in _as_list(os.getenv("JARVIS_ALLOWED_ROOTS")):
            candidate = Path(raw).expanduser().resolve(strict=False)
            if candidate not in roots:
                roots.append(candidate)
        if not roots:
            roots = [home]
        if workspace not in roots:
            roots.append(workspace)

        db_path = _as_path(os.getenv("JARVIS_DB_PATH"), BASE_DIR / "data" / "jarvis.sqlite3")
        assert db_path is not None
        db_path.parent.mkdir(parents=True, exist_ok=True)
        uploads_dir = db_path.parent / "uploads"
        if uploads_dir not in roots:
            roots.append(uploads_dir)

        obsidian_vault = _as_path(os.getenv("OBSIDIAN_VAULT"))

        rag_roots: List[Path] = []
        for raw in _as_list(os.getenv("JARVIS_RAG_ROOTS")):
            candidate = Path(raw).expanduser().resolve(strict=False)
            if candidate.exists() and candidate not in rag_roots:
                rag_roots.append(candidate)
        if not rag_roots:
            defaults = [
                obsidian_vault,
                workspace if workspace != home else None,
                home / "Documentos",
                home / "Downloads",
                BASE_DIR,
            ]
            for candidate in defaults:
                if candidate is None:
                    continue
                resolved = Path(candidate).expanduser().resolve(strict=False)
                if resolved.exists() and resolved not in rag_roots:
                    rag_roots.append(resolved)

        return cls(
            telegram_token=(os.getenv("TELEGRAM_BOT_TOKEN") or "").strip(),
            allowed_user_ids=_as_ids(os.getenv("TELEGRAM_ALLOWED_USER_IDS")),
            ollama_url=(os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/"),
            ollama_model=(os.getenv("OLLAMA_MODEL") or "qwen2.5:3b").strip(),
            ollama_timeout=_as_float(os.getenv("OLLAMA_TIMEOUT"), 300.0),
            ollama_num_ctx=_as_int(os.getenv("OLLAMA_NUM_CTX"), 8192),
            max_iterations=_as_int(os.getenv("JARVIS_MAX_ITERATIONS"), 8),
            history_limit=_as_int(os.getenv("JARVIS_HISTORY_LIMIT"), 40),
            db_path=db_path,
            allowed_roots=roots,
            shell_enabled=_as_bool(os.getenv("JARVIS_SHELL_ENABLED"), True),
            shell_timeout=_as_int(os.getenv("JARVIS_SHELL_TIMEOUT"), 120),
            owner_name=(os.getenv("JARVIS_OWNER_NAME") or "Julia").strip(),
            timezone=(os.getenv("JARVIS_TIMEZONE") or "America/Sao_Paulo").strip(),
            workspace=workspace,
            obsidian_vault=obsidian_vault,
            notion_token=(os.getenv("NOTION_TOKEN") or "").strip() or None,
            notion_default_page=(os.getenv("NOTION_DEFAULT_PAGE") or "").strip() or None,
            stripe_api_key=(os.getenv("STRIPE_API_KEY") or "").strip() or None,
            github_token=(os.getenv("GITHUB_TOKEN") or "").strip() or None,
            github_default_repo=(os.getenv("GITHUB_DEFAULT_REPO") or "").strip() or None,
            google_credentials=_as_path(
                os.getenv("GOOGLE_CREDENTIALS"), BASE_DIR / "credentials.json"
            ) or (BASE_DIR / "credentials.json"),
            google_token=_as_path(os.getenv("GOOGLE_TOKEN"), BASE_DIR / "token.json")
            or (BASE_DIR / "token.json"),
            whisper_enabled=_as_bool(os.getenv("JARVIS_WHISPER_ENABLED"), True),
            whisper_model=(os.getenv("JARVIS_WHISPER_MODEL") or "small").strip(),
            whisper_language=(os.getenv("JARVIS_WHISPER_LANGUAGE") or "pt").strip(),
            briefing_enabled=_as_bool(os.getenv("JARVIS_BRIEFING_ENABLED"), True),
            briefing_hour=_as_int(os.getenv("JARVIS_BRIEFING_HOUR"), 6),
            briefing_minute=_as_int(os.getenv("JARVIS_BRIEFING_MINUTE"), 0),
            briefing_max_iterations=_as_int(
                os.getenv("JARVIS_BRIEFING_MAX_ITERATIONS"), 4
            ),
            rag_enabled=_as_bool(os.getenv("JARVIS_RAG_ENABLED"), True),
            rag_model=(os.getenv("JARVIS_RAG_MODEL") or "nomic-embed-text").strip(),
            rag_roots=rag_roots,
            rag_extensions=(
                os.getenv("JARVIS_RAG_EXTENSIONS")
                or ".md,.txt,.pdf,.docx,.csv,.xlsx"
            ).strip(),
            ms_graph_client_id=(os.getenv("MS_GRAPH_CLIENT_ID") or "").strip() or None,
            ms_graph_tenant=(os.getenv("MS_GRAPH_TENANT") or "organizations").strip(),
            ms_graph_token=_as_path(
                os.getenv("MS_GRAPH_TOKEN"),
                BASE_DIR / "data" / "ms_graph_token.json",
            ),
            ms_graph_scopes=_as_list(os.getenv("MS_GRAPH_SCOPES"))
            or list(DEFAULT_MS_SCOPES),
            tts_enabled=_as_bool(os.getenv("JARVIS_TTS_ENABLED"), True),
            tts_auto=_as_bool(os.getenv("JARVIS_TTS_AUTO"), False),
            tts_voice=(os.getenv("JARVIS_TTS_VOICE") or "pt-BR-FranciscaNeural").strip(),
            tts_rate=(os.getenv("JARVIS_TTS_RATE") or "+0%").strip(),
            tts_max_chars=_as_int(os.getenv("JARVIS_TTS_MAX_CHARS"), 3000),
        )
