"""Peças compartilhadas pelas avaliações (evals/).

Nada aqui executa ferramentas nem precisa de credenciais: as settings são
fictícias e apontam para uma pasta temporária.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
EVALS_DIR = BASE_DIR / "evals"
DATA_DIR = EVALS_DIR / "data"
FIXTURES_DIR = EVALS_DIR / "fixtures"
RESULTS_DIR = EVALS_DIR / "results"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from jarvis.config import Settings  # noqa: E402

# 127.0.0.1 e não "localhost": no Windows cada conexão nova do httpx com
# "localhost" perde ~2 s tentando IPv6 (::1) antes do IPv4 (medido em 08/11/2026),
# o que contaminaria a latência medida. Ver evals/PENDENCIAS.md.
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_SEED = 42


# ------------------------------------------------------------------ settings


def make_eval_settings(
    root: Path,
    model: str = "qwen2.5:3b",
    ollama_url: str = DEFAULT_OLLAMA_URL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    num_ctx: int = 8192,
    timeout: float = 300.0,
) -> Settings:
    """Settings fictícias com os mesmos padrões do .env.example."""
    root = Path(root).resolve()
    (root / "data").mkdir(parents=True, exist_ok=True)
    vault = root / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    return Settings(
        telegram_token="eval",
        allowed_user_ids={1},
        ollama_url=ollama_url.rstrip("/"),
        ollama_model=model,
        ollama_timeout=timeout,
        ollama_num_ctx=num_ctx,
        max_iterations=8,
        history_limit=40,
        db_path=root / "data" / "eval.sqlite3",
        allowed_roots=[root],
        shell_enabled=True,
        shell_timeout=120,
        owner_name="Julia",
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
        rag_model=embed_model,
        rag_roots=[root],
    )


# --------------------------------------------------------------------- dados


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(str(path), encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError("{}:{}: JSON inválido: {}".format(path, number, exc))
            if not isinstance(row, dict):
                raise ValueError("{}:{}: cada linha deve ser um objeto".format(path, number))
            rows.append(row)
    ids = [row.get("id") for row in rows]
    duplicates = sorted({i for i in ids if ids.count(i) > 1 and i is not None})
    if duplicates:
        raise ValueError("{}: ids repetidos: {}".format(path, ", ".join(duplicates)))
    return rows


def file_sha256(path: Path, length: int = 12) -> str:
    """Hash do arquivo de dados: identifica a versão do dataset usada no run."""
    digest = hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:length]


def dir_sha256(path: Path, length: int = 12) -> str:
    digest = hashlib.sha256()
    for item in sorted(Path(path).rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(item.read_bytes())
    return digest.hexdigest()[:length]


# ---------------------------------------------------------- comparação de args


def normalize(value: Any) -> str:
    """Minúsculas, sem acento e com espaços colapsados. Listas viram texto."""
    if isinstance(value, (list, tuple)):
        value = " ".join(normalize(item) for item in value)
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    elif isinstance(value, bool):
        value = "true" if value else "false"
    elif isinstance(value, float) and value.is_integer():
        value = int(value)
    text = unicodedata.normalize("NFKD", str(value).lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def match_value(actual: Any, matcher: Any) -> bool:
    """Compara um argumento com um matcher (ver evals/README.md)."""
    if actual is None:
        return False
    text = normalize(actual)
    if isinstance(matcher, dict):
        if "contains" in matcher:
            return normalize(matcher["contains"]) in text
        if "contains_all" in matcher:
            return all(normalize(item) in text for item in matcher["contains_all"])
        if "any_of" in matcher:
            return any(text == normalize(item) for item in matcher["any_of"])
        if "regex" in matcher:
            return re.search(matcher["regex"], text) is not None
        raise ValueError("matcher desconhecido: {}".format(matcher))
    return text == normalize(matcher)


def match_args(
    actual: Dict[str, Any], expected: Optional[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """Todos os argumentos-chave batem? Devolve (ok, nomes que falharam)."""
    if not expected:
        return True, []
    failed = [key for key, matcher in expected.items() if not match_value(actual.get(key), matcher)]
    return not failed, failed


# ------------------------------------------------------------------ metadados


def git_commit() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = proc.stdout.strip()
    if not commit:
        return None
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    return commit + ("-dirty" if dirty else "")


def ollama_info(url: str) -> Dict[str, Any]:
    import httpx

    info: Dict[str, Any] = {"url": url, "version": None, "models": []}
    try:
        info["version"] = httpx.get(url + "/api/version", timeout=5.0).json().get("version")
        tags = httpx.get(url + "/api/tags", timeout=5.0).json()
        info["models"] = sorted(item.get("name", "") for item in tags.get("models", []))
    except Exception as exc:  # noqa: BLE001 - metadado opcional
        info["error"] = str(exc)
    return info


def model_available(url: str, model: str) -> bool:
    names = ollama_info(url).get("models") or []
    wanted = model if ":" in model else model + ":latest"
    return wanted in names or model in names


def unload_model(url: str, model: str) -> None:
    """Pede ao Ollama para tirar o modelo da GPU (um modelo por vez)."""
    import httpx

    try:
        httpx.post(url + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=30.0)
    except Exception:  # noqa: BLE001
        pass


def gpu_name() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else None


def run_metadata(url: str) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        # platform.platform() relata "Windows-10" também no Windows 11 (build 22000+).
        "platform": platform.platform(),
        "gpu": gpu_name(),
        "ollama": {k: v for k, v in ollama_info(url).items() if k != "models"},
    }


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-")


def save_json(payload: Dict[str, Any], name: str, out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    with open(str(path), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def fmt_pct(value: Optional[float], hits: Optional[int] = None, total: Optional[int] = None) -> str:
    """0.8125 -> '81,2%' (uma casa; com '(13/16)' se os contadores vierem)."""
    if value is None:
        return "—"
    text = "{:.1f}%".format(value * 100).replace(".", ",")
    if hits is not None and total is not None:
        text += " ({}/{})".format(hits, total)
    return text


def fmt_num(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "—"
    return "{:.{}f}".format(value, digits).replace(".", ",")
