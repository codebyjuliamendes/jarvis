#!/usr/bin/env python3
"""Demonstração da observabilidade com o agente real (Ollama), sem Telegram.

Cria uma pasta de trabalho TEMPORÁRIA com dois arquivos fictícios, roda alguns
pedidos pelo ``Agent`` de verdade (mesmo prompt, mesmas ferramentas, mesmo laço
do bot) e imprime o texto que o ``/metrics`` mostraria. O ``data/`` real não é
tocado: o banco de métricas fica na pasta temporária (ou em ``--db``).

Uso:
    .venv/Scripts/python.exe scripts/demo_metrics.py               # qwen2.5:3b
    .venv/Scripts/python.exe scripts/demo_metrics.py --model qwen2.5:1.5b
    .venv/Scripts/python.exe scripts/demo_metrics.py --log-format json

Só ferramentas de arquivo, restritas à pasta temporária; terminal desligado.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from jarvis.agent import Agent  # noqa: E402
from jarvis.config import Settings  # noqa: E402
from jarvis.llm import OllamaClient  # noqa: E402
from jarvis.logging_setup import setup_logging  # noqa: E402
from jarvis.memory import Memory  # noqa: E402
from jarvis.metrics_store import MetricsStore, metrics_report  # noqa: E402
from jarvis.tools import build_registry  # noqa: E402

DEMO_TOOLS = ["list_dir", "read_file", "write_file", "find_files", "search_files"]

PROMPTS = [
    "Oi! Em uma frase: o que você consegue fazer por mim?",
    "Liste os arquivos da pasta de trabalho.",
    "Leia o arquivo notas.md e me diga quais são os próximos passos.",
    "Crie o arquivo resumo.txt com o texto: demo de observabilidade ok",
    # Arquivo inexistente: mostra uma ferramenta falhando nas métricas.
    "Leia o arquivo relatorio_2025.txt e resuma.",
]

NOTAS = """# Projeto Jarvis — notas (fictício)

Próximos passos:
1. Medir latência p95 do agente em uso real.
2. Trocar o host do Ollama para 127.0.0.1 no Windows.
3. Revisar as ferramentas que mais falham.
"""


def make_settings(root: Path, url: str, model: str) -> Settings:
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    return Settings(
        telegram_token="demo",
        allowed_user_ids={1},
        ollama_url=url,
        ollama_model=model,
        ollama_timeout=300.0,
        ollama_num_ctx=8192,
        max_iterations=6,
        history_limit=10,
        db_path=data / "jarvis.sqlite3",
        allowed_roots=[root],
        shell_enabled=False,
        shell_timeout=30,
        owner_name="Julia",
        timezone="America/Sao_Paulo",
        workspace=root,
        obsidian_vault=None,
        notion_token=None,
        notion_default_page=None,
        stripe_api_key=None,
        github_token=None,
        github_default_repo=None,
        google_credentials=root / "credentials.json",
        google_token=root / "token.json",
        rag_enabled=False,
    )


async def run(args: argparse.Namespace, root: Path) -> int:
    (root / "notas.md").write_text(NOTAS, encoding="utf-8")
    (root / "tarefas.txt").write_text("- revisar PR\n- escrever README\n", encoding="utf-8")

    settings = make_settings(root, args.url, args.model)
    llm = OllamaClient(settings.ollama_url, settings.ollama_model, num_ctx=settings.ollama_num_ctx)
    if not await llm.is_available():
        print("Ollama não respondeu em {}.".format(settings.ollama_url), file=sys.stderr)
        return 2

    db_path = Path(args.db) if args.db else settings.db_path.parent / "metrics.sqlite3"
    store = MetricsStore(db_path)
    memory = Memory(settings.db_path)
    agent = Agent(settings, build_registry(settings), memory, llm=llm, metrics=store)

    try:
        for index, prompt in enumerate(PROMPTS, start=1):
            started = time.perf_counter()
            reply = await agent.respond(
                index, prompt, remember=False, allow_tools=DEMO_TOOLS, source="demo"
            )
            print(
                "\n[{}] {}\n    -> ({:.1f} s) {}".format(
                    index,
                    prompt,
                    time.perf_counter() - started,
                    " ".join(reply.split())[:240],
                ),
                flush=True,
            )
        print("\n========== /metrics ==========")
        print(metrics_report(store, time.time()))
        print("==============================")
        if args.db or args.keep:
            print("Banco de métricas: {}".format(db_path))
    finally:
        memory.close()
        store.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--url", default=os.getenv("OLLAMA_URL") or "http://127.0.0.1:11434")
    parser.add_argument("--db", default="", help="arquivo SQLite das métricas (padrão: pasta temporária)")
    parser.add_argument("--log-format", default="text", choices=["text", "json"])
    parser.add_argument("--keep", action="store_true", help="não apaga a pasta temporária")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # acentos no console do Windows
        sys.stderr.reconfigure(encoding="utf-8")
    setup_logging(args.log_format, "INFO")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    root = Path(tempfile.mkdtemp(prefix="jarvis-demo-")).resolve()
    try:
        return asyncio.run(run(args, root))
    finally:
        if args.keep:
            print("Pasta da demo mantida em {}".format(root))
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
