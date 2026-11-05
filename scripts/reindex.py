#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from jarvis.config import Settings  # noqa: E402
from jarvis.rag import RagIndex  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reindexa a memória local do Jarvis (RAG)."
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Pastas/arquivos específicos (pode repetir). Padrão: JARVIS_RAG_ROOTS.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reindexa mesmo os arquivos que não mudaram.",
    )
    parser.add_argument("--max-files", type=int, default=400)
    parser.add_argument("--max-chunks", type=int, default=4000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    settings = Settings.load()
    if not settings.rag_enabled:
        print("RAG desativado (JARVIS_RAG_ENABLED=false).")
        return 0
    paths = [Path(item).expanduser() for item in args.path] or None
    rag = RagIndex(settings)
    try:
        print(
            rag.index(
                paths=paths,
                force=args.force,
                max_files=max(1, args.max_files),
                max_chunks=max(1, args.max_chunks),
            )
        )
        print("Estatísticas: {}".format(rag.stats()))
    finally:
        rag.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
