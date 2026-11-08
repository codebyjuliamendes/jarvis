#!/usr/bin/env python3
"""Demonstração do RAG com citação de fontes, com o agente real (Ollama), sem Telegram.

Indexa os documentos fictícios de evals/fixtures/ numa pasta TEMPORÁRIA com a
configuração padrão do RAG (ou --mode), roda algumas perguntas pelo ``Agent`` de
verdade, só com a ferramenta ``knowledge_search``, e imprime a resposta e se ela
citou a fonte no formato pedido. O ``data/`` real não é tocado.

    .venv/Scripts/python.exe scripts/demo_rag.py                  # qwen2.5:3b
    .venv/Scripts/python.exe scripts/demo_rag.py --mode hybrid --show-tool
"""
from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from evals import common  # noqa: E402
from jarvis.agent import Agent  # noqa: E402
from jarvis.llm import OllamaClient  # noqa: E402
from jarvis.memory import Memory  # noqa: E402
from jarvis.rag import RAG_MODES, RagIndex  # noqa: E402
from jarvis.tools import build_registry  # noqa: E402

# As perguntas dizem "nos meus documentos": sem isso, o qwen2.5:3b responde de cabeça
# sem chamar knowledge_search (medido em 08/11/2026, 0 de 3 chamadas; ver
# evals/PENDENCIAS.md). A demo mostra a citação, não a decisão de buscar.
QUESTIONS = [
    ("Nos meus documentos, qual o valor mensal que eu pago de aluguel?", "contrato_aluguel.pdf"),
    ("Procura nas minhas notas: até quando dá para cancelar o hotel em Alfama sem pagar nada?",
     "viagem_lisboa.md"),
    ("Segundo o manual que eu guardei, como faço o reset de fábrica do roteador?",
     "manual_roteador.txt"),
]
CITATION = re.compile(r"fonte:\s*([\w.\-]+)", re.IGNORECASE)


async def run(args: argparse.Namespace, root: Path) -> int:
    corpus = root / "corpus"
    shutil.copytree(str(common.FIXTURES_DIR), str(corpus))
    settings = common.make_eval_settings(root, model=args.model, ollama_url=args.url)
    settings = replace(settings, shell_enabled=False, max_iterations=4)
    if args.mode:
        settings = replace(settings, rag_mode=args.mode)
    llm = OllamaClient(settings.ollama_url, settings.ollama_model, num_ctx=settings.ollama_num_ctx)
    if not await llm.is_available():
        print("Ollama não respondeu em {}.".format(settings.ollama_url), file=sys.stderr)
        return 2

    rag = RagIndex(settings)
    memory = Memory(settings.db_path)
    try:
        print("RAG: {}".format(rag.describe()))
        print(rag.index(paths=[corpus], force=True).splitlines()[0])
        registry = build_registry(settings, rag=rag)
        if args.show_tool:
            original = registry.execute

            async def traced(name, arguments, *rest, **kwargs):
                output = await original(name, arguments, *rest, **kwargs)
                print("\n    [ferramenta {} {}]\n    {}".format(
                    name, arguments, "\n    ".join(output.splitlines()[:6])))
                return output

            registry.execute = traced  # type: ignore[assignment]
        agent = Agent(settings, registry, memory, llm=llm)
        cited_right = 0
        for index, (question, expected) in enumerate(QUESTIONS, start=1):
            started = time.perf_counter()
            reply = await agent.respond(
                index, question, remember=False, allow_tools=["knowledge_search"], source="demo"
            )
            cited = CITATION.findall(reply)
            ok = expected in cited
            cited_right += int(ok)
            print("\n[{}] {}\n    -> ({:.1f} s) {}\n    citou {}: {}".format(
                index, question, time.perf_counter() - started, reply.strip().replace("\n", "\n       "),
                expected, "sim" if ok else "não (citações: {})".format(cited or "nenhuma")))
        print("\nRespostas que citaram o arquivo certo: {}/{}".format(cited_right, len(QUESTIONS)))
    finally:
        memory.close()
        rag.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--url", default=common.DEFAULT_OLLAMA_URL)
    parser.add_argument("--mode", choices=RAG_MODES, default=None)
    parser.add_argument("--show-tool", action="store_true", help="mostra a saída da ferramenta")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        return asyncio.run(run(args, Path(tmp).resolve()))


if __name__ == "__main__":
    sys.exit(main())
