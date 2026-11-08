#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from jarvis.agent import Agent  # noqa: E402
from jarvis.briefing import BriefingService  # noqa: E402
from jarvis.config import Settings  # noqa: E402
from jarvis.llm import ToolsNotSupported  # noqa: E402
from jarvis.memory import Memory  # noqa: E402
from jarvis.rag import RagIndex  # noqa: E402
from jarvis.reminders import ReminderService  # noqa: E402
from jarvis.tools import build_registry  # noqa: E402
from jarvis.tools.reminders import parse_when  # noqa: E402

IS_WINDOWS = os.name == "nt"


def make_settings(root: Path) -> Settings:
    vault = root / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    return Settings(
        telegram_token="teste",
        allowed_user_ids={1},
        ollama_url="http://localhost:11434",
        ollama_model="fake:latest",
        ollama_timeout=5.0,
        ollama_num_ctx=2048,
        max_iterations=5,
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


class ScriptedLLM:
    def __init__(self, responses: List[Any]) -> None:
        self.responses = list(responses)
        self.tools_supported = True
        self.calls: List[Any] = []

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.calls.append((messages, tools))
        if not self.responses:
            return {"content": "fim"}
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class NoNativeToolsLLM(ScriptedLLM):
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if tools is not None:
            self.tools_supported = False
            raise ToolsNotSupported("modelo sem suporte a tools")
        return await super().chat(messages, tools)


class FakeRag(RagIndex):
    """RAG sem Ollama: embeddings determinísticos para o smoke test."""

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * 64
        for word in re.findall(r"\w+", text.lower()):
            vector[sum(ord(char) for char in word) % 64] += 1.0
        return vector

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Sem isto a indexação ia ao Ollama real (/api/embed) quando ele está
        # ligado com o nomic-embed-text, e os vetores de 768 dimensões não
        # batiam com os de 64 da busca: o teste dependia do estado da máquina.
        return [self._embed(text) for text in texts]


async def run() -> int:
    checks = 0
    skipped: List[str] = []

    def check(condition: bool, label: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError("FALHOU: " + label)
        checks += 1
        print("ok - " + label)

    with tempfile.TemporaryDirectory() as tmp:
        # resolve(): no Windows o TEMP pode vir em formato curto 8.3 (ex.: ABCDEF~1)
        # e as ferramentas comparam caminhos já resolvidos, como o Settings.load faz.
        root = Path(tmp).resolve()
        settings = make_settings(root)
        reminders = ReminderService(settings)
        reminders.start()
        rag = FakeRag(replace(settings, rag_roots=[root]))
        registry = build_registry(settings, reminders, rag)
        memory = Memory(settings.db_path)
        try:
            check(len(registry.names()) >= 25, "registro com >= 25 ferramentas")
            check("run_shell" in registry.names(), "tool run_shell presente")
            check("web_search" in registry.names(), "tool web_search presente")
            check("send_file" in registry.names(), "tool send_file presente")
            check("reminder" in registry.names(), "tool reminder presente")
            check("knowledge_search" in registry.names(), "tool knowledge_search presente")
            check("read_pdf" in registry.names(), "tool read_pdf presente")
            check("read_docx" in registry.names(), "tool read_docx presente")
            check("read_image" in registry.names(), "tool read_image presente")
            check("microsoft" in registry.names(), "tool microsoft presente")
            check("speak" in registry.names(), "tool speak presente")

            result = await registry.execute(
                "write_file", {"path": "src/nota.txt", "content": "linha 1\nlinha 2\n"}
            )
            check("escrito" in result.lower(), "write_file cria arquivo")
            check((root / "src" / "nota.txt").exists(), "arquivo existe no disco")

            result = await registry.execute("read_file", {"path": "src/nota.txt"})
            check("1: linha 1" in result and "2: linha 2" in result, "read_file numera linhas")

            result = await registry.execute(
                "replace_in_file",
                {"path": "src/nota.txt", "old": "linha 2", "new": "linha dois"},
            )
            check("Substituído" in result, "replace_in_file edita")

            result = await registry.execute("list_dir", {"path": "src"})
            check("nota.txt" in result, "list_dir lista")

            result = await registry.execute(
                "search_files", {"pattern": "linha dois", "path": "src", "glob": "*.txt"}
            )
            check("nota.txt" in result, "search_files encontra conteúdo")

            result = await registry.execute(
                "find_files", {"name": "*.txt", "path": "src"}
            )
            check("nota.txt" in result, "find_files encontra por nome")

            if IS_WINDOWS:
                # run_shell usa executable="/bin/bash" (projeto feito para Linux/macOS);
                # no Windows nativo esse caminho não existe. Pendência registrada em
                # evals/PENDENCIAS.md. As checagens de bloqueio abaixo continuam valendo.
                skipped.append("run_shell executa (usa /bin/bash; não suportado no Windows)")
                print("pulado - run_shell executa (Windows: run_shell depende de /bin/bash)")
            else:
                result = await registry.execute("run_shell", {"command": "echo oi"})
                check("oi" in result and "exit code: 0" in result, "run_shell executa")

            result = await registry.execute("run_shell", {"command": "sudo ls"})
            check("Bloqueado" in result, "run_shell bloqueia sudo")

            # Precisa ser um arquivo que EXISTE fora da raiz: no Windows /etc/passwd
            # não existe e o erro seria "caminho não existe", não o bloqueio.
            outside = (
                os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "win.ini")
                if IS_WINDOWS
                else "/etc/passwd"
            )
            result = await registry.execute("read_file", {"path": outside})
            check("fora das pastas permitidas" in result, "bloqueia leitura fora da raiz")

            result = await registry.execute(
                "write_file", {"path": ".ssh/authorized_keys", "content": "x"}
            )
            check("bloqueado" in result.lower(), "bloqueia escrita em pasta sensível")

            if IS_WINDOWS:
                await registry.execute("git", {"args": ["init", "-q", "."]})
            else:
                await registry.execute("run_shell", {"command": "git init -q ."})
            result = await registry.execute("git", {"args": ["status", "--short"]})
            check("exit code: 0" in result, "git status funciona")

            result = await registry.execute("web_fetch", {"url": "http://localhost:8080"})
            check("Bloqueado" in result, "web_fetch bloqueia localhost")
            result = await registry.execute("web_fetch", {"url": "ftp://example.com"})
            check("Bloqueado" in result, "web_fetch exige http/https")
            result = await registry.execute("send_file", {"path": "src/nota.txt"})
            check("chat" in result.lower(), "send_file sem chat orienta")

            now = datetime.now().astimezone()
            when = parse_when("10m", settings.timezone)
            check(
                580 <= (when - now).total_seconds() <= 620,
                "parse_when entende minutos",
            )
            tomorrow = parse_when("amanhã 09:00", settings.timezone)
            check(
                tomorrow.hour == 9 and tomorrow.minute == 0,
                "parse_when entende 'amanhã HH:MM'",
            )
            absolute = parse_when("2030-01-02 18:30", settings.timezone)
            check(
                absolute.year == 2030 and absolute.hour == 18,
                "parse_when entende data absoluta",
            )
            invalid_rejected = False
            try:
                parse_when("quando der vontade", settings.timezone)
            except ValueError:
                invalid_rejected = True
            check(invalid_rejected, "parse_when rejeita texto inválido")

            result = await registry.execute(
                "reminder",
                {"action": "add", "when": "5m", "text": "beber água"},
                context={"chat_id": 1},
            )
            check("agendado" in result, "reminder add via ferramenta")
            result = await registry.execute(
                "reminder", {"action": "list"}, context={"chat_id": 1}
            )
            check("beber água" in result, "reminder list via ferramenta")
            pending = reminders.list_pending(1)
            check(
                len(pending) == 1 and pending[0]["text"] == "beber água",
                "lembrete persistido no banco",
            )
            result = await registry.execute(
                "reminder",
                {"action": "delete", "id": pending[0]["id"]},
                context={"chat_id": 1},
            )
            check("cancelado" in result, "reminder delete via ferramenta")
            check(not reminders.list_pending(1), "lista vazia após cancelar")

            result = await registry.execute("github", {"action": "list_repos"})
            check("não configurado" in result, "github sem token explica configuração")

            result = await registry.execute("notion", {"action": "search", "query": "x"})
            check("não configurado" in result, "notion sem token explica configuração")

            result = await registry.execute("stripe", {"action": "overview"})
            check("não configurado" in result, "stripe sem chave explica configuração")

            result = await registry.execute("google", {"action": "calendar_list"})
            check("Google não configurado" in result, "google sem token explica configuração")

            result = await registry.execute("microsoft", {"action": "me"})
            check(
                "Microsoft 365 não configurado" in result,
                "microsoft sem client id explica configuração",
            )
            result = await registry.execute("speak", {}, context={"chat_id": 1})
            check("Informe o campo 'text'" in result, "speak exige texto")

            result = await registry.execute(
                "obsidian",
                {"action": "write", "path": "teste.md", "content": "oi mundo", "mode": "overwrite"},
            )
            check("Nota" in result, "obsidian write")
            result = await registry.execute(
                "obsidian", {"action": "read", "path": "teste.md"}
            )
            check("oi mundo" in result, "obsidian read")
            result = await registry.execute(
                "obsidian", {"action": "search", "query": "mundo"}
            )
            check("teste.md" in result, "obsidian search")
            result = await registry.execute(
                "obsidian", {"action": "list", "folder": ""}
            )
            check("teste.md" in result, "obsidian list")

            (root / "dados.csv").write_text("nome,valor\nA,10\nB,20\n", encoding="utf-8")
            result = await registry.execute(
                "spreadsheet", {"action": "read", "path": "dados.csv"}
            )
            check("A\t10" in result and "B\t20" in result, "spreadsheet lê CSV")

            memory.add(1, "user", "oi")
            memory.add(1, "assistant", "olá")
            check(len(memory.history(1)) == 2, "memória guarda histórico")
            memory.set_pref(1, "tts_auto", "1")
            check(memory.get_pref(1, "tts_auto") == "1", "memória guarda preferência")
            memory.clear(1)
            check(len(memory.history(1)) == 0, "memória limpa histórico")

            fake = ScriptedLLM(
                [
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "write_file",
                                    "arguments": {
                                        "path": "agente_nativo.txt",
                                        "content": "feito",
                                    },
                                }
                            }
                        ]
                    },
                    {"content": "Pronto, escrevi o arquivo."},
                ]
            )
            agent = Agent(settings, registry, memory, llm=fake)
            reply = await agent.respond(10, "crie agente_nativo.txt com feito")
            check("Pronto" in reply, "agente nativo devolve resposta final")
            check(
                (root / "agente_nativo.txt").read_text(encoding="utf-8") == "feito",
                "agente nativo executou a ferramenta",
            )
            check(len(fake.calls) == 2, "agente nativo fez 2 chamadas ao LLM")

            fake_text = ScriptedLLM(
                [
                    {
                        "content": (
                            '{"tool": "write_file", "args": '
                            '{"path": "agente_texto.txt", "content": "abc"}}'
                        )
                    },
                    {"content": "Feito no modo texto."},
                ]
            )
            fake_text.tools_supported = False
            agent_text = Agent(settings, registry, memory, llm=fake_text)
            reply = await agent_text.respond(11, "crie agente_texto.txt")
            check("Feito" in reply, "agente em modo texto devolve resposta final")
            check(
                (root / "agente_texto.txt").read_text(encoding="utf-8") == "abc",
                "agente em modo texto executou a ferramenta",
            )

            fallback = NoNativeToolsLLM(
                [
                    {
                        "content": '{"tool": "write_file", "args": {"path": "fallback.txt", "content": "ok"}}'
                    },
                    {"content": "Fallback concluído."},
                ]
            )
            fallback_agent = Agent(settings, registry, memory, llm=fallback)
            reply = await fallback_agent.respond(12, "crie fallback.txt")
            check("Fallback" in reply, "fallback automático quando tools não é suportado")
            check(
                (root / "fallback.txt").read_text(encoding="utf-8") == "ok",
                "fallback executou a ferramenta",
            )

            audit_path = root / "data" / "audit.log"
            check(audit_path.exists(), "auditoria cria arquivo de log")
            check(
                "write_file" in audit_path.read_text(encoding="utf-8"),
                "auditoria registra ferramenta executada",
            )

            import fitz

            pdf_doc = fitz.open()
            pdf_page = pdf_doc.new_page()
            pdf_page.insert_text((72, 72), "Prova de Calculo I - limite e derivada")
            pdf_path = root / "prova.pdf"
            pdf_doc.save(str(pdf_path))
            pdf_doc.close()
            result = await registry.execute("read_pdf", {"path": "prova.pdf"})
            check("limite e derivada" in result, "read_pdf extrai texto")

            from docx import Document as WordDocument

            word = WordDocument()
            word.add_heading("Relatorio", 0)
            word.add_paragraph("Faturamento do mes: R$ 10.000")
            word_path = root / "relatorio.docx"
            word.save(str(word_path))
            result = await registry.execute("read_docx", {"path": "relatorio.docx"})
            check("Faturamento" in result, "read_docx extrai texto")

            memoria = root / "memoria"
            memoria.mkdir()
            (memoria / "reuniao.md").write_text(
                "A reunião do projeto Falcon será na sexta-feira às 14h.",
                encoding="utf-8",
            )
            result = rag.index(paths=[memoria], force=True)
            check("Indexação concluída" in result, "RAG indexa arquivo de teste")
            result = await registry.execute(
                "knowledge_search", {"query": "quando é a reunião do Falcon?"}
            )
            check("reuniao.md" in result, "knowledge_search encontra trecho")

            briefing_llm = ScriptedLLM([{"content": "Briefing de teste"}])
            briefing_agent = Agent(settings, registry, memory, llm=briefing_llm)
            briefing = BriefingService(settings, briefing_agent, reminders)
            reply = await briefing.generate(1)
            check("Briefing de teste" in reply, "briefing gera texto pelo agente")
            briefing.close()

        finally:
            # Fecha as conexões SQLite antes de o TemporaryDirectory apagar a
            # pasta: no Windows um arquivo aberto não pode ser removido (WinError 32).
            memory.close()
            rag.close()
            reminders.close()

    print("\n{} verificações passaram.".format(checks))
    if skipped:
        print("{} verificação(ões) pulada(s) nesta plataforma:".format(len(skipped)))
        for label in skipped:
            print("  - " + label)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run()))
    except AssertionError as exc:
        print("\n" + str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
