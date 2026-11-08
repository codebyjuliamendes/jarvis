from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple

from . import observability
from .config import Settings
from .llm import OllamaClient, OllamaError, OllamaUnavailable, ToolsNotSupported
from .memory import Memory
from .tool_parsing import extract_text_tool, parse_tool_call
from .tools.base import ToolRegistry
from .tools.knowledge import ensure_citation

if TYPE_CHECKING:  # evita importar sqlite3 só para a anotação
    from .metrics_store import MetricsStore

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        settings: Settings,
        tools: ToolRegistry,
        memory: Memory,
        llm: Optional[OllamaClient] = None,
        metrics: Optional[MetricsStore] = None,
    ) -> None:
        self.settings = settings
        self.tools = tools
        self.memory = memory
        # Opcional: sem gravador (testes, smoke, evals) nada é persistido.
        self.metrics = metrics
        self.llm = llm or OllamaClient(
            url=settings.ollama_url,
            model=settings.ollama_model,
            timeout=settings.ollama_timeout,
            num_ctx=settings.ollama_num_ctx,
        )

    def _system_prompt(
        self,
        text_mode: bool = False,
        allow_tools: Optional[Iterable[str]] = None,
    ) -> str:
        now = datetime.now().astimezone()
        roots = ", ".join(str(root) for root in self.settings.allowed_roots)
        shell = "habilitado" if self.settings.shell_enabled else "desabilitado"
        prompt = (
            "Você é o Jarvis, assistente pessoal de {owner}. "
            "Você roda na máquina dela e conversa pelo Telegram.\n\n"
            "PERSONALIDADE: direto, prático e em português do Brasil. Sem enrolação.\n\n"
            "CONTEXTO:\n"
            "- Data/hora: {now}\n"
            "- Fuso configurado: {tz}\n"
            "- Pasta de trabalho: {workspace}\n"
            "- Pastas permitidas: {roots}\n"
            "- Terminal: {shell}\n\n"
            "COMO AGIR:\n"
            "- Use as ferramentas sempre que precisar de dados reais (arquivos, terminal, git, "
            "GitHub, Notion, Obsidian, Google Agenda/Gmail, Microsoft 365/Outlook/Teams, "
            "Stripe, planilhas). Nunca invente.\n"
            "- Para perguntas sobre conteúdo antigo, notas, PDFs ou documentos, use "
            "knowledge_search antes de responder. Use read_pdf/read_docx/read_image para ler "
            "documentos específicos.\n"
            "- Ao responder com o que veio do knowledge_search, cite as fontes no fim: "
            "(fonte: arquivo, trecho N). Não invente fontes.\n"
            "- Antes de ações destrutivas ou irreversíveis (apagar, sobrescrever, "
            "git push --force, enviar e-mail), peça confirmação à usuária.\n"
            "- Se uma ferramenta não estiver configurada, diga exatamente o que falta no .env.\n"
            "- Respostas curtas para Telegram. Use listas. Cite caminhos de arquivos.\n"
            "- Quando fizer sentido, termine com o próximo passo em uma linha.\n"
            "- Se a tarefa for grande, divida em etapas e execute uma por vez.\n"
        ).format(
            owner=self.settings.owner_name,
            now=now.strftime("%d/%m/%Y %H:%M"),
            tz=self.settings.timezone,
            workspace=self.settings.workspace,
            roots=roots,
            shell=shell,
        )
        if text_mode:
            prompt += "\n" + self.tools.text_protocol_prompt(only=allow_tools)
        return prompt

    async def respond(
        self,
        chat_id: int,
        text: str,
        remember: bool = True,
        use_history: bool = True,
        allow_tools: Optional[Iterable[str]] = None,
        max_iterations: Optional[int] = None,
        source: str = "chat",
    ) -> str:
        model = str(getattr(self.llm, "model", None) or self.settings.ollama_model)
        with observability.trace_scope(model=model, source=source, chat_id=chat_id) as trace:
            try:
                reply = await self._respond(
                    trace,
                    chat_id,
                    text,
                    remember=remember,
                    use_history=use_history,
                    allow_tools=allow_tools,
                    max_iterations=max_iterations,
                )
            except BaseException as exc:
                trace.set_outcome("error", observability.short_error(exc))
                self._observe(trace)
                raise
            self._observe(trace)
            return reply

    def _observe(self, trace: observability.RequestTrace) -> None:
        """Loga o resumo da requisição e grava a métrica. Nunca levanta exceção."""
        try:
            trace.finish()
            row = trace.as_row()
            logger.info(
                "requisição concluída: %s em %.2fs (%d iterações, %d ferramenta(s))",
                row["outcome"],
                row["latency_s"],
                row["iterations"],
                row["tool_calls"],
                extra={
                    "fields": {
                        key: row[key]
                        for key in (
                            "source",
                            "model",
                            "mode",
                            "outcome",
                            "latency_s",
                            "llm_calls",
                            "prompt_tokens",
                            "completion_tokens",
                            "tokens_per_s",
                            "iterations",
                            "tool_calls",
                            "error",
                        )
                    }
                },
            )
        except Exception:  # noqa: BLE001 - medir nunca pode derrubar a resposta
            logger.debug("Falha ao logar a requisição", exc_info=True)
        if self.metrics is None:
            return
        try:
            self.metrics.record(trace)
        except Exception:  # noqa: BLE001
            logger.warning("Falha ao gravar métricas da requisição", exc_info=True)

    async def _respond(
        self,
        trace: observability.RequestTrace,
        chat_id: int,
        text: str,
        remember: bool,
        use_history: bool,
        allow_tools: Optional[Iterable[str]],
        max_iterations: Optional[int],
    ) -> str:
        history = (
            self.memory.history(chat_id, self.settings.history_limit)
            if use_history
            else []
        )
        tool_filter: Optional[Set[str]] = (
            set(allow_tools) if allow_tools is not None else None
        )
        limit = max(1, int(max_iterations or self.settings.max_iterations))
        text_mode = not self.llm.tools_supported
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": self._system_prompt(
                    text_mode=text_mode, allow_tools=tool_filter
                ),
            }
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": text})

        final = ""
        rag_outputs: List[str] = []  # saídas do knowledge_search, para garantir a citação
        for _ in range(limit):
            trace.iterations += 1
            try:
                message = await self.llm.chat(
                    messages,
                    tools=(
                        None
                        if text_mode
                        else self.tools.schemas(only=tool_filter)
                    ),
                )
            except ToolsNotSupported:
                text_mode = True
                messages[0] = {
                    "role": "system",
                    "content": self._system_prompt(
                        text_mode=True, allow_tools=tool_filter
                    ),
                }
                continue
            except OllamaUnavailable as exc:
                trace.mode = "text" if text_mode else "native"
                trace.set_outcome("error", observability.short_error(exc))
                return str(exc)
            except OllamaError as exc:
                trace.mode = "text" if text_mode else "native"
                trace.set_outcome("error", observability.short_error(exc))
                return "Erro no Ollama: {}".format(exc)

            tool_calls = message.get("tool_calls") or []
            content = (message.get("content") or "").strip()

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                )
                for call in tool_calls:
                    name, args = self._parse_tool_call(call)
                    result = await self.tools.execute(
                        name, args, context={"chat_id": chat_id}
                    )
                    if name == "knowledge_search":
                        rag_outputs.append(result)
                    messages.append(
                        {"role": "tool", "content": result, "tool_name": name}
                    )
                continue

            if text_mode:
                parsed = self._extract_text_tool(content)
                if parsed is not None:
                    name, args = parsed
                    result = await self.tools.execute(
                        name, args, context={"chat_id": chat_id}
                    )
                    if name == "knowledge_search":
                        rag_outputs.append(result)
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Resultado da ferramenta '{}':\n{}\n\n"
                                "Continue a tarefa. Se já terminou, responda em texto normal, "
                                "sem JSON."
                            ).format(name, result),
                        }
                    )
                    continue

            # Modelos pequenos nem sempre citam: se usou o RAG e não citou, o agente
            # anexa as fontes consultadas (ver jarvis/tools/knowledge.py).
            final = ensure_citation(content, rag_outputs) if rag_outputs else content
            break

        trace.mode = "text" if text_mode else "native"
        trace.set_outcome("ok" if final else "max_iterations")
        if not final:
            final = (
                "Não consegui concluir em {} passos. "
                "Tente dividir a tarefa em partes menores."
            ).format(limit)

        if remember:
            self.memory.add(chat_id, "user", text)
            self.memory.add(chat_id, "assistant", final)
        return final

    # Mantidos como métodos estáticos por compatibilidade; a lógica vive em
    # jarvis/tool_parsing.py para ser reutilizada pelas avaliações.
    @staticmethod
    def _parse_tool_call(call: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        return parse_tool_call(call)

    @staticmethod
    def _extract_text_tool(content: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        return extract_text_tool(content)
