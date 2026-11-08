from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .config import Settings
from .llm import OllamaClient, OllamaError, OllamaUnavailable, ToolsNotSupported
from .memory import Memory
from .tool_parsing import extract_text_tool, parse_tool_call
from .tools.base import ToolRegistry


class Agent:
    def __init__(
        self,
        settings: Settings,
        tools: ToolRegistry,
        memory: Memory,
        llm: Optional[OllamaClient] = None,
    ) -> None:
        self.settings = settings
        self.tools = tools
        self.memory = memory
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
        for _ in range(limit):
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
                return str(exc)
            except OllamaError as exc:
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

            final = content
            break

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
