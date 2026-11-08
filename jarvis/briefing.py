from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

import pytz

from .agent import Agent
from .config import Settings
from .reminders import ReminderService
from .telegram_api import send_message

logger = logging.getLogger(__name__)

WEEKDAYS = [
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
]

BRIEFING_TOOLS = {
    "google",
    "microsoft",
    "obsidian",
    "notion",
    "reminder",
    "web_search",
    "web_fetch",
    "knowledge_search",
}

BRIEFING_PROMPT = """Você é o Jarvis, assistente pessoal de {owner}.
Gere o briefing matinal de hoje, {date}.

Contexto já coletado pelo sistema:
{context}

Tarefa:
1. Se o Google estiver configurado, consulte a agenda de hoje e os e-mails importantes/não lidos das últimas 24h.
   Se o Microsoft 365 estiver configurado, consulte também o Outlook e o calendário.
2. Considere os lembretes pendentes, notas recentes do Obsidian e tarefas do Notion que façam sentido para hoje.
3. Se houver mensagens importantes no Teams, inclua um resumo curto.
4. Se algo for urgente ou tiver prazo hoje, destaque no topo.
5. Use web_search apenas se precisar de contexto externo (clima, notícias relevantes etc.).
6. Responda em português do Brasil, curto e prático, com as seções:
   Prioridades, Agenda, E-mails, Teams, Lembretes, Próximo passo.
7. Se uma integração não estiver configurada, escreva "não configurado" na seção e siga.
8. Não invente compromissos, e-mails, prazos ou tarefas. Use apenas o que as ferramentas retornarem.
9. Responda em TEXTO normal (markdown simples). NÃO responda em JSON.
"""


class BriefingService:
    """Briefing diário enviado pelo Telegram via APScheduler."""

    def __init__(
        self,
        settings: Settings,
        agent: Agent,
        reminders: Optional[ReminderService] = None,
    ) -> None:
        self.settings = settings
        self.agent = agent
        self.reminders = reminders
        self._scheduler: Optional[Any] = None

    def _timezone(self):
        return pytz.timezone(self.settings.timezone)

    def start(self) -> None:
        if not self.settings.briefing_enabled:
            logger.info("Briefing diário desativado (JARVIS_BRIEFING_ENABLED=false).")
            return
        if not self.settings.allowed_user_ids:
            logger.warning("Briefing desativado: nenhum usuário autorizado configurado.")
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError as exc:
            logger.warning("Briefing desativado: APScheduler não instalado (%s).", exc)
            return

        timezone = self._timezone()
        self._scheduler = BackgroundScheduler(timezone=timezone)
        trigger = CronTrigger(
            hour=self.settings.briefing_hour,
            minute=self.settings.briefing_minute,
            timezone=timezone,
        )
        self._scheduler.add_job(
            self._run,
            trigger,
            id="daily-briefing",
            replace_existing=True,
            coalesce=True,
            misfire_grace_time=3600,
        )
        self._scheduler.start()
        logger.info(
            "Briefing diário agendado para %02d:%02d (%s).",
            self.settings.briefing_hour,
            self.settings.briefing_minute,
            self.settings.timezone,
        )

    def _context(self, chat_id: int) -> str:
        lines = []
        if self.reminders is not None and self.reminders.is_started:
            rows = self.reminders.list_pending(chat_id)
            if rows:
                lines.append("Lembretes pendentes:")
                for row in rows:
                    lines.append("- {}: {}".format(row.get("due_at"), row.get("text")))
            else:
                lines.append("Lembretes pendentes: nenhum.")
        else:
            lines.append("Lembretes pendentes: serviço indisponível.")

        google = "configurado" if self.settings.google_token.exists() else "não configurado"
        microsoft = "configurado" if (
            self.settings.ms_graph_client_id
            and self.settings.ms_graph_token is not None
            and self.settings.ms_graph_token.exists()
        ) else "não configurado"
        obsidian = (
            "configurado ({})".format(self.settings.obsidian_vault)
            if self.settings.obsidian_vault is not None
            else "não configurado"
        )
        notion = "configurado" if self.settings.notion_token else "não configurado"
        lines.append("Google Agenda/Gmail: {}".format(google))
        lines.append("Microsoft 365 (Outlook/Teams): {}".format(microsoft))
        lines.append("Obsidian: {}".format(obsidian))
        lines.append("Notion: {}".format(notion))
        lines.append("Workspace: {}".format(self.settings.workspace))
        return "\n".join(lines)

    async def generate(self, chat_id: int) -> str:
        now = datetime.now(self._timezone())
        date = "{} ({})".format(now.strftime("%d/%m/%Y"), WEEKDAYS[now.weekday()])
        prompt = BRIEFING_PROMPT.format(
            owner=self.settings.owner_name,
            date=date,
            context=self._context(chat_id),
        )
        return await self.agent.respond(
            chat_id,
            prompt,
            remember=False,
            use_history=False,
            allow_tools=BRIEFING_TOOLS,
            max_iterations=self.settings.briefing_max_iterations,
            source="briefing",
        )

    def _run(self) -> None:
        chat_id = sorted(self.settings.allowed_user_ids)[0]
        try:
            text = asyncio.run(self.generate(chat_id))
        except Exception:  # noqa: BLE001 - scheduler não deve morrer
            logger.exception("Falha ao gerar o briefing diário")
            text = "Não consegui gerar o briefing de hoje. Verifique os logs do Jarvis."
        if not (text or "").strip():
            text = "Briefing vazio: nenhuma informação coletada."
        ok, detail = send_message(self.settings.telegram_token, chat_id, text)
        if ok:
            logger.info("Briefing diário enviado para o chat %s", chat_id)
        else:
            logger.warning("Falha ao enviar briefing diário: %s", detail)

    def close(self) -> None:
        if self._scheduler is not None:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass
            self._scheduler = None
