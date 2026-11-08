from __future__ import annotations

import logging
import sys

import httpx

from .agent import Agent
from .briefing import BriefingService
from .config import Settings
from .logging_setup import setup_logging
from .memory import Memory
from .metrics_store import MetricsStore
from .rag import RagIndex
from .reminders import ReminderService
from .telegram_bot import JarvisBot
from .tools import build_registry

logger = logging.getLogger(__name__)


def _preflight(agent: Agent) -> None:
    url = agent.settings.ollama_url + "/api/tags"
    try:
        response = httpx.get(url, timeout=4.0)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        logger.warning(
            "Ollama não respondeu em %s. Rode `ollama serve` ou ./scripts/install_ollama.sh.",
            agent.settings.ollama_url,
        )
        return
    models = [
        item.get("name", "")
        for item in data.get("models", [])
        if item.get("name")
    ]
    wanted = agent.settings.ollama_model
    short = wanted.split(":")[0]
    if not any(model == wanted or model.split(":")[0] == short for model in models):
        logger.warning(
            "O modelo '%s' não está baixado. Rode: ollama pull %s",
            wanted,
            wanted,
        )


def main() -> int:
    settings = Settings.load()
    # Padrão "text" = o formato de sempre; JARVIS_LOG_FORMAT=json para uma linha
    # JSON por evento, com o id da requisição.
    setup_logging(settings.log_format, settings.log_level)
    # Não registra URLs do Telegram (contêm o token) nem detalhes de HTTP.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if not settings.telegram_token:
        print(
            "TELEGRAM_BOT_TOKEN não configurado.\n"
            "Crie um bot com o @BotFather, copie .env.example para .env e preencha.",
            file=sys.stderr,
        )
        return 1

    memory = Memory(settings.db_path)
    reminders = ReminderService(settings)
    try:
        reminders.start()
    except RuntimeError as exc:
        logger.warning("Lembretes desativados: %s", exc)

    rag = None
    if settings.rag_enabled:
        try:
            rag = RagIndex(settings)
        except Exception as exc:  # noqa: BLE001 - RAG é opcional
            logger.warning("Memória/RAG desativada: %s", exc)

    metrics = None
    if settings.metrics_enabled:
        try:
            metrics = MetricsStore(settings.resolved_metrics_db_path())
        except Exception as exc:  # noqa: BLE001 - métricas são opcionais
            logger.warning("Métricas desativadas: %s", exc)

    registry = build_registry(settings, reminders, rag)
    agent = Agent(settings, registry, memory, metrics=metrics)

    logger.info("Ferramentas carregadas: %s", ", ".join(registry.names()))
    try:
        _preflight(agent)
    except Exception:  # noqa: BLE001
        logger.debug("Preflight falhou", exc_info=True)

    briefing = BriefingService(settings, agent, reminders)
    try:
        briefing.start()
    except Exception:  # noqa: BLE001
        logger.warning("Briefing diário desativado", exc_info=True)

    bot = JarvisBot(settings, agent, memory, reminders, briefing, rag)
    try:
        bot.run()
    finally:
        if metrics is not None:
            metrics.close()
        briefing.close()
        if rag is not None:
            rag.close()
        reminders.close()
        memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
