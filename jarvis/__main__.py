from __future__ import annotations

import logging
import sys

import httpx

from .agent import Agent
from .briefing import BriefingService
from .config import Settings
from .memory import Memory
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Não registra URLs do Telegram (contêm o token) nem detalhes de HTTP.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    settings = Settings.load()
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

    registry = build_registry(settings, reminders, rag)
    agent = Agent(settings, registry, memory)

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
        briefing.close()
        if rag is not None:
            rag.close()
        reminders.close()
        memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
