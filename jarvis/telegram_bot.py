from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from . import observability
from .agent import Agent
from .briefing import BriefingService
from .config import Settings
from .memory import Memory
from .metrics_store import metrics_report
from .rag import RagIndex
from .reminders import ReminderService
from .telegram_api import send_audio
from .tools.reminders import parse_when
from .tts import TextToSpeech
from .utils import run_blocking
from .voice import VoiceTranscriber

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "Sou o seu Jarvis local. Falo com você pelo Telegram e executo tarefas na sua máquina.\n\n"
    "Posso:\n"
    "- ler/editar arquivos e rodar comandos no terminal\n"
    "- usar git e consultar GitHub\n"
    "- ler/escrever notas (Obsidian/Notion)\n"
    "- ver agenda e e-mails (Google e Microsoft 365)\n"
    "- consultar Stripe e planilhas\n"
    "- buscar e ler páginas na web\n"
    "- transcrever seus áudios e responder em texto ou áudio\n"
    "- ler PDFs, DOCX, planilhas e imagens (OCR)\n"
    "- buscar na memória local indexada (notas, PDFs, documentos)\n"
    "- criar lembretes que chegam aqui no Telegram\n"
    "- enviar um briefing diário no horário configurado\n\n"
    "Comandos:\n"
    "/start - apresentação\n"
    "/help - esta ajuda\n"
    "/status - modelo, Ollama e integrações\n"
    "/metrics - latência, erros e ferramentas (24 h e 7 dias)\n"
    "/briefing - gera o briefing de hoje agora\n"
    "/reindex - reindexa os arquivos na memória local\n"
    "/voice on|off - liga/desliga respostas em áudio\n"
    "/remind <quando> <texto> - cria lembrete (ex.: /remind 10m tomar água)\n"
    "/reminders - lista lembretes pendentes\n"
    "/reset - limpar o histórico desta conversa\n\n"
    "Dica: fale como pediria a uma pessoa. Ex.: \"roda os testes do projeto X\", "
    "\"o que tenho amanhã na agenda?\", \"resume as issues abertas do repo Y\".\n"
    "Você também pode mandar um áudio que eu transcrevo e respondo, "
    "ou um PDF/DOCX/imagem que eu leio."
)


def split_message(text: str, limit: int = 3900) -> List[str]:
    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > limit and current:
            chunks.append(current)
            current = ""
        if len(line) > limit:
            for start in range(0, len(line), limit):
                piece = line[start : start + limit]
                if len(current) + len(piece) > limit and current:
                    chunks.append(current)
                    current = ""
                current += piece
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks or [""]


def _safe_filename(name: str) -> str:
    base = Path(name).name or "arquivo"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return (cleaned or "arquivo")[:120]


class JarvisBot:
    def __init__(
        self,
        settings: Settings,
        agent: Agent,
        memory: Memory,
        reminders: Optional[ReminderService] = None,
        briefing: Optional[BriefingService] = None,
        rag: Optional[RagIndex] = None,
    ) -> None:
        self.settings = settings
        self.agent = agent
        self.memory = memory
        self.reminders = reminders
        self.briefing = briefing
        self.rag = rag
        self.voice = VoiceTranscriber(settings)
        self.tts = TextToSpeech(settings)
        self._locks: Dict[int, asyncio.Lock] = {}
        self._background_tasks = set()
        self.app = Application.builder().token(settings.telegram_token).build()
        self._register_handlers()

    def _lock_for(self, chat_id: int) -> asyncio.Lock:
        lock = self._locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[chat_id] = lock
        return lock

    def _register_handlers(self) -> None:
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("help", self._cmd_help))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("metrics", self._cmd_metrics))
        self.app.add_handler(CommandHandler("briefing", self._cmd_briefing))
        self.app.add_handler(CommandHandler("reindex", self._cmd_reindex))
        self.app.add_handler(CommandHandler("voice", self._cmd_voice))
        self.app.add_handler(CommandHandler("remind", self._cmd_remind))
        self.app.add_handler(CommandHandler("reminders", self._cmd_reminders))
        self.app.add_handler(CommandHandler("reset", self._cmd_reset))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text)
        )
        self.app.add_handler(
            MessageHandler(filters.VOICE | filters.AUDIO, self._on_voice)
        )
        self.app.add_handler(
            MessageHandler(filters.Document.ALL | filters.PHOTO, self._on_document)
        )
        self.app.add_handler(
            MessageHandler(filters.VIDEO, self._on_unsupported)
        )
        self.app.add_error_handler(self._on_error)

    async def _guard(self, update: Update) -> bool:
        user = update.effective_user
        chat = update.effective_chat
        message = update.message
        if user is None or chat is None or message is None:
            return False

        if not self.settings.allowed_user_ids:
            await message.reply_text(
                "Acesso ainda não configurado.\n\n"
                "Seu ID do Telegram é: {}\n\n"
                "Adicione em TELEGRAM_ALLOWED_USER_IDS no arquivo .env e reinicie o Jarvis.".format(
                    user.id
                )
            )
            logger.warning(
                "Nenhum TELEGRAM_ALLOWED_USER_IDS configurado; informando ID %s", user.id
            )
            return False

        if user.id not in self.settings.allowed_user_ids:
            logger.warning("Acesso negado para user_id=%s", user.id)
            await message.reply_text("Acesso negado. Seu ID: {}".format(user.id))
            return False

        return True

    async def _reply(self, update: Update, text: str) -> None:
        message = update.message
        if message is None:
            return
        for chunk in split_message(text or "(sem resposta)"):
            try:
                await message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
            except BadRequest:
                try:
                    await message.reply_text(chunk)
                except TelegramError:
                    logger.exception("Falha ao enviar mensagem")
            except TelegramError:
                logger.exception("Falha ao enviar mensagem")

    async def _cmd_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._guard(update):
            return
        await self._reply(
            update,
            "Oi, {}! Sou seu Jarvis. Estou rodando localmente com o modelo {}.\n\n"
            "Manda a tarefa. Se quiser ver o que está configurado, use /status.\n"
            "Se eu não souber algo, eu falo em vez de inventar.".format(
                self.settings.owner_name, self.settings.ollama_model
            ),
        )

    async def _cmd_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._guard(update):
            return
        await self._reply(update, HELP_TEXT)

    async def _cmd_reset(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._guard(update):
            return
        self.memory.clear(update.effective_chat.id)
        await self._reply(update, "Histórico desta conversa apagado.")

    async def _cmd_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._guard(update):
            return

        ollama_ok = await self.agent.llm.is_available()
        models = await self.agent.llm.list_models() if ollama_ok else []
        model_loaded = any(
            model == self.settings.ollama_model
            or model.split(":")[0] == self.settings.ollama_model.split(":")[0]
            for model in models
        )

        def mark(ok: bool) -> str:
            return "ok" if ok else "não configurado"

        integrations = [
            "shell: {}".format("on" if self.settings.shell_enabled else "off"),
            "obsidian: {}".format(mark(self.settings.obsidian_vault is not None)),
            "notion: {}".format(mark(bool(self.settings.notion_token))),
            "stripe: {}".format(mark(bool(self.settings.stripe_api_key))),
            "github: {}".format(mark(bool(self.settings.github_token))),
            "google: {}".format(mark(self.settings.google_token.exists())),
            "microsoft: {}".format(
                mark(
                    bool(self.settings.ms_graph_client_id)
                    and (
                        self.settings.ms_graph_token is not None
                        and self.settings.ms_graph_token.exists()
                    )
                )
            ),
        ]

        if not self.settings.whisper_enabled:
            voice_status = "desativada"
        elif VoiceTranscriber.available():
            voice_status = "on ({})".format(self.settings.whisper_model)
        else:
            voice_status = "instale ./scripts/setup_voice.sh"

        if not self.settings.tts_enabled:
            tts_status = "desativado"
        elif self.tts.available():
            tts_status = "on ({} - {})".format(
                "auto" if self.settings.tts_auto else "sob demanda",
                self.settings.tts_voice,
            )
        else:
            tts_status = "instale edge-tts"

        if self.reminders is None:
            reminders_status = "indisponível"
        else:
            pending = len(self.reminders.list_pending(update.effective_chat.id))
            reminders_status = "{} pendente(s)".format(pending)

        if self.briefing is not None and self.settings.briefing_enabled:
            briefing_status = "{:02d}:{:02d} ({})".format(
                self.settings.briefing_hour,
                self.settings.briefing_minute,
                self.settings.timezone,
            )
        else:
            briefing_status = "desativado"

        if self.rag is None:
            rag_status = "desativada"
        else:
            try:
                stats = await run_blocking(self.rag.stats)
                rag_status = "{} documento(s) / {} trecho(s)".format(
                    stats["documents"], stats["chunks"]
                )
            except Exception as exc:  # noqa: BLE001
                rag_status = "erro: {}".format(exc)

        lines = [
            "Modelo: {}".format(self.settings.ollama_model),
            "Ollama ({}): {}".format(
                self.settings.ollama_url, "online" if ollama_ok else "offline"
            ),
        ]
        if ollama_ok:
            lines.append("Modelo baixado: {}".format("sim" if model_loaded else "não"))
            if not model_loaded:
                lines.append(
                    "Baixe com: ollama pull {}".format(self.settings.ollama_model)
                )
        lines.append("Ferramentas: {} ({})".format(
            len(self.agent.tools.names()), ", ".join(self.agent.tools.names())
        ))
        lines.append("Voz: {}".format(voice_status))
        lines.append("TTS: {}".format(tts_status))
        lines.append("Lembretes: {}".format(reminders_status))
        lines.append("Briefing: {}".format(briefing_status))
        lines.append("Memória/RAG: {}".format(rag_status))
        lines.append("Workspace: {}".format(self.settings.workspace))
        lines.append("Integrações: " + " | ".join(integrations))
        if not self.settings.allowed_user_ids:
            lines.append("[atenção] Nenhum usuário autorizado configurado.")
        await self._reply(update, "\n".join(lines))

    async def _cmd_metrics(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._guard(update):
            return
        store = getattr(self.agent, "metrics", None)
        if store is None:
            await self._reply(
                update,
                "Métricas desativadas. Ligue com JARVIS_METRICS_ENABLED=true e reinicie.",
            )
            return
        try:
            text = await run_blocking(metrics_report, store, time.time())
        except Exception as exc:  # noqa: BLE001
            logger.exception("Falha ao ler as métricas")
            await self._reply(update, "Não consegui ler as métricas: {}".format(exc))
            return
        # Bloco monoespaçado: alinha as colunas e evita que "_" vire itálico.
        await self._reply(update, "```\n{}\n```".format(text))

    async def _keep_typing(
        self, context: ContextTypes.DEFAULT_TYPE, chat_id: int
    ) -> None:
        try:
            while True:
                await context.bot.send_chat_action(
                    chat_id=chat_id, action=ChatAction.TYPING
                )
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            return
        except TelegramError:
            return

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _index_upload(self, path: Path) -> None:
        try:
            result = await run_blocking(
                self.rag.index, [path], max_files=1, max_chunks=300
            )
            logger.info(
                "Upload indexado (%s): %s", path.name, result.replace("\n", " | ")
            )
        except Exception:  # noqa: BLE001 - indexação é opcional
            logger.exception("Falha ao indexar upload %s", path)

    async def _maybe_send_audio(self, chat_id: int, text: str) -> None:
        if not self.settings.tts_enabled or not self.tts.available():
            return
        pref = self.memory.get_pref(chat_id, "tts_auto")
        enabled = self.settings.tts_auto if pref is None else pref == "1"
        if not enabled:
            return
        content = (text or "").strip()
        if not content:
            return
        path = (
            Path(self.settings.db_path).parent
            / "tts"
            / "tts-{}-{}.mp3".format(
                chat_id, datetime.now().strftime("%Y%m%d%H%M%S%f")
            )
        )
        try:
            await self.tts.save(content, path)
            ok, detail = await run_blocking(
                send_audio, self.settings.telegram_token, chat_id, path
            )
            if not ok:
                logger.warning("Falha ao enviar áudio TTS: %s", detail)
        except Exception:  # noqa: BLE001 - áudio é opcional
            logger.exception("Falha ao gerar áudio TTS")
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    async def _run_agent(
        self,
        update: Update,
        chat_id: int,
        text: str,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        lock = self._lock_for(chat_id)
        if lock.locked():
            await self._reply(
                update, "Ainda estou processando a mensagem anterior. Um instante."
            )
            return

        # Um id por mensagem: o agente reaproveita este, então os logs do envio
        # da resposta (e do áudio) também saem com o mesmo request_id.
        with observability.request_id_scope():
            async with lock:
                typing_task = asyncio.create_task(self._keep_typing(context, chat_id))
                try:
                    reply = await self.agent.respond(chat_id, text)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Erro inesperado no agente")
                    reply = "Deu erro aqui: {}".format(exc)
                finally:
                    typing_task.cancel()

            await self._reply(update, reply)
            await self._maybe_send_audio(chat_id, reply)

    async def _on_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._guard(update):
            return
        message = update.message
        chat = update.effective_chat
        if message is None or chat is None:
            return
        text = (message.text or "").strip()
        if not text:
            return

        await self._run_agent(update, chat.id, text, context)

    async def _on_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._guard(update):
            return
        message = update.message
        chat = update.effective_chat
        if message is None or chat is None:
            return
        media = message.voice or message.audio
        if media is None:
            return

        if not self.settings.whisper_enabled:
            await self._reply(
                update,
                "Transcrição de áudio está desativada (JARVIS_WHISPER_ENABLED=false).",
            )
            return
        if not VoiceTranscriber.available():
            await self._reply(
                update,
                "Para eu entender áudios, instale a transcrição local:\n"
                "./scripts/setup_voice.sh\n(depois reinicie o Jarvis)",
            )
            return

        suffix = ".oga" if message.voice else ".mp3"
        tmp_dir = Path(tempfile.mkdtemp(prefix="jarvis-voz-"))
        audio_path = tmp_dir / ("audio" + suffix)
        try:
            try:
                tg_file = await media.get_file()
                await tg_file.download_to_drive(custom_path=str(audio_path))
            except TelegramError:
                logger.exception("Falha ao baixar áudio do Telegram")
                await self._reply(update, "Não consegui baixar o áudio do Telegram.")
                return
            try:
                transcript = await run_blocking(self.voice.transcribe, audio_path)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Falha ao transcrever áudio")
                await self._reply(
                    update, "Não consegui transcrever o áudio: {}".format(exc)
                )
                return
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        transcript = (transcript or "").strip()
        if not transcript:
            await self._reply(
                update, "Não entendi o áudio. Tente falar mais perto do microfone."
            )
            return

        await self._reply(update, "Ouvi: {}".format(transcript))
        await self._run_agent(update, chat.id, transcript, context)

    async def _cmd_briefing(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._guard(update):
            return
        if self.briefing is None:
            await self._reply(update, "Briefing indisponível neste momento.")
            return
        chat = update.effective_chat
        if chat is None:
            return
        await self._reply(update, "Gerando o briefing de hoje...")
        try:
            text = await self.briefing.generate(chat.id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Falha ao gerar briefing manual")
            await self._reply(update, "Erro ao gerar briefing: {}".format(exc))
            return
        await self._reply(update, text or "Briefing vazio.")
        if text:
            await self._maybe_send_audio(chat.id, text)

    async def _cmd_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._guard(update):
            return
        chat = update.effective_chat
        if chat is None:
            return
        if not self.settings.tts_enabled:
            await self._reply(
                update, "TTS está desativado (JARVIS_TTS_ENABLED=false)."
            )
            return
        if not self.tts.available():
            await self._reply(
                update,
                "edge-tts não instalado. Rode `pip install -r requirements.txt` no .venv.",
            )
            return

        pref = self.memory.get_pref(chat.id, "tts_auto")
        enabled = self.settings.tts_auto if pref is None else pref == "1"
        args = [arg.lower() for arg in (context.args or [])]
        if not args or args[0] in {"status", "estado"}:
            await self._reply(
                update,
                "Áudio automático: {}.\nVoz: {}\nUse /voice on ou /voice off.".format(
                    "ligado" if enabled else "desligado",
                    self.settings.tts_voice,
                ),
            )
            return
        if args[0] in {"on", "ligar", "ativar", "1"}:
            self.memory.set_pref(chat.id, "tts_auto", "1")
            await self._reply(update, "Áudio automático ligado.")
            return
        if args[0] in {"off", "desligar", "desativar", "0"}:
            self.memory.set_pref(chat.id, "tts_auto", "0")
            await self._reply(update, "Áudio automático desligado.")
            return
        await self._reply(update, "Uso: /voice on, /voice off ou /voice status.")

    async def _cmd_reindex(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._guard(update):
            return
        if self.rag is None:
            await self._reply(
                update,
                "Memória/RAG desativada. Defina JARVIS_RAG_ENABLED=true no .env.",
            )
            return
        await self._reply(
            update,
            "Indexando arquivos na memória local... isso pode demorar.",
        )
        try:
            result = await run_blocking(self.rag.index)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Falha ao reindexar")
            await self._reply(update, "Erro ao indexar: {}".format(exc))
            return
        await self._reply(update, result)

    async def _cmd_remind(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._guard(update):
            return
        chat = update.effective_chat
        if chat is None:
            return
        if self.reminders is None or not self.reminders.is_started:
            await self._reply(
                update, "Lembretes indisponíveis: o agendador não iniciou."
            )
            return

        args = list(context.args or [])
        if len(args) < 2:
            await self._reply(
                update,
                "Uso: /remind <quando> <texto>\n"
                "Ex.: /remind 10m tomar água\n"
                "/remind amanhã 09:00 reunião\n"
                "/remind 2026-11-20 18:30 prova",
            )
            return

        when = None
        text = ""
        for size in (2, 1, 3):
            if len(args) - size < 1:
                continue
            try:
                candidate = parse_when(
                    " ".join(args[:size]), self.settings.timezone
                )
            except ValueError:
                continue
            when = candidate
            text = " ".join(args[size:]).strip()
            break

        if when is None or not text:
            await self._reply(
                update,
                "Não entendi a data/hora. Ex.: /remind 10m tomar água "
                "ou /remind amanhã 09:00 reunião",
            )
            return

        try:
            reminder_id = self.reminders.add(chat.id, when, text)
        except (ValueError, RuntimeError) as exc:
            await self._reply(update, "Erro: {}".format(exc))
            return

        await self._reply(
            update,
            "Lembrete #{} agendado para {}: {}".format(
                reminder_id, when.strftime("%d/%m/%Y %H:%M"), text
            ),
        )

    async def _cmd_reminders(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._guard(update):
            return
        chat = update.effective_chat
        if chat is None:
            return
        if self.reminders is None:
            await self._reply(update, "Lembretes indisponíveis neste momento.")
            return

        rows = self.reminders.list_pending(chat.id)
        if not rows:
            await self._reply(
                update,
                "Nenhum lembrete pendente. Crie com /remind 10m <texto>.",
            )
            return

        lines = ["Lembretes pendentes:"]
        for row in rows:
            try:
                due = datetime.fromisoformat(row["due_at"])
                when_text = due.strftime("%d/%m/%Y %H:%M")
            except (TypeError, ValueError):
                when_text = str(row.get("due_at") or "-")
            lines.append("#{} - {} - {}".format(row["id"], when_text, row["text"]))
        await self._reply(update, "\n".join(lines))

    async def _on_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._guard(update):
            return
        message = update.message
        chat = update.effective_chat
        if message is None or chat is None:
            return

        document = message.document
        photo = message.photo[-1] if message.photo else None
        if document is None and photo is None:
            return

        if document is not None:
            original_name = document.file_name or "documento"
            if (document.file_size or 0) > 20 * 1024 * 1024:
                await self._reply(update, "Arquivo grande demais (limite 20 MB).")
                return
            media = document
        else:
            original_name = "foto.jpg"
            media = photo

        upload_dir = Path(self.settings.db_path).parent / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = upload_dir / "{}-{}".format(stamp, _safe_filename(original_name))

        try:
            tg_file = await media.get_file()
            await tg_file.download_to_drive(custom_path=str(target))
        except TelegramError as exc:
            logger.exception("Falha ao baixar documento do Telegram")
            await self._reply(update, "Não consegui baixar o arquivo: {}".format(exc))
            return

        suffix = target.suffix.lower()
        if suffix == ".pdf":
            instruction = "Use read_pdf para ler o PDF em {}".format(target)
        elif suffix == ".docx":
            instruction = "Use read_docx para ler o DOCX em {}".format(target)
        elif suffix in {".xlsx", ".xlsm", ".csv", ".tsv"}:
            instruction = (
                "Use spreadsheet com action=read para ler a planilha em {}".format(target)
            )
        elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            instruction = "Use read_image (OCR) para ler a imagem em {}".format(target)
        else:
            instruction = "Use read_file para ler o arquivo em {}".format(target)

        caption = (message.caption or "").strip()
        if caption:
            instruction += "\n\nPedido da usuária: {}".format(caption)
        else:
            instruction += (
                "\n\nResuma o conteúdo em português e destaque os pontos principais."
            )

        await self._reply(
            update,
            "Recebi {} e salvei em {}. Analisando...".format(
                _safe_filename(original_name), target
            ),
        )
        if self.settings.rag_enabled and self.rag.supports(target):
            self._spawn(self._index_upload(target))
        await self._run_agent(update, chat.id, instruction, context)

    async def _on_unsupported(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._guard(update):
            return
        await self._reply(
            update,
            "Por enquanto não analiso vídeos. Mas aceito áudio, PDF, DOCX, "
            "planilha e imagem (OCR).",
        )

    async def _on_error(
        self, update: Optional[Update], context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        logger.error("Erro no Telegram: %s", context.error, exc_info=context.error)

    def run(self) -> None:
        logger.info(
            "Jarvis no ar | modelo=%s | workspace=%s | usuários autorizados=%s",
            self.settings.ollama_model,
            self.settings.workspace,
            sorted(self.settings.allowed_user_ids) or "NENHUM (o bot informará o ID)",
        )
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)
