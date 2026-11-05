from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import Settings
from ..rag import RagIndex
from ..reminders import ReminderService
from .base import ToolRegistry
from .business import BusinessTools
from .dev import DevTools
from .documents import DocumentTools
from .google import GoogleTools
from .knowledge import KnowledgeTools
from .microsoft import MicrosoftTools
from .productivity import ProductivityTools
from .reminders import ReminderTools
from .system import SystemTools
from .telegram_tools import TelegramTools
from .voice_tools import VoiceTools
from .web import WebTools


def build_registry(
    settings: Settings,
    reminders: Optional[ReminderService] = None,
    rag: Optional[RagIndex] = None,
) -> ToolRegistry:
    audit_path = Path(settings.db_path).parent / "audit.log"
    registry = ToolRegistry(audit_path=audit_path)
    SystemTools(settings, registry)
    DevTools(settings, registry)
    ProductivityTools(settings, registry)
    GoogleTools(settings, registry)
    MicrosoftTools(settings, registry)
    BusinessTools(settings, registry)
    WebTools(settings, registry)
    TelegramTools(settings, registry)
    DocumentTools(settings, registry)
    VoiceTools(settings, registry)
    if rag is not None:
        KnowledgeTools(settings, rag, registry)
    if reminders is not None:
        ReminderTools(reminders, registry)
    return registry
