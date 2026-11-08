from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .. import observability
from ..utils import run_blocking, truncate

_SENSITIVE_KEYWORDS = (
    "token",
    "password",
    "senha",
    "secret",
    "api_key",
    "apikey",
    "authorization",
    "credential",
)

_CONTENT_KEYS = {"content", "text", "body", "new", "old", "value", "message"}


def _safe_args(args: Dict[str, Any]) -> Dict[str, str]:
    safe: Dict[str, str] = {}
    for key, value in args.items():
        key_text = str(key)
        if key_text.startswith("_"):
            continue
        lowered = key_text.lower()
        if any(word in lowered for word in _SENSITIVE_KEYWORDS):
            safe[key_text] = "***"
            continue
        if lowered in _CONTENT_KEYS:
            safe[key_text] = "<{} chars>".format(len(str(value)))
            continue
        text = str(value)
        safe[key_text] = text[:200] + ("..." if len(text) > 200 else "")
    return safe


class ToolRegistry:
    def __init__(self, audit_path: Optional[Path] = None) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._audit_path = Path(audit_path) if audit_path else None
        self._audit_lock = threading.Lock()
        if self._audit_path is not None:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], str],
    ) -> None:
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler,
        }

    def names(self) -> List[str]:
        return sorted(self._tools)

    def schemas(
        self, only: Optional[Iterable[str]] = None
    ) -> List[Dict[str, Any]]:
        allowed = set(only) if only is not None else None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in self._tools.values()
            if allowed is None or tool["name"] in allowed
        ]

    def describe_for_prompt(self, only: Optional[Iterable[str]] = None) -> str:
        allowed = set(only) if only is not None else None
        lines: List[str] = []
        for name in self.names():
            if allowed is not None and name not in allowed:
                continue
            tool = self._tools[name]
            schema = json.dumps(tool["parameters"], ensure_ascii=False, indent=2)
            lines.append(
                "- {}: {}\n  parâmetros (JSON Schema): {}".format(
                    name, tool["description"], schema
                )
            )
        return "\n".join(lines)

    def text_protocol_prompt(self, only: Optional[Iterable[str]] = None) -> str:
        return (
            "MODO TEXTO (sem function calling nativo).\n"
            "Para usar uma ferramenta, responda APENAS com um objeto JSON válido, "
            "sem explicações:\n"
            '{"tool": "nome_da_ferramenta", "args": {"param": "valor"}}\n\n'
            "Ferramentas disponíveis:\n"
            + self.describe_for_prompt(only=only)
            + "\n\nDepois de receber o resultado, continue a tarefa. "
            "Quando terminar, responda normalmente em texto (sem JSON)."
        )

    def _audit(
        self,
        name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        result: str,
        error: Optional[str] = None,
    ) -> None:
        if self._audit_path is None:
            return
        record = {
            "ts": datetime.now().astimezone().isoformat(),
            "tool": name,
            "chat_id": (context or {}).get("chat_id"),
            "args": _safe_args(args),
            "result_chars": len(result or ""),
            "error": error,
        }
        source = (context or {}).get("source")
        if source:  # ex.: "mcp"; o bot não passa e o formato dele não muda
            record["source"] = source
        try:
            with self._audit_lock:
                with self._audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    async def execute(
        self,
        name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        tool = self._tools.get(name)
        if tool is None:
            observability.record_tool_call(name, 0.0, False, "ferramenta desconhecida")
            return "Erro: ferramenta desconhecida '{}'. Disponíveis: {}".format(
                name, ", ".join(self.names())
            )
        if not isinstance(args, dict):
            args = {}
        else:
            args = dict(args)
        if context:
            for key, value in context.items():
                args.setdefault("_" + key, value)
        started = time.perf_counter()
        try:
            result = await run_blocking(tool["handler"], args)
        except Exception as exc:  # noqa: BLE001 - erro volta para o modelo
            message = "Erro ao executar '{}': {}".format(name, exc)
            observability.record_tool_call(
                name, time.perf_counter() - started, False, observability.short_error(exc)
            )
            self._audit(name, args, context, message, error=str(exc))
            return message
        final = truncate(result)
        # As ferramentas costumam capturar o erro e devolver "Erro: ..." em texto.
        failed = final.lstrip().startswith("Erro")
        observability.record_tool_call(
            name,
            time.perf_counter() - started,
            not failed,
            final.strip()[:300] if failed else None,
        )
        self._audit(name, args, context, final)
        return final
