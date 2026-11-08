"""Rastro por requisição: id da requisição e métricas acumuladas em memória.

Tudo vive em ``contextvars``: cada tarefa asyncio (cada conversa) enxerga a sua
própria requisição, então conversas simultâneas não misturam números mesmo
compartilhando o mesmo ``OllamaClient`` e o mesmo ``ToolRegistry``.

Os ganchos (``record_llm_response``, ``record_llm_error``, ``record_tool_call``)
não fazem nada quando não há requisição ativa — é o caso dos testes, do smoke e
das avaliações — e nunca levantam exceção.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional

from .metrics_utils import NS_PER_SECOND, aggregate_tokens_per_second

logger = logging.getLogger(__name__)

ERROR_MAX_CHARS = 300

_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "jarvis_request_id", default=None
)
_trace: contextvars.ContextVar[Optional["RequestTrace"]] = contextvars.ContextVar(
    "jarvis_request_trace", default=None
)


def new_request_id() -> str:
    """12 caracteres hexadecimais (uuid4): curto para log, folgado para colisão."""
    return uuid.uuid4().hex[:12]


def current_request_id() -> Optional[str]:
    return _request_id.get()


def current_trace() -> Optional["RequestTrace"]:
    return _trace.get()


def short_error(exc: BaseException) -> str:
    text = "{}: {}".format(type(exc).__name__, exc)
    return text[:ERROR_MAX_CHARS]


@contextlib.contextmanager
def request_id_scope(request_id: Optional[str] = None) -> Iterator[str]:
    """Define o id da requisição para tudo que rodar dentro do bloco.

    Reaproveita um id já definido por um escopo de fora (ex.: o handler do
    Telegram abre o escopo e o agente usa o mesmo id).
    """
    existing = _request_id.get()
    if request_id is None and existing is not None:
        yield existing
        return
    rid = request_id or new_request_id()
    token = _request_id.set(rid)
    try:
        yield rid
    finally:
        _request_id.reset(token)


@dataclass
class LlmCall:
    duration_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    eval_ns: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ToolCall:
    tool: str
    duration_s: float
    ok: bool
    error: Optional[str] = None


@dataclass
class RequestTrace:
    """Acumulador de uma requisição. Só listas em memória: custo desprezível."""

    request_id: str
    model: str
    source: str = "chat"
    chat_id: Optional[int] = None
    ts: float = field(default_factory=time.time)
    started: float = field(default_factory=time.perf_counter)
    mode: str = "native"
    iterations: int = 0
    outcome: str = "ok"
    error: Optional[str] = None
    latency_s: Optional[float] = None
    llm_calls: List[LlmCall] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)

    def set_outcome(self, outcome: str, error: Optional[str] = None) -> None:
        self.outcome = outcome
        self.error = str(error)[:ERROR_MAX_CHARS] if error else None

    def finish(self) -> None:
        """Fecha o relógio de parede da requisição."""
        self.latency_s = time.perf_counter() - self.started

    # ------------------------------------------------------------ agregados

    @staticmethod
    def _sum(values: List[Optional[float]]) -> Optional[float]:
        present = [value for value in values if value is not None]
        return sum(present) if present else None

    def prompt_tokens(self) -> Optional[int]:
        total = self._sum([call.prompt_tokens for call in self.llm_calls])
        return int(total) if total is not None else None

    def completion_tokens(self) -> Optional[int]:
        total = self._sum([call.completion_tokens for call in self.llm_calls])
        return int(total) if total is not None else None

    def eval_s(self) -> Optional[float]:
        total = self._sum([call.eval_ns for call in self.llm_calls])
        return total / NS_PER_SECOND if total is not None else None

    def llm_s(self) -> float:
        return sum(call.duration_s for call in self.llm_calls)

    def tokens_per_s(self) -> Optional[float]:
        # Mesma conta das avaliações: soma de tokens ÷ soma do tempo de geração.
        return aggregate_tokens_per_second(
            {"eval_count": call.completion_tokens, "eval_duration": call.eval_ns}
            for call in self.llm_calls
        )

    def as_row(self) -> Dict[str, Any]:
        """Linha da tabela ``metrics`` (mesmos nomes de coluna)."""
        latency = self.latency_s
        if latency is None:
            latency = time.perf_counter() - self.started
        return {
            "request_id": self.request_id,
            "ts": self.ts,
            "source": self.source,
            "chat_id": self.chat_id,
            "model": self.model,
            "mode": self.mode,
            "latency_s": latency,
            "llm_calls": len(self.llm_calls),
            "llm_s": self.llm_s(),
            "prompt_tokens": self.prompt_tokens(),
            "completion_tokens": self.completion_tokens(),
            "eval_s": self.eval_s(),
            "tokens_per_s": self.tokens_per_s(),
            "iterations": self.iterations,
            "tool_calls": len(self.tool_calls),
            "outcome": self.outcome,
            "error": self.error,
        }


@contextlib.contextmanager
def trace_scope(
    model: str,
    source: str = "chat",
    chat_id: Optional[int] = None,
) -> Iterator[RequestTrace]:
    """Abre uma requisição: define o id (ou reaproveita o de fora) e o rastro."""
    with request_id_scope() as rid:
        trace = RequestTrace(request_id=rid, model=model, source=source, chat_id=chat_id)
        token = _trace.set(trace)
        try:
            yield trace
        finally:
            _trace.reset(token)


# ---------------------------------------------------------------- ganchos


def _as_int(value: Any) -> Optional[int]:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def record_llm_response(data: Mapping[str, Any], duration_s: float) -> None:
    """Chamado pelo ``OllamaClient`` depois de cada resposta bem-sucedida."""
    try:
        trace = _trace.get()
        if trace is None:
            return
        eval_ns = data.get("eval_duration")
        trace.llm_calls.append(
            LlmCall(
                duration_s=float(duration_s),
                prompt_tokens=_as_int(data.get("prompt_eval_count")),
                completion_tokens=_as_int(data.get("eval_count")),
                eval_ns=float(eval_ns) if isinstance(eval_ns, (int, float)) else None,
            )
        )
    except Exception:  # noqa: BLE001 - medir nunca pode atrapalhar
        logger.debug("Falha ao registrar resposta do LLM", exc_info=True)


def record_llm_error(exc: BaseException, duration_s: float) -> None:
    """Chamado pelo ``OllamaClient`` quando a chamada falha."""
    try:
        trace = _trace.get()
        if trace is None:
            return
        trace.llm_calls.append(LlmCall(duration_s=float(duration_s), error=short_error(exc)))
    except Exception:  # noqa: BLE001
        logger.debug("Falha ao registrar erro do LLM", exc_info=True)


def record_tool_call(
    tool: str, duration_s: float, ok: bool, error: Optional[str] = None
) -> None:
    """Chamado pelo ``ToolRegistry.execute`` a cada ferramenta."""
    try:
        trace = _trace.get()
        if trace is not None:
            trace.tool_calls.append(
                ToolCall(
                    tool=str(tool),
                    duration_s=float(duration_s),
                    ok=bool(ok),
                    error=(str(error)[:ERROR_MAX_CHARS] if error else None),
                )
            )
        logger.debug(
            "ferramenta executada: %s (%.3fs, %s)",
            tool,
            duration_s,
            "ok" if ok else "falhou",
            extra={"fields": {"tool": tool, "duration_s": round(duration_s, 4), "ok": bool(ok)}},
        )
    except Exception:  # noqa: BLE001
        logger.debug("Falha ao registrar ferramenta", exc_info=True)
