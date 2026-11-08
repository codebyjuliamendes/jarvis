"""Servidor MCP (Model Context Protocol) do Jarvis — módulo opcional.

Expõe as ferramentas do ``ToolRegistry`` para clientes MCP (Claude Desktop, Claude Code,
MCP Inspector) por stdio::

    .venv-mcp/bin/python -m jarvis.mcp_server            # serve por stdio
    .venv-mcp/bin/python -m jarvis.mcp_server --check    # mostra o que fica exposto

O SDK oficial (``mcp``) exige Python >= 3.10 e fica em ``requirements-mcp.txt``; o resto do
Jarvis continua em 3.8. Por isso o SDK só é importado dentro de ``build_server()`` e de
``main()``: a política de exposição e a ponte com o registro (``resolve_exposed_tools``,
``McpBridge``) funcionam e são testadas em 3.8 sem o pacote instalado. O arquivo mantém
sintaxe 3.8 de propósito.

Segurança: a execução vai pelo mesmo ``ToolRegistry.execute`` do agente, que chama os
mesmos handlers, que chamam o ``security.py`` (pastas permitidas, arquivos de credenciais,
comandos destrutivos, SSRF). Este módulo só decide **quais** ferramentas aparecem; não há
checagem paralela. Ver ``docs/mcp.md``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from . import __version__, observability
from .tools.base import ToolRegistry

# Nome fixo: com `python -m`, __name__ seria "__main__" nos logs.
logger = logging.getLogger("jarvis.mcp_server")

SERVER_NAME = "jarvis"

# Só leem (arquivos dentro de JARVIS_ALLOWED_ROOTS, a memória local, a web pública).
# É o conjunto exposto quando JARVIS_MCP_TOOLS está vazio.
READ_ONLY_TOOLS: FrozenSet[str] = frozenset(
    {
        "read_file",
        "list_dir",
        "search_files",
        "find_files",
        "read_pdf",
        "read_docx",
        "read_image",
        "knowledge_search",
        "web_search",
        "web_fetch",  # passa pelo security.check_url (bloqueia SSRF)
    }
)
# Saem da máquina para a internet: o cliente MCP pode mostrar isso ao usuário.
OPEN_WORLD_TOOLS: FrozenSet[str] = frozenset({"web_search", "web_fetch"})
# Precisam de um chat do Telegram (enviam arquivo/áudio para ele): nunca expostas.
TELEGRAM_ONLY_TOOLS: FrozenSet[str] = frozenset({"send_file", "speak"})
# Atalho aceito em JARVIS_MCP_TOOLS para o conjunto de leitura inteiro.
READ_ONLY_ALIAS = "leitura"

UNSAFE_HINT = "JARVIS_MCP_ALLOW_UNSAFE=true"


@dataclass(frozen=True)
class ExposurePolicy:
    """Resultado da allowlist: o que é exposto e o que foi recusado (com o motivo)."""

    tools: Tuple[str, ...]
    refused: Dict[str, str] = field(default_factory=dict)
    allow_unsafe: bool = False

    def is_read_only(self, name: str) -> bool:
        return name in READ_ONLY_TOOLS

    def describe(self) -> str:
        lines = ["Ferramentas expostas pelo servidor MCP ({}):".format(len(self.tools))]
        for name in self.tools:
            kind = "leitura" if self.is_read_only(name) else "ESCREVE/EXECUTA/ENVIA"
            lines.append("  - {} ({})".format(name, kind))
        if not self.tools:
            lines.append("  (nenhuma)")
        if self.refused:
            lines.append("Pedidas e recusadas:")
            for name in sorted(self.refused):
                lines.append("  - {}: {}".format(name, self.refused[name]))
        return "\n".join(lines)


def resolve_exposed_tools(
    available: Iterable[str],
    requested: Optional[Sequence[str]] = None,
    allow_unsafe: bool = False,
) -> ExposurePolicy:
    """Aplica a allowlist.

    - ``requested`` vazio/None: o conjunto de leitura (``READ_ONLY_TOOLS``) que estiver
      carregado no registro.
    - ``requested`` com nomes: exatamente esses (serve para restringir), mais o atalho
      ``leitura``. Uma ferramenta fora do conjunto de leitura só entra com
      ``allow_unsafe=True`` — duas chaves: o nome na lista **e** a variável ligada.
    - ``send_file``/``speak`` nunca entram; nomes desconhecidos são ignorados.
    """
    available_set = set(available)
    names: List[str] = []
    for raw in requested or []:
        item = str(raw).strip()
        if not item:
            continue
        if item.lower() == READ_ONLY_ALIAS:
            names.extend(sorted(READ_ONLY_TOOLS))
        else:
            names.append(item)
    if not names:
        names = sorted(READ_ONLY_TOOLS)
        implicit = True
    else:
        implicit = False

    exposed: List[str] = []
    refused: Dict[str, str] = {}
    for name in names:
        if name in exposed or name in refused:
            continue
        if name not in available_set:
            # No padrão, uma ferramenta de leitura ausente (ex.: knowledge_search com o RAG
            # desligado) não é um erro de configuração: só não aparece.
            if not implicit:
                refused[name] = "não existe ou não foi carregada (ex.: RAG desligado)"
            continue
        if name in TELEGRAM_ONLY_TOOLS:
            refused[name] = "depende de um chat do Telegram; não faz sentido via MCP"
            continue
        if name not in READ_ONLY_TOOLS and not allow_unsafe:
            refused[name] = "escreve, executa ou envia: exige também {}".format(UNSAFE_HINT)
            continue
        exposed.append(name)
    return ExposurePolicy(tools=tuple(sorted(exposed)), refused=refused, allow_unsafe=allow_unsafe)


def _looks_like_error(text: str) -> bool:
    head = text.lstrip()
    return head.startswith("Erro") or head.startswith("Bloqueado")


class McpBridge:
    """Liga o protocolo ao ``ToolRegistry`` sem depender do SDK (testável em 3.8)."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: ExposurePolicy,
        metrics: Optional[Any] = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.metrics = metrics
        self._schemas = {
            item["function"]["name"]: item["function"]
            for item in registry.schemas(only=policy.tools)
        }

    def tool_specs(self) -> List[Dict[str, Any]]:
        """Nome, descrição, JSON Schema e dicas de cada ferramenta exposta."""
        specs: List[Dict[str, Any]] = []
        for name in self.policy.tools:
            function = self._schemas.get(name)
            if function is None:
                continue
            read_only = self.policy.is_read_only(name)
            specs.append(
                {
                    "name": name,
                    "description": function["description"],
                    "input_schema": function["parameters"],
                    "read_only": read_only,
                    "open_world": name in OPEN_WORLD_TOOLS,
                }
            )
        return specs

    async def call(self, name: str, arguments: Optional[Dict[str, Any]]) -> Tuple[str, bool]:
        """Executa uma ferramenta exposta. Devolve (texto, é_erro)."""
        if name not in self.policy.tools:
            # Recusa antes do registro: ferramenta fora da allowlist nunca executa.
            logger.warning("MCP: chamada recusada a ferramenta não exposta: %s", name)
            return (
                "Erro: a ferramenta '{}' não está exposta neste servidor MCP. "
                "Disponíveis: {}".format(name, ", ".join(self.policy.tools) or "(nenhuma)"),
                True,
            )
        args = dict(arguments) if isinstance(arguments, dict) else {}
        with observability.trace_scope(model="", source="mcp") as trace:
            trace.mode = "mcp"
            try:
                text = await self.registry.execute(name, args, context={"source": "mcp"})
            except BaseException as exc:  # execute já captura; isto é só rede de segurança
                trace.set_outcome("error", observability.short_error(exc))
                self._observe(trace)
                raise
            failed = _looks_like_error(text)
            if failed:
                trace.set_outcome("error", text.strip()[: observability.ERROR_MAX_CHARS])
            self._observe(trace)
        return text, failed

    def _observe(self, trace: observability.RequestTrace) -> None:
        try:
            trace.finish()
            logger.info(
                "MCP: %s em %.3fs (%s)",
                ", ".join(call.tool for call in trace.tool_calls) or "-",
                trace.latency_s or 0.0,
                trace.outcome,
            )
        except Exception:  # noqa: BLE001 - medir nunca pode atrapalhar
            logger.debug("Falha ao logar a chamada MCP", exc_info=True)
        if self.metrics is None:
            return
        try:
            self.metrics.record(trace)
        except Exception:  # noqa: BLE001
            logger.warning("Falha ao gravar métricas da chamada MCP", exc_info=True)


def build_server(bridge: McpBridge) -> Any:
    """Monta o ``mcp.server.lowlevel.Server``. Exige o SDK (Python >= 3.10)."""
    import mcp.types as types
    from mcp.server.lowlevel import Server

    def _tool(spec: Dict[str, Any]) -> Any:
        return types.Tool(
            name=spec["name"],
            description=spec["description"],
            input_schema=spec["input_schema"],
            annotations=types.ToolAnnotations(
                read_only_hint=spec["read_only"],
                destructive_hint=not spec["read_only"],
                open_world_hint=spec["open_world"],
            ),
        )

    tools = [_tool(spec) for spec in bridge.tool_specs()]

    async def on_list_tools(ctx: Any, params: Any) -> Any:
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx: Any, params: Any) -> Any:
        text, is_error = await bridge.call(params.name, params.arguments)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)], is_error=is_error
        )

    return Server(
        SERVER_NAME,
        version=__version__,
        instructions=(
            "Ferramentas do Jarvis (assistente pessoal local). Arquivos ficam restritos às "
            "pastas permitidas pela dona; credenciais e endereços internos são bloqueados."
        ),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def serve_stdio(server: Any) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def build_runtime(settings: Any) -> Tuple[McpBridge, List[Any]]:
    """Registro + RAG + métricas a partir das ``Settings``, como o bot faz.

    Devolve a ponte e a lista de recursos a fechar no fim.
    """
    from .metrics_store import MetricsStore
    from .rag import RagIndex
    from .tools import build_registry

    closables: List[Any] = []
    rag = None
    if settings.rag_enabled:
        try:
            rag = RagIndex(settings)
            closables.append(rag)
        except Exception as exc:  # noqa: BLE001 - RAG é opcional
            logger.warning("Memória/RAG desativada no MCP: %s", exc)
    metrics = None
    if settings.metrics_enabled:
        try:
            metrics = MetricsStore(settings.resolved_metrics_db_path())
            closables.append(metrics)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Métricas desativadas no MCP: %s", exc)
    # Sem ReminderService: lembretes precisam do agendador e do Telegram.
    registry = build_registry(settings, reminders=None, rag=rag)
    policy = resolve_exposed_tools(
        registry.names(), settings.mcp_tools, allow_unsafe=settings.mcp_allow_unsafe
    )
    return McpBridge(registry, policy, metrics=metrics), closables


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m jarvis.mcp_server",
        description="Servidor MCP (stdio) com as ferramentas do Jarvis.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="mostra as ferramentas expostas (e as recusadas) e sai, sem subir o servidor",
    )
    options = parser.parse_args(argv)

    from .config import BASE_DIR, Settings
    from .logging_setup import setup_logging

    # Como o run.sh: caminhos relativos do .env (JARVIS_DB_PATH=data/...) valem a partir da
    # raiz do repositório, qualquer que seja a pasta de onde o cliente MCP lança o processo.
    os.chdir(str(BASE_DIR))
    settings = Settings.load()
    setup_logging(settings.log_format, settings.log_level)  # stderr: stdout é do protocolo
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    bridge, closables = build_runtime(settings)
    try:
        if options.check:
            try:  # console do Windows em cp1252: acentos saem tortos sem isto
                sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                pass
            print(bridge.policy.describe())
            return 0
        for name, reason in sorted(bridge.policy.refused.items()):
            logger.warning("MCP: ferramenta '%s' não exposta: %s", name, reason)
        logger.info("MCP: expondo %s", ", ".join(bridge.policy.tools) or "(nada)")

        if sys.version_info < (3, 10):
            print(
                "O servidor MCP exige Python >= 3.10 (o SDK `mcp` não roda em {}.{}). "
                "Crie o venv: uv venv --python 3.12 .venv-mcp".format(*sys.version_info[:2]),
                file=sys.stderr,
            )
            return 2
        try:
            import anyio
        except ImportError:
            anyio = None  # type: ignore[assignment]
        try:
            server = build_server(bridge)
        except ImportError:
            print(
                "SDK `mcp` não instalado. Rode: pip install -r requirements.txt "
                "-r requirements-mcp.txt (num Python >= 3.10).",
                file=sys.stderr,
            )
            return 2
        assert anyio is not None  # dependência do próprio SDK
        anyio.run(serve_stdio, server)
        return 0
    finally:
        for item in closables:
            try:
                item.close()
            except Exception:  # noqa: BLE001
                logger.debug("Falha ao fechar recurso", exc_info=True)


if __name__ == "__main__":
    raise SystemExit(main())
