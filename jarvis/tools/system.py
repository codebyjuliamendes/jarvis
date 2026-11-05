from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from .. import security
from ..config import Settings
from ..security import SecurityError
from ..utils import human_size
from .base import ToolRegistry


class SystemTools:
    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self._register()

    def _resolve(self, raw: str, must_exist: bool = False) -> Path:
        return security.resolve_path(
            raw,
            roots=self.settings.allowed_roots,
            base=self.settings.workspace,
            must_exist=must_exist,
        )

    def _register(self) -> None:
        self.registry.register(
            "run_shell",
            "Executa um comando no bash dentro das pastas permitidas. Use para rodar testes, "
            "builds, scripts, instalar dependências e comandos não interativos.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Comando bash."},
                    "cwd": {
                        "type": "string",
                        "description": "Diretório de trabalho (opcional).",
                    },
                },
                "required": ["command"],
            },
            self._run_shell,
        )
        self.registry.register(
            "read_file",
            "Lê um arquivo de texto. Retorna linhas numeradas. Use para inspecionar código e docs.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Caminho do arquivo."},
                    "start_line": {"type": "integer", "description": "Linha inicial (1)."},
                    "end_line": {"type": "integer", "description": "Linha final."},
                },
                "required": ["path"],
            },
            self._read_file,
        )
        self.registry.register(
            "write_file",
            "Cria ou sobrescreve um arquivo de texto. Use com cuidado: sobrescreve o conteúdo.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Caminho do arquivo."},
                    "content": {"type": "string", "description": "Conteúdo completo."},
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Criar pastas intermediárias (padrão true).",
                    },
                },
                "required": ["path", "content"],
            },
            self._write_file,
        )
        self.registry.register(
            "replace_in_file",
            "Substitui um trecho exato de texto dentro de um arquivo. Ideal para edições pequenas.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Caminho do arquivo."},
                    "old": {"type": "string", "description": "Texto exato a substituir."},
                    "new": {"type": "string", "description": "Novo texto."},
                    "all": {
                        "type": "boolean",
                        "description": "Substituir todas as ocorrências (padrão false).",
                    },
                },
                "required": ["path", "old", "new"],
            },
            self._replace_in_file,
        )
        self.registry.register(
            "list_dir",
            "Lista arquivos e pastas de um diretório.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Diretório (padrão: workspace)."}
                },
                "required": [],
            },
            self._list_dir,
        )
        self.registry.register(
            "search_files",
            "Busca um padrão regex dentro dos arquivos de uma pasta (similar a grep -rn).",
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Expressão regular."},
                    "path": {"type": "string", "description": "Pasta inicial."},
                    "glob": {
                        "type": "string",
                        "description": "Filtro de nomes, ex.: '*.py'. Padrão '*'.",
                    },
                    "max_results": {"type": "integer", "description": "Máximo de resultados."},
                },
                "required": ["pattern"],
            },
            self._search_files,
        )
        self.registry.register(
            "find_files",
            "Encontra arquivos/pastas pelo nome (glob), ex.: '*.pdf' ou 'test_*'.",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Glob do nome, ex.: '*.py'."},
                    "path": {"type": "string", "description": "Pasta inicial."},
                    "max_results": {"type": "integer", "description": "Máximo de resultados."},
                },
                "required": ["name"],
            },
            self._find_files,
        )

    def _run_shell(self, args: Dict[str, Any]) -> str:
        if not self.settings.shell_enabled:
            return "Terminal desabilitado. Defina JARVIS_SHELL_ENABLED=true no .env."
        command = str(args.get("command", "")).strip()
        if not command:
            return "Informe o campo 'command'."
        try:
            security.check_shell_command(command)
        except SecurityError as exc:
            return "Bloqueado: {}".format(exc)

        try:
            cwd = self._resolve(args.get("cwd") or str(self.settings.workspace), must_exist=True)
        except SecurityError as exc:
            return "Erro: {}".format(exc)
        if not cwd.is_dir():
            return "O cwd não é um diretório: {}".format(cwd)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=self.settings.shell_timeout,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return "O comando excedeu {}s e foi interrompido.".format(
                self.settings.shell_timeout
            )
        except Exception as exc:  # noqa: BLE001
            return "Erro ao executar: {}".format(exc)

        parts: List[str] = []
        if proc.stdout and proc.stdout.strip():
            parts.append(proc.stdout.rstrip())
        if proc.stderr and proc.stderr.strip():
            parts.append("[stderr]\n" + proc.stderr.rstrip())
        parts.append("[exit code: {}]".format(proc.returncode))
        return "\n".join(parts) if parts else "(sem saída)"

    def _read_file(self, args: Dict[str, Any]) -> str:
        raw = str(args.get("path", "")).strip()
        if not raw:
            return "Informe o campo 'path'."
        try:
            path = self._resolve(raw, must_exist=True)
        except SecurityError as exc:
            return "Erro: {}".format(exc)
        if not path.is_file():
            return "Não é um arquivo: {}".format(path)

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            return "Erro ao ler: {}".format(exc)

        lines = text.splitlines()
        total = len(lines)
        try:
            start = max(1, int(args.get("start_line") or 1))
        except (TypeError, ValueError):
            start = 1
        end_raw = args.get("end_line")
        try:
            end = int(end_raw) if end_raw else min(total, start + 399)
        except (TypeError, ValueError):
            end = min(total, start + 399)
        end = max(start, min(end, total))

        selected = lines[start - 1 : end]
        numbered = "\n".join(
            "{}: {}".format(number, line)
            for number, line in enumerate(selected, start=start)
        )
        header = "Arquivo: {} (linhas {}-{} de {})\n".format(path, start, end, total)
        return header + (numbered or "(arquivo vazio)")

    def _write_file(self, args: Dict[str, Any]) -> str:
        raw = str(args.get("path", "")).strip()
        if not raw:
            return "Informe o campo 'path'."
        content = str(args.get("content", ""))
        try:
            path = self._resolve(raw)
        except SecurityError as exc:
            return "Erro: {}".format(exc)
        try:
            if args.get("create_dirs", True):
                path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return "Erro ao escrever: {}".format(exc)
        return "Arquivo escrito: {} ({} bytes)".format(path, len(content.encode("utf-8")))

    def _replace_in_file(self, args: Dict[str, Any]) -> str:
        raw = str(args.get("path", "")).strip()
        old = str(args.get("old", ""))
        new = str(args.get("new", ""))
        if not raw:
            return "Informe o campo 'path'."
        if not old:
            return "O campo 'old' não pode ser vazio."
        try:
            path = self._resolve(raw, must_exist=True)
        except SecurityError as exc:
            return "Erro: {}".format(exc)
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return "Erro ao ler: {}".format(exc)

        occurrences = text.count(old)
        if occurrences == 0:
            return "Trecho não encontrado em {}.".format(path)
        replace_all = bool(args.get("all", False))
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        try:
            path.write_text(updated, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return "Erro ao escrever: {}".format(exc)
        changed = occurrences if replace_all else 1
        return "Substituído(s) {} trecho(s) em {}.".format(changed, path)

    def _list_dir(self, args: Dict[str, Any]) -> str:
        raw = str(args.get("path") or str(self.settings.workspace)).strip()
        try:
            path = self._resolve(raw, must_exist=True)
        except SecurityError as exc:
            return "Erro: {}".format(exc)
        if not path.is_dir():
            return "Não é um diretório: {}".format(path)

        entries: List[str] = []
        skipped = 0
        try:
            children = sorted(path.iterdir(), key=lambda item: item.name.lower())
        except Exception as exc:  # noqa: BLE001
            return "Erro ao listar: {}".format(exc)

        for child in children:
            if security.is_sensitive(child):
                skipped += 1
                continue
            try:
                if child.is_dir():
                    entries.append("[dir]  {}/".format(child.name))
                else:
                    entries.append(
                        "[file] {} ({})".format(child.name, human_size(child.stat().st_size))
                    )
            except OSError:
                entries.append("[?]    {}".format(child.name))
            if len(entries) >= 250:
                entries.append("... (lista truncada em 250 itens)")
                break

        if skipped:
            entries.append("({} item(ns) sensível(is) omitido(s))".format(skipped))
        return "Diretório: {}\n{}".format(path, "\n".join(entries) or "(vazio)")

    def _search_files(self, args: Dict[str, Any]) -> str:
        pattern = str(args.get("pattern", ""))
        if not pattern:
            return "Informe o campo 'pattern'."
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return "Regex inválido: {}".format(exc)
        try:
            root = self._resolve(args.get("path") or str(self.settings.workspace), must_exist=True)
        except SecurityError as exc:
            return "Erro: {}".format(exc)
        if not root.is_dir():
            return "Não é um diretório: {}".format(root)

        name_glob = str(args.get("glob") or "*")
        try:
            max_results = max(1, min(int(args.get("max_results") or 50), 300))
        except (TypeError, ValueError):
            max_results = 50

        results: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name
                for name in dirnames
                if not security.is_sensitive(Path(dirpath) / name)
            ]
            for filename in filenames:
                if not fnmatch.fnmatch(filename, name_glob):
                    continue
                file_path = Path(dirpath) / filename
                if security.is_sensitive(file_path):
                    continue
                try:
                    if file_path.stat().st_size > 2_000_000:
                        continue
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except (OSError, UnicodeError):
                    continue
                for line_number, line in enumerate(content.splitlines(), start=1):
                    if regex.search(line):
                        results.append(
                            "{}:{}: {}".format(file_path, line_number, line.strip()[:300])
                        )
                        if len(results) >= max_results:
                            break
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

        if not results:
            return "Nenhum resultado para {!r} em {}.".format(pattern, root)
        suffix = "\n... (limite de {} resultados)".format(max_results) if len(results) >= max_results else ""
        return "\n".join(results) + suffix

    def _find_files(self, args: Dict[str, Any]) -> str:
        name_glob = str(args.get("name", "")).strip()
        if not name_glob:
            return "Informe o campo 'name'."
        try:
            root = self._resolve(args.get("path") or str(self.settings.workspace), must_exist=True)
        except SecurityError as exc:
            return "Erro: {}".format(exc)
        try:
            max_results = max(1, min(int(args.get("max_results") or 100), 500))
        except (TypeError, ValueError):
            max_results = 100

        results: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name
                for name in dirnames
                if not security.is_sensitive(Path(dirpath) / name)
            ]
            for filename in filenames + dirnames:
                if fnmatch.fnmatch(filename, name_glob):
                    candidate = Path(dirpath) / filename
                    if security.is_sensitive(candidate):
                        continue
                    results.append(str(candidate))
                    if len(results) >= max_results:
                        break
            if len(results) >= max_results:
                break

        if not results:
            return "Nenhum arquivo encontrado com {!r} em {}.".format(name_glob, root)
        suffix = "\n... (limite de {} resultados)".format(max_results) if len(results) >= max_results else ""
        return "\n".join(results) + suffix
