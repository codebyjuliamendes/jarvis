from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .. import security
from ..config import Settings
from ..security import SecurityError
from .base import ToolRegistry


class DevTools:
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
            "git",
            "Executa comandos git num repositório (status, diff, log, add, commit, branch, pull...). "
            "Não abre editor. Push com --force é bloqueado por segurança.",
            {
                "type": "object",
                "properties": {
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Argumentos do git, ex.: ['status', '--short'].",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Pasta do repositório (padrão: workspace).",
                    },
                },
                "required": ["args"],
            },
            self._git,
        )
        self.registry.register(
            "github",
            "Consulta e age no GitHub via API: repositórios, issues, pull requests, commits, "
            "criar issue e comentar. Requer GITHUB_TOKEN.",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "list_repos",
                            "repo_info",
                            "list_issues",
                            "get_issue",
                            "list_prs",
                            "get_pr",
                            "recent_commits",
                            "create_issue",
                            "comment_issue",
                        ],
                        "description": "Ação a executar.",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repositório no formato owner/nome (opcional se GITHUB_DEFAULT_REPO).",
                    },
                    "number": {
                        "type": "integer",
                        "description": "Número da issue ou PR.",
                    },
                    "title": {"type": "string", "description": "Título (create_issue)."},
                    "body": {"type": "string", "description": "Corpo/comentário."},
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "Filtro de estado (padrão open).",
                    },
                    "limit": {"type": "integer", "description": "Máximo de itens (padrão 10)."},
                },
                "required": ["action"],
            },
            self._github,
        )

    def _git(self, args: Dict[str, Any]) -> str:
        raw = args.get("args")
        if isinstance(raw, str):
            try:
                command_args = shlex.split(raw)
            except ValueError as exc:
                return "Argumentos inválidos: {}".format(exc)
        elif isinstance(raw, list):
            command_args = [str(item) for item in raw]
        else:
            return "O campo 'args' deve ser uma lista de strings ou uma string."

        if not command_args:
            return "Informe os argumentos do git (ex.: ['status'])."

        lowered = [item.lower() for item in command_args]
        if lowered[0] == "push" and any(
            item in {"--force", "-f", "--force-with-lease"} for item in lowered
        ):
            return (
                "Bloqueado: `git push --force`. Se for realmente necessário, "
                "rode manualmente no terminal."
            )
        if lowered[:2] == ["config", "--global"]:
            return "Bloqueado: `git config --global`. Edite a configuração manualmente."
        if lowered[0] == "clean" and any(
            item.startswith("-") and "f" in item for item in lowered[1:]
        ):
            return "Bloqueado: `git clean -f` pode apagar arquivos não versionados."

        try:
            cwd = self._resolve(args.get("cwd") or str(self.settings.workspace), must_exist=True)
        except SecurityError as exc:
            return "Erro: {}".format(exc)
        if not cwd.is_dir():
            return "O cwd não é um diretório: {}".format(cwd)

        env = {
            **dict(__import__("os").environ),
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_EDITOR": "true",
            "GIT_PAGER": "cat",
        }
        try:
            proc = subprocess.run(
                ["git"] + command_args,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=self.settings.shell_timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return "O git excedeu {}s e foi interrompido.".format(self.settings.shell_timeout)
        except Exception as exc:  # noqa: BLE001
            return "Erro ao executar git: {}".format(exc)

        parts: List[str] = []
        if proc.stdout.strip():
            parts.append(proc.stdout.rstrip())
        if proc.stderr.strip():
            parts.append("[stderr]\n" + proc.stderr.rstrip())
        parts.append("[exit code: {}]".format(proc.returncode))
        return "\n".join(parts)

    def _gh_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Any], Optional[str]]:
        if not self.settings.github_token:
            return None, (
                "GitHub não configurado. Crie um token em "
                "https://github.com/settings/tokens (escopo repo) e defina GITHUB_TOKEN no .env."
            )
        headers = {
            "Authorization": "Bearer " + self.settings.github_token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "jarvis-local",
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.request(
                    method,
                    "https://api.github.com" + path,
                    headers=headers,
                    params=params,
                    json=body,
                )
        except httpx.HTTPError as exc:
            return None, "Erro de rede no GitHub: {}".format(exc)

        if response.status_code == 401:
            return None, "GitHub recusou o token (401). Verifique GITHUB_TOKEN."
        if response.status_code == 403 and "rate limit" in response.text.lower():
            return None, "Limite de requisições do GitHub atingido. Tente mais tarde."
        if response.status_code >= 400:
            return None, "GitHub HTTP {}: {}".format(
                response.status_code, response.text[:600]
            )
        try:
            return response.json(), None
        except ValueError:
            return None, "Resposta inválida do GitHub (não era JSON)."

    def _repo(self, args: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        repo = str(args.get("repo") or self.settings.github_default_repo or "").strip()
        if not repo or "/" not in repo:
            return None, (
                "Informe 'repo' no formato owner/nome "
                "(ou defina GITHUB_DEFAULT_REPO no .env)."
            )
        return repo, None

    def _github(self, args: Dict[str, Any]) -> str:
        action = str(args.get("action", "")).strip().lower()
        try:
            limit = max(1, min(int(args.get("limit") or 10), 50))
        except (TypeError, ValueError):
            limit = 10
        state = str(args.get("state") or "open").lower()
        if state not in {"open", "closed", "all"}:
            state = "open"

        if action == "list_repos":
            data, err = self._gh_request(
                "GET", "/user/repos", params={"sort": "updated", "per_page": limit}
            )
            if err:
                return err
            lines = []
            for repo in data or []:
                visibility = "privado" if repo.get("private") else "público"
                lines.append(
                    "- {} ({}, {}) {}".format(
                        repo.get("full_name"),
                        visibility,
                        repo.get("language") or "sem linguagem",
                        repo.get("description") or "",
                    ).rstrip()
                )
            return "\n".join(lines) or "Nenhum repositório encontrado."

        if action == "repo_info":
            repo, err = self._repo(args)
            if err:
                return err
            data, err = self._gh_request("GET", "/repos/{}".format(repo))
            if err:
                return err
            return (
                "Repo: {}\nDescrição: {}\nLinguagem: {}\nBranch padrão: {}\n"
                "Estrelas: {} | Forks: {} | Issues abertas: {}\nURL: {}".format(
                    data.get("full_name"),
                    data.get("description") or "-",
                    data.get("language") or "-",
                    data.get("default_branch") or "-",
                    data.get("stargazers_count", 0),
                    data.get("forks_count", 0),
                    data.get("open_issues_count", 0),
                    data.get("html_url"),
                )
            )

        if action in {"list_issues", "list_prs"}:
            repo, err = self._repo(args)
            if err:
                return err
            if action == "list_issues":
                data, err = self._gh_request(
                    "GET",
                    "/repos/{}/issues".format(repo),
                    params={"state": state, "per_page": limit},
                )
            else:
                data, err = self._gh_request(
                    "GET",
                    "/repos/{}/pulls".format(repo),
                    params={"state": state, "per_page": limit},
                )
            if err:
                return err
            lines = []
            for item in data or []:
                if action == "list_issues" and "pull_request" in item:
                    continue
                labels = ", ".join(label.get("name", "") for label in item.get("labels", []))
                suffix = " [{}]".format(labels) if labels else ""
                lines.append(
                    "#{} [{}] {}{}".format(
                        item.get("number"), item.get("state"), item.get("title"), suffix
                    )
                )
            kind = "issues" if action == "list_issues" else "pull requests"
            return "\n".join(lines) or "Nenhum(a) {} com estado '{}'.".format(kind, state)

        if action in {"get_issue", "get_pr"}:
            repo, err = self._repo(args)
            if err:
                return err
            number = args.get("number")
            if not number:
                return "Informe 'number'."
            if action == "get_issue":
                data, err = self._gh_request(
                    "GET", "/repos/{}/issues/{}".format(repo, number)
                )
            else:
                data, err = self._gh_request(
                    "GET", "/repos/{}/pulls/{}".format(repo, number)
                )
            if err:
                return err
            labels = ", ".join(label.get("name", "") for label in data.get("labels", []))
            body = (data.get("body") or "")[:5000]
            lines = [
                "#{} {}".format(data.get("number"), data.get("title")),
                "Estado: {} | Autor: {} | Labels: {}".format(
                    data.get("state"),
                    (data.get("user") or {}).get("login", "-"),
                    labels or "-",
                ),
                "URL: {}".format(data.get("html_url")),
            ]
            if action == "get_pr":
                head = data.get("head") or {}
                base = data.get("base") or {}
                lines.append(
                    "Branch: {} -> {} | +{} / -{} em {} arquivo(s) | mergeable: {}".format(
                        head.get("ref"),
                        base.get("ref"),
                        data.get("additions", 0),
                        data.get("deletions", 0),
                        data.get("changed_files", 0),
                        data.get("mergeable"),
                    )
                )
            lines.append("")
            lines.append(body or "(sem descrição)")
            return "\n".join(lines)

        if action == "recent_commits":
            repo, err = self._repo(args)
            if err:
                return err
            data, err = self._gh_request(
                "GET", "/repos/{}/commits".format(repo), params={"per_page": limit}
            )
            if err:
                return err
            lines = []
            for commit in data or []:
                info = commit.get("commit") or {}
                author = (info.get("author") or {}).get("name", "-")
                date = ((info.get("author") or {}).get("date") or "")[:10]
                message = (info.get("message") or "").splitlines()[0]
                lines.append(
                    "- {} {} {}: {}".format(
                        (commit.get("sha") or "")[:7], date, author, message
                    )
                )
            return "\n".join(lines) or "Nenhum commit encontrado."

        if action == "create_issue":
            repo, err = self._repo(args)
            if err:
                return err
            title = str(args.get("title") or "").strip()
            if not title:
                return "Informe 'title' para criar a issue."
            data, err = self._gh_request(
                "POST",
                "/repos/{}/issues".format(repo),
                body={"title": title, "body": str(args.get("body") or "")},
            )
            if err:
                return err
            return "Issue #{} criada: {}".format(data.get("number"), data.get("html_url"))

        if action == "comment_issue":
            repo, err = self._repo(args)
            if err:
                return err
            number = args.get("number")
            body = str(args.get("body") or "").strip()
            if not number or not body:
                return "Informe 'number' e 'body'."
            data, err = self._gh_request(
                "POST",
                "/repos/{}/issues/{}/comments".format(repo, number),
                body={"body": body},
            )
            if err:
                return err
            return "Comentário criado: {}".format(data.get("html_url"))

        return (
            "Ação desconhecida. Use uma de: list_repos, repo_info, list_issues, "
            "get_issue, list_prs, get_pr, recent_commits, create_issue, comment_issue."
        )
