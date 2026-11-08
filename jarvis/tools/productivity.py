from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .. import security
from ..config import Settings
from ..security import SecurityError
from .base import ToolRegistry


class ProductivityTools:
    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self._register()

    # ------------------------------------------------------------------ Obsidian
    def _vault(self) -> Tuple[Optional[Path], Optional[str]]:
        vault = self.settings.obsidian_vault
        if vault is None:
            return None, (
                "Obsidian não configurado. Defina OBSIDIAN_VAULT=/caminho/do/vault no .env."
            )
        if not vault.is_dir():
            return None, "OBSIDIAN_VAULT não é um diretório válido: {}".format(vault)
        return vault, None

    def _vault_path(self, raw: str, must_exist: bool = False) -> Path:
        vault = self.settings.obsidian_vault
        assert vault is not None
        path = Path(str(raw))
        if path.suffix == "":
            path = Path(str(path) + ".md")
        return security.resolve_path(
            str(path), roots=[vault], base=vault, must_exist=must_exist
        )

    def _obsidian(self, args: Dict[str, Any]) -> str:
        vault, err = self._vault()
        if err:
            return err
        assert vault is not None
        action = str(args.get("action", "")).strip().lower()
        try:
            limit = max(1, min(int(args.get("limit") or 20), 100))
        except (TypeError, ValueError):
            limit = 20

        if action == "search":
            query = str(args.get("query") or "").strip()
            if not query:
                return "Informe 'query'."
            needle = query.lower()
            matches: List[str] = []
            for path in sorted(vault.rglob("*.md")):
                if ".obsidian" in path.parts or security.is_sensitive(path):
                    continue
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if needle in path.name.lower() or needle in content.lower():
                    index = content.lower().find(needle)
                    if index >= 0:
                        snippet = content[max(0, index - 120) : index + 220]
                    else:
                        snippet = content[:300]
                    snippet = " ".join(snippet.split())
                    matches.append(
                        "{}: ...{}...".format(path.relative_to(vault), snippet)
                    )
                    if len(matches) >= limit:
                        break
            return "\n".join(matches) or "Nenhuma nota encontrada para {!r}.".format(query)

        if action == "read":
            raw = str(args.get("path") or "").strip()
            if not raw:
                return "Informe 'path'."
            try:
                path = self._vault_path(raw, must_exist=True)
            except SecurityError as exc:
                return "Erro: {}".format(exc)
            if not path.is_file():
                return "Não é um arquivo: {}".format(path)
            try:
                return "Nota: {}\n\n{}".format(
                    path.relative_to(vault),
                    path.read_text(encoding="utf-8", errors="replace"),
                )
            except OSError as exc:
                return "Erro ao ler: {}".format(exc)

        if action == "write":
            raw = str(args.get("path") or "").strip()
            content = str(args.get("content") or "")
            if not raw:
                return "Informe 'path'."
            try:
                path = self._vault_path(raw)
            except SecurityError as exc:
                return "Erro: {}".format(exc)
            mode = str(args.get("mode") or "append").lower()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if mode == "overwrite":
                    path.write_text(content, encoding="utf-8")
                else:
                    with path.open("a", encoding="utf-8") as handle:
                        if path.exists() and path.stat().st_size > 0:
                            handle.write("\n")
                        handle.write(content)
            except OSError as exc:
                return "Erro ao escrever: {}".format(exc)
            return "Nota {}: {}".format(
                "sobrescrita" if mode == "overwrite" else "atualizada",
                path.relative_to(vault),
            )

        if action == "list":
            folder = str(args.get("folder") or "").strip()
            try:
                root = self._vault_path(folder, must_exist=True) if folder else vault
            except SecurityError as exc:
                return "Erro: {}".format(exc)
            if root.is_file():
                return str(root.relative_to(vault))
            lines: List[str] = []
            for path in sorted(root.rglob("*.md")):
                if ".obsidian" in path.parts or security.is_sensitive(path):
                    continue
                lines.append(str(path.relative_to(vault)))
                if len(lines) >= limit:
                    break
            return "\n".join(lines) or "Nenhuma nota encontrada."

        return "Ação desconhecida. Use: search, read, write, list."

    # -------------------------------------------------------------------- Notion
    def _notion_request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Any], Optional[str]]:
        if not self.settings.notion_token:
            return None, (
                "Notion não configurado. Crie uma integração em "
                "https://www.notion.so/my-integrations, compartilhe as páginas com ela "
                "e defina NOTION_TOKEN no .env."
            )
        headers = {
            "Authorization": "Bearer " + self.settings.notion_token,
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.request(
                    method,
                    "https://api.notion.com/v1" + path,
                    headers=headers,
                    json=body,
                    params=params,
                )
        except httpx.HTTPError as exc:
            return None, "Erro de rede no Notion: {}".format(exc)
        if response.status_code == 401:
            return None, "Notion recusou o token (401). Verifique NOTION_TOKEN."
        if response.status_code == 404:
            return None, (
                "Página não encontrada (404). Compartilhe a página com a integração "
                "no Notion (menu ... > Connections)."
            )
        if response.status_code >= 400:
            return None, "Notion HTTP {}: {}".format(
                response.status_code, response.text[:600]
            )
        if response.status_code == 204 or not response.content:
            return {}, None
        try:
            return response.json(), None
        except ValueError:
            return None, "Resposta inválida do Notion."

    @staticmethod
    def _rich_text(text: str) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        remaining = text
        while remaining:
            piece = remaining[:1900]
            remaining = remaining[1900:]
            chunks.append({"type": "text", "text": {"content": piece}})
        return chunks or [{"type": "text", "text": {"content": ""}}]

    def _paragraph_blocks(self, content: str) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for line in content.splitlines():
            if not line.strip():
                continue
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": self._rich_text(line)},
                }
            )
        return blocks

    @staticmethod
    def _notion_title(obj: Dict[str, Any]) -> str:
        if obj.get("object") == "database":
            title = obj.get("title") or []
            return "".join(part.get("plain_text", "") for part in title) or "Sem título"
        properties = obj.get("properties") or {}
        for value in properties.values():
            if value.get("type") == "title":
                return (
                    "".join(
                        part.get("plain_text", "") for part in value.get("title", [])
                    )
                    or "Sem título"
                )
        return "Sem título"

    def _notion_block_line(self, block: Dict[str, Any]) -> str:
        block_type = block.get("type") or ""
        data = block.get(block_type) or {}
        rich = data.get("rich_text") or []
        text = "".join(part.get("plain_text", "") for part in rich)

        if block_type == "heading_1":
            return "# " + text
        if block_type == "heading_2":
            return "## " + text
        if block_type == "heading_3":
            return "### " + text
        if block_type == "bulleted_list_item":
            return "- " + text
        if block_type == "numbered_list_item":
            return "1. " + text
        if block_type == "to_do":
            return ("[x] " if data.get("checked") else "[ ] ") + text
        if block_type == "quote":
            return "> " + text
        if block_type == "code":
            language = data.get("language") or ""
            return "```{}\n{}\n```".format(language, text)
        if block_type == "divider":
            return "---"
        if block_type == "child_page":
            return "[subpágina] " + (data.get("title") or "")
        if block_type in {"bookmark", "link_preview"}:
            return data.get("url") or ""
        if block_type == "image":
            url = (data.get("file") or {}).get("url") or data.get("external", {}).get("url", "")
            return "[imagem] " + url
        return text

    def _notion_blocks(self, block_id: str, depth: int = 0) -> Tuple[str, Optional[str]]:
        lines: List[str] = []
        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            data, err = self._notion_request(
                "GET", "/blocks/{}/children".format(block_id), params=params
            )
            if err:
                return "\n".join(lines), err
            for block in data.get("results", []):
                line = self._notion_block_line(block)
                if line:
                    lines.append("  " * depth + line)
                if (
                    block.get("has_children")
                    and depth < 1
                    and block.get("type") != "child_page"
                ):
                    sub, _ = self._notion_blocks(block.get("id", ""), depth + 1)
                    if sub:
                        lines.append(sub)
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
        return "\n".join(lines), None

    def _notion(self, args: Dict[str, Any]) -> str:
        action = str(args.get("action", "")).strip().lower()

        if action == "search":
            query = str(args.get("query") or "").strip()
            try:
                limit = max(1, min(int(args.get("limit") or 10), 50))
            except (TypeError, ValueError):
                limit = 10
            body: Dict[str, Any] = {"page_size": limit}
            if query:
                body["query"] = query
            data, err = self._notion_request("POST", "/search", body=body)
            if err:
                return err
            lines = []
            for item in data.get("results", []):
                lines.append(
                    "- [{}] {} (id: {}) {}".format(
                        item.get("object"),
                        self._notion_title(item),
                        item.get("id"),
                        item.get("url") or "",
                    ).rstrip()
                )
            return "\n".join(lines) or "Nenhum resultado."

        if action == "read":
            page_id = str(args.get("page_id") or "").strip()
            if not page_id:
                return "Informe 'page_id'."
            page, err = self._notion_request("GET", "/pages/{}".format(page_id))
            if err:
                return err
            content, blocks_err = self._notion_blocks(page_id)
            title = self._notion_title(page)
            header = "Página: {}\nURL: {}\n".format(title, page.get("url") or "-")
            if blocks_err:
                return header + "\n[aviso] " + blocks_err
            return header + "\n" + (content or "(página sem conteúdo textual)")

        if action == "append":
            page_id = str(
                args.get("page_id") or self.settings.notion_default_page or ""
            ).strip()
            content = str(args.get("content") or "")
            if not page_id:
                return "Informe 'page_id' (ou defina NOTION_DEFAULT_PAGE)."
            if not content.strip():
                return "Informe 'content'."
            blocks = self._paragraph_blocks(content)
            appended = 0
            for start in range(0, len(blocks), 100):
                batch = blocks[start : start + 100]
                _, err = self._notion_request(
                    "POST",
                    "/blocks/{}/children".format(page_id),
                    body={"children": batch},
                )
                if err:
                    return "Erro ao anexar: {}".format(err)
                appended += len(batch)
            return "Anexados {} bloco(s) na página {}.".format(appended, page_id)

        if action == "create_page":
            parent = str(
                args.get("parent_page_id") or self.settings.notion_default_page or ""
            ).strip()
            title = str(args.get("title") or "").strip()
            content = str(args.get("content") or "")
            if not parent:
                return "Informe 'parent_page_id' (ou defina NOTION_DEFAULT_PAGE)."
            if not title:
                return "Informe 'title'."
            body = {
                "parent": {"page_id": parent},
                "properties": {
                    "title": {"title": [{"type": "text", "text": {"content": title}}]}
                },
                "children": self._paragraph_blocks(content),
            }
            data, err = self._notion_request("POST", "/pages", body=body)
            if err:
                return err
            return "Página criada: {} ({})".format(
                self._notion_title(data), data.get("url") or data.get("id")
            )

        return "Ação desconhecida. Use: search, read, append, create_page."

    # --------------------------------------------------------------- Planilhas
    def _spreadsheet(self, args: Dict[str, Any]) -> str:
        action = str(args.get("action") or "read").lower()
        if action != "read":
            return "Ação desconhecida. Use: read."
        raw = str(args.get("path") or "").strip()
        if not raw:
            return "Informe 'path'."
        try:
            path = security.resolve_path(
                raw,
                roots=self.settings.allowed_roots,
                base=self.settings.workspace,
                must_exist=True,
            )
        except SecurityError as exc:
            return "Erro: {}".format(exc)
        if not path.is_file():
            return "Não é um arquivo: {}".format(path)

        try:
            max_rows = max(1, min(int(args.get("max_rows") or 50), 500))
        except (TypeError, ValueError):
            max_rows = 50
        suffix = path.suffix.lower()

        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            text = None
            for encoding in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    text = path.read_text(encoding=encoding)
                    break
                except (UnicodeDecodeError, OSError):
                    continue
            if text is None:
                return "Não consegui ler o arquivo."
            rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
            return self._format_table(path, rows, max_rows, [])

        if suffix in {".xlsx", ".xlsm"}:
            try:
                from openpyxl import load_workbook
            except ImportError:
                return "openpyxl não instalado. Rode `pip install openpyxl` no .venv."
            try:
                workbook = load_workbook(filename=str(path), read_only=True, data_only=True)
            except Exception as exc:  # noqa: BLE001
                return "Erro ao abrir planilha: {}".format(exc)
            sheet_names = list(workbook.sheetnames)
            sheet_name = str(args.get("sheet") or "").strip() or sheet_names[0]
            if sheet_name not in workbook.sheetnames:
                workbook.close()
                return "Aba '{}' não existe. Abas: {}.".format(
                    sheet_name, ", ".join(sheet_names)
                )
            sheet = workbook[sheet_name]
            rows = []
            for index, row in enumerate(sheet.iter_rows(values_only=True)):
                if index >= max_rows:
                    break
                rows.append(["" if cell is None else str(cell) for cell in row])
            workbook.close()
            return self._format_table(path, rows, max_rows, sheet_names, sheet_name)

        return "Formato não suportado: {}. Use CSV, TSV, XLSX ou XLSM.".format(suffix or "?")

    @staticmethod
    def _format_table(
        path: Path,
        rows: List[List[str]],
        max_rows: int,
        sheet_names: List[str],
        sheet_name: Optional[str] = None,
    ) -> str:
        if not rows:
            return "Planilha vazia: {}".format(path)
        max_columns = 30
        lines: List[str] = ["Arquivo: {}".format(path)]
        if sheet_names:
            lines.append("Abas: {}".format(", ".join(sheet_names)))
            if sheet_name:
                lines.append("Aba: {}".format(sheet_name))
        lines.append("Linhas lidas: {}".format(min(len(rows), max_rows)))
        for row in rows[:max_rows]:
            cells = [
                str(cell).replace("\t", " ").replace("\n", " ")[:200]
                for cell in row[:max_columns]
            ]
            lines.append("\t".join(cells))
        if len(rows) >= max_rows:
            lines.append("... (limitado a {} linhas)".format(max_rows))
        return "\n".join(lines)

    def _register(self) -> None:
        self.registry.register(
            "obsidian",
            "Acessa um vault do Obsidian: buscar notas, ler, escrever/append e listar. "
            "Requer OBSIDIAN_VAULT.",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "read", "write", "list"],
                    },
                    "query": {"type": "string", "description": "Busca (search)."},
                    "path": {
                        "type": "string",
                        "description": "Caminho relativo da nota (read/write).",
                    },
                    "content": {"type": "string", "description": "Conteúdo (write)."},
                    "mode": {
                        "type": "string",
                        "enum": ["append", "overwrite"],
                        "description": "Modo de escrita (padrão append).",
                    },
                    "folder": {"type": "string", "description": "Pasta (list)."},
                    "limit": {"type": "integer", "description": "Máximo de itens."},
                },
                "required": ["action"],
            },
            self._obsidian,
        )
        self.registry.register(
            "notion",
            "Acessa o Notion: buscar páginas, ler conteúdo, anexar texto e criar página. "
            "Requer NOTION_TOKEN e página compartilhada com a integração.",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "read", "append", "create_page"],
                    },
                    "query": {"type": "string", "description": "Busca (search)."},
                    "page_id": {"type": "string", "description": "ID da página (read/append)."},
                    "parent_page_id": {
                        "type": "string",
                        "description": "Página pai (create_page).",
                    },
                    "title": {"type": "string", "description": "Título (create_page)."},
                    "content": {"type": "string", "description": "Texto (append/create_page)."},
                    "limit": {"type": "integer", "description": "Máximo de resultados."},
                },
                "required": ["action"],
            },
            self._notion,
        )
        self.registry.register(
            "spreadsheet",
            "Lê planilhas CSV/TSV/XLSX e retorna uma prévia em texto para análise.",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read"],
                    },
                    "path": {"type": "string", "description": "Caminho do arquivo."},
                    "sheet": {"type": "string", "description": "Nome da aba (XLSX)."},
                    "max_rows": {"type": "integer", "description": "Máximo de linhas."},
                },
                "required": ["action", "path"],
            },
            self._spreadsheet,
        )
