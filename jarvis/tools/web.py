from __future__ import annotations

from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from .. import security
from ..config import Settings
from ..security import SecurityError
from .base import ToolRegistry

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 Jarvis/0.2"
)


class WebTools:
    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self._register()

    def _register(self) -> None:
        self.registry.register(
            "web_search",
            "Busca na web (DuckDuckGo) e retorna títulos, URLs e trechos. "
            "Use para notícias, documentação e informações atuais.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Termos da busca."},
                    "max_results": {
                        "type": "integer",
                        "description": "Máximo de resultados (padrão 5).",
                    },
                },
                "required": ["query"],
            },
            self._web_search,
        )
        self.registry.register(
            "web_fetch",
            "Baixa uma página/URL e retorna o texto principal. Use depois de web_search "
            "para ler uma fonte específica.",
            {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL http/https."},
                    "max_chars": {
                        "type": "integer",
                        "description": "Máximo de caracteres (padrão 8000).",
                    },
                },
                "required": ["url"],
            },
            self._web_fetch,
        )

    @staticmethod
    def _resolve_ddg_url(href: str) -> str:
        if not href:
            return ""
        if href.startswith("//"):
            href = "https:" + href
        parsed = urlparse(href)
        host = (parsed.hostname or "").lower()
        if "duckduckgo.com" in host and parsed.path.startswith("/l/"):
            params = parse_qs(parsed.query)
            if params.get("uddg"):
                return unquote(params["uddg"][0])
        return href

    def _parse_ddg(self, html_text: str, limit: int) -> List[Tuple[str, str, str]]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_text, "html.parser")
        results: List[Tuple[str, str, str]] = []
        for block in soup.select(".result"):
            classes = block.get("class") or []
            if "result--ad" in classes:
                continue
            link = block.select_one("a.result__a")
            if link is None:
                continue
            title = link.get_text(" ", strip=True)
            url = self._resolve_ddg_url(link.get("href") or "")
            snippet_node = block.select_one(".result__snippet")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            if title and url:
                results.append((title, url, snippet))
            if len(results) >= limit:
                break
        return results

    def _parse_ddg_lite(self, html_text: str, limit: int) -> List[Tuple[str, str, str]]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_text, "html.parser")
        links = soup.select("a.result-link")
        snippets = soup.select(".result-snippet")
        results: List[Tuple[str, str, str]] = []
        for index, link in enumerate(links):
            if len(results) >= limit:
                break
            title = link.get_text(" ", strip=True)
            url = self._resolve_ddg_url(link.get("href") or "")
            snippet = (
                snippets[index].get_text(" ", strip=True) if index < len(snippets) else ""
            )
            if title and url:
                results.append((title, url, snippet))
        return results

    @staticmethod
    def _format(results: List[Tuple[str, str, str]]) -> str:
        lines = []
        for index, (title, url, snippet) in enumerate(results, start=1):
            lines.append("{}. {}".format(index, title))
            lines.append("   {}".format(url))
            if snippet:
                lines.append("   {}".format(snippet))
        return "\n".join(lines)

    def _web_search(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return "Informe o campo 'query'."
        try:
            limit = max(1, min(int(args.get("max_results") or 5), 10))
        except (TypeError, ValueError):
            limit = 5

        try:
            from bs4 import BeautifulSoup  # noqa: F401
        except ImportError:
            return "beautifulsoup4 não instalado. Rode ./setup.sh novamente."

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        }
        try:
            with httpx.Client(
                timeout=25.0, follow_redirects=True, headers=headers
            ) as client:
                response = client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query, "kl": "br-pt"},
                )
                response.raise_for_status()
                results = self._parse_ddg(response.text, limit)

                if not results:
                    response = client.get(
                        "https://lite.duckduckgo.com/lite/",
                        params={"q": query},
                    )
                    response.raise_for_status()
                    results = self._parse_ddg_lite(response.text, limit)
        except httpx.HTTPError as exc:
            return "Erro na busca web: {}".format(exc)

        if not results:
            return (
                "Nenhum resultado encontrado para {!r}. "
                "Se você já tem uma URL, use web_fetch."
            ).format(query)
        return "Resultados para {!r}:\n\n{}".format(query, self._format(results))

    def _web_fetch(self, args: Dict[str, Any]) -> str:
        raw_url = str(args.get("url") or "").strip()
        if not raw_url:
            return "Informe o campo 'url'."
        try:
            url = security.check_url(raw_url)
        except SecurityError as exc:
            return "Bloqueado: {}".format(exc)
        try:
            max_chars = max(1000, min(int(args.get("max_chars") or 8000), 20000))
        except (TypeError, ValueError):
            max_chars = 8000

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/json,text/plain",
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        }
        chunks: List[bytes] = []
        total = 0
        try:
            with httpx.Client(
                timeout=25.0, follow_redirects=True, headers=headers
            ) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    final_url = str(response.url)
                    for chunk in response.iter_bytes():
                        chunks.append(chunk)
                        total += len(chunk)
                        if total >= 3_000_000:
                            break
        except httpx.HTTPError as exc:
            return "Erro ao buscar a URL: {}".format(exc)

        raw = b"".join(chunks)
        encoding = "utf-8"
        if "charset=" in content_type.lower():
            encoding = (
                content_type.lower().split("charset=")[-1].split(";")[0].strip()
            )
        try:
            text = raw.decode(encoding, errors="replace")
        except LookupError:
            text = raw.decode("utf-8", errors="replace")

        title = ""
        if "html" in content_type.lower() or text.lstrip().startswith("<"):
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(text, "html.parser")
                for tag in soup(
                    ["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]
                ):
                    tag.decompose()
                if soup.title:
                    title = soup.title.get_text(" ", strip=True)
                extracted = soup.get_text("\n")
                lines = [" ".join(line.split()) for line in extracted.splitlines()]
                text = "\n".join(line for line in lines if line)
            except ImportError:
                pass

        text = text.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (conteúdo truncado)"

        header_lines = ["URL: {}".format(final_url)]
        if title:
            header_lines.append("Título: {}".format(title))
        return "\n".join(header_lines) + "\n\n" + (text or "(sem texto extraído)")
