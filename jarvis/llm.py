from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class OllamaError(Exception):
    """Erro genérico do Ollama."""


class OllamaUnavailable(OllamaError):
    """Ollama não está rodando ou não respondeu."""


class ToolsNotSupported(OllamaError):
    """O modelo não suporta function calling nativo (HTTP 400)."""


class OllamaClient:
    def __init__(
        self,
        url: str,
        model: str,
        timeout: float = 300.0,
        num_ctx: int = 8192,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
        # Opções extras do Ollama (ex.: {"seed": 42} nas avaliações). Padrão: nenhuma.
        self.extra_options: Dict[str, Any] = dict(extra_options or {})
        self.tools_supported = True

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(connect=10.0, read=self.timeout, write=30.0, pool=10.0)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Envia a conversa e devolve só a mensagem do assistente."""
        data = await self.chat_full(messages, tools=tools)
        return data["message"]

    async def chat_full(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Como chat(), mas devolve a resposta inteira do Ollama.

        Inclui as métricas (eval_count, eval_duration, prompt_eval_count,
        total_duration...), usadas pelas avaliações e pela observabilidade.
        """
        options: Dict[str, Any] = {
            "temperature": 0.2,
            "num_ctx": self.num_ctx,
        }
        options.update(self.extra_options)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if tools and self.tools_supported:
            payload["tools"] = tools

        try:
            async with httpx.AsyncClient(timeout=self._timeout()) as client:
                response = await client.post(self.url + "/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise OllamaUnavailable(
                "Não consegui conectar no Ollama em {}. "
                "Rode `ollama serve` (ou `./scripts/install_ollama.sh` se não instalou).".format(
                    self.url
                )
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaUnavailable(
                "O Ollama demorou mais de {:.0f}s para responder. "
                "Tente um modelo menor (ex.: qwen2.5:3b) ou aumente OLLAMA_TIMEOUT.".format(
                    self.timeout
                )
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError("Erro de rede falando com o Ollama: {}".format(exc)) from exc

        if response.status_code == 400 and tools and self.tools_supported:
            body = response.text.lower()
            if "tool" in body or "function" in body:
                self.tools_supported = False
                raise ToolsNotSupported(response.text[:500])

        if response.status_code >= 400:
            raise OllamaError(
                "Ollama retornou HTTP {}: {}".format(response.status_code, response.text[:800])
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError("Resposta inválida do Ollama (não era JSON).") from exc

        message = data.get("message") if isinstance(data, dict) else None
        if not message:
            raise OllamaError("O Ollama não retornou nenhuma mensagem.")
        return data

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(4.0)) as client:
                response = await client.get(self.url + "/api/tags")
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(6.0)) as client:
                response = await client.get(self.url + "/api/tags")
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError):
            return []
        return [item.get("name", "") for item in data.get("models", []) if item.get("name")]
