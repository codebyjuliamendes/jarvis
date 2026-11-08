"""Interpretação das chamadas de ferramenta devolvidas pelo LLM.

Usado pelo agente (jarvis/agent.py) e pelas avaliações (evals/), para que a
avaliação meça exatamente o que o agente faria com a resposta do modelo.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_CANDIDATE_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_tool_call(call: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Normaliza um item de ``tool_calls`` (function calling nativo).

    Aceita ``{"function": {"name", "arguments"}}`` ou o objeto sem o envelope.
    ``arguments`` pode vir como dict ou string JSON; o que não for dict vira ``{}``.
    """
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict):
        function = call if isinstance(call, dict) else {}
    name = function.get("name") or ""
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (TypeError, ValueError):
            args = {}
    if not isinstance(args, dict):
        args = {}
    return str(name), args


def extract_text_tool(content: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Extrai ``{"tool": ..., "args": {...}}`` de uma resposta em texto (modo JSON).

    Tenta primeiro blocos cercados por crases e depois o maior trecho entre
    chaves. Aceita as chaves ``tool``/``name`` e ``args``/``arguments``.
    Devolve ``None`` se nada válido for encontrado.
    """
    if not content:
        return None
    candidates: List[str] = []
    candidates.extend(_JSON_BLOCK_RE.findall(content))
    candidates.extend(_JSON_CANDIDATE_RE.findall(content))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("tool") or data.get("name")
        args = data.get("args") or data.get("arguments") or {}
        if isinstance(name, str) and isinstance(args, dict):
            return name, args
    return None
