"""Renderiza conversas exatamente como o Ollama monta o prompt do ``qwen2.5``.

O modelo ajustado vai rodar atrás do Ollama, então o texto de treino precisa ser
o mesmo que o Ollama entrega ao modelo na inferência. Este módulo reproduz o
template Go do ``qwen2.5`` (``ollama show qwen2.5:3b --modelfile``), que é o
mesmo copiado em ``finetune/Modelfile``.

Achado importante (Ollama 0.40.14, conferido com ``check_template.py``): o
template oficial imprime cada ferramenta com ``{{ .Function }}``, e nessa
versão o tipo da função não tem mais um ``String()`` que devolva JSON. O Go
imprime a struct com ``%v``:

    {"type": "function", "function": {git Executa comandos git... {object <nil> <nil> [args] {"args":{...},"cwd":{...}}}}}

ou seja: nome e descrição soltos, ``<nil> <nil>`` (``$defs`` e ``items`` vazios),
a lista ``required`` no formato do Go (``[a b]``) e só as ``properties`` em JSON
(esse tipo tem ``String()``), com as chaves em ordem alfabética, como todo ``map``
do Go. É isso que o ``qwen2.5`` do Ollama vê hoje — no
eval da linha de base também. Reproduzimos o formato byte a byte em vez de
"corrigir", para não criar diferença entre treino e serviço.

Só usa a biblioteca padrão: roda no Python 3.8 do projeto e no Colab.
"""  # noqa: E501 - o exemplo do Go fica numa linha só
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Ordem dos campos da struct de propriedade do Ollama (type, items, description, enum).
_PROPERTY_FIELDS = ("type", "items", "description", "enum")

_TOOLS_HEADER = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>"
)
_TOOLS_FOOTER = (
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)

# SYSTEM padrão do Modelfile oficial: o Ollama só usa quando a conversa não tem system.
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


# ---------------------------------------------------------------- JSON do Go


def go_json(value: Any, sort_keys: bool = False) -> str:
    """``json.Marshal`` do Go: compacto, UTF-8 cru e escape de ``<``, ``>`` e ``&``.

    ``sort_keys`` imita um ``map`` do Go (chaves em ordem alfabética).
    """
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys)
    return (
        text.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def _go_value(value: Any) -> str:
    """``%v`` do Go para os campos simples que aparecem na struct."""
    if value is None:
        return "<nil>"
    if isinstance(value, (list, tuple)):
        return "[" + " ".join(_go_value(item) for item in value) + "]"
    return str(value)


def _property(prop: Dict[str, Any]) -> Dict[str, Any]:
    return {field: prop[field] for field in _PROPERTY_FIELDS if field in prop}


def render_function(function: Dict[str, Any]) -> str:
    """O que ``{{ .Function }}`` imprime para uma ferramenta (ver docstring do módulo)."""
    params = function.get("parameters") or {}
    properties = params.get("properties") or {}
    # Mapa do Go: as chaves saem em ordem alfabética (bytes), não na ordem do schema.
    props_json = go_json({name: _property(properties[name]) for name in sorted(properties)})
    return "{{{} {} {{{} <nil> <nil> {} {}}}}}".format(
        function.get("name", ""),
        function.get("description", ""),
        params.get("type", ""),
        _go_value(list(params.get("required") or [])),
        props_json,
    )


def render_arguments(arguments: Dict[str, Any]) -> str:
    """``{{ .Function.Arguments }}`` em chamadas já feitas (histórico): JSON do Go.

    Os argumentos são um ``map`` no Go, então as chaves saem em ordem alfabética
    (conferido com ``check_template.py``, caso "histórico com dois argumentos").
    """
    return go_json(arguments, sort_keys=True)


def target_arguments(arguments: Dict[str, Any]) -> str:
    """Argumentos na resposta que o modelo deve gerar.

    O parser do Ollama aceita qualquer JSON válido; usamos o estilo que o Qwen
    já produz (``json.dumps`` com espaços), na ordem das propriedades do schema.
    """
    return json.dumps(arguments, ensure_ascii=False)


# ---------------------------------------------------------------- mensagens


def _tool_calls(message: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function", call)
        args = function.get("arguments") or {}
        if isinstance(args, str):
            args = json.loads(args)
        calls.append((function["name"], args))
    return calls


def _render_assistant(message: Dict[str, Any], for_target: bool) -> str:
    content = message.get("content") or ""
    if content:
        return content
    calls = _tool_calls(message)
    if not calls:
        return ""
    fmt = target_arguments if for_target else render_arguments
    body = "".join(
        '{{"name": "{}", "arguments": {}}}\n'.format(name, fmt(args)) for name, args in calls
    )
    return "<tool_call>\n" + body + "</tool_call>"


def _collate(messages: Sequence[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Igual ao ``collate`` do Ollama: junta os system e funde papéis repetidos."""
    system: List[str] = []
    collated: List[Dict[str, Any]] = []
    for message in messages:
        if message["role"] == "system":
            system.append(message.get("content") or "")
        if collated and collated[-1]["role"] == message["role"]:
            merged = dict(collated[-1])
            merged["content"] = (merged.get("content") or "") + "\n\n" + (message.get("content") or "")
            collated[-1] = merged
        else:
            collated.append(dict(message))
    return "\n\n".join(system), collated


def render_prompt(
    messages: Sequence[Dict[str, Any]],
    tools: Optional[Sequence[Dict[str, Any]]] = None,
    default_system: str = DEFAULT_SYSTEM,
) -> str:
    """Texto que o Ollama manda ao modelo para ``messages`` (sem a resposta nova).

    Se a última mensagem não for do assistente, termina com
    ``<|im_start|>assistant\\n`` (o modelo gera a partir daí).
    """
    system, msgs = _collate(messages)
    if not system:
        system = default_system
    tools = list(tools or [])
    out: List[str] = []
    if system or tools:
        out.append(IM_START + "system")
        if system:
            out.append("\n" + system)
        if tools:
            out.append(_TOOLS_HEADER)
            for tool in tools:
                function = tool.get("function", tool)
                out.append('\n{"type": "function", "function": ' + render_function(function) + "}")
            out.append(_TOOLS_FOOTER)
        out.append(IM_END + "\n")
    for index, message in enumerate(msgs):
        last = index == len(msgs) - 1
        role = message["role"]
        if role == "user":
            out.append(IM_START + "user\n" + (message.get("content") or "") + IM_END + "\n")
        elif role == "assistant":
            out.append(IM_START + "assistant\n" + _render_assistant(message, for_target=False))
            if not last:
                out.append(IM_END + "\n")
        elif role == "tool":
            out.append(
                IM_START + "user\n<tool_response>\n" + (message.get("content") or "")
                + "\n</tool_response>" + IM_END + "\n"
            )
        if role != "assistant" and last:
            out.append(IM_START + "assistant\n")
    return "".join(out)


def render_example(
    example: Dict[str, Any], tools: Optional[Sequence[Dict[str, Any]]] = None
) -> Tuple[str, str]:
    """(prompt, resposta) de um exemplo de treino.

    ``tools``: os schemas (``finetune/data/tools.json``); se omitido, usa
    ``example["tools"]``. A resposta é a **última** mensagem do assistente
    seguida de ``<|im_end|>`` (o token de parada do Qwen); a perda do treino é
    calculada só sobre ela.
    """
    messages = list(example["messages"])
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("o exemplo precisa terminar com uma mensagem do assistente")
    prompt = render_prompt(messages[:-1], tools if tools is not None else example.get("tools"))
    completion = _render_assistant(messages[-1], for_target=True) + IM_END
    return prompt, completion
