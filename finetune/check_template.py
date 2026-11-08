#!/usr/bin/env python3
"""Confere ``finetune/chat_format.py`` contra o Ollama local.

Para cada conversa de teste, manda a mesma coisa de dois jeitos (descarregando o
modelo antes de cada chamada, para o cache de prompt não interferir):

1. ``/api/chat`` com ``messages`` + ``tools`` (o Ollama aplica o template);
2. ``/api/generate`` com ``raw: true`` e o texto de ``render_prompt`` (template nenhum).

Se o renderizador reproduz o template, as duas chamadas têm o **mesmo número de
tokens de prompt** e, com temperatura 0 e seed fixo, a **mesma saída**. Usa o
system prompt e os 25 schemas reais do Jarvis. Leve: só inferência, sem treino.

    .venv/Scripts/python.exe finetune/check_template.py --model qwen2.5:1.5b

Saída 0 = tudo bateu; 1 = alguma divergência (o texto de treino não seria o da inferência).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import httpx  # noqa: E402

from finetune.chat_format import render_prompt  # noqa: E402


def _conversations(system: str, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sys_msg = {"role": "system", "content": system}
    return [
        {"name": "pedido com ferramenta", "tools": tools,
         "messages": [sys_msg, {"role": "user", "content": "me lembra de regar as plantas em 15 minutos"}]},
        {"name": "conversa simples", "tools": tools,
         "messages": [sys_msg, {"role": "user", "content": "bom dia! como você está?"}]},
        {"name": "depois do resultado da ferramenta", "tools": tools,
         "messages": [
             sys_msg,
             {"role": "user", "content": "o que tem na pasta projetos?"},
             {"role": "assistant", "content": "", "tool_calls": [
                 {"function": {"name": "list_dir", "arguments": {"path": "projetos"}}}]},
             {"role": "tool", "content": "[dir] jarvis\n[dir] site\nnotas.md"},
         ]},
        {"name": "histórico com dois argumentos", "tools": tools,
         "messages": [
             sys_msg,
             {"role": "user", "content": "cria o arquivo ideias.md com o texto 'testar o LoRA'"},
             {"role": "assistant", "content": "", "tool_calls": [
                 {"function": {"name": "write_file",
                               "arguments": {"path": "ideias.md", "content": "testar o LoRA"}}}]},
             {"role": "tool", "content": "Arquivo salvo: ideias.md (13 caracteres)"},
             {"role": "assistant", "content": "Pronto, salvei ideias.md."},
             # Pedir a cópia literal faz o modelo revelar a ordem das chaves que viu no
             # histórico: distingue "ordem alfabética" (map do Go) de "ordem original".
             {"role": "user", "content": "Copie literalmente, caractere por caractere, o JSON "
              "dos argumentos da chamada de ferramenta que você fez antes. Responda só com o "
              "JSON entre crases."},
         ]},
        {"name": "sem ferramentas", "tools": [],
         "messages": [sys_msg, {"role": "user", "content": "quanto é 12 vezes 7?"}]},
        {"name": "sem system (usa o SYSTEM do Modelfile)", "tools": tools,
         "messages": [{"role": "user", "content": "lista os arquivos da pasta fotos"}]},
    ]


def _real_system_and_tools() -> Any:
    from evals import common
    from jarvis.agent import Agent
    from jarvis.memory import Memory
    from jarvis.rag import RagIndex
    from jarvis.reminders import ReminderService
    from jarvis.tools import build_registry

    with tempfile.TemporaryDirectory() as tmp:
        settings = common.make_eval_settings(Path(tmp))
        rag = RagIndex(settings)
        reminders = ReminderService(settings)
        memory = Memory(settings.db_path)
        try:
            registry = build_registry(settings, reminders, rag)
            agent = Agent(settings, registry, memory, llm=object())
            return agent._system_prompt(), registry.schemas()
        finally:
            memory.close()
            rag.close()
            reminders.close()



def _call_key(name: str, arguments: Any) -> str:
    return "CALL {} {}".format(name, json.dumps(arguments, ensure_ascii=False, sort_keys=True))


def _normalize_text(text: str) -> str:
    """Saída crua do /api/generate -> forma comparável.

    No modo ``raw`` o Ollama não devolve os tokens especiais ``<tool_call>`` e
    ``</tool_call>`` no texto; se a saída é uma chamada, comparamos nome + args.
    """
    stripped = text.replace("<tool_call>", "").replace("</tool_call>", "").strip()
    if stripped.startswith("{"):
        try:
            call = json.loads(stripped.splitlines()[0])
            return _call_key(call["name"], call.get("arguments") or {})
        except (ValueError, KeyError, IndexError, TypeError):
            pass
    return stripped


def _normalize_chat(message: Dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if calls:
        function = calls[0].get("function") or {}
        return _call_key(function.get("name", ""), function.get("arguments") or {})
    return _normalize_text(message.get("content") or "")


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--num-predict", type=int, default=60)
    args = parser.parse_args(argv)

    system, tools = _real_system_and_tools()
    options = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": args.num_predict}
    url = args.ollama_url.rstrip("/")
    failures = 0
    with httpx.Client(timeout=300.0) as client:

        def unload() -> None:
            # Descarrega o modelo: zera o cache de prompt do Ollama, senão a 2ª chamada
            # reaproveitaria o prefixo e o prompt_eval_count viria menor.
            client.post(url + "/api/generate", json={"model": args.model, "keep_alive": 0})

        for conv in _conversations(system, tools):
            unload()
            chat = client.post(url + "/api/chat", json={
                "model": args.model, "stream": False, "options": options,
                "messages": conv["messages"], "tools": conv["tools"] or None,
            }).json()
            unload()
            raw = client.post(url + "/api/generate", json={
                "model": args.model, "stream": False, "raw": True, "options": options,
                "prompt": render_prompt(conv["messages"], conv["tools"]),
            }).json()
            if "error" in chat or "error" in raw:
                print("ERRO do Ollama:", chat.get("error") or raw.get("error"))
                return 2
            chat_out = _normalize_chat(chat.get("message") or {})
            raw_out = _normalize_text(raw.get("response") or "")
            same_tokens = chat.get("prompt_eval_count") == raw.get("prompt_eval_count")
            same_output = chat_out == raw_out
            ok = same_tokens and same_output
            failures += 0 if ok else 1
            print("[{}] {:<42} tokens chat/raw = {}/{}  saída igual: {}".format(
                "ok " if ok else "ERR", conv["name"], chat.get("prompt_eval_count"),
                raw.get("prompt_eval_count"), "sim" if same_output else "NÃO"))
            if not same_output:
                print("      chat: {!r}\n      raw:  {!r}".format(chat_out[:200], raw_out[:200]))
        unload()
    print("\n{}".format("Renderizador idêntico ao template do Ollama." if not failures
                        else "{} divergência(s).".format(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
