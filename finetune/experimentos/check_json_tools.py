#!/usr/bin/env python3
"""Confere o que o ``{{ json .Function }}`` do Modelfile experimental entrega ao modelo.

Mesmo método do ``finetune/check_template.py``: a mesma conversa vai pelo ``/api/chat`` do
modelo experimental (o Ollama aplica o template) e pelo ``/api/generate`` com ``raw: true`` e
o texto renderizado em Python com as ferramentas em JSON do Go (``json.Marshal``: compacto,
campos na ordem da struct, ``properties`` em ordem alfabética, campos vazios omitidos). Se o
número de tokens de prompt e a saída (temperatura 0, seed fixa) baterem, o template entrega
exatamente esse JSON. Para comparação, mostra também os tokens do template oficial.

    ollama create jarvis-qwen-json-tools -f finetune/experimentos/Modelfile.json-tools
    .venv/Scripts/python.exe finetune/experimentos/check_json_tools.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import httpx  # noqa: E402

from finetune import chat_format, check_template  # noqa: E402


def json_function(function: Dict[str, Any]) -> str:
    """``json.Marshal`` da ``api.ToolFunction`` do Ollama (palpite conferido pelos tokens)."""
    params = function.get("parameters") or {}
    properties = params.get("properties") or {}
    out_params: Dict[str, Any] = {"type": params.get("type", "")}
    for key in ("$defs", "items"):
        if params.get(key):
            out_params[key] = params[key]
    if params.get("required"):  # omitempty: 2 das 25 ferramentas não têm obrigatórios
        out_params["required"] = list(params["required"])
    out_params["properties"] = {
        name: chat_format._property(properties[name]) for name in sorted(properties)
    }
    return chat_format.go_json({
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "parameters": out_params,
    })


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="jarvis-qwen-json-tools")
    parser.add_argument("--base", default="qwen2.5:3b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args(argv)

    system, tools = check_template._real_system_and_tools()
    options = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 60}
    url = args.ollama_url.rstrip("/")
    conversations = check_template._conversations(system, tools)[:2]
    failures = 0
    with httpx.Client(timeout=600.0) as client:

        def unload(model: str) -> None:
            client.post(url + "/api/generate", json={"model": model, "keep_alive": 0})

        def chat(model: str, conv: Dict[str, Any]) -> Dict[str, Any]:
            unload(model)
            return client.post(url + "/api/chat", json={
                "model": model, "stream": False, "options": options,
                "messages": conv["messages"], "tools": conv["tools"] or None,
            }).json()

        for conv in conversations:
            base = chat(args.base, conv)
            unload(args.base)
            experimental = chat(args.model, conv)
            original = chat_format.render_function
            chat_format.render_function = json_function  # render_prompt usa o nome do módulo
            try:
                prompt = chat_format.render_prompt(conv["messages"], conv["tools"])
            finally:
                chat_format.render_function = original
            unload(args.model)
            raw = client.post(url + "/api/generate", json={
                "model": args.model, "stream": False, "raw": True, "options": options,
                "prompt": prompt,
            }).json()
            for data in (base, experimental, raw):
                if "error" in data:
                    print("ERRO do Ollama:", data["error"])
                    return 2
            same_tokens = experimental.get("prompt_eval_count") == raw.get("prompt_eval_count")
            out_chat = check_template._normalize_chat(experimental.get("message") or {})
            out_raw = check_template._normalize_text(raw.get("response") or "")
            ok = same_tokens and out_chat == out_raw
            failures += 0 if ok else 1
            print("[{}] {:<26} tokens: oficial (struct do Go) {} · json no template {} · "
                  "renderizado em Python {} · saída igual: {}".format(
                      "ok " if ok else "ERR", conv["name"], base.get("prompt_eval_count"),
                      experimental.get("prompt_eval_count"), raw.get("prompt_eval_count"),
                      "sim" if out_chat == out_raw else "NÃO"))
        unload(args.model)
    print("\n{}".format("O template entrega o JSON esperado." if not failures
                        else "{} divergência(s).".format(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
