"""Junta o adaptador LoRA ao modelo base e salva em safetensors para o Ollama importar.

O merge é feito na CPU, em bf16 (o 3B ocupa ~6 GB de RAM), sem quantização: o
Ollama quantiza depois, no ``ollama create --quantize q4_K_M``.

    .venv-train/Scripts/python.exe finetune/merge_adapter.py
    ollama create jarvis-qwen-lora -f finetune/Modelfile.local --quantize q4_K_M
"""
from __future__ import annotations

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", default=os.path.join(HERE, "out", "jarvis-qwen-lora-adapter"))
    p.add_argument("--out", default=os.path.join(HERE, "out", "merged"))
    args = p.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": "cpu"})
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    print("salvo em", args.out)


if __name__ == "__main__":
    main()
