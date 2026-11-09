"""Converte o modelo juntado (safetensors) em GGUF F16 que o Ollama importa.

Por que existe: o importador de safetensors do Ollama 0.40.14 só aceita
arquiteturas MLX e recusa ``Qwen2ForCausalLM``. Em vez de depender do
conversor do llama.cpp, este script reaproveita os METADADOS do GGUF oficial do
``qwen2.5`` que já está no Ollama (arquitetura, hiperparâmetros e o tokenizador
inteiro: vocabulário, merges, tokens especiais), e só troca os pesos pelos do
modelo ajustado. Assim o tokenizador é idêntico ao do modelo base por
construção.

    .venv-train/Scripts/python.exe finetune/to_gguf.py --base-gguf <blob do qwen2.5:3b>
    ollama create jarvis-qwen-lora -f finetune/Modelfile.local

Quantização: o Ollama 0.40.14 só quantiza na importação de safetensors (MLX) e o
gguf-py não implementa q4_K_M; por isso o padrão é Q8_0 (quase sem perda). Para
comparar de forma justa, o modelo BASE passa pelo mesmo caminho (``--merged``
apontando para os safetensors do Qwen2.5-3B-Instruct original).

O caminho do blob sai de ``ollama show qwen2.5:3b --modelfile`` (linha FROM).
Mapeamento de nomes: o mesmo do conversor do llama.cpp para ``qwen2`` (sem
permutação de q/k, que é coisa do ``llama``).
"""
from __future__ import annotations

import argparse
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

_LAYER = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}
_SKIP_KEYS = {"general.file_type", "general.quantization_version"}


def gguf_name(hf_name: str) -> str | None:
    if hf_name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if hf_name == "model.norm.weight":
        return "output_norm.weight"
    if hf_name == "lm_head.weight":
        return None  # embeddings amarradas no Qwen2.5-3B: o GGUF oficial não tem output.weight
    m = re.match(r"model\.layers\.(\d+)\.(.+)$", hf_name)
    if m and m.group(2) in _LAYER:
        return "blk.{}.{}".format(m.group(1), _LAYER[m.group(2)])
    raise KeyError("tensor sem mapeamento: " + hf_name)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-gguf", required=True, help="GGUF oficial do qwen2.5 (blob do Ollama)")
    p.add_argument("--merged", default=os.path.join(HERE, "out", "merged"))
    p.add_argument("--out", default=os.path.join(HERE, "out", "jarvis-qwen-lora-q8_0.gguf"))
    p.add_argument("--quant", choices=["f16", "q8_0"], default="q8_0",
                   help="q8_0: matrizes em Q8_0 (o gguf-py não faz q4_K_M)")
    args = p.parse_args(argv)

    import numpy as np
    from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter, LlamaFileType, quants
    from safetensors import safe_open

    reader = GGUFReader(args.base_gguf)
    arch = reader.fields["general.architecture"].contents()
    writer = GGUFWriter(args.out, arch)
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key in _SKIP_KEYS or key == "general.architecture":
            continue
        vtype = GGUFValueType(field.types[0])
        sub = GGUFValueType(field.types[-1]) if vtype == GGUFValueType.ARRAY else None
        writer.add_key_value(key, field.contents(), vtype, sub_type=sub)
    writer.add_file_type(LlamaFileType.MOSTLY_Q8_0 if args.quant == "q8_0" else LlamaFileType.MOSTLY_F16)
    writer.add_quantization_version(2)

    expected = {t.name for t in reader.tensors}
    written = set()
    import torch
    files = [f for f in os.listdir(args.merged) if f.endswith(".safetensors")]
    for fname in sorted(files):
        with safe_open(os.path.join(args.merged, fname), framework="pt") as st:
            for hf_name in st.keys():
                name = gguf_name(hf_name)
                if name is None:
                    continue
                t = st.get_tensor(hf_name)
                # normas e bias em F32 (como no GGUF oficial); matrizes em F16 ou Q8_0
                if t.ndim == 1:
                    writer.add_tensor(name, t.to(torch.float32).numpy())
                elif args.quant == "q8_0":
                    data = t.to(torch.float32).numpy()
                    q = quants.quantize(data, GGMLQuantizationType.Q8_0)
                    writer.add_tensor(name, q, raw_shape=q.shape, raw_dtype=GGMLQuantizationType.Q8_0)
                else:
                    writer.add_tensor(name, t.to(torch.float16).numpy().astype(np.float16))
                written.add(name)
    missing, extra = expected - written, written - expected
    if missing or extra:
        raise SystemExit("tensores diferentes do GGUF base: faltam {} sobram {}".format(
            sorted(missing)[:5], sorted(extra)[:5]))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print("ok:", args.out, len(written), "tensores")


if __name__ == "__main__":
    main()
