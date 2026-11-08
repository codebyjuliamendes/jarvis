"""Treino QLoRA local (GPU de 6 GB) do Qwen2.5-Instruct no dataset do Jarvis.

Versão local do ``train_lora.ipynb`` (que usa Unsloth no Colab). Mesmos dados,
mesmo renderizador (``chat_format.render_example``) e mesma perda: só sobre a
resposta do assistente.

Como cabe em 6 GB: cada exemplo tem ~4.000 tokens de prompt (system + 25
ferramentas) e só ~30 de resposta. O backward completo sobre 4.000 tokens passou
de 7 GB e levou 82 s por exemplo nesta GPU (transbordando para a RAM). Aqui o
prompt passa pelo modelo SEM gradiente e vira cache de chaves/valores; só a
resposta passa com gradiente, atendendo a esse cache.

Isso é um gradiente truncado: o LoRA muda o prompt no forward (a perda medida é
exata), mas o gradiente não volta pelas representações do prompt, só pelas da
resposta. É a aproximação que torna o treino viável em 6 GB; está documentada no
finetune/README.md.

Uso (venv de treino com torch CUDA, ver finetune/README.md):

    .venv-train/Scripts/python.exe finetune/train_local.py --model Qwen/Qwen2.5-3B-Instruct

Requer Python 3.10+ (torch/transformers atuais); não roda no venv 3.8 do bot.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from chat_format import render_example  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--out", default=os.path.join(HERE, "out", "jarvis-qwen-lora-adapter"))
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--val-sample", type=int, default=80, help="exemplos do val usados na perda")
    p.add_argument("--eval-every", type=int, default=25, help="passos de otimização")
    p.add_argument("--max-minutes", type=float, default=0, help="para antes (0 = sem limite)")
    p.add_argument("--max-train", type=int, default=0, help="limita exemplos (0 = todos)")
    p.add_argument("--seed", type=int, default=42)
    db = os.path.join(ROOT, "evals", "mlflow.db").replace("\\", "/")
    p.add_argument("--mlflow-uri", default="sqlite:///" + db)
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tools = json.load(open(os.path.join(HERE, "data", "tools.json"), encoding="utf-8"))
    train_rows = load_jsonl(os.path.join(HERE, "data", "train.jsonl"))
    val_rows = load_jsonl(os.path.join(HERE, "data", "val.jsonl"))
    random.shuffle(train_rows)
    if args.max_train:
        train_rows = train_rows[: args.max_train]
    val_rows = random.Random(0).sample(val_rows, min(args.val_sample, len(val_rows)))

    tok = AutoTokenizer.from_pretrained(args.model)
    for special in ("<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"):
        assert len(tok(special, add_special_tokens=False)["input_ids"]) == 1, special

    def encode(row):
        prompt, completion = render_example(row, tools)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
        return p_ids + c_ids, len(c_ids)

    train = [encode(r) for r in train_rows]
    val = [encode(r) for r in val_rows]
    lengths = sorted(len(x[0]) for x in train)
    print("treino {} | val {} | tokens p50 {} max {}".format(
        len(train), len(val), lengths[len(lengths) // 2], lengths[-1]), flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")
    # Sem prepare_model_for_kbit_training: ele converte embeddings e normas para fp32
    # (o embedding do Qwen tem 152 mil x 2048 e passaria de 1,2 GB), o que estoura 6 GB.
    # Sem gradient checkpointing: só os ~30 tokens da resposta guardam ativações.
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def loss_of(ids, n_comp):
        x = torch.tensor([ids], device="cuda")
        # 1) prompt (menos o último token) sem gradiente -> cache de chaves/valores
        with torch.no_grad():
            cache = model(input_ids=x[:, : -n_comp - 1], use_cache=True,
                          logits_to_keep=1).past_key_values  # sem logits do prompt inteiro
        # 2) último token do prompt + resposta com gradiente, atendendo ao cache.
        #    A posição i prevê o token i+1; a última posição (depois do <|im_end|>) sai.
        out = model(input_ids=x[:, -n_comp - 1:], past_key_values=cache, use_cache=True)
        logits = out.logits[0, :-1].float()
        target = x[0, -n_comp:]
        return torch.nn.functional.cross_entropy(logits, target)

    @torch.no_grad()
    def val_loss():
        model.eval()
        total = sum(loss_of(ids, n).item() for ids, n in val)
        model.train()
        return total / len(val)

    import bitsandbytes as bnbopt
    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnbopt.optim.PagedAdamW8bit(params, lr=args.lr, weight_decay=0.01)
    total_steps = max(1, math.ceil(len(train) * args.epochs / args.grad_accum))
    warmup = max(1, int(0.05 * total_steps))

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        return args.lr * max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    run = None
    if not args.no_mlflow:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("jarvis-finetune-lora")
        run = mlflow.start_run(run_name="qlora-local-" + args.model.split("/")[-1])
        mlflow.log_params({"model": args.model, "epochs": args.epochs, "lr": args.lr,
                           "grad_accum": args.grad_accum, "rank": args.rank, "alpha": args.alpha,
                           "train_examples": len(train), "val_sample": len(val),
                           "quant": "nf4-4bit", "gpu": torch.cuda.get_device_name(0)})

    history = []
    before = val_loss()
    print("val loss antes: {:.4f}".format(before), flush=True)
    history.append({"step": 0, "val_loss": before})
    if run:
        mlflow.log_metric("val_loss", before, step=0)

    started = time.time()
    model.train()
    step, micro, running, best = 0, 0, [], before
    order = []
    while len(order) < int(len(train) * args.epochs):
        idx = list(range(len(train)))
        random.shuffle(idx)
        order.extend(idx)
    order = order[: int(len(train) * args.epochs)]
    stopped_early = False
    for i in order:
        ids, n = train[i]
        loss = loss_of(ids, n) / args.grad_accum
        loss.backward()
        running.append(loss.item() * args.grad_accum)
        micro += 1
        if micro <= 3:
            print("  exemplo {} ({} tokens): {:.1f} s desde o início | VRAM pico {:.2f} GB".format(
                micro, len(ids), time.time() - started, torch.cuda.max_memory_allocated() / 1e9),
                flush=True)
        if micro % args.grad_accum:
            continue
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tr = sum(running) / len(running)
        running = []
        elapsed = (time.time() - started) / 60
        print("passo {}/{} perda {:.4f} | {:.1f} min | VRAM pico {:.2f} GB".format(
            step, total_steps, tr, elapsed, torch.cuda.max_memory_allocated() / 1e9), flush=True)
        history.append({"step": step, "train_loss": tr})
        if run:
            mlflow.log_metric("train_loss", tr, step=step)
        if step % args.eval_every == 0 or step == total_steps:
            vl = val_loss()
            history.append({"step": step, "val_loss": vl})
            print("  val loss {:.4f}".format(vl), flush=True)
            if run:
                mlflow.log_metric("val_loss", vl, step=step)
            best = min(best, vl)
        if args.max_minutes and elapsed > args.max_minutes:
            stopped_early = True
            print("limite de tempo atingido", flush=True)
            break

    after = val_loss()
    minutes = (time.time() - started) / 60
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    report = {
        "modelo_base": args.model, "quant": "nf4 4-bit (QLoRA)", "lora_rank": args.rank,
        "lora_alpha": args.alpha, "lr": args.lr, "grad_accum": args.grad_accum,
        "epocas": args.epochs, "passos": step, "parou_por_tempo": stopped_early,
        "exemplos_treino": len(train), "amostra_val": len(val),
        "val_loss_antes": round(before, 4), "val_loss_depois": round(after, 4),
        "minutos_treino": round(minutes, 1), "gpu": torch.cuda.get_device_name(0),
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "history": history,
    }
    with open(os.path.join(HERE, "out", "train_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    if run:
        mlflow.log_metrics({"val_loss_final": after, "train_minutes": minutes})
        mlflow.end_run()
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
