# Fine-tuning LoRA: antes e depois

**Leia antes dos números.** O ajustado acertou 78 de 78 nas 3 rodadas. Isso mede o eval de
78 casos, não generalização: os modelos de frase do dataset de treino foram escritos por quem
tinha lido o eval. Os filtros de vazamento tiram cópia e paráfrase próxima (Jaccard de
4-gramas ≥ 0,35), mas não a política ("anotações -> `knowledge_search`", "agenda do Google ->
`google`"), que coincide com os gabaritos. O teste honesto é um conjunto novo, escrito por outra
pessoa ou tirado de pedidos reais do Telegram; ele não existe ainda
(ver `evals/PENDENCIAS.md`, seção Fine-tuning).

Como foi treinado (local, `finetune/train_local.py`): Qwen2.5-3B-Instruct em 4 bits (QLoRA
nf4), LoRA r=16/alpha=32 nas 7 projeções, lr 2e-4, lote efetivo 16, 101 de 106 passos de 1
época (parou no limite de 210 min; 219,6 min com as avaliações), RTX 4050 Laptop de 6 GB,
pico de 5,08 GB de VRAM. Perda no val (40 exemplos de modelos de frase que o treino não viu):
0,917 antes, 0,099 depois. Gradiente truncado: o prompt (~3.900 tokens) passa sem gradiente
e só a resposta tem gradiente (ver o docstring do script). Relatório:
`20261108T213400_finetune_train_report.json`; curvas no MLflow (`jarvis-finetune-lora`).

Comparação justa: o Ollama 0.40.14 não importa Qwen2 em safetensors nem quantiza GGUF, então
o ajustado virou GGUF **Q8_0** com `finetune/to_gguf.py`. A **base** passou pelo mesmo caminho
(`qwen2.5-3b-base-q8`), e ela empata com o `qwen2.5:3b` oficial (q4_K_M): 5 / 6 casos
discordantes, p = 1,0. Isso confere a conversão (tokenizador e pesos) e isola o efeito do LoRA.

## Tabela gerada

Gerado por `evals/compare_models.py --repeats 3` em 2026-11-08T21:49:16-03:00 (commit `441c949`, Ollama 0.40.14).
Máquina: Windows-10-10.0.26100-SP0 · GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB. Um modelo por vez, seed 42 igual em todas as rodadas (o que varia é o não determinismo da GPU), temperatura 0,2, `num_ctx` 8192.

Tabela refeita com `--from-json` a partir dos JSON gravados (sem chamar o Ollama); commit e data de cada rodada estão no `meta` de cada arquivo.

Dataset `evals/data/tool_selection.jsonl` (78 casos, sha256 `a0afd6d2936e`), 25 ferramentas reais.

IC 95% principal: bootstrap percentil sobre os casos (10000 reamostragens, seed 20261108; cada caso entra com a fração de rodadas em que acertou). Wilson: conferência, sobre a média de acertos com n = nº de casos. Desvio: amostral, entre as rodadas. Ver `evals/stats.py`.

| Modelo | Modo | Acertos por rodada | Média | Desvio entre rodadas | IC 95% bootstrap | IC 95% Wilson | Casos instáveis | Args (média ± desvio) |
|---|---|---|---|---|---|---|---|---|
| `qwen2.5-3b-base-q8` | native | 65 / 65 / 65 (de 78) | 83,3% | 0,0 p.p. | [75,2; 91,0%] | [73,5; 90,0%] | 5 | 81,4% ± 0,9 p.p. |
| `jarvis-qwen-lora` | native | 78 / 78 / 78 (de 78) | 100,0% | 0,0 p.p. | [100,0; 100,0%] | [95,3; 100,0%] | 0 | 96,7% ± 0,0 p.p. |
| `qwen2.5:3b` | native | 67 / 67 / 66 (de 78) | 85,5% | 0,7 p.p. | [77,8; 92,3%] | [76,0; 91,6%] | 5 | 84,7% ± 0,9 p.p. |

## Comparações pareadas (mesmos casos)

McNemar exato bilateral sobre o acerto por maioria das rodadas; "só A / só B" são os casos discordantes. Também rodada a rodada (rodada i × rodada i), como sensibilidade. Diferença = média por caso da fração de acertos de A menos a de B, com IC bootstrap pareado. Com 3 comparações, o limiar corrigido por Bonferroni é p < 0,0167.

| Comparação | Modo | Só A / só B | p (maioria) | p por rodada | Diferença A − B | IC 95% da diferença | Significativa? |
|---|---|---|---|---|---|---|---|
| `qwen2.5-3b-base-q8` × `jarvis-qwen-lora` | native | 0 / 12 | < 0,001 | < 0,001, < 0,001, < 0,001 | -16,7 p.p. | [-24,8; -9,0] p.p. | sim |
| `qwen2.5-3b-base-q8` × `qwen2.5:3b` | native | 5 / 6 | 1,000 | 0,774, 0,774, 1,000 | -2,1 p.p. | [-9,8; +5,1] p.p. | não |
| `jarvis-qwen-lora` × `qwen2.5:3b` | native | 11 / 0 | < 0,001 | < 0,001, < 0,001, < 0,001 | +14,5 p.p. | [+7,7; +22,2] p.p. | sim |

Arquivos brutos (cada rodada, cada caso): `20261108T214245_tools_qwen2.5-3b-base-q8_native_x3.json`, `20261108T214647_tools_jarvis-qwen-lora_native_x3.json`, `20261108T045503_tools_qwen2.5-3b_native_x3.json`
