# Seleção de ferramentas com repetições

Gerado por `evals/compare_models.py --repeats 3` em 2026-11-08T05:07:46-03:00 (commit `faf0410-dirty`, Ollama 0.40.14).
Máquina: Windows-10-10.0.26100-SP0 · GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB. Um modelo por vez, seed 42 igual em todas as rodadas (o que varia é o não determinismo da GPU), temperatura 0,2, `num_ctx` 8192.

Tabela refeita com `--from-json` a partir dos JSON gravados (sem chamar o Ollama); commit e data de cada rodada estão no `meta` de cada arquivo.

Dataset `evals/data/tool_selection.jsonl` (78 casos, sha256 `a0afd6d2936e`), 25 ferramentas reais.

IC 95% principal: bootstrap percentil sobre os casos (10000 reamostragens, seed 20261108; cada caso entra com a fração de rodadas em que acertou). Wilson: conferência, sobre a média de acertos com n = nº de casos. Desvio: amostral, entre as rodadas. Ver `evals/stats.py`.

| Modelo | Modo | Acertos por rodada | Média | Desvio entre rodadas | IC 95% bootstrap | IC 95% Wilson | Casos instáveis | Args (média ± desvio) |
|---|---|---|---|---|---|---|---|---|
| `qwen2.5:1.5b` | native | 43 / 42 / 42 (de 78) | 54,3% | 0,7 p.p. | [44,0; 64,5%] | [43,3; 64,9%] | 12 | 41,0% ± 1,6 p.p. |
| `qwen2.5:1.5b` | json | 45 / 38 / 41 (de 78) | 53,0% | 4,5 p.p. | [42,3; 63,2%] | [42,0; 63,7%] | 10 | 32,8% ± 4,3 p.p. |
| `qwen2.5:3b` | native | 67 / 67 / 66 (de 78) | 85,5% | 0,7 p.p. | [77,8; 92,3%] | [76,0; 91,6%] | 5 | 84,7% ± 0,9 p.p. |
| `qwen2.5:3b` | json | 42 / 37 / 41 (de 78) | 51,3% | 3,4 p.p. | [40,6; 61,5%] | [40,4; 62,1%] | 10 | 45,9% ± 4,3 p.p. |
| `llama3.2:3b` | native | 56 / 58 / 58 (de 78) | 73,5% | 1,5 p.p. | [63,7; 82,5%] | [62,8; 82,0%] | 6 | 62,8% ± 0,9 p.p. |

## Comparações pareadas (mesmos casos)

McNemar exato bilateral sobre o acerto por maioria das rodadas; "só A / só B" são os casos discordantes. Também rodada a rodada (rodada i × rodada i), como sensibilidade. Diferença = média por caso da fração de acertos de A menos a de B, com IC bootstrap pareado. Com 4 comparações, o limiar corrigido por Bonferroni é p < 0,0125.

| Comparação | Modo | Só A / só B | p (maioria) | p por rodada | Diferença A − B | IC 95% da diferença | Significativa? |
|---|---|---|---|---|---|---|---|
| `qwen2.5:1.5b` × `qwen2.5:3b` | json | 15 / 15 | 1,000 | 0,711, 1,000, 1,000 | +1,7 p.p. | [-11,1; +14,1] p.p. | não |
| `qwen2.5:1.5b` × `qwen2.5:3b` | native | 4 / 30 | < 0,001 | < 0,001, < 0,001, < 0,001 | -31,2 p.p. | [-43,2; -18,8] p.p. | sim |
| `qwen2.5:1.5b` × `llama3.2:3b` | native | 10 / 27 | 0,008 | 0,041, 0,011, 0,011 | -19,2 p.p. | [-32,9; -6,0] p.p. | sim |
| `qwen2.5:3b` × `llama3.2:3b` | native | 13 / 4 | 0,049 | 0,019, 0,064, 0,096 | +12,0 p.p. | [+2,1; +22,2] p.p. | não |

Arquivos brutos (cada rodada, cada caso): `20261108T035921_tools_qwen2.5-1.5b_native_x3.json`, `20261108T043725_tools_qwen2.5-1.5b_json_x3.json`, `20261108T045503_tools_qwen2.5-3b_native_x3.json`, `20261108T045827_tools_qwen2.5-3b_json_x3.json`, `20261108T050159_tools_llama3.2-3b_native_x3.json`
