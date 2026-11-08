# Seleção de ferramentas com repetições

Gerado por `evals/compare_models.py --repeats 3` em 2026-11-08T05:18:27-03:00 (commit `972cf56`, Ollama 0.40.14).
Máquina: Windows-10-10.0.26100-SP0 · GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB. Um modelo por vez, seed 42 igual em todas as rodadas (o que varia é o não determinismo da GPU), temperatura 0,2, `num_ctx` 8192.

Tabela refeita com `--from-json` a partir dos JSON gravados (sem chamar o Ollama); commit e data de cada rodada estão no `meta` de cada arquivo.

Dataset `evals/data/tool_selection.jsonl` (78 casos, sha256 `a0afd6d2936e`), 25 ferramentas reais.

IC 95% principal: bootstrap percentil sobre os casos (10000 reamostragens, seed 20261108; cada caso entra com a fração de rodadas em que acertou). Wilson: conferência, sobre a média de acertos com n = nº de casos. Desvio: amostral, entre as rodadas. Ver `evals/stats.py`.

| Modelo | Modo | Acertos por rodada | Média | Desvio entre rodadas | IC 95% bootstrap | IC 95% Wilson | Casos instáveis | Args (média ± desvio) |
|---|---|---|---|---|---|---|---|---|
| `qwen2.5:3b` | native | 67 / 67 / 66 (de 78) | 85,5% | 0,7 p.p. | [77,8; 92,3%] | [76,0; 91,6%] | 5 | 84,7% ± 0,9 p.p. |
| `jarvis-qwen-json-tools` | native | 66 / 67 / 68 (de 78) | 85,9% | 1,3 p.p. | [78,2; 92,7%] | [76,5; 91,9%] | 4 | 82,5% ± 2,5 p.p. |

## Comparações pareadas (mesmos casos)

McNemar exato bilateral sobre o acerto por maioria das rodadas; "só A / só B" são os casos discordantes. Também rodada a rodada (rodada i × rodada i), como sensibilidade. Diferença = média por caso da fração de acertos de A menos a de B, com IC bootstrap pareado. Com 1 comparações, o limiar corrigido por Bonferroni é p < 0,0500.

| Comparação | Modo | Só A / só B | p (maioria) | p por rodada | Diferença A − B | IC 95% da diferença | Significativa? |
|---|---|---|---|---|---|---|---|
| `qwen2.5:3b` × `jarvis-qwen-json-tools` | native | 3 / 3 | 1,000 | 1,000, 1,000, 0,727 | -0,4 p.p. | [-6,4; +5,1] p.p. | não |

Arquivos brutos (cada rodada, cada caso): `20261108T045503_tools_qwen2.5-3b_native_x3.json`, `20261108T051555_tools_jarvis-qwen-json-tools_native_x3.json`
