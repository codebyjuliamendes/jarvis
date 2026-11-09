# Resultados das avaliações

Gerado por `evals/compare_models.py` em 2026-11-08T03:10:49-03:00 (commit `0fa0ad2`, Ollama 0.40.14).
Máquina: Windows-10-10.0.26200-SP0 · GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB. Um modelo por vez, seed 42, `num_ctx` 8192.

## Seleção de ferramentas

Dataset `evals/data/tool_selection.jsonl` (78 casos, sha256 `a0afd6d2936e`), 25 ferramentas reais.

| Modelo | Modo | Acurácia ferramenta | Acurácia args | JSON inválido | Latência p50 (s) | Latência p95 (s) | tokens/s |
|---|---|---|---|---|---|---|---|
| `qwen2.5:1.5b` | native | 56,4% (44/78) | 44,3% (27/61) | 0,0% (0/78) | 0,40 | 0,93 | 118,7 |
| `qwen2.5:1.5b` | json | 47,4% (37/78) | 32,8% (20/61) | 9,0% (7/78) | 0,40 | 1,04 | 115,4 |
| `qwen2.5:3b` | native | 82,1% (64/78) | 80,3% (49/61) | 0,0% (0/78) | 0,58 | 1,12 | 69,9 |
| `qwen2.5:3b` | json | 47,4% (37/78) | 39,3% (24/61) | 5,1% (4/78) | 0,69 | 1,81 | 66,9 |
| `llama3.2:3b` | native | 74,4% (58/78) | 65,6% (40/61) | 0,0% (0/78) | 0,77 | 1,29 | 65,2 |
| `llama3.2:3b` | json | 53,8% (42/78) | 39,3% (24/61) | 20,5% (16/78) | 0,81 | 1,56 | 61,3 |

### Por categoria

| Modelo | Modo | Direto | Ambíguo | Sem ferramenta | Args (dada a ferramenta certa) | Chamadas vazadas no texto | Ferramenta inexistente |
|---|---|---|---|---|---|---|---|
| `qwen2.5:1.5b` | native | 53,2% (33/62) | 37,5% (3/8) | 100,0% (8/8) | 84,4% (27/32) | 4 | 0 |
| `qwen2.5:1.5b` | json | 45,2% (28/62) | 37,5% (3/8) | 75,0% (6/8) | 74,1% (20/27) | 0 | 1 |
| `qwen2.5:3b` | native | 85,5% (53/62) | 75,0% (6/8) | 62,5% (5/8) | 94,2% (49/52) | 0 | 0 |
| `qwen2.5:3b` | json | 43,5% (27/62) | 25,0% (2/8) | 100,0% (8/8) | 92,3% (24/26) | 0 | 0 |
| `llama3.2:3b` | native | 83,9% (52/62) | 75,0% (6/8) | 0,0% (0/8) | 78,4% (40/51) | 4 | 0 |
| `llama3.2:3b` | json | 53,2% (33/62) | 62,5% (5/8) | 50,0% (4/8) | 75,0% (24/32) | 0 | 5 |

## Recuperação (RAG)

Dataset `evals/data/rag_retrieval.jsonl` (36 perguntas, sha256 `63a1ac64abdb`). O resultado depende só do modelo de embeddings, não do modelo de chat.

| Embeddings | Corpus | recall@1 | recall@5 | MRR | Latência da busca p50 (s) |
|---|---|---|---|---|---|
| `nomic-embed-text` · bm25 | 13 docs / 25 trechos | 83,3% (30/36) | 100,0% (36/36) | 0,912 | 0,001 |

Arquivos brutos (com cada caso): `20261108T030020_tools_qwen2.5-1.5b_native.json`, `20261108T030601_tools_qwen2.5-1.5b_json.json`, `20261108T030706_tools_qwen2.5-3b_native.json`, `20261108T030816_tools_qwen2.5-3b_json.json`, `20261108T030929_tools_llama3.2-3b_native.json`, `20261108T031044_tools_llama3.2-3b_json.json`, `20261108T031049_rag_nomic-embed-text_bm25.json`

## Comparação com a rodada anterior (subprojeto 1, commit `f66b0f7`)

Esta tabela foi refeita no subprojeto 5b porque o system prompt mudou no subprojeto 4 (uma
linha pedindo para citar as fontes do `knowledge_search`) e a anterior não correspondia mais ao
código. Mesmos dataset, seed, temperatura (0,2) e `num_ctx`. A máquina recebeu uma
atualização do Windows entre as duas rodadas (build 26100 → 26200 no metadado).

| Modelo | Modo | Ferramenta antes → agora | Args antes → agora | Previsões que mudaram |
|---|---|---|---|---|
| `qwen2.5:1.5b` | native | 44 → 44 | 26 → 27 | 4 (2 ganhos, 2 perdas) |
| `qwen2.5:1.5b` | json | 37 → 37 | 16 → 20 | 13 |
| `qwen2.5:3b` | native | 66 → 64 | 50 → 49 | 6 (1 ganho, 3 perdas) contra a rodada de `014925` |
| `qwen2.5:3b` | json | 40 → 37 | 26 → 24 | 10 |
| `llama3.2:3b` | native | 58 → 58 | 35 → 40 | 5 (2 ganhos, 2 perdas) |
| `llama3.2:3b` | json | 38 → 42 | 15 → 24 | 17; JSON inválido 24 → 16 |

Como ler:

- **No modo nativo, as diferenças estão dentro do ruído de uma rodada.** O `qwen2.5:3b` já
  tinha sido refeito com o prompt novo (`20261108T014925_tools_qwen2.5-3b_native.json`, 66/78);
  agora, com o mesmo prompt e o mesmo código, deu 64/78. Ou seja: 2 casos de diferença sem
  nenhuma mudança relevante, o que dá a medida do não determinismo (temperatura 0,2 com seed
  fixo não garante a mesma saída na GPU). As perdas foram `ts-019` (commit → sem ferramenta),
  `ts-032` (Google Agenda → `reminder`) e `ts-076` (ambíguo: `web_search` → `knowledge_search`).
- **A maior mudança é o `llama3.2:3b` em argumentos** (+5 casos no nativo, +9 no JSON) e o
  JSON inválido dele (24 → 16; com caminho do Windows sem escape, 8 → 3). O
  `llama3.2:3b` nativo, por outro lado, passou a chamar ferramenta em **todas** as 8 conversas
  sem ferramenta (antes acertava 1). Não dá para atribuir isso à linha nova do prompt com uma
  rodada só; o que é robusto é a ordem entre os modelos, que não mudou.
- **O padrão continua sendo `qwen2.5:3b` nativo**: melhor em ferramenta e em argumentos nas
  duas rodadas.
- **RAG**: a linha agora é o padrão do bot (`bm25`), com os mesmos números do
  [`RAG.md`](RAG.md) (83,3% / 100,0% / 0,912). A linha antiga (vetorial sem prefixos, 44,4% /
  72,2% / 0,544) continua no `RAG.md` como "antes".

Arquivos da rodada anterior (mantidos para auditoria): `20261108T005533_tools_qwen2.5-1.5b_native.json`,
`20261108T010114_tools_qwen2.5-1.5b_json.json`, `20261108T010229_tools_qwen2.5-3b_native.json`,
`20261108T010336_tools_qwen2.5-3b_json.json`, `20261108T010500_tools_llama3.2-3b_native.json`,
`20261108T010614_tools_llama3.2-3b_json.json`, `20261108T014925_tools_qwen2.5-3b_native.json`,
`20261108T010625_compare.json`.

## Três rodadas por modelo: variância, IC 95% e teste pareado (extras)

Feito depois da tabela acima, com `compare_models.py --repeats 3` (mesmo dataset, seed 42 em
todas as rodadas, temperatura 0,2, `num_ctx` 8192; o código da avaliação de ferramentas é o
mesmo do `0fa0ad2`). Tabela gerada, com o p de cada rodada: [`REPEATS.md`](REPEATS.md); cada
rodada e cada caso nos `*_x3.json`; os runs no MLflow (`tools <modelo> <modo> x3` e o pai
`compare x3`).

| Modelo | Modo | Acertos por rodada | Média | Desvio entre rodadas | IC 95% (bootstrap sobre os casos) | IC 95% (Wilson) |
|---|---|---|---|---|---|---|
| `qwen2.5:1.5b` | native | 43 / 42 / 42 | 54,3% | 0,7 p.p. | [44,0; 64,5%] | [43,3; 64,9%] |
| **`qwen2.5:3b`** | **native** | **67 / 67 / 66** | **85,5%** | **0,7 p.p.** | **[77,8; 92,3%]** | [76,0; 91,6%] |
| `llama3.2:3b` | native | 56 / 58 / 58 | 73,5% | 1,5 p.p. | [63,7; 82,5%] | [62,8; 82,0%] |
| `qwen2.5:1.5b` | json | 45 / 38 / 41 | 53,0% | 4,5 p.p. | [42,3; 63,2%] | [42,0; 63,7%] |
| `qwen2.5:3b` | json | 42 / 37 / 41 | 51,3% | 3,4 p.p. | [40,6; 61,5%] | [40,4; 62,1%] |
| `llama3.2:3b` | json | 44 / 43 / 45 | 56,4% | 1,3 p.p. | [46,2; 67,1%] | [45,4; 66,9%] |

Comparação pareada no modo nativo (McNemar exato bilateral sobre os casos discordantes, com o
acerto de cada caso pela maioria das 3 rodadas; 6 comparações no `REPEATS.md` → limiar de
Bonferroni p < 0,0083; eram 4 → 0,0125 antes da bateria do `llama3.2:3b` JSON):

| Comparação | Só A acerta / só B acerta | p (maioria) | p por rodada | Diferença A − B (IC 95% bootstrap pareado) |
|---|---|---|---|---|
| `qwen2.5:3b` × `llama3.2:3b` | 13 / 4 | **0,049** | 0,019 · 0,064 · 0,096 | +12,0 p.p. [+2,1; +22,2] |
| `qwen2.5:1.5b` × `qwen2.5:3b` | 4 / 30 | < 0,001 | < 0,001 nas 3 | −31,2 p.p. [−43,2; −18,8] |
| `qwen2.5:1.5b` × `llama3.2:3b` | 10 / 27 | 0,008 | 0,041 · 0,011 · 0,011 | −19,2 p.p. [−32,9; −6,0] |
| `qwen2.5:1.5b` × `qwen2.5:3b` (json) | 15 / 15 | 1,000 | 0,711 · 1,000 · 1,000 | +1,7 p.p. [−11,1; +14,1] |

Como ler:

- **A variância entre rodadas é pequena perto da incerteza da amostra.** No nativo, rodadas
  idênticas variam 1–2 casos (desvio de 0,7–1,5 ponto); o IC 95% sobre os 78 casos tem ~15
  pontos de largura. O número que limita a conclusão é o tamanho do dataset, não o não
  determinismo da GPU. No modo JSON a variação entre rodadas é maior (3,4–4,5 pontos).
- **`qwen2.5:3b` > `llama3.2:3b`: evidência moderada, não conclusiva.** A diferença de 12
  pontos tem IC bootstrap que não cruza o zero e McNemar p = 0,049 na maioria das rodadas, mas
  rodada a rodada o p vai de 0,019 a 0,096, e com a correção de Bonferroni (0,0125 para 4 comparações, 0,0083 para 6) não é
  significativa. A escolha do `qwen2.5:3b` continua defensável (melhor nas 3 rodadas, em
  argumentos e em latência), mas "melhor com significância estatística" não é o que os dados
  dizem.
- **O `qwen2.5:1.5b` perde dos dois 3B com folga** (p < 0,001 e p = 0,008, ambos abaixo do
  limiar corrigido).
- **No modo JSON o tamanho do modelo não importa** (15 × 15 casos discordantes entre os qwen;
  16 × 17 entre `qwen2.5:3b` e `llama3.2:3b`, p = 1,0): o gargalo é o formato, não o modelo.
  Os três ficam entre 51% e 57% de média.
- **A média do `qwen2.5:3b` nativo subiu de 82,1% (64/78) para 85,5%** sem mudança no código
  da avaliação: as rodadas desta noite deram 67, 67 e 66; as anteriores, 66 e 64. Com as 5
  rodadas juntas, 66,0 ± 1,2 casos (84,6%). É a mesma variância medida acima, e é por isso que
  a tabela principal deve ser lida com o IC ao lado.

Ressalvas desta rodada:

- **O `llama3.2:3b` no modo JSON foi refeito em 08/11 à noite** (a 1ª tentativa foi
  interrompida por falta de memória): 44 / 43 / 45, 56,4%. O `REPEATS.md` foi refeito com
  `compare_models.py --repeats 3 --from-json` com as 6 baterias.
- **Timeouts do `qwen2.5:1.5b`.** Em 2, 3 e 4 casos das rodadas nativas (e 0, 2 e 1 no JSON) o
  modelo não parou de gerar em 300 s (o eval não limita `num_predict`, como o agente). Contam
  como erro de ferramenta. Na rodada anterior tinha sido 1 caso. Não muda nenhuma conclusão
  (o 1,5B já era o pior), mas deixa a rodada lenta (~12 min em vez de ~1 min).
- O metadado `git_commit` dos `*_x3.json` diz `faf0410-dirty`: a árvore tinha mudanças em
  arquivos de documentação e do RAG enquanto rodava; `evals/eval_tools.py` e `jarvis/` estavam
  iguais ao commit `7efaa66`.

## Fine-tuning (08/11/2026)

O `jarvis-qwen-lora` (Qwen2.5-3B + LoRA treinado localmente, GGUF Q8_0) fez 78/78 nas 3
rodadas (100%; Wilson [95,3; 100]) contra 65/65/65 (83,3%) da mesma base convertida do mesmo
jeito; McNemar 0 / 12, p < 0,001. Os modelos de frase do treino foram escritos por quem tinha
lido este eval, então o número mede o eval, não generalização. Tabela, método e ressalvas em
[`FINETUNE.md`](FINETUNE.md).

