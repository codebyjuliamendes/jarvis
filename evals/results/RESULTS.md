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
