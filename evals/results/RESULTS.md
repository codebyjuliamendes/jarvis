# Resultados das avaliações

Gerado por `evals/compare_models.py` em 2026-11-08T01:06:25-03:00 (commit `f66b0f7`, Ollama 0.40.14).
Máquina: Windows-10-10.0.26100-SP0 · GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB. Um modelo por vez, seed 42, `num_ctx` 8192.

## Seleção de ferramentas

Dataset `evals/data/tool_selection.jsonl` (78 casos, sha256 `a0afd6d2936e`), 25 ferramentas reais.

| Modelo | Modo | Acurácia ferramenta | Acurácia args | JSON inválido | Latência p50 (s) | Latência p95 (s) | tokens/s |
|---|---|---|---|---|---|---|---|
| `qwen2.5:1.5b` | native | 56,4% (44/78) | 42,6% (26/61) | 0,0% (0/78) | 0,42 | 1,66 | 121,4 |
| `qwen2.5:1.5b` | json | 47,4% (37/78) | 26,2% (16/61) | 7,7% (6/78) | 0,41 | 1,09 | 114,9 |
| `qwen2.5:3b` | native | 84,6% (66/78) | 82,0% (50/61) | 0,0% (0/78) | 0,65 | 1,26 | 64,5 |
| `qwen2.5:3b` | json | 51,3% (40/78) | 42,6% (26/61) | 6,4% (5/78) | 0,64 | 2,02 | 65,5 |
| `llama3.2:3b` | native | 74,4% (58/78) | 57,4% (35/61) | 0,0% (0/78) | 0,72 | 1,32 | 64,0 |
| `llama3.2:3b` | json | 48,7% (38/78) | 24,6% (15/61) | 30,8% (24/78) | 0,87 | 1,56 | 60,3 |

### Por categoria

| Modelo | Modo | Direto | Ambíguo | Sem ferramenta | Args (dada a ferramenta certa) | Chamadas vazadas no texto | Ferramenta inexistente |
|---|---|---|---|---|---|---|---|
| `qwen2.5:1.5b` | native | 53,2% (33/62) | 37,5% (3/8) | 100,0% (8/8) | 81,2% (26/32) | 6 | 0 |
| `qwen2.5:1.5b` | json | 45,2% (28/62) | 25,0% (2/8) | 87,5% (7/8) | 59,3% (16/27) | 0 | 1 |
| `qwen2.5:3b` | native | 87,1% (54/62) | 87,5% (7/8) | 62,5% (5/8) | 94,3% (50/53) | 0 | 0 |
| `qwen2.5:3b` | json | 48,4% (30/62) | 25,0% (2/8) | 100,0% (8/8) | 89,7% (26/29) | 0 | 0 |
| `llama3.2:3b` | native | 82,3% (51/62) | 75,0% (6/8) | 12,5% (1/8) | 70,0% (35/50) | 4 | 0 |
| `llama3.2:3b` | json | 45,2% (28/62) | 62,5% (5/8) | 62,5% (5/8) | 55,6% (15/27) | 0 | 5 |

## Recuperação (RAG)

Dataset `evals/data/rag_retrieval.jsonl` (36 perguntas, sha256 `63a1ac64abdb`). O resultado depende só do modelo de embeddings, não do modelo de chat.

| Embeddings | Corpus | recall@1 | recall@5 | MRR | Latência da busca p50 (s) |
|---|---|---|---|---|---|
| `nomic-embed-text` | 13 docs / 25 trechos | 44,4% (16/36) | 72,2% (26/36) | 0,544 | 0,083 |

Arquivos brutos (com cada caso): `20261108T005533_tools_qwen2.5-1.5b_native.json`, `20261108T010114_tools_qwen2.5-1.5b_json.json`, `20261108T010229_tools_qwen2.5-3b_native.json`, `20261108T010336_tools_qwen2.5-3b_json.json`, `20261108T010500_tools_llama3.2-3b_native.json`, `20261108T010614_tools_llama3.2-3b_json.json`, `20261108T010625_rag_nomic-embed-text.json`
