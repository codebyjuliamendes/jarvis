# RAG: comparação de configurações

Gerado por `evals/eval_rag.py --grid` em 2026-11-08T01:43:58-03:00 (commit `8954b5f`, Ollama 0.40.14).
Máquina: Windows-10-10.0.26200-SP0 · GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB. Embeddings `nomic-embed-text`; corpus 13 documentos / 25 trechos; dataset `evals/data/rag_retrieval.jsonl` (36 perguntas, sha256 `63a1ac64abdb`).

Divisão fixa: ids ímpares = dev (18), pares = teste (18). O padrão do bot é escolhido pelo MRR no dev; o teste só confirma. Latência = `RagIndex.search` inteiro (inclui o embedding da pergunta no Ollama, exceto no bm25).

| Configuração | recall@1 | recall@5 | MRR | MRR dev (18) | MRR teste (18) | Latência média (ms) | Latência p95 (ms) |
|---|---|---|---|---|---|---|---|
| vector numpy sem prefixos | 44,4% (16/36) | 72,2% (26/36) | 0,544 | 0,537 | 0,551 | 98,5 | 142,4 |
| vector sqlite-vec sem prefixos | 44,4% (16/36) | 72,2% (26/36) | 0,544 | 0,537 | 0,551 | 102,0 | 126,4 |
| vector sqlite-vec + prefixos | 52,8% (19/36) | 80,6% (29/36) | 0,614 | 0,594 | 0,634 | 146,3 | 239,0 |
| bm25 | 83,3% (30/36) | 100,0% (36/36) | 0,912 | 0,907 | 0,917 | 1,2 | 2,6 |
| hybrid sem prefixos (RRF k=60) | 63,9% (23/36) | 94,4% (34/36) | 0,773 | 0,759 | 0,787 | 114,0 | 148,2 |
| hybrid + prefixos (RRF k=60) | 72,2% (26/36) | 100,0% (36/36) | 0,845 | 0,838 | 0,852 | 97,6 | 129,4 |

## Sensibilidade do k do RRF (informativo, não usado para escolher)

| Configuração | recall@1 | recall@5 | MRR | MRR dev (18) | MRR teste (18) | Latência média (ms) | Latência p95 (ms) |
|---|---|---|---|---|---|---|---|
| hybrid sem prefixos (RRF k=10) | 69,4% (25/36) | 100,0% (36/36) | 0,815 | 0,801 | 0,829 | 101,3 | 125,5 |
| hybrid sem prefixos (RRF k=30) | 66,7% (24/36) | 94,4% (34/36) | 0,787 | 0,787 | 0,787 | 101,3 | 150,0 |
| hybrid sem prefixos (RRF k=100) | 63,9% (23/36) | 94,4% (34/36) | 0,773 | 0,759 | 0,787 | 104,7 | 139,4 |
| hybrid + prefixos (RRF k=10) | 72,2% (26/36) | 100,0% (36/36) | 0,845 | 0,838 | 0,852 | 107,1 | 141,7 |
| hybrid + prefixos (RRF k=30) | 72,2% (26/36) | 100,0% (36/36) | 0,845 | 0,838 | 0,852 | 92,5 | 147,3 |
| hybrid + prefixos (RRF k=100) | 72,2% (26/36) | 100,0% (36/36) | 0,845 | 0,838 | 0,852 | 100,4 | 139,4 |

Arquivos brutos (com cada pergunta): `20261108T014358_rag_nomic-embed-text_vector-numpy.json`, `20261108T014405_rag_nomic-embed-text_vector-sqlite-vec.json`, `20261108T014414_rag_nomic-embed-text_vector-sqlite-vec-prefixos.json`, `20261108T014417_rag_nomic-embed-text_bm25.json`, `20261108T014424_rag_nomic-embed-text_hybrid-k60.json`, `20261108T014431_rag_nomic-embed-text_hybrid-prefixos-k60.json`, `20261108T014437_rag_nomic-embed-text_hybrid-k10.json`, `20261108T014445_rag_nomic-embed-text_hybrid-k30.json`, `20261108T014451_rag_nomic-embed-text_hybrid-k100.json`, `20261108T014459_rag_nomic-embed-text_hybrid-prefixos-k10.json`, `20261108T014506_rag_nomic-embed-text_hybrid-prefixos-k30.json`, `20261108T014512_rag_nomic-embed-text_hybrid-prefixos-k100.json`

## Conjunto de paráfrases (extras)

O conjunto original favorece a busca por palavras: em média **65,0%** das palavras de conteúdo
de cada pergunta aparecem no trecho certo. `evals/data/rag_paraphrase.jsonl` tem 36 perguntas
com os mesmos documentos e trechos esperados, escritas pelo sentido, com sinônimos e sem nomes
próprios quando dá: sobreposição média de **7,9%** (19 das 36 sem nenhuma palavra em comum),
medida por `evals/lexical_overlap.py` com os mesmos tokens do BM25. O critério para trocar o
padrão foi commitado junto com os dados, antes de qualquer busca nessas perguntas
(`docs/superpowers/specs/2026-11-08-rag-parafrases-design.md`). Tabela completa:
[`RAG_PARAPHRASE.md`](RAG_PARAPHRASE.md). A grade original foi refeita nesta noite, sem gravar,
com o mesmo código: números idênticos aos de cima.

| Configuração | MRR original | MRR paráfrases | Média (critério) | recall@1 paráfrases | recall@5 paráfrases |
|---|---|---|---|---|---|
| vetorial sqlite-vec, sem prefixos | 0,544 | 0,162 | 0,353 | 5,6% (2/36) | 36,1% (13/36) |
| vetorial sqlite-vec + prefixos | 0,614 | 0,191 | 0,402 | 5,6% (2/36) | 38,9% (14/36) |
| **bm25 (padrão)** | **0,912** | 0,283 | **0,598** | 19,4% (7/36) | 44,4% (16/36) |
| híbrido (RRF k=60), sem prefixos | 0,773 | 0,307 | 0,540 | 22,2% (8/36) | 47,2% (17/36) |
| híbrido (RRF k=60) + prefixos | 0,845 | **0,349** | 0,597 | 19,4% (7/36) | **63,9% (23/36)** |

- **Todos os modos despencam.** O melhor MRR nas paráfrases é 0,349. A busca vetorial, que
  deveria ser a vantagem aqui, é a pior (0,19 com prefixos): o `nomic-embed-text` foi treinado
  sobretudo em inglês, e em pt-BR com sinônimos ("moradia" × "aluguel", "salgado mineiro" ×
  "pão de queijo") ele não aproxima as frases. O BM25 ainda acha algo porque as palavras da
  pergunta aparecem em outros trechos do mesmo documento.
- **Critério aplicado:** média do MRR nos dois conjuntos. `bm25` 0,5977 × híbrido com prefixos
  0,5970 — empate técnico, e a regra de empate (diferença < 0,03) já mandava manter o atual.
  **O padrão continua `bm25`**; `jarvis/config.py` e `.env.example` não mudaram.
- **O que o critério não pega:** no recall@5 das paráfrases o híbrido com prefixos acha 23 de
  36 contra 16 do BM25, sem perder nada no recall@5 do original (100% nos dois). Como o
  `knowledge_search` devolve 5 trechos ao modelo, isso pode importar mais que o MRR no uso real.
  A troca para `JARVIS_RAG_MODE=hybrid` fica como decisão da Júlia (o critério foi escrito com o
  MRR e não vou mudá-lo depois de ver o resultado).
- **Próximo passo de verdade:** um modelo de embeddings multilíngue (ex.: `bge-m3`,
  `paraphrase-multilingual`) medido com estas mesmas 72 perguntas. Não baixei nenhum (limite de
  tamanho desta máquina e desta noite).
