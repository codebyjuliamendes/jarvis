# RAG: comparação de configurações (paráfrases)

Gerado por `evals/eval_rag.py --grid` em 2026-11-08T05:10:44-03:00 (commit `c971115`, Ollama 0.40.14).
Máquina: Windows-10-10.0.26100-SP0 · GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB. Embeddings `nomic-embed-text`; corpus 13 documentos / 25 trechos; dataset `evals/data/rag_paraphrase.jsonl` (36 perguntas, sha256 `9a64dc118a12`).

Conjunto de paráfrases: mesmos documentos e trechos esperados do original, perguntas escritas com sinônimos e baixa sobreposição lexical (`evals/lexical_overlap.py`). É só teste: nada do RAG foi ajustado com ele. O critério para o padrão foi fixado antes desta grade (`docs/superpowers/specs/2026-11-08-rag-parafrases-design.md`). As colunas dev/teste são só informativas aqui. Latência = `RagIndex.search` inteiro (inclui o embedding da pergunta no Ollama, exceto no bm25).

| Configuração | recall@1 | recall@5 | MRR | MRR dev (18) | MRR teste (18) | Latência média (ms) | Latência p95 (ms) |
|---|---|---|---|---|---|---|---|
| vector numpy sem prefixos | 5,6% (2/36) | 36,1% (13/36) | 0,162 | 0,144 | 0,181 | 96,7 | 165,3 |
| vector sqlite-vec sem prefixos | 5,6% (2/36) | 36,1% (13/36) | 0,162 | 0,144 | 0,181 | 107,3 | 164,8 |
| vector sqlite-vec + prefixos | 5,6% (2/36) | 38,9% (14/36) | 0,191 | 0,194 | 0,187 | 106,9 | 145,7 |
| bm25 | 19,4% (7/36) | 44,4% (16/36) | 0,283 | 0,306 | 0,261 | 0,8 | 1,2 |
| hybrid sem prefixos (RRF k=60) | 22,2% (8/36) | 47,2% (17/36) | 0,307 | 0,335 | 0,280 | 114,7 | 164,1 |
| hybrid + prefixos (RRF k=60) | 19,4% (7/36) | 63,9% (23/36) | 0,349 | 0,368 | 0,331 | 104,2 | 125,7 |

Arquivos brutos (com cada pergunta): `20261108T051044_rag_nomic-embed-text_parafrases_vector-numpy.json`, `20261108T051051_rag_nomic-embed-text_parafrases_vector-sqlite-vec.json`, `20261108T051058_rag_nomic-embed-text_parafrases_vector-sqlite-vec-prefixos.json`, `20261108T051102_rag_nomic-embed-text_parafrases_bm25.json`, `20261108T051109_rag_nomic-embed-text_parafrases_hybrid-k60.json`, `20261108T051116_rag_nomic-embed-text_parafrases_hybrid-prefixos-k60.json`
