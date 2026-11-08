# RAG híbrido do Jarvis — especificação

Data: 2026-11-08 · Branch: `vitrine-ai` · Subprojeto 4 (melhorias no RAG)

## Objetivo

Melhorar a recuperação do `RagIndex` com técnicas padrão e **medir cada uma isoladamente**
com o `evals/eval_rag.py` do subprojeto 1:

1. **sqlite-vec** para a busca vetorial (KNN em C dentro do SQLite), com queda automática
   para a implementação atual (NumPy sobre o JSON) se a extensão não carregar.
2. **Busca híbrida**: FTS5/BM25 + vetorial, combinadas por Reciprocal Rank Fusion (RRF).
3. **Prefixos do `nomic-embed-text`** (`search_document: ` / `search_query: `).
4. **Citação de fontes**: a ferramenta `knowledge_search` devolve arquivo + trecho, e o
   system prompt manda citar.

Linha de base (subprojeto 1, reproduzida antes de mexer): recall@1 44,4% (16/36),
recall@5 72,2% (26/36), MRR 0,544.

Fora de escopo: reranker com LLM/cross-encoder, mudar o chunking, outros modelos de
embeddings, ANN (o sqlite-vec 0.1.x faz busca exata, o que basta para milhares de trechos).

## Configuração (`.env`)

| Variável | Valores | Padrão |
|---|---|---|
| `JARVIS_RAG_MODE` | `vector`, `bm25`, `hybrid` | decidido pela medição (abaixo) |
| `JARVIS_RAG_RRF_K` | inteiro > 0 | `60` (valor do artigo do RRF, fixado *a priori*) |
| `JARVIS_RAG_PREFIXES` | `auto`, `true`, `false` | decidido pela medição; `auto` = liga só se o modelo tiver "nomic" no nome |
| `JARVIS_RAG_VECTOR_BACKEND` | `auto`, `sqlite-vec`, `numpy` | `auto` (tenta sqlite-vec, cai para NumPy com aviso no log) |

## Esquema e migração (transparente)

- A tabela `chunks` continua sendo a fonte da verdade, **com o embedding em JSON** como
  antes. Índices existentes continuam funcionando sem reindexar.
- `chunks_fts` (FTS5, `tokenize='unicode61 remove_diacritics 2'`, rowid = id do trecho) e
  `chunks_vec` (vec0, `distance_metric=cosine`, rowid = id do trecho) são **índices
  derivados**. Na abertura, se o conjunto de ids não bate com `chunks`, são reconstruídos
  a partir dela (sem chamar o Ollama). Mudou a dimensão do embedding (troca de modelo):
  `chunks_vec` é recriada.
- `documents.embed_sig` (coluna nova, `ALTER TABLE`) guarda `modelo|prefixo`. Ao indexar,
  um arquivo com assinatura diferente da atual é re-embedado mesmo sem ter mudado — trocar
  o modelo ou ligar os prefixos se resolve no próximo `/reindex`, sem `force`. Linhas
  antigas (assinatura vazia) equivalem a "mesmo modelo, sem prefixo".
- Sem FTS5 no SQLite (não é o caso do Python 3.8.20 do `.venv`, SQLite 3.46.0): o modo
  `bm25` vira `vector` e o `hybrid` usa só a perna vetorial, com aviso no log.
- Sem Ollama na hora da busca: `hybrid` responde só com BM25 (aviso no log) em vez de
  devolver nada.

## Busca

- **BM25**: a pergunta vira tokens (`\w+`, minúsculas), sem uma lista curta e genérica de
  stopwords do português (artigos, preposições, pronomes — escolhida antes de medir, não
  ajustada), cada token entre aspas e unidos por `OR`. Ordem por `bm25()` do FTS5.
- **Vetorial**: cosseno; sqlite-vec (`MATCH ... AND k = ?`) ou NumPy.
- **Híbrida**: top-20 de cada perna (constante fixada *a priori*), RRF:
  `score(d) = Σ 1 / (k + posição_d)`. Empate: ordem da perna vetorial.

## Citação de fontes

`search()` passa a devolver também `name` (nome do arquivo), `chunks_total` e `mode`.
`knowledge_search` formata cada resultado como `[n] arquivo.md, trecho i de N`, uma linha
`caminho: ...` e o texto, e termina com a instrução de citar no formato
`(fonte: arquivo.md, trecho i)`. O system prompt ganha uma linha com a mesma regra. Teste de
formato no pytest; exemplo real com `qwen2.5:3b`.

## Avaliação e cuidado com overfitting

`eval_rag.py` ganha `--mode`, `--prefixes on|off`, `--rrf-k`, `--vector-backend` e
`--grid` (roda todas as combinações e gera uma tabela). Cada rodada é um run no MLflow.

O dataset tem só 36 perguntas escritas por uma pessoa. Regras fixadas **antes** de medir:

- Nada é ajustado olhando pergunta por pergunta. Chunking e top-20 não mudam; k do RRF fica
  em 60.
- **Divisão dev/teste fixa**: ids ímpares (`rag-001`, `rag-003`, ...) = dev (18),
  pares = teste (18). O padrão (`JARVIS_RAG_MODE` e prefixos) é escolhido pelo **MRR no
  dev**; empate → a opção mais simples/rápida. O teste só confirma (ou não), e o resultado
  é relatado de qualquer jeito.
- Sensibilidade do k do RRF (10/30/60/100) é relatada como informação, não usada para
  escolher.
- Com 18 perguntas por metade, 1 pergunta = 5,6 pontos: diferenças pequenas não são
  conclusivas, e a análise diz isso.

## Decisões tomadas na implementação (com a medida)

- **Padrão `JARVIS_RAG_MODE=bm25`, `JARVIS_RAG_PREFIXES=auto`.** Pela regra acima: MRR dev
  bm25 0,907 > hybrid+prefixos 0,838 > hybrid 0,759 > vector+prefixos 0,594 > vector 0,537;
  o teste confirmou a ordem. Tabela em `evals/results/RAG.md`.
- **Instrução de citar no fim da saída da ferramenta**, com o 1º resultado como exemplo. No
  começo, o `qwen2.5:3b` não citou na demo; no fim, citou (o arquivo certo; o trecho copiado
  do exemplo).
- **Fontes anexadas pelo agente** (`Fontes consultadas: ...`, 3 primeiros resultados) quando
  o modelo usa o `knowledge_search` e não cita.
- **A grade refaz a configuração** se o Ollama cair no meio (até 2 vezes): o runner recusou
  conexão duas vezes depois de ~7 rodadas seguidas.
- **`evals/bench_vector_backend.py`**: no corpus do eval a busca é dominada pelo embedding no
  Ollama, então o ganho do sqlite-vec foi medido à parte (índice sintético, 768 dimensões).
