# RAG com paráfrases — especificação e critério fixado antes de medir

Data: 2026-11-08 · Extra do subprojeto 4 (RAG híbrido).

## Problema

O padrão do RAG (`bm25`) foi escolhido no `evals/data/rag_retrieval.jsonl`, onde em média
**65%** das palavras de conteúdo de cada pergunta aparecem no trecho certo
(`evals/lexical_overlap.py`). Esse conjunto favorece a busca por palavras: no uso real as
pessoas perguntam com outras palavras ("quanto pago de moradia" e não "valor do aluguel").

## Conjunto novo: `evals/data/rag_paraphrase.jsonl`

- 36 perguntas, uma para cada pergunta do conjunto original, com **os mesmos documentos e os
  mesmos trechos esperados** (`expected_doc`, `expected_snippet`), ids `par-001`…`par-036`
  (a paridade do número preserva a divisão dev/teste, que aqui é só informativa).
- Escritas a partir do **sentido** do trecho, com sinônimos e perífrases, sem nomes próprios
  quando dá (sem "Falcon", "Lisboa", "Airbnb", "L'Hôpital", "Origami"...).
- A redação foi ajustada **só** contra a medida de sobreposição lexical (uma pergunta,
  `par-014`, foi reescrita porque a primeira versão repetia quatro palavras do trecho). **Nenhum
  resultado de busca foi visto antes de este arquivo e o critério abaixo entrarem num commit.**
- Sobreposição medida antes de qualquer busca: **7,9%** em média (19 das 36 com 0%), contra
  **65,0%** do original (1 com 0%).
- Limitações conhecidas desde já: quem escreveu as paráfrases é quem escreveu o original e leu
  os trechos (não é uma pessoa de fora nem uso real); algumas perguntas exigem conhecimento de
  mundo (origami → "dobrado em estilo japonês", pão de queijo → "salgado mineiro"), o que é
  justamente o caso em que a busca vetorial deveria ajudar. Com 19 perguntas sem nenhuma
  palavra em comum, o `bm25` devolve lista vazia nelas por construção.
- **As paráfrases são só teste.** Nada no RAG (stopwords, top-20 do híbrido, k do RRF, tamanho
  dos trechos, prefixos) será ajustado olhando estes resultados.

## Grade

A mesma grade principal do `eval_rag.py --grid` (k = 60), rodada nos dois conjuntos com o
mesmo código e o mesmo índice: vetorial NumPy sem prefixos, vetorial sqlite-vec sem/com
prefixos, `bm25`, híbrido sem/com prefixos. A sensibilidade do k fica como informação.

## Critério para o padrão (fixado antes de rodar a grade nas paráfrases)

Candidatos: as 5 configurações distintas da grade principal (a linha NumPy repete a
sqlite-vec sem prefixos e fica de fora da escolha).

1. **Métrica:** média simples do MRR nos dois conjuntos,
   `(MRR_original + MRR_paráfrases) / 2`. Pesos iguais porque nenhum dos dois representa
   melhor o uso real: o original é otimista para palavras, o de paráfrases é pessimista.
2. **Restrição:** o escolhido não pode ficar mais de **0,10** de MRR abaixo do melhor no
   conjunto original (proteção contra trocar um ganho em paráfrases por uma perda grande nas
   perguntas literais, que também acontecem).
3. **Empate a favor do atual:** se o melhor pela métrica superar o `bm25` por menos de
   **0,03** de média, o padrão continua `bm25` (1 ms por busca, sem chamar o Ollama na
   consulta). 0,03 ≈ uma pergunta inteira (1/36 = 0,028) em cada conjunto.
4. Se o critério apontar outro modo, o padrão muda em `jarvis/config.py`,
   `jarvis/rag.py` (`DEFAULT_MODE`) e `.env.example`, com o motivo no README e no `RAG.md`.

## O que não muda

Divisão dev/teste do conjunto original, stopwords, `HYBRID_CANDIDATES = 20`, k = 60 do RRF,
tamanho dos trechos. Pesos no RRF e reranker continuam fora (sobreajuste com 72 perguntas).
