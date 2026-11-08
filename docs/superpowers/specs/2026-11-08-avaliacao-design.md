# Avaliação automatizada do Jarvis — especificação

Data: 2026-11-08 · Branch: `vitrine-ai` · Subprojeto 1 (avaliação) + 5a (testes mínimos)

## Objetivo

Medir com dados, e não com impressão, duas decisões que o Jarvis toma o tempo todo:

1. **Seleção de ferramenta** — dado um pedido em pt-BR, o LLM escolhe a ferramenta certa
   (ou nenhuma) e preenche os argumentos-chave?
2. **Recuperação do RAG** — dada uma pergunta, o `RagIndex` traz o trecho certo entre os
   primeiros resultados?

E comparar modelos locais pequenos (qwen2.5:1.5b, qwen2.5:3b, llama3.2:3b) em qualidade,
latência e vazão, para justificar o modelo padrão.

Fora de escopo: avaliar a resposta final em linguagem natural (LLM-as-judge), avaliar a
execução real das ferramentas, modelos > 3B.

## Arquivos

| Arquivo | Papel |
|---|---|
| `jarvis/metrics_utils.py` | percentis, resumo de latência e tokens/s a partir de `eval_count`/`eval_duration` (reutilizável pela futura observabilidade) |
| `jarvis/tool_parsing.py` | parsing das chamadas de ferramenta extraído do `agent.py` sem mudar comportamento (`parse_tool_call`, `extract_text_tool`) |
| `jarvis/llm.py` | ganha `chat_full()` (resposta completa do Ollama, com métricas) e `extra_options`; `chat()` continua devolvendo só a mensagem |
| `evals/common.py` | settings fictícias, construção do registro real, comparação de argumentos, leitura de JSONL, gravação de resultados |
| `evals/eval_tools.py` | avaliação de seleção de ferramentas |
| `evals/eval_rag.py` | avaliação de recuperação |
| `evals/compare_models.py` | roda as duas para vários modelos, grava JSON e tabela Markdown |
| `evals/make_fixtures.py` | gera o corpus fictício (inclui o .pdf e o .docx) |
| `evals/fixtures/` | corpus fictício (.md, .txt, .pdf, .docx) |
| `evals/data/tool_selection.jsonl` | ≥ 50 casos de seleção de ferramenta |
| `evals/data/rag_retrieval.jsonl` | ≥ 30 perguntas de recuperação |
| `evals/results/` | resultados (JSON com data e modelo) + `RESULTS.md` |
| `evals/README.md`, `evals/PENDENCIAS.md` | documentação e pendências honestas |
| `tests/` | pytest sem Ollama (mocks) |

## Formato dos JSONL

### `tool_selection.jsonl` (um objeto por linha)

```json
{"id": "ts-001", "prompt": "me lembra de tomar remédio daqui a 10 minutos",
 "expected_tool": "reminder",
 "expected_args": {"action": "add", "when": {"regex": "10\\s*(m|min)"}, "text": {"contains": "rem"}},
 "acceptable_tools": [],
 "category": "direto"}
```

- `expected_tool`: nome de uma ferramenta do registro real, ou `null` quando o certo é
  **não** chamar ferramenta.
- `expected_args` (opcional): só os argumentos-chave. Cada valor é um *matcher*:
  - escalar → igualdade após normalização (minúsculas, sem acento, espaços colapsados);
  - `{"contains": "x"}` → substring normalizada (listas viram texto unido por espaço);
  - `{"any_of": ["a", "b"]}` → igual a um dos valores;
  - `{"regex": "..."}` → `re.search` sobre o valor normalizado.
- `acceptable_tools` (opcional): outras ferramentas também aceitas em casos ambíguos.
- `category`: `direto`, `ambiguo` ou `sem_ferramenta`.

### `rag_retrieval.jsonl`

```json
{"id": "rag-001", "question": "quanto custa o aluguel?", "expected_doc": "contrato_aluguel.pdf",
 "expected_snippet": "R$ 2.350,00"}
```

## Como cada avaliação funciona

### Seleção de ferramentas (`eval_tools.py`)

- Constrói o **registro real** (`build_registry` com `RagIndex` e `ReminderService` em pasta
  temporária, sem Ollama) e o **system prompt real** do `Agent` (`_system_prompt`).
- Para cada caso faz **uma** chamada ao Ollama, sem histórico, e **não executa** nada.
- Modo `native`: envia os schemas em `tools`; decisão = primeiro item de `tool_calls`
  interpretado por `parse_tool_call` (a mesma função do agente). Se o servidor responder que
  o modelo não suporta tools, cai para o modo `json`, como o agente.
- Modo `json`: system prompt com `text_protocol_prompt`, sem `tools`; decisão =
  `extract_text_tool(content)` (a mesma função do agente).
- Uma chamada de aquecimento antes de medir (carrega o modelo na GPU), fora das métricas.
- `seed` fixo (42) via `extra_options`; temperatura e `num_ctx` iguais aos do agente.

### Recuperação (`eval_rag.py`)

- Copia `evals/fixtures/` para uma pasta temporária, indexa com o `RagIndex` **real**
  (`nomic-embed-text` pelo Ollama) e consulta `search(pergunta, k=5)`.
- Um resultado é **acerto** quando o arquivo é `expected_doc` **e** o texto do trecho contém
  `expected_snippet` (normalização de espaços e caixa; PDF quebra linhas).

## Métricas e cálculo

| Métrica | Cálculo |
|---|---|
| Acurácia de ferramenta | casos em que a ferramenta prevista ∈ {`expected_tool`} ∪ `acceptable_tools` (`null` == nenhuma chamada) ÷ total |
| Acurácia de argumentos | entre os casos com `expected_args`: ferramenta correta **e** todos os matchers satisfeitos ÷ casos com `expected_args`. Também registramos a versão condicional (só entre os que acertaram a ferramenta) |
| Taxa de JSON inválido | casos em que o modelo **tentou** uma chamada e o parser do agente não conseguiu extrair ferramenta + args: modo `json` → resposta contém `{` e `extract_text_tool` devolve `None`; modo `native` → `arguments` veio como string que não é JSON. ÷ total |
| Latência média, p50, p95 | relógio de parede por chamada (`time.perf_counter`), em segundos. Percentil por interpolação linear entre ordens (mesmo método padrão do NumPy) |
| tokens/s | Σ `eval_count` ÷ Σ (`eval_duration` / 1e9) das respostas do Ollama (tokens gerados por segundo de geração; não inclui o processamento do prompt) |
| recall@k | perguntas com pelo menos um acerto entre os k primeiros ÷ total (k = 1 e 5) |
| MRR | média de 1/posição do primeiro acerto (0 se não houver acerto no top-5) |

Também registramos, sem entrar na tabela principal: acurácia por categoria, chamadas
"vazadas" (no modo nativo, JSON de ferramenta no `content` em vez de `tool_calls` — o agente
não executa isso), erros de chamada e a matriz de confusão.

## Decisões

- **Uma chamada por caso, sem executar**: isola a decisão do modelo; executar exigiria
  credenciais e teria efeitos colaterais.
- **Reutilizar o parsing do agente** em vez de reimplementar: a avaliação mede exatamente o
  que o agente faria. A extração para `jarvis/tool_parsing.py` mantém os métodos estáticos
  do `Agent` delegando, e os testes de parsing são escritos **antes** da extração.
- **`chat_full` no cliente** em vez de chamar o Ollama direto: o eval usa o mesmo payload do
  agente; `chat()` não muda.
- **RAG independe do modelo de chat**: o resultado depende só do modelo de embeddings. O
  `compare_models.py` roda o RAG uma vez por modelo de embeddings e o mostra em tabela
  própria, para não sugerir uma diferença entre LLMs que não existe.
- **Resultados versionados** (`evals/results/*.json`) para que a tabela do README seja
  reproduzível e auditável.
- **Sem dependências novas**: tudo com a biblioteca padrão, `httpx`, `numpy`, PyMuPDF e
  python-docx que o projeto já usa; `pytest` só para os testes.
- **Python 3.8**: `from __future__ import annotations`, `typing.List/Dict/Optional`.
