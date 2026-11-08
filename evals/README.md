# Avaliações do Jarvis

Duas decisões que o Jarvis toma o tempo todo, medidas com dados:

1. **Seleção de ferramentas** — dado um pedido em pt-BR, o LLM escolhe a ferramenta certa
   (ou nenhuma) e preenche os argumentos certos?
2. **Recuperação (RAG)** — dada uma pergunta, a busca semântica traz o trecho certo?

E uma comparação entre modelos locais pequenos para escolher o padrão do bot. A tabela mais
recente fica em [`results/RESULTS.md`](results/RESULTS.md); a especificação, em
[`docs/superpowers/specs/2026-11-08-avaliacao-design.md`](../docs/superpowers/specs/2026-11-08-avaliacao-design.md).
O que não funcionou ou ficou de fora está em [`PENDENCIAS.md`](PENDENCIAS.md).

## Como rodar

Pré-requisitos: Ollama ligado e os modelos baixados (`ollama pull qwen2.5:3b`, etc.).
Comandos a partir da raiz do repositório (no Linux/macOS, troque `.venv/Scripts/python.exe`
por `.venv/bin/python`):

```bash
# dependências só das avaliações (MLflow + pytest); o bot não precisa delas
uv pip install --python .venv/Scripts/python.exe -r requirements-evals.txt

# comparação completa (o que gerou a tabela do README principal)
.venv/Scripts/python.exe evals/compare_models.py qwen2.5:1.5b qwen2.5:3b llama3.2:3b --modes native,json

# avaliações isoladas
.venv/Scripts/python.exe evals/eval_tools.py --model qwen2.5:3b --mode native -v
.venv/Scripts/python.exe evals/eval_tools.py --model qwen2.5:3b --mode json --limit 10
.venv/Scripts/python.exe evals/eval_rag.py --embed-model nomic-embed-text -v

# testes unitários (não precisam do Ollama)
.venv/Scripts/python.exe -m pytest -q
```

Opções úteis: `--limit N` (só os N primeiros casos), `--no-save` (não grava JSON),
`--no-mlflow`, `--ollama-url`. Os modelos rodam **um por vez**: o `compare_models.py`
descarrega cada um da GPU antes do próximo.

> Por padrão os evals falam com `http://127.0.0.1:11434`, não `localhost`: no Windows cada
> conexão nova com `localhost` perdia ~2 s tentando IPv6 antes do IPv4, o que contaminaria a
> latência (ver PENDENCIAS.md).

## 1. Seleção de ferramentas — `eval_tools.py`

**Dados:** [`data/tool_selection.jsonl`](data/tool_selection.jsonl), 78 pedidos realistas em
pt-BR. Cobre as **25 ferramentas do registro real** (um teste garante isso), 8 casos em que o
certo é **não** chamar ferramenta e 8 casos ambíguos.

```json
{"id": "ts-059", "category": "direto", "prompt": "me lembra de tomar remédio daqui a 10 minutos",
 "expected_tool": "reminder",
 "expected_args": {"action": "add", "when": {"regex": "10\\s*(m|min)"}, "text": {"contains": "remedio"}}}
```

| Campo | Significado |
|---|---|
| `expected_tool` | ferramenta esperada, ou `null` quando o certo é responder sem ferramenta |
| `acceptable_tools` | outras respostas aceitas em casos ambíguos (ex.: `knowledge_search` ou `obsidian`) |
| `expected_args` | só os argumentos-chave. Valor simples = igualdade; `{"contains": ...}`, `{"contains_all": [...]}`, `{"any_of": [...]}`, `{"regex": ...}`. Tudo comparado em minúsculas, sem acento e com espaços colapsados |
| `category` | `direto`, `ambiguo` ou `sem_ferramenta` |

**Como funciona:** monta o registro real de ferramentas e o **system prompt real** do
`Agent`, e faz **uma** chamada ao Ollama por pedido, sem histórico e **sem executar** a
ferramenta. A resposta é interpretada pelas mesmas funções do agente
(`jarvis/tool_parsing.py`):

- **modo `native`**: schemas enviados em `tools`; vale o primeiro item de `tool_calls`;
- **modo `json`** (fallback do agente para modelos sem function calling): o system prompt
  ganha o protocolo de texto e vale o JSON extraído do texto.

Antes de medir há uma chamada de aquecimento (carrega o modelo na GPU). Seed fixo (42);
temperatura (0,2) e `num_ctx` (8192) iguais aos do agente.

**Métricas**

| Métrica | Cálculo |
|---|---|
| Acurácia de ferramenta | acertos ÷ total; acerto = ferramenta prevista ∈ {esperada} ∪ aceitáveis (`null` = nenhuma chamada) |
| Acurácia de args | entre os casos com `expected_args`: ferramenta certa **e** todos os argumentos-chave certos. O JSON traz também a versão condicional (só entre os que acertaram a ferramenta) |
| JSON inválido | o modelo tentou chamar uma ferramenta e o parser do agente não conseguiu extrair: no modo `json`, a resposta tem `{` mas não vira chamada; no nativo, `arguments` veio como string que não é JSON |
| Latência média / p50 / p95 | relógio de parede por chamada, em segundos; percentil por interpolação linear (igual ao NumPy) |
| tokens/s | Σ `eval_count` ÷ Σ `eval_duration` das respostas do Ollama (só geração, sem o prompt) |

Também registrados: acurácia por categoria, chamadas "vazadas" (no modo nativo, JSON de
ferramenta escrito no texto em vez de `tool_calls` — o agente não executa isso), ferramentas
inexistentes e a lista de erros `esperada -> prevista`.

## 2. Recuperação — `eval_rag.py`

**Corpus:** [`fixtures/`](fixtures/), 13 documentos **fictícios** em .md, .txt, .pdf e .docx
(contrato de aluguel, ata de reunião, relatório trimestral, regulamento de condomínio,
receitas, roteiro de viagem, notas de aula, manual de roteador...). Há distratores de
propósito: outra reunião de projeto, a ata do condomínio de outro ano, outra viagem, outra
matéria — mesmo tema, valores e datas diferentes. Gerado por
[`make_fixtures.py`](make_fixtures.py) (o PDF com PyMuPDF e o DOCX com python-docx).

**Dados:** [`data/rag_retrieval.jsonl`](data/rag_retrieval.jsonl), 36 perguntas escritas com
outras palavras, cada uma com o documento e um trecho literal esperado:

```json
{"id": "rag-007", "question": "Qual o valor mensal que eu pago de aluguel?",
 "expected_doc": "contrato_aluguel.pdf", "expected_snippet": "R$ 2.350,00"}
```

**Como funciona:** copia os fixtures para uma pasta temporária, indexa com o `RagIndex` real
(mesmo chunking e mesmo modelo de embeddings do bot) e consulta `search(pergunta, k=5)`.
Acerto = arquivo esperado **e** trecho que contém o snippet (normalizando espaços, caixa e
acentos, porque o PDF quebra linhas).

| Métrica | Cálculo |
|---|---|
| recall@1 | perguntas com acerto na 1ª posição ÷ total |
| recall@5 | perguntas com acerto entre as 5 primeiras ÷ total |
| MRR | média de 1/posição do primeiro acerto (0 se não houver no top-5) |

O corpus é pequeno (25 trechos), então recall@5 é otimista: 5 resultados são 20% do índice.
recall@1 e MRR dizem mais.

### Modos de busca e grade (subprojeto 4)

O `eval_rag.py` mede qualquer configuração do `RagIndex`; sem flags, mede os **padrões do
bot** (lidos de `jarvis/config.py`, hoje `bm25` com prefixos `auto`):

```bash
.venv/Scripts/python.exe evals/eval_rag.py --mode vector --prefixes off --vector-backend numpy  # linha de base
.venv/Scripts/python.exe evals/eval_rag.py --mode hybrid --prefixes on --rrf-k 60
.venv/Scripts/python.exe evals/eval_rag.py --grid    # 6 configurações + sensibilidade do k -> results/RAG.md
```

| Opção | Valores |
|---|---|
| `--mode` | `vector`, `bm25`, `hybrid` |
| `--prefixes` | `on`/`off` (`search_document:`/`search_query:` do `nomic-embed-text`) |
| `--rrf-k` | k do Reciprocal Rank Fusion (padrão 60) |
| `--vector-backend` | `auto`, `sqlite-vec`, `numpy` |
| `--grid` | roda a grade inteira; `--no-sensitivity` pula a variação do k |

Além do total, as métricas saem por **metade fixa** do dataset — ids ímpares = `dev`, pares =
`test`, decidido antes de medir. O padrão do bot foi escolhido pelo MRR no dev, e o teste só
confirma; a regra está na
[spec do subprojeto 4](../docs/superpowers/specs/2026-11-08-rag-hibrido-design.md). A
latência agora traz também a média. Se o Ollama cair no meio da grade, a configuração é
refeita do zero (até 2 vezes, com aviso).

`bench_vector_backend.py` mede só a perna vetorial (sqlite-vec × NumPy) num índice sintético
de 768 dimensões, para N trechos — no corpus do eval os dois empatam porque o tempo é do
Ollama.

## 3. Comparação de modelos — `compare_models.py`

Roda a seleção de ferramentas para cada modelo e modo e o RAG para cada modelo de
embeddings. Grava em [`results/`](results/):

- um JSON por execução (`<data>_tools_<modelo>_<modo>.json`, `<data>_rag_<modelo>.json`)
  com metadados (data, commit, versão do Ollama, GPU), métricas e **cada caso** com a
  resposta do modelo — dá para auditar qualquer número da tabela;
- `<data>_compare.json` (resumo) e `RESULTS.md` (tabela).

Os JSON são versionados de propósito: a tabela do README é reproduzível e auditável.

**Sem dados da máquina.** O system prompt mostra a pasta de trabalho (um diretório temporário
dentro da home) e o modelo copia esse caminho nas respostas. `common.save_json` passa tudo
por [`sanitize.py`](sanitize.py), que troca a home e o nome do usuário (com qualquer escape,
sem acento, truncado, nome curto 8.3) por `<HOME>`/`<USER>`. Os números não mudam: as
métricas são calculadas antes de gravar. Para conferir ou limpar arquivos já gravados:

```bash
.venv/Scripts/python.exe evals/sanitize.py --check evals/results   # sai com 1 se achar algo
.venv/Scripts/python.exe evals/sanitize.py evals/results           # regrava limpo
```

### Repetições e intervalo de confiança (`--repeats N`)

Uma rodada só não basta: com o mesmo código, o mesmo seed (42) e temperatura 0,2, o
`qwen2.5:3b` nativo deu 66/78 numa rodada e 64/78 na seguinte (a GPU não é determinística).
`eval_tools.py --repeats N` e `compare_models.py --repeats N` (padrão 1) fazem N rodadas
idênticas e gravam um JSON por (modelo, modo) com todas as rodadas
(`<data>_tools_<modelo>_<modo>_xN.json`), além de `<data>_compare_xN.json` e da tabela em
`--markdown` (a rodada de 08/11 está em [`results/REPEATS.md`](results/REPEATS.md)).
Estatística em [`stats.py`](stats.py), só stdlib:

- **Média e desvio-padrão amostral entre as rodadas** — mede o não determinismo.
- **IC 95% principal: bootstrap percentil sobre os casos** (10.000 reamostragens, seed fixa).
  Cada caso entra com a fração de rodadas em que acertou; reamostrar casos (e não as 3×78
  respostas) mantém juntas as rodadas do mesmo caso e cobre a incerteza maior, que é a da
  amostra de 78 pedidos.
- **IC de Wilson** sobre a média de acertos com n = 78: fórmula fechada, boa perto de 0/1;
  ignora a variação entre rodadas e serve de conferência.
- **Comparação pareada (`compare_models`): McNemar exato** bilateral sobre os casos
  discordantes, com o acerto de cada caso por maioria das rodadas (N ímpar); também rodada a
  rodada, como sensibilidade, e o IC bootstrap pareado da diferença. Com várias comparações, a
  tabela usa o limiar de Bonferroni (0,05 / nº de comparações).

```bash
.venv/Scripts/python.exe evals/compare_models.py qwen2.5:1.5b qwen2.5:3b llama3.2:3b     --modes native,json --repeats 3 --skip-rag --markdown evals/results/REPEATS.md
```

## 4. MLflow

Cada execução de `eval_tools.py`, `eval_rag.py` e `compare_models.py` vira um **run** no
MLflow (o `compare_models` é um run pai com um run filho por modelo/modo e por modelo de
embeddings):

- **params:** modelo, modo pedido e efetivo, nº de exemplos, hash sha256 do JSONL (versão do
  dataset), hash dos fixtures, commit git (com `-dirty` se havia mudança local), seed,
  `num_ctx`, versão do Ollama, GPU;
- **métricas:** `tool_accuracy`, `args_accuracy`, `args_accuracy_given_tool`,
  `invalid_json_rate`, `latency_p50_s`, `latency_p95_s`, `tokens_per_s`, acurácia por
  categoria; `recall_at_1`, `recall_at_5`, `mrr` (também `_dev`/`_test`), latência da
  busca (média, p50, p95); no RAG, os params incluem `rag_mode`, `prefixes`, `rrf_k` e
  `vector_backend` (a `--grid` é um run pai com um filho por configuração);
- **artefatos:** o JSON de resultados e a tabela Markdown.

O MLflow é **opcional**: sem ele instalado (ou com `JARVIS_MLFLOW_ENABLED=false`, ou com
`--no-mlflow`), os evals rodam igual e só avisam.

Configuração (no `.env`, ver `.env.example`):

| Variável | Padrão |
|---|---|
| `JARVIS_MLFLOW_ENABLED` | `true` |
| `JARVIS_MLFLOW_TRACKING_URI` | vazio = `sqlite:///<repo>/evals/mlflow.db` (artefatos em `evals/mlartifacts/`) |
| `JARVIS_MLFLOW_EXPERIMENT` | `jarvis-evals` |

O banco e os artefatos do MLflow ficam fora do git (são locais e pesados); os JSON de
`results/` continuam versionados.

**Abrir a interface e comparar runs:**

```bash
.venv/Scripts/python.exe -m mlflow ui --backend-store-uri sqlite:///evals/mlflow.db --port 5000
# abra http://127.0.0.1:5000
```

1. Entre no experimento **jarvis-evals**. Os runs filhos aparecem sob o run
   `compare ...` (expanda com o `+`).
2. Marque os runs que quer comparar (ex.: os três `tools ... native`) e clique em
   **Compare**: a tela mostra params lado a lado e gráficos de métricas (use *Parallel
   Coordinates* ou *Scatter* com `latency_p50_s` × `tool_accuracy` para ver o custo de
   cada ponto de acurácia).
3. Na lista de runs, adicione colunas `tool_accuracy`, `latency_p50_s`, `tokens_per_s` e
   filtre, por exemplo, `params.effective_mode = "native"` ou
   `params.dataset_sha256 = "<hash>"` para só comparar runs com o mesmo dataset.
4. Em cada run, a aba **Artifacts** tem o JSON com todos os casos e a `tabela.md`.

## Estrutura

```
evals/
  common.py          settings fictícias, matchers de args, hashes, metadados
  eval_tools.py      seleção de ferramentas
  eval_rag.py        recuperação (modos vector/bm25/hybrid, --grid -> results/RAG.md;
                     --dataset paraphrase -> results/RAG_PARAPHRASE.md)
  lexical_overlap.py sobreposição de palavras entre pergunta e trecho certo
  stats.py           média, desvio, IC bootstrap/Wilson, McNemar exato (stdlib)
  bench_vector_backend.py  microbenchmark sqlite-vec x NumPy
  compare_models.py  comparação + RESULTS.md
  sanitize.py        tira home/usuário da máquina dos JSON antes de gravar
  report.py          tabelas Markdown
  tracking.py        MLflow opcional
  make_fixtures.py   gera o corpus fictício
  data/              datasets (JSONL)
  fixtures/          corpus fictício
  results/           resultados versionados
jarvis/metrics_utils.py  percentis, latência e tokens/s (reutilizável pela observabilidade)
jarvis/tool_parsing.py   parsing das chamadas, compartilhado com o agente
tests/                   pytest sem Ollama
```
