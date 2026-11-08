# Observabilidade do Jarvis — especificação

Data: 2026-11-08 · Branch: `vitrine-ai` · Subprojeto 3 (observabilidade)

## Objetivo

Responder com dados, a qualquer momento e pelo próprio Telegram, três perguntas sobre o
Jarvis em uso real:

1. **Está rápido?** Latência por requisição (p50/p95), tokens de prompt e de resposta,
   vazão do modelo (tokens/s).
2. **Está funcionando?** Taxa de erro das requisições e de cada ferramenta.
3. **O que ele faz?** Quais ferramentas são mais usadas e quais mais falham; quantas
   iterações do laço do agente cada pedido consome.

E deixar cada linha de log ligada à requisição que a gerou (id da requisição), em JSON
quando configurado, para dar para filtrar um pedido inteiro com `jq`/`grep`.

Fora de escopo: dashboards externos (Prometheus/Grafana/OpenTelemetry), tracing
distribuído, alertas, retenção/rotação automática, métricas do RAG/indexação.

## Princípios

- **Não mudar o comportamento.** As respostas do agente são idênticas com e sem métricas.
  O `Agent` só grava se receber um gravador (`metrics=`); sem ele (testes, smoke, evals),
  nada muda.
- **Falha ao medir nunca derruba a resposta.** Toda gravação fica em `try/except`; o erro
  vai para o log em nível `WARNING` e a resposta segue.
- **Barato.** Durante a requisição só se acumula em memória (listas num objeto de
  contexto). No fim, **uma** transação SQLite: 1 `INSERT` na tabela `metrics` + 1
  `executemany` em `metric_tool_calls`. Nada é gravado dentro do laço.
- **Sem conteúdo.** Não se grava o texto do pedido, da resposta nem os argumentos das
  ferramentas — só números, nomes de ferramenta e a mensagem de erro (truncada em 300
  caracteres). O `audit.log` existente continua sendo o registro de argumentos (já
  higienizados).
- **Só biblioteca padrão.** `sqlite3`, `contextvars`, `logging`, `json`, `uuid`.
- **Python 3.8.**

## Onde fica o banco: arquivo separado

`data/metrics.sqlite3` (ao lado de `JARVIS_DB_PATH`; configurável por
`JARVIS_METRICS_DB_PATH`), **não** o `data/jarvis.sqlite3` do histórico. Motivos:

- **Natureza diferente.** O `jarvis.sqlite3` guarda conteúdo pessoal (conversas,
  preferências); métricas são telemetria. Dá para apagar, copiar ou compartilhar o
  arquivo de métricas (por exemplo, para analisar num notebook) sem expor conversas.
- **Ciclo de vida independente.** Zerar as métricas é `rm data/metrics.sqlite3`; o
  `/reset` e o backup do histórico não são afetados, e uma migração de esquema das
  métricas não toca a tabela de mensagens.
- **Sem disputa de trava.** O `Memory` usa uma conexão com `threading.Lock`; um arquivo
  próprio com conexão própria não compete com as escritas do histórico.
- Custo: um arquivo a mais em `data/` (já ignorado pelo git).

## Esquema (versão 1, `PRAGMA user_version = 1`)

```sql
CREATE TABLE metrics (
    request_id        TEXT PRIMARY KEY,   -- 12 hex (uuid4), o mesmo dos logs
    ts                REAL NOT NULL,      -- início, epoch UTC (para janelas de tempo)
    created_at        TEXT NOT NULL,      -- início, ISO 8601 UTC (leitura humana)
    source            TEXT NOT NULL,      -- chat | briefing | demo ...
    chat_id           INTEGER,
    model             TEXT NOT NULL,
    mode              TEXT NOT NULL,      -- native | text (protocolo de ferramentas usado no fim)
    latency_s         REAL NOT NULL,      -- relógio de parede do respond() inteiro
    llm_calls         INTEGER NOT NULL,   -- chamadas ao /api/chat (inclui as que falharam)
    llm_s             REAL NOT NULL,      -- soma do tempo de parede dessas chamadas
    prompt_tokens     INTEGER,            -- soma de prompt_eval_count
    completion_tokens INTEGER,            -- soma de eval_count
    eval_s            REAL,               -- soma de eval_duration (s)
    tokens_per_s      REAL,               -- completion_tokens / eval_s (ponderado)
    iterations        INTEGER NOT NULL,   -- voltas do laço do agente
    tool_calls        INTEGER NOT NULL,
    outcome           TEXT NOT NULL,      -- ok | error | max_iterations
    error             TEXT                -- "Classe: mensagem" (300 chars) se outcome != ok
);
CREATE INDEX idx_metrics_ts ON metrics(ts);

CREATE TABLE metric_tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT NOT NULL REFERENCES metrics(request_id),
    seq         INTEGER NOT NULL,         -- ordem dentro da requisição
    tool        TEXT NOT NULL,
    duration_s  REAL NOT NULL,
    ok          INTEGER NOT NULL,         -- 1 sucesso, 0 falha
    error       TEXT
);
CREATE INDEX idx_tool_calls_request ON metric_tool_calls(request_id);
```

Os tokens e a vazão saem das métricas que o próprio Ollama devolve
(`prompt_eval_count`, `eval_count`, `eval_duration`), reaproveitando
`jarvis/metrics_utils.py` (`aggregate_tokens_per_second`, `percentile`). A vazão é
ponderada (soma de tokens ÷ soma de tempo), como nas avaliações.

### O que conta como falha de ferramenta

As ferramentas do Jarvis quase nunca levantam exceção: capturam o erro e devolvem um texto
`"Erro: ..."` para o modelo ler. Por isso `ok = 0` quando:

- o handler levantou exceção (o registro devolve `"Erro ao executar '...'"`);
- a ferramenta não existe (`"Erro: ferramenta desconhecida"`);
- o texto devolvido começa com `"Erro"`.

Ferramentas que reportam problema com outra frase (por exemplo, "Notion não configurado")
contam como sucesso. É uma heurística declarada, não uma verdade — está em
`evals/PENDENCIAS.md`.

### Desfecho da requisição

- `ok`: o modelo deu uma resposta final.
- `max_iterations`: o laço acabou sem resposta final (o agente devolve "Não consegui
  concluir em N passos").
- `error`: o Ollama ficou indisponível / devolveu erro (o agente devolve a mensagem de
  erro ao usuário, como hoje) ou uma exceção inesperada escapou do `respond()` (é gravada
  e relançada, sem engolir).

A **taxa de erro** do `/metrics` é `(error + max_iterations) / total`, com as duas partes
mostradas separadamente.

## Arquitetura

| Arquivo | Papel |
|---|---|
| `jarvis/observability.py` | `request_id` em `contextvars`; `RequestTrace` (acumulador em memória); ganchos `record_llm_response`, `record_llm_error`, `record_tool_call` que não fazem nada sem uma requisição ativa |
| `jarvis/metrics_store.py` | `MetricsStore` (SQLite: esquema, `record(trace)`, `fetch_since(ts)`); `summarize()` e `format_summary()` — funções puras |
| `jarvis/logging_setup.py` | `JsonFormatter`, filtro que injeta `request_id`, `setup_logging(format, level)` |
| `jarvis/llm.py` | `chat_full()` chama o gancho com a resposta (tokens) e o tempo de parede; em erro, o gancho de erro |
| `jarvis/tools/base.py` | `execute()` mede cada ferramenta e chama o gancho |
| `jarvis/agent.py` | abre a requisição (`request_scope`), conta iterações e desfecho, grava no fim; parâmetro opcional `metrics=` e `source=` |
| `jarvis/utils.py` | `run_blocking` passa a copiar o contexto para a thread (para o `request_id` aparecer nos logs das ferramentas — equivalente ao `asyncio.to_thread` do 3.9) |
| `jarvis/telegram_bot.py` | comando `/metrics` (só usuários autorizados, via `_guard`); abre o escopo da requisição antes do agente |
| `jarvis/__main__.py` | configura logs e cria o `MetricsStore` |
| `scripts/demo_metrics.py` | roda o agente real (Ollama, sem Telegram) num banco temporário e imprime o `/metrics` |

Por que `contextvars` e não passar um objeto por parâmetro: o `OllamaClient` e o
`ToolRegistry` são compartilhados entre conversas simultâneas; um atributo "última
resposta" no cliente seria uma condição de corrida. Cada tarefa asyncio tem a sua cópia do
contexto, então duas conversas ao mesmo tempo não misturam métricas. E não é preciso mudar
nenhuma assinatura pública além de dois parâmetros opcionais no `Agent`.

## `/metrics`

Resumo das últimas 24 h e 7 dias, em bloco monoespaçado:

- requisições (e quantas com erro / sem concluir), taxa de erro;
- latência p50 / p95 / máx.;
- iterações médias, tokens médios de prompt e de resposta, vazão (tokens/s);
- top 5 ferramentas por uso (chamadas, falhas, p50 de duração);
- ferramentas que mais falham (taxa de falha, mínimo de 1 falha).

`summarize(requests, tool_calls)` e `format_summary(...)` são puras (testadas sem banco e
sem Telegram). O handler só faz `_guard` → consulta (em thread) → formata → responde.

## Logs estruturados

`JARVIS_LOG_FORMAT=text` (padrão — o formato atual, inalterado) ou `json` (uma linha JSON
por evento: `ts`, `level`, `logger`, `msg`, `request_id`, e campos extras como
`latency_s`, `tool`, `outcome`). `JARVIS_LOG_LEVEL` (padrão `INFO`). O `request_id` vem
do `contextvar` e é o mesmo gravado na tabela `metrics`, então dá para ir do `/metrics` ao
log e voltar.

Eventos novos, com dados só numéricos/nomes: `requisição concluída` (INFO, um por pedido)
e `ferramenta executada` (DEBUG, uma por chamada).

## Configuração nova (`.env.example`)

```
JARVIS_METRICS_ENABLED=true
JARVIS_METRICS_DB_PATH=        # padrão: data/metrics.sqlite3
JARVIS_LOG_FORMAT=text         # text | json
JARVIS_LOG_LEVEL=INFO
```

## Correção junto: host do Ollama

`OLLAMA_URL` padrão passa de `http://localhost:11434` para `http://127.0.0.1:11434`
(pendência 3 do subprojeto 1: no Windows cada conexão com `localhost` custava ~2 s). Commit
próprio.

## Testes (sem Ollama)

- gravação e leitura no `MetricsStore` (inclui tabela filha e janela de tempo);
- `summarize`: p50/p95, taxa de erro, ranking e falhas de ferramentas, janela vazia;
- `format_summary`: conteúdo e janela vazia;
- agente instrumentado com LLM roteirizado: tokens somados, iterações, ferramentas com
  sucesso e falha, desfechos `error` e `max_iterations`;
- gravador que levanta exceção → resposta do agente idêntica;
- `chat_full` com transporte HTTP falso alimenta o rastro;
- formatter JSON e propagação do `request_id` para threads (`run_blocking`).
