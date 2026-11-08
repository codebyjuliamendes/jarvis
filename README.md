# Jarvis pessoal (Telegram + Ollama)

[![CI](https://github.com/codebyjuliamendes/jarvis/actions/workflows/ci.yml/badge.svg)](https://github.com/codebyjuliamendes/jarvis/actions/workflows/ci.yml)
[![Licença: MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://docs.astral.sh/ruff/)

Assistente pessoal com **LLM local** (Ollama) que conversa pelo **Telegram** e age com **25
ferramentas reais** — terminal, arquivos, git/GitHub, Google, Microsoft 365, Notion, RAG nas
suas notas. Medido, não suposto: o `qwen2.5:3b` escolhe a ferramenta certa em **82,1%** de 78
pedidos em pt-BR (85,5% na média de 3 rodadas, IC 95% de 77,8% a 92,3%; 0,58 s por
decisão numa GPU de notebook), e trocar a busca vetorial pela
busca por palavras (BM25) levou o recall@1 do RAG de **44,4% para 83,3%**. Observabilidade
por requisição (`/metrics`), LoRA preparado com checagem de vazamento, **servidor MCP** que
leva as mesmas ferramentas (e as mesmas travas) ao Claude Desktop/Code, **441 testes** sem
Ollama, CI e Docker.

![Demonstração: servidor MCP só leitura, avaliação com IC 95% e McNemar, testes](docs/demo.gif)

<sub>Gravado a partir da saída real dos comandos (o `--check` do MCP, a tabela gravada em `evals/results/REPEATS.md` e o pytest).</sub>

| Seção | Destaque |
|---|---|
| [Avaliação](#avaliação) | 78 casos de seleção de ferramentas (3 rodadas, IC 95%, McNemar) e 36+36 de RAG (original e paráfrases), MLflow, JSON de cada caso versionado |
| [Fine-tuning](#fine-tuning) | dataset sintético de 2.134 conversas, template idêntico ao do Ollama; treino no Colab **pendente** |
| [RAG](#rag) | BM25 (FTS5), vetorial (sqlite-vec) e híbrido (RRF), com citação de fonte |
| [Observabilidade](#observabilidade) | latência, tokens e ferramentas por requisição; logs JSON com `request_id` |
| [Servidor MCP](#servidor-mcp) | ferramentas do Jarvis no Claude Desktop/Code por stdio; só leitura por padrão, mesmas checagens do `security.py` |
| [Engenharia](#engenharia) | testes, ruff, GitHub Actions, Docker, segurança |

<!-- GIF de demonstração: ver "Gravar o GIF de demonstração" em Engenharia. Quando existir:
![Jarvis no Telegram](docs/demo.gif)
-->

O Jarvis roda **na sua máquina**: a inferência é local e o Telegram é só a interface. Além
das ferramentas, lê PDFs/DOCX/imagens (com OCR), transcreve áudios, responde em voz e envia um
briefing diário pela manhã.

> Você escolheu rodar no seu computador. Isso significa: privacidade total e custo zero
> de API, mas o Jarvis só responde quando o computador estiver ligado e com internet.

---

## O que ele faz

- Conversa em português pelo Telegram (texto ou áudio transcrito localmente).
- Lê, cria e edita arquivos dentro das pastas permitidas.
- Roda comandos no terminal (testes, builds, scripts) com bloqueio de comandos destrutivos.
- Usa `git` e consulta/cria issues e comentários no GitHub via API.
- Lê, busca e escreve notas no Obsidian.
- Busca, lê, anexa e cria páginas no Notion.
- Lista/cria eventos na Google Agenda; busca, lê e envia e-mails pelo Gmail.
- Acessa Microsoft 365: Outlook (e-mails/calendário) e Teams (equipes, canais e chats).
- Responde em áudio com voz natural (edge-tts), sob demanda ou automático.
- Consulta saldo, receita, clientes, faturas e payment links do Stripe.
- Lê planilhas CSV/TSV/XLSX.
- Lê PDFs (com OCR em páginas escaneadas), DOCX e imagens.
- Mantém uma memória local (RAG) nas suas notas e documentos: busca por palavras (BM25),
  semântica ou híbrida, citando arquivo e trecho.
- Envia um briefing diário às 6h (agenda, e-mails, lembretes e prioridades).
- Busca e lê páginas na web (DuckDuckGo + extração de texto).
- Envia arquivos da máquina para o Telegram.
- Cria lembretes persistentes que chegam no Telegram na hora marcada.

Cada integração é opcional: se não estiver configurada, o Jarvis explica o que falta
em vez de inventar.

---

## Arquitetura

```mermaid
flowchart LR
    U([Você no Telegram]) <-->|long polling| B[Bot<br/>telegram_bot.py]
    B -->|texto, áudio transcrito,<br/>PDF/DOCX/imagem| A[Agente<br/>loop de ferramentas]
    A <-->|function calling nativo<br/>ou fallback JSON| O[(Ollama local<br/>qwen2.5:3b)]
    A --> T[25 ferramentas<br/>terminal · arquivos · git/GitHub<br/>Google · Microsoft 365 · Notion<br/>Obsidian · Stripe · web · lembretes]
    A --> K[knowledge_search<br/>RAG: BM25 · vetorial · híbrido]
    K --> R[(rag.sqlite3<br/>FTS5 + sqlite-vec)]
    K -.->|embeddings| O
    T --> S{{security.py<br/>pastas permitidas<br/>comandos bloqueados}}
    A --> M[(metrics.sqlite3<br/>latência · tokens · ferramentas)]
    M --> B
    E[evals/<br/>78 + 36 casos] -.->|mesmo parsing<br/>e system prompt| A
    E -.-> F[(MLflow local)]
```

- **Sem porta aberta / sem ngrok**: o bot só faz conexões de saída para o Telegram.
- Histórico por conversa em SQLite (`data/jarvis.sqlite3`); métricas num SQLite separado
  (`data/metrics.sqlite3`), resumidas pelo comando `/metrics`.
- Suporta function calling nativo do Ollama; se o modelo não suportar, cai
  automaticamente para um modo de ferramentas via JSON.
- As avaliações usam o mesmo system prompt, os mesmos schemas e o mesmo código de parsing do
  agente: o número medido é o do bot, não o de uma cópia.

---

## Avaliação

O Jarvis tem uma suíte de avaliação automatizada em [`evals/`](evals/README.md) que mede, com
dados, as duas decisões das quais tudo depende:

- **Seleção de ferramentas:** 78 pedidos em pt-BR cobrindo as 25 ferramentas reais, mais casos
  ambíguos e casos em que o certo é **não** chamar ferramenta. Uma chamada ao LLM por pedido,
  com o system prompt e os schemas reais do agente, sem executar nada; a resposta é
  interpretada pelo mesmo código de parsing do agente.
- **Recuperação (RAG):** 36 perguntas sobre um corpus fictício (.md, .txt, .pdf, .docx) com
  distratores, indexado com o `RagIndex` real.

Cada execução também fica registrada no **MLflow** (params, métricas, JSON e tabela como
artefatos) para comparar runs pela interface.

Resultados medidos em 08/11/2026 (commit `0fa0ad2`) num notebook com RTX 4050 Laptop (6 GB),
Ollama 0.40.14, um modelo por vez, seed 42 ([tabela completa, por categoria e comparação com a
rodada anterior](evals/results/RESULTS.md); os JSON com cada caso estão em
[`evals/results/`](evals/results/)):

| Modelo | Modo | Acurácia ferramenta | Acurácia args | JSON inválido | Latência p50 (s) | Latência p95 (s) | tokens/s |
|---|---|---|---|---|---|---|---|
| `qwen2.5:1.5b` | nativo | 56,4% (44/78) | 44,3% (27/61) | 0,0% | 0,40 | 0,93 | 118,7 |
| `qwen2.5:1.5b` | JSON | 47,4% (37/78) | 32,8% (20/61) | 9,0% | 0,40 | 1,04 | 115,4 |
| **`qwen2.5:3b`** | **nativo** | **82,1% (64/78)** | **80,3% (49/61)** | 0,0% | 0,58 | 1,12 | 69,9 |
| `qwen2.5:3b` | JSON | 47,4% (37/78) | 39,3% (24/61) | 5,1% | 0,69 | 1,81 | 66,9 |
| `llama3.2:3b` | nativo | 74,4% (58/78) | 65,6% (40/61) | 0,0% | 0,77 | 1,29 | 65,2 |
| `llama3.2:3b` | JSON | 53,8% (42/78) | 39,3% (24/61) | 20,5% | 0,81 | 1,56 | 61,3 |

| RAG (`nomic-embed-text`, 13 docs / 25 trechos) | recall@1 | recall@5 | MRR |
|---|---|---|---|
| Antes: só vetorial, sem prefixos | 44,4% (16/36) | 72,2% (26/36) | 0,544 |
| **Depois (padrão): BM25** — ver [RAG](#rag) | **83,3% (30/36)** | **100,0% (36/36)** | **0,912** |

**Análise.**

- **O padrão continua sendo `qwen2.5:3b` com function calling nativo.** Foi o melhor em
  ferramenta (82,1%) e em argumentos (80,3%; 94,2% quando acerta a ferramenta), com latência
  mediana de 0,58 s por decisão. O `llama3.2:3b`, de tamanho e latência parecidos, fica quase
  8 pontos atrás em ferramenta e quase 15 em argumentos, e chamou ferramenta em todas as 8 conversas
  em que o certo era só responder. O `qwen2.5:1.5b` gera tokens 1,7 vez mais rápido, mas em
  31 dos seus 34 erros nem chama ferramenta e responde em texto: perde quase 26 pontos, uma
  troca que não compensa.
- **O fallback JSON é muito pior em todos os modelos** (47–54%). Ele existe para não quebrar
  com modelos sem function calling, mas na prática esses modelos não deveriam ser usados no
  Jarvis. No `llama3.2:3b`, 16 dos 78 casos (20,5%) vieram em JSON que o parser não aproveita:
  3 por caminhos do Windows (`<HOME>\AppData\...`) sem escapar as barras e a maioria dos
  outros fora do protocolo (argumentos sem o nome da ferramenta, ou `args` como lista).
- **O ponto fraco do `qwen2.5:3b` é a conversa sem ferramenta:** chamou ferramenta em 3 de 8
  casos (ex.: `knowledge_search` para "quanto é 15% de 240?"). É o alvo do
  [fine-tuning](#fine-tuning), e a avaliação mede o efeito.
- **O RAG era o elo mais fraco** (recall@1 de 44,4%, fora do top-5 em 10 de 36). O
  subprojeto 4 mediu prefixos, BM25 e busca híbrida; o resultado e as ressalvas estão na
  seção [RAG](#rag).
- **Quanto uma rodada varia, e o que isso muda (3 rodadas por modelo).** Rodadas idênticas
  variam 1–2 casos no modo nativo (temperatura 0,2 e seed fixo não garantem a mesma saída na
  GPU), mas a incerteza maior é a amostra de 78 pedidos: o IC 95% tem ~15 pontos de largura.

  | Modelo (nativo) | Acertos nas 3 rodadas | Média | Desvio | IC 95% (bootstrap sobre os casos) |
  |---|---|---|---|---|
  | `qwen2.5:1.5b` | 43 / 42 / 42 | 54,3% | 0,7 p.p. | [44,0; 64,5%] |
  | **`qwen2.5:3b`** | **67 / 67 / 66** | **85,5%** | 0,7 p.p. | **[77,8; 92,3%]** |
  | `llama3.2:3b` | 56 / 58 / 58 | 73,5% | 1,5 p.p. | [63,7; 82,5%] |

  `qwen2.5:3b` × `llama3.2:3b`, pareado por caso: 13 casos só o qwen acerta, 4 só o llama;
  McNemar exato p = 0,049 e diferença de +12,0 pontos (IC 95% [+2,1; +22,2]). É evidência
  moderada a favor do qwen, **não** conclusiva: rodada a rodada o p vai de 0,019 a 0,096 e não
  passa na correção de Bonferroni para as 4 comparações (0,0125). O `qwen2.5:1.5b` perde dos
  dois com folga (p < 0,001 e 0,008). Detalhes, modo JSON e ressalvas em
  [`RESULTS.md`](evals/results/RESULTS.md#três-rodadas-por-modelo-variância-ic-95-e-teste-pareado-extras)
  e [`REPEATS.md`](evals/results/REPEATS.md). As latências valem para esta máquina.

Para reproduzir:

```bash
uv pip install --python .venv/Scripts/python.exe -r requirements-evals.txt   # MLflow (opcional) + pytest
.venv/Scripts/python.exe evals/compare_models.py qwen2.5:1.5b qwen2.5:3b llama3.2:3b --modes native,json
.venv/Scripts/python.exe evals/compare_models.py qwen2.5:1.5b qwen2.5:3b llama3.2:3b \
    --repeats 3 --skip-rag --markdown evals/results/REPEATS.md     # média, desvio, IC 95%, McNemar
.venv/Scripts/python.exe -m pytest -q    # testes unitários, sem Ollama
```

---

## Fine-tuning

Preparação de um **LoRA** para o `qwen2.5` acertar mais a escolha de ferramenta — inclusive
quando o certo é não chamar nenhuma. Tudo em [`finetune/`](finetune/README.md). **O treino
ainda não foi feito**: ele roda no Google Colab gratuito (T4), não nesta máquina, e o modelo
ajustado só ganha número depois de medido pelo mesmo eval da seção anterior.

O que está pronto e testado:

- **Dataset sintético em pt-BR** gerado a partir dos schemas e do system prompt **reais**:
  2.134 conversas (1.689 treino / 445 val), as 25 ferramentas com 50–65 exemplos cada,
  **22,4% sem ferramenta** (com 99 "negativos difíceis", como "o que faz o git stash?") e 5%
  de continuações (resultado da ferramenta -> resposta). Determinístico (seed 42); treino e
  val divididos por modelo de frase.
- **Checagem de vazamento** contra os 78 casos do eval: duplicata exata, quase-duplicata
  (Jaccard de 4-gramas ≥ 0,35, limiar = 1,5 × a maior similaridade entre casos distintos do
  eval) e entidades do eval. **60 exemplos removidos**; a checagem roda de novo no `pytest`.
- **Texto de treino idêntico ao da inferência.** O Ollama 0.40.14 não manda as ferramentas do
  `qwen2.5` em JSON (imprime a struct do Go); o `apply_chat_template` do Hugging Face teria
  treinado num formato que o modelo nunca vê. `finetune/chat_format.py` reproduz o template
  do Ollama e `finetune/check_template.py` prova: mesmos tokens de prompt e mesma saída em 6
  conversas.
- **Notebook do Colab** (Unsloth, LoRA r=16/alpha=32, perda só na resposta, avaliação no
  val, exportação GGUF q4_K_M) e **Modelfile** com o template oficial — validado num modelo
  temporário sobre o `qwen2.5:1.5b`.

| Modelo (modo nativo, 78 casos) | Acurácia ferramenta | Acurácia args | Sem ferramenta |
|---|---|---|---|
| `qwen2.5:1.5b` (linha de base, medida) | 56,4% (44/78) | 44,3% (27/61) | 100,0% (8/8) |
| `qwen2.5:3b` (linha de base, medida) | 82,1% (64/78) | 80,3% (49/61) | 62,5% (5/8) |
| `jarvis-qwen-lora` (1,5B + LoRA) | pendente (treino no Colab) | pendente | pendente |

Os dois modelos erram em sentidos opostos: o 1,5B quase não chama ferramenta (31 dos 34
erros), o 3B chama demais em conversa simples. O dataset ataca os dois lados. Ressalvas que
valem para quando houver número: os modelos de frase foram escritos por quem tinha lido o
eval (os filtros pegam cópia e paráfrase, não a política), e 1 caso = 1,3 ponto. Lista
completa em [`evals/PENDENCIAS.md`](evals/PENDENCIAS.md#fine-tuning-subprojeto-2).

**Experimento sem treino (resultado nulo).** O Ollama 0.40.14 mostra as ferramentas ao
`qwen2.5` como struct do Go, não JSON. Um Modelfile com `{{ json .Function }}` (conferido: o
prompt passa a ter exatamente o JSON esperado) deu 85,9% contra 85,5% do template oficial em 3
rodadas, McNemar p = 1,0: não mudou nada mensurável e não foi adotado. Ver
[`finetune/experimentos/`](finetune/experimentos/README.md).

Para reproduzir:

```bash
.venv/Scripts/python.exe finetune/generate_dataset.py          # dados + relatório de vazamento
.venv/Scripts/python.exe finetune/check_template.py            # renderizador = template do Ollama
# treino: finetune/train_lora.ipynb no Colab (T4) -> jarvis-qwen-lora.Q4_K_M.gguf
ollama create jarvis-qwen-lora -f finetune/Modelfile
.venv/Scripts/python.exe evals/compare_models.py qwen2.5:1.5b jarvis-qwen-lora --skip-rag --markdown evals/results/FINETUNE.md
```

---

## RAG

A memória local (`knowledge_search`) tem três modos de busca, escolhidos por
`JARVIS_RAG_MODE`, todos dentro do mesmo SQLite (`data/rag.sqlite3`):

- **`bm25`** (padrão): busca por palavras com FTS5, sem acento e sem stopwords. Não chama o
  Ollama na consulta.
- **`vector`**: cosseno entre embeddings do `nomic-embed-text`, com os prefixos
  `search_document:`/`search_query:` com que o modelo foi treinado. Usa a extensão
  [sqlite-vec](https://github.com/asg017/sqlite-vec) quando ela carrega; se não carregar,
  cai sozinha para NumPy e avisa no log.
- **`hybrid`**: as duas, combinadas por Reciprocal Rank Fusion (top-20 de cada,
  `JARVIS_RAG_RRF_K`, padrão 60). Se o Ollama cair, responde só com o BM25.

As respostas citam a fonte: a ferramenta devolve arquivo e trecho de cada resultado e pede
`(fonte: arquivo, trecho N)`. Especificação:
[`docs/superpowers/specs/2026-11-08-rag-hibrido-design.md`](docs/superpowers/specs/2026-11-08-rag-hibrido-design.md).

### Antes e depois (medido)

Mesmo eval do subprojeto 1 (36 perguntas, 13 documentos fictícios com distratores, 25
trechos), uma rodada por configuração em 08/11/2026, RTX 4050 Laptop, Ollama 0.40.14. Cada
linha muda uma coisa em relação a uma anterior. Tabela completa, sensibilidade do k do RRF e
links para os JSON (com o ranking de cada pergunta) em
[`evals/results/RAG.md`](evals/results/RAG.md); cada rodada também está no MLflow.

| Configuração | recall@1 | recall@5 | MRR | MRR dev (18) | MRR teste (18) | Latência média |
|---|---|---|---|---|---|---|
| vetorial NumPy, sem prefixos (**antes**) | 44,4% (16/36) | 72,2% (26/36) | 0,544 | 0,537 | 0,551 | 98 ms |
| vetorial sqlite-vec, sem prefixos | 44,4% (16/36) | 72,2% (26/36) | 0,544 | 0,537 | 0,551 | 102 ms |
| vetorial sqlite-vec + prefixos | 52,8% (19/36) | 80,6% (29/36) | 0,614 | 0,594 | 0,634 | 146 ms |
| **BM25 (padrão, depois)** | **83,3% (30/36)** | **100,0% (36/36)** | **0,912** | **0,907** | **0,917** | **1,2 ms** |
| híbrido (RRF k=60), sem prefixos | 63,9% (23/36) | 94,4% (34/36) | 0,773 | 0,759 | 0,787 | 114 ms |
| híbrido (RRF k=60) + prefixos | 72,2% (26/36) | 100,0% (36/36) | 0,845 | 0,838 | 0,852 | 98 ms |

Latência = `RagIndex.search` inteiro; nos modos com vetor, ~95 ms são o embedding da pergunta
no Ollama. A variação entre as linhas vetoriais (98–146 ms) é ruído entre rodadas, não efeito
dos prefixos.

**Análise.**

- **O BM25 sozinho ganhou de tudo, inclusive do híbrido.** Os dados explicam: em média 65%
  das palavras de conteúdo da pergunta aparecem no trecho certo (só 1 das 36 não tem nenhuma),
  e as perguntas falam de coisas com nome — "Falcon", "Alfama", "L'Hôpital", "Luminária
  Origami". É o terreno da busca lexical. O vetorial deixa 7 perguntas fora do top-5; o BM25,
  nenhuma.
- **O híbrido ficou no meio.** O RRF dá peso igual a uma perna forte (BM25, MRR 0,91) e a
  uma fraca (vetorial, 0,61): em relação ao BM25 ele melhorou 1 pergunta (rag-034) e piorou
  7. Um RRF com pesos ou um reranker poderiam ajudar, mas escolher pesos nestas 36 perguntas
  seria sobreajuste.
- **Os prefixos do `nomic-embed-text` valem a pena:** +3 perguntas em recall@1 no vetorial
  (16 → 19) e no híbrido (23 → 26). Ficam ligados (`JARVIS_RAG_PREFIXES=auto`).
- **sqlite-vec não muda o ranking** (cosseno exato nos dois caminhos; os rankings são
  idênticos) **nem a latência neste corpus**, que é dominada pelo Ollama. O ganho aparece com
  mais trechos: no microbenchmark só da busca (`evals/bench_vector_backend.py`, 768 dimensões,
  média de 30 consultas) foram 9,0 → 4,7 ms com 25 trechos, 385 → 10,7 ms com 1.000 e
  1.773 → 19,7 ms com 4.000 (o limite de uma rodada de indexação do bot).
- **Por que `bm25` é o padrão:** a regra foi fixada na spec antes de medir — maior MRR na
  metade dev (ids ímpares). BM25 teve 0,907 contra 0,838 do híbrido com prefixos; a metade de
  teste confirmou a ordem (0,917 contra 0,852). De quebra, a consulta fica ~80x mais rápida e
  não depende do Ollama. **Quando trocar para `hybrid`:** se as perguntas tiverem pouca palavra
  em comum com os arquivos (sinônimos, paráfrases, outra língua) — exatamente o que este
  dataset quase não tem.
- **Cuidado com sobreajuste.** As 36 perguntas foram escritas por uma pessoa que conhecia os
  documentos, o que favorece a sobreposição de palavras. Nada foi ajustado olhando pergunta por
  pergunta: chunking igual ao de antes; stopwords, top-20 do híbrido e k=60 escolhidos antes de
  medir; a sensibilidade do k (10/30/60/100) é só informativa — com prefixos não muda nada, sem
  prefixos k=10 seria melhor (0,815 contra 0,773), e não foi adotado. 1 pergunta = 2,8 pontos
  no total e 5,6 em cada metade: a diferença entre BM25 e híbrido com prefixos (4 perguntas em
  recall@1) é real neste corpus, mas não prova que o BM25 ganha nas notas reais de alguém.

**Paráfrases (extras).** Para testar a ressalva acima, 36 perguntas novas com os mesmos trechos
esperados e sinônimos no lugar das palavras do texto (sobreposição lexical de 7,9%, contra 65,0%
do conjunto original), com o critério de troca do padrão commitado **antes** de rodar:

| Configuração | MRR original | MRR paráfrases | Média (critério) | recall@1 paráfrases | recall@5 paráfrases |
|---|---|---|---|---|---|
| vetorial sqlite-vec, sem prefixos | 0,544 | 0,162 | 0,353 | 5,6% (2/36) | 36,1% (13/36) |
| vetorial sqlite-vec + prefixos | 0,614 | 0,191 | 0,402 | 5,6% (2/36) | 38,9% (14/36) |
| **bm25 (padrão)** | **0,912** | 0,283 | **0,598** | 19,4% (7/36) | 44,4% (16/36) |
| híbrido (RRF k=60), sem prefixos | 0,773 | 0,307 | 0,540 | 22,2% (8/36) | 47,2% (17/36) |
| híbrido (RRF k=60) + prefixos | 0,845 | **0,349** | 0,597 | 19,4% (7/36) | **63,9% (23/36)** |

Todos os modos caem muito (o `nomic-embed-text` é fraco com sinônimos em pt-BR). Pela regra
fixada antes — média do MRR nos dois conjuntos, empate de 0,03 a favor do atual — o **BM25
continua o padrão** (0,598 × 0,597). O híbrido com prefixos acha bem mais trechos no top-5
das paráfrases (23 × 16) sem perder no original; como o agente recebe 5 trechos, trocar para
`hybrid` é uma opção razoável, registrada como decisão pendente em
[`PENDENCIAS.md`](evals/PENDENCIAS.md#rag-subprojeto-4). Detalhes em
[`RAG.md`](evals/results/RAG.md#conjunto-de-paráfrases-extras).

### Citação das fontes (exemplo real)

`scripts/demo_rag.py` indexa os mesmos documentos fictícios e roda o agente real
(`qwen2.5:3b`, só com `knowledge_search`). Resposta real de 08/11/2026:

> **Procura nas minhas notas: até quando dá para cancelar o hotel em Alfama sem pagar nada?**
>
> (fonte: viagem_lisboa.md, trecho 2) O cancelamento do hotel Miradouro Azul, na Rua das
> Gaivotas, no bairro de Alfama, pode ser feito sem custo até 20 de fevereiro.

O fato e o arquivo estão certos; o trecho não (a data está no trecho 1 — o modelo copiou o
exemplo da instrução, que usa o 1º resultado). Na mesma demo, o `qwen2.5:3b` **não chamou a
ferramenta** nas outras duas perguntas e respondeu de cabeça; isso é a seleção de ferramentas,
não o RAG (ver [`evals/PENDENCIAS.md`](evals/PENDENCIAS.md)). Quando o modelo usa a busca e não
cita, o agente anexa `Fontes consultadas: arquivo (trechos ...)` com os 3 primeiros
resultados — "consultadas", porque o agente não sabe quais o modelo aproveitou.

### Índices existentes

Não precisa reindexar para atualizar: a tabela `chunks` (texto + embedding em JSON) continua
sendo a fonte da verdade, e os índices FTS5 e sqlite-vec são reconstruídos a partir dela ao
abrir. Como os prefixos mudam os embeddings, cada documento guarda a assinatura
`modelo|prefixo`; o próximo `/reindex` re-embeda só o que estiver com assinatura antiga (o
modo `bm25` já funciona antes disso). Sem FTS5 no SQLite, `bm25` vira `vector` com aviso; o
SQLite 3.46 do Python 3.8 do `.venv` tem FTS5.

Para reproduzir:

```bash
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # inclui sqlite-vec
.venv/Scripts/python.exe evals/eval_rag.py --grid                     # todas as configurações
.venv/Scripts/python.exe evals/eval_rag.py --mode hybrid --prefixes on -v
.venv/Scripts/python.exe evals/bench_vector_backend.py --sizes 25,1000,4000
.venv/Scripts/python.exe scripts/demo_rag.py --show-tool              # citação com o agente real
```

---

## Observabilidade

Avaliação diz se o modelo escolhe bem num dataset; observabilidade diz como o Jarvis se
comporta **no uso real**. Cada mensagem vira uma linha numa tabela SQLite, e o comando
`/metrics` no Telegram resume as últimas 24 h e os últimos 7 dias. A especificação está em
[`docs/superpowers/specs/2026-11-08-observabilidade-design.md`](docs/superpowers/specs/2026-11-08-observabilidade-design.md).

**O que é medido, por requisição** (sem gravar o texto do pedido, da resposta nem os
argumentos das ferramentas):

- latência total (relógio de parede do `Agent.respond`) e tempo gasto dentro do LLM;
- tokens de prompt e de resposta e vazão (tokens/s), tirados do que o próprio Ollama devolve
  (`prompt_eval_count`, `eval_count`, `eval_duration`), com o mesmo código das avaliações
  (`jarvis/metrics_utils.py`);
- iterações do laço do agente, modo (function calling nativo ou JSON) e desfecho: `ok`,
  `error` (Ollama fora do ar, HTTP de erro, exceção) ou `max_iterations` (não concluiu);
- cada ferramenta chamada: nome, duração e sucesso/falha. Conta como falha a exceção, a
  ferramenta inexistente e o texto que começa com `Erro` (as ferramentas do Jarvis devolvem o
  erro em texto para o modelo ler, em vez de levantar exceção).

**Esquema** (`data/metrics.sqlite3`, separado do histórico em `data/jarvis.sqlite3` para dar
para apagar ou compartilhar a telemetria sem expor conversas):

| Tabela | Colunas principais |
|---|---|
| `metrics` | `request_id` (PK), `ts`, `created_at`, `source` (chat/briefing/demo), `chat_id`, `model`, `mode`, `latency_s`, `llm_calls`, `llm_s`, `prompt_tokens`, `completion_tokens`, `eval_s`, `tokens_per_s`, `iterations`, `tool_calls`, `outcome`, `error` |
| `metric_tool_calls` | `request_id` (FK), `seq`, `tool`, `duration_s`, `ok`, `error` |

**Como é gravado.** Durante a requisição só se acumula em memória, num objeto guardado em
`contextvars` (cada conversa simultânea tem o seu, então os números não se misturam mesmo com
o cliente Ollama e o registro de ferramentas compartilhados). No fim, uma única transação. Com
WAL e `synchronous=NORMAL`, medido nesta máquina: **0,2 ms p50 / 0,5 ms p95 por requisição**
(eram 13 ms / 26 ms com o fsync padrão a cada commit). Falha ao gravar vira um `WARNING` no
log e a resposta segue igual, coberto por teste.

**Exemplo real.** Saída de `scripts/demo_metrics.py`, que roda o agente de verdade (mesmo
prompt, ferramentas e laço do bot, sem Telegram) com `qwen2.5:3b` em 5 pedidos, numa pasta e
num banco temporários. Medido em 08/11/2026 na RTX 4050 Laptop (6 GB), começando com o modelo
descarregado:

```
== Últimas 24 h ==
Requisições: 5 | erro: 0 | sem concluir: 0 | taxa de erro: 0%
Latência: p50 1.7 s | p95 3.6 s | máx 4.1 s
Por requisição: 1.8 iterações | 1813 tokens de prompt | 83 de resposta
Vazão do modelo: 69.2 tokens/s
Modelos: qwen2.5:3b
Ferramentas: 4 chamada(s), 1 falha(s)
Mais usadas:
  read_file          2x  p50    3 ms  falhas 1
  list_dir           1x  p50   15 ms  falhas 0
  write_file         1x  p50    3 ms  falhas 0
Mais falham:
  read_file        1/2 (50%)

== Últimos 7 dias ==
(igual: todas as requisições da demo são de agora)
```

Como ler: o máximo de 4,1 s é o primeiro pedido, que inclui carregar o modelo na GPU; os
outros ficaram entre 1,1 e 1,9 s com duas idas ao LLM cada (decidir a ferramenta e responder).
A falha do `read_file` é proposital: o 5º pedido pede um arquivo que não existe, e o modelo
respondeu corretamente que não o encontrou. Os ~1.000 tokens de prompt por chamada são quase
todos system prompt + schemas das ferramentas, o que aponta onde cortar latência. Com 5
requisições, o p95 é só ilustrativo.

```bash
.venv/Scripts/python.exe scripts/demo_metrics.py                    # qwen2.5:3b, logs em texto
.venv/Scripts/python.exe scripts/demo_metrics.py --log-format json  # logs JSON
```

**Logs estruturados.** `JARVIS_LOG_FORMAT=json` troca o formato de texto de sempre (que
continua o padrão) por uma linha JSON por evento. Toda linha tem o `request_id`, o mesmo da
tabela `metrics`, inclusive as que saem de dentro das ferramentas (o `run_blocking` copia o
contexto para a thread). Linha real da demo:

```json
{"ts": "2026-11-08T04:23:41.213+00:00", "level": "INFO", "logger": "jarvis.agent", "msg": "requisição concluída: ok em 1.88s (2 iterações, 1 ferramenta(s))", "request_id": "05bd83571b72", "source": "demo", "model": "qwen2.5:3b", "mode": "native", "outcome": "ok", "latency_s": 1.878363, "llm_calls": 2, "prompt_tokens": 2106, "completion_tokens": 117, "tokens_per_s": 73.32076644641192, "iterations": 2, "tool_calls": 1, "error": null}
```

Para seguir um pedido ou achar os lentos:

```bash
python -m jarvis 2> jarvis.log                                   # com JARVIS_LOG_FORMAT=json
jq -c 'select(.request_id == "05bd83571b72")' jarvis.log          # tudo de uma requisição
jq -c 'select(.msg | startswith("requisição concluída")) | select(.latency_s > 5)' jarvis.log
sqlite3 data/metrics.sqlite3 "select tool, count(*), sum(1-ok) from metric_tool_calls group by tool"
```

Com `JARVIS_LOG_LEVEL=DEBUG` sai também uma linha `ferramenta executada` por chamada, com
`tool`, `duration_s` e `ok`. Limitações e o que ficou de fora:
[`evals/PENDENCIAS.md`](evals/PENDENCIAS.md#observabilidade-subprojeto-3).

---

## Servidor MCP

O Jarvis também é um servidor [MCP](https://modelcontextprotocol.io): o módulo opcional
[`jarvis/mcp_server.py`](jarvis/mcp_server.py) expõe as ferramentas do bot por **stdio** para
Claude Desktop, Claude Code ou qualquer cliente MCP, com o SDK oficial em Python. Guia
completo, com a configuração dos clientes: [`docs/mcp.md`](docs/mcp.md).

- **Sem duplicar nada:** cada ferramenta vira um `Tool` MCP com o mesmo nome, descrição e
  JSON Schema do `ToolRegistry`, e `tools/call` vai pelo mesmo `ToolRegistry.execute` do
  agente — logo pelas mesmas checagens do `security.py` (pastas permitidas, arquivos de
  credenciais, comandos destrutivos, SSRF), pelo `audit.log` e pelas métricas
  (`source="mcp"`, aparecem no `/metrics`).
- **Só leitura local por padrão:** `read_file`, `list_dir`, `search_files`, `find_files`,
  `read_pdf`, `read_docx`, `read_image` e `knowledge_search` (RAG). `web_search` e `web_fetch`
  ficam **fora** do padrão: junto com `read_file`, um cliente vítima de prompt injection
  poderia mandar um arquivo local numa URL. Religar: `JARVIS_MCP_TOOLS=leitura,web_search,web_fetch`
  (são leitura, não precisam da segunda chave).
  O que escreve, executa, envia ou mexe em dinheiro exige **duas chaves**: o nome em
  `JARVIS_MCP_TOOLS` **e** `JARVIS_MCP_ALLOW_UNSAFE=true`. Ferramenta fora da allowlist não
  aparece e, se chamada, é recusada antes do registro.
- **Python:** o SDK `mcp` exige 3.10+, o bot segue em 3.8. O servidor tem dependência
  própria ([`requirements-mcp.txt`](requirements-mcp.txt)) e só importa o SDK quando sobe;
  a allowlist é testada em 3.8 e o protocolo (inclusive por stdio, com o cliente do SDK)
  num job de CI em 3.12.

```bash
uv venv --python 3.12 .venv-mcp
uv pip install --python .venv-mcp -r requirements.txt -r requirements-mcp.txt
.venv-mcp/bin/python -m jarvis.mcp_server --check     # o que fica exposto e por quê

# Claude Code (Claude Desktop: ver docs/mcp.md)
claude mcp add jarvis --env PYTHONPATH="$PWD" -- "$PWD/.venv-mcp/bin/python" -m jarvis.mcp_server
```

Exemplo real (cliente do SDK por stdio, `.env` padrão): `knowledge_search` devolve os trechos
com a fonte; `read_file` em `C:\Windows\win.ini` volta com `isError` e a mensagem do
`security.py` ("caminho fora das pastas permitidas"); `run_shell` volta "não está exposta
neste servidor MCP". Pendências: [`evals/PENDENCIAS.md`](evals/PENDENCIAS.md#mcp-subprojeto-6).

---

## Engenharia

O que sustenta os números acima e deixa o projeto fácil de rodar e de auditar.

- **Testes:** 441 testes `pytest` que rodam **sem Ollama** (LLM e embeddings falsos; o
  `conftest` aponta o Ollama para uma porta descartável): parsing das chamadas de ferramenta,
  segurança, RAG (BM25, vetorial, híbrido, queda para NumPy), métricas, avaliações, sanitizador,
  dataset do fine-tuning, o template do `qwen2.5` e o servidor MCP (434 em Python 3.8; os 7
  do protocolo MCP pedem o SDK, Python 3.10+). Mais o `scripts/smoke_test.py`, que exercita
  as ferramentas reais (arquivos, terminal, git, PDF/DOCX, lembretes, briefing, laço do agente)
  com um LLM roteirizado.
- **Lint:** [ruff](https://docs.astral.sh/ruff/) com alvo Python 3.8 (`E`, `W`, `F`, `I`, `B`,
  `UP`), configurado em [`pyproject.toml`](pyproject.toml) com cada exceção justificada.
- **CI:** [GitHub Actions](.github/workflows/ci.yml) em todo push e PR — ruff, `pytest` e o
  smoke test em Python 3.8 e 3.12 (`ubuntu-latest`). No Linux o smoke test executa o
  `run_shell` de verdade. Um job à parte em 3.12 instala o SDK do MCP e roda os testes do
  servidor, inclusive a integração por stdio.
- **Docker:** [`Dockerfile`](Dockerfile) (Python 3.11 slim, usuário sem privilégios, sem
  segredos na imagem) e [`docker-compose.yml`](docker-compose.yml) com o Jarvis e o Ollama
  oficial; GPU opcional. Ver [Rodar com Docker](#rodar-com-docker).
- **Segurança:** pastas permitidas e bloqueio de arquivos de credenciais, filtro de comandos
  destrutivos (inclusive `rm` recursivo em `/`, `~`, `$HOME` escrito de qualquer forma), SSRF
  básico no `web_fetch`, long polling sem porta aberta. Ver [Segurança](#segurança).
- **Reprodutibilidade:** seeds fixos, hash do dataset e commit em cada resultado, JSON com cada
  caso versionado, MLflow local. Os resultados passam por um sanitizador
  ([`evals/sanitize.py`](evals/sanitize.py)) que tira a home e o usuário da máquina antes de
  gravar.
- **Limitações documentadas:** o que não funcionou, o que não foi medido e as decisões discutíveis estão em
  [`evals/PENDENCIAS.md`](evals/PENDENCIAS.md); cada subprojeto tem uma especificação em
  [`docs/superpowers/specs/`](docs/superpowers/specs/).

```bash
.venv/bin/python -m pytest -q          # testes (Windows: .venv/Scripts/python.exe)
.venv/bin/python scripts/smoke_test.py # ferramentas reais com LLM roteirizado
.venv/bin/ruff check .                 # lint (pip install ruff==0.17.4)
```

### Gravar o GIF de demonstração

O espaço do GIF no topo está comentado até existir uma gravação real (nada de mockup). Roteiro
sugerido, ~20 s:

1. No Telegram Desktop (ou no celular espelhado), mande: *"lista os arquivos da pasta notas"*,
   depois *"procura nas minhas notas até quando dá para cancelar o hotel"* (mostra a citação
   da fonte) e por fim `/metrics`.
2. Grave só a janela do chat: no Windows, [ScreenToGif](https://www.screentogif.com/); no
   macOS/Linux, [Kap](https://getkap.co/) ou [Peek](https://github.com/phw/peek). 10–12 fps e
   ~720 px de largura bastam.
3. Reduza o tamanho (alvo < 5 MB): `gifsicle -O3 --lossy=80 --colors 128 demo.gif -o docs/demo.gif`.
4. Confira que não aparecem token do bot, ID do Telegram, nomes reais ou caminhos da sua
   máquina (use os documentos fictícios de `evals/fixtures/` como notas).
5. Tire o comentário em volta de `![Jarvis no Telegram](docs/demo.gif)` no topo deste README.

---

## Requisitos

- Linux ou macOS, Python **3.8+** — ou só Docker (ver [Rodar com Docker](#rodar-com-docker)).
- [Ollama](https://ollama.com) (o script instala sem `sudo` em `~/.local`).
- Conta no Telegram.
- 8 GB de RAM ou mais. **Sem GPU**, use modelos 3B; 7B funciona, mas é mais lento.
- Internet para o Telegram (a inferência é local).
- Para OCR: `tesseract-ocr` (e `tesseract-ocr-por` para português). Sem isso, PDFs
  digitais e DOCX continuam funcionando; só OCR de imagens/PDFs escaneados fica indisponível.

---

## Início rápido

### 1. Instalar dependências

```bash
cd ~/jarvis
./setup.sh
```

Isso cria o `.venv`, instala tudo e gera um `.env` a partir do `.env.example`.

Opcional (voz): instale a transcrição local de áudios depois:

```bash
./scripts/setup_voice.sh
```

O modelo do Whisper é baixado no primeiro áudio. Em CPU, `small` é um bom equilíbrio;
use `base` para mais velocidade ou `medium` para mais precisão.

### 2. Instalar o Ollama e baixar um modelo

```bash
# instala o servidor (sem sudo, ~1.5 GB) e baixa o modelo
./scripts/install_ollama.sh qwen2.5:3b
```

Se a máquina aguentar, `qwen2.5:7b` responde melhor (mais lento em CPU).
Outros modelos com suporte a ferramentas: `llama3.2:3b`, `llama3.1:8b`.

Para a memória/RAG, baixe também o modelo de embeddings:

```bash
ollama pull nomic-embed-text
```

O `install_ollama.sh` já tenta baixar esse modelo automaticamente (pule com
`JARVIS_SKIP_EMBEDDINGS=1`).

Confira:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

### 3. Criar o bot no Telegram

1. Abra o Telegram e fale com o **@BotFather**.
2. Envie `/newbot`, escolha nome e usuário.
3. Copie o **token**.
4. Edite o `.env`:

```dotenv
TELEGRAM_BOT_TOKEN=123456:ABC...
OLLAMA_MODEL=qwen2.5:3b
```

### 4. Rodar

```bash
./run.sh
```

No Telegram, mande `/start`.

Se aparecer **“Acesso ainda não configurado”**, ele mostra o seu ID numérico.
Copie para o `.env` e reinicie:

```dotenv
TELEGRAM_ALLOWED_USER_IDS=123456789
```

Pronto. Mande `/status` para ver modelo, Ollama e integrações.
E `/metrics` para latência, erros e ferramentas mais usadas (24 h e 7 dias).

---

## Rodar com Docker

Alternativa ao `setup.sh`: o Jarvis e o Ollama oficial em dois contêineres
([`docker-compose.yml`](docker-compose.yml)). A imagem do bot ([`Dockerfile`](Dockerfile)) é
Python 3.11 slim com tesseract e git, roda como usuário sem privilégios e **não leva
segredos**: o `.env` entra só em tempo de execução (`env_file`) e o `.dockerignore` deixa
`.env`, `credentials.json` e `token.json` fora do contexto de build.

```bash
cp .env.example .env              # preencha TELEGRAM_BOT_TOKEN e TELEGRAM_ALLOWED_USER_IDS
mkdir -p data workspace           # criadas por você, ficam graváveis pelo usuário do contêiner (uid 1000)
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:3b
docker compose exec ollama ollama pull nomic-embed-text
docker compose logs -f jarvis
```

- **O que fica onde:** `./data` → `/app/data` (histórico, métricas, índice do RAG, uploads,
  tokens do Google/Microsoft); `./workspace` (ou `JARVIS_HOST_WORKSPACE=/caminho`) →
  `/home/jarvis/workspace`, a única pasta que o Jarvis lê, escreve e indexa; os modelos ficam
  no volume `ollama-models`.
- **O compose sobrescreve** `OLLAMA_URL` (`http://ollama:11434`), `JARVIS_WORKSPACE`,
  `JARVIS_ALLOWED_ROOTS` e `JARVIS_DB_PATH`, então o mesmo `.env` serve para rodar fora do
  Docker. Para o Google, aponte `GOOGLE_CREDENTIALS`/`GOOGLE_TOKEN` para `/app/data/...`.
- **GPU NVIDIA (opcional)**, com o NVIDIA Container Toolkit instalado no host:
  `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`. Sem GPU, o
  Ollama roda na CPU; prefira `qwen2.5:3b`.
- O Ollama não publica porta no host (só o bot fala com ele). Para usá-lo de fora, descomente
  `ports` no compose (fica restrito a `127.0.0.1`).
- O terminal do agente (`run_shell`) roda dentro do contêiner, como usuário sem privilégios:
  uma camada a mais além do filtro de comandos. Voz (faster-whisper) não está na imagem.

> Conferido em 08/11/2026 (Docker 29.6.1): `docker compose build jarvis` constrói a imagem
> (739 MB, ~1 min), o smoke test **dentro do contêiner** passa inteiro (63 de 63, inclusive o
> `run_shell`, que no Windows é pulado), o processo roda como `uid=1000(jarvis)` e, sem
> `.env`, o bot sai com a mensagem de configuração. O serviço `ollama` (imagem oficial) e o
> `docker compose up` completo com o Telegram não foram executados aqui.

---

## Como usar nos seus projetos

Ajuste no `.env`:

```dotenv
# pastas que o Jarvis pode acessar (vírgula)
JARVIS_ALLOWED_ROOTS=/home/julinha/projetos,/home/julinha/faculdade

# pasta padrão quando você não especifica caminho
JARVIS_WORKSPACE=/home/julinha/projetos
```

Exemplos de pedidos:

- “entra no projeto `api-clientes`, roda os testes e me diz o que falhou”
- “faz um resumo do `git diff` do repositório `faculdade/tcc`”
- “cria uma branch `feat/login` e um commit com as mudanças atuais”
- “lista as issues abertas de `minhaorg/meurepo` e prioriza por label”
- “anota no Obsidian, em `Faculdade/TCC.md`, que a defesa é dia 20”
- “o que tenho na agenda amanhã?”
- “busca e-mails não lidos do professor nas últimas 48h”
- “quanto faturei no Stripe nos últimos 30 dias?”
- “lê `vendas.xlsx`, aba `Março`, e resume os 10 maiores valores”
- “pesquisa na web as notícias de hoje sobre a minha área e me manda um resumo”
- “me manda o arquivo `relatorio.pdf` que está na pasta `projetos`”
- “me lembra em 10 minutos de tomar água”
- “anota que amanhã às 9h tenho reunião com o cliente”

> Dica: para tarefas de código mais complexas, um modelo 7B ajuda bastante.
> Troque `OLLAMA_MODEL` e rode `ollama pull <modelo>`.

---

## Configuração (`.env`)

| Variável | Obrigatória | Descrição |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | sim | Token do @BotFather |
| `TELEGRAM_ALLOWED_USER_IDS` | sim | IDs autorizados, separados por vírgula |
| `OLLAMA_URL` | não | Padrão `http://127.0.0.1:11434` (no Windows, `localhost` custa ~2 s por conexão) |
| `OLLAMA_MODEL` | não | Padrão `qwen2.5:3b` |
| `OLLAMA_TIMEOUT` | não | Segundos de espera (padrão 300) |
| `OLLAMA_NUM_CTX` | não | Contexto (padrão 8192) |
| `JARVIS_ALLOWED_ROOTS` | não | Pastas permitidas (padrão: sua home) |
| `JARVIS_WORKSPACE` | não | Pasta padrão de trabalho |
| `JARVIS_SHELL_ENABLED` | não | `true`/`false` para liberar terminal |
| `JARVIS_SHELL_TIMEOUT` | não | Timeout de comandos em segundos |
| `JARVIS_MAX_ITERATIONS` | não | Máximo de passos do agente por mensagem |
| `JARVIS_HISTORY_LIMIT` | não | Mensagens lembradas por conversa |
| `JARVIS_OWNER_NAME` | não | Como o Jarvis te chama |
| `JARVIS_TIMEZONE` | não | Fuso usado no contexto e nos lembretes |
| `JARVIS_WHISPER_ENABLED` | não | `true`/`false` para transcrever áudios |
| `JARVIS_WHISPER_MODEL` | não | `tiny`, `base`, `small` (padrão), `medium` |
| `JARVIS_WHISPER_LANGUAGE` | não | Idioma da transcrição (padrão `pt`) |
| `JARVIS_BRIEFING_ENABLED` | não | `true`/`false` para o briefing diário |
| `JARVIS_BRIEFING_HOUR` | não | Hora do briefing (padrão `6`) |
| `JARVIS_BRIEFING_MINUTE` | não | Minuto do briefing (padrão `0`) |
| `JARVIS_BRIEFING_MAX_ITERATIONS` | não | Passos máximos do agente no briefing (padrão `4`) |
| `JARVIS_RAG_ENABLED` | não | `true`/`false` para a memória local |
| `JARVIS_RAG_MODEL` | não | Modelo de embeddings (padrão `nomic-embed-text`) |
| `JARVIS_RAG_ROOTS` | não | Pastas indexadas, separadas por vírgula |
| `JARVIS_RAG_EXTENSIONS` | não | Extensões indexadas (padrão `.md,.txt,.pdf,.docx,.csv,.xlsx`) |
| `JARVIS_RAG_MODE` | não | `bm25` (padrão), `vector` ou `hybrid` (ver [RAG](#rag)) |
| `JARVIS_RAG_RRF_K` | não | k do Reciprocal Rank Fusion no modo `hybrid` (padrão `60`) |
| `JARVIS_RAG_PREFIXES` | não | Prefixos do `nomic-embed-text`: `auto` (padrão), `true`, `false` |
| `JARVIS_RAG_VECTOR_BACKEND` | não | `auto` (padrão: sqlite-vec se carregar, senão NumPy), `sqlite-vec`, `numpy` |
| `OBSIDIAN_VAULT` | não | Caminho do vault |
| `NOTION_TOKEN` | não | Token da integração do Notion |
| `NOTION_DEFAULT_PAGE` | não | Página pai padrão |
| `STRIPE_API_KEY` | não | Chave restrita (rk_) ou de teste (sk_test_) |
| `GITHUB_TOKEN` | não | Token com escopo `repo` |
| `GITHUB_DEFAULT_REPO` | não | `owner/repo` padrão |
| `GOOGLE_CREDENTIALS` | não | `credentials.json` do Google Cloud |
| `GOOGLE_TOKEN` | não | `token.json` gerado pelo script |
| `MS_GRAPH_CLIENT_ID` | não | Application (client) ID do app no Azure |
| `MS_GRAPH_TENANT` | não | Directory (tenant) ID ou `organizations` |
| `MS_GRAPH_TOKEN` | não | Cache de token (padrão `data/ms_graph_token.json`) |
| `MS_GRAPH_SCOPES` | não | Escopos Graph separados por vírgula |
| `JARVIS_TTS_ENABLED` | não | `true`/`false` para voz de saída |
| `JARVIS_TTS_AUTO` | não | Falar automaticamente todas as respostas |
| `JARVIS_TTS_VOICE` | não | Voz edge-tts (padrão `pt-BR-FranciscaNeural`) |
| `JARVIS_TTS_RATE` | não | Velocidade, ex.: `+0%`, `+15%`, `-10%` |
| `JARVIS_TTS_MAX_CHARS` | não | Limite de caracteres por áudio (padrão 3000) |
| `JARVIS_METRICS_ENABLED` | não | `true` (padrão) grava métricas por requisição; `/metrics` mostra o resumo |
| `JARVIS_METRICS_DB_PATH` | não | Banco das métricas (padrão `data/metrics.sqlite3`) |
| `JARVIS_LOG_FORMAT` | não | `text` (padrão, formato de sempre) ou `json` (uma linha JSON com `request_id`) |
| `JARVIS_LOG_LEVEL` | não | `INFO` (padrão), `DEBUG`, `WARNING`... |

---

## Integrações

### GitHub

1. Crie um token clássico em <https://github.com/settings/tokens> com escopo `repo`.
2. `.env`: `GITHUB_TOKEN=ghp_...` e, se quiser, `GITHUB_DEFAULT_REPO=owner/repo`.

> O `gh` do snap está quebrado nesta máquina (glibc), por isso o Jarvis usa a API REST
> diretamente. Não precisa do `gh`.

### Notion

1. Crie uma integração em <https://www.notion.so/my-integrations>.
2. Copie o “Internal Integration Secret” para `NOTION_TOKEN`.
3. No Notion, abra a página > `...` > **Connections** > adicione a integração.
4. Opcional: `NOTION_DEFAULT_PAGE=<id da página>`.

### Obsidian

```dotenv
OBSIDIAN_VAULT=/home/julinha/Documentos/MeuVault
```

### Google Agenda e Gmail

1. No <https://console.cloud.google.com/>, crie um projeto.
2. Ative **Google Calendar API** e **Gmail API**.
3. Configure a tela de consentimento OAuth (Externo, modo teste; adicione seu e-mail
   como usuário de teste).
4. Crie credenciais > **ID do cliente OAuth** > **Aplicativo para computador**.
5. Baixe o JSON e salve como `~/jarvis/credentials.json`.
6. Rode uma vez:

```bash
.venv/bin/python scripts/google_auth.py
```

7. Autorize no navegador. O `token.json` é criado e renovado automaticamente.

### Microsoft 365 (Outlook e Teams)

Funciona com conta **trabalho/escola** (Microsoft 365). Usa o Microsoft Graph com
device code, então não precisa de servidor nem redirecionamento público.

1. No <https://portal.azure.com/>, entre em **Microsoft Entra ID** >
   **Registros de aplicativo** > **Novo registro**.
2. Nome: `Jarvis`. Tipo de conta: **somente esta organização** (ou multilocatário,
   se quiser usar em várias empresas).
3. Em **Autenticação**, habilite **Fluxos de cliente público** = **Sim**
   (necessário para device code).
4. Em **Permissões de API** > **Microsoft Graph** > **Permissões delegadas**, adicione:
   `User.Read`, `Mail.Read`, `Mail.Send`, `Calendars.ReadWrite`, `Chat.Read`,
   `Team.ReadBasic.All`, `Channel.ReadBasic.All`.
5. Clique em **Conceder consentimento do administrador** (se você não for admin,
   peça ao admin do tenant). Sem isso, o Teams pode retornar 403.
6. Copie o **Application (client) ID** e o **Directory (tenant) ID** para o `.env`:

```dotenv
MS_GRAPH_CLIENT_ID=00000000-0000-0000-0000-000000000000
MS_GRAPH_TENANT=00000000-0000-0000-0000-000000000000
```

7. Autorize uma vez:

```bash
.venv/bin/python scripts/ms_graph_auth.py
```

Ele mostra um código e um link (`microsoft.com/devicelogin`). Depois disso, o token
fica em `data/ms_graph_token.json` e é renovado sozinho.

Além de listar/criar eventos e ler/enviar e-mails, o Jarvis também marca e-mails como
lidos/não lidos e remove eventos (pedindo confirmação conforme o bom senso).

> Se o tenant bloquear o consentimento do Teams, remova os escopos de Teams de
> `MS_GRAPH_SCOPES` e rode o script de novo. Outlook e calendário continuam funcionando.
> Leitura de mensagens de **canais** do Teams normalmente exige
> `ChannelMessage.Read.All` com consentimento do admin.

### Stripe

```dotenv
STRIPE_API_KEY=rk_test_...
```

Prefira chave restrita (`rk_`) ou de teste (`sk_test_`). O Jarvis avisa quando a chave
é `live`.

### Planilhas

Basta apontar o caminho no pedido (dentro das pastas permitidas). Ex.: `vendas.xlsx`.

### Voz (transcrição local)

```bash
./scripts/setup_voice.sh
```

Depois é só mandar um áudio no Telegram. O Jarvis transcreve com Whisper local
(`faster-whisper`), mostra o que entendeu e responde como se fosse texto.

- `JARVIS_WHISPER_MODEL=small` (padrão) — bom equilíbrio em CPU.
- `JARVIS_WHISPER_LANGUAGE=pt` — idioma principal.
- Para desligar: `JARVIS_WHISPER_ENABLED=false`.

O áudio é baixado, transcrito e apagado da máquina em seguida. Nada é enviado para a nuvem.

### Voz de saída (TTS com edge-tts)

O Jarvis pode responder em áudio usando o `edge-tts` (vozes neurais da Microsoft).
Como esse serviço é online, **o texto da resposta é enviado à Microsoft** para gerar
o áudio. O resto continua local.

```text
/voice status   # mostra se o áudio automático está ligado e qual voz
/voice on       # toda resposta também vira áudio
/voice off      # só texto (padrão)
```

Você também pode pedir “manda isso em áudio” ou “fala isso” e o agente usa a ferramenta
`speak`. O padrão é **sob demanda** (`JARVIS_TTS_AUTO=false`).

```dotenv
JARVIS_TTS_ENABLED=true
JARVIS_TTS_AUTO=false
JARVIS_TTS_VOICE=pt-BR-FranciscaNeural
JARVIS_TTS_RATE=+0%
```

Outras vozes em português: `pt-BR-AntonioNeural`, `pt-PT-FranciscaNeural`.
Para desligar: `JARVIS_TTS_ENABLED=false`.

### Lembretes

```text
/remind 10m tomar água
/remind 2h ligar para o cliente
/remind hoje 18:00 revisar o TCC
/remind amanhã 09:00 reunião
/remind 2026-11-20 18:30 prova
/reminders
```

Os lembretes ficam no SQLite (`data/jarvis.sqlite3`), sobrevivem a reinícios e são
enviados pelo próprio bot no horário marcado. O fuso usado é `JARVIS_TIMEZONE`.
O computador e o Jarvis precisam estar ligados na hora do lembrete.

### Briefing diário

Todo dia às **6h** (configurável), o Jarvis gera e envia um resumo com prioridades,
agenda, e-mails importantes, Teams, lembretes e próximo passo. Ele usa Google
Agenda/Gmail, Microsoft 365 (Outlook/Teams), Obsidian, Notion, lembretes e, se precisar,
busca na web. Integrações não configuradas são apenas ignoradas.

```text
/briefing    # gera o briefing de hoje na hora
```

```dotenv
JARVIS_BRIEFING_ENABLED=true
JARVIS_BRIEFING_HOUR=6
JARVIS_BRIEFING_MINUTE=0
```

O computador precisa estar ligado às 6h. Se o horário passar com a máquina desligada,
o agendador tenta executar quando o serviço voltar (margem de 1 hora).

### Memória local (RAG)

O Jarvis indexa suas notas e documentos e consulta por palavras (BM25, padrão), por
significado ou pelas duas juntas (ver a seção [RAG](#rag), com a medição de cada modo). Os
embeddings são gerados **localmente** pelo Ollama (`nomic-embed-text`).

- Pastas padrão: vault do Obsidian (se configurado), `~/Documentos`, `~/Downloads` e o
  próprio projeto. Ajuste com `JARVIS_RAG_ROOTS=/caminho/a,/caminho/b`.
- Indexa `.md`, `.txt`, `.pdf`, `.docx`, `.csv`, `.xlsx` (configurável).
- A indexação é incremental: arquivos sem mudança são ignorados.
- Arquivos removidos/movidos saem do índice na próxima indexação completa.
- Documentos enviados pelo Telegram são indexados automaticamente em segundo plano.
- Pastas/arquivos sensíveis (`.ssh`, `.config`, `.env`, credenciais, chaves) são pulados.
- Banco separado: `data/rag.sqlite3`.

```text
/reindex     # reindexa as pastas configuradas
```

Pela linha de comando:

```bash
.venv/bin/python scripts/reindex.py                 # incremental
.venv/bin/python scripts/reindex.py --force         # tudo de novo
.venv/bin/python scripts/reindex.py --path ~/Notas  # só uma pasta/arquivo
```

A primeira indexação pode demorar. O agente usa `knowledge_search` automaticamente
quando você pergunta sobre conteúdo antigo. Também há `knowledge_index` para indexar
uma pasta específica.

### PDF, DOCX e OCR

- `read_pdf`: extrai texto de PDFs; usa OCR automaticamente em páginas escaneadas.
- `read_docx`: extrai texto e tabelas de arquivos Word.
- `read_image`: faz OCR de prints/fotos.
- Você pode **enviar um PDF/DOCX/planilha/imagem direto no Telegram**: o arquivo é salvo
  em `data/uploads/`, lido e resumido. A legenda da mensagem vira o pedido.

OCR requer `tesseract` instalado. O resto funciona sem ele.

### Web e arquivos

- `web_search`: busca no DuckDuckGo e devolve títulos, links e trechos.
- `web_fetch`: baixa uma URL e extrai o texto principal (bloqueia localhost/rede interna).
- `send_file`: envia um arquivo das pastas permitidas para a conversa atual
  (limite do Telegram ~50 MB).

---

## Segurança

- **Só os IDs em `TELEGRAM_ALLOWED_USER_IDS` conseguem usar.** Qualquer outra pessoa
  recebe “acesso negado”. Se o token vazar, revogue no @BotFather (`/revoke`).
- **Arquivos** são restritos a `JARVIS_ALLOWED_ROOTS`. Pastas/arquivos de credenciais
  (`.ssh`, `.config`, `.git`, `.env`, `credentials.json`, `token.json`, chaves etc.)
  são bloqueados.
- **Terminal**: comandos perigosos (`sudo`, `mkfs`, `dd`, `shutdown`, fork bomb,
  `curl | sh` etc.) são bloqueados, e `git push --force`, `git clean -f` e
  `git config --global` também. O `rm` recursivo em alvo crítico (`/`, `/*`, `~`, `$HOME`,
  `/etc`, `/usr`...) é barrado em qualquer forma de escrever as flags (`-rf`, `-fr`,
  `-r -f`, `--recursive --force`, `/bin/rm`, dentro de `sh -c` ou depois de `&&`). Isso **não é um sandbox completo**: o terminal é
  poderoso. Se quiser desligar, use `JARVIS_SHELL_ENABLED=false`.
- `.env`, `credentials.json`, `token.json` e o banco SQLite estão no `.gitignore`.
- O bot usa **long polling**: não abre porta na sua rede.
- O **servidor MCP** (opcional) fala por stdio, sem porta, e expõe só ferramentas de leitura
  local por padrão (a web fica de fora, para não abrir caminho de exfiltração); as que escrevem/executam/enviam exigem `JARVIS_MCP_TOOLS` **e**
  `JARVIS_MCP_ALLOW_UNSAFE=true`, e passam pelas mesmas travas acima. Ver
  [`docs/mcp.md`](docs/mcp.md).
- O Jarvis pede confirmação antes de ações irreversíveis (o modelo é instruído a isso,
  mas os bloqueios acima são a rede de segurança).

---

## Deixar rodando (systemd de usuário)

```bash
mkdir -p ~/.config/systemd/user
cp ~/jarvis/scripts/jarvis.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now jarvis
systemctl --user status jarvis
journalctl --user -u jarvis -f
```

Para continuar rodando mesmo sem sessão aberta:

```bash
loginctl enable-linger "$USER"
```

O `run.sh` sobe o Ollama automaticamente se ele estiver offline.

### Backup e reindexação automáticos (opcional)

```bash
cp ~/jarvis/scripts/jarvis-backup.service ~/jarvis/scripts/jarvis-backup.timer \
   ~/jarvis/scripts/jarvis-reindex.service ~/jarvis/scripts/jarvis-reindex.timer \
   ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now jarvis-backup.timer jarvis-reindex.timer
```

- **Backup diário às 3:30** em `~/jarvis-backups/jarvis-AAAAmmdd-HHMMSS.tar.gz`
  (`.env`, `data/` exceto logs/uploads, credenciais; mantém 30 dias).
  Ajuste com `JARVIS_BACKUP_DIR`.
- **Reindexação diária às 4:00** da memória local.
- Rode um backup manual: `./scripts/backup.sh`.
- Restaurar: pare o Jarvis, extraia o `.tar.gz` em `~/jarvis` e suba de novo.

---

## Problemas comuns

| Sintoma | O que fazer |
|---|---|
| `TELEGRAM_BOT_TOKEN não configurado` | Preencha o token no `.env` |
| “Acesso ainda não configurado” | Copie o ID mostrado para `TELEGRAM_ALLOWED_USER_IDS` e reinicie |
| “Não consegui conectar no Ollama” | `./scripts/install_ollama.sh` ou `ollama serve`; veja `data/ollama.log` |
| “O modelo X não está baixado” | `ollama pull X` |
| Respostas muito lentas | Use `qwen2.5:3b`; aumente `OLLAMA_TIMEOUT` |
| Google 403 | Ative Calendar/Gmail API e rode `scripts/google_auth.py` de novo |
| Notion 404 | Compartilhe a página com a integração (Connections) |
| GitHub 401 | Gere outro token e atualize `GITHUB_TOKEN` |
| Briefing não chegou | `systemctl --user status jarvis`; máquina ligada às 6h; veja `journalctl` |
| RAG: “modelo não encontrado” | `ollama pull nomic-embed-text` |
| OCR não funciona | Instale `tesseract-ocr` e `tesseract-ocr-por` |
| TTS retorna 403 | Atualize `edge-tts` (`pip install -U edge-tts`); a Microsoft muda o token às vezes |
| Microsoft 403 no Teams | Peça consentimento admin ou ajuste `MS_GRAPH_SCOPES` |
| Bot não responde | Veja os logs do terminal ou `journalctl --user -u jarvis -f` |

Teste as ferramentas sem modelo:

```bash
.venv/bin/python scripts/smoke_test.py
```

Ele valida arquivos, terminal, git, web, lembretes, bloqueios de segurança, memória,
RAG, PDF/DOCX, briefing, Microsoft/TTS e o loop do agente (63 verificações no Linux; no
Windows a checagem "run_shell executa" é pulada, ver `evals/PENDENCIAS.md`).

Testes unitários (pytest, sem Ollama): `.venv/bin/python -m pytest -q`. Ver
[Engenharia](#engenharia).

---

## Estrutura do projeto

```
jarvis/
├── jarvis/
│   ├── __main__.py          # entrada
│   ├── config.py            # .env -> Settings
│   ├── security.py          # raízes permitidas, segredos, comandos perigosos
│   ├── memory.py            # histórico SQLite
│   ├── reminders.py         # lembretes persistentes + APScheduler
│   ├── briefing.py          # briefing diário às 6h
│   ├── rag.py               # memória local: BM25 (FTS5), vetorial (sqlite-vec/NumPy), híbrida
│   ├── documents.py         # extração de PDF/DOCX/XLSX/OCR
│   ├── voice.py             # transcrição local (faster-whisper)
│   ├── tts.py               # voz de saída (edge-tts)
│   ├── graph_auth.py        # token Microsoft Graph (MSAL)
│   ├── telegram_api.py      # envio de mensagens, arquivos e áudio
│   ├── llm.py               # cliente Ollama (+ fallback sem tools)
│   ├── agent.py             # loop do agente
│   ├── observability.py     # request_id e rastro por requisição (contextvars)
│   ├── metrics_store.py     # tabela de métricas (SQLite) e resumo do /metrics
│   ├── metrics_utils.py     # percentis e tokens/s (avaliações e métricas)
│   ├── logging_setup.py     # logs em texto ou JSON
│   ├── telegram_bot.py      # bot, autorização, comandos, voz, documentos
│   ├── mcp_server.py        # servidor MCP (stdio, opcional, Python 3.10+)
│   └── tools/
│       ├── system.py        # terminal e arquivos
│       ├── dev.py           # git e GitHub
│       ├── productivity.py  # Obsidian, Notion, planilhas
│       ├── google.py        # Calendar e Gmail
│       ├── microsoft.py     # Outlook, Calendar e Teams (Graph)
│       ├── business.py      # Stripe
│       ├── web.py           # busca e leitura de páginas
│       ├── telegram_tools.py# envio de arquivos
│       ├── knowledge.py     # busca/indexação da memória local + citação
│       ├── documents.py     # leitura de PDF/DOCX/imagem
│       ├── voice_tools.py   # ferramenta speak (TTS)
│       └── reminders.py     # ferramenta de lembretes
├── evals/                   # avaliação de ferramentas e RAG, sanitizador (ver evals/README.md)
├── finetune/                # LoRA: dataset, vazamento, template, notebook, Modelfile
├── scripts/
│   ├── install_ollama.sh
│   ├── google_auth.py
│   ├── ms_graph_auth.py
│   ├── setup_voice.sh
│   ├── backup.sh
│   ├── reindex.py
│   ├── smoke_test.py
│   ├── demo_metrics.py      # agente real + /metrics num banco temporário
│   ├── demo_rag.py          # agente real + citação de fontes do RAG
│   ├── jarvis.service
│   ├── jarvis-backup.service
│   ├── jarvis-backup.timer
│   ├── jarvis-reindex.service
│   └── jarvis-reindex.timer
├── tests/                   # pytest sem Ollama
├── docs/mcp.md              # servidor MCP: clientes, allowlist, segurança
├── docs/superpowers/specs/  # especificação de cada subprojeto
├── .github/workflows/ci.yml # ruff + pytest + smoke (3.8 e 3.12) + MCP (3.12)
├── Dockerfile
├── docker-compose.yml       # Jarvis + Ollama
├── docker-compose.gpu.yml   # GPU NVIDIA (opcional)
├── pyproject.toml           # configuração do ruff
├── setup.sh
├── run.sh
├── requirements.txt
├── requirements-voice.txt
├── requirements-evals.txt
├── requirements-mcp.txt     # SDK do MCP (Python 3.10+)
├── LICENSE
└── .env.example
```

---

## Próximos passos possíveis

- **WhatsApp**: API oficial (Meta Cloud API) ou bridge QR; exige decisão/conta.
- **Google Drive** e outras integrações (Trello, Linear, Slack, Mercado Pago).
- **Rodar o treino do LoRA no Colab** e medir o antes/depois (ver [Fine-tuning](#fine-tuning)).
- **Migrar para VPS** para ficar 24h no ar (basta copiar `.env`, `data/` e instalar Ollama
  ou trocar o cliente por uma API).

---

## Licença

[MIT](LICENSE) © 2026 Júlia Mendes.
