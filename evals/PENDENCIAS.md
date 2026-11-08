# Pendências e limitações conhecidas

Registro honesto do que não funcionou, do que foi achado e não corrigido, e do que a
avaliação não mede. Nada aqui foi escondido da tabela de resultados.

## Bugs/achados no código do Jarvis (não corrigidos nesta etapa)

1. **`run_shell` não funciona no Windows nativo.** `jarvis/tools/system.py` chama
   `subprocess.run(..., shell=True, executable="/bin/bash")`. No Windows o caminho
   `/bin/bash` não existe (`WinError 3`), e mesmo com um bash válido o Python montaria
   `bash /c "comando"` (sintaxe do `cmd.exe`). O projeto foi escrito para Linux/macOS e o
   comportamento lá não foi alterado. O `scripts/smoke_test.py` pula só a checagem
   "run_shell executa" quando `os.name == "nt"` (o bloqueio de `sudo` continua testado).
   Correção sugerida: no Windows, usar `shell=True` sem `executable` (cmd.exe) ou procurar
   o bash do Git com `shutil.which("bash")` e chamá-lo com `["bash", "-lc", comando]`.

2. **[RESOLVIDO no subprojeto 5b, commit `fix(security): bloqueia variações de rm recursivo
   forçado`]** O regex foi trocado por uma análise do comando (segmentos, `shlex`, flags
   curtas/longas/abreviadas em qualquer ordem, prefixos como `env`/`nohup`, `sh -c`), e o
   bloqueio vale para `rm` recursivo — com ou sem `-f` — em `/`, `/*`, `~`, `$HOME`, `~outro`
   e diretórios de sistema de primeiro nível. Os três `xfail` viraram testes normais, com 35
   casos que devem bloquear e 17 que devem passar (`rm -rf build`, `git rm -r --cached /`...).
   Continua **não** sendo um sandbox: `find / -delete`, `python -c "import shutil..."`,
   variáveis montadas em tempo de execução (`X=/; rm -rf $X`) e `rm -rf .` dentro da home
   passam. Registro original: `rm -fr /`, `rm -r -f /` e `rm --recursive --force /` não
   eram bloqueados (o regex só reconhecia `-rf` com o `r` antes do `f`).

3. **[RESOLVIDO no subprojeto 3, commit `fix(config): usa 127.0.0.1 como host padrão do
   Ollama`]** O padrão do `OLLAMA_URL` (em `jarvis/config.py`, `.env.example` e na tabela
   do README) agora é `http://127.0.0.1:11434`, equivalente no Linux/macOS. Quem já tem um
   `.env` com `localhost` precisa trocar à mão. Continua aberto: o cliente HTTP ainda não é
   persistente (um `AsyncClient` por chamada). Registro original:
   **`localhost` custa ~2 s por conexão no Windows.** Medido em 08/11/2026 nesta máquina:
   uma chamada de embedding com `httpx` levou 2,16–2,20 s via `http://localhost:11434` e
   0,09–0,14 s via `http://127.0.0.1:11434` (o cliente tenta IPv6 `::1` antes do IPv4).
   Indexar o corpus de teste caiu de 40,0 s para 2,7 s só com essa troca. O
   `OllamaClient` cria um `httpx.AsyncClient` novo a cada chamada e o `RagIndex` usa
   `httpx.post` avulso, então o bot paga isso em toda mensagem se o `.env` usar
   `localhost` (o `.env.example` usa). Os evals usam `127.0.0.1` por padrão. Sugestões:
   recomendar `OLLAMA_URL=http://127.0.0.1:11434` no Windows e/ou reaproveitar um cliente
   HTTP persistente.

4. **Caminhos do Windows quebram o JSON no modo texto.** Na rodada de 08/11, 8 das 24
   respostas inválidas do `llama3.2:3b` no modo JSON tinham caminhos como
   `<HOME>\AppData\Local\Temp\...` com barras sem escape (o system prompt mostra o workspace com barras
   invertidas e o modelo copia). Não acontece no modo nativo nem com caminhos POSIX. A taxa
   "JSON inválido" conta também JSON válido fora do protocolo (só os argumentos, sem o
   nome da ferramenta, ou `args` como lista), que é a maioria dos outros 16 casos.
   Possíveis melhorias: mostrar caminhos com `/` no prompt e tolerar `{"action": ...}`
   sem `tool`; as duas mudam o comportamento do agente e ficaram para depois.

5. **O fallback JSON ignora a chave `parameters`.** `extract_text_tool` aceita
   `tool`/`name` e `args`/`arguments`; um modelo que responde
   `{"name": "x", "parameters": {...}}` (formato comum em modelos Llama) tem a ferramenta
   reconhecida, mas os argumentos viram `{}`. Comportamento fixado em teste
   (`test_extract_text_tool_parameters_nao_e_reconhecido`) e mantido sem mudança. Na rodada
   de 08/11 nenhuma resposta usou `parameters`, então não afetou os números.

6. **[RESOLVIDO no subprojeto 4: prefixos medidos e ligados por padrão, +3 perguntas em
   recall@1 no vetorial; ver a seção RAG abaixo]** **Embeddings sem os prefixos recomendados.** O `nomic-embed-text` foi treinado com os
   prefixos `search_query:` e `search_document:`; o `RagIndex` não os usa. Pode haver ganho
   de recall, mas não foi medido — é um bom próximo experimento (a avaliação já está pronta
   para comparar).

## Limitações da avaliação

- **Uma execução por configuração.** Seed fixo (42) e temperatura 0,2 (a do agente), sem
  repetições: não há intervalo de confiança. Com 78 casos, 1 caso = 1,3 ponto percentual;
  diferenças de poucos pontos entre modelos não são conclusivas.
- **Latência de um único notebook** (RTX 4050 Laptop 6 GB, Windows 11). Os números servem
  para comparar modelos entre si, não como garantia de desempenho em outra máquina.
- **Dataset escrito por uma pessoa**, sem anotação dupla. Os casos ambíguos aceitam mais de
  uma ferramenta; outra pessoa poderia discordar de algum gabarito.
- **Só a primeira decisão.** Não avaliamos conversas de vários turnos, a execução das
  ferramentas, nem a qualidade da resposta final em texto.
- **Corpus de RAG pequeno** (13 documentos, 25 trechos): recall@5 é otimista (no subprojeto 4
  o BM25 chegou a 100%, o que diz mais sobre o tamanho do corpus do que sobre o BM25).
- **Data relativa nos lembretes**: o system prompt real inclui a data/hora atual, então
  pedidos como "amanhã às 9h" aceitam tanto `amanhã 09:00` quanto uma data ISO com 09:00.

## Ambiente e histórico

- No Windows 11, `platform.platform()` ainda relata "Windows-10-10.0.26100"; o metadado
  dos resultados mostra isso.
- Instalar o MLflow 2.17.2 (última versão com Python 3.8) rebaixou o pacote `packaging`
  de 26.2 para 24.2 no `.venv`. Por isso ele fica em `requirements-evals.txt`, separado do
  bot.
- Histórico: no commit `5b1464c` um `git add` falhou no meio (a regra `data/` do
  `.gitignore` também pegava `evals/data`) e os fixtures entraram junto com o código; os
  datasets entraram no commit seguinte, `3eaadd4`, que também corrige o `.gitignore`. O
  histórico não foi reescrito.

## Observabilidade (subprojeto 3)

O que a instrumentação mede, o que não mede e as decisões que merecem revisão. A
especificação está em `docs/superpowers/specs/2026-11-08-observabilidade-design.md`.

1. **Falha de ferramenta é uma heurística.** As ferramentas devolvem o erro em texto em vez
   de levantar exceção. Conta como falha: exceção, ferramenta inexistente e texto que começa
   com `Erro`. Respostas como "Não é um arquivo: ...", "Informe o campo 'path'." ou
   "Notion não configurado" contam como **sucesso**. O certo seria as ferramentas devolverem
   um resultado estruturado (`ok`, `texto`); isso muda o contrato das 25 ferramentas e ficou
   para depois.
2. **Latência é a do agente, não a percebida.** `latency_s` cobre o `Agent.respond`. Fica de
   fora: download e transcrição de áudio (Whisper), extração de PDF/DOCX/OCR antes do agente,
   o envio da resposta ao Telegram e o TTS. Tempo até o primeiro token também não existe:
   o cliente usa `stream: false`.
3. **Tokens de prompt vêm do Ollama sem conferência.** Não verifiquei se o
   `prompt_eval_count` desconta o prefixo reaproveitado do cache KV entre chamadas da mesma
   conversa. Na demo os números batem com o prompt inteiro (~950 tokens na 1ª chamada,
   ~1.050 na 2ª, com o resultado da ferramenta), mas isso não foi testado em conversas
   longas.
4. **Uma chamada ao LLM que falha conta em `llm_calls`.** Inclusive a primeira tentativa
   que recebe `ToolsNotSupported` e faz o agente cair para o modo JSON (ela também consome
   uma iteração, como já acontecia antes). É intencional (mostra o custo real), mas pode
   surpreender quem compara `llm_calls` com `iterations`.
5. **Sem retenção.** As tabelas crescem para sempre (cada requisição ocupa algumas centenas
   de bytes; uso pessoal leva anos para pesar). Não há limpeza automática nem
   `/metrics reset`; zerar é apagar `data/metrics.sqlite3`.
6. **O `/metrics` mistura origens e modelos.** A coluna `source` separa `chat` de
   `briefing` (e da demo), e `model` registra o modelo, mas o resumo junta tudo. Se o
   modelo for trocado no meio da semana, os percentis de 7 dias misturam os dois (a linha
   "Modelos:" avisa quando há mais de um).
7. **`request_id` só existe onde há agente.** Mensagens de texto, áudio e documento passam
   pelo `_run_agent` e ganham id (inclusive os logs do envio da resposta). Comandos como
   `/status`, `/reindex`, `/remind`, os disparos de lembrete e o agendador do briefing
   (fora do `respond`) logam com `request_id: null`. O log da transcrição do áudio também
   fica sem id, porque acontece antes do `_run_agent`.
8. **Gravação síncrona no event loop.** `record()` roda na mesma thread do bot: 0,2 ms p50
   e 0,5 ms p95 medidos (300 gravações, Windows, WAL), com picos de ~10 ms quando o SQLite
   faz checkpoint do WAL. Aceitável para um bot pessoal; com mais carga, o caminho é uma
   fila com uma thread gravadora.
9. **`synchronous=NORMAL` pode perder as últimas métricas** numa queda de energia (não
   corrompe o banco). Para telemetria a troca vale; para o histórico de conversa o `Memory`
   continua com o padrão do SQLite.
10. **A demo é pequena.** `scripts/demo_metrics.py` roda 5 pedidos uma vez, só com
    ferramentas de arquivo numa pasta temporária. Serve para mostrar o pipeline de ponta a
    ponta com o modelo real, não como medida de desempenho: com n = 5 o p95 é ilustrativo, e
    o primeiro pedido inclui carregar o modelo na GPU.
11. **Não testei o `/metrics` num Telegram de verdade.** O handler é testado com o
    `JarvisBot` real e um `Update` falso (autorizado, não autorizado, métricas
    desligadas). A formatação usa bloco ``` do Markdown legado do Telegram; se o envio com
    Markdown falhar, o `_reply` já reenvia em texto puro, mas a aparência no app não foi
    conferida.
12. **Quem já tem `.env` com `OLLAMA_URL=http://localhost:11434`** continua pagando os ~2 s
    por conexão no Windows: a troca do padrão não reescreve `.env` existente.

## RAG (subprojeto 4)

A especificação está em `docs/superpowers/specs/2026-11-08-rag-hibrido-design.md`; a tabela,
em `evals/results/RAG.md`.

1. **O dataset favorece a busca por palavras.** Em média 65% das palavras de conteúdo de cada
   pergunta (tokens do `fts_query`, sem stopwords) aparecem no trecho certo, e só 1 das 36
   perguntas não tem nenhuma em comum (conta feita à parte, sobre os mesmos fixtures). As
   perguntas foram escritas por quem conhecia os documentos. O BM25 ganhou com folga aqui, mas
   isso **não** prova que ganha nas notas reais; para saber, faltam perguntas escritas por outra
   pessoa ou tiradas do uso real (com paráfrases e sinônimos) — de preferência antes de olhar os
   resultados de novo.
2. **Escolha do padrão em 18 perguntas.** A regra (maior MRR na metade dev) foi fixada antes de
   medir e a metade de teste confirmou, mas com 18 perguntas por metade 1 pergunta = 5,6 pontos.
   A diferença BM25 × híbrido com prefixos é de 4 perguntas em recall@1 no total.
3. **Decisões que não foram ajustadas, de propósito:** lista de stopwords (escrita antes de
   medir, mas pela mesma pessoa que leu os resultados), top-20 candidatos no híbrido, k=60 do
   RRF, tamanho dos trechos. RRF com pesos e reranker não foram testados: com 36 perguntas,
   qualquer peso escolhido aqui seria sobreajuste. A sensibilidade do k mostra que, sem
   prefixos, k=10 daria MRR 0,815 contra 0,773 — não adotado.
4. **FTS5 sem stemming em português.** O tokenizador `unicode61` tira acento e caixa, mas
   "reserva" não casa com "reservas". O FTS5 só traz stemmer de inglês (porter); não testei
   alternativas.
5. **A citação depende do modelo, e o `qwen2.5:3b` é fraco nisso.** Na demo
   (`scripts/demo_rag.py`, 3 perguntas, uma rodada): com a pergunta neutra ("Qual o valor
   mensal que eu pago de aluguel?") ele não chamou o `knowledge_search` em nenhuma das 3; com
   "nos meus documentos"/"procura nas minhas notas" chamou em 1 de 3. Quando chamou, citou o
   arquivo certo, mas o trecho errado: copiou o exemplo da instrução, que usa o 1º resultado
   (trecho 2; a resposta estava no trecho 1). Com a instrução no começo da saída, em vez do
   fim, ele não citou nada. n = 3 é demonstração, não medida: falta um eval de citação (com
   gabarito de arquivo+trecho por resposta, ou um juiz) e um eval de "decidir buscar" com
   perguntas neutras.
6. **"Fontes consultadas" não são "fontes usadas".** Quando o modelo usa o RAG e não cita, o
   agente anexa os 3 primeiros resultados da busca. Pode listar um trecho que o modelo ignorou.
   A detecção de "já citou" é o texto `fonte:` em qualquer lugar da resposta.
7. **O system prompt mudou (uma linha pedindo citação).** Só refiz o eval de ferramentas do
   `qwen2.5:3b` nativo: 84,6% / 82,0%, igual ao de antes, com 7 casos trocando de previsão
   (ganhos e perdas se anulam; não dá para separar efeito do prompt de não determinismo da
   GPU). Os outros modelos/modos do `RESULTS.md` não foram refeitos.
8. **Embeddings continuam sendo gerados na indexação, mesmo no modo `bm25`.** Assim trocar de
   modo é instantâneo, mas a indexação continua dependendo do Ollama. Não há opção para pular.
9. **O runner do Ollama falhou no meio da grade** duas vezes em três tentativas
   ("dial tcp 127.0.0.1:<porta>", depois de ~7 rodadas seguidas). O eval refaz a configuração;
   o `RagIndex` do bot **não** tem retry, então um `/reindex` pode terminar em "Erro na
   indexação" e precisar ser repetido. As grades interrompidas ficaram no MLflow (runs pais
   incompletos); a que vale é a do commit `8954b5f`, cujos JSON estão em `results/`.
10. **Latência é de uma rodada.** As linhas vetoriais variaram de 98 a 146 ms entre rodadas
    iguais; diferenças de dezenas de ms entre elas são ruído. O que é robusto: BM25 ~1 ms
    contra ~100 ms de qualquer modo que embede a pergunta.
11. **sqlite-vec só foi testado no Windows.** A versão 0.1.9 faz busca exata (sem ANN), o que
    basta para dezenas de milhares de trechos. A queda para NumPy foi testada simulando a falta
    do pacote (pytest), não numa plataforma real sem `enable_load_extension` (ex.: o Python do
    sistema no macOS).
12. **Janela de transição ao trocar modelo/prefixo.** Até o próximo `/reindex`, as consultas
    vetoriais usam o prefixo novo contra embeddings antigos (qualidade pior). Se a dimensão do
    embedding mudar, a tabela `chunks_vec` é recriada e os documentos antigos saem da busca
    vetorial até serem reindexados (a busca em NumPy já se comportava assim).


## Fine-tuning (subprojeto 2)

A especificação está em `docs/superpowers/specs/2026-11-08-finetune-design.md`; o passo a
passo, em `finetune/README.md`. **O que foi preparado está testado; o treino e a medição do
antes/depois NÃO foram feitos.**

1. **[PENDENTE] Treinar no Colab.** Por pedido explícito, nada foi treinado nesta máquina
   (nem pesos baixados do Hugging Face, nem torch/transformers/unsloth instalados). O
   `finetune/train_lora.ipynb` foi validado só como arquivo (`json.load`, `nbformat` 5.10.4,
   código compilado com `ast`), **nunca executado**: a instalação do Unsloth no Colab, os
   nomes de parâmetro do `TrainingArguments` (`eval_strategy`) e o nome do arquivo que o
   `save_pretrained_gguf` gera mudam entre versões e podem pedir ajuste na primeira rodada.
   O tempo de treino na T4 não foi medido (~13 milhões de tokens em 2 épocas).
2. **[PENDENTE] Medir o antes/depois** com
   `evals/compare_models.py qwen2.5:1.5b jarvis-qwen-lora --skip-rag --markdown evals/results/FINETUNE.md`
   e só então preencher a linha do modelo ajustado no README. A linha de base do 1,5B
   (56,4% / 42,6%) foi medida antes de o subprojeto 4 acrescentar uma linha ao system prompt;
   por isso a base tem de ser medida de novo na mesma execução.
3. **Os modelos de frase foram escritos por quem tinha lido o eval.** Os filtros de vazamento
   pegam cópia (exata), paráfrase próxima (Jaccard de 4-gramas ≥ 0,35; 60 removidos) e
   entidades do eval (0), mas não "ensinar para a prova" no nível da política — ex.:
   "anotações -> `knowledge_search`", "agenda do Google -> `google`", que coincidem com
   gabaritos do eval (e com o system prompt). Um ganho no eval de 78 casos pode ser em parte
   isso. O teste honesto de generalização é um conjunto novo, escrito por outra pessoa sem
   ver os modelos de frase, de preferência com pedidos reais do Telegram.
4. **O limiar de 0,35 é uma escolha**, com regra fixada antes (1,5 × a maior similaridade
   entre casos distintos do eval) e sensibilidade registrada (0,30 removeria 122; 0,50, só 7).
   O mais parecido que passou tem 0,349 ("pesquisa na internet: notícias sobre a taxa Selic"
   × "pesquisa na internet as notícias de hoje sobre inteligência artificial"): mesmo começo
   de frase, pedido diferente.
5. **Achado: o Ollama 0.40.14 manda as ferramentas do `qwen2.5` como struct do Go, não
   JSON** (`{{ .Function }}` imprime `{nome descrição {object <nil> <nil> [obrigatórios]
   {properties}}}`). Afeta a linha de base de todos os `qwen2.5` do subprojeto 1. O treino
   reproduz o formato (conferido por `finetune/check_template.py`: mesmos tokens e mesma
   saída em 6 conversas). Dois desdobramentos: (a) se uma atualização do Ollama mudar isso, o
   modelo ajustado passa a ver outro prompt — rodar o `check_template.py` depois de cada
   atualização; (b) **experimento barato e não feito:** um Modelfile com as ferramentas em
   JSON (`{{ json .Function }}` ou equivalente) para a base, medido com o mesmo eval — pode
   render pontos sem treino nenhum. Não testei se a função `json` existe nos templates desta
   versão.
6. **A paráfrase via Ollama ficou desligada no dataset versionado.** Funciona (`--paraphrase-frac`),
   mas o `qwen2.5:3b` produziu erros de português e, uma vez, caracteres chineses no meio
   da frase (agora descartados por filtro). O dataset é 100% de modelos de frase, com a
   variedade limitada a 334 formas de pedir e aos pools de entidades.
7. **Respostas sem ferramenta e de continuação são fixas e curtas.** O modelo pode aprender
   o estilo delas (ex.: "Pronto! Salvei em ..."); como a perda inclui o texto, isso também é
   treinado. As saídas de ferramenta das continuações são inventadas no formato real das 6
   ferramentas cobertas, não executadas.
8. **O val é sintético e da mesma família.** A divisão por modelo de frase evita que a mesma
   frase apareça nos dois lados, mas entidades e estilo são compartilhados; a perda no val e
   a "conferência de sanidade" do notebook medem ajuste à distribuição sintética, não o
   desempenho no Jarvis. O número que vale é o `eval_tools.py` via Ollama.
9. **Contexto de 4.608 tokens no treino, 8.192 no serviço.** Todo exemplo cabe (o notebook
   confere), mas conversas longas com histórico no Telegram ficam fora da distribuição de
   treino (só primeira decisão e uma continuação).
10. **Versionamento do dataset ligado ao código.** Testes falham se o `tools.json` não for
    igual ao registro atual ou se o system prompt dos exemplos não for o do `Agent` atual
    (fora data/hora e pasta). Nos dois casos, regere os dados (e treine de novo) antes de
    comparar números.


## Acabamento (subprojeto 5b)

A especificação está em `docs/superpowers/specs/2026-11-08-acabamento-design.md`.

1. **[DECISÃO DA JÚLIA] Commits antigos da branch ainda têm caminhos com o usuário do
   Windows.** Os JSON de `evals/results/` gravados antes do `evals/sanitize.py` traziam a
   home da máquina (o caminho da pasta temporária, que o modelo copiava nas respostas, e
   variações do nome inventadas por ele). No HEAD eles foram limpos e toda gravação nova
   passa pelo sanitizador, mas **o histórico não foi reescrito**: os commits dos subprojetos
   1 a 4 (a partir de `a122b5e`) ainda contêm esses caminhos, e o comentário do
   `scripts/smoke_test.py` com o nome curto 8.3 também existe em commits antigos. Antes de
   publicar o repositório, dá para fazer squash da branch `vitrine-ai` (ou `git filter-repo`
   só nesses arquivos); isso fica para ela decidir. Nenhum segredo estava nesses arquivos —
   só o nome da conta.
2. **O sanitizador é por padrão, não por lista fechada.** Troca o nome do usuário pelo
   prefixo (até 10 caracteres) seguido de qualquer continuação, a forma 8.3 e qualquer
   `X:` + `\Users\<alguém>`. Um nome que o modelo invente sem o mesmo começo (ex.: só o
   sobrenome) passaria. Os dados do fine-tuning (`finetune/data/`) usam homes fictícias
   (`/home/julia` e o equivalente no Windows) e não passam pelo sanitizador, de propósito:
   são texto de treino, e o teste do system prompt depende deles.
