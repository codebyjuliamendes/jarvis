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

2. **O filtro de comandos destrutivos deixa passar variações de `rm -rf /`.**
   `rm -fr /`, `rm -r -f /` e `rm --recursive --force /` não são bloqueados: o regex só
   reconhece `-rf` com o `r` antes do `f`. Ficou como teste `xfail` estrito em
   `tests/test_security.py` (vira "XPASS" e falha quando o filtro for corrigido, para lembrar
   de promovê-lo). Não corrigi porque muda política de segurança e merece revisão própria.

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
   `C:\Users\...` com barras sem escape (o system prompt mostra o workspace com barras
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

6. **Embeddings sem os prefixos recomendados.** O `nomic-embed-text` foi treinado com os
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
- **Corpus de RAG pequeno** (13 documentos, 25 trechos): recall@5 é otimista.
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
