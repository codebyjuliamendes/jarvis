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
