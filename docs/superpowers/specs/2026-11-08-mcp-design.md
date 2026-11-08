# Servidor MCP — especificação

Data: 2026-11-08 · Branch: `vitrine-ai` · Subprojeto 6

## Objetivo

Expor as ferramentas do Jarvis como um servidor [MCP](https://modelcontextprotocol.io)
(Model Context Protocol), para que qualquer cliente MCP — Claude Desktop, Claude Code, MCP
Inspector — use a mesma busca na memória local, leitura de arquivos e leitura da web que o
bot usa no Telegram, **com as mesmas travas de segurança** e sem reimplementar nada.

## Restrições

- O SDK oficial (`mcp`, hoje 2.x) exige **Python >= 3.10**; o bot continua em **3.8**.
  O servidor é um módulo opcional (`jarvis/mcp_server.py`) com dependência separada
  (`requirements-mcp.txt`). O SDK só é importado dentro de `build_server()`/`main()`, então
  `import jarvis.mcp_server` funciona em 3.8 e a política de exposição é testada lá também.
  O arquivo mantém sintaxe 3.8 (ruff com alvo `py38` continua valendo para ele).
- Desenvolvimento e testes do protocolo num venv à parte com Python 3.12 (`.venv-mcp/`,
  ignorado pelo git). Testes que precisam do SDK usam `pytest.importorskip("mcp")`.

## Desenho

1. **Reaproveitar o `ToolRegistry`.** `build_registry(settings, rag=...)` monta o mesmo
   registro do bot; cada ferramenta exposta vira um `Tool` MCP com o mesmo nome, descrição e
   JSON Schema (`parameters` → `inputSchema`). Servidor de baixo nível (`mcp.server.lowlevel.Server`)
   porque os schemas já existem — o `MCPServer`/FastMCP geraria schemas a partir de
   assinaturas Python, o que duplicaria a fonte da verdade.
2. **Mesmo caminho de execução.** `tools/call` vai para `ToolRegistry.execute`, que chama o
   handler da ferramenta, que chama `security.resolve_path` / `check_url` /
   `check_shell_command`. Nada de checagem paralela: um caminho fora de
   `JARVIS_ALLOWED_ROOTS` ou um arquivo de credenciais é recusado pelo mesmo código que
   protege o bot. O registro também grava o `audit.log` (agora com `"source": "mcp"`).
3. **Allowlist com padrão seguro (só leitura).**
   - Padrão: `read_file`, `list_dir`, `search_files`, `find_files`, `read_pdf`, `read_docx`,
     `read_image`, `knowledge_search`, `web_search`, `web_fetch` (este passa pelo
     `check_url`, que bloqueia SSRF).
   - `JARVIS_MCP_TOOLS` (lista separada por vírgula; aceita o atalho `leitura`) troca o
     conjunto — serve para **restringir** ou para pedir ferramentas que escrevem.
   - Ferramentas que escrevem, executam, enviam ou mexem em dinheiro (`write_file`,
     `replace_in_file`, `run_shell`, `git`, `github`, `obsidian`, `notion`, `spreadsheet`,
     `google`, `microsoft`, `stripe`, `knowledge_index`) só entram com **duas chaves**: o nome
     em `JARVIS_MCP_TOOLS` **e** `JARVIS_MCP_ALLOW_UNSAFE=true`. Com uma chave só, ficam de
     fora e o motivo vai para o log (stderr).
   - `send_file` e `speak` nunca são expostas: dependem de um chat do Telegram.
   - Nomes desconhecidos são ignorados com aviso. Ferramenta fora da allowlist não aparece
     em `tools/list` e, se chamada mesmo assim, é recusada **antes** do registro.
   - `python -m jarvis.mcp_server --check` mostra o que ficaria exposto e por quê, sem subir
     o servidor.
4. **Transporte stdio.** Nada pode escrever em stdout além do protocolo: logs vão para
   stderr (o `logging` já usa stderr) e o próprio SDK desvia o fd 1 enquanto serve.
   `main()` faz `chdir` para a raiz do repositório antes de carregar o `.env`, como o
   `run.sh`, para que caminhos relativos (`JARVIS_DB_PATH=data/...`) signifiquem o mesmo
   que no bot, qualquer que seja a pasta de onde o cliente lança o processo.
5. **Métricas.** Cada chamada abre um `trace_scope(source="mcp", model="")` e, com
   `JARVIS_METRICS_ENABLED`, é gravada no mesmo `metrics.sqlite3` — aparece no `/metrics`
   como requisição sem LLM (modelo vazio, 0 iterações, 1 ferramenta).

## Testes

- Python 3.8 (sem SDK): política de exposição (padrão, restrição, duas chaves, nomes
  desconhecidos, ferramentas do Telegram) e a ponte (`McpBridge`): ferramenta bloqueada não
  executa, caminho fora da raiz é recusado pelo `security.py`, métrica com `source="mcp"`.
- Python 3.12 (com SDK): cliente em processo (`mcp.Client(server)`) lista as ferramentas,
  chama `read_file`, `knowledge_search` (RAG BM25 real num índice temporário), recebe
  `isError` para caminho fora da raiz e para ferramenta não exposta; integração por **stdio**
  subindo `python -m jarvis.mcp_server` com o cliente do SDK.
- CI: job novo só em 3.12 que instala `requirements-mcp.txt` e roda os testes MCP.

## Critérios de aceite

- `.venv` (3.8): `pytest` sem regressão (373 + os novos de política/ponte), testes de
  protocolo pulados; `ruff check .` limpo; smoke 62 ok + 1 pulada.
- `.venv-mcp` (3.12): todos os testes, inclusive os MCP, passam.
- YAML do CI faz parse.
- Uma chamada real via cliente MCP contra o servidor com o `.env` da máquina.

## Fora de escopo

Transporte HTTP/SSE, autenticação, recursos e prompts MCP, editar a configuração real do
Claude Desktop/Code da máquina (só documentação), push.
