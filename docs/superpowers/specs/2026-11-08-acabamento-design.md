# Acabamento de engenharia — especificação

Data: 2026-11-08 · Branch: `vitrine-ai` · Subprojeto 5b

## Objetivo

Deixar o repositório pronto para ser lido por quem avalia o projeto: seguro onde já se sabia
que não era, sem dados pessoais nos artefatos versionados, com números coerentes com o código
atual, com lint e CI, empacotado em Docker e com um README que conta a história em um minuto
— sem prometer nada que não foi medido.

## Escopo (um ou mais commits pequenos por item)

1. **Segurança: variações de `rm` recursivo forçado.** O filtro de `jarvis/security.py` só
   reconhecia `-rf` com o `r` antes do `f`. Trocar o regex por uma análise do comando:
   dividir em segmentos (`;`, `&&`, `||`, `|`, quebra de linha), tokenizar com `shlex`,
   pular prefixos (`env`, `nice`, `nohup`, `time`, `command`, `builtin`, `exec`, `busybox`,
   atribuições `VAR=x`), reconhecer `rm` também por caminho (`/bin/rm`) e juntar as flags em
   qualquer forma (`-rf`, `-fr`, `-r -f`, `-R`, `--recursive`, `--force`, `--no-preserve-root`)
   até o `--`. Bloqueia quando é recursivo (com ou sem `-f`: `rm -r ~` apaga a home do mesmo jeito)
   **e** algum alvo é perigoso: `/`,
   `/*`, `~`, `~/`, `~/*`, `$HOME`, `${HOME}`, `$HOME/*`, `.`/`..`/`*` sozinhos não entram
   (seriam falso positivo demais para um bot que roda `rm -rf build`). Diretórios de sistema
   de primeiro nível (`/etc`, `/usr`, `/home`, `/boot`, `/var`, `/bin`...) também contam.
   `rm -rf build`, `rm -rf ./node_modules`, `rm -r pasta`, `rm -f /tmp/x` continuam passando.
   Os três testes `xfail` viram testes normais.
2. **Dados pessoais.** `evals/sanitize.py`: troca a home da máquina (qualquer separador e
   nível de escape), o nome do usuário (com e sem acento, `\u00e3`, truncado) e o nome curto
   8.3 (`ABCDEF~1`) por `<HOME>`/`<USER>`. Aplicado nos JSON versionados e chamado por
   `common.save_json`, de modo que toda gravação futura já sai limpa. O histórico da branch
   **não** é reescrito: fica registrado em `evals/PENDENCIAS.md`.
3. **Resultados coerentes com o HEAD.** Refazer `compare_models.py` com `qwen2.5:1.5b`,
   `qwen2.5:3b` e `llama3.2:3b`, nativo e JSON, mais o RAG no padrão atual, um modelo por vez,
   no commit atual. Atualizar `RESULTS.md` e o README com os números novos, data e commit.
   Mudanças grandes são explicadas, não escondidas.
4. **ruff** em `pyproject.toml`: alvo `py38`, regras `E`, `F`, `W`, `I`, `B`, `UP`, com os
   ignores justificados no próprio arquivo.
5. **CI** (`.github/workflows/ci.yml`): ruff + pytest em push e PR, Python 3.8 e 3.12,
   `ubuntu-latest`, sem Ollama. O smoke test entra se não depender do Ollama.
6. **Docker**: `Dockerfile` (python slim, usuário não-root, sem segredos), `docker-compose.yml`
   com o Jarvis e o `ollama/ollama` (volume de modelos, GPU opcional), `.dockerignore`.
   Nada é construído nem baixado nesta máquina; só validação de sintaxe.
7. **README**: badges, resumo com os números reais, diagrama Mermaid, seção Engenharia,
   espaço para o GIF de demonstração com instruções de gravação.
8. **Licença MIT** (Júlia Mendes, 2026).

## Critérios de aceite

- `pytest`: os 298 atuais continuam passando, os 3 `xfail` viram testes que passam, 0 `xfail`.
- `scripts/smoke_test.py`: 62 ok + 1 pulada (Windows), sem regressão.
- `ruff check .` limpo.
- `git grep -i "JuliaCristina\|JULIAC~1\|C:\\Users"` sem resultados no HEAD.
- Nenhum número no README que não esteja num arquivo de `evals/results/`.

## Fora de escopo

Reescrever histórico, push, rodar contêineres, treinar o LoRA, corrigir o `run_shell` no
Windows (registrado em `PENDENCIAS.md`).
