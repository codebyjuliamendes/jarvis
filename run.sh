#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Ambiente .venv não encontrado. Rode ./setup.sh primeiro." >&2
  exit 1
fi

# Ollama instalado no modo usuário (sem sudo)
export PATH="$HOME/.local/bin:$PATH"
if [ -d "$HOME/.local/lib/ollama" ]; then
  export OLLAMA_LIBRARY_PATH="$HOME/.local/lib/ollama"
fi

mkdir -p data
if ! curl -sf -m 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
  if command -v ollama >/dev/null 2>&1; then
    echo ">> Ollama offline; iniciando em background (data/ollama.log)"
    nohup ollama serve >> data/ollama.log 2>&1 &
    sleep 2
  fi
fi

# shellcheck disable=SC1091
. .venv/bin/activate
exec python -m jarvis
