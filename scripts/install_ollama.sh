#!/usr/bin/env bash
set -euo pipefail

# Instala o Ollama no modo usuário (sem sudo) em ~/.local.
# Uso: ./scripts/install_ollama.sh [modelo]
#   ./scripts/install_ollama.sh                 -> instala só o servidor
#   ./scripts/install_ollama.sh qwen2.5:3b      -> instala e baixa o modelo

MODEL="${1:-}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$HOME/.local/bin"
LIB_DIR="$HOME/.local/lib/ollama"
TMP_ZST="/tmp/ollama-linux-amd64.tar.zst"
TMP_TAR="/tmp/ollama-linux-amd64.tar"

PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

mkdir -p "$BIN_DIR" "$HOME/.local/lib" "$HOME/.local/share/ollama" "$PROJECT_DIR/data"

if ! command -v ollama >/dev/null 2>&1 && [ ! -x "$BIN_DIR/ollama" ]; then
  echo ">> Baixando Ollama para $HOME/.local (sem sudo, ~1.5 GB)"

  # O link direto de release do GitHub tem retornado 504 em alguns ambientes.
  # A API entrega uma URL assinada que funciona; usamos ela quando possível.
  ASSET_URL="$("$PYTHON_BIN" - <<'PY'
import json
import urllib.request

request = urllib.request.Request(
    "https://api.github.com/repos/ollama/ollama/releases/latest",
    headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "jarvis-installer",
    },
)
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
except Exception:
    raise SystemExit(0)

for asset in release.get("assets", []):
    if asset.get("name") == "ollama-linux-amd64.tar.zst":
        print(asset.get("url", ""))
        break
PY
)"

  if [ -z "$ASSET_URL" ]; then
    ASSET_URL="https://ollama.com/download/ollama-linux-amd64.tar.zst"
  fi

  if ! curl -fL --progress-bar -H "Accept: application/octet-stream" \
    "$ASSET_URL" -o "$TMP_ZST"; then
    echo "Falha ao baixar o Ollama. Baixe manualmente em https://ollama.com/download" >&2
    exit 1
  fi

  if ! "$PYTHON_BIN" - "$TMP_ZST" <<'PY'
import sys

with open(sys.argv[1], "rb") as handle:
    raise SystemExit(0 if handle.read(4) == b"\x28\xb5\x2f\xfd" else 1)
PY
  then
    echo "O arquivo baixado não é um .tar.zst válido." >&2
    echo "Baixe manualmente em https://ollama.com/download e extraia em ~/.local." >&2
    rm -f "$TMP_ZST"
    exit 1
  fi

  echo ">> Descompactando .tar.zst"
  if command -v zstd >/dev/null 2>&1; then
    zstd -d -c "$TMP_ZST" > "$TMP_TAR"
  else
    if ! "$PYTHON_BIN" -c "import zstandard" >/dev/null 2>&1; then
      echo ">> Instalando módulo zstandard (necessário para descompactar)"
      "$PYTHON_BIN" -m pip install --quiet zstandard \
        || "$PYTHON_BIN" -m pip install --quiet --user zstandard
    fi
    "$PYTHON_BIN" - "$TMP_ZST" "$TMP_TAR" <<'PY'
import sys
import zstandard

src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as fin, open(dst, "wb") as fout:
    zstandard.ZstdDecompressor().copy_stream(fin, fout)
PY
  fi

  tar -C "$HOME/.local" -xf "$TMP_TAR"
  rm -f "$TMP_ZST" "$TMP_TAR"

  if [ -x "$HOME/.local/ollama" ] && [ ! -e "$BIN_DIR/ollama" ]; then
    ln -sf "$HOME/.local/ollama" "$BIN_DIR/ollama"
  fi
  if [ -d "$HOME/.local/lib/ollama" ]; then
    LIB_DIR="$HOME/.local/lib/ollama"
  fi
fi

export PATH="$BIN_DIR:$PATH"
if [ -d "$LIB_DIR" ]; then
  export OLLAMA_LIBRARY_PATH="$LIB_DIR"
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "O binário do Ollama não foi encontrado em $BIN_DIR." >&2
  echo "Baixe manualmente em https://ollama.com/download e extraia em ~/.local." >&2
  exit 1
fi

echo ">> Ollama: $(ollama --version 2>/dev/null || echo 'versão desconhecida')"

# Sobe o servidor se ainda não estiver respondendo
if ! curl -sf -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo ">> Iniciando 'ollama serve' em background (log: data/ollama.log)"
  nohup ollama serve >"$PROJECT_DIR/data/ollama.log" 2>&1 &
  sleep 3
fi

if curl -sf -m 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo ">> Servidor Ollama respondendo em http://localhost:11434"
else
  echo ">> Aviso: o servidor não respondeu. Veja data/ollama.log" >&2
fi

if [ -n "$MODEL" ]; then
  echo ">> Baixando modelo $MODEL (pode demorar e ocupar vários GB)"
  ollama pull "$MODEL"
fi

if [ "${JARVIS_SKIP_EMBEDDINGS:-0}" != "1" ]; then
  echo ">> Garantindo modelo de embeddings para a memória/RAG"
  ollama pull nomic-embed-text || {
    echo ">> Aviso: não consegui baixar nomic-embed-text. Rode: ollama pull nomic-embed-text" >&2
  }
fi

cat <<EOF

Para iniciar o servidor manualmente depois:
  export PATH="\$HOME/.local/bin:\$PATH"
  ollama serve

Modelos recomendados (sem GPU):
  ollama pull qwen2.5:3b    # mais rápido
  ollama pull qwen2.5:7b    # melhor qualidade, mais lento
EOF
