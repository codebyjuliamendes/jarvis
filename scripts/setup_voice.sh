#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "Ambiente .venv não encontrado. Rode ./setup.sh primeiro." >&2
  exit 1
fi

# shellcheck disable=SC1091
. .venv/bin/activate

echo ">> Instalando faster-whisper (transcrição local, CPU)"
pip install -r requirements-voice.txt

cat <<'EOF'

Pronto. O modelo será baixado no primeiro áudio recebido.
Padrão: JARVIS_WHISPER_MODEL=small (bom equilíbrio em português).
Para mais velocidade: JARVIS_WHISPER_MODEL=base
Para mais precisão:  JARVIS_WHISPER_MODEL=medium (bem mais lento em CPU)

Reinicie o Jarvis depois de instalar.
EOF
