#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo ">> Criando ambiente virtual (.venv) com $($PYTHON --version)"
"$PYTHON" -m venv .venv

# shellcheck disable=SC1091
. .venv/bin/activate

echo ">> Atualizando pip"
python -m pip install --upgrade pip >/dev/null

echo ">> Instalando dependências"
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo ">> Criei .env a partir do .env.example"
else
  echo ">> .env já existe, mantido"
fi

chmod +x run.sh scripts/*.sh 2>/dev/null || true

if ! command -v tesseract >/dev/null 2>&1; then
  cat <<'EOF'
>> Aviso: tesseract não encontrado. O OCR de PDFs/imagens ficará indisponível.
   Em Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-por
EOF
fi

cat <<'EOF'

Pronto para configurar:

1. Edite o .env e preencha TELEGRAM_BOT_TOKEN (via @BotFather).
2. Instale/rode o Ollama e baixe um modelo:
     ./scripts/install_ollama.sh qwen2.5:3b
   (em máquina sem GPU, qwen2.5:3b é o mais rápido)
3. Inicie o Jarvis:
     ./run.sh
4. No Telegram, mande /start. Se aparecer "acesso não configurado",
   copie o ID mostrado para TELEGRAM_ALLOWED_USER_IDS no .env e reinicie.

Opcional: teste as ferramentas sem modelo com
     .venv/bin/python scripts/smoke_test.py
EOF
