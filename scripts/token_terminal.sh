#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

clear
echo "=============================================="
echo " Jarvis - configuracao do token do Telegram"
echo "=============================================="
echo
echo "Cole o token do @BotFather e aperte Enter."
echo "Ele nao aparece na tela enquanto voce cola."
echo

.venv/bin/python scripts/configure_telegram.py
status=$?

echo
if [ "$status" -eq 0 ]; then
  echo "Token salvo com sucesso."
  echo "Volte ao Jarvis e diga: pronto"
else
  echo "Nao foi possivel salvar o token. Tente de novo."
fi
echo
read -r -p "Pressione Enter para fechar esta janela..." _
