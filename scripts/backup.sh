#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEST="${JARVIS_BACKUP_DIR:-$HOME/jarvis-backups}"
mkdir -p "$DEST"
chmod 700 "$DEST"

STAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE="$DEST/jarvis-$STAMP.tar.gz"

FILES=()
[ -f .env ] && FILES+=(".env")
[ -d data ] && FILES+=("data")
[ -f credentials.json ] && FILES+=("credentials.json")
[ -f token.json ] && FILES+=("token.json")

if [ ${#FILES[@]} -eq 0 ]; then
  echo "Nada para backup." >&2
  exit 1
fi

tar -czf "$ARCHIVE" \
  --exclude='data/*.log' \
  --exclude='data/uploads' \
  "${FILES[@]}"

chmod 600 "$ARCHIVE"
find "$DEST" -maxdepth 1 -name 'jarvis-*.tar.gz' -mtime +30 -delete

echo "Backup criado: $ARCHIVE"
