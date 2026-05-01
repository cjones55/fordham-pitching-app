#!/bin/zsh
set -euo pipefail

REPO_DIR="/Users/chrisjones/Documents/Codex/2026-04-30/github-com-cjones55-fordham-pitching-app"
SERVICE_NAME="fordham-trackman-ftp"
ACCOUNT_NAME="Fordham"
LOG_DIR="$REPO_DIR/logs"
PYTHON_BIN="/Users/chrisjones/anaconda3/bin/python3"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

PASSWORD="$(security find-generic-password -a "$ACCOUNT_NAME" -s "$SERVICE_NAME" -w 2>/dev/null || true)"
if [[ -z "$PASSWORD" ]]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') Missing TrackMan password in macOS Keychain for service $SERVICE_NAME." >&2
  exit 1
fi

export FTP_PASSWORD="$PASSWORD"

"$PYTHON_BIN" scripts/import_trackman_2026.py \
  --protocol ftp \
  --port 21 \
  --host ftp.trackmanbaseball.com \
  --username Fordham \
  --remote-dir v3/2026 \
  --month 01 \
  --month 02 \
  --month 03 \
  --month 04 \
  --month 05 \
  --month 06 \
  --month 07 \
  --month 08 \
  --month 09 \
  --month 10 \
  --month 11 \
  --month 12 \
  --timeout 600 \
  --skip-existing
