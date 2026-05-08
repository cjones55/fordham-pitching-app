#!/bin/zsh
# Daily TrackMan update — downloads new game files then pushes to GitHub
# so the Streamlit Cloud app picks them up automatically.
#
# Runs via macOS LaunchAgent every 24 hours.
# Install with: zsh scripts/setup_daily_trackman_update.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_NAME="fordham-trackman-ftp"
ACCOUNT_NAME="Fordham"
LOG_DIR="$REPO_DIR/logs"
PYTHON_BIN="/Users/chrisjones/anaconda3/bin/python3"
GIT_BIN="$(command -v git)"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

TS() { date '+%Y-%m-%d %H:%M:%S'; }

echo "$(TS) ── Daily TrackMan update starting ──────────────────────────"

# ── 1. Get TrackMan FTP password from macOS Keychain ─────────────────────────
PASSWORD="$(security find-generic-password -a "$ACCOUNT_NAME" -s "$SERVICE_NAME" -w 2>/dev/null || true)"
if [[ -z "$PASSWORD" ]]; then
  echo "$(TS) ERROR: TrackMan password not found in Keychain (service=$SERVICE_NAME)." >&2
  echo "Run: zsh scripts/setup_daily_trackman_update.sh to store it." >&2
  exit 1
fi
export FTP_PASSWORD="$PASSWORD"

# ── 2. Download new game files from TrackMan FTP ─────────────────────────────
echo "$(TS) Connecting to TrackMan FTP..."
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
echo "$(TS) FTP import complete."

# ── 3. Commit and push new Fordham game files to GitHub ──────────────────────
NEW_FILES=$("$GIT_BIN" ls-files --others --exclude-standard data/ | wc -l | tr -d ' ')
MODIFIED=$("$GIT_BIN" diff --name-only data/ | wc -l | tr -d ' ')
TOTAL=$(( NEW_FILES + MODIFIED ))

if [[ "$TOTAL" -gt 0 ]]; then
  echo "$(TS) Found $NEW_FILES new + $MODIFIED modified Fordham files. Committing..."
  "$GIT_BIN" add data/*.csv 2>/dev/null || true
  "$GIT_BIN" commit -m "Auto-update game data $(date '+%Y-%m-%d')" 2>/dev/null \
    && echo "$(TS) Fordham data committed." \
    || echo "$(TS) Fordham data already staged — continuing."
else
  echo "$(TS) No new Fordham game files."
fi

# ── 4. Rebuild scouting Parquet and push everything to GitHub ────────────────
# This always runs so cloud apps stay current with new scouting data.
echo "$(TS) Rebuilding scouting Parquet from updated CSVs..."
if ! "$PYTHON_BIN" "$REPO_DIR/scripts/build_scouting_parquet.py"; then
  echo "$(TS) WARNING: Parquet rebuild failed — skipping push." >&2
else
  # Pull latest before pushing (handles race with other push in step 3)
  "$GIT_BIN" pull --rebase --autostash --quiet 2>/dev/null || true

  "$GIT_BIN" add \
    "$REPO_DIR/scouting_data_1.parquet" \
    "$REPO_DIR/scouting_data_2.parquet" 2>/dev/null || true

  if ! "$GIT_BIN" diff --cached --quiet; then
    "$GIT_BIN" commit -m "Update scouting data $(date '+%Y-%m-%d')"
    "$GIT_BIN" push \
      && echo "$(TS) Pushed updated Parquet — Streamlit Cloud will redeploy." \
      || echo "$(TS) WARNING: git push failed. Committed locally." >&2
  else
    echo "$(TS) Parquet unchanged — no push needed."
  fi
fi

echo "$(TS) ── Update complete ─────────────────────────────────────────"
