#!/bin/zsh
# Daily TrackMan update — downloads new game files then pushes to GitHub
# so the Streamlit Cloud app picks them up automatically.
#
# Runs via macOS LaunchAgent every 24 hours.
# Install with: zsh scripts/setup_daily_trackman_update.sh

set -euo pipefail

REPO_DIR="/Users/chrisjones/Documents/Codex/2026-04-30/github-com-cjones55-fordham-pitching-app"
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
# Only data/ is tracked in git — scouting_2026_trackman/ is gitignored.
# Counting untracked new files in data/:
NEW_FILES=$("$GIT_BIN" ls-files --others --exclude-standard data/ | wc -l | tr -d ' ')
MODIFIED=$("$GIT_BIN" diff --name-only data/ | wc -l | tr -d ' ')
TOTAL=$(( NEW_FILES + MODIFIED ))

if [[ "$TOTAL" -eq 0 ]]; then
  echo "$(TS) No new Fordham game files — nothing to push."
else
  echo "$(TS) Found $NEW_FILES new + $MODIFIED modified files in data/. Committing..."

  # Pull latest remote changes first; --autostash handles any unstaged changes safely
  "$GIT_BIN" pull --rebase --autostash --quiet || {
    echo "$(TS) WARNING: git pull failed — skipping push to avoid conflict." >&2
    exit 0
  }

  "$GIT_BIN" add data/*.csv 2>/dev/null || true

  COMMIT_MSG="Auto-update game data $(date '+%Y-%m-%d')"
  "$GIT_BIN" commit -m "$COMMIT_MSG" || {
    echo "$(TS) Nothing to commit (files may already be staged)."
    exit 0
  }

  "$GIT_BIN" push && echo "$(TS) Pushed $TOTAL file(s) — Streamlit Cloud will redeploy." || {
    echo "$(TS) WARNING: git push failed. Check SSH/HTTPS auth. Files are committed locally." >&2
    exit 1
  }
fi

# ── 4. Rebuild scouting Parquet and push so cloud apps stay current ──────────
echo "$(TS) Rebuilding scouting_data.parquet from updated CSVs..."
"$PYTHON_BIN" "$REPO_DIR/scripts/build_scouting_parquet.py" && {
  "$GIT_BIN" add "$REPO_DIR/scouting_data.parquet"
  if ! "$GIT_BIN" diff --cached --quiet; then
    "$GIT_BIN" commit -m "Update scouting_data.parquet $(date '+%Y-%m-%d')"
    "$GIT_BIN" push && echo "$(TS) Pushed updated scouting_data.parquet." || {
      echo "$(TS) WARNING: push failed. Parquet committed locally." >&2
    }
  else
    echo "$(TS) scouting_data.parquet unchanged — no push needed."
  fi
} || echo "$(TS) WARNING: Parquet rebuild failed." >&2

echo "$(TS) ── Update complete ─────────────────────────────────────────"
