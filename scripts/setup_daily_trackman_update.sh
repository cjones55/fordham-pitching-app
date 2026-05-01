#!/bin/zsh
set -euo pipefail

REPO_DIR="/Users/chrisjones/Documents/Codex/2026-04-30/github-com-cjones55-fordham-pitching-app"
SERVICE_NAME="fordham-trackman-ftp"
ACCOUNT_NAME="Fordham"
PLIST_PATH="$HOME/Library/LaunchAgents/com.fordham.trackman.update.plist"
RUNNER="$REPO_DIR/scripts/run_daily_trackman_update.sh"
LOG_DIR="$REPO_DIR/logs"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
chmod +x "$RUNNER"

echo "This stores your TrackMan password in your local macOS Keychain."
echo "It will not write the password into GitHub or the repo."
printf "TrackMan FTP password for %s: " "$ACCOUNT_NAME"
read -rs TRACKMAN_PASSWORD
echo

if [[ -z "$TRACKMAN_PASSWORD" ]]; then
  echo "No password entered. Setup cancelled." >&2
  exit 1
fi

security add-generic-password \
  -a "$ACCOUNT_NAME" \
  -s "$SERVICE_NAME" \
  -w "$TRACKMAN_PASSWORD" \
  -U

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.fordham.trackman.update</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$RUNNER</string>
  </array>

  <key>StartInterval</key>
  <integer>86400</integer>

  <key>RunAtLoad</key>
  <true/>

  <key>StandardOutPath</key>
  <string>$LOG_DIR/trackman_update.out.log</string>

  <key>StandardErrorPath</key>
  <string>$LOG_DIR/trackman_update.err.log</string>
</dict>
</plist>
PLIST

launchctl unload "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl load "$PLIST_PATH"

echo "Daily TrackMan updater installed."
echo "It runs every 24 hours and also runs once immediately after setup."
echo "Logs:"
echo "  $LOG_DIR/trackman_update.out.log"
echo "  $LOG_DIR/trackman_update.err.log"
