#!/usr/bin/env bash
# Runs on YOUR machine (Git Bash on Windows, or a Mac/Linux terminal).
# Syncs this working tree to the OCI VM and runs deploy/oci/setup.sh there,
# which restarts the app. Nothing needs to be committed or pushed to GitHub
# first -- whatever is in this folder right now is what gets deployed.
#
#   bash deploy/oci/push.sh
#
# What is NOT synced (so nothing on the VM gets clobbered): everything in
# .gitignore -- the VM's own venv/, cache/, .env, and the data/*.json state
# files the alert job maintains -- plus .git and the SSH keys.
#
# Env overrides:
#   TRENDLINE_HOST     ssh target        (default ubuntu@132.226.186.192)
#   TRENDLINE_SSH_KEY  private key path  (default ./ssh-key-2026-07-13.key)
#   TRENDLINE_APP_DIR  remote directory  (default TrendLine, i.e. ~/TrendLine)
#   PORT               app port          (default 5000, passed to setup.sh)
set -euo pipefail

cd "$(dirname "$0")/../.."   # project root

HOST="${TRENDLINE_HOST:-ubuntu@132.226.186.192}"
KEY="${TRENDLINE_SSH_KEY:-ssh-key-2026-07-13.key}"
APP_DIR="${TRENDLINE_APP_DIR:-TrendLine}"
PORT="${PORT:-5000}"

if [ ! -f "$KEY" ]; then
    echo "SSH key not found: $KEY  (set TRENDLINE_SSH_KEY)" >&2
    exit 1
fi
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$HOST")

# Build tar excludes from .gitignore (drop comments/blank lines and trailing
# slashes -- GNU tar's default unanchored matching then treats "venv" as
# "any path component named venv"), plus the things git already ignores
# implicitly.
# mapfile (bash 4+) isn't available in macOS's stock /bin/bash (3.2) -- a
# plain while-read loop works on every bash this script claims to support.
EXCLUDES=()
while IFS= read -r line; do
    EXCLUDES+=("--exclude=$line")
done < <(grep -vE '^\s*(#|$)' .gitignore | sed 's:/*$::')
EXCLUDES+=(--exclude=.git --exclude=.claude --exclude='*.key' --exclude='*.key.pub')

echo "==> syncing $(pwd) -> $HOST:$APP_DIR"
"${SSH[@]}" "mkdir -p '$APP_DIR'"
tar -czf - "${EXCLUDES[@]}" . | "${SSH[@]}" "tar -xzf - -C '$APP_DIR'"

# Windows checkouts can carry CRLF; make sure the scripts run on Linux.
"${SSH[@]}" "sed -i 's/\r\$//' '$APP_DIR'/deploy/oci/*"

echo "==> running setup on the VM"
"${SSH[@]}" "cd '$APP_DIR' && PORT=$PORT bash deploy/oci/setup.sh"
