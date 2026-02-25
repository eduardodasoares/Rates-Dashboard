#!/bin/bash
# Daily signal logger — runs at market close on weekdays.
# Appends stdout/stderr to logs/cron.log.
# Safe to run multiple times: skips if today already logged.

PROJECT="/Users/eduardosoares/Documents/Repo Trading Coding Projects/Rates Dashboard"
PYTHON="/Users/eduardosoares/Documents/Repo Trading Coding Projects/venv/bin/python"
LOGFILE="$PROJECT/logs/cron.log"

cd "$PROJECT" || exit 1

echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOGFILE"
"$PYTHON" main.py log >> "$LOGFILE" 2>&1
echo "" >> "$LOGFILE"
