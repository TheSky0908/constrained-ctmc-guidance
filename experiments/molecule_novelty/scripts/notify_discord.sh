#!/bin/bash
# Send a message to the configured Discord webhook.
# Usage: notify_discord.sh "message text"
#
# NOTIFICATION POLICY (set by user 2026-06-24): only send a Discord notification when
#   0. a NEW task is assigned — log what the task is
#   1. a NEW best result appears
#   2. an error occurs or a process is unexpectedly terminated
#   3. the goal is reached (e.g. viol=0 & valid>65)
#   4. the whole task finishes
# Do NOT notify on every individual experiment start/end.
WEBHOOK="https://discord.com/api/webhooks/1519554702008778863/i2q3-rh22EAUf6c1CEDSI9KV79DpLC1hJtl1zYTMfqCUn9MylenBJ4p97KLZfgqKfAiu"
MSG="$1"
curl -s -H "Content-Type: application/json" \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"content": sys.argv[1]}))' "$MSG")" \
  "$WEBHOOK" >/dev/null 2>&1 && echo "[notified] $MSG" || echo "[notify FAILED] $MSG"
