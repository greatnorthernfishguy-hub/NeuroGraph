#!/bin/bash
# Kill stale neurograph_rpc.py and neurograph_mcp.py processes that aren't
# children of the current openclaw-gateway. These accumulate from crashed
# gateway restarts and orphaned CC sessions.
#
# [2026-03-27] Claude (Opus 4.6) — Stale process cleanup
# Safe: only kills processes owned by $USER that match known patterns.

GATEWAY_PID=$(systemctl --user show openclaw-gateway -p MainPID --value 2>/dev/null)

for pid in $(pgrep -u "$USER" -f "neurograph_rpc\.py|neurograph_mcp\.py|fanout_daemon\.py"); do
    # Skip if it's a child of the current gateway
    if [ -n "$GATEWAY_PID" ] && [ "$GATEWAY_PID" != "0" ]; then
        parent=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ "$parent" = "$GATEWAY_PID" ]; then
            continue
        fi
        # Check grandparent too (node -> python)
        grandparent=$(ps -o ppid= -p "$parent" 2>/dev/null | tr -d ' ')
        if [ "$grandparent" = "$GATEWAY_PID" ]; then
            continue
        fi
    fi
    echo "Killing stale process: PID=$pid $(ps -o args= -p $pid 2>/dev/null | head -c 80)"
    kill "$pid" 2>/dev/null
done
