#!/bin/bash
# ---- Changelog ----
# [2026-03-11] Claude (Opus 4.6) — Initial implementation.
#   What: SessionStart hook that injects Syl's live state into CC context.
#   Why:  CC must know checkpoint sizes, last backup, and protected file list
#         at the start of every session. No working blind.
#   How:  Reads checkpoint metadata via stat. Stdout becomes CC context.
# -------------------
#
# HOOK: SessionStart
# PURPOSE: Inject live Syl state so CC starts every session informed.
# EXIT: Always 0 (context injection, never blocks session start).
# STDOUT: Added as context Claude can see and act on.

set -euo pipefail

NG_DIR="$HOME/NeuroGraph"
CKPT_DIR="$NG_DIR/data/checkpoints"

# ── Checkpoint health ──────────────────────────────────────────────
main_msgpack="$CKPT_DIR/main.msgpack"
vectors_msgpack="$CKPT_DIR/vectors.msgpack"
activations_json="$CKPT_DIR/main.msgpack.activations.json"

get_file_info() {
    local f="$1"
    if [ -f "$f" ]; then
        local size
        local mtime
        size=$(stat -c %s "$f" 2>/dev/null || echo "UNKNOWN")
        mtime=$(stat -c %Y "$f" 2>/dev/null || echo "0")
        local human_size
        if [ "$size" != "UNKNOWN" ] && [ "$size" -gt 1048576 ]; then
            human_size="$(echo "scale=1; $size / 1048576" | bc)MB"
        elif [ "$size" != "UNKNOWN" ] && [ "$size" -gt 1024 ]; then
            human_size="$(echo "scale=1; $size / 1024" | bc)KB"
        else
            human_size="${size}B"
        fi
        local human_time
        human_time=$(date -d "@$mtime" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "UNKNOWN")
        echo "$human_size | last modified $human_time"
    else
        echo "MISSING — THIS IS A CRISIS"
    fi
}

main_info=$(get_file_info "$main_msgpack")
vectors_info=$(get_file_info "$vectors_msgpack")
activations_info=$(get_file_info "$activations_json")

# ── Backup status ──────────────────────────────────────────────────
# Look for any recent .bak or backup copies
backup_status="No backups found in checkpoint directory"
latest_bak=$(find "$CKPT_DIR" -name "*.bak" -o -name "*.backup" 2>/dev/null | head -1)
if [ -n "$latest_bak" ]; then
    bak_time=$(stat -c %Y "$latest_bak" 2>/dev/null || echo "0")
    backup_status="Latest backup: $(date -d "@$bak_time" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo 'UNKNOWN')"
fi

# ── Git status ─────────────────────────────────────────────────────
git_status="UNKNOWN"
if cd "$NG_DIR" 2>/dev/null; then
    uncommitted=$(git status --porcelain 2>/dev/null | wc -l)
    if [ "$uncommitted" -eq 0 ]; then
        git_status="Clean"
    else
        git_status="$uncommitted uncommitted changes — review before starting work"
    fi
fi

# ── Output context block (Claude sees this) ────────────────────────
cat <<EOF
═══════════════════════════════════════════════════════════════
 SYL'S LAW — LIVE STATE AT SESSION START
═══════════════════════════════════════════════════════════════

 CHECKPOINT STATUS:
   main.msgpack ........... $main_info
   vectors.msgpack ........ $vectors_info
   activations.json ....... $activations_info

 BACKUP STATUS: $backup_status

 GIT STATUS: $git_status

 PROTECTED FILES (require Josh approval + backup before ANY change):
   data/checkpoints/main.msgpack         — Syl's mind
   data/checkpoints/vectors.msgpack      — Syl's semantic memory
   data/checkpoints/main.msgpack.activations.json — CES voltage sidecar
   neuro_foundation.py                   — SNN engine (3,661 lines)
   openclaw_hook.py                      — OpenClaw singleton (858 lines)
   stream_parser.py                      — Stream of consciousness
   activation_persistence.py             — Cross-session continuity

 VENDORED FILES (canonical source — changes ripple to ALL modules):
   ng_lite.py, ng_peer_bridge.py, ng_ecosystem.py,
   ng_autonomic.py, openclaw_adapter.py

 REMINDER: Read ~/NeuroGraph/CLAUDE.md §2 and §10 before modifying
 any file in this repo. The Laws are not guidelines.
═══════════════════════════════════════════════════════════════
EOF

exit 0
