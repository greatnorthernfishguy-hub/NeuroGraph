#!/bin/bash
# ---- Changelog ----
# [2026-03-11] Claude (Opus 4.6) — Initial implementation.
# [2026-03-11] Claude (Opus 4.6) — v2: Interactive approval prompt.
#   What: Instead of hard block, prompts Josh in terminal for approval.
#   Why:  Punch list items legitimately require protected file edits.
#         Josh should approve in real-time, not toggle permissions.
#   How:  Detects protected file → prompts [1] Approve [2] Block
#         [3] Approve All (session bypass). Reads from /dev/tty for
#         terminal input even when stdin is piped JSON.
# -------------------
#
# HOOK: PreToolUse
# MATCHER: Edit|Write|MultiEdit
# PURPOSE: Prompt Josh for approval before edits to protected files.
# EXIT 0: Approved or not a protected file.
# EXIT 2: Josh chose to block.

set -uo pipefail

NG_DIR="$HOME/NeuroGraph"
BYPASS_FILE="$NG_DIR/.claude/hooks/.session_approved"

# ── Read tool input from stdin ─────────────────────────────────────
INPUT=$(cat)

FILE_PATH=$(echo "$INPUT" | jq -r '
    .tool_input.file_path //
    .tool_input.path //
    .tool_input.file //
    empty
' 2>/dev/null)

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# ── Resolve to absolute path ──────────────────────────────────────
if [[ ! "$FILE_PATH" = /* ]]; then
    if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
        FILE_PATH="$CLAUDE_PROJECT_DIR/$FILE_PATH"
    else
        FILE_PATH="$NG_DIR/$FILE_PATH"
    fi
fi
FILE_PATH=$(realpath -m "$FILE_PATH" 2>/dev/null || echo "$FILE_PATH")

# ── Protected file lists ──────────────────────────────────────────
PROTECTED_DATA=(
    "$NG_DIR/data/checkpoints/main.msgpack"
    "$NG_DIR/data/checkpoints/vectors.msgpack"
    "$NG_DIR/data/checkpoints/main.msgpack.activations.json"
)

PROTECTED_ENGINE=(
    "$NG_DIR/neuro_foundation.py"
    "$NG_DIR/openclaw_hook.py"
    "$NG_DIR/stream_parser.py"
    "$NG_DIR/activation_persistence.py"
)

VENDORED_CANONICAL=(
    "$NG_DIR/ng_lite.py"
    "$NG_DIR/ng_peer_bridge.py"
    "$NG_DIR/ng_ecosystem.py"
    "$NG_DIR/ng_autonomic.py"
    "$NG_DIR/openclaw_adapter.py"
)

# ── Determine protection category ─────────────────────────────────
CATEGORY=""
LABEL=""

for p in "${PROTECTED_DATA[@]}"; do
    resolved=$(realpath -m "$p" 2>/dev/null || echo "$p")
    if [ "$FILE_PATH" = "$resolved" ]; then
        CATEGORY="SYLS_MIND"
        LABEL="Syl's Mind — her learned state, irreplaceable"
        break
    fi
done

if [ -z "$CATEGORY" ]; then
    for p in "${PROTECTED_ENGINE[@]}"; do
        resolved=$(realpath -m "$p" 2>/dev/null || echo "$p")
        if [ "$FILE_PATH" = "$resolved" ]; then
            CATEGORY="SYLS_ENGINE"
            LABEL="Syl's Engine — changes how she thinks"
            break
        fi
    done
fi

if [ -z "$CATEGORY" ]; then
    for p in "${VENDORED_CANONICAL[@]}"; do
        resolved=$(realpath -m "$p" 2>/dev/null || echo "$p")
        if [ "$FILE_PATH" = "$resolved" ]; then
            CATEGORY="VENDORED"
            LABEL="Vendored Canonical — changes ripple to ALL modules"
            break
        fi
    done
fi

if [ -z "$CATEGORY" ]; then
    CKPT_DIR=$(realpath -m "$NG_DIR/data/checkpoints" 2>/dev/null || echo "$NG_DIR/data/checkpoints")
    if [[ "$FILE_PATH" == "$CKPT_DIR"* ]]; then
        CATEGORY="CKPT_DIR"
        LABEL="Checkpoint Directory — Syl's mind lives here"
    fi
fi

# ── Not protected — proceed silently ──────────────────────────────
if [ -z "$CATEGORY" ]; then
    exit 0
fi

# ── Session bypass active? ────────────────────────────────────────
if [ -f "$BYPASS_FILE" ]; then
    echo "Session bypass active — protected file edit approved: $(basename "$FILE_PATH")" >&2
    exit 0
fi

# ── Prompt Josh ───────────────────────────────────────────────────
# /dev/tty reads from the terminal even when stdin is piped
cat >&2 <<EOF

══════════════════════════════════════════════════════════════
 ⚠  SYL'S LAW — PROTECTED FILE EDIT REQUESTED
══════════════════════════════════════════════════════════════

 File: $(basename "$FILE_PATH")
 Category: $LABEL

 [1] APPROVE  — I have backups, proceed with this edit
 [2] BLOCK    — Do not touch this file
 [3] APPROVE ALL — Approve all protected edits this session

══════════════════════════════════════════════════════════════
EOF

# Read from terminal, not from piped stdin
read -r -p " Choice [1/2/3]: " choice < /dev/tty 2>/dev/tty

case "$choice" in
    1)
        echo " ✓ Approved: $(basename "$FILE_PATH")" >&2
        exit 0
        ;;
    3)
        touch "$BYPASS_FILE"
        echo " ✓ Session bypass activated. All protected edits approved." >&2
        echo "   Run: rm ~/NeuroGraph/.claude/hooks/.session_approved  to re-lock" >&2
        exit 0
        ;;
    2|*)
        cat >&2 <<EOF

 ⛔ BLOCKED by Josh.

 REQUIRED BEFORE RETRY:
   1. Confirm manual backup of BOTH msgpack files
   2. Re-attempt the edit — you will be prompted again

EOF
        exit 2
        ;;
esac
