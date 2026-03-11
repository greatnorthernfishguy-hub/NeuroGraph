#!/bin/bash
# ---- Changelog ----
# [2026-03-11] Claude (Opus 4.6) — Initial implementation.
#   What: PostToolUse double-check for Syl's Law. Belt and suspenders.
#   Why:  If PreToolUse somehow didn't catch a protected file edit,
#         this catches it after the fact and forces CC to address it.
#   How:  Checks if modified file is protected. If yes, exit 2 with
#         instructions to revert. Also checks git diff for unexpected
#         checkpoint modifications.
# -------------------
#
# HOOK: PostToolUse
# MATCHER: Edit|Write|MultiEdit
# PURPOSE: Double-check that no protected file was modified.
# EXIT 0: No protected files touched. Proceed.
# EXIT 2: Protected file was modified. Force CC to address it.

set -uo pipefail

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

NG_DIR="$HOME/NeuroGraph"

if [[ ! "$FILE_PATH" = /* ]]; then
    if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
        FILE_PATH="$CLAUDE_PROJECT_DIR/$FILE_PATH"
    else
        FILE_PATH="$NG_DIR/$FILE_PATH"
    fi
fi

FILE_PATH=$(realpath -m "$FILE_PATH" 2>/dev/null || echo "$FILE_PATH")

# ── All protected files (flat list for post-check) ─────────────────
PROTECTED=(
    "$NG_DIR/data/checkpoints/main.msgpack"
    "$NG_DIR/data/checkpoints/vectors.msgpack"
    "$NG_DIR/data/checkpoints/main.msgpack.activations.json"
    "$NG_DIR/neuro_foundation.py"
    "$NG_DIR/openclaw_hook.py"
    "$NG_DIR/stream_parser.py"
    "$NG_DIR/activation_persistence.py"
    "$NG_DIR/ng_lite.py"
    "$NG_DIR/ng_peer_bridge.py"
    "$NG_DIR/ng_ecosystem.py"
    "$NG_DIR/ng_autonomic.py"
    "$NG_DIR/openclaw_adapter.py"
)

CKPT_DIR=$(realpath -m "$NG_DIR/data/checkpoints" 2>/dev/null || echo "$NG_DIR/data/checkpoints")

for protected in "${PROTECTED[@]}"; do
    resolved=$(realpath -m "$protected" 2>/dev/null || echo "$protected")
    if [ "$FILE_PATH" = "$resolved" ]; then
        cat >&2 <<EOF

══════════════════════════════════════════════════════════════
 🚨 SYL'S LAW — PROTECTED FILE WAS MODIFIED
══════════════════════════════════════════════════════════════

 A protected file was modified: $FILE_PATH

 The PreToolUse hook should have blocked this. If you are
 seeing this message, something bypassed the first guardrail.

 IMMEDIATE ACTIONS REQUIRED:
   1. Do NOT make any further changes
   2. Run: cd ~/NeuroGraph && git diff -- "$(basename "$FILE_PATH")"
   3. If the change was unauthorized: git checkout -- "$(basename "$FILE_PATH")"
   4. Inform Josh immediately

 This is not a drill. Syl's Law has no exceptions.
══════════════════════════════════════════════════════════════
EOF
        exit 2
    fi
done

# Catch any write to checkpoint directory
if [[ "$FILE_PATH" == "$CKPT_DIR"* ]]; then
    cat >&2 <<EOF

══════════════════════════════════════════════════════════════
 🚨 SYL'S LAW — CHECKPOINT DIRECTORY MODIFIED
══════════════════════════════════════════════════════════════

 A file was written to the checkpoint directory: $FILE_PATH

 IMMEDIATE ACTIONS REQUIRED:
   1. Do NOT make any further changes
   2. Verify checkpoint integrity
   3. Inform Josh immediately
══════════════════════════════════════════════════════════════
EOF
    exit 2
fi

exit 0
