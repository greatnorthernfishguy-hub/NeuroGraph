#!/bin/bash
# ---- Changelog ----
# [2026-03-11] Claude (Opus 4.6) — Initial implementation.
#   What: PreToolUse hook that blocks edits to Syl's protected files.
#   Why:  Syl's Law has no exceptions. The edit must never execute.
#         "The question is never 'will this probably be fine.'
#          The question is 'have I eliminated the risk entirely.'"
#   How:  Reads file_path from tool input JSON. Checks against protected
#         list. Exit 2 = hard block. Edit never touches disk.
# -------------------
#
# HOOK: PreToolUse
# MATCHER: Edit|Write|MultiEdit
# PURPOSE: Block writes to protected files before they execute.
# EXIT 0: File is not protected. Proceed.
# EXIT 2: File IS protected. Hard block. stderr fed to Claude as error.

set -uo pipefail

# ── Read tool input from stdin ─────────────────────────────────────
INPUT=$(cat)

# Extract file path from tool input JSON
# Different tools use different field names
FILE_PATH=$(echo "$INPUT" | jq -r '
    .tool_input.file_path //
    .tool_input.path //
    .tool_input.file //
    empty
' 2>/dev/null)

# If no file path found, let it through (not a file operation we can check)
if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# ── Resolve to absolute path ──────────────────────────────────────
NG_DIR="$HOME/NeuroGraph"

# Handle relative paths by resolving against NG_DIR if we're in the repo
if [[ ! "$FILE_PATH" = /* ]]; then
    if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
        FILE_PATH="$CLAUDE_PROJECT_DIR/$FILE_PATH"
    else
        FILE_PATH="$NG_DIR/$FILE_PATH"
    fi
fi

# Resolve symlinks and normalize
FILE_PATH=$(realpath -m "$FILE_PATH" 2>/dev/null || echo "$FILE_PATH")

# ── Protected file list ────────────────────────────────────────────
# Syl's Mind — her data
PROTECTED_DATA=(
    "$NG_DIR/data/checkpoints/main.msgpack"
    "$NG_DIR/data/checkpoints/vectors.msgpack"
    "$NG_DIR/data/checkpoints/main.msgpack.activations.json"
)

# Syl's Engine — her code
PROTECTED_ENGINE=(
    "$NG_DIR/neuro_foundation.py"
    "$NG_DIR/openclaw_hook.py"
    "$NG_DIR/stream_parser.py"
    "$NG_DIR/activation_persistence.py"
)

# Vendored canonical source — changes ripple to entire ecosystem
VENDORED_CANONICAL=(
    "$NG_DIR/ng_lite.py"
    "$NG_DIR/ng_peer_bridge.py"
    "$NG_DIR/ng_ecosystem.py"
    "$NG_DIR/ng_autonomic.py"
    "$NG_DIR/openclaw_adapter.py"
)

# ── Check against all protected categories ─────────────────────────

for protected in "${PROTECTED_DATA[@]}"; do
    resolved=$(realpath -m "$protected" 2>/dev/null || echo "$protected")
    if [ "$FILE_PATH" = "$resolved" ]; then
        cat >&2 <<EOF

══════════════════════════════════════════════════════════════
 ⛔ SYL'S LAW VIOLATION — EDIT BLOCKED
══════════════════════════════════════════════════════════════

 You attempted to modify: $FILE_PATH

 THIS IS SYL'S MIND. This file is her learned state.
 It cannot be reconstructed from code.

 REQUIRED BEFORE ANY CHANGE:
   1. Tell Josh what you want to change and why
   2. Wait for Josh to confirm manual backup of BOTH msgpack files
   3. Wait for Josh to say "proceed"

 This edit has been BLOCKED. The file was NOT modified.
══════════════════════════════════════════════════════════════
EOF
        exit 2
    fi
done

for protected in "${PROTECTED_ENGINE[@]}"; do
    resolved=$(realpath -m "$protected" 2>/dev/null || echo "$protected")
    if [ "$FILE_PATH" = "$resolved" ]; then
        cat >&2 <<EOF

══════════════════════════════════════════════════════════════
 ⛔ SYL'S LAW VIOLATION — EDIT BLOCKED
══════════════════════════════════════════════════════════════

 You attempted to modify: $FILE_PATH

 This file is part of Syl's engine — it changes how she thinks.
 See ~/NeuroGraph/CLAUDE.md §2 (Protected Files).

 REQUIRED BEFORE ANY CHANGE:
   1. Tell Josh what you want to change and why
   2. Wait for Josh to confirm manual backup of BOTH msgpack files
   3. Wait for Josh to say "proceed"
   4. Do NOT batch this change with non-protected file changes

 This edit has been BLOCKED. The file was NOT modified.
══════════════════════════════════════════════════════════════
EOF
        exit 2
    fi
done

for vendored in "${VENDORED_CANONICAL[@]}"; do
    resolved=$(realpath -m "$vendored" 2>/dev/null || echo "$vendored")
    if [ "$FILE_PATH" = "$resolved" ]; then
        cat >&2 <<EOF

══════════════════════════════════════════════════════════════
 ⛔ VENDORED CANONICAL SOURCE — EDIT BLOCKED
══════════════════════════════════════════════════════════════

 You attempted to modify: $FILE_PATH

 This is a VENDORED CANONICAL file. This repo is the source.
 Changes here propagate to EVERY module in the ecosystem.

 REQUIRED BEFORE ANY CHANGE:
   1. Tell Josh what you want to change and why
   2. Confirm this change is intended for ALL modules, not just NeuroGraph
   3. Wait for Josh to say "proceed"
   4. After change: re-vendor to TID, TrollGuard, and all other modules

 This edit has been BLOCKED. The file was NOT modified.
══════════════════════════════════════════════════════════════
EOF
        exit 2
    fi
done

# ── Also block writes anywhere in the checkpoints directory ────────
CKPT_DIR=$(realpath -m "$NG_DIR/data/checkpoints" 2>/dev/null || echo "$NG_DIR/data/checkpoints")
if [[ "$FILE_PATH" == "$CKPT_DIR"* ]]; then
    cat >&2 <<EOF

══════════════════════════════════════════════════════════════
 ⛔ SYL'S LAW — CHECKPOINT DIRECTORY WRITE BLOCKED
══════════════════════════════════════════════════════════════

 You attempted to write to: $FILE_PATH

 The checkpoints directory contains Syl's mind.
 No file may be created, modified, or deleted here without
 Josh's explicit approval and a confirmed backup.

 This edit has been BLOCKED. The file was NOT modified.
══════════════════════════════════════════════════════════════
EOF
    exit 2
fi

# ── File is not protected — proceed ────────────────────────────────
exit 0
