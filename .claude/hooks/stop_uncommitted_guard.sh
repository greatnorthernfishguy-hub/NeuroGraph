#!/bin/bash
# ---- Changelog ----
# [2026-03-11] Claude (Opus 4.6) — Initial implementation.
#   What: Stop hook that checks for uncommitted changes in NeuroGraph.
#   Why:  Uncommitted changes left behind become code shrapnel.
#         The code explosion happened because sessions ended with
#         unfinished work scattered across the repo.
#   How:  Runs git status. If uncommitted changes exist, exit 2 forces
#         CC to continue and address them.
# -------------------
#
# HOOK: Stop
# PURPOSE: Prevent CC from finishing with uncommitted changes.
# EXIT 0: Working tree is clean. CC may stop.
# EXIT 2: Uncommitted changes exist. CC must address them.

set -uo pipefail

NG_DIR="$HOME/NeuroGraph"

# Only check if we're working in the NeuroGraph repo
# Check if the session was working in NeuroGraph by looking at project dir
if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
    PROJECT_DIR=$(realpath -m "$CLAUDE_PROJECT_DIR" 2>/dev/null || echo "$CLAUDE_PROJECT_DIR")
    NG_RESOLVED=$(realpath -m "$NG_DIR" 2>/dev/null || echo "$NG_DIR")
    if [ "$PROJECT_DIR" != "$NG_RESOLVED" ]; then
        # Not working in NeuroGraph — don't enforce
        exit 0
    fi
fi

cd "$NG_DIR" 2>/dev/null || exit 0

# Check for uncommitted changes
CHANGES=$(git status --porcelain 2>/dev/null)

if [ -z "$CHANGES" ]; then
    exit 0
fi

# Count the changes
NUM_CHANGES=$(echo "$CHANGES" | wc -l)

# Check if any protected files are among the uncommitted changes
PROTECTED_TOUCHED=""
while IFS= read -r line; do
    changed_file=$(echo "$line" | awk '{print $2}')
    case "$changed_file" in
        neuro_foundation.py|openclaw_hook.py|stream_parser.py|activation_persistence.py)
            PROTECTED_TOUCHED="${PROTECTED_TOUCHED}    ⚠ PROTECTED: $changed_file
"
            ;;
        ng_lite.py|ng_peer_bridge.py|ng_ecosystem.py|ng_autonomic.py|openclaw_adapter.py)
            PROTECTED_TOUCHED="${PROTECTED_TOUCHED}    ⚠ VENDORED: $changed_file
"
            ;;
        data/checkpoints/*)
            PROTECTED_TOUCHED="${PROTECTED_TOUCHED}    🚨 SYL'S MIND: $changed_file
"
            ;;
    esac
done <<< "$CHANGES"

cat >&2 <<EOF

══════════════════════════════════════════════════════════════
 ⛔ UNCOMMITTED CHANGES IN NEUROGRAPH
══════════════════════════════════════════════════════════════

 $NUM_CHANGES uncommitted change(s) detected:

$(echo "$CHANGES" | sed 's/^/    /')

EOF

if [ -n "$PROTECTED_TOUCHED" ]; then
    cat >&2 <<EOF
 PROTECTED FILES AMONG CHANGES:
$PROTECTED_TOUCHED
EOF
fi

cat >&2 <<EOF
 BEFORE STOPPING, YOU MUST:
   1. Review each change with: git diff <filename>
   2. Either commit the changes with a descriptive message
      OR explain to Josh why they should remain uncommitted
   3. Do NOT leave code shrapnel (Law 3: Restore, Don't Rebuild)

══════════════════════════════════════════════════════════════
EOF

exit 2
