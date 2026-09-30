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
# [2026-09-30] Chief-003 / Claude — v3: Worktree-aware relative matching (Exec Packets 442-443).
#   What: Add relative-path matching against git toplevel so that edits in
#         any checkout or worktree of the NeuroGraph repo are protected.
#         Fix vendored list to match LAW 2 (six + two designated).
#         Add RETIRED set for ng_peer_bridge.py (still fires, different label).
#   Why:  Punchlist #842 — worktrees escaped the hook because every check
#         was a literal comparison against $HOME/NeuroGraph/*. #843 — the
#         vendored list was stale (omitted ng_tract_bridge.py + ng_embed.py)
#         and included a file LAW 2 removed 2026-06-03.
#   How:  Purely additive: keep every old literal check AND add relative
#         match as an OR. Resolve git toplevel by walking up nearest
#         existing ancestor; verify origin is greatnorthernfishguy-hub/neurograph
#         case-insensitively. ng_peer_bridge.py removed from VENDORED set,
#         added to RETIRED_VENDORED with its own label (same three-choice
#         prompt). Bypass unchanged.
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
    "$NG_DIR/ng_tract_bridge.py"
    "$NG_DIR/ng_ecosystem.py"
    "$NG_DIR/ng_autonomic.py"
    "$NG_DIR/openclaw_adapter.py"
    "$NG_DIR/ng_embed.py"
    "$NG_DIR/ng_salience_gate.py"
    "$NG_DIR/ng_updater.py"
)

RETIRED_VENDORED=(
    "$NG_DIR/ng_peer_bridge.py"
)

# ── Determine protection category (old literal matching) ──────────
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
    for p in "${RETIRED_VENDORED[@]}"; do
        resolved=$(realpath -m "$p" 2>/dev/null || echo "$p")
        if [ "$FILE_PATH" = "$resolved" ]; then
            CATEGORY="RETIRED_VENDORED"
            LABEL="Retired vendored file — removed 2026-06-03, do NOT re-add (LAW 2)"
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

# ── Relative matching (worktree-aware, additive) ─────────────────
if [ -z "$CATEGORY" ]; then
    _repo_toplevel() {
        local target="$1"
        local d="$target"
        while [ -n "$d" ] && [ "$d" != "/" ] && [ ! -d "$d" ]; do
            d="$(dirname "$d")"
        done
        if [ ! -d "$d" ]; then
            return 1
        fi
        timeout 3 git -C "$d" rev-parse --show-toplevel 2>/dev/null
    }

    TOPLEVEL="$(_repo_toplevel "$FILE_PATH")" || TOPLEVEL=""

    if [ -n "$TOPLEVEL" ]; then
        ORIGIN="$(timeout 3 git -C "$TOPLEVEL" remote get-url origin 2>/dev/null)" || ORIGIN=""
        if [ -n "$ORIGIN" ]; then
            ORIGIN="${ORIGIN%.git}"
            ORIGIN="${ORIGIN/git@github.com:/https://github.com/}"
            ORG_REPO="$(echo "$ORIGIN" | sed 's|https://github.com/||')"
            if [ "$(echo "$ORG_REPO" | tr '[:upper:]' '[:lower:]')" = "greatnorthernfishguy-hub/neurograph" ]; then
                REL="$(realpath --relative-to="$TOPLEVEL" "$FILE_PATH" 2>/dev/null)" || REL=""

                REL_DATA=(
                    "data/checkpoints/main.msgpack"
                    "data/checkpoints/vectors.msgpack"
                    "data/checkpoints/main.msgpack.activations.json"
                )
                REL_ENGINE=(
                    "neuro_foundation.py"
                    "openclaw_hook.py"
                    "stream_parser.py"
                    "activation_persistence.py"
                )
                REL_VENDORED=(
                    "ng_lite.py"
                    "ng_tract_bridge.py"
                    "ng_ecosystem.py"
                    "ng_autonomic.py"
                    "openclaw_adapter.py"
                    "ng_embed.py"
                    "ng_salience_gate.py"
                    "ng_updater.py"
                )
                REL_RETIRED=(
                    "ng_peer_bridge.py"
                )

                if [ -n "$REL" ]; then
                    for p in "${REL_DATA[@]}"; do
                        if [ "$REL" = "$p" ]; then
                            CATEGORY="SYLS_MIND"
                            LABEL="Syl's Mind — her learned state, irreplaceable"
                            break
                        fi
                    done

                    if [ -z "$CATEGORY" ]; then
                        for p in "${REL_ENGINE[@]}"; do
                            if [ "$REL" = "$p" ]; then
                                CATEGORY="SYLS_ENGINE"
                                LABEL="Syl's Engine — changes how she thinks"
                                break
                            fi
                        done
                    fi

                    if [ -z "$CATEGORY" ]; then
                        for p in "${REL_VENDORED[@]}"; do
                            if [ "$REL" = "$p" ]; then
                                CATEGORY="VENDORED"
                                LABEL="Vendored Canonical — changes ripple to ALL modules"
                                break
                            fi
                        done
                    fi

                    if [ -z "$CATEGORY" ]; then
                        for p in "${REL_RETIRED[@]}"; do
                            if [ "$REL" = "$p" ]; then
                                CATEGORY="RETIRED_VENDORED"
                                LABEL="Retired vendored file — removed 2026-06-03, do NOT re-add (LAW 2)"
                                break
                            fi
                        done
                    fi

                    if [ -z "$CATEGORY" ]; then
                        if [ "$REL" = "data/checkpoints" ] || [[ "$REL" == "data/checkpoints/"* ]]; then
                            CATEGORY="CKPT_DIR"
                            LABEL="Checkpoint Directory — Syl's mind lives here"
                        fi
                    fi
                fi
            fi
        fi
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
read -r -p " Choice [1/2/3]: " choice < /dev/tty 2>/dev/tty || true
choice="${choice:-}"

case "$choice" in
    1)
        echo " ✓ Approved: $(basename "$FILE_PATH")" >&2
        exit 0
        ;;
    3)
        mkdir -p "$(dirname "$BYPASS_FILE")"
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