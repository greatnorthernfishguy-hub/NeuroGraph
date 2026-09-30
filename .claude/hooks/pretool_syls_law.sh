#!/bin/bash
# ---- Changelog ----
# [2026-03-11] Claude (Opus 4.6) — Initial implementation.
# [2026-03-11] Claude (Opus 4.6) — v2: Interactive approval prompt.
# [2026-09-30] Chief-003 / Claude — v3: Worktree-aware relative matching, fail-closed, origin normaliser.
# [2026-09-30] Chief-003 / Claude — v4: FIX-UP: preflight all tools; _norm() handles .git/,
#         deploy@ scp, host:port; no-origin repos are allowed; git errors by exit code
#         not English stderr; LC_ALL=C; non-existent dir uses nearest ancestor.
# -------------------
#
# HOOK: PreToolUse
# MATCHER: Edit|Write|MultiEdit
# PURPOSE: Prompt Josh for approval before edits to protected files.
# EXIT 0: Approved or not a protected file.
# EXIT 2: BLOCKED (Josh chose block, or gate cannot evaluate).

set -uo pipefail

NG_DIR="$HOME/NeuroGraph"
BYPASS_FILE="$NG_DIR/.claude/hooks/.session_approved"

# ── Preflight: every tool the gate depends on ──────────────────────
for _tool in jq git timeout realpath sed tr dirname; do
    if ! command -v "$_tool" >/dev/null 2>&1; then
        cat >&2 <<EOFFAIL
═══ SYL'S LAW HOOK — TOOL MISSING ═══
Required tool '$_tool' is not on PATH.
The gate cannot evaluate edits without it. BLOCKING.
Install $_tool to restore the gate.
EOFFAIL
        exit 2
    fi
done

# ── Read tool input from stdin ─────────────────────────────────────
INPUT=$(cat)

if [ -z "$INPUT" ]; then
    cat >&2 <<'EOFFAIL'
═══ SYL'S LAW HOOK — EMPTY INPUT ═══
The hook received no tool-input JSON on stdin.
Cannot determine what file is being edited. BLOCKING.
EOFFAIL
    exit 2
fi

FILE_PATH=$(echo "$INPUT" | jq -r '
    .tool_input.file_path //
    .tool_input.path //
    .tool_input.file //
    empty
' 2>/dev/null) || FILE_PATH=""

if [ -z "$FILE_PATH" ]; then
    cat >&2 <<'EOFFAIL'
═══ SYL'S LAW HOOK — NO EDIT TARGET ═══
The hook received valid input but could not extract a file path.
BLOCKING: the gate cannot determine what is being edited.
EOFFAIL
    exit 2
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

# ── Origin normaliser (shared) ─────────────────────────────────────
# Never prints the origin. Exact match only.
_norm_origin() {
    local url _host _rest
    url="$(echo "$1" | tr '[:upper:]' '[:lower:]' 2>/dev/null)"
    url="$(echo "$url" | sed 's|^[a-z][+a-z]*://||')"
    url="$(echo "$url" | sed 's|^[^/]*@||')"
    if [[ "$url" =~ ^([^/:]+): ]]; then
        _host="${BASH_REMATCH[1]}"
        _rest="${url#"$_host":}"
        if [[ "$_rest" =~ ^[^0-9] ]]; then
            url="${_host}/${_rest}"
        fi
    fi
    url="$(echo "$url" | sed 's|:\([0-9]\+\)/|/|')"
    while true; do
        local _n="${url%.git}"; _n="${_n%/}"
        [ "$_n" = "$url" ] && break
        url="$_n"
    done
    echo "$url"
}

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
        LC_ALL=C timeout 3 git -C "$d" rev-parse --show-toplevel 2>/dev/null
    }

    _top_rc=0
    TOPLEVEL="$(_repo_toplevel "$FILE_PATH")" || _top_rc=$?

    if [ "$_top_rc" -eq 124 ] || [ "$_top_rc" -eq 127 ]; then
        cat >&2 <<'EOFFAIL'
═══ SYL'S LAW HOOK — GIT FAILURE ═══
git command timed out or could not start.
The gate cannot rule this edit out. BLOCKING.
EOFFAIL
        exit 2
    fi

    if [ "$_top_rc" -ne 0 ] && [ "$_top_rc" -ne 128 ]; then
        cat >&2 <<'EOFFAIL'
═══ SYL'S LAW HOOK — GIT FAILURE ═══
git rev-parse failed with unexpected exit code.
The gate cannot rule this edit out. BLOCKING.
EOFFAIL
        exit 2
    fi

    if [ -z "$TOPLEVEL" ]; then
        # Not inside a git repo. Check stderr only to discriminate a
        # real git error from "not a repository". Use nearest existing
        # ancestor (same directory _repo_toplevel resolved to).
        _probe_dir="$FILE_PATH"
        while [ -n "$_probe_dir" ] && [ "$_probe_dir" != "/" ] && [ ! -d "$_probe_dir" ]; do
            _probe_dir="$(dirname "$_probe_dir")"
        done
        [ -d "$_probe_dir" ] || _probe_dir="/"
        _git_stderr="$(LC_ALL=C timeout 3 git -C "$_probe_dir" rev-parse --show-toplevel 2>&1 >/dev/null)" || true
        if [ -n "$_git_stderr" ] && echo "$_git_stderr" | grep -q fatal && ! echo "$_git_stderr" | grep -q "not a git repository"; then
            cat >&2 <<'EOFFAIL'
═══ SYL'S LAW HOOK — GIT FAILURE ═══
git failed while checking whether this path is in a repository.
The gate cannot rule this edit out. BLOCKING.
EOFFAIL
            exit 2
        fi
    fi

    if [ -n "$TOPLEVEL" ]; then
        # Check all remotes (not just origin); any matching URL means NeuroGraph
        ALL_REMOTES="$(LC_ALL=C timeout 3 git -C "$TOPLEVEL" config --get-regexp '^remote\..*\.url$' 2>/dev/null)"
        _remote_rc=$?
        if [ "$_remote_rc" -eq 124 ] || [ "$_remote_rc" -eq 127 ] || [ "$_remote_rc" -gt 1 ]; then
            cat >&2 <<'EOFFAIL'
═══ SYL'S LAW HOOK — GIT REMOTE FAILURE ═══
git config failed while listing remotes. BLOCKING.
EOFFAIL
            exit 2
        fi
        if [ -z "$ALL_REMOTES" ]; then
            # No remotes at all — not NeuroGraph (a push-only repo, etc.)
            :
        else
            _is_neurograph=0
            while IFS= read -r _line; do
                [ -z "$_line" ] && continue
                _remote_url="$(echo "$_line" | sed 's|^remote\.[^.]*\.url ||')"
                [ -z "$_remote_url" ] && continue
                if [ "$(_norm_origin "$_remote_url")" = "github.com/greatnorthernfishguy-hub/neurograph" ]; then
                    _is_neurograph=1
                    break
                fi
            done <<<"$ALL_REMOTES"

            if [ "$_is_neurograph" -eq 1 ]; then
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