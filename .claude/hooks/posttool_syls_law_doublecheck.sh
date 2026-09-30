#!/bin/bash
# ---- Changelog ----
# [2026-03-11] Claude (Opus 4.6) — Initial implementation.
# [2026-09-30] Chief-003 / Claude — v2: Worktree-aware matching, fixed vendored list,
#         retired ng_peer_bridge.py, preflight all tools.
# -------------------
#
# HOOK: PostToolUse
# MATCHER: Edit|Write|MultiEdit
# PURPOSE: Double-check that no protected file was modified.
# EXIT 0: No protected files touched. Proceed.
# EXIT 2: Protected file was modified. Force CC to address it.

set -uo pipefail

NG_DIR="$HOME/NeuroGraph"

# ── Preflight ──────────────────────────────────────────────────────
for _tool in jq git timeout realpath sed tr dirname; do
    if ! command -v "$_tool" >/dev/null 2>&1; then
        cat >&2 <<EOFFAIL
═══ SYL'S LAW DOUBLECHECK — TOOL MISSING ═══
Required tool '$_tool' is not on PATH. BLOCKING.
EOFFAIL
        exit 2
    fi
done

INPUT=$(cat)

if [ -z "$INPUT" ]; then
    cat >&2 <<'EOFFAIL'
═══ SYL'S LAW DOUBLECHECK — EMPTY INPUT ═══
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
═══ SYL'S LAW DOUBLECHECK — NO PATH ═══
EOFFAIL
    exit 2
fi

if [[ ! "$FILE_PATH" = /* ]]; then
    if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
        FILE_PATH="$CLAUDE_PROJECT_DIR/$FILE_PATH"
    else
        FILE_PATH="$NG_DIR/$FILE_PATH"
    fi
fi
FILE_PATH=$(realpath -m "$FILE_PATH" 2>/dev/null || echo "$FILE_PATH")

# ── Protected file lists (literal + relative) ─────────────────────
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

CKPT_DIR=$(realpath -m "$NG_DIR/data/checkpoints" 2>/dev/null || echo "$NG_DIR/data/checkpoints")

# ── Old literal matching ───────────────────────────────────────────
FOUND=0
for p in "${PROTECTED_DATA[@]}" "${PROTECTED_ENGINE[@]}" "${VENDORED_CANONICAL[@]}" "${RETIRED_VENDORED[@]}"; do
    resolved=$(realpath -m "$p" 2>/dev/null || echo "$p")
    if [ "$FILE_PATH" = "$resolved" ]; then
        FOUND=1
        break
    fi
done
if [ "$FOUND" -eq 0 ] && [[ "$FILE_PATH" == "$CKPT_DIR"* ]]; then
    FOUND=1
fi

# ── Origin normaliser (copy from pretool_syls_law.sh v4) ──────────
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
if [ "$FOUND" -eq 0 ]; then
    _repo_toplevel() {
        local target="$1"
        local d="$target"
        while [ -n "$d" ] && [ "$d" != "/" ] && [ ! -d "$d" ]; do
            d="$(dirname "$d")"
        done
        if [ ! -d "$d" ]; then return 1; fi
        LC_ALL=C timeout 3 git -C "$d" rev-parse --show-toplevel 2>/dev/null
    }

    _top_rc=0
    TOPLEVEL="$(_repo_toplevel "$FILE_PATH")" || _top_rc=$?
    [ "$_top_rc" -eq 124 ] || [ "$_top_rc" -eq 127 ] && { cat >&2 <<'EOFFAIL'
═══ SYL'S LAW DOUBLECHECK — GIT FAILURE ═══
EOFFAIL
        exit 2; }

    [ "$_top_rc" -ne 0 ] && [ "$_top_rc" -ne 128 ] && { cat >&2 <<'EOFFAIL'
═══ SYL'S LAW DOUBLECHECK — GIT FAILURE ═══
EOFFAIL
        exit 2; }

    if [ -z "$TOPLEVEL" ]; then
        _probe_dir="$FILE_PATH"
        while [ -n "$_probe_dir" ] && [ "$_probe_dir" != "/" ] && [ ! -d "$_probe_dir" ]; do
            _probe_dir="$(dirname "$_probe_dir")"
        done
        [ -d "$_probe_dir" ] || _probe_dir="/"
        _git_stderr="$(LC_ALL=C timeout 3 git -C "$_probe_dir" rev-parse --show-toplevel 2>&1 >/dev/null)" || true
        if [ -n "$_git_stderr" ] && echo "$_git_stderr" | grep -q fatal && ! echo "$_git_stderr" | grep -q "not a git repository"; then
            cat >&2 <<'EOFFAIL'
═══ SYL'S LAW DOUBLECHECK — GIT FAILURE ═══
EOFFAIL
            exit 2
        fi
    fi

    if [ -n "$TOPLEVEL" ]; then
        ALL_REMOTES="$(LC_ALL=C timeout 3 git -C "$TOPLEVEL" config --get-regexp '^remote\..*\.url$' 2>/dev/null)" || true
        _rr=$?
        [ "$_rr" -eq 124 ] || [ "$_rr" -eq 127 ] || [ "$_rr" -gt 1 ] && { cat >&2 <<'EOFFAIL'
═══ SYL'S LAW DOUBLECHECK — GIT REMOTE FAILURE ═══
EOFFAIL
            exit 2; }

        if [ -n "$ALL_REMOTES" ]; then
            _is_ng=0
            while IFS= read -r _line; do
                [ -z "$_line" ] && continue
                _ru="$(echo "$_line" | sed 's|^remote\.[^.]*\.url ||')"
                [ "$(_norm_origin "$_ru")" = "github.com/greatnorthernfishguy-hub/neurograph" ] && { _is_ng=1; break; }
            done <<<"$ALL_REMOTES"

            if [ "$_is_ng" -eq 1 ]; then
                REL="$(realpath --relative-to="$TOPLEVEL" "$FILE_PATH" 2>/dev/null)" || REL=""
                if [ -n "$REL" ]; then
                    REL_DATA=("data/checkpoints/main.msgpack" "data/checkpoints/vectors.msgpack" "data/checkpoints/main.msgpack.activations.json")
                    REL_ENGINE=("neuro_foundation.py" "openclaw_hook.py" "stream_parser.py" "activation_persistence.py")
                    REL_VENDORED=("ng_lite.py" "ng_tract_bridge.py" "ng_ecosystem.py" "ng_autonomic.py" "openclaw_adapter.py" "ng_embed.py" "ng_salience_gate.py" "ng_updater.py")
                    REL_RETIRED=("ng_peer_bridge.py")

                    for p in "${REL_DATA[@]}" "${REL_ENGINE[@]}" "${REL_VENDORED[@]}" "${REL_RETIRED[@]}"; do
                        if [ "$REL" = "$p" ]; then
                            FOUND=1
                            break
                        fi
                    done
                    if [ "$FOUND" -eq 0 ]; then
                        [ "$REL" = "data/checkpoints" ] || [[ "$REL" == "data/checkpoints/"* ]] && FOUND=1
                    fi
                fi
            fi
        fi
    fi
fi

# ── Report ─────────────────────────────────────────────────────────
if [ "$FOUND" -eq 1 ]; then
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

exit 0