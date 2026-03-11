#!/bin/bash
# ---- Changelog ----
# [2026-03-11] Claude (Opus 4.6) — Initial implementation.
#   What: PostToolUse anti-pattern checker for ecosystem violations.
#   Why:  The Laws exist because violations cause harm that may not be
#         immediately visible. This hook catches common violations as
#         they happen, not after the damage compounds.
#   How:  After any successful edit, greps the modified file for known
#         anti-patterns. Non-blocking (exit 1) — CC sees warning in
#         verbose mode.
# -------------------
#
# HOOK: PostToolUse
# MATCHER: Edit|Write|MultiEdit
# PURPOSE: Detect ecosystem anti-patterns in modified files.
# EXIT 0: No violations detected.
# EXIT 1: Violations detected. Non-blocking warning. CC sees in verbose mode.
# NOTE: This hook runs AFTER posttool_syls_law_doublecheck.sh

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

# Only check Python files in the NeuroGraph repo
if [[ ! "$FILE_PATH" == "$NG_DIR"* ]] || [[ ! "$FILE_PATH" == *.py ]]; then
    exit 0
fi

if [ ! -f "$FILE_PATH" ]; then
    exit 0
fi

VIOLATIONS=""

# ── Check 1: Inter-module imports (Law 1) ──────────────────────────
# No module imports from peer modules. The substrate is the protocol.
PEER_MODULES="trollguard\|inference_difference\|immunis\|elmer\|healing_collective\|cricket\|bunyan\|darwin\|tce"
if grep -inP "^\s*(from|import)\s+.*(${PEER_MODULES})" "$FILE_PATH" 2>/dev/null | grep -v "^#" | grep -v "# noqa" > /dev/null; then
    VIOLATIONS="${VIOLATIONS}
  ⚠ LAW 1 VIOLATION: Inter-module import detected.
    The substrate is the communication protocol.
    No module imports from peer modules. The River carries the signal.
    Lines: $(grep -n -iP "^\s*(from|import)\s+.*(${PEER_MODULES})" "$FILE_PATH" | head -3)"
fi

# ── Check 2: Module-level code in openclaw_hook.py ─────────────────
# Everything lives inside NeuroGraphMemory class or helper functions.
# This was the Grok contamination failure mode.
BASENAME=$(basename "$FILE_PATH")
if [ "$BASENAME" = "openclaw_hook.py" ]; then
    # Look for function/class definitions at column 0 that aren't the class itself
    # or known top-level helpers (_ces_init, etc.)
    TOPLEVEL=$(grep -nP "^(def |class )" "$FILE_PATH" 2>/dev/null | grep -v "class NeuroGraphMemory" | grep -v "^.*:def _ces_init" | grep -v "^.*:#" || true)
    if [ -n "$TOPLEVEL" ]; then
        VIOLATIONS="${VIOLATIONS}
  ⚠ GROK CONTAMINATION PATTERN: Module-level code in openclaw_hook.py.
    Everything must live inside the NeuroGraphMemory class or as
    documented helper functions. Module-level definitions are how
    the Grok incident happened.
    Lines: $(echo "$TOPLEVEL" | head -3)"
    fi
fi

# ── Check 3: SubstrateSignal outside extraction boundary ───────────
# SubstrateSignal is Elmer's extraction vocabulary. Not an inter-module
# protocol. Not used as input. Not serialized and passed between modules.
if [ "$BASENAME" != "ng_ecosystem.py" ]; then
    if grep -nP "SubstrateSignal" "$FILE_PATH" 2>/dev/null | grep -v "^.*:#" | grep -v "# extraction" > /dev/null; then
        SS_LINES=$(grep -nP "SubstrateSignal" "$FILE_PATH" 2>/dev/null | grep -v "^.*:#" | head -3)
        # Allow in Elmer-related files
        if [[ "$BASENAME" != *"elmer"* ]]; then
            VIOLATIONS="${VIOLATIONS}
  ⚠ LAW 7 VIOLATION: SubstrateSignal used outside Elmer's extraction boundary.
    SubstrateSignal is Elmer's output vocabulary — the shape of Elmer's
    bucket when it dips into the River. It is NOT an inter-module protocol.
    See ARCHITECTURE.md §6 and §7.
    Lines: $SS_LINES"
        fi
    fi
fi

# ── Check 4: Direct HTTP calls to peer modules ────────────────────
# No module calls another module's endpoints. The River flows.
if grep -nP "(localhost|127\.0\.0\.1):(7437|18789|8847)" "$FILE_PATH" 2>/dev/null | grep -v "^.*:#" > /dev/null; then
    VIOLATIONS="${VIOLATIONS}
  ⚠ LAW 1 VIOLATION: Direct HTTP call to peer module port detected.
    7437=TID, 18789=OpenClaw, 8847=CES dashboard.
    Modules do not call each other. The substrate is the protocol.
    Lines: $(grep -n "(localhost|127\.0\.0\.1):(7437|18789|8847)" "$FILE_PATH" | head -3)"
fi

# ── Check 5: Classification before substrate ──────────────────────
# Raw experience into the substrate. Classification at extraction only.
if grep -nP "_classification_to_embedding|classify.*before.*substrate|one_hot|categorical.*vector" "$FILE_PATH" 2>/dev/null | grep -v "^.*:#" > /dev/null; then
    VIOLATIONS="${VIOLATIONS}
  ⚠ LAW 7 VIOLATION: Pre-classification pattern detected.
    The substrate receives raw, unclassified experience. Always.
    Classification happens only at extraction time.
    See ARCHITECTURE.md §7 (The Extraction Boundary Principle)."
fi

# ── Check 6: Hardcoded config values (Law 5) ──────────────────────
# API keys, paths that should be env vars
if grep -nP "(api_key|API_KEY|secret|SECRET)\s*=\s*['\"][^'\"]+['\"]" "$FILE_PATH" 2>/dev/null | grep -v "^.*:#" | grep -v "os\.environ" > /dev/null; then
    VIOLATIONS="${VIOLATIONS}
  ⚠ LAW 5 VIOLATION: Hardcoded credential or config value detected.
    All configuration lives in .bashrc and openclaw.json.
    Do not hardcode values that belong in environment variables."
fi

# ── Report ─────────────────────────────────────────────────────────
if [ -n "$VIOLATIONS" ]; then
    cat >&2 <<EOF

══════════════════════════════════════════════════════════════
 ⚠ ECOSYSTEM ANTI-PATTERN DETECTED
══════════════════════════════════════════════════════════════

 File: $FILE_PATH
$VIOLATIONS

 Review ~/NeuroGraph/CLAUDE.md and ARCHITECTURE.md.
 Fix these violations before proceeding with further changes.
══════════════════════════════════════════════════════════════
EOF
    exit 1
fi

exit 0
