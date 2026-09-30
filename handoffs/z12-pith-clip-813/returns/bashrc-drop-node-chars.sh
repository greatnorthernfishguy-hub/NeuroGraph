#!/usr/bin/env bash
# ---- Changelog ----
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 STAGED .bashrc edit (NOT APPLIED)
# What: removes the single line `export CC_PITH_PROVIDER_NODE_CHARS=...` from ~/.bashrc.
# Why: Exec P413 -- CCs maintain .bashrc; the owning worker removes the export in the S4
#   batched step that deploys the code which no longer requires it (rides S4's go, P342).
#   The variable no longer exists in cc_ng_organism.py (#813).
# How: apply = count-guard -> timestamped backup -> delete by LINE PATTERN (never a line
#   number) -> name-only verify. reverse = byte-exact restore. NEVER prints file content
#   or values: only variable NAMES and paths (the file holds secrets).
#
# WHEN IT RUNS (S4 batch): strictly AFTER both merges are DEPLOYED --
#   (1) the NeuroGraph merge (cc_ng_organism.py no longer reads the variable), AND
#   (2) the docs merge (scripts/cc-ng-service.py preflight no longer REQUIRES it) --
#   and BEFORE the service restart that re-reads the environment. Running it earlier
#   makes the CURRENT preflight refuse launch ("missing canonical export").
#   Leaving the export in place is always safe, so if either merge is not live, do not run.
#
# Usage:   bash bashrc-drop-node-chars.sh apply|verify|reverse
#          BASHRC=/path/to/file overrides the target (tests use a temp file; default ~/.bashrc)
# -------------------
set -euo pipefail

TARGET_BASHRC="${BASHRC:-$HOME/.bashrc}"
PATTERN='^export CC_PITH_PROVIDER_NODE_CHARS='
NAME='CC_PITH_PROVIDER_NODE_CHARS'
KEEP=(CC_PITH_PROVIDER_ROOTS CC_PITH_PROVIDER_MEMBERS CC_PITH_PROVIDER_DEPTH
      CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS CC_PITH_PROVIDER_MAX_QUEST_CHARS)
LATEST="${TARGET_BASHRC}.bak-813.latest"       # holds the path of the newest backup
POSTSHA="${TARGET_BASHRC}.bak-813.postsha"     # sha256 of the file right after apply
REMOVED="${TARGET_BASHRC}.bak-813.line"        # the one removed line (a non-secret config line)

count_matches() { grep -c -- "$PATTERN" "$TARGET_BASHRC" || true; }

verify() {
  bash -n "$TARGET_BASHRC"                     # syntax only; does not execute the file
  echo "syntax: ok"
  echo "CC_PITH_PROVIDER_* exports now present (names only):"
  grep -oE '^export CC_PITH_PROVIDER_[A-Z_]+' "$TARGET_BASHRC" | sed 's/^export /  /'
  if [ "$(count_matches)" != "0" ]; then echo "FAIL: $NAME still present" >&2; return 1; fi
  for keep in "${KEEP[@]}"; do
    grep -q "^export ${keep}=" "$TARGET_BASHRC" || { echo "FAIL: $keep missing" >&2; return 1; }
  done
  echo "verify: $NAME absent; the ${#KEEP[@]} other provider exports present"
}

apply() {
  local n; n="$(count_matches)"
  if [ "$n" != "1" ]; then
    echo "refusing: expected exactly 1 line matching '$PATTERN', found $n (nothing changed)" >&2
    return 1
  fi
  local ts backup; ts="$(date +%Y%m%d-%H%M%S)"; backup="${TARGET_BASHRC}.bak-813-${ts}"
  cp -p -- "$TARGET_BASHRC" "$backup"
  grep -- "$PATTERN" "$TARGET_BASHRC" > "$REMOVED"
  sed -i "/${PATTERN}/d" "$TARGET_BASHRC"
  printf '%s\n' "$backup" > "$LATEST"
  sha256sum "$TARGET_BASHRC" | cut -d' ' -f1 > "$POSTSHA"
  echo "removed 1 line matching '$PATTERN' (name only); backup: $backup"
  verify
}

reverse() {
  [ -f "$LATEST" ] || { echo "refusing: no backup pointer ($LATEST)" >&2; return 1; }
  local backup; backup="$(cat "$LATEST")"
  [ -f "$backup" ] || { echo "refusing: backup missing: $backup" >&2; return 1; }
  if [ -f "$POSTSHA" ] && [ "$(sha256sum "$TARGET_BASHRC" | cut -d' ' -f1)" = "$(cat "$POSTSHA")" ]; then
    cp -p -- "$backup" "$TARGET_BASHRC"        # untouched since apply -> byte-exact restore
    echo "reversed: restored $TARGET_BASHRC byte-exact from $backup"
  else
    # Something else edited the file after apply (other S4 batch changes): do NOT
    # clobber them -- put back only the one removed line.
    [ -f "$REMOVED" ] || { echo "refusing: file changed since apply and removed-line record missing" >&2; return 1; }
    [ "$(count_matches)" = "0" ] || { echo "refusing: $NAME already present" >&2; return 1; }
    cat -- "$REMOVED" >> "$TARGET_BASHRC"
    echo "reversed: file changed since apply; re-appended the one removed line ($NAME) only"
  fi
  echo "$NAME present again: $(count_matches) line"
}

case "${1:-}" in
  apply) apply ;;
  verify) verify ;;
  reverse) reverse ;;
  *) echo "usage: $0 apply|verify|reverse   (BASHRC=/path overrides ~/.bashrc)" >&2; exit 2 ;;
esac
