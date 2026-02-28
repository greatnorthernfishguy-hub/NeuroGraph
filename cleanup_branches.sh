#!/bin/bash
# NeuroGraph Branch Cleanup Script
# Generated 2026-02-28
#
# This script deletes stale remote branches that are either:
#   - Already merged into main (7 branches)
#   - Trivial/unwanted unmerged changes (3 branches)
#
# It leaves 5 branches with real unmerged work for manual review.
# It does NOT touch main or any code files.
#
# Usage: ./cleanup_branches.sh

set -e

echo "=== NeuroGraph Branch Cleanup ==="
echo ""

# -------------------------------------------------------
# GROUP 1: Already merged into main (safe — all work is in main)
# -------------------------------------------------------
MERGED_BRANCHES=(
    "claude/add-update-mechanism-Xt2pY"
    "claude/ces-initialization-check-BqKBL"
    "claude/enable-file-uploads-3XKYC"
    "claude/fix-ingestor-cli-timeout-pNNae"
    "claude/fix-neurograph-api-keys-0MnFy"
    "claude/fix-neurograph-issues-eFyqc"
    "claude/neurograph-user-guide-5vOB6"
)

echo "Deleting 7 already-merged branches..."
for branch in "${MERGED_BRANCHES[@]}"; do
    echo "  Deleting $branch"
    git push origin --delete "$branch" 2>&1 || echo "  (already deleted or not found)"
done

echo ""

# -------------------------------------------------------
# GROUP 2: Unmerged but trivial/unwanted (safe to delete)
# -------------------------------------------------------
# - grok-review-optimizations: only leftover comment wording fixes;
#   the real Grok optimizations were already merged in PR #17-18
# - neurograph-phase-1-setup: only adds changelog text to code comments
# - predictive-coding-engine: BSL 1.1 license change; AGPL is already on main
TRIVIAL_BRANCHES=(
    "claude/grok-review-optimizations-Xbu7F"
    "claude/neurograph-phase-1-setup-sJNS6"
    "claude/predictive-coding-engine-jfr6u"
)

echo "Deleting 3 trivial/unwanted unmerged branches..."
for branch in "${TRIVIAL_BRANCHES[@]}"; do
    echo "  Deleting $branch"
    git push origin --delete "$branch" 2>&1 || echo "  (already deleted or not found)"
done

echo ""

# -------------------------------------------------------
# GROUP 3: Kept for review (NOT deleted)
# -------------------------------------------------------
echo "=== 5 branches KEPT for your review ==="
echo ""
echo "  deploy-neurograph-bzx2g"
echo "    neurograph_paths.py — unified path resolution (+339 lines)"
echo ""
echo "  fix-git-updater-divergent-W488E"
echo "    GitUpdater shallow clone fix + .gitignore expansion (+53 lines)"
echo ""
echo "  implement-module-integration-spec-vjgGa"
echo "    Module Integration Spec v2: ng_ecosystem.py, openclaw_adapter.py (+2,394 lines)"
echo ""
echo "  neurograph-auto-knowledge-ZNN3l"
echo "    neurograph-patch MANIFEST + GUI fallback for Phase 7 (+21 lines)"
echo ""
echo "  review-neurograph-spec-LYOPe"
echo "    Module discovery canonical paths, remove ghost locations (+22 lines)"
echo ""
echo "To review a branch:  git diff main...origin/claude/<branch-name>"
echo "To merge a branch:   git checkout main && git merge origin/claude/<branch-name>"
echo ""
echo "=== Cleanup complete ==="

# Clean up remote tracking refs
git remote prune origin 2>/dev/null || true
