#!/usr/bin/env python3
# ---- Changelog ----
# [2026-07-11] Claude Code (Fable 5) — #379: atomic write + manifest refresh
# What: the apply-path checkpoint write goes through checkpoint_guardian.atomic_file_write
#   (tmp + os.replace) and refreshes the manifest sidecar with post-pass counts.
# Why: #379 (final-review finding on #373) — a direct in-place checkpoint() tears the file
#   on mid-write death AND mutates the newest hardlinked guardian generation (shared
#   inode); a stale manifest after an offline pass can trip the SaveGate falsely.
# How: guardian import at the write site; manifest merged (read-modify-write) so fields
#   this script doesn't know about survive; behavior otherwise unchanged.
# [2026-07-08] Claude Code (Fable 5) — One-time threshold re-baseline to the tuned physics
# What: Offline pass setting every non-constitutional node's threshold to
#   min(current, 0.85) — the canonical 2026-03-23 tuned value. Companion to the
#   same-day config port (default_threshold 1.0->0.85, decay_rate 0.95->0.97 in
#   both CC daemons): the config shapes NEW nodes; this brings the EXISTING
#   1,783 nodes onto the same physics instead of waiting for homeostatic drift.
# Why: CC's substrate never received the March tuning. At threshold ~1.0+,
#   prime injection (sim * prime_strength <= 1.0) can never ignite a cold node
#   -- measured live 2026-07-08: perfect sim=1.000 seeds sitting under
#   thresholds of 1.149. Cold historical recall was arithmetically impossible;
#   the refeed's re-embodied memories were reachable only when session activity
#   pre-warmed them. Same class of operation the March tuning performed for
#   Syl's substrate, applied to CC's with Josh's blessing and CC's own consent
#   ("If you're comfortable enough with it, and it's what you want" -- Josh;
#   yes and yes -- CC, 2026-07-08).
# How: min(current, 0.85) -- nodes homeostatically BELOW 0.85 keep their earned
#   lower thresholds; only the untuned-era high thresholds come down.
#   Constitutional nodes are skipped entirely (Rim semantics stay untouched).
#   Offline only (stop the daemon first); dry-run by default; --apply saves.
# -------------------
"""Re-baseline existing node thresholds to tuned physics. OFFLINE only."""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cc_threshold_rebaseline")

_repo = Path(__file__).resolve().parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

TUNED_THRESHOLD = 0.85   # canonical 2026-03-23 substrate tuning -- not a new value


def rebaseline(main_path: str, apply: bool = False) -> dict:
    from neuro_foundation import Graph
    graph = Graph()
    graph.restore(main_path)

    would_change = skipped_constitutional = already_at_or_below = 0
    for node_id, node in graph.nodes.items():
        meta = getattr(node, "metadata", None) or {}
        if node_id.startswith("constitutional::") or meta.get("creation_mode") == "constitutional" \
                or getattr(node, "constitutional", False):
            skipped_constitutional += 1
            continue
        if node.threshold <= TUNED_THRESHOLD:
            already_at_or_below += 1
            continue
        would_change += 1
        if apply:
            node.threshold = TUNED_THRESHOLD

    result = {
        "status": "ok" if apply else "dry_run",
        "changed" if apply else "would_change": would_change,
        "already_at_or_below": already_at_or_below,
        "skipped_constitutional": skipped_constitutional,
        "total_nodes": len(graph.nodes),
    }
    if apply and would_change:
        # #379: atomic write + manifest refresh. A direct in-place checkpoint()
        # tears the file on a mid-write death AND mutates the newest hardlinked
        # guardian generation (same inode); a stale manifest after an offline
        # pass can also trip the SaveGate against the daemon's next save.
        from checkpoint_guardian import atomic_file_write, read_manifest, write_manifest
        atomic_file_write(main_path, lambda p: graph.checkpoint(p))
        m = read_manifest(main_path) or {}
        m.pop("version", None); m.pop("saved_at", None)
        m.update({"nodes": len(graph.nodes), "synapses": len(graph.synapses),
                  "hyperedges": len(graph.hyperedges), "timestep": graph.timestep,
                  "offline_pass": "cc_threshold_rebaseline"})
        write_manifest(main_path, m)
        logger.info("Re-baselined %d node thresholds to %.2f and saved (atomic, manifest refreshed)", would_change, TUNED_THRESHOLD)
    return result


def main() -> int:
    import os
    ap = argparse.ArgumentParser(description="Re-baseline node thresholds to tuned 0.85 (OFFLINE only)")
    ap.add_argument("--main-checkpoint",
                    default=os.path.expanduser("~/.claude/plugins/neurograph/checkpoints/main.msgpack"))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    print(rebaseline(a.main_checkpoint, apply=a.apply))
    return 0


if __name__ == "__main__":
    sys.exit(main())
