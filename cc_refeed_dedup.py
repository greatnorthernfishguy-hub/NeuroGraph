#!/usr/bin/env python3
# ---- Changelog ----
# [2026-07-08] Claude Code (Fable 5) — Post-refeed orphan dedup
# What: Offline pass deleting vdb entries that are BOTH orphaned (no graph node)
#   AND superseded by a re-embodied refeed twin (content sha1 in the refeed
#   journal AND cc:conv::<sha1> alive in the graph). Companion to cc_refeed.py.
# Why: The refeed (2026-07-07/08, 1575 memories) re-embodied orphaned content
#   under new cc:conv:: nodes but left the original orphan vdb entries in place.
#   Both match a query equally, so orphans hog top-k seed slots where the
#   orphan-seed fix (b2213df) correctly skips them — starving the harvest of
#   live seeds for exactly the historical queries the refeed was meant to
#   restore. Confirmed live 2026-07-08: lenia-history query returned an empty
#   Active Recall while recent-content queries fired fine.
# How: Same offline pattern as cleanup_cc_tool_noise.py — caller stops the
#   daemon first (single-writer). Triple gate per deletion: orphaned + hash in
#   journal + live cc:conv:: twin present. Anything failing any gate is kept.
#   Dry-run by default; --apply deletes and saves the vdb only (graph untouched).
# -------------------
"""Delete refeed-superseded orphan vdb entries. OFFLINE only -- stop the daemon first."""
from __future__ import annotations
import argparse
import hashlib
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cc_refeed_dedup")

_repo = Path(__file__).resolve().parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))


def find_superseded(graph, vdb, journal_hashes: set) -> list:
    """vdb entry ids that are orphaned AND journal-fed AND re-embodied live."""
    out = []
    for node_id, content in vdb.content.items():
        if node_id in graph.nodes:
            continue                                  # live -> keep
        h = hashlib.sha1(content.encode()).hexdigest()
        if h not in journal_hashes:
            continue                                  # never refed -> keep
        if f"cc:conv::{h}" not in graph.nodes:
            continue                                  # twin not alive -> keep
        out.append(node_id)
    return out


def run(main_path: str, vectors_path: str, journal_path: str, apply: bool = False) -> dict:
    from neuro_foundation import Graph
    from universal_ingestor import SimpleVectorDB

    graph = Graph()
    graph.restore(main_path)
    vdb = SimpleVectorDB()
    vdb.load(vectors_path)
    journal = set(Path(journal_path).read_text().split()) if Path(journal_path).exists() else set()

    before = vdb.count()
    victims = find_superseded(graph, vdb, journal)
    logger.info("Superseded orphans: %d of %d vdb entries (journal covers %d hashes)",
                len(victims), before, len(journal))
    if not apply:
        return {"status": "dry_run", "would_delete": len(victims), "vdb_before": before}

    deleted = sum(1 for nid in victims if vdb.delete(nid))
    vdb.save(vectors_path)
    return {"status": "ok", "deleted": deleted,
            "vdb_before": before, "vdb_after": vdb.count()}


def main() -> int:
    import os
    ws = os.path.expanduser("~/.claude/plugins/neurograph")
    ap = argparse.ArgumentParser(description="Delete refeed-superseded orphan vdb entries (OFFLINE only)")
    ap.add_argument("--main-checkpoint", default=os.path.join(ws, "checkpoints", "main.msgpack"))
    ap.add_argument("--vectors-checkpoint", default=os.path.join(ws, "checkpoints", "vectors.msgpack"))
    ap.add_argument("--journal", default=os.path.join(ws, "refeed_journal.txt"))
    ap.add_argument("--apply", action="store_true", help="Actually delete + save (default: dry run)")
    a = ap.parse_args()
    r = run(a.main_checkpoint, a.vectors_checkpoint, a.journal, apply=a.apply)
    print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
