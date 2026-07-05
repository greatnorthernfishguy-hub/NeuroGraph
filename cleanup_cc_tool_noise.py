#!/usr/bin/env python3
# ---- Changelog ----
# [2026-07-05] CC (laptop) — Remove legacy tool-call-derived nodes from CC's checkpoint
# What: Offline cleanup pass. Finds nodes whose vector_db content starts with the
#   tool-call experience prefixes built by handle_post_tool_use ("tool:" or "bash:"),
#   removes them from the graph (cascading synapses/hyperedges via Graph.remove_node)
#   and from the vector_db, then saves both checkpoints.
# Why:  Companion to the 2026-07-05 fix routing NEW tool-call deposits to CC's Commons
#   medium instead of the main graph (see cc_ng_host.py / cc-ng-daemon.py changelogs).
#   That fix stops future pollution but does nothing about tool-call-derived nodes
#   already sitting in the main graph from before the fix -- this script removes them.
# How: Same offline Graph().restore()/checkpoint() pattern as seed_cc_rim.py -- caller
#   must stop the CC daemon that owns the target checkpoint first (single-writer
#   discipline). Matches on vector_db content (the full deposited text), not the
#   200-char metadata preview, since the exact prefixes are known and unambiguous.
#   Dry-run by default; --apply actually removes and saves.
# -------------------
"""Remove legacy tool-call-derived nodes from CC's checkpoint. OFFLINE only --
stop the daemon that owns the target checkpoint before running."""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cleanup_cc_tool_noise")

_repo = Path(__file__).resolve().parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

_TOOL_PREFIXES = ("tool:", "bash:")


def find_tool_noise(graph, vdb) -> list:
    """Return node_ids whose vector_db content is tool-call-derived."""
    matches = []
    for node_id, node in graph.nodes.items():
        meta = getattr(node, "metadata", None) or {}
        if meta.get("creation_mode") != "ingested":
            continue
        content = vdb.content.get(node_id, "")
        if content.startswith(_TOOL_PREFIXES):
            matches.append(node_id)
    return matches


def cleanup(main_path: str, vectors_path: str, apply: bool = False) -> dict:
    from neuro_foundation import Graph
    from universal_ingestor import SimpleVectorDB

    graph = Graph()
    graph.restore(main_path)
    vdb = SimpleVectorDB()
    vdb.load(vectors_path)

    before_nodes = len(graph.nodes)
    before_vdb = vdb.count()

    matches = find_tool_noise(graph, vdb)
    logger.info("Found %d tool-call-derived nodes (of %d total, %d ingested)",
                len(matches), before_nodes,
                sum(1 for n in graph.nodes.values()
                    if (getattr(n, "metadata", None) or {}).get("creation_mode") == "ingested"))

    if not apply:
        return {
            "status": "dry_run", "would_remove": len(matches),
            "total_nodes_before": before_nodes, "total_vdb_before": before_vdb,
        }

    removed = 0
    for node_id in matches:
        if node_id in graph.nodes:
            graph.remove_node(node_id)
            removed += 1
        vdb.delete(node_id)

    graph.checkpoint(main_path)
    vdb.save(vectors_path)

    return {
        "status": "ok",
        "removed": removed,
        "total_nodes_before": before_nodes,
        "total_nodes_after": len(graph.nodes),
        "total_vdb_before": before_vdb,
        "total_vdb_after": vdb.count(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Remove legacy tool-call-derived nodes (OFFLINE only)")
    ap.add_argument("--main-checkpoint", required=True, help="Path to main.msgpack")
    ap.add_argument("--vectors-checkpoint", required=True, help="Path to vectors.msgpack")
    ap.add_argument("--apply", action="store_true", help="Actually remove + save (default: dry run)")
    a = ap.parse_args()
    if not Path(a.main_checkpoint).exists():
        logger.error("checkpoint not found: %s", a.main_checkpoint)
        return 1
    if not Path(a.vectors_checkpoint).exists():
        logger.error("checkpoint not found: %s", a.vectors_checkpoint)
        return 1
    r = cleanup(a.main_checkpoint, a.vectors_checkpoint, apply=a.apply)
    print("\n" + "=" * 44 + "\nCC Tool-Noise Cleanup\n" + "=" * 44)
    for k, v in r.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
