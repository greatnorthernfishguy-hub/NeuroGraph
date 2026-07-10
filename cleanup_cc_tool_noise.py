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
# [2026-07-06] Claude Code (Sonnet 5) — Catch vector_db-only orphaned tool noise
# What: find_tool_noise() only ever matched nodes reachable via graph.nodes.items() --
#   any vector_db entry whose graph node had ALREADY been pruned (by #237's
#   _collect_orphan_nodes(), or any other removal path) was invisible to it, so its
#   tool:/bash: content was never deleted from vdb. Added find_orphan_vdb_tool_noise()
#   to scan vdb.content directly for tool:/bash: entries with no matching graph node,
#   and cleanup() now removes both classes (still-in-graph and orphan-only).
# Why:  Confirmed empirically on the laptop's live checkpoint (read-only diagnostic,
#   2026-07-06): only 5 of vdb's 12,219 entries were reachable via the graph (177 live
#   nodes total), but 9,085 vdb entries still carried tool:/bash: content -- 9,080 of
#   them orphaned (no graph node at all). This is why old tool-call content kept
#   surfacing via cc_pattern_completion_recall()'s direct vector search even after the
#   2026-07-05 cleanup pass: that pass's node-driven scan could only ever reach content
#   still attached to a live node, and CC's graph naturally orphan-prunes low-activity
#   nodes over time (the same #237 mechanism as Syl's graph) -- so tool-noise nodes that
#   had already been orphan-pruned by the time the first cleanup ran were untouched.
# How: Second scan directly over vdb.content.items() (not gated by graph membership),
#   skip anything already caught by find_tool_noise() to avoid double-counting.
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
    """Return node_ids (still in the graph) whose vector_db content is tool-call-derived."""
    matches = []
    for node_id, node in graph.nodes.items():
        meta = getattr(node, "metadata", None) or {}
        if meta.get("creation_mode") != "ingested":
            continue
        content = vdb.content.get(node_id, "")
        if content.startswith(_TOOL_PREFIXES):
            matches.append(node_id)
    return matches


def find_orphan_vdb_tool_noise(graph, vdb, already_matched: set) -> list:
    """Return vector_db-only node_ids (no corresponding graph node) with tool-call
    content -- content orphaned by graph pruning (#237) before the first cleanup ran,
    invisible to find_tool_noise()'s graph-driven scan."""
    matches = []
    for node_id, content in vdb.content.items():
        if node_id in already_matched or node_id in graph.nodes:
            continue
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

    graph_matches = find_tool_noise(graph, vdb)
    orphan_matches = find_orphan_vdb_tool_noise(graph, vdb, set(graph_matches))
    logger.info("Found %d tool-call-derived nodes still in graph, %d orphaned vdb-only "
                "entries (of %d total nodes, %d total vdb entries)",
                len(graph_matches), len(orphan_matches), before_nodes, before_vdb)

    if not apply:
        return {
            "status": "dry_run",
            "would_remove_graph": len(graph_matches),
            "would_remove_orphan_vdb": len(orphan_matches),
            "would_remove_total": len(graph_matches) + len(orphan_matches),
            "total_nodes_before": before_nodes, "total_vdb_before": before_vdb,
        }

    removed_graph = 0
    for node_id in graph_matches:
        if node_id in graph.nodes:
            graph.remove_node(node_id)
            removed_graph += 1
        vdb.delete(node_id)

    removed_orphan = 0
    for node_id in orphan_matches:
        if vdb.delete(node_id):
            removed_orphan += 1

    # #379: atomic writes + manifest refresh (see cc_threshold_rebaseline.py's
    # note). The manifest update is load-bearing HERE: this script mass-deletes
    # nodes, and a stale higher node-count in the manifest would make the
    # SaveGate refuse the daemon's next legitimate save.
    from checkpoint_guardian import atomic_file_write, read_manifest, write_manifest
    atomic_file_write(main_path, lambda p: graph.checkpoint(p))
    atomic_file_write(vectors_path, lambda p: vdb.save(p))
    m = read_manifest(main_path) or {}
    m.pop("version", None); m.pop("saved_at", None)
    m.update({"nodes": len(graph.nodes), "synapses": len(graph.synapses),
              "hyperedges": len(graph.hyperedges), "timestep": graph.timestep,
              "vdb_count": vdb.count(), "offline_pass": "cleanup_cc_tool_noise"})
    write_manifest(main_path, m)

    return {
        "status": "ok",
        "removed_graph": removed_graph,
        "removed_orphan_vdb": removed_orphan,
        "removed_total": removed_graph + removed_orphan,
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
