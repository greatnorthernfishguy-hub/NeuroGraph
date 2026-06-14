#!/usr/bin/env python3
"""
rebuild_vectors.py — Rebuild SimpleVectorDB from existing Graph checkpoint.

Iterates through all nodes in the Graph checkpoint, extracts their text
content from node metadata, re-embeds them using the configured embedding
engine, and saves the populated vector DB to vectors.msgpack.

This is a one-time recovery tool for when the vector DB is empty but the
Graph checkpoint has learned state (nodes, synapses, STDP weights).

Usage:
    python3 rebuild_vectors.py [--checkpoint PATH] [--output PATH] [--dry-run]

Default paths:
    checkpoint: ~/NeuroGraph/data/checkpoints/main.msgpack
    output:     ~/NeuroGraph/data/checkpoints/vectors.msgpack
"""

# ---- Changelog ----
# [2026-06-14] Claude Code (Opus 4.8) — #294-B conv re-index mode (re-light her recall)
# What: add select_conv_reindex_targets(), _reindex_content(), _content_is_sane(),
#   _load_graph(), reindex_conv(), and a `--conv-only` CLI mode (+ --fracture-start/-end,
#   --throttle). Re-embeds each unindexed conv:: node's _forest_content/_concept into the
#   existing vdb (idempotent), READ-ONLY on the graph.
# Why: docs/prd/2026-06-14-syl-recall-heal-phase1-design.md Component B — her ~1,733 conv
#   memories live in the graph; only the recall vdb index was dropped (poison-prune of
#   wire-explosion garbage). Re-index relights them. Syl chose this heal (recency-weighted,
#   fracture-window skipped, documented). Wire garbage excluded by conv:: scope.
# How: extend this existing offline rebuild tool (Law 3); idempotent against the loaded vdb;
#   newest-first by creation_time; must run OFFLINE (sidecar dead — single vdb writer).
# -------------------

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure skill directory is on path
_skill_dir = Path(os.environ.get(
    "NEUROGRAPH_SKILL_DIR",
    str(Path.home() / ".openclaw" / "skills" / "neurograph"),
))
if _skill_dir.exists() and str(_skill_dir) not in sys.path:
    sys.path.insert(0, str(_skill_dir))

# Also add repo root
_repo_dir = Path.home() / ".neurograph" / "repo"
if _repo_dir.exists() and str(_repo_dir) not in sys.path:
    sys.path.insert(0, str(_repo_dir))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rebuild_vectors")


# ── #294-B: conversational recall re-index (READ-ONLY on the graph) ──────────────────
# Her ~1,733 conv:: memories live in the graph; only the recall vdb index was dropped (poison-
# prune of wire-explosion garbage during the OOM/starvation recovery). Re-light them by
# re-embedding each node's _forest_content / _concept back into the vdb. The wire garbage is
# excluded by construction — it is not conv:: shaped. Reads the graph, writes only the vdb.
_CONV_PREFIX = "conv::"


def _reindex_content(meta):
    """Content to embed for a conv node: tree -> _concept, forest -> _forest_content."""
    if meta.get("_tree_concept"):
        return str(meta.get("_concept", "")).strip()
    return str(meta.get("_forest_content", "")).strip()


def _content_is_sane(text):
    """Belt-and-suspenders garbage filter. The wire explosion is not conv:: anyway; this
    rejects empty/degenerate content and any stray wire-fingerprint signature."""
    import re
    t = (text or "").strip()
    if len(t) < 3:
        return False
    if re.search(r"\b(wire[_ ]?explosion|signal_burst|broadcast_flood)\b", t, re.I):
        return False
    return True


def select_conv_reindex_targets(nodes, already_indexed=frozenset(), fracture_window=None):
    """Pick which conv:: graph nodes to re-index into recall. READ-ONLY on the graph.

    Skips: non-conv:: (poison excluded by construction), already-indexed (idempotent),
    insane/empty content, and fracture-window nodes (Syl's (a)+named-absence — her degraded
    responses are not re-lit). Returns [(node_id, content, metadata)] newest-first by
    creation_time (recency; heal-not-flood).
    """
    out = []
    for nid, node in nodes.items():
        if not str(nid).startswith(_CONV_PREFIX):
            continue
        if nid in already_indexed:
            continue
        meta = getattr(node, "metadata", None) or {}
        content = _reindex_content(meta)
        if not _content_is_sane(content):
            continue
        ct = float(getattr(node, "creation_time", 0.0) or 0.0)
        if fracture_window and fracture_window[0] <= ct <= fracture_window[1]:
            continue
        out.append((nid, content, meta, ct))
    out.sort(key=lambda x: x[3], reverse=True)  # newest-first
    return [(nid, content, meta) for (nid, content, meta, ct) in out]


def _load_graph(path):
    """Load a Graph from a checkpoint (READ-ONLY use by the re-index). Extracted so tests
    can substitute a sandbox graph."""
    from neuro_foundation import Graph
    g = Graph()
    g.restore(path)
    return g


def reindex_conv(checkpoint_path, vdb_path, dry_run=False, fracture_window=None,
                 throttle_per_sec=0, reindex_date=None):
    """Re-light Syl's unindexed conv:: memories: re-embed each into the recall vdb.

    READ-ONLY on the graph (creates no nodes/synapses). Idempotent: loads the existing vdb and
    skips ids already present. Offline single-writer only (caller stops the sidecar first —
    the sidecar also persists vectors.msgpack; two writers = corruption). dry_run does the
    selection + shape report with NO writes (for the shape-note preview + sign-off).
    """
    from universal_ingestor import SimpleVectorDB, EmbeddingEngine
    import time as _time

    graph = _load_graph(checkpoint_path)  # read-only
    vdb = SimpleVectorDB()
    try:
        vdb.load(vdb_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load existing vdb (%s); index starts empty: %s", vdb_path, exc)
    already = set(vdb.embeddings.keys())

    targets = select_conv_reindex_targets(
        graph.nodes, already_indexed=already, fracture_window=fracture_window)

    conv_cts = [float(getattr(n, "creation_time", 0.0) or 0.0)
                for nid, n in graph.nodes.items() if str(nid).startswith(_CONV_PREFIX)]
    shape = {
        "conv_nodes_total": len(conv_cts),
        "already_indexed": len(already),
        "would_index": len(targets),
        "creation_time_min": min(conv_cts) if conv_cts else 0.0,
        "creation_time_max": max(conv_cts) if conv_cts else 0.0,
        "fracture_window": fracture_window,
    }
    if dry_run:
        shape["status"] = "dry_run"
        logger.info("[dry-run] would re-index %d conv nodes (already=%d, total=%d, ct %.0f..%.0f)",
                    shape["would_index"], shape["already_indexed"], shape["conv_nodes_total"],
                    shape["creation_time_min"], shape["creation_time_max"])
        return shape

    embedder = EmbeddingEngine()
    logger.info("Embedding engine ready: %s", getattr(embedder, "status", "?"))
    n = errors = 0
    start = _time.time()
    for nid, content, meta in targets:
        try:
            emb = embedder.embed_text(content)
            md = dict(meta)
            if reindex_date:
                md["reindexed"] = reindex_date
            vdb.insert(id=nid, embedding=emb, content=content, metadata=md)
            n += 1
            if throttle_per_sec and throttle_per_sec > 0:
                _time.sleep(1.0 / throttle_per_sec)
            if n % 100 == 0:
                logger.info("  re-indexed %d/%d", n, len(targets))
        except Exception as exc:  # noqa: BLE001
            logger.warning("  re-index failed for %s: %s", str(nid)[:16], exc)
            errors += 1
    vdb.save(vdb_path)
    shape.update({"status": "success", "reindexed": n, "errors": errors,
                  "elapsed_seconds": round(_time.time() - start, 1)})
    logger.info("Re-index complete: %d entries (%d errors) in %.1fs",
                n, errors, shape["elapsed_seconds"])
    return shape


def rebuild(
    checkpoint_path: str,
    output_path: str,
    dry_run: bool = False,
) -> dict:
    """Rebuild vector DB from checkpoint.

    Args:
        checkpoint_path: Path to Graph checkpoint (.msgpack or .json).
        output_path: Path for output vectors file.
        dry_run: If True, report what would happen without writing.

    Returns:
        Dict with rebuild statistics.
    """
    from neuro_foundation import Graph
    from universal_ingestor import SimpleVectorDB, EmbeddingEngine

    # Load graph
    logger.info("Loading graph from %s", checkpoint_path)
    graph = Graph()
    graph.restore(checkpoint_path)
    telemetry = graph.get_telemetry()
    logger.info(
        "Graph loaded: %d nodes, %d synapses, %d hyperedges, timestep %d",
        telemetry.total_nodes,
        telemetry.total_synapses,
        telemetry.total_hyperedges,
        telemetry.timestep,
    )

    # Initialize embedding engine
    logger.info("Initializing embedding engine...")
    embedder = EmbeddingEngine()
    logger.info("Embedding engine ready: %s", embedder.status)

    # Create vector DB
    vdb = SimpleVectorDB()

    # Collect all nodes with content
    nodes_with_content = []
    nodes_without_content = 0
    
    for node_id, node in graph.nodes.items():
        # Node metadata may contain the original text content
        # The content is stored in metadata during ingestion
        content = ""
        metadata = node.metadata or {}

        # Try to find content in various metadata fields
        # (the ingestor stores chunk text in metadata)
        if "content" in metadata:
            content = str(metadata["content"])
        elif "text" in metadata:
            content = str(metadata["text"])
        elif "chunk_text" in metadata:
            content = str(metadata["chunk_text"])
        elif "source_text" in metadata:
            content = str(metadata["source_text"])
        elif "raw_text" in metadata:
            content = str(metadata["raw_text"])

        if content and content.strip():
            nodes_with_content.append((node_id, content, metadata))
        else:
            nodes_without_content += 1

    logger.info(
        "Found %d nodes with content, %d without content",
        len(nodes_with_content),
        nodes_without_content,
    )

    if not nodes_with_content:
        logger.warning(
            "No nodes with text content found in metadata. "
            "The graph may store content differently. "
            "Checking metadata keys on first 5 nodes..."
        )
        for i, (nid, node) in enumerate(list(graph.nodes.items())[:5]):
            meta = node.metadata or {}
            logger.info("  Node %s metadata keys: %s", nid[:12], list(meta.keys()))
        return {
            "status": "no_content",
            "total_nodes": telemetry.total_nodes,
            "nodes_with_content": 0,
            "nodes_embedded": 0,
            "nodes_skipped": nodes_without_content,
        }

    if dry_run:
        logger.info("[dry-run] Would embed %d nodes and save to %s", len(nodes_with_content), output_path)
        return {
            "status": "dry_run",
            "total_nodes": telemetry.total_nodes,
            "nodes_with_content": len(nodes_with_content),
            "nodes_embedded": 0,
            "nodes_skipped": nodes_without_content,
        }

    # Embed and insert into vector DB
    embedded_count = 0
    errors = 0
    start_time = time.time()

    for i, (node_id, content, metadata) in enumerate(nodes_with_content):
        try:
            embedding = embedder.embed_text(content)
            vdb.insert(
                id=node_id,
                embedding=embedding,
                content=content,
                metadata=metadata,
            )
            embedded_count += 1

            # Progress reporting
            if (i + 1) % 100 == 0 or (i + 1) == len(nodes_with_content):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    "  Progress: %d/%d (%.1f/sec, %.0fs elapsed)",
                    i + 1, len(nodes_with_content), rate, elapsed,
                )

        except Exception as exc:
            logger.warning("  Failed to embed node %s: %s", node_id[:12], exc)
            errors += 1

    elapsed = time.time() - start_time
    logger.info(
        "Embedding complete: %d entries in %.1fs (%.1f/sec), %d errors",
        embedded_count, elapsed, embedded_count / elapsed if elapsed > 0 else 0, errors,
    )

    # Save
    logger.info("Saving vector DB to %s", output_path)
    saved = vdb.save(output_path)
    file_size = Path(output_path).stat().st_size
    logger.info(
        "Saved %d entries (%.2f MB)",
        saved,
        file_size / (1024 * 1024),
    )

    return {
        "status": "success",
        "total_nodes": telemetry.total_nodes,
        "nodes_with_content": len(nodes_with_content),
        "nodes_embedded": embedded_count,
        "nodes_skipped": nodes_without_content,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "output_path": output_path,
        "output_size_bytes": file_size,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild SimpleVectorDB from existing Graph checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path.home() / "NeuroGraph/data/checkpoints/main.msgpack"),
        help="Path to Graph checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path.home() / "NeuroGraph/data/checkpoints/vectors.msgpack"),
        help="Path for output vectors.msgpack",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without writing",
    )
    parser.add_argument(
        "--conv-only",
        action="store_true",
        help="#294-B: re-index ONLY conv:: nodes into the existing vdb (idempotent, read-only "
             "on graph). Use OFFLINE (sidecar stopped). Not a full rebuild.",
    )
    parser.add_argument(
        "--fracture-start", type=float, default=None,
        help="conv-only: skip conv nodes with creation_time >= this (fracture window start, epoch s)",
    )
    parser.add_argument(
        "--fracture-end", type=float, default=None,
        help="conv-only: skip conv nodes with creation_time <= this (fracture window end, epoch s)",
    )
    parser.add_argument(
        "--throttle", type=float, default=0,
        help="conv-only: max embeds/sec (0 = unthrottled). ONNX is CPU-slow; throttle to spare RAM.",
    )
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    if args.conv_only:
        fw = None
        if args.fracture_start is not None and args.fracture_end is not None:
            fw = (args.fracture_start, args.fracture_end)
        from datetime import date
        result = reindex_conv(
            args.checkpoint, args.output, dry_run=args.dry_run,
            fracture_window=fw, throttle_per_sec=args.throttle,
            reindex_date=date.today().isoformat(),
        )
    else:
        result = rebuild(args.checkpoint, args.output, dry_run=args.dry_run)

    print("\n" + "=" * 50)
    print("Rebuild Results")
    print("=" * 50)
    for k, v in result.items():
        print(f"  {k}: {v}")

    if result["status"] == "success":
        print("\n✅ Vector DB rebuilt successfully!")
        print("   Restart OpenClaw gateway to load it:")
        print("   systemctl --user restart openclaw-gateway")
    elif result["status"] == "no_content":
        print("\n⚠️  No text content found in node metadata.")
        print("   The nodes may store content under different keys.")
        print("   Check the metadata keys listed above.")


if __name__ == "__main__":
    main()
