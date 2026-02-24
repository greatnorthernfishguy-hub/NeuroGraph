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
    checkpoint: ~/.openclaw/neurograph/checkpoints/main.msgpack
    output:     ~/.openclaw/neurograph/checkpoints/vectors.msgpack
"""

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
        default=str(Path.home() / ".openclaw/neurograph/checkpoints/main.msgpack"),
        help="Path to Graph checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path.home() / ".openclaw/neurograph/checkpoints/vectors.msgpack"),
        help="Path for output vectors.msgpack",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without writing",
    )
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

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
