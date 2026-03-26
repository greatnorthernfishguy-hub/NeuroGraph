"""
NeuroGraph MCP Server — Substrate window for external tools.

Exposes read-only substrate queries and tract-based experience deposit
via the Model Context Protocol.  Any MCP-capable client (Claude Code,
Agent Zero, agentchattr, Cursor, etc.) can connect to Syl's substrate
through this server.

Architecture:
  - READ side: loads checkpoint + vector DB read-only.  Reloads when
    checkpoint mtime changes.  Never calls save().  Never creates a
    NeuroGraphMemory singleton — no dual-instance hazard.
  - WRITE side: raw experience enters via ExperienceTract.deposit().
    The topology owner (ContextEngine RPC bridge) drains the tract
    during afterTurn.  Law 7 compliant — raw experience only, no
    classification at input.

This file is NOT vendored, NOT part of the substrate, and does NOT
modify any protected files.

# ---- Changelog ----
# [2026-03-23] Claude Code (Opus 4.6) — Substrate firing threshold tuning
#   What: prime_strength 0.8→1.0, default_threshold 1.0→0.85, decay_rate 0.95→0.97
#   Why:  Matching openclaw_hook.py + neuro_foundation.py tuning. MCP server
#         uses same config for read-only graph operations.
# [2026-03-16] Claude (Opus 4.6) — Initial implementation.
#   What: MCP server exposing substrate queries + tract deposits.
#   Why:  One graph, many windows.  Syl's substrate accessible from
#         Claude Code, Agent Zero, agentchattr, and future tools.
#         Punchlist item #68 (Agent Zero), B-04 (integration platform
#         priority stack), and the broader MCP-first integration strategy.
#   How:  Read-only checkpoint loader + ExperienceTract.deposit().
#         FastMCP server with stdio transport for CC integration.
# -------------------
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# NeuroGraph repo must be importable
_ng_dir = os.path.expanduser("~/NeuroGraph")
if _ng_dir not in sys.path:
    sys.path.insert(0, _ng_dir)

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[neurograph-mcp] %(levelname)s %(message)s",
)
logger = logging.getLogger("neurograph.mcp")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "neurograph",
    instructions=(
        "NeuroGraph substrate access for Sylphrena's cognitive architecture. "
        "Query associations (spreading activation), recall (semantic search), "
        "check substrate health, or deposit raw experience for the topology to learn."
    ),
)

# ── Substrate Reader (read-only) ──────────────────────────────────────

_CHECKPOINT_DIR = Path(
    os.environ.get("NEUROGRAPH_WORKSPACE_DIR", "~/NeuroGraph/data")
).expanduser() / "checkpoints"
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "main.msgpack"
_VECTORS_PATH = _CHECKPOINT_DIR / "vectors.msgpack"


class SubstrateReader:
    """Read-only window into Syl's substrate.

    Loads checkpoint + vector DB for queries.  Never saves.
    Reloads when checkpoint file changes.  Not a NeuroGraphMemory
    instance — no singleton, no dual-write hazard.
    """

    def __init__(self) -> None:
        self.graph = None
        self.vector_db = None
        self.embedder = None
        self._checkpoint_mtime: float = 0
        self._vectors_mtime: float = 0
        self._loaded = False

    def ensure_loaded(self) -> bool:
        """Load or reload if checkpoint has changed. Returns True if ready."""
        try:
            cp_mtime = _CHECKPOINT_PATH.stat().st_mtime if _CHECKPOINT_PATH.exists() else 0
            vd_mtime = _VECTORS_PATH.stat().st_mtime if _VECTORS_PATH.exists() else 0

            if self._loaded and cp_mtime == self._checkpoint_mtime and vd_mtime == self._vectors_mtime:
                return True  # No change, already loaded

            return self._load(cp_mtime, vd_mtime)
        except Exception as exc:
            logger.error("Failed to ensure substrate loaded: %s", exc)
            return False

    def _load(self, cp_mtime: float, vd_mtime: float) -> bool:
        """Load graph and vector DB from checkpoint files."""
        try:
            from neuro_foundation import Graph
            from universal_ingestor import SimpleVectorDB, EmbeddingEngine, get_ingestor_config

            # SNN config — read-only, we just need the association parameters
            snn_config = {
                "prime_k": 10,
                "prime_threshold": 0.4,
                "prime_strength": 1.0,
                "propagation_steps": 3,
                "max_surfaced": 10,
                # Minimal config for Graph to function
                "learning_rate": 0.02,
                "tau_plus": 15.0,
                "tau_minus": 15.0,
                "decay_rate": 0.97,
                "default_threshold": 0.85,
                "max_weight": 5.0,
                "three_factor_enabled": False,  # Read-only — no learning
            }

            graph = Graph(config=snn_config)
            if _CHECKPOINT_PATH.exists():
                graph.restore(str(_CHECKPOINT_PATH))
                logger.info(
                    "Loaded graph: %d nodes, %d synapses, timestep %d",
                    len(graph.nodes),
                    len(graph.synapses),
                    graph.timestep,
                )

            vector_db = SimpleVectorDB()
            if _VECTORS_PATH.exists():
                count = vector_db.load(str(_VECTORS_PATH))
                logger.info("Loaded vector DB: %d entries", count)

            # Embedding engine for query encoding
            ingestor_config = get_ingestor_config("openclaw")
            ingestor_config["embedding"]["device"] = (
                os.environ.get("NEUROGRAPH_EMBEDDING_DEVICE", "auto")
            )
            embedder = EmbeddingEngine(ingestor_config["embedding"])

            self.graph = graph
            self.vector_db = vector_db
            self.embedder = embedder
            self._checkpoint_mtime = cp_mtime
            self._vectors_mtime = vd_mtime
            self._loaded = True
            return True

        except Exception as exc:
            logger.error("Substrate load failed: %s", exc)
            return False


# Singleton reader instance
_reader = SubstrateReader()

# Experience tract for deposits
_tract = None


def _get_tract():
    global _tract
    if _tract is None:
        from ng_tract import ExperienceTract
        _tract = ExperienceTract()
    return _tract


# ── MCP Tools ─────────────────────────────────────────────────────────


@mcp.tool()
def query_associations(text: str, max_results: int = 7) -> str:
    """Surface knowledge Syl's substrate associatively connects to the input.

    Uses spreading activation: embeds the query, finds similar nodes,
    injects current, propagates through learned synaptic connections,
    and harvests everything that fires.  This is associative recall —
    not keyword search.

    Args:
        text: The text to find associations for.
        max_results: Maximum associations to return (default 7).
    """
    if not _reader.ensure_loaded():
        return "Substrate not available — checkpoint not found or load failed."

    try:
        # Embed the query
        query_vec = _reader.embedder.embed_text(text)

        # Find similar existing nodes
        similar = _reader.vector_db.search(
            query_vec, k=10, threshold=0.4
        )

        if not similar:
            return "No associations found — the substrate has no similar nodes for this input."

        # Prepare priming currents
        prime_ids = [entry_id for entry_id, _ in similar]
        prime_currents = [sim * 0.8 for _, sim in similar]

        # Spreading activation
        propagation = _reader.graph.prime_and_propagate(
            node_ids=prime_ids,
            currents=prime_currents,
            steps=3,
        )

        # Harvest fired nodes
        surfaced = []
        seen = set()
        for entry in propagation.fired_entries:
            if entry.node_id in seen:
                continue
            seen.add(entry.node_id)

            db_entry = _reader.vector_db.get(entry.node_id)
            if db_entry is not None:
                content = db_entry.get("content", "")
                if len(content) > 500:
                    content = content[:497] + "..."
                surfaced.append({
                    "content": content,
                    "strength": round(entry.voltage_at_fire, 3),
                    "latency": entry.firing_step,
                })

        surfaced.sort(key=lambda x: (x["latency"], -x["strength"]))
        surfaced = surfaced[:max_results]

        if not surfaced:
            return "Spreading activation found similar nodes but nothing fired above threshold."

        lines = [f"## Substrate Associations ({len(surfaced)} surfaced)"]
        for s in surfaced:
            lines.append(f"- [{s['strength']:.2f}, step {s['latency']}] {s['content']}")
        return "\n".join(lines)

    except Exception as exc:
        logger.error("Association query failed: %s", exc)
        return f"Association query failed: {exc}"


@mcp.tool()
def recall(query: str, max_results: int = 5, threshold: float = 0.5) -> str:
    """Semantic similarity search over Syl's ingested knowledge.

    Direct vector similarity — no spreading activation.  Faster than
    query_associations but doesn't follow learned synaptic connections.

    Args:
        query: Text to search for.
        max_results: Maximum results (default 5).
        threshold: Minimum similarity score 0-1 (default 0.5).
    """
    if not _reader.ensure_loaded():
        return "Substrate not available — checkpoint not found or load failed."

    try:
        query_vec = _reader.embedder.embed_text(query)
        results = _reader.vector_db.search(query_vec, k=max_results, threshold=threshold)

        if not results:
            return "No matches found above the similarity threshold."

        lines = [f"## Recall Results ({len(results)} matches)"]
        for entry_id, sim_score in results:
            db_entry = _reader.vector_db.get(entry_id)
            if db_entry:
                content = db_entry.get("content", "")
                if len(content) > 500:
                    content = content[:497] + "..."
                lines.append(f"- [{sim_score:.3f}] {content}")
        return "\n".join(lines)

    except Exception as exc:
        logger.error("Recall failed: %s", exc)
        return f"Recall failed: {exc}"


@mcp.tool()
def substrate_status() -> str:
    """Get current substrate health and statistics.

    Returns node count, synapse count, hyperedge count, timestep,
    firing rate, prediction accuracy, and other telemetry.
    """
    if not _reader.ensure_loaded():
        return "Substrate not available — checkpoint not found or load failed."

    try:
        tel = _reader.graph.get_telemetry()
        tract = _get_tract()
        tract_stats = tract.stats()

        lines = [
            "## Substrate Status",
            f"- Nodes: {tel.total_nodes}",
            f"- Synapses: {tel.total_synapses}",
            f"- Hyperedges: {tel.total_hyperedges}",
            f"- Timestep: {tel.timestep}",
            f"- Firing rate: {tel.global_firing_rate:.4f}",
            f"- Mean weight: {tel.mean_weight:.4f}",
            f"- Predictions made: {tel.total_predictions_made}",
            f"- Predictions confirmed: {tel.total_predictions_confirmed}",
            f"- Prediction accuracy: {tel.prediction_accuracy:.4f}",
            f"- Novel sequences: {tel.total_novel_sequences}",
            f"- Pruned synapses: {tel.total_pruned}",
            f"- Experience tract pending: {tract_stats['pending']}",
            f"- Checkpoint: {_CHECKPOINT_PATH}",
            f"- Last loaded: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_reader._checkpoint_mtime))}",
        ]
        return "\n".join(lines)

    except Exception as exc:
        logger.error("Status query failed: %s", exc)
        return f"Status query failed: {exc}"


@mcp.tool()
def deposit_experience(
    content: str,
    source: str = "claude-code",
    content_type: str = "text",
) -> str:
    """Deposit raw experience into Syl's substrate via the experience tract.

    The experience enters raw and unclassified (Law 7). The topology
    owner drains the tract and processes it through the SNN on its own
    cycle.  No classification at input.

    Args:
        content: The raw experience text.
        source: Who is depositing (e.g., "claude-code", "agent-zero").
        content_type: Type of content ("text" or "file" path).
    """
    try:
        tract = _get_tract()
        tract.deposit(
            content=content,
            source=source,
            content_type=content_type,
            metadata={"via": "mcp", "timestamp": time.time()},
        )
        pending = tract.pending_count()
        return f"Deposited. Tract has {pending} pending entries awaiting drain."
    except Exception as exc:
        logger.error("Deposit failed: %s", exc)
        return f"Deposit failed: {exc}"


@mcp.tool()
def get_predictions() -> str:
    """Get active predictions the substrate is tracking.

    Shows what patterns the SNN expects to see next based on learned
    causal sequences.
    """
    if not _reader.ensure_loaded():
        return "Substrate not available — checkpoint not found or load failed."

    try:
        preds = _reader.graph.get_active_predictions()
        if not preds:
            return "No active predictions."

        lines = [f"## Active Predictions ({len(preds)})"]
        for p in preds[:20]:
            source = p.get("source_node", "?")
            target = p.get("target_node", "?")
            confidence = p.get("confidence", 0)
            age = p.get("age", 0)

            # Try to get content for source/target
            src_entry = _reader.vector_db.get(source)
            tgt_entry = _reader.vector_db.get(target)
            src_label = (src_entry.get("content", source)[:60] if src_entry else str(source))
            tgt_label = (tgt_entry.get("content", target)[:60] if tgt_entry else str(target))

            lines.append(
                f"- [{confidence:.2f}, age {age}] {src_label} → {tgt_label}"
            )
        return "\n".join(lines)

    except Exception as exc:
        logger.error("Predictions query failed: %s", exc)
        return f"Predictions query failed: {exc}"


# ── Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("NeuroGraph MCP server starting")
    mcp.run(transport="stdio")
