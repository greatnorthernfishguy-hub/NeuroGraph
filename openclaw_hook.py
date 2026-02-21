"""
NeuroGraph OpenClaw Integration Hook

Singleton NeuroGraphMemory class that integrates NeuroGraph's cognitive
architecture into the OpenClaw AI assistant framework. Provides automatic
ingestion, STDP learning, semantic recall, and cross-session persistence.

Usage:
    from openclaw_hook import NeuroGraphMemory

    ng = NeuroGraphMemory.get_instance()
    ng.on_message("User said something interesting about recursion")
    context = ng.recall("recursion")
    print(ng.stats())
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from neuro_foundation import Graph, CheckpointMode
from neurograph_paths import get_neurograph_home
from universal_ingestor import (
    UniversalIngestor,
    SimpleVectorDB,
    SourceType,
    get_ingestor_config,
)

logger = logging.getLogger("neurograph")


# OpenClaw-tuned SNN config: fast learning, tight causal windows
OPENCLAW_SNN_CONFIG = {
    "learning_rate": 0.02,
    "tau_plus": 15.0,
    "tau_minus": 15.0,
    "A_plus": 1.0,
    "A_minus": 1.2,
    "decay_rate": 0.95,
    "default_threshold": 1.0,
    "refractory_period": 2,
    "max_weight": 5.0,
    "target_firing_rate": 0.05,
    "scaling_interval": 100,
    "weight_threshold": 0.01,
    "grace_period": 500,
    "inactivity_threshold": 1000,
    "co_activation_window": 5,
    "initial_sprouting_weight": 0.1,
    # Predictive coding
    "prediction_threshold": 3.0,
    "prediction_pre_charge_factor": 0.3,
    "prediction_window": 10,
    "prediction_chain_decay": 0.7,
    "prediction_max_chain_depth": 3,
    "prediction_confirm_bonus": 0.01,
    "prediction_error_penalty": 0.02,
    "prediction_max_active": 1000,
    "surprise_sprouting_weight": 0.1,
    "three_factor_enabled": False,
    # Hypergraph
    "he_pattern_completion_strength": 0.3,
    "he_member_weight_lr": 0.05,
    "he_threshold_lr": 0.01,
    "he_discovery_window": 10,
    "he_discovery_min_co_fires": 5,
    "he_discovery_min_nodes": 3,
    "he_consolidation_overlap": 0.8,
    "he_experience_threshold": 100,
}


class NeuroGraphMemory:
    """Singleton cognitive memory layer for OpenClaw integration.

    Wraps NeuroGraph's Graph + UniversalIngestor + SimpleVectorDB into a
    single interface for message-level ingestion, learning, and recall.

    Auto-saves every ``auto_save_interval`` messages (default 10).
    Loads from the latest checkpoint on initialization if one exists.
    """

    _instance: Optional[NeuroGraphMemory] = None

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._workspace_dir = Path(
            workspace_dir or str(get_neurograph_home())
        ).expanduser()

        self._checkpoint_dir = self._workspace_dir / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._checkpoint_path = self._checkpoint_dir / "main.msgpack"

        # Merge user config over OpenClaw defaults
        snn_config = {**OPENCLAW_SNN_CONFIG, **(config or {})}
        self.graph = Graph(config=snn_config)

        # Restore from checkpoint if one exists
        if self._checkpoint_path.exists():
            try:
                self.graph.restore(str(self._checkpoint_path))
                logger.info(
                    "Restored graph from %s (%d nodes, %d synapses)",
                    self._checkpoint_path,
                    len(self.graph.nodes),
                    len(self.graph.synapses),
                )
            except Exception as exc:
                logger.warning("Failed to restore checkpoint: %s", exc)

        # Vector DB for semantic search
        self.vector_db = SimpleVectorDB()

        # Ingestor with OpenClaw project config
        ingestor_config = get_ingestor_config("openclaw")
        self.ingestor = UniversalIngestor(
            self.graph, self.vector_db, config=ingestor_config
        )

        self._message_count = 0
        self.auto_save_interval = 10

    @classmethod
    def get_instance(
        cls,
        workspace_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NeuroGraphMemory:
        """Return the singleton instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls(workspace_dir=workspace_dir, config=config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def on_message(self, text: str, source_type: Optional[SourceType] = None) -> Dict[str, Any]:
        """Ingest a message, run one STDP learning step, and auto-save.

        Args:
            text: Raw message content to ingest.
            source_type: Override auto-detection (TEXT, MARKDOWN, CODE, etc.).

        Returns:
            Dict with ingestion stats and learning results.
        """
        if not text or not text.strip():
            return {"status": "skipped", "reason": "empty_input"}

        # Stage 1-5: Extract → Chunk → Embed → Register → Associate
        result = self.ingestor.ingest(text, source_type=source_type)

        # Run SNN learning step
        step_result = self.graph.step()

        # Update novelty probation for ingested nodes
        graduated = self.ingestor.update_probation()

        self._message_count += 1

        # Auto-save
        if self._message_count % self.auto_save_interval == 0:
            self.save()

        return {
            "status": "ingested",
            "nodes_created": len(result.nodes_created),
            "synapses_created": len(result.synapses_created),
            "hyperedges_created": len(result.hyperedges_created),
            "chunks": result.chunks_created,
            "fired": len(step_result.fired_node_ids),
            "graduated": len(graduated),
            "message_count": self._message_count,
        }

    def recall(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Semantic similarity search over ingested knowledge.

        Args:
            query: Text to search for.
            k: Maximum results to return.
            threshold: Minimum similarity score (0-1).

        Returns:
            List of dicts with 'content', 'similarity', 'node_id', 'metadata'.
        """
        return self.ingestor.query_similar(query, k=k, threshold=threshold)

    def step(self, n: int = 1) -> List[Any]:
        """Run N SNN learning steps without ingestion."""
        results = []
        for _ in range(n):
            results.append(self.graph.step())
        return results

    def save(self) -> str:
        """Save graph state to checkpoint. Returns the checkpoint path."""
        self.graph.checkpoint(str(self._checkpoint_path), mode=CheckpointMode.FULL)
        logger.info("Checkpoint saved to %s", self._checkpoint_path)
        return str(self._checkpoint_path)

    def stats(self) -> Dict[str, Any]:
        """Return current graph statistics and telemetry."""
        tel = self.graph.get_telemetry()
        return {
            "version": "0.4.0",
            "timestep": tel.timestep,
            "nodes": tel.total_nodes,
            "synapses": tel.total_synapses,
            "hyperedges": tel.total_hyperedges,
            "firing_rate": round(tel.global_firing_rate, 4),
            "mean_weight": round(tel.mean_weight, 4),
            "predictions_made": tel.total_predictions_made,
            "predictions_confirmed": tel.total_predictions_confirmed,
            "prediction_accuracy": round(tel.prediction_accuracy, 4),
            "novel_sequences": tel.total_novel_sequences,
            "pruned": tel.total_pruned,
            "sprouted": tel.total_sprouted,
            "vector_db_count": self.vector_db.count(),
            "checkpoint": str(self._checkpoint_path),
            "message_count": self._message_count,
        }

    def ingest_file(self, path: str, source_type: Optional[SourceType] = None) -> Dict[str, Any]:
        """Ingest a file from disk."""
        p = Path(path).expanduser()
        if not p.exists():
            return {"status": "error", "reason": f"File not found: {path}"}

        content = p.read_text(errors="replace")

        # Auto-detect source type from extension
        if source_type is None:
            ext = p.suffix.lower()
            type_map = {
                ".py": SourceType.CODE,
                ".js": SourceType.CODE,
                ".ts": SourceType.CODE,
                ".md": SourceType.MARKDOWN,
                ".html": SourceType.URL,
                ".htm": SourceType.URL,
                ".pdf": SourceType.PDF,
            }
            source_type = type_map.get(ext, SourceType.TEXT)

        return self.on_message(content, source_type=source_type)

    def ingest_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Dict[str, Any]]:
        """Ingest all matching files from a directory.

        Args:
            directory: Path to directory.
            extensions: File extensions to include (e.g. ['.py', '.md']).
                       Default: ['.py', '.js', '.ts', '.md', '.txt']
            recursive: Whether to recurse into subdirectories.

        Returns:
            List of ingestion results per file.
        """
        if extensions is None:
            extensions = [".py", ".js", ".ts", ".md", ".txt"]

        d = Path(directory).expanduser()
        if not d.is_dir():
            return [{"status": "error", "reason": f"Not a directory: {directory}"}]

        results = []
        pattern = "**/*" if recursive else "*"
        for fp in sorted(d.glob(pattern)):
            if fp.is_file() and fp.suffix.lower() in extensions:
                res = self.ingest_file(str(fp))
                res["file"] = str(fp)
                results.append(res)

        # Save after batch ingestion
        self.save()
        return results
