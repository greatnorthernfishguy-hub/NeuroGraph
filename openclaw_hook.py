"""
NeuroGraph OpenClaw Integration Hook

Singleton NeuroGraphMemory class that integrates NeuroGraph's cognitive
architecture into the OpenClaw AI assistant framework. Provides automatic
ingestion, STDP learning, semantic recall, and cross-session persistence.

NeuroGraph acts as the Tier 3 SNN backend for the E-T Systems ecosystem.
When peer modules (TrollGuard, The-Inference-Difference, Cricket) are
co-located on the same host, NeuroGraphMemory:
  - Writes learning events to the shared learning directory so peers
    can absorb patterns via NGPeerBridge (Tier 2)
  - Provides the full SNN substrate that peers upgrade to via
    NGSaaSBridge (Tier 3)
  - Participates in the ET Module Manager for unified discovery,
    status reporting, and coordinated updates

Writes structured operational logs to ``{workspace}/memory/`` so that
OpenClaw's memory system can parse ingestion events, learning progress,
and recall results without relying on stdout.

Usage:
    from openclaw_hook import NeuroGraphMemory

    ng = NeuroGraphMemory.get_instance()
    ng.on_message("User said something interesting about recursion")
    context = ng.recall("recursion")
    print(ng.stats())

# ---- Changelog ----
# [2026-02-22] Claude (Opus 4.6) — CES integration (Phase 9).
#   What: Added Cognitive Enhancement Suite — StreamParser (real-time
#         Ollama embedding + node nudging), ActivationPersistence (JSON
#         sidecar for cross-session voltage state), SurfacingMonitor
#         (priority queue of relevant concepts for prompt injection),
#         CESMonitor (health context + HTTP dashboard + rotating logger).
#   Why:  CES adds real-time cognitive capabilities: continuous attention
#         streaming, activation warmth across sessions, and automatic
#         surfacing of relevant knowledge without explicit search.
#   Settings: ces.enabled defaults to True, all CES imports guarded by
#         try/except so core NeuroGraph works without CES files present.
#   How:  CES modules initialized in __init__ after peer bridge.
#         on_message() feeds stream parser + calls surfacing monitor.
#         save() writes activation sidecar.  stats() includes CES status.
#
# [2026-02-17] Claude (Opus 4.6) — ET Module Manager integration.
#   What: Added NGPeerBridge connection, shared learning event writing,
#         ET Module Manager registration, peer module discovery, and
#         Tier 3 upgrade offering via get_peer_modules().
#   Why:  NeuroGraph is the Tier 3 SNN backend for all E-T Systems
#         modules.  This integration enables automatic cross-module
#         learning: when NeuroGraph ingests or learns, it writes events
#         to the shared directory so sibling modules benefit.
#   Settings: peer_bridge_enabled defaults to True, sync_interval=50
#         (more frequent than default 100 because NeuroGraph processes
#         more events), shared_dir=~/.et_modules/shared_learning/.
#   How:  NGPeerBridge initialized in __init__ (guarded by try/except
#         for graceful degradation).  on_message() writes learning
#         events after ingestion.  stats() includes peer bridge status.
# -------------------
#
# ---- Grok Review Changelog (v0.7.1) ----
# Accepted: Added file size guard in ingest_file() — warns and skips files
#     above 50MB to prevent excessive memory use when ingesting large binaries
#     that were accidentally placed in the ingest path.
# Rejected: 'No locks around graph access — concurrent on_message() could
#     race' — NeuroGraphMemory is a singleton within a single Python process.
#     OpenClaw calls on_message() sequentially per session.  Adding a
#     threading.Lock would add overhead with zero benefit.  If multi-threaded
#     access is ever needed, the lock should be added at the caller level
#     (e.g., an async wrapper), not inside the singleton.
# Rejected: '_load_checkpoint() assumes msgpack always succeeds' — Lines
#     150-161 already wrap graph.restore() in try/except Exception, log the
#     error, and continue with a fresh graph.  This was implemented in the
#     original Phase 5 code.
# Rejected: 'Config Overload: no merge with user overrides' — Line 147
#     does exactly this: {**OPENCLAW_SNN_CONFIG, **(config or {})} merges
#     user config over defaults, with user keys taking precedence.
# Rejected: 'Auto-knowledge ranking lacks dedup' — _harvest_associations()
#     lines 374-377 explicitly deduplicate via a `seen` set before ranking.
# -------------------------------------------
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from neuro_foundation import Graph, CheckpointMode, PropagationResult
from universal_ingestor import (
    UniversalIngestor,
    SimpleVectorDB,
    SourceType,
    get_ingestor_config,
    MEDIA_EXTENSIONS,
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
    # Auto-knowledge / Associative recall
    "auto_knowledge_enabled": True,
    "prime_k": 10,
    "prime_threshold": 0.4,
    "prime_strength": 0.8,
    "propagation_steps": 3,
    "max_surfaced": 10,
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
            workspace_dir
            or os.environ.get("NEUROGRAPH_WORKSPACE_DIR", "~/.openclaw/neurograph")
        ).expanduser()

        self._checkpoint_dir = self._workspace_dir / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._memory_dir = self._workspace_dir / "memory"
        self._memory_dir.mkdir(parents=True, exist_ok=True)

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

        # Ingestor with OpenClaw project config, respecting embedding_device
        ingestor_config = get_ingestor_config("openclaw")

        # Allow callers / env to override the embedding device mode
        embedding_device = (
            (config or {}).get("embedding_device")
            or os.environ.get("NEUROGRAPH_EMBEDDING_DEVICE")
            or "auto"
        )
        ingestor_config["embedding"]["device"] = embedding_device

        self.ingestor = UniversalIngestor(
            self.graph, self.vector_db, config=ingestor_config
        )

        # Log embedding backend status to memory/ for OpenClaw to parse
        self._write_memory_event("embedding_status", self.ingestor.embedder.status)

        self._message_count = 0
        self.auto_save_interval = 10

        # --- ET Module Manager: Peer bridge for cross-module learning ---
        # NeuroGraph is the Tier 3 backend.  We also participate as a
        # Tier 2 peer so sibling modules can absorb our learning events
        # from the shared directory even without a direct SaaS connection.
        self._peer_bridge = None
        peer_config = (config or {}).get("peer_bridge", {})
        if peer_config.get("enabled", True):
            try:
                from ng_peer_bridge import NGPeerBridge
                self._peer_bridge = NGPeerBridge(
                    module_id="neurograph",
                    shared_dir=peer_config.get("shared_dir"),
                    sync_interval=peer_config.get("sync_interval", 50),
                    relevance_threshold=peer_config.get(
                        "relevance_threshold", 0.3
                    ),
                )
                logger.info("NGPeerBridge connected for cross-module learning")
            except Exception as exc:
                logger.info(
                    "NGPeerBridge not available (standalone mode): %s", exc
                )

        # --- CES: Cognitive Enhancement Suite ---
        # Optional real-time cognitive modules: stream parser (Ollama
        # embedding + node nudging), activation persistence (cross-session
        # voltage state), surfacing monitor (priority queue of relevant
        # concepts), and CES monitoring (health + HTTP dashboard + logs).
        self._ces_config = None
        self._stream_parser = None
        self._activation_persistence = None
        self._surfacing_monitor = None
        self._ces_monitor = None

        ces_conf = (config or {}).get("ces", {})
        if ces_conf.get("enabled", True):
            try:
                from ces_config import load_ces_config
                from stream_parser import StreamParser
                from activation_persistence import ActivationPersistence
                from surfacing import SurfacingMonitor
                from ces_monitoring import CESMonitor

                self._ces_config = load_ces_config(ces_conf)
                self._stream_parser = StreamParser(
                    self.graph,
                    self.vector_db,
                    self._ces_config,
                    fallback_embedder=self.ingestor.embedder.embed_text,
                )
                self._activation_persistence = ActivationPersistence(
                    self._ces_config
                )
                self._surfacing_monitor = SurfacingMonitor(
                    self.graph, self.vector_db, self._ces_config
                )
                self._ces_monitor = CESMonitor(self, self._ces_config)
                self._ces_monitor._surfacing_monitor = self._surfacing_monitor

                # Restore activation state if checkpoint exists
                if self._checkpoint_path.exists():
                    self._activation_persistence.restore(
                        self.graph, str(self._checkpoint_path)
                    )

                self._ces_monitor.start()
                logger.info("CES modules initialized")
            except Exception as exc:
                logger.info("CES not available: %s", exc)

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
    # Memory logging (structured output for OpenClaw)
    # ------------------------------------------------------------------

    def _write_memory_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write a structured event to the memory/ directory.

        Each event is a JSON line appended to ``memory/events.jsonl``.
        OpenClaw's memory system can tail this file for ingestion/learning
        events instead of parsing stdout.
        """
        event = {
            "timestamp": time.time(),
            "event": event_type,
            "data": data,
        }
        try:
            events_path = self._memory_dir / "events.jsonl"
            with open(events_path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write memory event: %s", exc)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def on_message(self, text: str, source_type: Optional[SourceType] = None) -> Dict[str, Any]:
        """Ingest a message, run one STDP learning step, and auto-save.

        When ``auto_knowledge_enabled`` is True (the default), this method
        also performs **spreading activation harvest**: it primes similar
        existing nodes, propagates activation through the SNN's learned
        synaptic structure, and returns any knowledge that "lights up" as
        a ``surfaced`` list.  This is the cortex-like recall — you don't
        search for it, the network *just knows*.

        Args:
            text: Raw message content to ingest.
            source_type: Override auto-detection (TEXT, MARKDOWN, CODE, etc.).

        Returns:
            Dict with ingestion stats, learning results, and surfaced
            knowledge (if auto_knowledge_enabled).
        """
        if not text or not text.strip():
            return {"status": "skipped", "reason": "empty_input"}

        # Stage 1-5: Extract → Chunk → Embed → Register → Associate
        result = self.ingestor.ingest(text, source_type=source_type)
        new_node_ids = set(result.nodes_created)

        # --- AUTO-KNOWLEDGE: Spreading Activation Harvest ---
        surfaced: List[Dict[str, Any]] = []
        snn_config = self.graph.config
        if snn_config.get("auto_knowledge_enabled", True) and self.vector_db.count() > 0:
            surfaced = self._harvest_associations(text, new_node_ids)

        # Run SNN learning step (separate from propagation — this one
        # applies STDP, structural plasticity, predictions, etc.)
        step_result = self.graph.step()

        # CES: Feed stream parser (async background processing)
        if self._stream_parser is not None:
            self._stream_parser.feed(text)

        # CES: Surfacing monitor — scan fired nodes for relevant concepts
        ces_surfaced: List[Dict[str, Any]] = []
        if self._surfacing_monitor is not None:
            self._surfacing_monitor.after_step(step_result)
            ces_surfaced = self._surfacing_monitor.get_surfaced()

        # Update novelty probation for ingested nodes
        graduated = self.ingestor.update_probation()

        self._message_count += 1

        # Auto-save
        if self._message_count % self.auto_save_interval == 0:
            self.save()

        event_data = {
            "status": "ingested",
            "nodes_created": len(result.nodes_created),
            "synapses_created": len(result.synapses_created),
            "hyperedges_created": len(result.hyperedges_created),
            "chunks": result.chunks_created,
            "fired": len(step_result.fired_node_ids),
            "graduated": len(graduated),
            "message_count": self._message_count,
            "surfaced": surfaced,
            "ces_surfaced": ces_surfaced,
        }

        # Write to memory/ for OpenClaw consumption
        self._write_memory_event("ingestion", event_data)

        # Write learning event to shared directory for peer modules.
        # This is how sibling modules (TrollGuard, TID, etc.) absorb
        # NeuroGraph's learning without a direct SaaS connection.
        self._write_peer_learning_event(text, result, step_result)

        return event_data

    def _harvest_associations(
        self,
        text: str,
        exclude_node_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic priming + spreading activation harvest.

        Embeds the input text, finds similar existing nodes via the vector DB,
        injects current into those nodes, runs N SNN steps, and harvests
        everything that fires.  The result is knowledge the network
        *associatively connects* with the input — no explicit search needed.

        Returns:
            List of surfaced knowledge dicts sorted by association strength.
        """
        if exclude_node_ids is None:
            exclude_node_ids = set()

        snn_config = self.graph.config
        prime_k = snn_config.get("prime_k", 10)
        prime_threshold = snn_config.get("prime_threshold", 0.4)
        prime_strength = snn_config.get("prime_strength", 0.8)
        propagation_steps = snn_config.get("propagation_steps", 3)
        max_surfaced = snn_config.get("max_surfaced", 10)

        try:
            # Embed the input and find similar existing nodes
            query_vec = self.ingestor.embedder.embed_text(text)
            similar = self.vector_db.search(
                query_vec, k=prime_k, threshold=prime_threshold
            )

            # Filter out newly created nodes (they ARE the input)
            prime_ids = []
            prime_currents = []
            for entry_id, sim_score in similar:
                if entry_id not in exclude_node_ids:
                    prime_ids.append(entry_id)
                    prime_currents.append(sim_score * prime_strength)

            if not prime_ids:
                return []

            # Spreading activation through learned synaptic connections
            propagation = self.graph.prime_and_propagate(
                node_ids=prime_ids,
                currents=prime_currents,
                steps=propagation_steps,
            )

            # Harvest content from fired nodes
            surfaced = []
            seen = set()
            for entry in propagation.fired_entries:
                if entry.node_id in exclude_node_ids:
                    continue  # Skip input nodes
                if entry.node_id in seen:
                    continue  # Deduplicate
                seen.add(entry.node_id)

                db_entry = self.vector_db.get(entry.node_id)
                if db_entry is not None:
                    surfaced.append({
                        "node_id": entry.node_id,
                        "content": db_entry.get("content", ""),
                        "metadata": db_entry.get("metadata", {}),
                        "latency": entry.firing_step,
                        "strength": entry.voltage_at_fire,
                        "was_predicted": entry.was_predicted,
                    })

            # Sort: lower latency first, then higher strength
            surfaced.sort(key=lambda x: (x["latency"], -x["strength"]))
            return surfaced[:max_surfaced]

        except Exception as exc:
            logger.debug("Auto-knowledge harvest failed: %s", exc)
            return []

    def _write_peer_learning_event(
        self, text: str, result: Any, step_result: Any
    ) -> None:
        """Write a learning event to the shared peer directory.

        Called after each ingestion so sibling modules can absorb
        NeuroGraph's patterns via NGPeerBridge.  Uses the ingestor's
        embedding engine to generate the event embedding.
        """
        if self._peer_bridge is None:
            return

        try:
            import numpy as np

            # Embed the ingested text for cross-module similarity matching
            embedding = self.ingestor.embedder.embed_text(text)

            # Record as a peer learning event
            self._peer_bridge.record_outcome(
                embedding=embedding,
                target_id=f"ingestion_{self._message_count}",
                success=True,
                module_id="neurograph",
                metadata={
                    "nodes_created": len(result.nodes_created),
                    "fired": len(step_result.fired_node_ids),
                    "text_preview": text[:200],
                },
            )
        except Exception as exc:
            logger.debug("Peer learning event write failed: %s", exc)

    def get_peer_modules(self) -> List[Dict[str, Any]]:
        """Discover peer E-T Systems modules on this host.

        NeuroGraph is the Tier 3 SNN backend.  This method finds
        co-located modules that could benefit from a full SNN upgrade.

        Returns:
            List of dicts with module_id, display_name, version, tier.
        """
        try:
            from et_modules.manager import ETModuleManager
            manager = ETModuleManager()
            statuses = manager.status()
            peers = []
            for mid, status in statuses.items():
                if mid == "neurograph":
                    continue
                peers.append({
                    "module_id": mid,
                    "display_name": status.manifest.display_name,
                    "version": status.manifest.version,
                    "health": status.health,
                    "tier": status.tier,
                    "ng_lite_connected": status.ng_lite_connected,
                })
            return peers
        except Exception as exc:
            logger.debug("Peer module discovery failed: %s", exc)
            return []

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

    def associate(self, text: str, k: int = 10, steps: int = 3) -> List[Dict[str, Any]]:
        """Associative recall: surface knowledge the network connects to this input.

        Unlike ``recall()`` which does pure vector similarity (cosine search),
        this routes through the SNN's learned synaptic structure — surfacing
        knowledge based on causal connections, pattern completion, and
        prediction chains.  This is the difference between searching a
        database and *remembering*.

        Args:
            text: Input text to associate from.
            k: Maximum results to return.
            steps: SNN propagation steps (more = deeper associations).

        Returns:
            List of dicts with 'content', 'metadata', 'latency', 'strength',
            'was_predicted', 'node_id'.
        """
        if not text or not text.strip():
            return []

        # Temporarily override config for this call
        old_max = self.graph.config.get("max_surfaced", 10)
        old_steps = self.graph.config.get("propagation_steps", 3)
        self.graph.config["max_surfaced"] = k
        self.graph.config["propagation_steps"] = steps
        try:
            return self._harvest_associations(text)
        finally:
            self.graph.config["max_surfaced"] = old_max
            self.graph.config["propagation_steps"] = old_steps

    def step(self, n: int = 1) -> List[Any]:
        """Run N SNN learning steps without ingestion."""
        results = []
        for _ in range(n):
            results.append(self.graph.step())
        return results

    def save(self) -> str:
        """Save graph state to checkpoint. Returns the checkpoint path."""
        self.graph.checkpoint(str(self._checkpoint_path), mode=CheckpointMode.FULL)
        # CES: Save activation sidecar alongside checkpoint
        if self._activation_persistence is not None:
            self._activation_persistence.save(
                self.graph, str(self._checkpoint_path)
            )
        logger.info("Checkpoint saved to %s", self._checkpoint_path)
        return str(self._checkpoint_path)

    def stats(self) -> Dict[str, Any]:
        """Return current graph statistics and telemetry."""
        tel = self.graph.get_telemetry()
        result = {
            "version": "0.6.0",
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
            "memory_dir": str(self._memory_dir),
            "embedding": self.ingestor.embedder.status,
            "message_count": self._message_count,
            "auto_knowledge": self.graph.config.get("auto_knowledge_enabled", True),
        }

        # Peer bridge status
        if self._peer_bridge is not None:
            result["peer_bridge"] = self._peer_bridge.get_stats()
        else:
            result["peer_bridge"] = {"connected": False}

        # CES subsystem status
        if self._ces_config is not None:
            result["ces"] = {
                "stream_parser": (
                    self._stream_parser.get_stats()
                    if self._stream_parser
                    else None
                ),
                "surfacing": (
                    self._surfacing_monitor.get_stats()
                    if self._surfacing_monitor
                    else None
                ),
                "persistence": (
                    self._activation_persistence.get_stats()
                    if self._activation_persistence
                    else None
                ),
                "monitor": (
                    self._ces_monitor.get_health()
                    if self._ces_monitor
                    else None
                ),
            }

        return result

    def ces_stats(self) -> Dict[str, Any]:
        """Return dedicated CES (Cognitive Enhancement Suite) statistics.

        Returns a dict with status of each CES subsystem: stream_parser,
        surfacing, persistence, monitor.  Returns {"enabled": False} when
        CES is not initialized.
        """
        if self._ces_config is None:
            return {"enabled": False}

        return {
            "enabled": True,
            "stream_parser": (
                self._stream_parser.get_stats()
                if self._stream_parser
                else None
            ),
            "surfacing": (
                self._surfacing_monitor.get_stats()
                if self._surfacing_monitor
                else None
            ),
            "persistence": (
                self._activation_persistence.get_stats()
                if self._activation_persistence
                else None
            ),
            "monitor": (
                self._ces_monitor.get_health()
                if self._ces_monitor
                else None
            ),
        }

    def ingest_file(self, path: str, source_type: Optional[SourceType] = None) -> Dict[str, Any]:
        """Ingest a file from disk.

        Files above 50 MB are skipped to prevent excessive memory use
        (Grok review: large file guard).
        """
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"status": "error", "reason": f"File not found: {p}"}

        # Guard against very large files (Grok review optimization)
        max_file_bytes = 50 * 1024 * 1024  # 50 MB
        try:
            file_size = p.stat().st_size
            if file_size > max_file_bytes:
                logger.warning(
                    "Skipping %s — file size %d bytes exceeds 50 MB limit",
                    p, file_size,
                )
                return {
                    "status": "skipped",
                    "reason": f"File too large ({file_size} bytes, limit {max_file_bytes})",
                }
        except OSError:
            pass  # stat failed, proceed anyway

        # Auto-detect source type from extension
        if source_type is None:
            ext = p.suffix.lower()
            type_map = {
                ".py": SourceType.CODE,
                ".js": SourceType.CODE,
                ".ts": SourceType.CODE,
                ".java": SourceType.CODE,
                ".go": SourceType.CODE,
                ".rs": SourceType.CODE,
                ".c": SourceType.CODE,
                ".cpp": SourceType.CODE,
                ".rb": SourceType.CODE,
                ".php": SourceType.CODE,
                ".md": SourceType.MARKDOWN,
                ".markdown": SourceType.MARKDOWN,
                ".html": SourceType.HTML,
                ".htm": SourceType.HTML,
                ".pdf": SourceType.PDF,
                ".json": SourceType.JSON,
                ".csv": SourceType.CSV,
            }
            source_type = type_map.get(ext)
            if source_type is None and ext in MEDIA_EXTENSIONS:
                source_type = SourceType.MEDIA
            if source_type is None:
                source_type = SourceType.TEXT

        # Media files are handled by the extractor directly (it reads
        # metadata from the file path, not the file content).
        if source_type == SourceType.MEDIA:
            result = self.ingestor.ingest(str(p), source_type=SourceType.MEDIA)
            return {
                "status": "ingested",
                "nodes_created": len(result.nodes_created),
                "synapses_created": len(result.synapses_created),
                "chunks": result.chunks_created,
                "media_type": result.metadata.get("extraction_metadata", {}).get("media_type", "unknown"),
            }

        # PDF files are handled by the extractor directly (reads the file).
        if source_type == SourceType.PDF:
            result = self.ingestor.ingest(str(p), source_type=SourceType.PDF)
            step_result = self.graph.step()
            return {
                "status": "ingested",
                "nodes_created": len(result.nodes_created),
                "synapses_created": len(result.synapses_created),
                "chunks": result.chunks_created,
                "fired": len(step_result.fired_node_ids),
            }

        content = p.read_text(errors="replace")
        return self.on_message(content, source_type=source_type)

    def ingest_url(self, url: str) -> Dict[str, Any]:
        """Fetch and ingest content from a URL."""
        return self.on_message(url, source_type=SourceType.URL)

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
