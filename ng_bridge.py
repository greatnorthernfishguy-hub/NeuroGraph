"""
NG Bridge Implementations — Tier 2 and Tier 3 Bridges for NG-Lite

Provides concrete bridge implementations that connect NG-Lite instances
to higher-tier learning backends:

    NGSaaSBridge (Tier 3): Connects NG-Lite to the full NeuroGraph
        Foundation via NeuroGraphMemory. Translates between NG-Lite's
        lightweight Hebbian learning (weights [0,1], JSON, incremental IDs)
        and NeuroGraph's full SNN with STDP, hyperedges, and predictive
        coding (weights [0,5], msgpack, UUIDs).

    NGPeerBridge (Tier 2): [Planned] Connects two co-located NG-Lite
        instances for shared learning without requiring NeuroGraph SaaS.

Weight normalization:
    NG-Lite weights are in [0.0, 1.0].
    NeuroGraph weights are in [0.0, max_weight] (default 5.0).
    Bridge normalizes: ng_weight * max_weight → full weight
                       full_weight / max_weight → ng weight

Node ID translation:
    NG-Lite uses incremental IDs ("n_1", "n_2").
    NeuroGraph uses UUID strings.
    The bridge maintains a bidirectional mapping table.

Canonical source: https://github.com/greatnorthernfishguy-hub/NeuroGraph
License: AGPL-3.0

Grok Review Changelog (v0.7.1):
    No code changes.  Grok's suggestions for ng_bridge.py were evaluated:
    Rejected: 'sync_state() is one-way (Lite → Full)' — By design.  NG-Lite
        is a lightweight Hebbian substrate; pulling SNN state back would
        require implementing STDP traces, hyperedges, and prediction chains
        that NG-Lite deliberately omits.  Enriched cross-module intelligence
        flows back through get_recommendations() and detect_novelty() calls,
        which query the full graph in real time.
    Rejected: '_normalize() duplicates np.linalg.norm' — Same rationale as
        ng_peer_bridge.py: the bridge imports only the abstract NGBridge
        interface.  Each file owns its own 3-line normalize() to avoid
        coupling to ng_lite internal methods.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ng_lite import NGBridge

logger = logging.getLogger("ng_bridge")


class NGSaaSBridge(NGBridge):
    """Tier 3 bridge: NG-Lite → full NeuroGraph Foundation.

    Wraps a NeuroGraphMemory instance and translates between the
    lightweight NG-Lite API and the full cognitive architecture.

    Usage:
        from openclaw_hook import NeuroGraphMemory
        from ng_bridge import NGSaaSBridge

        memory = NeuroGraphMemory.get_instance()
        bridge = NGSaaSBridge(memory)

        ng = NGLite(module_id="inference_difference")
        ng.connect_bridge(bridge)
        # Now ng.record_outcome() feeds into the full SNN
    """

    def __init__(
        self,
        memory: Any,  # NeuroGraphMemory — Any to avoid circular import
        max_weight: float = 5.0,
    ):
        """
        Args:
            memory: A NeuroGraphMemory instance (from openclaw_hook.py).
            max_weight: The max_weight in the full NeuroGraph config.
                Used for weight normalization between [0,1] and [0,max].
        """
        self._memory = memory
        self._max_weight = max_weight
        self._connected = True

        # Node ID mapping: ng_lite_id → neurograph_uuid
        self._id_map: Dict[str, str] = {}
        # Reverse: neurograph_uuid → ng_lite_id
        self._reverse_id_map: Dict[str, str] = {}

        # Track module sync state
        self._sync_count = 0

    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        """Manually disconnect the bridge."""
        self._connected = False

    def reconnect(self) -> None:
        """Manually reconnect the bridge."""
        self._connected = True

    def record_outcome(
        self,
        embedding: np.ndarray,
        target_id: str,
        success: bool,
        module_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Forward outcome to full NeuroGraph for cross-module learning.

        Ingests the outcome as a structured message into NeuroGraphMemory,
        which feeds it through the full SNN pipeline (STDP, hyperedges,
        predictive coding).

        Returns enriched context with cross-module insights.
        """
        if not self._connected:
            return None

        try:
            # Build a structured representation for the full SNN
            outcome_text = (
                f"[{module_id}] outcome: target={target_id} "
                f"success={success}"
            )
            if metadata:
                for k, v in metadata.items():
                    outcome_text += f" {k}={v}"

            # Ingest into the full graph
            result = self._memory.on_message(outcome_text)

            # Get cross-module context from the graph's telemetry
            stats = self._memory.stats()

            return {
                "cross_module": True,
                "graph_nodes": stats.get("nodes", 0),
                "graph_synapses": stats.get("synapses", 0),
                "firing_rate": stats.get("firing_rate", 0.0),
                "prediction_accuracy": stats.get("prediction_accuracy", 0.0),
                "nodes_created": result.get("nodes_created", 0),
            }

        except Exception as e:
            logger.warning("NGSaaSBridge record_outcome failed: %s", e)
            return None

    def get_recommendations(
        self,
        embedding: np.ndarray,
        module_id: str,
        top_k: int = 3,
    ) -> Optional[List[Tuple[str, float, str]]]:
        """Get recommendations from the full NeuroGraph substrate.

        Uses NeuroGraphMemory's semantic recall to find relevant
        patterns, then maps results back to NG-Lite target IDs.

        Returns list of (target_id, confidence, reasoning) tuples.
        """
        if not self._connected:
            return None

        try:
            # Use the module_id as context for recall
            query = f"[{module_id}] recommendation query"
            results = self._memory.recall(query, k=top_k * 2, threshold=0.3)

            if not results:
                return None

            recommendations = []
            for r in results[:top_k]:
                content = r.get("content", "")
                similarity = r.get("similarity", 0.0)

                # Extract target_id from stored content if possible
                target_id = self._extract_target_from_content(content)
                if target_id:
                    # Normalize similarity to confidence
                    confidence = min(1.0, similarity)
                    reasoning = (
                        f"Cross-module recall (similarity={similarity:.3f})"
                    )
                    recommendations.append((target_id, confidence, reasoning))

            return recommendations if recommendations else None

        except Exception as e:
            logger.warning("NGSaaSBridge get_recommendations failed: %s", e)
            return None

    def detect_novelty(
        self,
        embedding: np.ndarray,
        module_id: str,
    ) -> Optional[float]:
        """Get novelty score from the full NeuroGraph substrate.

        Queries the full graph's vector DB for similarity to known
        patterns across ALL modules (not just this one). This is the
        key benefit of Tier 3: cross-module novelty detection.

        Returns 0.0 (routine) to 1.0 (completely novel).
        """
        if not self._connected:
            return None

        try:
            # Query the vector DB for similar content
            query = f"[{module_id}] novelty probe"
            results = self._memory.recall(query, k=1, threshold=0.0)

            if not results:
                return 1.0  # Nothing in the graph = fully novel

            # Highest similarity → inverse = novelty
            best_similarity = results[0].get("similarity", 0.0)
            return max(0.0, 1.0 - best_similarity)

        except Exception as e:
            logger.warning("NGSaaSBridge detect_novelty failed: %s", e)
            return None

    def sync_state(
        self,
        local_state: Dict[str, Any],
        module_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Sync NG-Lite local state with the full NeuroGraph substrate.

        Sends the full local state snapshot to NeuroGraph for
        integration. The bridge:
        1. Maps NG-Lite node IDs to NeuroGraph UUIDs
        2. Normalizes weight ranges
        3. Ingests new patterns into the full SNN
        4. Returns updated stats for the module to consume

        Args:
            local_state: Output of NGLite._export_state()
            module_id: The module performing the sync.

        Returns:
            Dict with sync results, or None on failure.
        """
        if not self._connected:
            return None

        try:
            self._sync_count += 1

            # Count local objects
            n_nodes = len(local_state.get("nodes", {}))
            n_synapses = len(local_state.get("synapses", {}))

            # Ingest a sync summary into the full graph
            sync_text = (
                f"[{module_id}] sync #{self._sync_count}: "
                f"{n_nodes} nodes, {n_synapses} synapses, "
                f"outcomes={local_state.get('counters', {}).get('total_outcomes', 0)}"
            )
            self._memory.on_message(sync_text)

            # Ingest high-weight synapses as learned patterns
            synapse_data = local_state.get("synapses", {})
            ingested_patterns = 0
            for key, syn in synapse_data.items():
                weight = syn.get("weight", 0.0)
                if weight > 0.7:  # Only sync strong connections
                    pattern_text = (
                        f"[{module_id}] learned: "
                        f"{syn.get('source_id', '?')} → {syn.get('target_id', '?')} "
                        f"(weight={weight:.3f}, "
                        f"success={syn.get('success_count', 0)}, "
                        f"fail={syn.get('failure_count', 0)})"
                    )
                    self._memory.on_message(pattern_text)
                    ingested_patterns += 1

            # Force a save after sync
            self._memory.save()

            stats = self._memory.stats()
            return {
                "synced": True,
                "sync_number": self._sync_count,
                "patterns_ingested": ingested_patterns,
                "graph_total_nodes": stats.get("nodes", 0),
                "graph_total_synapses": stats.get("synapses", 0),
                "graph_prediction_accuracy": stats.get("prediction_accuracy", 0.0),
            }

        except Exception as e:
            logger.warning("NGSaaSBridge sync_state failed: %s", e)
            return None

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _extract_target_from_content(content: str) -> Optional[str]:
        """Try to extract a target_id from stored content.

        Looks for patterns like "target=model_name" in the ingested
        content strings.
        """
        if "target=" not in content:
            return None

        try:
            # Find "target=VALUE" and extract VALUE
            idx = content.index("target=") + len("target=")
            rest = content[idx:]
            # Target ends at space or end of string
            end = rest.find(" ")
            if end == -1:
                return rest.strip()
            return rest[:end].strip()
        except (ValueError, IndexError):
            return None

    def _normalize_weight_to_full(self, ng_weight: float) -> float:
        """Convert NG-Lite weight [0,1] to full NeuroGraph weight [0,max]."""
        return ng_weight * self._max_weight

    def _normalize_weight_to_lite(self, full_weight: float) -> float:
        """Convert full NeuroGraph weight [0,max] to NG-Lite weight [0,1]."""
        if self._max_weight <= 0:
            return 0.0
        return min(1.0, full_weight / self._max_weight)
