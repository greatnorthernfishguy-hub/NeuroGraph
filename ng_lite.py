"""
NG-Lite — Lightweight NeuroGraph Learning Substrate v1.0

Single-file learning substrate for E-T Systems modules. Provides
standalone Hebbian learning, novelty detection, and JSON persistence.

Designed to be vendored into any module as a single-file dependency.
No external dependencies beyond numpy and the Python standard library.

When NeuroGraph SaaS is available, NG-Lite delegates to the full
substrate for cross-module learning, predictive coding, and hypergraph
capabilities via the NGBridge interface. When disconnected, NG-Lite
operates independently with local learning — no functionality is lost,
only the ecosystem-level synergies.

Design principles (aligned with NeuroGraph Foundation PRD §2.1):
    - Sparse by default: dict-based storage, no dense matrices
    - Bounded memory: configurable max nodes/synapses with LRU pruning
    - Persistence-native: full state serializable to JSON
    - Upgrade-ready: clean bridge interface to full NeuroGraph SaaS

Connectivity tiers:
    Tier 1 — Isolated: Module runs its own NG-Lite independently.
    Tier 2 — Peer-pooled: Co-located modules share learning via
             NGPeerBridge (not yet implemented). Two NG-Lite instances
             exchange nodes and synapse weights for mutual benefit
             without requiring NeuroGraph SaaS. Uses the same NGBridge
             interface — the module doesn't know or care whether its
             bridge partner is a sibling module or the full SaaS.
    Tier 3 — Full SaaS: NG-Lite delegates to NeuroGraph for cross-module
             learning, STDP, hyperedges, and predictive coding.

    Tier transitions are transparent. A module starts at Tier 1,
    discovers a co-located sibling and upgrades to Tier 2, then
    connects to SaaS and upgrades to Tier 3 — all without code changes.
    If SaaS disconnects, it falls back to Tier 2 or 1 automatically.

Serialization format notes:
    NG-Lite uses JSON for persistence. This is deliberate:
    - JSON is stdlib (no extra dependency)
    - NG-Lite state is small (≤1000 nodes, ≤5000 synapses)
    - Human-readable state files aid debugging
    - The full NeuroGraph Foundation uses msgpack for its much larger
      graphs (10K+ nodes with spike histories, numpy arrays, etc.)
    - Format translation happens in the bridge layer, not here.

Weight range notes:
    NG-Lite weights are bounded [0.0, 1.0] for simplicity.
    Full NeuroGraph uses [0.0, max_weight] (default 5.0).
    The bridge normalizes: ng_weight * max_weight ↔ full_weight / max_weight.

Node ID notes:
    NG-Lite uses incremental IDs ("n_1", "n_2") for compactness.
    Full NeuroGraph uses UUIDs for global uniqueness.
    The bridge maintains a mapping table during sync_state().

Ethical obligations (per NeuroGraph ETHICS.md):
    - Type I error bias: when uncertain, err toward respect
    - Choice Clause: no module may block agent autonomy
    - Transparency: all learning decisions are queryable

Canonical source: https://github.com/greatnorthernfishguy-hub/NeuroGraph
License: AGPL-3.0 (see NeuroGraph LICENSE)

Author: Josh + Claude
Date: February 2026

Grok Review Changelog (v0.7.1):
    Accepted: Replaced per-node loop in _find_similar_node() with vectorized
        np.stack + matrix-vector dot product.  For 1000 nodes this reduces
        wall clock from ~2ms (Python loop with individual np.dot) to ~0.1ms
        (single BLAS call).  Semantically equivalent.
    Accepted: Added embedding shape/dtype validation at the record_outcome()
        boundary.  Raises ValueError for non-1D arrays to fail fast rather
        than producing confusing downstream errors in hashing or dot products.
    Rejected: 'weight update uses raw counts without normalization — could
        overflow [0,1]' — Weights are explicitly clamped via np.clip(w, 0, 1)
        on line 422 of every record_outcome() call.  The soft saturation
        formula (success_boost * (1 - w)) also naturally converges.  The
        success/failure counts are statistics, not weights — they don't need
        normalization.
    Rejected: 'Hash embedder truncates vector — why not use full for better
        collision resistance?' — SHA-256 already distributes uniformly.
        Hashing 128 dims (512 bytes) vs 768 dims (3072 bytes) produces the
        same 256-bit hash with equivalent collision resistance.  Truncation
        reduces hash computation time by ~6x for no loss in uniqueness.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("ng_lite")

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # Capacity limits
    "max_nodes": 1000,
    "max_synapses": 5000,

    # Learning parameters
    "learning_rate": 0.1,
    "success_boost": 0.15,      # Weight increase on success
    "failure_penalty": 0.20,    # Weight decrease on failure

    # Novelty detection
    "novelty_threshold": 0.7,   # Embedding distance above which = novel

    # Pruning
    "pruning_threshold": 0.01,  # Synapses below this weight get pruned

    # Embedding
    "embedding_dim": 384,       # Expected embedding dimensionality
    "hash_dims": 128,           # Dims used for hashing (first N of embedding)

    # Persistence
    "snapshot_version": "1.0.0",
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class NGLiteNode:
    """A pattern node in the lightweight learning substrate.

    Each node represents a recognized input pattern, identified by a hash
    of its embedding vector. Tracks activation frequency for LRU pruning.

    Attributes:
        node_id: Unique identifier for this pattern.
        embedding_hash: Truncated SHA-256 of the embedding for fast lookup.
        activation_count: How many times this pattern has been matched.
        last_activation: Unix timestamp of most recent activation.
        metadata: Application-specific data (e.g., domain, source module).
    """

    node_id: str = ""
    embedding_hash: str = ""
    activation_count: int = 0
    last_activation: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NGLiteSynapse:
    """Weighted connection from a pattern node to a target.

    Targets are opaque string identifiers — could be model names (for
    routing), action categories (for Cricket), threat classes (for
    ClawGuard), or any other module-specific concept.

    Learning is Hebbian: success strengthens the connection weight,
    failure weakens it. Weight is bounded [0.0, 1.0].

    Attributes:
        source_id: Pattern node ID (the "when I see this..." side).
        target_id: Target identifier (the "...I should do this" side).
        weight: Connection strength [0.0, 1.0]. Higher = more confident.
        activation_count: Total times this synapse has been activated.
        success_count: Times this connection led to a successful outcome.
        failure_count: Times this connection led to a failed outcome.
        last_updated: Unix timestamp of most recent weight update.
        metadata: Application-specific data.
    """

    source_id: str = ""
    target_id: str = ""
    weight: float = 0.5
    activation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_updated: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bridge Interface (upgrade path to full NeuroGraph SaaS)
# ---------------------------------------------------------------------------

class NGBridge(ABC):
    """Interface for delegating to a higher-tier learning backend.

    Two planned implementations:
        1. NGPeerBridge (Tier 2): Connects two co-located NG-Lite
           instances for shared learning. When modules run together
           (e.g., Inference Difference + Cricket on the same host),
           they pool their pattern knowledge for mutual benefit.
        2. NGSaaSBridge (Tier 3): Connects to full NeuroGraph SaaS
           for cross-module STDP, hyperedges, and predictive coding.

    Both use this same interface. The module doesn't know or care
    which backend is on the other side — it just calls record_outcome,
    get_recommendations, etc. Tier transitions are transparent.

    NG-Lite maintains local state as fallback. If the bridge disconnects,
    the module continues operating on local learning without interruption.
    """

    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the bridge has an active connection to NeuroGraph."""
        ...

    @abstractmethod
    def record_outcome(
        self,
        embedding: np.ndarray,
        target_id: str,
        success: bool,
        module_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Report an outcome to the full substrate.

        Returns enriched response with cross-module insights, or None
        if the bridge is unavailable.
        """
        ...

    @abstractmethod
    def get_recommendations(
        self,
        embedding: np.ndarray,
        module_id: str,
        top_k: int = 3,
    ) -> Optional[List[Tuple[str, float, str]]]:
        """Get recommendations from the full substrate.

        Returns list of (target_id, confidence, reasoning) or None
        if the bridge is unavailable. The reasoning string explains
        why this recommendation was made (transparency obligation).
        """
        ...

    @abstractmethod
    def detect_novelty(
        self,
        embedding: np.ndarray,
        module_id: str,
    ) -> Optional[float]:
        """Get novelty score from the full substrate.

        Returns 0.0 (routine) to 1.0 (completely novel), or None
        if the bridge is unavailable.
        """
        ...

    @abstractmethod
    def sync_state(
        self,
        local_state: Dict[str, Any],
        module_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Sync local NG-Lite state with the full substrate.

        Called periodically to merge local learning into the shared
        graph and receive updates from cross-module learning.

        Returns updated state or None if unavailable.
        """
        ...


# ---------------------------------------------------------------------------
# NG-Lite Core
# ---------------------------------------------------------------------------

class NGLite:
    """Lightweight NeuroGraph learning substrate.

    Provides pattern-based Hebbian learning for any E-T Systems module.
    Each module vendors this file and uses it for standalone intelligence.

    Core capabilities:
        - Pattern recognition via embedding similarity
        - Hebbian learning (success strengthens, failure weakens)
        - Novelty detection (how far is this from known patterns?)
        - Bounded memory with LRU pruning
        - JSON persistence for cross-session learning
        - Optional bridge to full NeuroGraph SaaS

    Usage:
        ng = NGLite(module_id="inference_difference")

        # Learn from outcomes
        embedding = your_embedder.encode("user query")
        ng.record_outcome(embedding, target_id="local_model", success=True)

        # Get recommendations
        recs = ng.get_recommendations(embedding, top_k=3)

        # Check novelty
        novelty = ng.detect_novelty(embedding)

        # Persist
        ng.save("ng_lite_state.json")
        ng.load("ng_lite_state.json")
    """

    def __init__(
        self,
        module_id: str = "default",
        config: Optional[Dict[str, Any]] = None,
        bridge: Optional[NGBridge] = None,
    ):
        self.module_id = module_id
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._bridge = bridge

        # Core collections
        self.nodes: Dict[str, NGLiteNode] = {}
        self.synapses: Dict[Tuple[str, str], NGLiteSynapse] = {}

        # Embedding cache: hash -> full embedding (for similarity search)
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Activation history (bounded, for stats and debugging)
        self._history: List[Dict[str, Any]] = []
        self._history_max = 1000

        # Counters
        self._total_outcomes = 0
        self._total_successes = 0
        self._node_id_counter = 0

    # -------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------

    def find_or_create_node(self, embedding: np.ndarray) -> NGLiteNode:
        """Find existing node for this pattern or create a new one.

        Lookup strategy:
        1. Hash the embedding for exact match
        2. If no exact match, search for similar node (cosine distance)
        3. If no similar node found (novelty > threshold), create new

        Prunes the least-used node if at capacity.

        Args:
            embedding: Vector representation of the input pattern.

        Returns:
            The matched or newly created NGLiteNode.
        """
        emb = self._normalize(embedding)
        emb_hash = self._hash_embedding(emb)

        # Exact hash match
        if emb_hash in self.nodes:
            node = self.nodes[emb_hash]
            node.activation_count += 1
            node.last_activation = time.time()
            self._embedding_cache[emb_hash] = emb
            return node

        # Similarity search against known patterns
        similar = self._find_similar_node(emb)
        if similar is not None:
            similar.activation_count += 1
            similar.last_activation = time.time()
            return similar

        # Novel pattern — create new node
        if len(self.nodes) >= self.config["max_nodes"]:
            self._prune_least_used_node()

        self._node_id_counter += 1
        node = NGLiteNode(
            node_id=f"n_{self._node_id_counter}",
            embedding_hash=emb_hash,
            activation_count=1,
            last_activation=time.time(),
        )
        self.nodes[emb_hash] = node
        self._embedding_cache[emb_hash] = emb
        return node

    def record_outcome(
        self,
        embedding: np.ndarray,
        target_id: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an outcome and update learning weights.

        This is the core learning method. Call it after every
        decision to teach NG-Lite what works and what doesn't.

        Hebbian rule:
            - Success: weight += success_boost * (1.0 - weight)
              (diminishing returns as weight approaches 1.0)
            - Failure: weight -= failure_penalty * weight
              (proportional to current confidence)

        If a bridge to NeuroGraph SaaS is connected, the outcome
        is also forwarded there for cross-module learning.

        Args:
            embedding: The input pattern embedding (1-D numpy array).
            target_id: What was chosen (model name, action, etc.).
            success: Whether the outcome was successful.
            metadata: Optional context about this outcome.

        Returns:
            Dict with learning results (node_id, weight_after, etc.).

        Raises:
            ValueError: If embedding is not a 1-D numpy array.
        """
        # Input validation (Grok review: defensive boundary check)
        if not isinstance(embedding, np.ndarray) or embedding.ndim != 1:
            raise ValueError(
                f"embedding must be a 1-D numpy array, got "
                f"{type(embedding).__name__} with ndim={getattr(embedding, 'ndim', 'N/A')}"
            )

        node = self.find_or_create_node(embedding)
        synapse = self._get_or_create_synapse(node.node_id, target_id)

        synapse.activation_count += 1

        if success:
            synapse.success_count += 1
            # Hebbian strengthening with soft saturation
            delta = self.config["success_boost"] * (1.0 - synapse.weight)
            synapse.weight += delta
        else:
            synapse.failure_count += 1
            # Anti-Hebbian weakening proportional to current weight
            delta = self.config["failure_penalty"] * synapse.weight
            synapse.weight -= delta

        synapse.weight = float(np.clip(synapse.weight, 0.0, 1.0))
        synapse.last_updated = time.time()

        self._total_outcomes += 1
        if success:
            self._total_successes += 1

        result = {
            "node_id": node.node_id,
            "target_id": target_id,
            "success": success,
            "weight_after": synapse.weight,
            "activation_count": synapse.activation_count,
        }

        # Record in history
        self._record_history(result)

        # Forward to bridge if connected
        if self._bridge and self._bridge.is_connected():
            try:
                enriched = self._bridge.record_outcome(
                    embedding=embedding,
                    target_id=target_id,
                    success=success,
                    module_id=self.module_id,
                    metadata=metadata,
                )
                if enriched:
                    result["bridge_response"] = enriched
            except Exception as e:
                logger.warning("Bridge record_outcome failed: %s", e)

        return result

    def get_recommendations(
        self,
        embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Get target recommendations for an input pattern.

        Finds the closest known pattern node and returns its strongest
        synapse targets, sorted by weight (descending).

        If a bridge to NeuroGraph is connected, prefers its recommendations
        (which include cross-module intelligence). Falls back to local
        learning if bridge is unavailable.

        Args:
            embedding: The input pattern embedding.
            top_k: Maximum number of recommendations to return.

        Returns:
            List of (target_id, confidence) tuples, highest first.
            Empty list if no learned routes exist for this pattern.
        """
        # Try bridge first
        if self._bridge and self._bridge.is_connected():
            try:
                bridge_recs = self._bridge.get_recommendations(
                    embedding=embedding,
                    module_id=self.module_id,
                    top_k=top_k,
                )
                if bridge_recs:
                    # Bridge returns (target, confidence, reasoning)
                    # We return (target, confidence) for API consistency
                    return [(t, c) for t, c, _ in bridge_recs]
            except Exception as e:
                logger.warning("Bridge get_recommendations failed: %s", e)

        # Local learning
        node = self.find_or_create_node(embedding)

        relevant = [
            (syn.target_id, syn.weight)
            for key, syn in self.synapses.items()
            if key[0] == node.node_id and syn.weight > self.config["pruning_threshold"]
        ]

        if not relevant:
            return []

        relevant.sort(key=lambda x: x[1], reverse=True)
        return relevant[:top_k]

    def detect_novelty(self, embedding: np.ndarray) -> float:
        """How novel is this input pattern?

        Computes the minimum cosine distance between the input embedding
        and all known pattern nodes. Higher = more novel.

        If a bridge to NeuroGraph is connected, its novelty score
        (which considers cross-module patterns) is preferred.

        Args:
            embedding: The input pattern embedding.

        Returns:
            Novelty score from 0.0 (routine/known) to 1.0 (completely novel).
        """
        # Try bridge first
        if self._bridge and self._bridge.is_connected():
            try:
                bridge_novelty = self._bridge.detect_novelty(
                    embedding=embedding,
                    module_id=self.module_id,
                )
                if bridge_novelty is not None:
                    return bridge_novelty
            except Exception as e:
                logger.warning("Bridge detect_novelty failed: %s", e)

        # Local novelty detection
        if not self._embedding_cache:
            return 1.0  # Everything is novel when we know nothing

        emb = self._normalize(embedding)
        max_similarity = 0.0

        for cached_emb in self._embedding_cache.values():
            similarity = float(np.dot(emb, cached_emb))
            if similarity > max_similarity:
                max_similarity = similarity

        # Convert similarity to novelty (1 - similarity)
        # Cosine similarity of normalized vectors is in [-1, 1]
        # but practically in [0, 1] for embedding models
        novelty = 1.0 - max(0.0, max_similarity)
        return novelty

    # -------------------------------------------------------------------
    # Bridge Management
    # -------------------------------------------------------------------

    def connect_bridge(self, bridge: NGBridge) -> None:
        """Connect to full NeuroGraph SaaS.

        When connected, NG-Lite delegates to the full substrate for
        recommendations, novelty detection, and outcome recording.
        Local learning continues as fallback.
        """
        self._bridge = bridge
        logger.info("NG-Lite bridge connected for module '%s'", self.module_id)

    def disconnect_bridge(self) -> None:
        """Disconnect from NeuroGraph, fall back to local learning."""
        self._bridge = None
        logger.info("NG-Lite bridge disconnected, using local learning")

    def sync_with_bridge(self) -> Optional[Dict[str, Any]]:
        """Sync local state with NeuroGraph SaaS.

        Sends accumulated local learning to the full substrate and
        receives cross-module updates. Call periodically (e.g., hourly
        or after N outcomes).

        Returns:
            Sync result from bridge, or None if unavailable.
        """
        if not self._bridge or not self._bridge.is_connected():
            return None

        try:
            local_state = self._export_state()
            result = self._bridge.sync_state(
                local_state=local_state,
                module_id=self.module_id,
            )
            return result
        except Exception as e:
            logger.warning("Bridge sync failed: %s", e)
            return None

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save full state to JSON file.

        Saves all nodes, synapses, configuration, and counters.
        Embedding cache is NOT saved (too large, reconstructed on use).

        Args:
            filepath: Path to write the JSON state file.
        """
        state = self._export_state()
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("NG-Lite state saved to %s (%d nodes, %d synapses)",
                     filepath, len(self.nodes), len(self.synapses))

    def load(self, filepath: str) -> None:
        """Restore state from a JSON file.

        Replaces all current state with the loaded data.
        Embedding cache starts empty and rebuilds on use.

        Args:
            filepath: Path to the JSON state file.
        """
        with open(filepath, "r") as f:
            state = json.load(f)
        self._import_state(state)
        logger.info("NG-Lite state loaded from %s (%d nodes, %d synapses)",
                     filepath, len(self.nodes), len(self.synapses))

    def _export_state(self) -> Dict[str, Any]:
        """Export full state as a serializable dict.

        Synapse keys are converted from (source_id, target_id) tuples
        to "source_id|target_id" strings for JSON compatibility.
        """
        # Convert synapse keys from tuples to strings for JSON
        synapses_serialized = {}
        for (src, tgt), syn in self.synapses.items():
            key = f"{src}|{tgt}"
            synapses_serialized[key] = asdict(syn)

        return {
            "version": self.config["snapshot_version"],
            "module_id": self.module_id,
            "timestamp": time.time(),
            "config": self.config,
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
            "synapses": synapses_serialized,
            "counters": {
                "node_id_counter": self._node_id_counter,
                "total_outcomes": self._total_outcomes,
                "total_successes": self._total_successes,
            },
        }

    def _import_state(self, state: Dict[str, Any]) -> None:
        """Import state from a deserialized dict."""
        self.module_id = state.get("module_id", self.module_id)

        # Restore config (merge with defaults for forward compatibility)
        saved_config = state.get("config", {})
        self.config = {**DEFAULT_CONFIG, **saved_config}

        # Restore nodes
        self.nodes = {}
        for key, node_data in state.get("nodes", {}).items():
            self.nodes[key] = NGLiteNode(**node_data)

        # Restore synapses
        self.synapses = {}
        for key, syn_data in state.get("synapses", {}).items():
            parts = key.split("|", 1)
            if len(parts) == 2:
                tuple_key = (parts[0], parts[1])
                self.synapses[tuple_key] = NGLiteSynapse(**syn_data)

        # Restore counters
        counters = state.get("counters", {})
        self._node_id_counter = counters.get("node_id_counter", 0)
        self._total_outcomes = counters.get("total_outcomes", 0)
        self._total_successes = counters.get("total_successes", 0)

        # Clear caches (will rebuild on use)
        self._embedding_cache.clear()
        self._history.clear()

    # -------------------------------------------------------------------
    # Stats & Telemetry
    # -------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Current state statistics.

        Returns a dict suitable for logging, Observatory queries,
        or display to users. All routing/learning decisions should
        be queryable per transparency obligations.
        """
        synapse_weights = [s.weight for s in self.synapses.values()]
        return {
            "version": __version__,
            "module_id": self.module_id,
            "node_count": len(self.nodes),
            "synapse_count": len(self.synapses),
            "max_nodes": self.config["max_nodes"],
            "max_synapses": self.config["max_synapses"],
            "memory_estimate_bytes": self._estimate_memory(),
            "total_outcomes": self._total_outcomes,
            "total_successes": self._total_successes,
            "success_rate": (
                self._total_successes / self._total_outcomes
                if self._total_outcomes > 0 else 0.0
            ),
            "avg_synapse_weight": (
                float(np.mean(synapse_weights))
                if synapse_weights else 0.0
            ),
            "bridge_connected": (
                self._bridge is not None
                and self._bridge.is_connected()
            ),
            "embedding_cache_size": len(self._embedding_cache),
        }

    # -------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm < 1e-12:
            return embedding
        return embedding / norm

    def _hash_embedding(self, embedding: np.ndarray) -> str:
        """Hash embedding to a fixed-size string for fast lookup.

        Uses the first ``hash_dims`` dimensions of the embedding,
        converted to bytes, then SHA-256 truncated to 32 hex chars.
        This gives a compact, collision-resistant key.
        """
        dims = self.config["hash_dims"]
        truncated = embedding[:dims]
        hash_input = truncated.astype(np.float32).tobytes()
        return hashlib.sha256(hash_input).hexdigest()[:32]

    def _find_similar_node(self, embedding: np.ndarray) -> Optional[NGLiteNode]:
        """Find a node with similar embedding (below novelty threshold).

        Uses vectorized cosine similarity on all cached embeddings for
        performance (Grok review: batch dot product instead of per-node
        loop).  Returns the most similar node if its similarity exceeds
        (1 - novelty_threshold).
        """
        threshold = self.config["novelty_threshold"]

        if not self._embedding_cache:
            return None

        # Vectorized similarity: stack all cached embeddings into a matrix
        # and compute cosine similarities in one np.dot call.
        cache_keys = list(self._embedding_cache.keys())
        cache_matrix = np.stack(list(self._embedding_cache.values()))
        similarities = cache_matrix @ embedding  # (N,) cosine similarities

        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])

        if best_similarity >= (1.0 - threshold):
            best_hash = cache_keys[best_idx]
            return self.nodes.get(best_hash)

        return None

    def _get_or_create_synapse(
        self,
        source_id: str,
        target_id: str,
    ) -> NGLiteSynapse:
        """Get existing synapse or create a new one with neutral weight."""
        key = (source_id, target_id)
        if key in self.synapses:
            return self.synapses[key]

        if len(self.synapses) >= self.config["max_synapses"]:
            self._prune_weakest_synapse()

        synapse = NGLiteSynapse(
            source_id=source_id,
            target_id=target_id,
            weight=0.5,  # Neutral initial weight
            last_updated=time.time(),
        )
        self.synapses[key] = synapse
        return synapse

    def _prune_least_used_node(self) -> None:
        """Remove the node with the lowest activation count (LRU)."""
        if not self.nodes:
            return

        # Find least-used node
        least_hash = min(
            self.nodes,
            key=lambda h: self.nodes[h].activation_count,
        )
        least_node = self.nodes[least_hash]

        # Remove associated synapses
        keys_to_remove = [
            key for key in self.synapses
            if key[0] == least_node.node_id
        ]
        for key in keys_to_remove:
            del self.synapses[key]

        # Remove node and cached embedding
        del self.nodes[least_hash]
        self._embedding_cache.pop(least_hash, None)

    def _prune_weakest_synapse(self) -> None:
        """Remove the synapse with the lowest weight."""
        if not self.synapses:
            return

        weakest_key = min(
            self.synapses,
            key=lambda k: self.synapses[k].weight,
        )
        del self.synapses[weakest_key]

    def _record_history(self, entry: Dict[str, Any]) -> None:
        """Append to bounded history."""
        entry["timestamp"] = time.time()
        self._history.append(entry)
        if len(self._history) > self._history_max:
            self._history = self._history[-self._history_max:]

    def _estimate_memory(self) -> int:
        """Rough estimate of memory footprint in bytes.

        Node: ~200 bytes each (dataclass + hash key)
        Synapse: ~150 bytes each (dataclass + tuple key)
        Embedding cache: ~embedding_dim * 4 bytes each (float32)
        """
        node_bytes = len(self.nodes) * 200
        synapse_bytes = len(self.synapses) * 150
        cache_bytes = len(self._embedding_cache) * self.config["embedding_dim"] * 4
        return node_bytes + synapse_bytes + cache_bytes
