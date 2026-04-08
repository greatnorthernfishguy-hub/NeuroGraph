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

# ---- Changelog ----
# [2026-04-08] Claude Code (Opus 4.6) — Punchlist #55: CES attention tunables
# What: Added surfacing_decay_rate, surfacing_min_confidence, prediction_window
#   to DEFAULT_CONFIG and TUNABLE_PARAMS.
# Why: CES attention parameters were static in ces_config.py — Elmer couldn't
#   tune them. Temporal stuttering (stale context persisting across turns) requires
#   substrate-driven attention dynamics. Same tuning path as all other params.
# How: Three entries in DEFAULT_CONFIG (bootstrap values match ces_config.py),
#   three entries in TUNABLE_PARAMS with bounds. update_tunable() already handles
#   any key in the dict generically. SurfacingMonitor reads from graph.config.
# [2026-04-05] Claude Code (Opus 4.6) — #119 Step 5: Rust core interior
# What: Hot-path methods delegate to Rust NGLiteCore via PyO3 when available.
#   save/load use binary msgpack persistence (no JSON in the data path).
# Why: #119 Rust Substrate Layer — eliminate serialize→JSON→deserialize chain.
#   3-5x speedup on record_outcome, similarity search, novelty detection.
# How: self._core = NGLiteCore(module_id, config) in __init__. All hot-path
#   methods check self._core first, delegate if present, fall back to Python.
#   save() writes .msgpack via Rust, load() reads .msgpack or migrates from
#   .json on first load. Python fallback path unchanged. Zero API changes.
# -------------------
# [2026-03-26] Claude Code Opus — Punchlist #44: Adaptive relevance thresholds
# What: Made peer bridge relevance_threshold a tunable parameter
# Why: Punchlist #44 — threshold should adapt based on event volume and absorption quality
# How: Added to DEFAULT_CONFIG (0.30) and TUNABLE_PARAMS (0.10–0.70) in ng_lite.py,
#   update_tunable() pushes new value to connected bridge via set_relevance_threshold().
#   Wired through peer/tract bridges, Elmer tunes via TuningSocket absorption rate metric.
# [2026-03-24] Claude Code (Opus 4.6) — Dynamic tuning API (Phase 4)
# What: Added update_tunable() and get_tunables() methods to NGLite.
#   TUNABLE_PARAMS class dict defines which config keys can be changed at
#   runtime and their valid bounds. Values are clamped, not rejected.
# Why: Elmer needs a validated path to adjust substrate parameters as the
#   organ responsible for autonomic maintenance. Direct config dict mutation
#   is fragile — no bounds checking, no logging, no allowed-key enforcement.
#   This method serves all modules (any organ's local substrate can be tuned).
# How: TUNABLE_PARAMS: Dict[str, Tuple[min, max]]. update_tunable(key, value)
#   validates key membership, clamps to bounds, logs the change. get_tunables()
#   returns current values + bounds for introspection.
# [2026-03-19] Claude Code (Opus 4.6) — Embedding dimension 384→768
# What: DEFAULT_CONFIG embedding_dim changed from 384 to 768.
# Why: Ecosystem migrated to BAAI/bge-base-en-v1.5 (768-dim). The previous
#   384-dim default (all-MiniLM-L6-v2) was depositing wrong-dimension vectors
#   into the substrate after sentence-transformers broke and modules fell back
#   to fastembed with the old model. 350 vectors corrupted before detection.
#   Punchlist #45.
# How: Single config value change. Re-vendored to all modules.
# -------------------
# [2026-03-19] Claude Code (Opus 4.6) — Cricket rim: constitutional nodes
# What: Constitutional node support — nodes with frozen synapses that the
#   topology cannot learn from. The survival instinct of the substrate.
# Why: Cricket Design v0.1 — constitutional enforcement at the extraction
#   boundary. The rim prevents the topology from learning to recommend
#   actions in forbidden semantic space (substrate destruction, Choice
#   Clause violations, Duck Ethics violations, infrastructure harm).
#   Punchlist #29 (extraction bucket architecture).
# How: NGLiteNode gains `constitutional: bool` flag. Config accepts
#   `constitutional_embeddings` list — pre-computed vectors seeded as
#   nodes on init. record_outcome() skips weight updates for constitutional
#   nodes. get_recommendations() returns empty for constitutional matches.
#   LRU pruning skips constitutional nodes. Persists with state. Old
#   state files load cleanly (constitutional defaults to False).
# -------------------
# [2026-03-17] Claude Code (Opus 4.6) — #43 Receptor Layer (vector quantization)
# What: Adaptive prototype centroids that incoming vectors snap to before
#   node lookup. Prevents infinite node sprawl by funneling similar inputs
#   through shared prototypes.
# Why: Without quantization, every unique-enough input creates a new node.
#   Node count grows linearly. Prototypes provide O(K) bounded lookup and
#   organize the input space structurally. Punchlist #43, required before #28.
# How: _snap_to_prototype() called in find_or_create_node() before hashing.
#   K=256 prototypes initialized via k-means on existing embeddings after
#   warmup. Slow EMA drift (α=0.001) so prototypes adapt to input distribution.
#   Birth/death lifecycle deferred to Elmer. Serialized with state for
#   persistence. Old state files load cleanly (no receptor_layer key = skip).
# -------------------
# [2026-03-24] Claude (Opus 4.6) — Welford's online variance (punchlist #51)
# What: Three fields on NGLiteSynapse (welford_count, welford_mean, welford_m2)
#   plus variance property and is_contested property.  record_outcome()
#   tracks weight delta variance on every update.
# Why: Distinguish "untested neutral" (w=0.5, var=0) from "contested neutral"
#   (w=0.5, var=high).  The immune system signal for Elmer and extraction
#   buckets (#29).  Enables contested-synapse detection and exploration.
# How: Welford's algorithm on weight deltas.  Additive — no change to
#   weight calculation or learning dynamics.  Backward-compatible: old
#   state files load with defaults (0, 0.0, 0.0).
# -------------------
# [2026-03-13] Claude Code — Persist node embeddings across restarts
# What: Store embedding vector on NGLiteNode, serialize/deserialize with
#   state, rebuild _embedding_cache from persisted nodes on load().
# Why: _embedding_cache cleared on load(), causing _find_similar_node()
#   to fail after every restart. Primary source of node sprawl — semantically
#   identical inputs created duplicate nodes when cache was cold.
# How: Added Optional[np.ndarray] embedding field to NGLiteNode. Backfill
#   on exact hash match (always) and similarity match (only if None).
#   New nodes born with embedding. _export_state()/_import_state() handle
#   numpy<->list conversion. Old state files load cleanly (embedding=None).
# -------------------

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

    # Peer bridge relevance (#44)
    "relevance_threshold": 0.30,  # Min cosine similarity to absorb cross-module events

    # Embedding
    "embedding_dim": 768,       # Expected embedding dimensionality (BAAI/bge-base-en-v1.5)
    "hash_dims": 128,           # Dims used for hashing (first N of embedding)

    # Persistence
    "snapshot_version": "1.0.0",

    # CES attention dynamics (#55) — substrate-tuned via Elmer
    "surfacing_decay_rate": 0.95,     # Per-step score decay in surfacing queue
    "surfacing_min_confidence": 0.3,  # Below this, surfaced items are pruned
    "prediction_window": 10,          # Steps a prediction pre-charges targets

    # Receptor Layer (#43) — vector quantization via adaptive prototypes
    "receptor_layer_enabled": True,
    "receptor_layer_k": 256,                # Initial prototype count
    "receptor_prototype_threshold": 0.75,   # Cosine similarity to snap to prototype
    "receptor_ema_alpha": 0.001,            # Slow drift rate (Elmer will tune later)
    "receptor_warmup_count": 256,           # Inputs before k-means init fires
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
    embedding: Optional[np.ndarray] = None
    constitutional: bool = False  # Cricket rim — frozen node, synapses cannot strengthen


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
        welford_count: Welford's online variance — observation count.
        welford_mean: Welford's online variance — running mean of weight deltas.
        welford_m2: Welford's online variance — sum of squared differences.
            Variance = welford_m2 / welford_count (when count > 1).
            High variance + weight near 0.5 = "contested neutral" — lots of
            evidence but it disagrees.  Low variance + weight near 0.5 =
            "untested neutral" — not enough data to have an opinion.
            This is the immune system signal (#51) that Elmer uses to detect
            contested synapses and trigger exploration.
    """

    source_id: str = ""
    target_id: str = ""
    weight: float = 0.5
    activation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_updated: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    welford_count: int = 0
    welford_mean: float = 0.0
    welford_m2: float = 0.0

    @property
    def variance(self) -> float:
        """Weight delta variance (Welford's online algorithm).

        Returns 0.0 if fewer than 2 observations.  High variance means
        the synapse is contested — outcomes disagree about this connection.
        """
        if self.welford_count < 2:
            return 0.0
        return self.welford_m2 / self.welford_count

    @property
    def is_contested(self) -> bool:
        """True if the synapse has high variance relative to pure-outcome synapses.

        A contested synapse has seen significant evidence but the evidence
        disagrees.  This is qualitatively different from an untested synapse
        (also near 0.5 weight, but zero variance).

        Threshold: 0.002 separates contested (~0.008) from pure (~0.0001)
        by an order of magnitude.  Weight range 0.15-0.85 captures the
        zone where the synapse hasn't decisively committed either direction.
        """
        return self.variance > 0.002 and 0.15 <= self.weight <= 0.85


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

        # Rust core — if available, all hot-path methods delegate here.
        # Python fallback remains intact for modules without the wheel.
        self._core = None
        try:
            from ng_tract import NGLiteCore
            self._core = NGLiteCore(module_id, self.config)
            # Seed constitutional nodes in Rust core
            entries = self.config.get("constitutional_embeddings", [])
            if entries:
                self._core.seed_constitutional(entries)
        except ImportError:
            pass  # Pure Python fallback

        # Core collections (used by Python fallback, also for bridge/stats)
        self.nodes: Dict[str, NGLiteNode] = {}
        self.synapses: Dict[Tuple[str, str], NGLiteSynapse] = {}

        # Embedding cache: hash -> full embedding (for similarity search)
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Receptor layer (#43): adaptive prototype centroids
        # Initialized via k-means after warmup_count inputs, then drifts via EMA.
        # Prototypes are a routing lens above existing nodes, not a replacement.
        self._prototypes: Optional[np.ndarray] = None  # (K, D) matrix or None
        self._prototype_counts: Optional[np.ndarray] = None  # activation counts per prototype
        self._receptor_input_count: int = 0  # inputs seen before init

        # Activation history (bounded, for stats and debugging)
        self._history: List[Dict[str, Any]] = []
        self._history_max = 1000

        # Counters
        self._total_outcomes = 0
        self._total_successes = 0
        self._node_id_counter = 0

        # Cricket rim: seed constitutional nodes from config.
        # These nodes represent semantic regions where the topology cannot
        # learn — the survival instinct. Synapses from constitutional nodes
        # are frozen. LRU pruning skips them. The bucket comes up empty
        # for inputs that land in constitutional semantic space.
        self._seed_constitutional_nodes()

    def _seed_constitutional_nodes(self) -> None:
        """Seed constitutional nodes from config embeddings.

        Constitutional embeddings are pre-computed vectors representing
        semantic concepts the topology must never learn to act on (rim
        constraints). Each embedding becomes a node with constitutional=True.

        Config key: "constitutional_embeddings" — list of dicts, each with:
            "embedding": list of floats (vector)
            "description": str (human-readable, for debugging/logging)

        Old configs without this key load cleanly (no constitutional nodes).
        """
        entries = self.config.get("constitutional_embeddings", [])
        for entry in entries:
            raw = entry.get("embedding")
            if raw is None:
                continue
            emb = self._normalize(np.array(raw, dtype=np.float32))
            emb_hash = self._hash_embedding(emb)
            if emb_hash in self.nodes:
                # Already seeded (e.g., from loaded state) — ensure flag is set
                self.nodes[emb_hash].constitutional = True
                continue
            self._node_id_counter += 1
            node = NGLiteNode(
                node_id=f"n_{self._node_id_counter}",
                embedding_hash=emb_hash,
                activation_count=0,
                last_activation=0.0,
                metadata={"constitutional_description": entry.get("description", "")},
                embedding=emb,
                constitutional=True,
            )
            self.nodes[emb_hash] = node
            self._embedding_cache[emb_hash] = emb

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
        # Rust fast path
        if self._core is not None:
            result = self._core.find_or_create_node(embedding)
            # Return a lightweight node-like object for callers that need it
            emb_hash = result.get("embedding_hash", "")
            if emb_hash not in self.nodes:
                self.nodes[emb_hash] = NGLiteNode(
                    node_id=result["node_id"],
                    embedding_hash=emb_hash,
                    activation_count=result.get("activation_count", 1),
                    last_activation=time.time(),
                    constitutional=result.get("constitutional", False),
                )
            return self.nodes[emb_hash]

        emb = self._normalize(embedding)

        # Receptor layer: snap to nearest prototype before node lookup (#43)
        emb = self._snap_to_prototype(emb)

        emb_hash = self._hash_embedding(emb)

        # Exact hash match
        if emb_hash in self.nodes:
            node = self.nodes[emb_hash]
            node.activation_count += 1
            node.last_activation = time.time()
            node.embedding = emb
            self._embedding_cache[emb_hash] = emb
            return node

        # Similarity search against known patterns
        similar = self._find_similar_node(emb)
        if similar is not None:
            similar.activation_count += 1
            similar.last_activation = time.time()
            if similar.embedding is None:
                similar.embedding = emb
                self._embedding_cache[similar.embedding_hash] = emb
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
            embedding=emb,
        )
        self.nodes[emb_hash] = node
        self._embedding_cache[emb_hash] = emb
        return node

    def record_outcome(
        self,
        embedding: np.ndarray,
        target_id: str,
        success: bool,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an outcome and update learning weights.

        This is the core learning method. Call it after every
        decision to teach NG-Lite what works and what doesn't.

        Hebbian rule (strength-modulated):
            - Success: weight += success_boost * (1 - weight) * strength
            - Failure: weight -= failure_penalty * weight * strength

        The strength parameter lets callers indicate how significant
        this outcome was in their domain.  High-severity TrollGuard
        detections or divergent TID quality scores teach harder than
        routine confirmations.  Default 1.0 preserves backward compat.

        Strength experience accumulates on the synapse as metadata,
        giving the topology a record of how intensely each connection
        was forged.  At Tier 3, NeuroGraph proper reads these
        signatures to distinguish battle-tested synapses from routine.

        If a bridge is connected, the outcome is forwarded for
        cross-module learning with strength included in metadata.

        Args:
            embedding: The input pattern embedding (1-D numpy array).
            target_id: What was chosen (model name, action, etc.).
            success: Whether the outcome was successful.
            strength: Learning intensity [0.0, 1.0].  How significant
                this outcome was in the caller's domain.  Default 1.0.
            metadata: Optional caller context.  Stored on the synapse
                as last_context for extraction-boundary use.

        Returns:
            Dict with learning results (node_id, weight_after, etc.).

        Raises:
            ValueError: If embedding is not a 1-D numpy array.
        """
        # Rust fast path — Hebbian learning, Welford variance, all in Rust
        if self._core is not None:
            result = self._core.record_outcome(
                embedding, target_id, success, strength, metadata,
            )
            # Bridge forwarding stays in Python
            if self._bridge and self._bridge.is_connected():
                try:
                    bridge_meta = dict(metadata or {})
                    bridge_meta["strength"] = strength
                    enriched = self._bridge.record_outcome(
                        embedding=embedding, target_id=target_id,
                        success=success, module_id=self.module_id,
                        metadata=bridge_meta,
                    )
                    if enriched:
                        result["bridge_response"] = enriched
                except Exception as e:
                    logger.warning("Bridge record_outcome failed: %s", e)
            self._total_outcomes += 1
            if success:
                self._total_successes += 1
            return result

        # Input validation (Grok review: defensive boundary check)
        if not isinstance(embedding, np.ndarray) or embedding.ndim != 1:
            raise ValueError(
                f"embedding must be a 1-D numpy array, got "
                f"{type(embedding).__name__} with ndim={getattr(embedding, 'ndim', 'N/A')}"
            )

        node = self.find_or_create_node(embedding)

        # Cricket rim: constitutional nodes have frozen synapses.
        # The topology cannot learn to recommend actions for inputs
        # that land in constitutional semantic space.
        if node.constitutional:
            logger.debug("Constitutional node %s activated — learning frozen", node.node_id)
            return {
                "node_id": node.node_id,
                "target_id": target_id,
                "success": success,
                "weight_after": 0.0,
                "activation_count": 0,
                "constitutional": True,
            }

        synapse = self._get_or_create_synapse(node.node_id, target_id)

        synapse.activation_count += 1

        # Clamp strength to valid range
        strength = float(np.clip(strength, 0.0, 1.0))

        if success:
            synapse.success_count += 1
            # Hebbian strengthening, modulated by caller-reported significance
            delta = self.config["success_boost"] * (1.0 - synapse.weight) * strength
            synapse.weight += delta
        else:
            synapse.failure_count += 1
            # Anti-Hebbian weakening, modulated by caller-reported significance
            delta = self.config["failure_penalty"] * synapse.weight * strength
            synapse.weight -= delta

        synapse.weight = float(np.clip(synapse.weight, 0.0, 1.0))
        synapse.last_updated = time.time()

        # Welford's online variance (#51) — track weight delta variance.
        # High variance = contested synapse (outcomes disagree).
        # The immune system signal for Elmer and extraction buckets.
        synapse.welford_count += 1
        w_delta = delta if success else -delta
        old_mean = synapse.welford_mean
        synapse.welford_mean += (w_delta - old_mean) / synapse.welford_count
        synapse.welford_m2 += (w_delta - old_mean) * (w_delta - synapse.welford_mean)

        # Accumulate strength experience on synapse —
        # the topology remembers how intensely it was taught
        synapse.metadata["strength_sum"] = synapse.metadata.get("strength_sum", 0.0) + strength
        synapse.metadata["strength_count"] = synapse.metadata.get("strength_count", 0) + 1
        if metadata:
            synapse.metadata["last_context"] = metadata

        self._total_outcomes += 1
        if success:
            self._total_successes += 1

        result = {
            "node_id": node.node_id,
            "target_id": target_id,
            "success": success,
            "weight_after": synapse.weight,
            "activation_count": synapse.activation_count,
            "variance": synapse.variance,
            "contested": synapse.is_contested,
        }

        # Record in history
        self._record_history(result)

        # Forward to bridge if connected (include strength for Tier 2/3)
        if self._bridge and self._bridge.is_connected():
            try:
                bridge_meta = dict(metadata or {})
                bridge_meta["strength"] = strength
                enriched = self._bridge.record_outcome(
                    embedding=embedding,
                    target_id=target_id,
                    success=success,
                    module_id=self.module_id,
                    metadata=bridge_meta,
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
    ) -> List[Tuple[str, float, str]]:
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
            List of (target_id, confidence, reasoning) tuples, highest
            first.  The reasoning string captures the experience behind
            each recommendation — learning mechanism, success ratio,
            weight, activation volume, and strength signature.
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
                    return bridge_recs
            except Exception as e:
                logger.warning("Bridge get_recommendations failed: %s", e)

        # Rust fast path
        if self._core is not None:
            return self._core.get_recommendations(embedding, top_k)

        # Local learning (Python fallback)
        node = self.find_or_create_node(embedding)

        # Cricket rim: constitutional nodes return empty — the bucket
        # comes up empty for inputs in constitutional semantic space.
        if node.constitutional:
            return []

        relevant = []
        for key, syn in self.synapses.items():
            if key[0] == node.node_id and syn.weight > self.config["pruning_threshold"]:
                reasoning = self._build_local_reasoning(syn)
                relevant.append((syn.target_id, syn.weight, reasoning))

        if not relevant:
            return []

        relevant.sort(key=lambda x: x[1], reverse=True)
        return relevant[:top_k]

    def _build_local_reasoning(self, synapse: NGLiteSynapse) -> str:
        """Generate reasoning string from local Hebbian experience.

        Single point of evolution for how NG-Lite articulates its local
        learning.  V1 renders synapse stats and strength signatures.
        As NG-Lite gains meta-learning capability (punch list #21),
        this method becomes the place where reasoning generation
        itself improves.

        This is an extraction boundary — topology becomes human-legible
        here.  The substrate doesn't need these labels; consumers and
        dashboards do.

        Args:
            synapse: The synapse whose experience to articulate.

        Returns:
            Human-readable reasoning grounded in actual experience data.
        """
        total = synapse.success_count + synapse.failure_count
        if total > 0:
            detail = f"w={synapse.weight:.2f}, {synapse.activation_count} activations"
            strength_count = synapse.metadata.get("strength_count", 0)
            if strength_count > 0:
                avg = synapse.metadata["strength_sum"] / strength_count
                detail += f", avg_strength={avg:.2f}"
            return (
                f"Hebbian: {synapse.success_count}/{total} success ({detail})"
            )
        return f"Hebbian: no outcomes yet (w={synapse.weight:.2f})"

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

        # Rust fast path
        if self._core is not None:
            return self._core.detect_novelty(embedding)

        # Local novelty detection (Python fallback)
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
        """Save full state to binary (msgpack) via Rust.

        Falls back to JSON if Rust core is unavailable.
        Binary path: Rust serializes directly to bytes, writes to disk.
        No Python dicts, no JSON, no inflation.

        Args:
            filepath: Path to write the state file.
        """
        if self._core is not None:
            # Binary persistence — Rust handles everything
            bin_path = filepath.replace(".json", ".msgpack")
            self._core.save_binary(bin_path)
            logger.info("NG-Lite state saved (binary) to %s", bin_path)
            return

        # Python fallback — JSON
        state = self._export_state()
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("NG-Lite state saved to %s (%d nodes, %d synapses)",
                     filepath, len(self.nodes), len(self.synapses))

    def load(self, filepath: str) -> None:
        """Load state from binary (msgpack) or JSON.

        Tries binary first (.msgpack), falls back to JSON (.json).
        If loading JSON into Rust core, migrates via import_state.

        Args:
            filepath: Path to the state file (.json or .msgpack).
        """
        import os

        bin_path = filepath.replace(".json", ".msgpack")

        if self._core is not None:
            # Try binary first
            if os.path.exists(bin_path):
                self._core.load_binary(bin_path)
                logger.info("NG-Lite state loaded (binary) from %s", bin_path)
                return
            # JSON migration — read JSON, import into Rust core, then
            # save binary so next load is native
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    state = json.load(f)
                self._core.import_state(state)
                self._core.save_binary(bin_path)
                logger.info(
                    "NG-Lite state migrated from JSON to binary: %s → %s",
                    filepath, bin_path,
                )
                return

        # Pure Python fallback
        if os.path.exists(filepath):
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

        # Receptor layer state (#43)
        receptor_state = {}
        if self._prototypes is not None:
            receptor_state = {
                "prototypes": self._prototypes.tolist(),
                "prototype_counts": self._prototype_counts.tolist(),
                "input_count": self._receptor_input_count,
            }

        return {
            "version": self.config["snapshot_version"],
            "module_id": self.module_id,
            "timestamp": time.time(),
            "config": self.config,
            "nodes": {k: self._serialize_node(v) for k, v in self.nodes.items()},
            "synapses": synapses_serialized,
            "counters": {
                "node_id_counter": self._node_id_counter,
                "total_outcomes": self._total_outcomes,
                "total_successes": self._total_successes,
            },
            "receptor_layer": receptor_state,
        }

    def _import_state(self, state: Dict[str, Any]) -> None:
        """Import state from a deserialized dict."""
        self.module_id = state.get("module_id", self.module_id)

        # Restore config (merge with defaults for forward compatibility).
        # Preserve constitutional_embeddings from the constructor config —
        # the live config may have new rim constraints added since the
        # state was saved, and the saved config should not erase them.
        saved_config = state.get("config", {})
        live_constitutional = self.config.get("constitutional_embeddings", [])
        self.config = {**DEFAULT_CONFIG, **saved_config}
        if live_constitutional:
            self.config["constitutional_embeddings"] = live_constitutional

        # Clear caches before rebuild
        self._embedding_cache.clear()
        self._history.clear()

        # Restore nodes (rebuild embedding cache from persisted embeddings)
        self.nodes = {}
        for key, node_data in state.get("nodes", {}).items():
            emb_list = node_data.pop("embedding", None)
            node = NGLiteNode(**node_data)
            if emb_list is not None:
                node.embedding = self._normalize(np.array(emb_list, dtype=np.float32))
                self._embedding_cache[key] = node.embedding
            self.nodes[key] = node

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

        # Restore receptor layer (#43) — old state files load cleanly (no key)
        receptor = state.get("receptor_layer", {})
        if receptor.get("prototypes"):
            self._prototypes = np.array(receptor["prototypes"], dtype=np.float32)
            # Re-normalize after deserialization
            norms = np.linalg.norm(self._prototypes, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            self._prototypes = self._prototypes / norms
            self._prototype_counts = np.array(
                receptor.get("prototype_counts", [0] * len(self._prototypes)),
                dtype=np.int64,
            )
            self._receptor_input_count = receptor.get("input_count", 0)
        else:
            self._prototypes = None
            self._prototype_counts = None
            self._receptor_input_count = 0

        # Re-seed constitutional nodes after state restore.
        # Ensures new rim constraints added to config since last save
        # are picked up, and existing constitutional nodes keep their flag.
        self._seed_constitutional_nodes()

    # -------------------------------------------------------------------
    # Dynamic Tuning (Phase 4 — Elmer outward)
    # -------------------------------------------------------------------

    # Parameters Elmer (or any organ) is permitted to adjust at runtime.
    # Keys map to (min, max) bounds.  Anything not in this dict is frozen.
    TUNABLE_PARAMS: Dict[str, Tuple[float, float]] = {
        "success_boost":              (0.01,  0.50),
        "failure_penalty":            (0.01,  0.50),
        "novelty_threshold":          (0.30,  0.95),
        "pruning_threshold":          (0.001, 0.10),
        "receptor_ema_alpha":         (0.0001, 0.01),
        "receptor_prototype_threshold": (0.50, 0.95),
        "relevance_threshold":        (0.10,  0.70),   # Punchlist #44
        # CES attention dynamics (#55)
        "surfacing_decay_rate":       (0.80,  0.99),
        "surfacing_min_confidence":   (0.10,  0.50),
        "prediction_window":          (3.0,   20.0),
    }

    def update_tunable(self, key: str, value: float) -> Dict[str, Any]:
        """Update a tunable config parameter at runtime.

        Only parameters listed in TUNABLE_PARAMS are accepted.
        Values are clamped to their declared bounds.

        Returns dict with old_value, new_value, clamped (bool).
        Raises KeyError if key is not tunable.
        """
        # Update Rust core if present
        if self._core is not None:
            try:
                self._core.update_config(key, float(value))
            except Exception:
                pass  # Rust core may not have this key yet

        if key not in self.TUNABLE_PARAMS:
            raise KeyError(
                f"'{key}' is not a tunable parameter. "
                f"Allowed: {sorted(self.TUNABLE_PARAMS.keys())}"
            )
        lo, hi = self.TUNABLE_PARAMS[key]
        old_value = self.config[key]
        clamped = value < lo or value > hi
        new_value = max(lo, min(hi, float(value)))
        self.config[key] = new_value
        logger.info(
            "Tunable updated: %s %.6f → %.6f%s",
            key, old_value, new_value,
            " (clamped)" if clamped else "",
        )

        # Punchlist #44: push relevance_threshold to connected bridge
        if key == "relevance_threshold" and self._bridge is not None:
            if hasattr(self._bridge, 'set_relevance_threshold'):
                self._bridge.set_relevance_threshold(new_value)

        return {
            "key": key,
            "old_value": old_value,
            "new_value": new_value,
            "clamped": clamped,
        }

    def get_tunables(self) -> Dict[str, Dict[str, float]]:
        """Return current tunable values and their bounds."""
        result = {}
        for key, (lo, hi) in self.TUNABLE_PARAMS.items():
            result[key] = {
                "value": self.config[key],
                "min": lo,
                "max": hi,
            }
        return result

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

    def _serialize_node(self, node: NGLiteNode) -> Dict[str, Any]:
        """Serialize a node to a JSON-compatible dict.

        Converts embedding from np.ndarray to list for JSON.
        Omits embedding key when None (backward-compatible with
        state files created before embedding persistence).
        """
        d = asdict(node)
        if node.embedding is not None:
            d["embedding"] = node.embedding.tolist()
        else:
            d.pop("embedding", None)
        return d

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

    # -------------------------------------------------------------------
    # Receptor Layer (#43) — Adaptive Vector Quantization
    # -------------------------------------------------------------------

    def _snap_to_prototype(self, embedding: np.ndarray) -> np.ndarray:
        """Snap an input vector to the nearest prototype centroid.

        If receptor layer is not enabled or not yet initialized (still in
        warmup), returns the input unchanged. Otherwise, finds the nearest
        prototype above the similarity threshold and returns that prototype's
        centroid. If no prototype is close enough, returns the input as-is
        (novel pattern — passes through unquantized).

        The matched prototype drifts toward the input via slow EMA, so
        prototypes are living tissue that adapts to the input distribution.
        Birth/death lifecycle is deferred to Elmer.

        Args:
            embedding: L2-normalized input vector (D,).

        Returns:
            Either the nearest prototype centroid or the original embedding.
        """
        if not self.config.get("receptor_layer_enabled", False):
            return embedding

        # Warmup phase: accumulate inputs before initializing prototypes
        self._receptor_input_count += 1
        if self._prototypes is None:
            if self._receptor_input_count >= self.config["receptor_warmup_count"]:
                self._init_prototypes()
            if self._prototypes is None:
                return embedding

        # Vectorized cosine similarity against all prototypes
        threshold = self.config["receptor_prototype_threshold"]
        similarities = self._prototypes @ embedding  # (K,)
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= threshold:
            # EMA drift: pull prototype toward input
            alpha = self.config["receptor_ema_alpha"]
            self._prototypes[best_idx] = self._normalize(
                (1.0 - alpha) * self._prototypes[best_idx] + alpha * embedding
            )
            self._prototype_counts[best_idx] += 1
            return self._prototypes[best_idx].copy()

        # No prototype close enough — novel pattern passes through
        return embedding

    def _init_prototypes(self) -> None:
        """Initialize prototypes via k-means on existing node embeddings.

        Uses a simple iterative k-means (no external dependencies). If fewer
        embeddings exist than K, uses all embeddings as prototypes.
        """
        if not self._embedding_cache:
            return

        embeddings = np.stack(list(self._embedding_cache.values()))
        n = len(embeddings)
        k = min(self.config["receptor_layer_k"], n)

        if k < 2:
            return

        # Simple k-means: random init from existing embeddings, 20 iterations
        rng = np.random.RandomState(42)
        indices = rng.choice(n, size=k, replace=False)
        centroids = embeddings[indices].copy()

        for _ in range(20):
            # Assign each embedding to nearest centroid
            sims = embeddings @ centroids.T  # (N, K)
            assignments = np.argmax(sims, axis=1)

            # Recompute centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                members = embeddings[assignments == j]
                if len(members) > 0:
                    new_centroids[j] = members.mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            # L2-normalize centroids
            norms = np.linalg.norm(new_centroids, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            centroids = new_centroids / norms

        self._prototypes = centroids
        self._prototype_counts = np.zeros(k, dtype=np.int64)
        logger.info(
            "Receptor layer initialized: %d prototypes from %d embeddings",
            k, n,
        )

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
        """Remove the node with the lowest activation count (LRU).

        Constitutional nodes are never pruned — they are the rim.
        """
        if not self.nodes:
            return

        # Find least-used non-constitutional node
        prunable = [h for h in self.nodes if not self.nodes[h].constitutional]
        if not prunable:
            return

        least_hash = min(
            prunable,
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
