# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Concrete LeniaSubstrate implementation wrapping NeuroGraph's Graph
# Why: Maps Lenia's abstract entity/distance interface to NG's nodes/synapses
# How: Wraps graph.nodes, graph.synapses, computes 5-metric distance vectors
# PRD: ~/docs/prd/Lenia_FlowGraph_Design_v0.1.md §4
# [2026-03-26] Claude Code (Opus 4.6) — Dual-pass embeddings + vector DB access
# What: Distance vector now 6 components (forest + tree embedding similarity).
#   Reads embeddings from SimpleVectorDB, distinguishes forest/tree nodes via
#   metadata._tree_concept flag. Cached embedding matrix for fast cosine sim.
# Why: Dual-pass gives kernel two semantic scales. Forest = broad concept,
#   tree = specific detail. Channels can weight each differently.
# How: vector_db reference passed at init. Embedding matrix built on populate().
#   Forest nodes get forest similarity in component 4, tree nodes get tree
#   similarity in component 5. Mixed pairs get the relevant component.
# -------------------

"""NeuroGraph-specific implementation of the LeniaSubstrate interface.

entity = graph node
distance = [topology_hops, synaptic_weight, cofire_frequency,
            hyperedge_membership, embedding_forest, embedding_tree]

Dual-pass: forest embeddings capture broad concept similarity,
tree embeddings capture specific detail similarity. Both come from
ng_embed.py's dual_record_outcome — already in the vector DB.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lenia.interface import LeniaSubstrate

logger = logging.getLogger(__name__)


class NeuroGraphSubstrate(LeniaSubstrate):
    """Wraps a neuro_foundation.py Graph + SimpleVectorDB as a LeniaSubstrate.

    The Graph and VectorDB are not owned — this is a view.
    Changes to the graph (node additions, STDP, etc.) are reflected
    here because we read from the graph directly.
    """

    def __init__(self, graph: Any, vector_db: Any = None):
        """
        Args:
            graph: neuro_foundation.py Graph instance.
            vector_db: SimpleVectorDB instance (for embeddings).
        """
        self._graph = graph
        self._vector_db = vector_db
        self._rebuild_index()

        # Embedding caches — built on first access or populate()
        self._embedding_matrix: Optional[np.ndarray] = None
        self._embedding_norms: Optional[np.ndarray] = None
        self._is_tree_node: Optional[np.ndarray] = None  # bool array
        self._embeddings_cached = False

    def _rebuild_index(self):
        """Build entity_id ↔ index mapping from current graph state."""
        self._entity_list = sorted(self._graph.nodes.keys())
        self._id_to_idx = {
            eid: idx for idx, eid in enumerate(self._entity_list)
        }
        # Cache for BFS distances
        self._topo_cache: Dict[Tuple[str, str], int] = {}
        # Adjacency list cache
        self._adj: Optional[Dict[str, List[str]]] = None
        # Co-fire tracking
        self._cofire_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self._cofire_window: int = 0
        # Invalidate embedding cache
        self._embeddings_cached = False

    def _build_adjacency(self):
        """Build adjacency list from synapses (cached)."""
        self._adj = defaultdict(list)
        for syn in self._graph.synapses.values():
            self._adj[syn.pre_id].append(syn.post_id)
            self._adj[syn.post_id].append(syn.pre_id)

    def _cache_embeddings(self):
        """Build embedding matrix from vector DB for fast cosine similarity.

        Identifies forest vs tree nodes from VDB metadata._tree_concept.
        """
        n = len(self._entity_list)
        if n == 0 or self._vector_db is None:
            self._embeddings_cached = True
            return

        # Determine embedding dimension from first available entry
        dim = None
        for eid in self._entity_list:
            emb = self._vector_db.embeddings.get(eid)
            if emb is not None:
                dim = len(emb)
                break

        if dim is None:
            logger.warning("No embeddings found in vector DB")
            self._embeddings_cached = True
            return

        self._embedding_matrix = np.zeros((n, dim), dtype=np.float64)
        self._is_tree_node = np.zeros(n, dtype=bool)

        found = 0
        for i, eid in enumerate(self._entity_list):
            emb = self._vector_db.embeddings.get(eid)
            if emb is not None:
                self._embedding_matrix[i] = emb
                found += 1

            # Check metadata for tree/forest distinction
            meta = self._vector_db.metadata.get(eid, {})
            if meta.get("_tree_concept", False):
                self._is_tree_node[i] = True

        # Precompute norms for fast cosine sim
        self._embedding_norms = np.linalg.norm(
            self._embedding_matrix, axis=1, keepdims=True
        )
        # Avoid division by zero
        self._embedding_norms = np.where(
            self._embedding_norms < 1e-10, 1.0, self._embedding_norms
        )

        self._embeddings_cached = True
        tree_count = int(self._is_tree_node.sum())
        logger.info(
            "Cached %d/%d embeddings (%d forest, %d tree, dim=%d)",
            found, n, found - tree_count, tree_count, dim,
        )

    def entities(self) -> List[str]:
        return self._entity_list

    def entity_count(self) -> int:
        return len(self._entity_list)

    def channel_count(self) -> int:
        raise NotImplementedError("Use ChannelRegistry.count instead")

    def entity_index(self, entity_id: str) -> int:
        return self._id_to_idx[entity_id]

    def index_to_entity(self, index: int) -> str:
        return self._entity_list[index]

    def distance_vector(self, source_id: str, target_id: str) -> np.ndarray:
        """Compute 6-component distance vector between two nodes.

        Components:
            0: topology (hop count, BFS)
            1: synaptic weight (direct connection weight, 0 if none)
            2: co-fire frequency (normalized)
            3: hyperedge membership (Jaccard similarity)
            4: embedding similarity — forest (broad concept)
            5: embedding similarity — tree (specific detail)
        """
        vec = np.zeros(6, dtype=np.float64)

        vec[0] = self._topology_distance(source_id, target_id)
        vec[1] = self._synaptic_distance(source_id, target_id)

        pair = (min(source_id, target_id), max(source_id, target_id))
        if self._cofire_window > 0:
            vec[2] = self._cofire_counts[pair] / self._cofire_window

        vec[3] = self._hyperedge_similarity(source_id, target_id)

        forest_sim, tree_sim = self._dual_embedding_similarity(
            source_id, target_id
        )
        vec[4] = forest_sim
        vec[5] = tree_sim

        return vec

    def neighbors(
        self, entity_id: str, max_distance: float
    ) -> List[Tuple[str, np.ndarray]]:
        """Find neighbors within max_distance (L2 norm of distance vector)."""
        results = []
        for other_id in self._entity_list:
            if other_id == entity_id:
                continue
            dvec = self.distance_vector(entity_id, other_id)
            if np.linalg.norm(dvec) <= max_distance:
                results.append((other_id, dvec))
        return results

    def record_cofires(self, fired_ids: List[str]):
        """Update co-fire counts for a set of nodes that fired together."""
        self._cofire_window += 1
        fired = set(fired_ids)
        for a in fired:
            for b in fired:
                if a < b:
                    self._cofire_counts[(a, b)] += 1

    def on_topology_change(self):
        """Called when nodes/synapses are added or removed."""
        self._rebuild_index()
        self._topo_cache.clear()
        self._adj = None

    # -- Private distance computation --

    def _topology_distance(self, a: str, b: str) -> float:
        """BFS hop count between two nodes."""
        if a == b:
            return 0.0

        pair = (min(a, b), max(a, b))
        if pair in self._topo_cache:
            return self._topo_cache[pair]

        if self._adj is None:
            self._build_adjacency()

        visited = {a}
        frontier = [a]
        depth = 0
        while frontier:
            depth += 1
            next_frontier = []
            for node in frontier:
                for neighbor in self._adj.get(node, []):
                    if neighbor == b:
                        self._topo_cache[pair] = float(depth)
                        return float(depth)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier

        result = float(len(self._entity_list))
        self._topo_cache[pair] = result
        return result

    def _synaptic_distance(self, a: str, b: str) -> float:
        """Direct synapse weight. Returns 0 if no direct connection."""
        if self._adj is None:
            self._build_adjacency()

        # Fast check: are they even neighbors?
        if b not in self._adj.get(a, []):
            return 0.0

        for syn in self._graph.synapses.values():
            if (syn.pre_id == a and syn.post_id == b) or (
                syn.pre_id == b and syn.post_id == a
            ):
                return float(syn.weight)
        return 0.0

    def _hyperedge_similarity(self, a: str, b: str) -> float:
        """Jaccard similarity of hyperedge membership."""
        hes_a = set()
        hes_b = set()
        for he in self._graph.hyperedges.values():
            members = set(
                he.node_ids if hasattr(he, "node_ids") else
                (he.member_nodes if hasattr(he, "member_nodes") else [])
            )
            if a in members:
                hes_a.add(he.hyperedge_id)
            if b in members:
                hes_b.add(he.hyperedge_id)

        if not hes_a and not hes_b:
            return 0.0
        intersection = hes_a & hes_b
        union = hes_a | hes_b
        return len(intersection) / len(union) if union else 0.0

    def _dual_embedding_similarity(
        self, a: str, b: str
    ) -> Tuple[float, float]:
        """Compute forest and tree embedding similarity separately.

        Returns (forest_similarity, tree_similarity).

        Logic:
        - Both forest nodes: similarity goes into forest component
        - Both tree nodes: similarity goes into tree component
        - One forest, one tree: similarity goes into tree component
          (the tree is a detail of a broader concept — the detail
          scale is more informative for mixed pairs)
        - No embeddings: both 0.0
        """
        if not self._embeddings_cached:
            self._cache_embeddings()

        if self._embedding_matrix is None:
            return 0.0, 0.0

        try:
            i = self._id_to_idx[a]
            j = self._id_to_idx[b]
        except KeyError:
            return 0.0, 0.0

        # Fast cosine similarity from cached matrix
        emb_a = self._embedding_matrix[i]
        emb_b = self._embedding_matrix[j]
        norm_a = self._embedding_norms[i, 0]
        norm_b = self._embedding_norms[j, 0]

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0, 0.0

        sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))

        is_tree_a = self._is_tree_node[i]
        is_tree_b = self._is_tree_node[j]

        if not is_tree_a and not is_tree_b:
            # Both forest: broad concept similarity
            return sim, 0.0
        elif is_tree_a and is_tree_b:
            # Both tree: specific detail similarity
            return 0.0, sim
        else:
            # Mixed: goes into tree (detail scale)
            return 0.0, sim
