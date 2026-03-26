# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Concrete LeniaSubstrate implementation wrapping NeuroGraph's Graph
# Why: Maps Lenia's abstract entity/distance interface to NG's nodes/synapses
# How: Wraps graph.nodes, graph.synapses, computes 5-metric distance vectors
# PRD: ~/docs/prd/Lenia_FlowGraph_Design_v0.1.md §4
# -------------------

"""NeuroGraph-specific implementation of the LeniaSubstrate interface.

entity = graph node
distance = [topology_hops, synaptic_weight, cofire_frequency,
            hyperedge_membership, embedding_similarity]
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lenia.interface import LeniaSubstrate

logger = logging.getLogger(__name__)


class NeuroGraphSubstrate(LeniaSubstrate):
    """Wraps a neuro_foundation.py Graph as a LeniaSubstrate.

    The Graph is not owned — this is a view over the existing graph.
    Changes to the graph (node additions, STDP, etc.) are reflected
    here because we read from the graph directly.
    """

    def __init__(self, graph: Any):
        """
        Args:
            graph: neuro_foundation.py Graph instance.
        """
        self._graph = graph
        self._rebuild_index()

    def _rebuild_index(self):
        """Build entity_id ↔ index mapping from current graph state."""
        self._entity_list = sorted(self._graph.nodes.keys())
        self._id_to_idx = {
            eid: idx for idx, eid in enumerate(self._entity_list)
        }
        # Cache for BFS distances
        self._topo_cache: Dict[Tuple[str, str], int] = {}
        # Co-fire tracking
        self._cofire_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self._cofire_window: int = 0

    def entities(self) -> List[str]:
        return self._entity_list

    def entity_count(self) -> int:
        return len(self._entity_list)

    def channel_count(self) -> int:
        # Delegated to channel registry, not stored here
        raise NotImplementedError("Use ChannelRegistry.count instead")

    def entity_index(self, entity_id: str) -> int:
        return self._id_to_idx[entity_id]

    def index_to_entity(self, index: int) -> str:
        return self._entity_list[index]

    def distance_vector(self, source_id: str, target_id: str) -> np.ndarray:
        """Compute 5-component distance vector between two nodes.

        Components:
            0: topology (hop count, BFS)
            1: synaptic weight (direct connection weight, 0 if no direct synapse)
            2: co-fire frequency (normalized)
            3: hyperedge membership (Jaccard similarity of shared HEs)
            4: embedding similarity (cosine similarity, 0 if no embeddings)
        """
        vec = np.zeros(5, dtype=np.float64)

        # 0: Topological distance (BFS hop count)
        vec[0] = self._topology_distance(source_id, target_id)

        # 1: Synaptic weight (direct connection)
        vec[1] = self._synaptic_distance(source_id, target_id)

        # 2: Co-fire frequency
        pair = (min(source_id, target_id), max(source_id, target_id))
        if self._cofire_window > 0:
            vec[2] = self._cofire_counts[pair] / self._cofire_window
        else:
            vec[2] = 0.0

        # 3: Hyperedge membership (Jaccard)
        vec[3] = self._hyperedge_similarity(source_id, target_id)

        # 4: Embedding similarity
        vec[4] = self._embedding_similarity(source_id, target_id)

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

    # -- Private distance computation --

    def _topology_distance(self, a: str, b: str) -> float:
        """BFS hop count between two nodes. Returns inf if unreachable."""
        if a == b:
            return 0.0

        pair = (min(a, b), max(a, b))
        if pair in self._topo_cache:
            return self._topo_cache[pair]

        # Build adjacency from synapses
        adj: Dict[str, List[str]] = defaultdict(list)
        for syn in self._graph.synapses.values():
            adj[syn.pre_id].append(syn.post_id)
            adj[syn.post_id].append(syn.pre_id)

        # BFS
        visited = {a}
        frontier = [a]
        depth = 0
        while frontier:
            depth += 1
            next_frontier = []
            for node in frontier:
                for neighbor in adj.get(node, []):
                    if neighbor == b:
                        self._topo_cache[pair] = float(depth)
                        return float(depth)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier

        # Unreachable — use a large but finite value
        result = float(len(self._entity_list))
        self._topo_cache[pair] = result
        return result

    def _synaptic_distance(self, a: str, b: str) -> float:
        """Direct synapse weight. Returns 0 if no direct connection."""
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
            members = set(he.node_ids) if hasattr(he, "node_ids") else set()
            if a in members:
                hes_a.add(he.hyperedge_id)
            if b in members:
                hes_b.add(he.hyperedge_id)

        if not hes_a and not hes_b:
            return 0.0
        intersection = hes_a & hes_b
        union = hes_a | hes_b
        return len(intersection) / len(union) if union else 0.0

    def _embedding_similarity(self, a: str, b: str) -> float:
        """Cosine similarity of node embeddings."""
        node_a = self._graph.nodes.get(a)
        node_b = self._graph.nodes.get(b)
        if node_a is None or node_b is None:
            return 0.0

        emb_a = getattr(node_a, "embedding", None)
        emb_b = getattr(node_b, "embedding", None)
        if emb_a is None or emb_b is None:
            return 0.0

        emb_a = np.asarray(emb_a, dtype=np.float64)
        emb_b = np.asarray(emb_b, dtype=np.float64)

        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
