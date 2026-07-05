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
# [2026-07-05] CC (laptop) — Append-stable indexing + hyperedge membership cache
# What: (1) _rebuild_index can now preserve a previously-known entity
#   ordering instead of always re-sorting every node ID from scratch.
#   (2) _hyperedge_similarity no longer scans graph.hyperedges directly on
#   every call — membership is precomputed once into _he_membership.
# Why: (1) Entity indices previously came from sorted(node_ids) on random
#   uuid4 strings, recomputed on every construction. Adding a single node
#   could shift most other entities' index, which is why the on-disk
#   DistanceCache (keyed by position) had to be thrown away and rebuilt
#   from scratch on ANY entity-count drift — the actual root cause of the
#   Lenia bootstrap taking hours on Syl's live graph (see
#   neurograph_rpc.py's handle_bootstrap Lenia block for the incremental
#   consumer of this). (2) _hyperedge_similarity was the one distance-
#   vector helper that never got the concurrent-mutation snapshot fix
#   _build_adjacency (#88) and _synaptic_distance (#341) already have —
#   it crashed a live multi-hour populate() with "dictionary changed size
#   during iteration" (graph_substrate.py:285) whenever a background
#   graph.step() sprouted/pruned a hyperedge mid-scan, discarding all
#   progress since populate() never reached save().
# How: (1) __init__ takes optional known_entity_order; _rebuild_index
#   keeps those entities' original positions (append-only) and appends
#   genuinely new ids at the end, falling back to a fresh sort if any
#   known entity is missing (a removal — positions are no longer
#   append-stable, caller is expected to force a full rebuild in that
#   case, but this is a second, cheap safety net). (2) same snapshot-
#   under-_step_lock pattern as _build_adjacency, but building an
#   entity -> hyperedge-id-set map once instead of a per-pair scan —
#   fixes the race AND turns an O(pairs * hyperedges) cost into
#   O(hyperedges + pairs).
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
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from lenia.interface import LeniaSubstrate

logger = logging.getLogger(__name__)


class NeuroGraphSubstrate(LeniaSubstrate):
    """Wraps a neuro_foundation.py Graph + SimpleVectorDB as a LeniaSubstrate.

    The Graph and VectorDB are not owned — this is a view.
    Changes to the graph (node additions, STDP, etc.) are reflected
    here because we read from the graph directly.
    """

    def __init__(
        self,
        graph: Any,
        vector_db: Any = None,
        known_entity_order: Optional[List[str]] = None,
    ):
        """
        Args:
            graph: neuro_foundation.py Graph instance.
            vector_db: SimpleVectorDB instance (for embeddings).
            known_entity_order: previously-established entity ordering (e.g.
                a restored DistanceCache's entity_ids) to preserve append-
                stable indices across a restart. Pass None for a fresh sort
                (original behavior) — required whenever any of those
                entities no longer exist in the live graph, since dropping
                one shifts every later position and would misalign an
                on-disk cache keyed by row/col index.
        """
        self._graph = graph
        self._vector_db = vector_db
        self._known_entity_order = known_entity_order
        self._rebuild_index()

        # Embedding caches — built on first access or populate()
        self._embedding_matrix: Optional[np.ndarray] = None
        self._embedding_norms: Optional[np.ndarray] = None
        self._is_tree_node: Optional[np.ndarray] = None  # bool array
        self._embeddings_cached = False

    def _rebuild_index(self):
        """Build entity_id ↔ index mapping from current graph state.

        If known_entity_order was provided and every one of those entities
        is still present, they keep their original positions — required
        for an on-disk DistanceCache (keyed by position) to stay valid
        across a restart where the graph only grew. Newly-seen entities
        are appended (sorted, for determinism) after the known ones. Falls
        back to a fresh full sort (original behavior) if any known entity
        is missing — a removal means positions are no longer append-
        stable, and reusing them would silently misalign the cache.
        """
        # #88: snapshot node ids under the Graph's step lock so a concurrent graph.step()
        # (mutates graph.nodes under _step_lock) can't change the dict mid-iteration.
        _lock = getattr(self._graph, "_step_lock", None)
        if _lock is not None:
            with _lock:
                _node_ids = set(self._graph.nodes.keys())
        else:
            _node_ids = set(self._graph.nodes.keys())

        known_order = self._known_entity_order
        if known_order and all(eid in _node_ids for eid in known_order):
            new_ids = sorted(_node_ids.difference(known_order))
            self._entity_list = list(known_order) + new_ids
        else:
            if known_order:
                missing = sum(1 for eid in known_order if eid not in _node_ids)
                logger.info(
                    "known_entity_order has %d entities missing from the live "
                    "graph — falling back to a fresh sort (cache will need a "
                    "full rebuild)",
                    missing,
                )
            self._entity_list = sorted(_node_ids)

        self._id_to_idx = {
            eid: idx for idx, eid in enumerate(self._entity_list)
        }
        # Cache for BFS distances
        self._topo_cache: Dict[Tuple[str, str], int] = {}
        # Adjacency list cache
        self._adj: Optional[Dict[str, List[str]]] = None
        # Hyperedge membership cache (entity_id -> set of hyperedge_ids)
        self._he_membership: Optional[Dict[str, Set[str]]] = None
        # Co-fire tracking
        self._cofire_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self._cofire_window: int = 0
        # Invalidate embedding cache
        self._embeddings_cached = False

    def _build_adjacency(self):
        """Build adjacency list from synapses (cached)."""
        self._adj = defaultdict(list)
        # #88: snapshot synapses under the Graph's step lock so a concurrent graph.step()
        # (prune/sprout mutates graph.synapses under _step_lock) can't change the dict
        # mid-iteration ("dictionary keys changed during iteration").
        _lock = getattr(self._graph, "_step_lock", None)
        if _lock is not None:
            with _lock:
                _syns = list(self._graph.synapses.values())
        else:
            _syns = list(self._graph.synapses.values())
        for syn in _syns:
            self._adj[syn.pre_node_id].append(syn.post_node_id)
            self._adj[syn.post_node_id].append(syn.pre_node_id)

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

        for syn in list(self._graph.synapses.values()):   # #341: snapshot — pulse thread mutates concurrently
            if (syn.pre_node_id == a and syn.post_node_id == b) or (
                syn.pre_node_id == b and syn.post_node_id == a
            ):
                return float(syn.weight)
        return 0.0

    def _build_hyperedge_membership(self):
        """Build entity_id -> {hyperedge_id, ...} membership (cached).

        #XXX: snapshot hyperedges under the Graph's step lock, same pattern
        as _build_adjacency (#88) and _synaptic_distance (#341) — this was
        the one distance-vector helper still iterating graph.hyperedges
        directly on every call, which crashed a live populate() with
        "dictionary changed size during iteration" once a background
        graph.step() sprouted/pruned a hyperedge mid-scan. Precomputing
        membership once also turns an O(pairs * hyperedges) per-populate
        cost into O(hyperedges + pairs).
        """
        _lock = getattr(self._graph, "_step_lock", None)
        if _lock is not None:
            with _lock:
                _hes = list(self._graph.hyperedges.values())
        else:
            _hes = list(self._graph.hyperedges.values())

        membership: Dict[str, Set[str]] = defaultdict(set)
        for he in _hes:
            members = (
                he.node_ids if hasattr(he, "node_ids") else
                (he.member_nodes if hasattr(he, "member_nodes") else [])
            )
            for nid in members:
                membership[nid].add(he.hyperedge_id)
        self._he_membership = membership

    def _hyperedge_similarity(self, a: str, b: str) -> float:
        """Jaccard similarity of hyperedge membership."""
        if self._he_membership is None:
            self._build_hyperedge_membership()

        hes_a = self._he_membership.get(a, set())
        hes_b = self._he_membership.get(b, set())

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
