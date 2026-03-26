# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Multi-metric kernel computer with sparse distance cache
# Why: Lenia kernels operate over composite distance, not single-metric
# How: Sparse cache per distance component, lazy invalidation, per-channel ranges
# PRD: ~/docs/prd/Lenia_FlowGraph_Design_v0.1.md §3, §4
# -------------------

"""Multi-metric kernel computer.

Distance between nodes is composite:
    [topology, synaptic_weight, cofire, hyperedge, embedding_similarity]

Each component cached independently with lazy invalidation.
Kernel functions operate on the full distance vector per channel.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse

from lenia.channels import ChannelRegistry
from lenia.functions import evaluate
from lenia.interface import LeniaSubstrate

logger = logging.getLogger(__name__)

# Distance vector component indices
DIST_TOPOLOGY = 0
DIST_SYNAPTIC = 1
DIST_COFIRE = 2
DIST_HYPEREDGE = 3
DIST_EMBEDDING = 4
NUM_DIST_COMPONENTS = 5


class DistanceCache:
    """Sparse, lazily-invalidated distance cache.

    Stores one sparse matrix per distance component.
    Dirty flags per (i, j, component). Rebuild on sweep.
    """

    def __init__(self, entity_count: int):
        self._n = entity_count
        # Each component is a sparse matrix (lil_matrix for efficient updates)
        self._components: List[sparse.lil_matrix] = [
            sparse.lil_matrix((entity_count, entity_count), dtype=np.float64)
            for _ in range(NUM_DIST_COMPONENTS)
        ]
        # Dirty entries: set of (row, col, component_idx)
        self._dirty: Set[Tuple[int, int, int]] = set()
        self._initialized = False

    @property
    def entity_count(self) -> int:
        return self._n

    def set_distance(
        self, i: int, j: int, component: int, value: float
    ):
        """Set a distance component for a pair."""
        self._components[component][i, j] = value
        self._components[component][j, i] = value  # symmetric

    def mark_dirty(self, i: int, j: int, component: int):
        """Mark a distance component as needing recomputation."""
        self._dirty.add((i, j, component))
        self._dirty.add((j, i, component))

    def mark_dirty_entity(self, entity_idx: int, component: int):
        """Mark all pairs involving an entity as dirty for a component."""
        for j in range(self._n):
            if j != entity_idx:
                self._dirty.add((entity_idx, j, component))
                self._dirty.add((j, entity_idx, component))

    def get_distance_vector(self, i: int, j: int) -> np.ndarray:
        """Return the full 5-component distance vector for a pair."""
        vec = np.zeros(NUM_DIST_COMPONENTS, dtype=np.float64)
        for c in range(NUM_DIST_COMPONENTS):
            vec[c] = self._components[c][i, j]
        return vec

    def get_neighbors(
        self, entity_idx: int, max_range: float
    ) -> List[Tuple[int, np.ndarray]]:
        """Return neighbors within max_range (L2 norm of distance vector).

        Returns list of (neighbor_idx, distance_vector).
        """
        neighbors = []
        for j in range(self._n):
            if j == entity_idx:
                continue
            vec = self.get_distance_vector(entity_idx, j)
            if np.linalg.norm(vec) <= max_range:
                neighbors.append((j, vec))
        return neighbors

    def sweep_dirty(self, substrate: LeniaSubstrate):
        """Recompute all dirty entries from the substrate.

        Called once per SNN step, before field ticks begin.
        """
        if not self._dirty:
            return

        entities = substrate.entities()
        for i, j, component in list(self._dirty):
            if i >= len(entities) or j >= len(entities):
                continue
            eid_i = substrate.index_to_entity(i)
            eid_j = substrate.index_to_entity(j)
            dist_vec = substrate.distance_vector(eid_i, eid_j)
            self._components[component][i, j] = dist_vec[component]

        self._dirty.clear()

    def resize(self, new_count: int):
        """Resize for entity addition/removal."""
        new_components = []
        for comp in self._components:
            new_mat = sparse.lil_matrix(
                (new_count, new_count), dtype=np.float64
            )
            copy_n = min(self._n, new_count)
            new_mat[:copy_n, :copy_n] = comp[:copy_n, :copy_n]
            new_components.append(new_mat)
        self._components = new_components
        self._n = new_count

    @property
    def dirty_count(self) -> int:
        return len(self._dirty)


class KernelComputer:
    """Computes kernel-weighted influence for the Lenia update engine.

    For each entity and channel, gathers neighbors within effective range,
    applies the channel's kernel function to the distance vector, and
    returns total weighted influence.
    """

    def __init__(
        self,
        cache: DistanceCache,
        registry: ChannelRegistry,
    ):
        self._cache = cache
        self._registry = registry

    def compute(
        self,
        field_state: np.ndarray,
        channel_id: int,
    ) -> np.ndarray:
        """Compute kernel-weighted influence for all entities on one channel.

        Args:
            field_state: (entity_count, channel_count) current field state.
            channel_id: Which channel to compute.

        Returns:
            (entity_count,) array of total influence per entity.
        """
        n = field_state.shape[0]
        channel_idx = self._registry.channel_index(channel_id)
        effective_range = self._registry.effective_range(channel_id)
        kernel_spec = self._registry.kernel_shape(channel_id)

        influences = np.zeros(n, dtype=np.float64)

        for i in range(n):
            neighbors = self._cache.get_neighbors(i, effective_range)
            if not neighbors:
                continue

            total = 0.0
            for j, dist_vec in neighbors:
                # Kernel maps distance vector magnitude to influence weight
                dist_mag = float(np.linalg.norm(dist_vec))
                weight = float(evaluate(kernel_spec, dist_mag))
                # Weight by neighbor's field state in this channel
                total += weight * field_state[j, channel_idx]

            influences[i] = total

        return influences

    def compute_all_channels(
        self, field_state: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """Compute influences for all channels.

        Returns dict mapping channel_id → influence array.
        """
        result = {}
        for cid in self._registry.channel_ids:
            result[cid] = self.compute(field_state, cid)
        return result
