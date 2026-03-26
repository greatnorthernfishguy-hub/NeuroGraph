# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: LeniaSubstrate abstract interface
# Why: Decouples Lenia dynamics from concrete substrate (NeuroGraph now, UniAI later)
# How: Pure state access contract — entities, field state, distances, energy injection
# PRD: ~/docs/prd/Lenia_FlowGraph_Design_v0.1.md §3
# -------------------

"""Abstract interface for any substrate that Lenia dynamics can operate on.

NeuroGraph implementation: entity = node, distance = graph metrics.
Future UniAI implementation: entity = weight position, distance = structural.

Same kernels, same growth functions, same conservation law. Different substrate.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class LeniaSubstrate(ABC):
    """Pure state access interface for Lenia dynamics.

    The dynamics (kernel, growth, conservation) operate ON this interface,
    not inside it. This is state access only.
    """

    @abstractmethod
    def entities(self) -> List[str]:
        """Return all entity IDs in the substrate."""

    @abstractmethod
    def entity_count(self) -> int:
        """Return number of entities."""

    @abstractmethod
    def channel_count(self) -> int:
        """Return current number of active channels."""

    @abstractmethod
    def distance_vector(self, source_id: str, target_id: str) -> np.ndarray:
        """Return multi-metric distance vector between two entities.

        Components: [topology, synaptic, cofire, hyperedge, embedding]
        """

    @abstractmethod
    def neighbors(
        self, entity_id: str, max_distance: float
    ) -> List[Tuple[str, np.ndarray]]:
        """Return entities within max_distance with their distance vectors.

        Distance is composite — max_distance applies to the L2 norm of the
        full distance vector.
        """

    @abstractmethod
    def entity_index(self, entity_id: str) -> int:
        """Return the integer index for an entity (for field store access)."""

    @abstractmethod
    def index_to_entity(self, index: int) -> str:
        """Return entity ID from integer index."""
