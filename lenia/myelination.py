# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Myelination observer — measures emergent pathway stability
# Why: Myelination is a continuous index, not a flag. Real computational effects.
# How: Tracks persistence, perturbation resistance, reconstruction per pathway
# PRD: ~/docs/prd/Lenia_FlowGraph_Design_v0.1.md §2, §3
# -------------------

"""Myelination observer.

Doesn't create or control myelination. Observes field dynamics and computes
the myelination index (0.0–1.0) for each pathway.

Myelination index produces real effects:
    - Signal speed: myelinated pathways need fewer intermediate update steps
    - Energy cost: lower energy per signal propagated
    - Fidelity: lower noise from neighboring fluctuations
"""

import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from lenia.config import LeniaConfig

logger = logging.getLogger(__name__)


class PathwayState:
    """Tracking state for one pathway."""

    def __init__(self, node_indices: List[int], window_size: int):
        self.node_indices = node_indices
        self._patterns: Deque[np.ndarray] = deque(maxlen=window_size)
        self._perturbation_scores: Deque[float] = deque(maxlen=window_size)
        self._reconstruction_events: int = 0
        self._reconstruction_attempts: int = 0
        self._disrupted = False
        self._pre_disruption_pattern: Optional[np.ndarray] = None

    @property
    def persistence(self) -> float:
        """How stable is the field pattern along this pathway?

        Cosine similarity between current and oldest pattern in window.
        """
        if len(self._patterns) < 2:
            return 0.0
        oldest = self._patterns[0]
        newest = self._patterns[-1]
        norm_old = np.linalg.norm(oldest)
        norm_new = np.linalg.norm(newest)
        if norm_old < 1e-10 or norm_new < 1e-10:
            return 0.0
        return float(np.dot(oldest, newest) / (norm_old * norm_new))

    @property
    def perturbation_resistance(self) -> float:
        """How little does this pathway deform when neighbors fluctuate?

        Inverse of mean perturbation score. 1.0 = perfectly resistant.
        """
        if not self._perturbation_scores:
            return 0.0
        mean_perturb = sum(self._perturbation_scores) / len(
            self._perturbation_scores
        )
        return 1.0 / (1.0 + mean_perturb)

    @property
    def reconstruction(self) -> float:
        """If disrupted, how often does the pattern reform?

        Ratio of successful reconstructions to attempts.
        """
        if self._reconstruction_attempts == 0:
            return 0.5  # no data = neutral
        return self._reconstruction_events / self._reconstruction_attempts

    def record_pattern(self, field_state: np.ndarray):
        """Record the current field pattern along this pathway."""
        pattern = field_state[self.node_indices, :].flatten()
        self._patterns.append(pattern.copy())

    def record_perturbation(
        self, field_state: np.ndarray, prev_state: np.ndarray
    ):
        """Measure how much the pathway pattern changed from previous state.

        Only meaningful when neighboring nodes had large deltas.
        """
        current = field_state[self.node_indices, :].flatten()
        previous = prev_state[self.node_indices, :].flatten()
        delta = np.linalg.norm(current - previous)
        self._perturbation_scores.append(delta)

    def check_disruption(self, field_state: np.ndarray, threshold: float):
        """Check if the pathway pattern was disrupted (deviates significantly)."""
        if len(self._patterns) < 2:
            return
        current = field_state[self.node_indices, :].flatten()
        reference = self._patterns[-2]  # previous recorded pattern
        norm_ref = np.linalg.norm(reference)
        if norm_ref < 1e-10:
            return
        deviation = np.linalg.norm(current - reference) / norm_ref

        if deviation > threshold and not self._disrupted:
            # Disruption detected
            self._disrupted = True
            self._pre_disruption_pattern = reference.copy()
            self._reconstruction_attempts += 1

        elif self._disrupted and deviation < threshold * 0.5:
            # Reconstruction detected
            self._disrupted = False
            self._reconstruction_events += 1
            self._pre_disruption_pattern = None


class MyelinationObserver:
    """Observes field dynamics and computes myelination index per pathway.

    Pure observer — doesn't create or control myelination.
    """

    def __init__(self, config: LeniaConfig):
        self._config = config
        self._pathways: Dict[str, PathwayState] = {}
        self._prev_field_state: Optional[np.ndarray] = None

    def register_pathway(
        self, pathway_id: str, node_indices: List[int]
    ):
        """Register a pathway to observe.

        Args:
            pathway_id: Identifier for the pathway.
            node_indices: Ordered list of entity indices forming the pathway.
        """
        self._pathways[pathway_id] = PathwayState(
            node_indices, self._config.persistence_window
        )

    def unregister_pathway(self, pathway_id: str):
        """Stop observing a pathway."""
        self._pathways.pop(pathway_id, None)

    def update(self, field_state: np.ndarray):
        """Update all pathway observations with current field state.

        Called after each field tick (by the update engine).
        """
        for pathway in self._pathways.values():
            pathway.record_pattern(field_state)

            if self._prev_field_state is not None:
                pathway.record_perturbation(
                    field_state, self._prev_field_state
                )
                pathway.check_disruption(
                    field_state, self._config.perturbation_sensitivity
                )

        self._prev_field_state = field_state.copy()

    def myelination_index(self, pathway_id: str) -> float:
        """Compute the myelination index for a pathway.

        Weighted combination of persistence, resistance, reconstruction.
        Returns 0.0–1.0.
        """
        pathway = self._pathways.get(pathway_id)
        if pathway is None:
            return 0.0

        w = self._config.myelination_weights
        raw = (
            w[0] * pathway.persistence
            + w[1] * pathway.perturbation_resistance
            + w[2] * pathway.reconstruction
        )
        return max(0.0, min(1.0, raw))

    def all_indices(self) -> Dict[str, float]:
        """Return myelination index for all registered pathways."""
        return {
            pid: self.myelination_index(pid)
            for pid in self._pathways
        }

    def coverage(self, threshold: float = None) -> float:
        """Fraction of pathways with myelination index above threshold."""
        if not self._pathways:
            return 0.0
        if threshold is None:
            threshold = self._config.myelination_coverage_threshold
        above = sum(
            1
            for pid in self._pathways
            if self.myelination_index(pid) > threshold
        )
        return above / len(self._pathways)
