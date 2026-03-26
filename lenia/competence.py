# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Competence meter — observable substrate maturation metrics
# Why: Apprentice/Journeyman/Master as emergent regime, not labels
# How: Three metrics: field entropy, stable pattern count, myelination coverage
# PRD: ~/docs/prd/Lenia_FlowGraph_Design_v0.1.md §2, §3
# -------------------

"""Competence meter — substrate maturation as observable properties.

Three floats. No labels. No Apprentice/Journeyman/Master strings.
The dynamics naturally produce different behavior at different
maturation levels because the field physics changes character.

Metrics (all 0.0–1.0):
    - field_entropy: energy distribution uniformity (high = Apprentice)
    - pattern_stability: fraction of persistent structures (low = Apprentice)
    - myelination_coverage: fraction of pathways myelinated (low = Apprentice)
"""

import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from lenia.config import LeniaConfig
from lenia.myelination import MyelinationObserver

logger = logging.getLogger(__name__)


class CompetenceMeter:
    """Computes substrate maturation metrics once per SNN step.

    Pure observer. Three floats stored and queryable.
    """

    def __init__(
        self,
        config: LeniaConfig,
        myelination_observer: MyelinationObserver,
    ):
        self._config = config
        self._myelination = myelination_observer

        # Current metric values
        self.field_entropy: float = 1.0  # start at max entropy (Apprentice)
        self.pattern_stability: float = 0.0
        self.myelination_coverage: float = 0.0

        # Pattern tracking for stability
        self._fingerprint_history: Deque[Dict[int, int]] = deque(
            maxlen=config.pattern_persistence_window
        )

    def update(self, field_state: np.ndarray):
        """Recompute all three metrics from current field state.

        Called once per SNN step (not per field tick — these are slow-moving).
        """
        self._compute_entropy(field_state)
        self._compute_pattern_stability(field_state)
        self._compute_myelination_coverage()

    def _compute_entropy(self, field_state: np.ndarray):
        """Shannon entropy of energy distribution, normalized to [0, 1]."""
        flat = field_state.flatten()
        total = flat.sum()
        if total < 1e-15:
            self.field_entropy = 1.0  # undefined → max entropy
            return

        # Normalize to probability distribution
        probs = flat / total
        # Filter zeros for log
        nonzero = probs[probs > 1e-15]
        entropy = -float(np.sum(nonzero * np.log(nonzero)))

        # Normalize by max possible entropy
        max_entropy = np.log(len(flat))
        if max_entropy > 0:
            self.field_entropy = entropy / max_entropy
        else:
            self.field_entropy = 1.0

    def _compute_pattern_stability(self, field_state: np.ndarray):
        """Count persistent field patterns.

        A pattern = a cluster of nodes with similar quantized field fingerprints.
        Persistent = cluster membership stable across the tracking window.
        """
        q = self._config.pattern_quantization
        n_entities = field_state.shape[0]

        # Quantize each node's field state to a fingerprint
        quantized = np.round(field_state / q) * q
        fingerprints = {}
        for i in range(n_entities):
            fp = tuple(quantized[i, :].tolist())
            fingerprints.setdefault(fp, []).append(i)

        # Count clusters above minimum size
        cluster_ids = {}
        cluster_counter = 0
        for fp, members in fingerprints.items():
            if len(members) >= self._config.min_cluster_size:
                for m in members:
                    cluster_ids[m] = cluster_counter
                cluster_counter += 1

        self._fingerprint_history.append(cluster_ids)

        if len(self._fingerprint_history) < 2:
            self.pattern_stability = 0.0
            return

        # Compare current clusters to oldest in window
        oldest = self._fingerprint_history[0]
        current = self._fingerprint_history[-1]

        if not oldest or not current:
            self.pattern_stability = 0.0
            return

        # Count entities that are in the same cluster as their oldest assignment
        stable_count = 0
        for entity_idx, current_cluster in current.items():
            if entity_idx in oldest:
                if oldest[entity_idx] == current_cluster:
                    stable_count += 1

        total_clustered = max(len(current), 1)
        self.pattern_stability = stable_count / total_clustered

    def _compute_myelination_coverage(self):
        """Fraction of observed pathways with myelination above threshold."""
        self.myelination_coverage = self._myelination.coverage(
            self._config.myelination_coverage_threshold
        )

    def profile(self) -> Dict[str, float]:
        """Return the maturation profile as a dict."""
        return {
            "field_entropy": round(self.field_entropy, 4),
            "pattern_stability": round(self.pattern_stability, 4),
            "myelination_coverage": round(self.myelination_coverage, 4),
        }
