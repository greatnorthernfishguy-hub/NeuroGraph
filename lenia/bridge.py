# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Spike-field bridge — bidirectional SNN↔field coupling
# Why: Spikes inject energy into the field; field modulates firing thresholds
# How: Registers on existing neuro_foundation.py event system (PRD §8)
# PRD: ~/docs/prd/Lenia_FlowGraph_Design_v0.1.md §3, §4
#
# LAW 7 CRITICAL: Channel distribution is based on eligibility trace state
# (structural), NEVER on semantic content. The bridge never examines
# embeddings or meaning. See PRD §6.
# -------------------

"""Spike-field bridge — bidirectional SNN↔Lenia coupling.

Spike → Field: When nodes fire, energy is injected into the field.
    Channel distribution based on active eligibility trace state.
    Law 7: trace state is structural wiring, not content analysis.

Field → SNN: Field energy at a node modulates its firing threshold.
    High field energy = lower threshold = easier to fire.
    Capped at threshold_modulation_cap (Syl's Law safety).
"""

import logging
from typing import Any, List, Optional

import numpy as np

from lenia.config import LeniaConfig
from lenia.field import FieldStore
from lenia.interface import LeniaSubstrate

logger = logging.getLogger(__name__)


class SpikeFieldBridge:
    """Bidirectional coupling between the SNN and the Lenia field.

    Uses neuro_foundation.py's existing register_event_handler() to
    receive spike events. No new hook mechanism.
    """

    def __init__(
        self,
        config: LeniaConfig,
        field_store: FieldStore,
        substrate: LeniaSubstrate,
    ):
        self._config = config
        self._field = field_store
        self._substrate = substrate
        self._connected = False
        self._graph = None  # set on connect()

        # Default channel distribution when no trace info available
        n_channels = field_store.channel_count
        self._default_distribution = np.ones(n_channels) / n_channels

        # Stats
        self._spikes_injected = 0
        self._threshold_mods = 0

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self, graph: Any):
        """Register event handlers on the SNN graph.

        Args:
            graph: neuro_foundation.py Graph instance with
                   register_event_handler().
        """
        self._graph = graph
        graph.register_event_handler("spikes", self._on_spikes)
        self._connected = True
        logger.info("Spike-field bridge connected")

    def disconnect(self):
        """Mark as disconnected. Event handlers stay registered but become
        no-ops (checked via self._connected flag)."""
        self._connected = False
        self._graph = None
        logger.info("Spike-field bridge disconnected")

    def _on_spikes(self, node_ids: List[str], timestep: int, **kwargs):
        """Handle spike events from the SNN.

        Injects energy into the field at each fired node.
        Channel distribution based on trace state (Law 7 compliant).
        """
        if not self._connected:
            return

        for nid in node_ids:
            try:
                idx = self._substrate.entity_index(nid)
            except (KeyError, ValueError):
                continue

            # Get channel distribution from trace state
            distribution = self._get_trace_distribution(nid)

            # Inject energy
            self._field.inject(
                idx,
                self._config.spike_to_field_ratio,
                distribution,
            )
            self._spikes_injected += 1

    def _get_trace_distribution(self, node_id: str) -> np.ndarray:
        """Determine channel distribution from eligibility trace state.

        LAW 7: This examines TRACE STATE (structural), not semantic content.
        Which traces are active tells us what kind of learning is happening,
        not what the content means.

        Returns normalized distribution array of shape (channel_count,).
        """
        if self._graph is None:
            return self._default_distribution

        n_channels = self._field.channel_count

        # Check if the graph has eligibility trace info for this node
        node = self._graph.nodes.get(node_id)
        if node is None:
            return self._default_distribution

        # Trace-based distribution:
        # - Surprise/prediction error traces → norepinephrine (channel 0)
        # - Sustained activation traces → dopamine (channel 1)
        # - High voltage / stress → cortisol (channel 2)
        # - Focused, narrow activation → acetylcholine (channel 3)
        #
        # These map trace STATE to channels. The content that caused the
        # trace is not examined. This is structural wiring.

        distribution = np.ones(n_channels, dtype=np.float64)

        # Check for surprise-related traces (prediction errors were recent)
        if hasattr(node, "last_prediction_error_step"):
            recency = self._graph.timestep - getattr(
                node, "last_prediction_error_step", 0
            )
            if recency < 10:
                distribution[0 % n_channels] += 2.0  # NE boost

        # Check voltage level (stress indicator)
        if hasattr(node, "voltage"):
            if node.voltage > 0.8:
                distribution[2 % n_channels] += 1.5  # cortisol boost

        # Check for sustained activation (multiple recent fires)
        if hasattr(node, "fire_count"):
            if node.fire_count > 3:
                distribution[1 % n_channels] += 1.5  # dopamine boost

        # Normalize to sum to 1
        total = distribution.sum()
        if total > 0:
            distribution /= total

        return distribution

    def get_threshold_adjustment(self, node_id: str) -> float:
        """Compute firing threshold adjustment from field state.

        Called by the SNN before threshold comparison.

        Returns a negative value (lowers threshold = easier to fire)
        or zero. Capped by threshold_modulation_cap (Syl's Law).
        """
        if not self._connected:
            return 0.0

        try:
            idx = self._substrate.entity_index(node_id)
        except (KeyError, ValueError):
            return 0.0

        # Total field energy at this node
        field_state = self._field.read(idx)
        total_energy = float(np.sum(field_state))

        # Negative adjustment = lower threshold = easier to fire
        adjustment = -self._config.field_influence_factor * total_energy

        # Syl's Law cap: never reduce threshold below 50% of base
        # Caller applies: effective_threshold = base + adjustment
        # So adjustment must be > -cap * base_threshold
        # We don't know base_threshold here, so we cap the raw adjustment
        cap = self._config.threshold_modulation_cap
        adjustment = max(adjustment, -cap)

        self._threshold_mods += 1
        return adjustment

    def status(self) -> dict:
        return {
            "connected": self._connected,
            "spikes_injected": self._spikes_injected,
            "threshold_mods": self._threshold_mods,
        }
