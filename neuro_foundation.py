"""
NeuroGraph Foundation - Core Cognitive Architecture (Phase 1 + 2 + 3)

Implements the Temporal Dynamics Layer: a sparse Spiking Neural Network (SNN)
with STDP plasticity, homeostatic regulation, structural plasticity,
a full hypergraph engine with pattern completion, adaptive plasticity,
hierarchical composition, automatic discovery, and consolidation,
and a predictive coding engine with prediction tracking, error events,
surprise-driven exploration, and three-factor learning.

Reference: NeuroGraph Foundation PRD v1.0, Sections 2-6, 9 (Phase 1),
           Section 4 (Phase 2 — Hypergraph Engine),
           and Section 5 (Phase 3 — Predictive Coding Engine).

Design principles (PRD §2.1):
    - Sparse by default: dict/set topology, no dense matrices
    - Dynamic topology: nodes/edges created and destroyed at runtime
    - Pluggable plasticity: learning rules are swappable strategy objects
    - Persistence-native: all state is serializable
"""

from __future__ import annotations

import copy
import json
import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np

try:
    import msgpack
except ImportError:
    msgpack = None


# ---------------------------------------------------------------------------
# Enumerations (PRD §2.2.2, §2.2.3)
# ---------------------------------------------------------------------------

class SynapseType(Enum):
    """Synapse functional type (PRD §2.2.2)."""
    EXCITATORY = auto()
    INHIBITORY = auto()
    MODULATORY = auto()


class ActivationMode(Enum):
    """Hyperedge activation mode (PRD §2.2.3)."""
    WEIGHTED_THRESHOLD = auto()
    K_OF_N = auto()
    ALL_OR_NONE = auto()
    GRADED = auto()


class CheckpointMode(Enum):
    """Persistence checkpoint mode (PRD §6.2)."""
    FULL = auto()
    INCREMENTAL = auto()
    FORK = auto()


# ---------------------------------------------------------------------------
# Ring Buffer for spike history
# ---------------------------------------------------------------------------

class RingBuffer:
    """Fixed-size ring buffer for storing recent spike times.

    Used by Node.spike_history to compute firing rates and detect bursts
    (PRD §2.2.1).
    """

    def __init__(self, capacity: int = 100):
        self._capacity = capacity
        self._buffer: Deque[float] = deque(maxlen=capacity)

    def append(self, value: float) -> None:
        self._buffer.append(value)

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)

    def __repr__(self) -> str:
        return f"RingBuffer(capacity={self._capacity}, size={len(self._buffer)})"

    @property
    def capacity(self) -> int:
        return self._capacity

    def to_list(self) -> List[float]:
        return list(self._buffer)

    @classmethod
    def from_list(cls, data: List[float], capacity: int = 100) -> "RingBuffer":
        rb = cls(capacity)
        for v in data:
            rb.append(v)
        return rb


# ---------------------------------------------------------------------------
# Core Data Structures (PRD §2.2)
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """Atomic unit of the graph wrapping neural state (PRD §2.2.1, Table 2.2.1).

    Each Node is a stateful computational unit with its own membrane potential,
    adaptive threshold, refractory state, and spike history.

    Attributes:
        node_id: Globally unique identifier (matches vector DB entry ID).
        voltage: Current membrane potential; accumulates input, resets on spike.
        threshold: Adaptive firing threshold; adjusted via intrinsic plasticity.
        resting_potential: Baseline voltage after reset (default 0.0).
        refractory_remaining: Timesteps left in refractory period; cannot fire while > 0.
        refractory_period: Duration of refractory period in timesteps (default 2).
        last_spike_time: Timestamp of most recent spike, used by STDP.
        spike_history: Rolling window of recent spike times (depth 100).
        firing_rate_ema: Exponential moving average of firing rate for homeostasis.
        intrinsic_excitability: Multiplier on incoming current (default 1.0).
        metadata: Application-specific key-value data.
        is_inhibitory: If True, outgoing spikes subtract from target voltage.
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    voltage: float = 0.0
    threshold: float = 1.0
    resting_potential: float = 0.0
    refractory_remaining: int = 0
    refractory_period: int = 2
    last_spike_time: float = -math.inf
    spike_history: RingBuffer = field(default_factory=lambda: RingBuffer(100))
    firing_rate_ema: float = 0.0
    intrinsic_excitability: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_inhibitory: bool = False


@dataclass
class Synapse:
    """Directed, weighted connection between two nodes (PRD §2.2.2, Table 2.2.2).

    First-class object with its own state for fine-grained plasticity tracking.

    Attributes:
        synapse_id: Unique identifier.
        pre_node_id: Source node (cause in causal links).
        post_node_id: Target node (effect in causal links).
        weight: Connection strength [0.0, max_weight]; shaped by plasticity.
        max_weight: Upper bound preventing runaway potentiation (default 5.0).
        delay: Propagation delay in timesteps (default 1).
        last_update_time: Timestamp of most recent plasticity update.
        eligibility_trace: Decaying trace for three-factor learning.
        creation_time: When synapse was created (age-based pruning).
        synapse_type: EXCITATORY, INHIBITORY, or MODULATORY.
    """

    synapse_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pre_node_id: str = ""
    post_node_id: str = ""
    weight: float = 0.1
    max_weight: float = 5.0
    delay: int = 1
    last_update_time: float = 0.0
    eligibility_trace: float = 0.0
    creation_time: float = 0.0
    synapse_type: SynapseType = SynapseType.EXCITATORY
    # Track peak weight for age-based pruning (PRD §3.3.1)
    peak_weight: float = 0.1
    # Steps spent below weight_threshold for weight-based pruning
    low_weight_steps: int = 0
    # Steps since last pre or post spike traversal
    inactive_steps: int = 0
    # Application-specific metadata (Phase 3: tracks creation_mode for
    # surprise-driven synapses)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hyperedge:
    """Set-valued relationship connecting arbitrary nodes (PRD §2.2.3, Table 2.2.3).

    Represents composite concepts: syndromes, threat signatures, code patterns.

    Attributes:
        hyperedge_id: Unique identifier.
        member_nodes: Node IDs in this relationship.
        member_weights: Per-member importance weights.
        activation_threshold: Weighted fraction of active members needed to fire.
        activation_mode: WEIGHTED_THRESHOLD, K_OF_N, ALL_OR_NONE, or GRADED.
        current_activation: Current activation level [0.0, 1.0].
        output_targets: Nodes receiving input when hyperedge fires.
        output_weight: Signal strength sent to output targets on activation.
        metadata: Application data (label, domain, creation_mode).
        is_learnable: Whether plasticity can modify weights/threshold.
        refractory_period: Minimum timesteps between firings (default 2).
            Prevents cascading feedback loops where a hyperedge's output
            re-activates its own members on the very next step.
        refractory_remaining: Timesteps left in current refractory window.
        activation_count: How many times this hyperedge has fired (for plasticity).
        pattern_completion_strength: Current injected into inactive members
            when the hyperedge fires from partial activation (PRD §4.2).
        child_hyperedges: IDs of child hyperedges for hierarchical composition
            (PRD §4.4).  Level-0 = leaf nodes only, level-N references level-(N-1).
        level: Hierarchy level. 0 = base (members are nodes only).
    """

    hyperedge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    member_nodes: Set[str] = field(default_factory=set)
    member_weights: Dict[str, float] = field(default_factory=dict)
    activation_threshold: float = 0.6
    activation_mode: ActivationMode = ActivationMode.WEIGHTED_THRESHOLD
    current_activation: float = 0.0
    output_targets: List[str] = field(default_factory=list)
    output_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_learnable: bool = True
    refractory_period: int = 2
    refractory_remaining: int = 0
    # Phase 2 fields
    activation_count: int = 0
    pattern_completion_strength: float = 0.3
    child_hyperedges: Set[str] = field(default_factory=set)
    level: int = 0
    # Phase 2.5: Dynamic pattern completion — EMA of recent activation rate
    recent_activation_ema: float = 0.0
    # Phase 2.5: Cross-level consistency — archived flag
    is_archived: bool = False


# ---------------------------------------------------------------------------
# Step Result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result returned from Graph.step() (PRD §8 step method).

    Attributes:
        timestep: The simulation timestep this result corresponds to.
        fired_node_ids: Node IDs that spiked this step.
        fired_hyperedge_ids: Hyperedge IDs that activated this step.
        synapses_pruned: Number of synapses pruned this step.
        synapses_sprouted: Number of new synapses created this step.
    """

    timestep: int = 0
    fired_node_ids: List[str] = field(default_factory=list)
    fired_hyperedge_ids: List[str] = field(default_factory=list)
    synapses_pruned: int = 0
    synapses_sprouted: int = 0
    predictions_confirmed: int = 0
    predictions_surprised: int = 0


# ---------------------------------------------------------------------------
# Prediction State (Phase 2.5 §1)
# ---------------------------------------------------------------------------

@dataclass
class PredictionState:
    """Tracks a pending prediction made by a hyperedge firing.

    When a hyperedge fires, it predicts that its output_targets will fire
    within ``prediction_window`` steps.  The Graph checks each step whether
    the predicted targets actually fired (confirmed) or the window expired
    without firing (surprise).

    Attributes:
        hyperedge_id: Which hyperedge made the prediction.
        predicted_targets: Node IDs expected to fire.
        prediction_strength: Confidence based on hyperedge activation level.
        prediction_timestamp: Timestep when the prediction was created.
        prediction_window: How many steps to wait before declaring surprise.
        confirmed_targets: Targets that have already fired within the window.
    """

    hyperedge_id: str = ""
    predicted_targets: Set[str] = field(default_factory=set)
    prediction_strength: float = 0.0
    prediction_timestamp: int = 0
    prediction_window: int = 5
    confirmed_targets: Set[str] = field(default_factory=set)


@dataclass
class SurpriseEvent:
    """Emitted when a prediction fails — an expected node did not fire.

    Attributes:
        hyperedge_id: Which hyperedge made the failed prediction.
        expected_node: The node that was predicted to fire but didn't.
        prediction_strength: How confident the prediction was.
        actual_nodes: Nodes that did fire during the prediction window.
        timestamp: When the surprise was detected (window expiry step).
    """

    hyperedge_id: str = ""
    expected_node: str = ""
    prediction_strength: float = 0.0
    actual_nodes: Set[str] = field(default_factory=set)
    timestamp: int = 0


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

@dataclass
class Telemetry:
    """Network statistics snapshot (PRD §2.2.4 Telemetry).

    Attributes:
        timestep: Current simulation time.
        total_nodes: Number of nodes in graph.
        total_synapses: Number of synapses.
        total_hyperedges: Number of hyperedges.
        global_firing_rate: Average firing rate across all nodes.
        mean_weight: Mean synapse weight.
        std_weight: Standard deviation of synapse weights.
        total_pruned: Cumulative synapses pruned.
        total_sprouted: Cumulative synapses sprouted.
        total_he_discovered: Cumulative hyperedges auto-discovered.
        total_he_consolidated: Cumulative hyperedge merges.
        mean_he_activation_count: Average firing count per hyperedge.
    """

    timestep: int = 0
    total_nodes: int = 0
    total_synapses: int = 0
    total_hyperedges: int = 0
    global_firing_rate: float = 0.0
    mean_weight: float = 0.0
    std_weight: float = 0.0
    total_pruned: int = 0
    total_sprouted: int = 0
    total_he_discovered: int = 0
    total_he_consolidated: int = 0
    mean_he_activation_count: float = 0.0
    # Phase 3: Predictive Coding telemetry
    prediction_accuracy: float = 0.0
    surprise_rate: float = 0.0
    active_predictions_count: int = 0
    total_predictions_made: int = 0
    total_predictions_confirmed: int = 0
    total_predictions_errors: int = 0
    total_novel_sequences: int = 0
    total_rewards_injected: int = 0
    # Phase 2.5: Experience distribution
    hyperedge_experience_distribution: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 3: Prediction Data Structures (PRD §5)
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """An active prediction that node B will fire after node A fired.

    Generated when a node fires with a strong causal link (weight > threshold)
    to downstream nodes.  The prediction pre-charges the target's voltage and
    is tracked until confirmed or expired (PRD §5.1).

    Attributes:
        prediction_id: Unique identifier.
        source_node_id: Node that fired and generated this prediction.
        target_node_id: Node predicted to fire.
        strength: Prediction strength based on synapse weight.
        confidence: Confidence score [0,1] based on weight, firing stability,
            and historical confirmation rate.
        created_at: Timestep when prediction was generated.
        expires_at: Timestep when prediction expires (prediction window).
        chain_depth: How many hops from the original stimulus (0 = direct).
        via_hyperedge: If prediction came from a hyperedge, its ID.
        pre_charge_applied: Voltage pre-charge applied to target.
    """

    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str = ""
    target_node_id: str = ""
    strength: float = 0.0
    confidence: float = 0.0
    created_at: int = 0
    expires_at: int = 0
    chain_depth: int = 0
    via_hyperedge: Optional[str] = None
    pre_charge_applied: float = 0.0


@dataclass
class PredictionOutcome:
    """Record of a resolved prediction (confirmed or error).

    Attributes:
        prediction: The original prediction.
        confirmed: Whether the target fired within the window.
        resolved_at: Timestep when the prediction was resolved.
        actual_firing_nodes: Nodes that actually fired (for error analysis).
    """

    prediction: Prediction = field(default_factory=Prediction)
    confirmed: bool = False
    resolved_at: int = 0
    actual_firing_nodes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Auto-Knowledge: Spreading Activation Harvest Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FiredEntry:
    """A node that fired during spreading activation propagation.

    Attributes:
        node_id: ID of the node that fired.
        firing_step: Which propagation step it fired on (0-indexed from
            the first propagation step).  Lower = stronger association.
        voltage_at_fire: Voltage at the moment of firing.
        was_predicted: Whether this node was a prediction target.
        source_distance: Hop count from nearest primed node (approximate,
            tracked via breadth-first through synapses).
    """

    node_id: str = ""
    firing_step: int = 0
    voltage_at_fire: float = 0.0
    was_predicted: bool = False
    source_distance: int = 0


@dataclass
class PropagationResult:
    """Result of a prime-and-propagate spreading activation harvest.

    Attributes:
        fired_entries: All nodes that fired during propagation, with metadata.
        steps_run: How many SNN steps were executed.
        nodes_primed: How many nodes received the initial priming current.
    """

    fired_entries: List[FiredEntry] = field(default_factory=list)
    steps_run: int = 0
    nodes_primed: int = 0


# ---------------------------------------------------------------------------
# Plasticity Rules (PRD §3 – pluggable strategy objects)
# ---------------------------------------------------------------------------

class PlasticityRule:
    """Base class for pluggable plasticity rules (PRD §2.1, §3).

    Subclass and override ``apply`` to create custom rules.
    """

    def apply(
        self,
        graph: "Graph",
        fired_node_ids: List[str],
        timestep: int,
    ) -> None:
        raise NotImplementedError


class STDPRule(PlasticityRule):
    """Spike-Timing-Dependent Plasticity (PRD §3.1).

    Mathematical specification (PRD §3.1.1):
        LTP (pre fires before post, Δt > 0):
            Δw = A_plus × exp(−Δt / τ_plus) × learning_rate
        LTD (pre fires after post, Δt < 0):
            Δw = −A_minus × exp(Δt / τ_minus) × learning_rate

    Weight-dependent scaling (PRD §3.1.2, Runaway Potentiation mitigation):
        LTP scaled by (max_weight − w) / max_weight  (soft saturation)

    Temporal aliasing (PRD §3.1.2):
        Δt = 0 treated as weak LTP at half strength.

    Critical: A_minus > A_plus (ratio 1.05–1.2) for stability.
    """

    def __init__(
        self,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        A_plus: float = 1.0,
        A_minus: float = 1.2,
        learning_rate: float = 0.01,
    ):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.learning_rate = learning_rate

    def _apply_dw(self, syn: Synapse, dw: float, timestep: int,
                  three_factor: bool) -> None:
        """Apply weight change directly or via eligibility trace.

        In three-factor mode (PRD §5.2), STDP creates the eligibility trace
        but weight change only commits when reward arrives via inject_reward.
        """
        if three_factor:
            syn.eligibility_trace += dw
        else:
            syn.weight = max(0.0, min(syn.weight + dw, syn.max_weight))
            if syn.weight > syn.peak_weight:
                syn.peak_weight = syn.weight
        syn.last_update_time = float(timestep)

    def apply(
        self,
        graph: "Graph",
        fired_node_ids: List[str],
        timestep: int,
    ) -> None:
        """Apply STDP to all synapses incident on fired nodes.

        When three_factor_enabled is set in graph config, weight changes
        accumulate in eligibility_trace instead of being applied directly
        (PRD §5.2 Three-Factor Learning).
        """
        three_factor = graph.config.get("three_factor_enabled", False)

        for post_id in fired_node_ids:
            post_node = graph.nodes[post_id]
            t_post = float(timestep)

            # Iterate over all incoming synapses to this post node
            incoming_syn_ids = graph._incoming.get(post_id, set())
            for syn_id in list(incoming_syn_ids):
                syn = graph.synapses.get(syn_id)
                if syn is None:
                    continue
                pre_node = graph.nodes.get(syn.pre_node_id)
                if pre_node is None:
                    continue

                t_pre = pre_node.last_spike_time
                if t_pre == -math.inf:
                    continue

                dt = t_post - t_pre

                if dt > 0:
                    # LTP: pre fired before post (causal)
                    raw_dw = self.A_plus * math.exp(-dt / self.tau_plus)
                    # Weight-dependent scaling (PRD §3.1.2)
                    scale = (syn.max_weight - syn.weight) / syn.max_weight
                    dw = raw_dw * self.learning_rate * max(scale, 0.0)
                elif dt < 0:
                    # LTD: pre fired after post (acausal)
                    raw_dw = -self.A_minus * math.exp(dt / self.tau_minus)
                    dw = raw_dw * self.learning_rate
                else:
                    # Temporal aliasing: Δt=0 → weak LTP at half strength (PRD §3.1.2)
                    raw_dw = self.A_plus * 0.5
                    scale = (syn.max_weight - syn.weight) / syn.max_weight
                    dw = raw_dw * self.learning_rate * max(scale, 0.0)

                self._apply_dw(syn, dw, timestep, three_factor)

            # Also handle outgoing synapses (post-before-pre → LTD from
            # perspective of those synapses where this node is pre)
            outgoing_syn_ids = graph._outgoing.get(post_id, set())
            for syn_id in list(outgoing_syn_ids):
                syn = graph.synapses.get(syn_id)
                if syn is None:
                    continue
                other_node = graph.nodes.get(syn.post_node_id)
                if other_node is None:
                    continue

                t_other = other_node.last_spike_time
                if t_other == -math.inf:
                    continue

                # From this synapse's perspective: pre (post_id) just fired,
                # and post (other_node) fired at t_other.
                # dt = t_other - t_post (post_node time relative to pre_node)
                dt = t_other - t_post

                if dt > 0:
                    # other fired after this node → LTP
                    raw_dw = self.A_plus * math.exp(-dt / self.tau_plus)
                    scale = (syn.max_weight - syn.weight) / syn.max_weight
                    dw = raw_dw * self.learning_rate * max(scale, 0.0)
                elif dt < 0:
                    # other fired before this node → LTD
                    raw_dw = -self.A_minus * math.exp(dt / self.tau_minus)
                    dw = raw_dw * self.learning_rate
                else:
                    continue  # already handled in incoming pass

                self._apply_dw(syn, dw, timestep, three_factor)


class HomeostaticRule(PlasticityRule):
    """Homeostatic plasticity (PRD §3.2).

    Maintains global stability without destroying learned structure.
    Operates on a slower timescale than STDP.

    Mechanisms (PRD §3.2.1):
        1. Synaptic Scaling (multiplicative): w_new = w_old × (target / actual)^factor
           Preserves relative weight ratios.  NOT normalization (PRD §3.2 note).
        2. Intrinsic Excitability: rate too low → increase; too high → decrease.
        3. Threshold Adaptation: drifts toward recent avg voltage at rate 0.001/step.
    """

    def __init__(
        self,
        target_firing_rate: float = 0.05,
        scaling_interval: int = 100,
        scaling_factor: float = 0.1,
        excitability_rate: float = 0.01,
        threshold_rate: float = 0.001,
        ema_alpha: float = 0.01,
    ):
        self.target_firing_rate = target_firing_rate
        self.scaling_interval = scaling_interval
        self.scaling_factor = scaling_factor
        self.excitability_rate = excitability_rate
        self.threshold_rate = threshold_rate
        self.ema_alpha = ema_alpha
        self._steps_since_scaling = 0

    def apply(
        self,
        graph: "Graph",
        fired_node_ids: List[str],
        timestep: int,
    ) -> None:
        fired_set = set(fired_node_ids)

        # Update firing rate EMA for every node
        for nid, node in graph.nodes.items():
            fired = 1.0 if nid in fired_set else 0.0
            node.firing_rate_ema = (
                (1.0 - self.ema_alpha) * node.firing_rate_ema
                + self.ema_alpha * fired
            )

        # Threshold adaptation: continuous, 0.001/step (PRD §3.2.1)
        for nid, node in graph.nodes.items():
            rate = node.firing_rate_ema
            if rate > self.target_firing_rate * 1.2:
                node.threshold += self.threshold_rate
            elif rate < self.target_firing_rate * 0.8:
                node.threshold = max(0.01, node.threshold - self.threshold_rate)

        self._steps_since_scaling += 1
        if self._steps_since_scaling < self.scaling_interval:
            return
        self._steps_since_scaling = 0

        # Synaptic scaling & intrinsic excitability (every N steps)
        for nid, node in graph.nodes.items():
            rate = node.firing_rate_ema
            if rate < 1e-9:
                # Silent node → boost excitability (PRD §3.1.2 Silent Death mitigation)
                node.intrinsic_excitability = min(
                    node.intrinsic_excitability * (1.0 + self.excitability_rate * 5),
                    5.0,
                )
                continue

            ratio = self.target_firing_rate / rate

            # Intrinsic excitability adjustment
            if ratio > 1.0:
                node.intrinsic_excitability = min(
                    node.intrinsic_excitability * (1.0 + self.excitability_rate),
                    5.0,
                )
            else:
                node.intrinsic_excitability = max(
                    node.intrinsic_excitability * (1.0 - self.excitability_rate),
                    0.1,
                )

            # Multiplicative synaptic scaling (PRD §3.2.1)
            # Scale incoming weights by (target/actual)^factor
            scale = ratio ** self.scaling_factor
            incoming_syn_ids = graph._incoming.get(nid, set())
            for syn_id in incoming_syn_ids:
                syn = graph.synapses.get(syn_id)
                if syn is None:
                    continue
                syn.weight = max(
                    0.0,
                    min(syn.weight * scale, syn.max_weight),
                )


class HyperedgePlasticityRule(PlasticityRule):
    """Hyperedge-level plasticity (PRD §4.3).

    When a hyperedge fires, adapt its internal structure:
        1. Member Weight Adaptation — consistently-active members during firing
           get higher weight; inactive members get lower weight.
        2. Threshold Learning — reward → lower threshold (more sensitive),
           punishment → raise threshold (more strict).
        3. Member Evolution — non-members that consistently co-fire with a
           hyperedge get added as new members with low initial weight.

    Operates on hyperedges with ``is_learnable=True``.
    """

    def __init__(
        self,
        member_weight_lr: float = 0.05,
        threshold_lr: float = 0.01,
        evolution_window: int = 50,
        evolution_min_co_fires: int = 10,
        evolution_initial_weight: float = 0.3,
    ):
        self.member_weight_lr = member_weight_lr
        self.threshold_lr = threshold_lr
        self.evolution_window = evolution_window
        self.evolution_min_co_fires = evolution_min_co_fires
        self.evolution_initial_weight = evolution_initial_weight

    def apply(
        self,
        graph: "Graph",
        fired_node_ids: List[str],
        timestep: int,
    ) -> None:
        """Adapt learnable hyperedges that fired this step."""
        fired_set = set(fired_node_ids)
        if not fired_set:
            return

        for hid, he in graph.hyperedges.items():
            if not he.is_learnable:
                continue
            # Only adapt hyperedges that actually fired this step
            if he.refractory_remaining != he.refractory_period:
                # Didn't just fire (refractory is set right after firing)
                continue

            # --- Member Weight Adaptation ---
            for nid in list(he.member_nodes):
                w = he.member_weights.get(nid, 1.0)
                if nid in fired_set:
                    # Active member during firing → strengthen
                    he.member_weights[nid] = min(w + self.member_weight_lr, 5.0)
                else:
                    # Inactive member during firing → weaken
                    he.member_weights[nid] = max(w - self.member_weight_lr * 0.5, 0.01)

            # --- Member Evolution: add co-firing non-members ---
            he_co_fire_counts = graph._he_co_fire_counts.get(hid)
            if he_co_fire_counts is not None:
                for nid in list(fired_set):
                    if nid in he.member_nodes:
                        continue
                    if nid not in graph.nodes:
                        continue
                    he_co_fire_counts[nid] = he_co_fire_counts.get(nid, 0) + 1
                    if he_co_fire_counts[nid] >= self.evolution_min_co_fires:
                        # Promote to member
                        he.member_nodes.add(nid)
                        he.member_weights[nid] = self.evolution_initial_weight
                        graph._node_hyperedges.setdefault(nid, set()).add(hid)
                        del he_co_fire_counts[nid]


# ---------------------------------------------------------------------------
# Graph Container (PRD §2.2.4, §8)
# ---------------------------------------------------------------------------

# Default configuration (PRD §9)
DEFAULT_CONFIG: Dict[str, Any] = {
    "decay_rate": 0.95,
    "default_threshold": 1.0,
    "refractory_period": 2,
    "tau_plus": 20.0,
    "tau_minus": 20.0,
    "A_plus": 1.0,
    "A_minus": 1.2,
    "learning_rate": 0.01,
    "max_weight": 5.0,
    "target_firing_rate": 0.05,
    "scaling_interval": 100,
    "weight_threshold": 0.01,
    "grace_period": 500,
    "inactivity_threshold": 1000,
    "co_activation_window": 5,
    "initial_sprouting_weight": 0.1,
    # Phase 3: Predictive Coding config
    "prediction_threshold": 3.0,         # Min synapse weight to generate prediction
    "prediction_pre_charge_factor": 0.3,  # Fraction of prediction strength for pre-charge
    "prediction_window": 10,             # Steps before prediction expires
    "prediction_chain_decay": 0.7,       # Strength decay per chain hop
    "prediction_max_chain_depth": 3,     # Max depth for prediction chains
    "prediction_confirm_bonus": 0.01,    # Weight bonus on confirmation (×confidence)
    "prediction_error_penalty": 0.02,    # Weight penalty on error (×confidence)
    "prediction_max_active": 1000,       # Max active predictions (memory limit)
    "surprise_sprouting_weight": 0.1,    # Initial weight for surprise-driven synapses
    "eligibility_trace_tau": 100,        # Decay time constant for eligibility traces
    "three_factor_enabled": False,       # Whether to use three-factor learning
    # Phase 2: Hypergraph Engine config
    "he_pattern_completion_strength": 0.3,
    "he_member_weight_lr": 0.05,
    "he_threshold_lr": 0.01,
    "he_discovery_window": 10,
    "he_discovery_min_co_fires": 5,
    "he_discovery_min_nodes": 3,
    "he_consolidation_overlap": 0.8,
    "he_member_evolution_window": 50,
    "he_member_evolution_min_co_fires": 10,
    "he_member_evolution_initial_weight": 0.3,
    # Phase 2.5: Prediction infrastructure
    "prediction_window": 5,
    "prediction_ema_alpha": 0.01,
    "he_experience_threshold": 100,
}


class Graph:
    """Manages all nodes, synapses, and hyperedges; orchestrates simulation,
    applies plasticity, and provides the public API (PRD §2.2.4, §8).

    Topology is fully sparse: dict-based adjacency indices, no dense matrices.

    Args:
        config: Override any key from ``DEFAULT_CONFIG``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # --- Core collections (sparse) ---
        self.nodes: Dict[str, Node] = {}
        self.synapses: Dict[str, Synapse] = {}
        self.hyperedges: Dict[str, Hyperedge] = {}

        # --- Sparse adjacency indices ---
        # node_id → set of synapse_ids
        self._outgoing: Dict[str, Set[str]] = {}
        self._incoming: Dict[str, Set[str]] = {}
        # node_id → set of hyperedge_ids the node belongs to
        self._node_hyperedges: Dict[str, Set[str]] = {}

        # --- Spike delay buffer: timestep → list of (target_node_id, current) ---
        self._delay_buffer: Dict[int, List[Tuple[str, float]]] = {}

        # --- Co-activation tracking for sprouting ---
        # Stores recent spike times per node for co-activation detection
        self._recent_spikes: Dict[str, Deque[int]] = {}

        # --- Phase 2: Hyperedge co-fire tracking for member evolution ---
        # hid → {node_id: co_fire_count}
        self._he_co_fire_counts: Dict[str, Dict[str, int]] = {}

        # --- Phase 2: Hyperedge discovery tracking ---
        # Tracks which sets of nodes fire together for automatic hyperedge creation
        # tuple(sorted node_ids) → fire_count
        self._he_discovery_counts: Dict[Tuple[str, ...], int] = {}
        self._he_discovery_last_reset: int = 0

        # --- Phase 2.5: Prediction tracking ---
        # prediction_id → PredictionState
        self._active_predictions: Dict[str, PredictionState] = {}
        self._prediction_counter: int = 0
        self._total_predictions: int = 0
        self._total_confirmed: int = 0
        self._total_surprised: int = 0
        # Nodes that fired during each prediction's window (rolling)
        self._prediction_window_fired: Dict[str, Set[str]] = {}
        # Archived hyperedges (for cross-level pruning)
        self._archived_hyperedges: Dict[str, Hyperedge] = {}

        # --- Plasticity rules ---
        self._plasticity_rules: List[PlasticityRule] = [
            STDPRule(
                tau_plus=self.config["tau_plus"],
                tau_minus=self.config["tau_minus"],
                A_plus=self.config["A_plus"],
                A_minus=self.config["A_minus"],
                learning_rate=self.config["learning_rate"],
            ),
            HomeostaticRule(
                target_firing_rate=self.config["target_firing_rate"],
                scaling_interval=self.config["scaling_interval"],
            ),
            HyperedgePlasticityRule(
                member_weight_lr=self.config["he_member_weight_lr"],
                threshold_lr=self.config["he_threshold_lr"],
                evolution_window=self.config["he_member_evolution_window"],
                evolution_min_co_fires=self.config["he_member_evolution_min_co_fires"],
                evolution_initial_weight=self.config["he_member_evolution_initial_weight"],
            ),
        ]

        # --- Phase 3: Predictive Coding state ---
        # Active predictions awaiting confirmation or expiry
        self.active_predictions: Dict[str, Prediction] = {}
        # Recent prediction outcomes for accuracy tracking
        self._prediction_outcomes: Deque[PredictionOutcome] = deque(maxlen=1000)
        # Per-synapse confirmation history: synapse_id → deque of bool
        self._synapse_confirmation_history: Dict[str, Deque[bool]] = {}
        # Nodes predicted this step (to avoid double-prediction)
        self._predicted_this_step: Set[str] = set()
        # Novel sequence log
        self._novel_sequence_log: List[Dict[str, Any]] = []
        # Reward history
        self._reward_history: List[Dict[str, Any]] = []
        # Prediction telemetry counters
        self._total_predictions_made = 0
        self._total_predictions_confirmed = 0
        self._total_predictions_errors = 0
        self._total_novel_sequences = 0
        self._total_rewards_injected = 0

        # --- Event handlers ---
        self._event_handlers: Dict[str, List[Callable]] = {}

        # --- Telemetry counters ---
        self._total_pruned = 0
        self._total_sprouted = 0
        self._total_he_discovered = 0
        self._total_he_consolidated = 0

        # --- Clock ---
        self.timestep: int = 0

        # --- Dirty flags for incremental checkpointing ---
        self._dirty_nodes: Set[str] = set()
        self._dirty_synapses: Set[str] = set()
        self._dirty_hyperedges: Set[str] = set()

        # --- Graph metadata with design changelog ---
        self.metadata: Dict[str, Any] = {
            "changelog": [
                {
                    "version": "0.1.0",
                    "description": "Phase 1 Core Foundation",
                    "notes": [
                        "STDP uses weight-dependent soft-saturation: LTP scaled by "
                        "(max_weight - w) / max_weight to prevent runaway potentiation "
                        "(PRD §3.1.2).",
                        "Spike history stored in fixed-capacity RingBuffer (default 100) "
                        "to bound memory per node while enabling firing-rate and burst "
                        "detection (PRD §2.2.1).",
                        "Homeostatic scaling is multiplicative (w * ratio^factor), NOT "
                        "divisive normalization, to preserve learned weight distributions "
                        "(PRD §3.2).",
                        "Hyperedges enforce a refractory period (default 2 steps) to "
                        "prevent cascading feedback loops from output-to-member cycles.",
                    ],
                },
                {
                    "version": "0.2.5",
                    "description": "Phase 2.5 Predictive Infrastructure",
                    "notes": [
                        "Prediction error events: hyperedge firings create predictions "
                        "for output targets. Confirmed if target fires within window, "
                        "SurpriseEvent emitted if window expires without firing.",
                        "Dynamic pattern completion: completion strength scales with "
                        "hyperedge experience (activation_count / threshold). New "
                        "hyperedges complete at 10% strength; experienced ones at 100%.",
                        "Cross-level consistency pruning: subsumption detection archives "
                        "redundant lower-level hyperedges when a higher-level one covers "
                        "identical members. Archived hyperedges preserved in metadata.",
                    ],
                },
                {
                    "version": "0.3.0",
                    "description": "Phase 3 Predictive Coding Engine",
                    "notes": [
                        "Synapse-level predictions: when a fired node has a strong "
                        "causal link (weight > prediction_threshold) to downstream "
                        "targets, a Prediction is registered and the target is "
                        "pre-charged by strength × 0.3 (PRD §5.1).",
                        "Prediction chains cascade through learned sequences (A→B→C) "
                        "with strength decaying ×0.7 per hop, up to max depth 3. "
                        "Cycle-safe via visited set passed through recursion.",
                        "Confirmed predictions strengthen the causal link (bonus × "
                        "confidence); expired predictions weaken it (penalty × "
                        "confidence) and trigger surprise-driven exploration that "
                        "sprouts speculative synapses to alternative firing nodes.",
                        "Three-factor learning: STDPRule._apply_dw routes weight "
                        "changes to eligibility_trace when three_factor_enabled=True. "
                        "Traces decay with τ=100 steps. Weight commits only on "
                        "inject_reward(strength, scope). Δw = trace × reward × lr.",
                        "Prediction confidence computed from synapse weight relative "
                        "to max_weight (60%) and historical confirmation rate (40%). "
                        "Higher confidence → stronger pre-charge and larger error "
                        "penalty.",
                        "Active predictions capped at 1000 with per-step cleanup of "
                        "expired entries. Outcomes stored in bounded deque (max 1000) "
                        "for accuracy tracking in Telemetry.",
                    ],
                },
                {
                    "version": "0.3.5",
                    "description": "Phase 3.5 Predictive State Persistence & Validation",
                    "notes": [
                        "Active predictions now survive checkpoint/restore. Both Phase 3 "
                        "synapse-level Predictions and Phase 2.5 HE-level PredictionStates "
                        "are serialized with all fields, preventing the system from "
                        "'forgetting' what it was expecting after reload.",
                        "Prediction support state persisted: PredictionOutcome history, "
                        "per-synapse confirmation history (deques), novel_sequence_log, "
                        "reward_history. These are needed for confidence calculations and "
                        "telemetry accuracy after restore.",
                        "Validation on restore: expired predictions dropped, predictions "
                        "referencing deleted nodes/hyperedges dropped, stale synapse "
                        "confirmation history entries dropped. Fail-fast prevents stale "
                        "predictions from corrupting confidence or emitting spurious events.",
                        "Backward compatible: v0.2.5 checkpoints restore cleanly with "
                        "empty prediction state. Checkpoint version bumped to 0.3.5.",
                    ],
                },
                {
                    "version": "0.4.0",
                    "description": "Phase 4 Universal Ingestor System",
                    "notes": [
                        "Five-stage ingestion pipeline: Extract → Chunk → Embed → "
                        "Register → Associate.  Converts raw data (text, markdown, "
                        "code, URLs, PDFs) into fully integrated NeuroGraph knowledge.",
                        "SimpleVectorDB: in-memory cosine-similarity search over "
                        "L2-normalized embeddings with content/metadata storage.",
                        "Novelty dampening: new ingested nodes start at reduced "
                        "intrinsic_excitability and boosted threshold, fading over "
                        "a probation period (linear/exponential/logarithmic curves) "
                        "to prevent destabilizing learned STDP pathways.",
                        "Three project configs: OpenClaw (code-aware, fast 0.3 "
                        "dampening), DSM (hierarchical, conservative 0.05), "
                        "Consciousness (semantic, exploratory 0.01).",
                    ],
                },
            ],
        }

    # -----------------------------------------------------------------------
    # Topology Management (PRD §2.2.4)
    # -----------------------------------------------------------------------

    def create_node(
        self,
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_inhibitory: bool = False,
    ) -> Node:
        """Register a node (PRD §8 create_node).

        Args:
            node_id: Optional explicit ID (auto-generated UUID if None).
            metadata: Application-specific key-value data.
            is_inhibitory: Whether outgoing spikes subtract from targets.

        Returns:
            The created Node.
        """
        nid = node_id or str(uuid.uuid4())
        if nid in self.nodes:
            raise ValueError(f"Node {nid} already exists")
        node = Node(
            node_id=nid,
            threshold=self.config["default_threshold"],
            refractory_period=self.config["refractory_period"],
            metadata=metadata or {},
            is_inhibitory=is_inhibitory,
        )
        self.nodes[nid] = node
        self._outgoing[nid] = set()
        self._incoming[nid] = set()
        self._node_hyperedges[nid] = set()
        self._recent_spikes[nid] = deque(maxlen=self.config["co_activation_window"] * 2)
        self._dirty_nodes.add(nid)
        return node

    def remove_node(self, node_id: str) -> None:
        """Remove node and all connected synapses; update hyperedges (PRD §8 remove_node)."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found")

        # Remove connected synapses (cascading deletion)
        syn_ids_to_remove = set()
        syn_ids_to_remove.update(self._outgoing.get(node_id, set()))
        syn_ids_to_remove.update(self._incoming.get(node_id, set()))
        for sid in syn_ids_to_remove:
            self._remove_synapse_internal(sid)

        # Remove from hyperedges
        for hid in list(self._node_hyperedges.get(node_id, set())):
            he = self.hyperedges.get(hid)
            if he:
                he.member_nodes.discard(node_id)
                he.member_weights.pop(node_id, None)
                if node_id in he.output_targets:
                    he.output_targets.remove(node_id)
                if len(he.member_nodes) == 0:
                    self._remove_hyperedge_internal(hid)
                else:
                    self._dirty_hyperedges.add(hid)

        # Clean up indices
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)
        self._node_hyperedges.pop(node_id, None)
        self._recent_spikes.pop(node_id, None)
        del self.nodes[node_id]
        self._dirty_nodes.discard(node_id)

    def create_synapse(
        self,
        pre_node_id: str,
        post_node_id: str,
        weight: float = 0.1,
        delay: int = 1,
        synapse_type: SynapseType = SynapseType.EXCITATORY,
        max_weight: Optional[float] = None,
    ) -> Synapse:
        """Create a directed synapse between two nodes.

        Args:
            pre_node_id: Source node ID.
            post_node_id: Target node ID.
            weight: Initial weight [0, max_weight].
            delay: Propagation delay in timesteps (≥1).
            synapse_type: EXCITATORY, INHIBITORY, or MODULATORY.
            max_weight: Per-synapse ceiling (defaults to config).

        Returns:
            The created Synapse.
        """
        if pre_node_id not in self.nodes:
            raise KeyError(f"Pre node {pre_node_id} not found")
        if post_node_id not in self.nodes:
            raise KeyError(f"Post node {post_node_id} not found")
        if pre_node_id == post_node_id:
            raise ValueError("Self-connections not allowed")

        mw = max_weight if max_weight is not None else self.config["max_weight"]
        syn = Synapse(
            pre_node_id=pre_node_id,
            post_node_id=post_node_id,
            weight=max(0.0, min(weight, mw)),
            max_weight=mw,
            delay=max(1, delay),
            synapse_type=synapse_type,
            creation_time=float(self.timestep),
            last_update_time=float(self.timestep),
            peak_weight=weight,
        )
        self.synapses[syn.synapse_id] = syn
        self._outgoing[pre_node_id].add(syn.synapse_id)
        self._incoming[post_node_id].add(syn.synapse_id)
        self._dirty_synapses.add(syn.synapse_id)
        return syn

    def _remove_synapse_internal(self, synapse_id: str) -> None:
        """Remove a synapse and clean up indices (no KeyError on missing)."""
        syn = self.synapses.pop(synapse_id, None)
        if syn is None:
            return
        self._outgoing.get(syn.pre_node_id, set()).discard(synapse_id)
        self._incoming.get(syn.post_node_id, set()).discard(synapse_id)
        self._dirty_synapses.discard(synapse_id)

    def remove_synapse(self, synapse_id: str) -> None:
        """Remove a synapse (public API)."""
        if synapse_id not in self.synapses:
            raise KeyError(f"Synapse {synapse_id} not found")
        self._remove_synapse_internal(synapse_id)

    def create_hyperedge(
        self,
        member_node_ids: Set[str],
        member_weights: Optional[Dict[str, float]] = None,
        activation_threshold: float = 0.6,
        activation_mode: ActivationMode = ActivationMode.WEIGHTED_THRESHOLD,
        output_targets: Optional[List[str]] = None,
        output_weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        is_learnable: bool = True,
    ) -> Hyperedge:
        """Create a hyperedge grouping multiple nodes (PRD §2.2.3)."""
        for nid in member_node_ids:
            if nid not in self.nodes:
                raise KeyError(f"Member node {nid} not found")

        mw = member_weights or {nid: 1.0 for nid in member_node_ids}
        he = Hyperedge(
            member_nodes=set(member_node_ids),
            member_weights=mw,
            activation_threshold=activation_threshold,
            activation_mode=activation_mode,
            output_targets=output_targets or [],
            output_weight=output_weight,
            metadata=metadata or {},
            is_learnable=is_learnable,
        )
        self.hyperedges[he.hyperedge_id] = he
        for nid in member_node_ids:
            self._node_hyperedges.setdefault(nid, set()).add(he.hyperedge_id)
        self._he_co_fire_counts[he.hyperedge_id] = {}
        self._dirty_hyperedges.add(he.hyperedge_id)
        return he

    def _remove_hyperedge_internal(self, hyperedge_id: str) -> None:
        he = self.hyperedges.pop(hyperedge_id, None)
        if he is None:
            return
        for nid in he.member_nodes:
            self._node_hyperedges.get(nid, set()).discard(hyperedge_id)
        self._he_co_fire_counts.pop(hyperedge_id, None)
        self._dirty_hyperedges.discard(hyperedge_id)

    def remove_hyperedge(self, hyperedge_id: str) -> None:
        if hyperedge_id not in self.hyperedges:
            raise KeyError(f"Hyperedge {hyperedge_id} not found")
        self._remove_hyperedge_internal(hyperedge_id)

    # -----------------------------------------------------------------------
    # Stimulation (PRD §8 stimulate / stimulate_batch)
    # -----------------------------------------------------------------------

    def stimulate(self, node_id: str, current: float) -> None:
        """Inject input current into a node (PRD §8 stimulate)."""
        node = self.nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node {node_id} not found")
        node.voltage += current * node.intrinsic_excitability

    def stimulate_batch(self, stimuli: List[Tuple[str, float]]) -> None:
        """Batch stimulus for search results (PRD §8 stimulate_batch)."""
        for node_id, current in stimuli:
            self.stimulate(node_id, current)

    # -----------------------------------------------------------------------
    # Simulation Loop (PRD §2.2.4, §8 step)
    # -----------------------------------------------------------------------

    def step(self) -> StepResult:
        """Advance one timestep (PRD §2.2.4 Simulation Loop, §8 step).

        Pipeline:
            1. Decay voltages toward resting potential
            2. Deliver delayed spikes from buffer
            3. Detect fired nodes (voltage ≥ threshold, not refractory)
            4. Reset fired node voltages; set refractory
            5. Propagate spikes through outgoing synapses (with delays)
            6. Evaluate hyperedges
            6b. Evaluate predictions (confirm/error) (PRD §5.1)
            6c. Generate new predictions from fired nodes (PRD §5.1)
            6d. Decay eligibility traces (PRD §5.2)
            7. Apply plasticity rules
            8. Structural plasticity (prune / sprout)
            9. Decrement refractory counters
            10. Record telemetry / emit events

        Returns:
            StepResult with fired nodes, hyperedges, pruning/sprouting counts.
        """
        self.timestep += 1
        result = StepResult(timestep=self.timestep)

        # 1. Voltage decay: v = v * decay_rate + (1-decay) * resting  (PRD §2.2.4)
        decay = self.config["decay_rate"]
        for node in self.nodes.values():
            node.voltage = node.voltage * decay + (1.0 - decay) * node.resting_potential

        # 2. Deliver delayed spikes arriving this timestep
        arrivals = self._delay_buffer.pop(self.timestep, [])
        for target_id, current in arrivals:
            target = self.nodes.get(target_id)
            if target is not None:
                target.voltage += current * target.intrinsic_excitability

        # 3. Detect firing nodes
        fired_ids: List[str] = []
        for nid, node in self.nodes.items():
            if node.refractory_remaining > 0:
                continue
            if node.voltage >= node.threshold:
                fired_ids.append(nid)

        # 4. Reset fired nodes and set refractory
        for nid in fired_ids:
            node = self.nodes[nid]
            node.voltage = node.resting_potential
            node.refractory_remaining = node.refractory_period
            node.last_spike_time = float(self.timestep)
            node.spike_history.append(float(self.timestep))
            # Track recent spikes for sprouting
            self._recent_spikes.setdefault(nid, deque(maxlen=20)).append(self.timestep)

        result.fired_node_ids = fired_ids

        # 5. Propagate spikes through outgoing synapses (with delay)
        for nid in fired_ids:
            node = self.nodes[nid]
            sign = -1.0 if node.is_inhibitory else 1.0
            for syn_id in self._outgoing.get(nid, set()):
                syn = self.synapses.get(syn_id)
                if syn is None:
                    continue
                # Effective current is weight × sign
                effective_type_sign = sign
                if syn.synapse_type == SynapseType.INHIBITORY:
                    effective_type_sign = -1.0
                current = syn.weight * effective_type_sign
                arrival = self.timestep + syn.delay
                self._delay_buffer.setdefault(arrival, []).append(
                    (syn.post_node_id, current)
                )
                syn.inactive_steps = 0  # Reset inactivity

        # 6. Evaluate hyperedges (PRD §4.2) — with dynamic pattern completion
        fired_set = set(fired_ids)
        fired_he_this_step: List[str] = []
        experience_threshold = self.config["he_experience_threshold"]
        ema_alpha = self.config["prediction_ema_alpha"]
        # Process by level so child hyperedges fire before parents
        max_level = max((he.level for he in self.hyperedges.values()), default=0)
        for level in range(max_level + 1):
            for hid, he in self.hyperedges.items():
                if he.level != level:
                    continue
                if he.is_archived:
                    continue  # Archived hyperedges don't participate
                activation = self._compute_hyperedge_activation(he, fired_set)
                he.current_activation = activation
                # Update activation EMA (Phase 2.5)
                fired_flag = 1.0 if (activation >= he.activation_threshold and he.refractory_remaining == 0) else 0.0
                he.recent_activation_ema = (
                    (1.0 - ema_alpha) * he.recent_activation_ema
                    + ema_alpha * fired_flag
                )
                if he.refractory_remaining > 0:
                    continue  # Still in refractory — cannot fire
                if activation >= he.activation_threshold:
                    result.fired_hyperedge_ids.append(hid)
                    fired_he_this_step.append(hid)
                    he.activation_count += 1
                    he.refractory_remaining = he.refractory_period

                    # Output injection — GRADED mode scales by activation level
                    effective_weight = he.output_weight
                    if he.activation_mode == ActivationMode.GRADED:
                        effective_weight *= activation
                    for target_id in he.output_targets:
                        target = self.nodes.get(target_id)
                        if target is not None:
                            target.voltage += effective_weight * target.intrinsic_excitability

                    # Dynamic pattern completion (Phase 2.5):
                    # Scale by experience: new HEs complete weakly, experienced ones fully.
                    # completion_strength = base × min(1.0, activation_count / threshold)
                    if he.pattern_completion_strength > 0:
                        learning_factor = min(1.0, he.activation_count / max(experience_threshold, 1))
                        effective_completion = he.pattern_completion_strength * learning_factor
                        if effective_completion > 0:
                            for nid in he.member_nodes:
                                if nid not in fired_set:
                                    node = self.nodes.get(nid)
                                    if node is not None and node.refractory_remaining == 0:
                                        node.voltage += (
                                            effective_completion
                                            * he.member_weights.get(nid, 1.0)
                                            * node.intrinsic_excitability
                                        )

                    # Prediction creation (Phase 2.5): predict output targets will fire
                    if he.output_targets:
                        pred_id = f"pred_{self._prediction_counter}"
                        self._prediction_counter += 1
                        pred = PredictionState(
                            hyperedge_id=hid,
                            predicted_targets=set(he.output_targets),
                            prediction_strength=activation,
                            prediction_timestamp=self.timestep,
                            prediction_window=self.config["prediction_window"],
                        )
                        self._active_predictions[pred_id] = pred
                        self._prediction_window_fired[pred_id] = set()
                        self._total_predictions += 1
                        self._emit("prediction_created", prediction_id=pred_id,
                                   hyperedge_id=hid, targets=list(he.output_targets))

                    self._emit("hyperedge_fired", hid=hid, activation=activation)

        # Decrement hyperedge refractory counters (skip those that just fired)
        fired_he_set = set(fired_he_this_step)
        for hid, he in self.hyperedges.items():
            if he.refractory_remaining > 0 and hid not in fired_he_set:
                he.refractory_remaining -= 1

        # 6b. Evaluate Phase 2.5 hyperedge-level predictions
        he_confirmed_this_step = 0
        he_surprised_this_step = 0
        he_preds_to_remove: List[str] = []
        for pid, pred_state in self._active_predictions.items():
            # Track nodes that fired during this prediction's window
            window_fired = self._prediction_window_fired.get(pid, set())
            window_fired.update(fired_set)
            self._prediction_window_fired[pid] = window_fired

            # Check for newly confirmed targets
            for target in pred_state.predicted_targets - pred_state.confirmed_targets:
                if target in fired_set:
                    pred_state.confirmed_targets.add(target)

            # Check if all targets confirmed
            if pred_state.confirmed_targets >= pred_state.predicted_targets:
                he_confirmed_this_step += 1
                self._total_confirmed += 1
                he_preds_to_remove.append(pid)
                target_list = list(pred_state.predicted_targets)
                self._emit(
                    "prediction_confirmed",
                    prediction_id=pid,
                    hyperedge_id=pred_state.hyperedge_id,
                    targets=target_list,
                    target_node=target_list[0] if target_list else None,
                    timestep=self.timestep,
                )
            # Check if window expired
            elif self.timestep - pred_state.prediction_timestamp >= pred_state.prediction_window:
                he_surprised_this_step += 1
                self._total_surprised += 1
                he_preds_to_remove.append(pid)
                for expected in pred_state.predicted_targets - pred_state.confirmed_targets:
                    surprise = SurpriseEvent(
                        hyperedge_id=pred_state.hyperedge_id,
                        expected_node=expected,
                        prediction_strength=pred_state.prediction_strength,
                        actual_nodes=window_fired,
                        timestamp=self.timestep,
                    )
                    self._emit(
                        "surprise",
                        surprise=surprise,
                        timestep=self.timestep,
                    )
        for pid in he_preds_to_remove:
            self._active_predictions.pop(pid, None)
            self._prediction_window_fired.pop(pid, None)
        result.predictions_confirmed = he_confirmed_this_step
        result.predictions_surprised = he_surprised_this_step

        # 6c. Evaluate Phase 3 synapse-level predictions (PRD §5.1)
        self._predicted_this_step.clear()
        self._evaluate_predictions(fired_set)

        # 6d. Generate new predictions from fired nodes (PRD §5.1)
        self._generate_predictions(fired_ids)

        # 6e. Decay eligibility traces (PRD §5.2)
        # Only process synapses with non-zero traces for performance
        if self.config.get("three_factor_enabled", False):
            trace_tau = self.config["eligibility_trace_tau"]
            trace_decay = math.exp(-1.0 / trace_tau) if trace_tau > 0 else 0.0
            for syn in self.synapses.values():
                if abs(syn.eligibility_trace) > 1e-12:
                    syn.eligibility_trace *= trace_decay

        # 6f. Clean up expired Phase 3 predictions
        self._cleanup_predictions()

        # 7. Apply plasticity rules
        if fired_ids:
            for rule in self._plasticity_rules:
                rule.apply(self, fired_ids, self.timestep)

        # 8. Structural plasticity
        pruned, sprouted = self._structural_plasticity(fired_ids)
        result.synapses_pruned = pruned
        result.synapses_sprouted = sprouted
        self._total_pruned += pruned
        self._total_sprouted += sprouted

        # 9. Decrement refractory counters
        #    Skip nodes that just fired this step — their full refractory
        #    period starts on the NEXT step (PRD §3.2.1: mandatory N-step rest).
        fired_this_step = set(fired_ids)
        for nid, node in self.nodes.items():
            if node.refractory_remaining > 0 and nid not in fired_this_step:
                node.refractory_remaining -= 1

        # 10. Track synapse inactivity
        for syn in self.synapses.values():
            syn.inactive_steps += 1

        # Emit spike events
        if fired_ids:
            self._emit("spikes", node_ids=fired_ids, timestep=self.timestep)

        return result

    def step_n(self, n: int) -> List[StepResult]:
        """Run n steps; returns all StepResults (PRD §8 step_n)."""
        results = []
        for _ in range(n):
            results.append(self.step())
        return results

    # -----------------------------------------------------------------------
    # Auto-Knowledge: Spreading Activation Harvest
    # -----------------------------------------------------------------------

    def prime_and_propagate(
        self,
        node_ids: List[str],
        currents: List[float],
        steps: int = 3,
    ) -> PropagationResult:
        """Prime nodes and propagate activation through the network.

        This is the SNN-level primitive for associative recall.  It runs a
        *read-mostly* simulation: activation dynamics (voltage decay, spike
        propagation, hyperedge evaluation, pattern completion, prediction
        pre-charging) are active, but plasticity rules and structural changes
        are NOT applied.  This prevents recall-driven activation from
        altering learned weights.

        Args:
            node_ids: Nodes to inject current into (semantic priming).
            currents: Current to inject into each node (parallel to node_ids).
            steps: Number of SNN steps to propagate.

        Returns:
            PropagationResult with all nodes that fired, ranked by latency.
        """
        if not node_ids:
            return PropagationResult(steps_run=steps, nodes_primed=0)

        # Save and restore state so propagation is non-destructive
        saved_voltages: Dict[str, float] = {}
        saved_refractory: Dict[str, int] = {}
        for nid, node in self.nodes.items():
            saved_voltages[nid] = node.voltage
            saved_refractory[nid] = node.refractory_remaining

        saved_he_refractory: Dict[str, int] = {}
        for hid, he in self.hyperedges.items():
            saved_he_refractory[hid] = he.refractory_remaining

        # Compute approximate distances from primed nodes
        primed_set = set(node_ids)
        distances: Dict[str, int] = {nid: 0 for nid in node_ids}
        # BFS to compute distances
        frontier = set(node_ids)
        for dist in range(1, steps + 1):
            next_frontier: Set[str] = set()
            for nid in frontier:
                for syn_id in self._outgoing.get(nid, set()):
                    syn = self.synapses.get(syn_id)
                    if syn and syn.post_node_id not in distances:
                        distances[syn.post_node_id] = dist
                        next_frontier.add(syn.post_node_id)
            frontier = next_frontier

        # Collect current prediction targets for was_predicted tagging
        predicted_targets: Set[str] = set()
        for pred in self.active_predictions.values():
            predicted_targets.add(pred.target_node_id)
        for pred_state in self._active_predictions.values():
            predicted_targets.update(pred_state.predicted_targets)

        # --- PRIME: inject current into specified nodes ---
        for nid, current in zip(node_ids, currents):
            node = self.nodes.get(nid)
            if node is not None:
                node.voltage += current * node.intrinsic_excitability

        # --- PROPAGATE: run N read-only SNN steps ---
        result = PropagationResult(
            steps_run=steps,
            nodes_primed=len(node_ids),
        )
        # Local delay buffer separate from the graph's real one
        prop_delay_buffer: Dict[int, List[Tuple[str, float]]] = {}
        prop_timestep = self.timestep

        decay = self.config["decay_rate"]
        experience_threshold = self.config["he_experience_threshold"]

        for step_idx in range(steps):
            prop_timestep += 1

            # 1. Voltage decay
            for node in self.nodes.values():
                node.voltage = node.voltage * decay + (1.0 - decay) * node.resting_potential

            # 2. Deliver delayed spikes from propagation buffer
            arrivals = prop_delay_buffer.pop(prop_timestep, [])
            for target_id, current in arrivals:
                target = self.nodes.get(target_id)
                if target is not None:
                    target.voltage += current * target.intrinsic_excitability

            # 3. Detect firing nodes
            fired_ids: List[str] = []
            for nid, node in self.nodes.items():
                if node.refractory_remaining > 0:
                    continue
                if node.voltage >= node.threshold:
                    fired_ids.append(nid)

            # 4. Reset fired nodes and set refractory
            for nid in fired_ids:
                node = self.nodes[nid]
                voltage_at_fire = node.voltage
                node.voltage = node.resting_potential
                node.refractory_remaining = node.refractory_period

                entry = FiredEntry(
                    node_id=nid,
                    firing_step=step_idx,
                    voltage_at_fire=voltage_at_fire,
                    was_predicted=nid in predicted_targets,
                    source_distance=distances.get(nid, steps + 1),
                )
                result.fired_entries.append(entry)

            # 5. Propagate spikes through outgoing synapses
            fired_set = set(fired_ids)
            for nid in fired_ids:
                node = self.nodes[nid]
                sign = -1.0 if node.is_inhibitory else 1.0
                for syn_id in self._outgoing.get(nid, set()):
                    syn = self.synapses.get(syn_id)
                    if syn is None:
                        continue
                    effective_type_sign = sign
                    if syn.synapse_type == SynapseType.INHIBITORY:
                        effective_type_sign = -1.0
                    current = syn.weight * effective_type_sign
                    arrival = prop_timestep + syn.delay
                    prop_delay_buffer.setdefault(arrival, []).append(
                        (syn.post_node_id, current)
                    )

            # 6. Evaluate hyperedges (pattern completion, output injection)
            max_level = max((he.level for he in self.hyperedges.values()), default=0)
            for level in range(max_level + 1):
                for hid, he in self.hyperedges.items():
                    if he.level != level or he.is_archived:
                        continue
                    activation = self._compute_hyperedge_activation(he, fired_set)
                    if he.refractory_remaining > 0:
                        continue
                    if activation >= he.activation_threshold:
                        he.refractory_remaining = he.refractory_period

                        # Output injection
                        effective_weight = he.output_weight
                        if he.activation_mode == ActivationMode.GRADED:
                            effective_weight *= activation
                        for target_id in he.output_targets:
                            target = self.nodes.get(target_id)
                            if target is not None:
                                target.voltage += effective_weight * target.intrinsic_excitability

                        # Pattern completion (pre-charge inactive members)
                        if he.pattern_completion_strength > 0:
                            learning_factor = min(
                                1.0, he.activation_count / max(experience_threshold, 1)
                            )
                            eff_completion = he.pattern_completion_strength * learning_factor
                            if eff_completion > 0:
                                for mnid in he.member_nodes:
                                    if mnid not in fired_set:
                                        mnode = self.nodes.get(mnid)
                                        if mnode and mnode.refractory_remaining == 0:
                                            mnode.voltage += (
                                                eff_completion
                                                * he.member_weights.get(mnid, 1.0)
                                                * mnode.intrinsic_excitability
                                            )

            # Decrement refractory counters (skip those that just fired)
            for nid, node in self.nodes.items():
                if node.refractory_remaining > 0 and nid not in fired_set:
                    node.refractory_remaining -= 1
            for hid, he in self.hyperedges.items():
                if he.refractory_remaining > 0 and hid not in set(fired_ids):
                    he.refractory_remaining -= 1

        # --- RESTORE: put all voltages and refractory counters back ---
        for nid, node in self.nodes.items():
            node.voltage = saved_voltages.get(nid, node.resting_potential)
            node.refractory_remaining = saved_refractory.get(nid, 0)
        for hid, he in self.hyperedges.items():
            he.refractory_remaining = saved_he_refractory.get(hid, 0)

        return result

    # -----------------------------------------------------------------------
    # Hyperedge Activation (PRD §4.2)
    # -----------------------------------------------------------------------

    def _compute_hyperedge_activation(
        self, he: Hyperedge, fired_set: Set[str]
    ) -> float:
        """Compute hyperedge activation level (PRD §4.2).

        WEIGHTED_THRESHOLD mode:
            activation = Σ(weight_i × is_active_i) / Σ(weight_i)
        """
        if not he.member_nodes:
            return 0.0

        if he.activation_mode == ActivationMode.WEIGHTED_THRESHOLD:
            total_w = sum(he.member_weights.get(nid, 1.0) for nid in he.member_nodes)
            if total_w == 0:
                return 0.0
            active_w = sum(
                he.member_weights.get(nid, 1.0)
                for nid in he.member_nodes
                if nid in fired_set
            )
            return active_w / total_w

        elif he.activation_mode == ActivationMode.K_OF_N:
            k = int(he.activation_threshold * len(he.member_nodes))
            active = sum(1 for nid in he.member_nodes if nid in fired_set)
            return 1.0 if active >= max(k, 1) else active / max(k, 1)

        elif he.activation_mode == ActivationMode.ALL_OR_NONE:
            all_active = all(nid in fired_set for nid in he.member_nodes)
            return 1.0 if all_active else 0.0

        elif he.activation_mode == ActivationMode.GRADED:
            active = sum(1 for nid in he.member_nodes if nid in fired_set)
            return active / len(he.member_nodes)

        return 0.0

    # -----------------------------------------------------------------------
    # Phase 3: Predictive Coding Engine (PRD §5)
    # -----------------------------------------------------------------------

    def _compute_prediction_confidence(
        self, synapse: Synapse
    ) -> float:
        """Compute prediction confidence for a synapse (PRD §5.1).

        Confidence is based on:
            1. Synapse weight relative to max_weight (strength of causal link)
            2. Confirmation history (how often predictions were correct)

        Returns:
            Confidence in [0.0, 1.0].
        """
        # Factor 1: weight strength
        weight_factor = synapse.weight / synapse.max_weight

        # Factor 2: confirmation history
        history = self._synapse_confirmation_history.get(synapse.synapse_id)
        if history and len(history) > 0:
            confirmation_rate = sum(1 for x in history if x) / len(history)
        else:
            confirmation_rate = 0.5  # Neutral prior

        return min(1.0, weight_factor * 0.6 + confirmation_rate * 0.4)

    def _generate_predictions(self, fired_ids: List[str]) -> None:
        """Generate predictions from fired nodes (PRD §5.1).

        When node A fires with a strong causal link to B (weight > threshold):
            - Pre-charge B's voltage (prediction strength × factor)
            - Register prediction in active_predictions
            - Cascade through learned chains with decaying strength

        Avoids double-predicting the same target from both synapse and
        hyperedge sources.
        """
        if not fired_ids:
            return

        threshold = self.config["prediction_threshold"]
        pre_charge_factor = self.config["prediction_pre_charge_factor"]
        window = self.config["prediction_window"]
        chain_decay = self.config["prediction_chain_decay"]
        max_depth = self.config["prediction_max_chain_depth"]
        max_active = self.config["prediction_max_active"]

        # Generate predictions for each fired node
        for nid in fired_ids:
            if len(self.active_predictions) >= max_active:
                break
            self._generate_predictions_from_node(
                source_id=nid,
                origin_id=nid,
                strength=1.0,
                chain_depth=0,
                threshold=threshold,
                pre_charge_factor=pre_charge_factor,
                window=window,
                chain_decay=chain_decay,
                max_depth=max_depth,
                max_active=max_active,
                visited=set(),
            )

    def _generate_predictions_from_node(
        self,
        source_id: str,
        origin_id: str,
        strength: float,
        chain_depth: int,
        threshold: float,
        pre_charge_factor: float,
        window: int,
        chain_decay: float,
        max_depth: int,
        max_active: int,
        visited: Set[str],
    ) -> None:
        """Recursively generate predictions through causal chains.

        Args:
            source_id: Node whose outgoing synapses we're examining.
            origin_id: Original node that started the prediction chain.
            strength: Current prediction strength (decays with depth).
            chain_depth: Current depth in the prediction chain.
            threshold: Min weight to trigger prediction.
            pre_charge_factor: Voltage pre-charge fraction.
            window: Prediction expiry window in timesteps.
            chain_decay: Strength multiplier per chain hop.
            max_depth: Maximum chain depth.
            max_active: Maximum active predictions allowed.
            visited: Nodes already visited in this chain (cycle prevention).
        """
        if chain_depth > max_depth:
            return
        if source_id in visited:
            return
        visited.add(source_id)

        for syn_id in self._outgoing.get(source_id, set()):
            if len(self.active_predictions) >= max_active:
                return

            syn = self.synapses.get(syn_id)
            if syn is None:
                continue

            # Only predict for strong causal links
            effective_weight = syn.weight
            if chain_depth == 0 and effective_weight < threshold:
                continue

            target_id = syn.post_node_id
            target = self.nodes.get(target_id)
            if target is None:
                continue

            # Skip if already predicted this step (avoid double-prediction)
            if target_id in self._predicted_this_step:
                continue

            # Calculate prediction strength for this hop
            pred_strength = strength * (effective_weight / syn.max_weight)
            if chain_depth > 0:
                pred_strength *= chain_decay

            # Skip very weak predictions
            if pred_strength < 0.01:
                continue

            confidence = self._compute_prediction_confidence(syn)

            # Pre-charge target voltage
            pre_charge = pred_strength * pre_charge_factor
            if target.refractory_remaining == 0:
                target.voltage += pre_charge * target.intrinsic_excitability

            # Register prediction
            pred = Prediction(
                source_node_id=origin_id,
                target_node_id=target_id,
                strength=pred_strength,
                confidence=confidence,
                created_at=self.timestep,
                expires_at=self.timestep + window,
                chain_depth=chain_depth,
                pre_charge_applied=pre_charge,
            )
            self.active_predictions[pred.prediction_id] = pred
            self._predicted_this_step.add(target_id)
            self._total_predictions_made += 1

            self._emit(
                "prediction_generated",
                source=origin_id,
                target=target_id,
                strength=pred_strength,
                confidence=confidence,
                chain_depth=chain_depth,
                timestep=self.timestep,
            )

            # Cascade predictions through chains
            if chain_depth < max_depth and effective_weight >= threshold:
                self._generate_predictions_from_node(
                    source_id=target_id,
                    origin_id=origin_id,
                    strength=pred_strength,
                    chain_depth=chain_depth + 1,
                    threshold=threshold,
                    pre_charge_factor=pre_charge_factor,
                    window=window,
                    chain_decay=chain_decay,
                    max_depth=max_depth,
                    max_active=max_active,
                    visited=visited,
                )

    def _evaluate_predictions(self, fired_set: Set[str]) -> None:
        """Check active predictions against fired nodes (PRD §5.1).

        Confirmed predictions strengthen the causal link.
        Failed predictions (expired) weaken the link and trigger surprise.
        """
        confirm_bonus = self.config["prediction_confirm_bonus"]
        error_penalty = self.config["prediction_error_penalty"]

        to_remove: List[str] = []

        for pid, pred in list(self.active_predictions.items()):
            if pred.target_node_id in fired_set:
                # Prediction confirmed
                self._on_prediction_confirmed(pred, confirm_bonus)
                to_remove.append(pid)

        for pid in to_remove:
            self.active_predictions.pop(pid, None)

    def _on_prediction_confirmed(
        self, pred: Prediction, confirm_bonus: float
    ) -> None:
        """Handle a confirmed prediction (PRD §5.1).

        Effects:
            - Strengthen the causal link (weight += bonus × confidence)
            - Record confirmation in synapse history
            - Emit PredictionConfirmed event
        """
        # Find the synapse from source to target
        syn = self._find_synapse(pred.source_node_id, pred.target_node_id)
        if syn is not None:
            bonus = confirm_bonus * pred.confidence
            if not self.config["three_factor_enabled"]:
                syn.weight = min(syn.weight + bonus, syn.max_weight)
            else:
                # In three-factor mode, add to eligibility trace
                syn.eligibility_trace += bonus
            syn.last_update_time = float(self.timestep)
            if syn.weight > syn.peak_weight:
                syn.peak_weight = syn.weight

            # Update confirmation history
            history = self._synapse_confirmation_history.setdefault(
                syn.synapse_id, deque(maxlen=100)
            )
            history.append(True)

        self._total_predictions_confirmed += 1

        outcome = PredictionOutcome(
            prediction=pred,
            confirmed=True,
            resolved_at=self.timestep,
            actual_firing_nodes=[pred.target_node_id],
        )
        self._prediction_outcomes.append(outcome)

        self._emit(
            "prediction_confirmed",
            source=pred.source_node_id,
            target=pred.target_node_id,
            strength=pred.strength,
            confidence=pred.confidence,
            delay=self.timestep - pred.created_at,
            timestep=self.timestep,
        )

    def _on_prediction_error(
        self, pred: Prediction, error_penalty: float, recent_fired: Set[str]
    ) -> None:
        """Handle a prediction error / surprise (PRD §5.1, §5.2).

        Effects:
            - Weaken the causal link (weight -= penalty × confidence)
            - Emit SurpriseEvent
            - Trigger surprise-driven exploration
        """
        syn = self._find_synapse(pred.source_node_id, pred.target_node_id)
        if syn is not None:
            penalty = error_penalty * pred.confidence
            if not self.config["three_factor_enabled"]:
                syn.weight = max(0.0, syn.weight - penalty)
            else:
                syn.eligibility_trace -= penalty
            syn.last_update_time = float(self.timestep)

            # Update confirmation history
            history = self._synapse_confirmation_history.setdefault(
                syn.synapse_id, deque(maxlen=100)
            )
            history.append(False)

        self._total_predictions_errors += 1

        outcome = PredictionOutcome(
            prediction=pred,
            confirmed=False,
            resolved_at=self.timestep,
            actual_firing_nodes=list(recent_fired),
        )
        self._prediction_outcomes.append(outcome)

        self._emit(
            "prediction_error",
            source=pred.source_node_id,
            expected_target=pred.target_node_id,
            strength=pred.strength,
            confidence=pred.confidence,
            actual_fired=list(recent_fired),
            timestep=self.timestep,
        )

        # Surprise-driven exploration (PRD §5.2)
        self._surprise_exploration(pred, recent_fired)

    def _surprise_exploration(
        self, pred: Prediction, recent_fired: Set[str]
    ) -> None:
        """Explore alternative pathways after prediction error (PRD §5.2).

        When A→B prediction fails, examine what DID fire instead.
        If node C fired when B was expected:
            - Create speculative synapse A→C (weight 0.1)
            - Tag as "surprise-driven" in metadata

        Also check for novel sequences (no learned patterns).
        """
        source_id = pred.source_node_id
        expected_id = pred.target_node_id
        sprout_weight = self.config["surprise_sprouting_weight"]

        # Find what fired instead of the expected target
        alternative_nodes = recent_fired - {expected_id}

        sprouted_count = 0
        for alt_id in alternative_nodes:
            if alt_id == source_id:
                continue
            if alt_id not in self.nodes:
                continue

            # Check if synapse already exists
            existing = self._find_synapse(source_id, alt_id)
            if existing is not None:
                continue

            # Create speculative synapse
            try:
                syn = self.create_synapse(
                    source_id, alt_id, weight=sprout_weight
                )
                syn.metadata = {"creation_mode": "surprise_driven",
                                "expected_target": expected_id,
                                "timestep": self.timestep}
                sprouted_count += 1
                self._total_sprouted += 1
            except (KeyError, ValueError):
                continue

        if sprouted_count > 0:
            self._emit(
                "surprise_sprouted",
                source=source_id,
                expected=expected_id,
                alternatives=list(alternative_nodes),
                count=sprouted_count,
                timestep=self.timestep,
            )

        # Novelty detection: check if any fired sequence has NO learned patterns
        if alternative_nodes and not self._has_learned_pattern(source_id, alternative_nodes):
            self._total_novel_sequences += 1
            novel_event = {
                "source": source_id,
                "firing_nodes": list(alternative_nodes),
                "timestep": self.timestep,
            }
            self._novel_sequence_log.append(novel_event)
            # Cap log size
            if len(self._novel_sequence_log) > 1000:
                self._novel_sequence_log = self._novel_sequence_log[-500:]

            self._emit(
                "novel_sequence",
                source=source_id,
                firing_nodes=list(alternative_nodes),
                timestep=self.timestep,
            )

    def _has_learned_pattern(
        self, source_id: str, targets: Set[str]
    ) -> bool:
        """Check if any of the targets have a learned connection from source.

        A "learned pattern" means there exists a synapse with weight above
        the prediction threshold.
        """
        threshold = self.config["prediction_threshold"]
        for syn_id in self._outgoing.get(source_id, set()):
            syn = self.synapses.get(syn_id)
            if syn and syn.post_node_id in targets and syn.weight >= threshold:
                return True
        return False

    def _cleanup_predictions(self) -> None:
        """Remove expired predictions and trigger error handling (PRD §5.1).

        Called each step to clean up predictions that have exceeded their
        window without confirmation.
        """
        error_penalty = self.config["prediction_error_penalty"]

        # Collect recently fired nodes for surprise analysis
        recent_window = self.config["prediction_window"]
        recent_fired: Set[str] = set()
        for nid, spikes in self._recent_spikes.items():
            for t in spikes:
                if self.timestep - t <= recent_window:
                    recent_fired.add(nid)
                    break

        expired: List[str] = []
        for pid, pred in self.active_predictions.items():
            if self.timestep > pred.expires_at:
                expired.append(pid)

        for pid in expired:
            pred = self.active_predictions.pop(pid)
            self._on_prediction_error(pred, error_penalty, recent_fired)

    def _find_synapse(
        self, pre_id: str, post_id: str
    ) -> Optional[Synapse]:
        """Find a synapse connecting pre_id → post_id, if one exists."""
        for syn_id in self._outgoing.get(pre_id, set()):
            syn = self.synapses.get(syn_id)
            if syn and syn.post_node_id == post_id:
                return syn
        return None

    # -----------------------------------------------------------------------
    # Phase 3: Public Prediction API (PRD §8)
    # -----------------------------------------------------------------------

    def get_predictions(self) -> List[Prediction]:
        """Current active predictions (PRD §8 get_predictions).

        Returns pre-charged unfired nodes that the system expects to fire.
        """
        return list(self.active_predictions.values())

    # -----------------------------------------------------------------------
    # Structural Plasticity (PRD §3.3)
    # -----------------------------------------------------------------------

    def _structural_plasticity(self, fired_ids: List[str]) -> Tuple[int, int]:
        """Apply pruning and sprouting rules (PRD §3.3).

        Returns:
            (num_pruned, num_sprouted)
        """
        pruned = self._prune_synapses()
        sprouted = self._sprout_synapses(fired_ids)
        return pruned, sprouted

    def _prune_synapses(self) -> int:
        """Prune weak/inactive synapses (PRD §3.3.1).

        Rules:
            Weight-based: weight < threshold for > grace_period steps → remove.
            Activity-based: unused for > inactivity_threshold steps → remove.
            Age-based: age > grace_period AND peak_weight < 2× initial → remove.
        """
        wt = self.config["weight_threshold"]
        grace = self.config["grace_period"]
        inactivity = self.config["inactivity_threshold"]
        initial_w = self.config["initial_sprouting_weight"]

        to_prune: List[str] = []
        for sid, syn in self.synapses.items():
            age = self.timestep - syn.creation_time

            # Weight-based pruning
            if syn.weight < wt:
                syn.low_weight_steps += 1
                if syn.low_weight_steps > grace:
                    to_prune.append(sid)
                    continue
            else:
                syn.low_weight_steps = 0

            # Activity-based pruning
            if syn.inactive_steps > inactivity:
                to_prune.append(sid)
                continue

            # Age-based pruning: speculative connections that never strengthened
            if age > grace and syn.peak_weight < 2.0 * initial_w:
                to_prune.append(sid)

        for sid in to_prune:
            self._remove_synapse_internal(sid)

        if to_prune:
            self._emit("pruned", count=len(to_prune), timestep=self.timestep)

        return len(to_prune)

    def _sprout_synapses(self, fired_ids: List[str]) -> int:
        """Create synapses between co-activating nodes (PRD §3.3.2).

        Co-activation rule: two nodes fire within co_activation_window,
        no synapse exists → create at initial_weight.

        Performance: capped at 10 new synapses per step to prevent
        explosive growth in highly active networks.
        """
        if not fired_ids:
            return 0

        window = self.config["co_activation_window"]
        initial_w = self.config["initial_sprouting_weight"]
        max_sprouts_per_step = 10
        count = 0

        # Build a set of recently-fired-but-not-this-step nodes for quick lookup
        fired_set = set(fired_ids)
        candidates: Set[str] = set()
        for nid, spikes in self._recent_spikes.items():
            if nid in fired_set:
                continue
            if any(
                0 < (self.timestep - t) <= window
                for t in spikes
            ):
                candidates.add(nid)

        if not candidates:
            return 0

        # Build a fast edge-existence index for fired nodes
        existing_pairs: Set[Tuple[str, str]] = set()
        for nid in fired_ids:
            for sid in self._outgoing.get(nid, set()):
                syn = self.synapses.get(sid)
                if syn:
                    existing_pairs.add((nid, syn.post_node_id))
            for sid in self._incoming.get(nid, set()):
                syn = self.synapses.get(sid)
                if syn:
                    existing_pairs.add((syn.pre_node_id, nid))

        for nid in fired_ids:
            if count >= max_sprouts_per_step:
                break
            for other_id in candidates:
                if count >= max_sprouts_per_step:
                    break
                # Check no existing synapse in either direction
                if (nid, other_id) in existing_pairs:
                    continue
                if (other_id, nid) in existing_pairs:
                    continue
                self.create_synapse(nid, other_id, weight=initial_w)
                existing_pairs.add((nid, other_id))
                count += 1

        if count > 0:
            self._emit("sprouted", count=count, timestep=self.timestep)

        return count

    # -----------------------------------------------------------------------
    # Query Methods (PRD §8)
    # -----------------------------------------------------------------------

    def get_active_nodes(self, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Nodes above voltage threshold (PRD §8 get_active_nodes)."""
        return [
            (nid, node.voltage)
            for nid, node in self.nodes.items()
            if node.voltage >= threshold
        ]

    def get_causal_chain(self, node_id: str, depth: int = 3) -> Dict[str, Any]:
        """Trace learned causality forward from a node (PRD §8 get_causal_chain).

        Returns a dict representing a DAG of causal connections.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found")
        return self._trace_causal(node_id, depth, set())

    def _trace_causal(
        self, node_id: str, depth: int, visited: Set[str]
    ) -> Dict[str, Any]:
        if depth <= 0 or node_id in visited:
            return {"node_id": node_id, "children": []}
        visited.add(node_id)
        children = []
        for sid in self._outgoing.get(node_id, set()):
            syn = self.synapses.get(sid)
            if syn and syn.weight > self.config["weight_threshold"]:
                child = self._trace_causal(syn.post_node_id, depth - 1, visited)
                child["weight"] = syn.weight
                child["delay"] = syn.delay
                children.append(child)
        children.sort(key=lambda c: c.get("weight", 0), reverse=True)
        return {"node_id": node_id, "children": children}

    def get_hyperedges(self, node_id: str) -> List[Hyperedge]:
        """Hyperedges containing this node (PRD §8 get_hyperedges)."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found")
        return [
            self.hyperedges[hid]
            for hid in self._node_hyperedges.get(node_id, set())
            if hid in self.hyperedges
        ]

    def get_active_predictions(self) -> Dict[str, "PredictionState"]:
        """Return currently active (pending) predictions (Phase 2.5)."""
        return dict(self._active_predictions)

    def get_archived_hyperedges(self) -> Dict[str, Hyperedge]:
        """Return archived (subsumed) hyperedges preserved for debugging (Phase 2.5)."""
        return dict(self._archived_hyperedges)

    def get_telemetry(self) -> Telemetry:
        """Network statistics snapshot (PRD §8 get_telemetry).

        Phase 2.5 additions:
            prediction_accuracy: confirmed / total predictions (0 if none).
            surprise_rate: surprises / total predictions (0 if none).
            hyperedge_experience_distribution: bucket histogram of activation_counts.
        """
        weights = [s.weight for s in self.synapses.values()]
        rates = [n.firing_rate_ema for n in self.nodes.values()]
        he_counts = [he.activation_count for he in self.hyperedges.values()]

        # Phase 2.5: Experience distribution buckets
        exp_dist: Dict[str, int] = {"0": 0, "1-9": 0, "10-99": 0, "100+": 0}
        for count in he_counts:
            if count == 0:
                exp_dist["0"] += 1
            elif count < 10:
                exp_dist["1-9"] += 1
            elif count < 100:
                exp_dist["10-99"] += 1
            else:
                exp_dist["100+"] += 1

        # Prediction accuracy — combine Phase 2.5 (HE-level) and Phase 3 (synapse-level)
        combined_confirmed = self._total_predictions_confirmed + self._total_confirmed
        combined_errors = self._total_predictions_errors + self._total_surprised
        total_resolved = combined_confirmed + combined_errors
        accuracy = (
            combined_confirmed / total_resolved
            if total_resolved > 0
            else 0.0
        )
        combined_made = self._total_predictions_made + self._total_predictions
        surprise_rate = (
            combined_errors / max(combined_made, 1)
        )

        return Telemetry(
            timestep=self.timestep,
            total_nodes=len(self.nodes),
            total_synapses=len(self.synapses),
            total_hyperedges=len(self.hyperedges),
            global_firing_rate=float(np.mean(rates)) if rates else 0.0,
            mean_weight=float(np.mean(weights)) if weights else 0.0,
            std_weight=float(np.std(weights)) if weights else 0.0,
            total_pruned=self._total_pruned,
            total_sprouted=self._total_sprouted,
            total_he_discovered=self._total_he_discovered,
            total_he_consolidated=self._total_he_consolidated,
            mean_he_activation_count=float(np.mean(he_counts)) if he_counts else 0.0,
            # Phase 3 telemetry
            prediction_accuracy=accuracy,
            surprise_rate=surprise_rate,
            active_predictions_count=len(self.active_predictions),
            total_predictions_made=self._total_predictions_made,
            total_predictions_confirmed=self._total_predictions_confirmed,
            total_predictions_errors=self._total_predictions_errors,
            total_novel_sequences=self._total_novel_sequences,
            total_rewards_injected=self._total_rewards_injected,
            hyperedge_experience_distribution=exp_dist,
        )

    # -----------------------------------------------------------------------
    # Plasticity Configuration (PRD §8)
    # -----------------------------------------------------------------------

    def set_plasticity_rules(self, rules: List[PlasticityRule]) -> None:
        """Configure active plasticity rules (PRD §8 set_plasticity_rules)."""
        self._plasticity_rules = list(rules)

    def inject_reward(
        self,
        strength: float,
        scope: Optional[Set[str]] = None,
    ) -> None:
        """Broadcast reward for three-factor learning (PRD §5.2, §8 inject_reward).

        Three-factor rule (PRD §5.2):
            Factor 1: Pre-spike (STDP creates eligibility trace)
            Factor 2: Post-spike (updates trace)
            Factor 3: Reward signal (confirms or rejects)
            Final: Δw = eligibility_trace × reward_strength

        Args:
            strength: Reward signal in [-1.0, 1.0].
                Positive = reinforce, negative = weaken.
            scope: Optional set of node IDs to limit reward scope.
                If None, reward applies globally to all synapses with traces.
        """
        lr = self.config["he_threshold_lr"]
        self._total_rewards_injected += 1
        self._reward_history.append({
            "strength": strength,
            "timestep": self.timestep,
            "scope_size": len(scope) if scope else None,
        })
        # Cap reward history
        if len(self._reward_history) > 1000:
            self._reward_history = self._reward_history[-500:]

        for syn in self.synapses.values():
            if abs(syn.eligibility_trace) < 1e-9:
                continue

            # Apply scope filter
            if scope is not None:
                if syn.pre_node_id not in scope and syn.post_node_id not in scope:
                    continue

            dw = syn.eligibility_trace * strength * self.config["learning_rate"]
            syn.weight = max(0.0, min(syn.weight + dw, syn.max_weight))
            syn.eligibility_trace *= 0.9  # Decay trace after use
            if syn.weight > syn.peak_weight:
                syn.peak_weight = syn.weight

        # Hyperedge threshold learning (PRD §4.3)
        for he in self.hyperedges.values():
            if not he.is_learnable:
                continue
            # Scope check for hyperedges
            if scope is not None:
                if not he.member_nodes.intersection(scope):
                    continue
            # Threshold adapts for recently-fired hyperedges
            if he.refractory_remaining > 0:
                if strength > 0:
                    he.activation_threshold = max(0.1, he.activation_threshold - lr * strength)
                else:
                    he.activation_threshold = min(1.0, he.activation_threshold - lr * strength)

        self._emit(
            "reward_injected",
            strength=strength,
            scope_size=len(scope) if scope else None,
            timestep=self.timestep,
        )

    # -----------------------------------------------------------------------
    # Phase 2: Hierarchical Hyperedges (PRD §4.4)
    # -----------------------------------------------------------------------

    def create_hierarchical_hyperedge(
        self,
        child_hyperedge_ids: Set[str],
        activation_threshold: float = 0.6,
        activation_mode: ActivationMode = ActivationMode.WEIGHTED_THRESHOLD,
        output_targets: Optional[List[str]] = None,
        output_weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Hyperedge:
        """Create a meta-hyperedge that groups child hyperedges (PRD §4.4).

        A hierarchical hyperedge "fires" when enough of its children fire.
        Its ``member_nodes`` is the union of all children's member_nodes,
        so it participates in the same activation check but at a higher level.

        Args:
            child_hyperedge_ids: IDs of existing hyperedges to compose.
            activation_threshold: Fraction of child members that must be active.
            activation_mode: Activation function.
            output_targets: Nodes to inject current when this meta-HE fires.
            output_weight: Injection strength.
            metadata: Application data.

        Returns:
            The new hierarchical Hyperedge (level = max(child levels) + 1).
        """
        all_member_nodes: Set[str] = set()
        max_child_level = 0
        for chid in child_hyperedge_ids:
            child = self.hyperedges.get(chid)
            if child is None:
                raise KeyError(f"Child hyperedge {chid} not found")
            all_member_nodes.update(child.member_nodes)
            max_child_level = max(max_child_level, child.level)

        for nid in all_member_nodes:
            if nid not in self.nodes:
                raise KeyError(f"Member node {nid} not found")

        he = Hyperedge(
            member_nodes=all_member_nodes,
            member_weights={nid: 1.0 for nid in all_member_nodes},
            activation_threshold=activation_threshold,
            activation_mode=activation_mode,
            output_targets=output_targets or [],
            output_weight=output_weight,
            metadata=metadata or {},
            is_learnable=True,
            child_hyperedges=set(child_hyperedge_ids),
            level=max_child_level + 1,
        )
        self.hyperedges[he.hyperedge_id] = he
        for nid in all_member_nodes:
            self._node_hyperedges.setdefault(nid, set()).add(he.hyperedge_id)
        self._he_co_fire_counts[he.hyperedge_id] = {}
        self._dirty_hyperedges.add(he.hyperedge_id)
        return he

    # -----------------------------------------------------------------------
    # Phase 2: Hyperedge Discovery (PRD §3.3.2 extended)
    # -----------------------------------------------------------------------

    def discover_hyperedges(self, fired_node_ids: List[str]) -> List[Hyperedge]:
        """Discover new hyperedges from co-activation patterns (PRD §4.3).

        Tracks groups of nodes that consistently fire together. When a group
        exceeds ``he_discovery_min_co_fires`` within ``he_discovery_window``,
        a new hyperedge is created.

        Args:
            fired_node_ids: Nodes that fired this step.

        Returns:
            List of newly discovered Hyperedges (may be empty).
        """
        if len(fired_node_ids) < self.config["he_discovery_min_nodes"]:
            return []

        window = self.config["he_discovery_window"]
        min_fires = self.config["he_discovery_min_co_fires"]
        min_nodes = self.config["he_discovery_min_nodes"]

        # Reset discovery counts periodically
        if self.timestep - self._he_discovery_last_reset > window * 2:
            self._he_discovery_counts.clear()
            self._he_discovery_last_reset = self.timestep

        # Build sorted key from fired nodes (order-independent)
        fired_sorted = tuple(sorted(fired_node_ids))

        # Track all subsets of size >= min_nodes would be too expensive,
        # so we track the full fired set as a co-activation pattern.
        self._he_discovery_counts[fired_sorted] = (
            self._he_discovery_counts.get(fired_sorted, 0) + 1
        )

        discovered: List[Hyperedge] = []

        if self._he_discovery_counts[fired_sorted] >= min_fires:
            # Check no existing hyperedge already covers this exact set
            fired_set = set(fired_node_ids)
            already_exists = any(
                he.member_nodes == fired_set
                for he in self.hyperedges.values()
            )
            if not already_exists and len(fired_set) >= min_nodes:
                # Verify all nodes still exist
                valid_nodes = {nid for nid in fired_set if nid in self.nodes}
                if len(valid_nodes) >= min_nodes:
                    he = self.create_hyperedge(
                        valid_nodes,
                        activation_threshold=0.6,
                        metadata={"creation_mode": "discovered", "timestep": self.timestep},
                    )
                    discovered.append(he)
                    self._total_he_discovered += 1
                    self._emit("hyperedge_discovered", hid=he.hyperedge_id)
            # Reset this pattern's counter
            del self._he_discovery_counts[fired_sorted]

        return discovered

    # -----------------------------------------------------------------------
    # Phase 2: Hyperedge Consolidation (PRD §4.3 extended)
    # -----------------------------------------------------------------------

    def consolidate_hyperedges(self) -> int:
        """Merge highly overlapping hyperedges and archive subsumed ones (PRD §4.3).

        Phase 2 behavior: Two hyperedges at same level with >80% member overlap
        (Jaccard) get merged into one, keeping the union of members and the lower
        threshold.

        Phase 2.5 addition — cross-level consistency pruning: after same-level
        merges, detect subsumption where a lower-level hyperedge has identical
        members to (or is a hierarchical child of) a higher-level one.  The
        lower-level edge is archived (``is_archived=True``, preserved in
        ``_archived_hyperedges`` for debugging) rather than deleted.

        Returns:
            Number of hyperedges removed or archived by consolidation.
        """
        overlap_threshold = self.config["he_consolidation_overlap"]
        he_list = [(hid, he) for hid, he in self.hyperedges.items()
                   if he.level == 0 and not he.is_archived]
        to_remove: Set[str] = set()
        merged_count = 0

        for i in range(len(he_list)):
            hid_a, he_a = he_list[i]
            if hid_a in to_remove:
                continue
            for j in range(i + 1, len(he_list)):
                hid_b, he_b = he_list[j]
                if hid_b in to_remove:
                    continue

                # Jaccard similarity
                intersection = he_a.member_nodes & he_b.member_nodes
                union = he_a.member_nodes | he_b.member_nodes
                if not union:
                    continue
                jaccard = len(intersection) / len(union)

                if jaccard >= overlap_threshold:
                    # Merge B into A: expand A's members, keep lower threshold
                    for nid in he_b.member_nodes - he_a.member_nodes:
                        he_a.member_nodes.add(nid)
                        he_a.member_weights[nid] = he_b.member_weights.get(nid, 1.0)
                        self._node_hyperedges.setdefault(nid, set()).add(hid_a)
                    he_a.activation_threshold = min(
                        he_a.activation_threshold,
                        he_b.activation_threshold,
                    )
                    he_a.activation_count += he_b.activation_count
                    to_remove.add(hid_b)
                    merged_count += 1

        for hid in to_remove:
            self._remove_hyperedge_internal(hid)

        # --- Phase 2.5: Cross-level consistency pruning (subsumption) ---
        archived_count = self._prune_subsumed_hyperedges()
        merged_count += archived_count

        self._total_he_consolidated += merged_count
        if merged_count > 0:
            self._emit("hyperedges_consolidated", count=merged_count)

        return merged_count

    def _prune_subsumed_hyperedges(self) -> int:
        """Archive lower-level hyperedges subsumed by higher-level ones.

        Subsumption criteria:
            - Jaccard similarity = 1.0 (exact member match), OR
            - The lower-level hyperedge is a child of the higher-level one
              AND they share identical members.
            - Keep the higher-level abstraction, archive the lower.

        Returns:
            Number of hyperedges archived.
        """
        archived_count = 0
        # Build lookup: level → list of (hid, he)
        by_level: Dict[int, List[Tuple[str, Hyperedge]]] = {}
        for hid, he in self.hyperedges.items():
            if he.is_archived:
                continue
            by_level.setdefault(he.level, []).append((hid, he))

        levels = sorted(by_level.keys())
        if len(levels) < 2:
            return 0

        for lower_level in levels:
            for higher_level in levels:
                if higher_level <= lower_level:
                    continue
                for hid_lo, he_lo in by_level.get(lower_level, []):
                    if he_lo.is_archived:
                        continue
                    for hid_hi, he_hi in by_level.get(higher_level, []):
                        if he_hi.is_archived:
                            continue
                        # Check exact member match (Jaccard = 1.0)
                        if he_lo.member_nodes == he_hi.member_nodes:
                            # Archive the lower-level one
                            he_lo.is_archived = True
                            self._archived_hyperedges[hid_lo] = he_lo
                            archived_count += 1
                            self._emit("hyperedge_archived",
                                       archived_id=hid_lo,
                                       subsumed_by=hid_hi,
                                       reason="exact_member_match")
                            break
                        # Check if lower is a child of higher with identical members
                        if hid_lo in he_hi.child_hyperedges:
                            if he_lo.member_nodes == he_hi.member_nodes:
                                he_lo.is_archived = True
                                self._archived_hyperedges[hid_lo] = he_lo
                                archived_count += 1
                                self._emit("hyperedge_archived",
                                           archived_id=hid_lo,
                                           subsumed_by=hid_hi,
                                           reason="child_subsumption")
                                break

        return archived_count

    # -----------------------------------------------------------------------
    # Event System (PRD §8 register_event_handler)
    # -----------------------------------------------------------------------

    def register_event_handler(self, event_type: str, callback: Callable) -> None:
        """Subscribe to events: spikes, predictions, pruning, etc. (PRD §8)."""
        self._event_handlers.setdefault(event_type, []).append(callback)

    def _emit(self, event_type: str, **kwargs: Any) -> None:
        for cb in self._event_handlers.get(event_type, []):
            cb(**kwargs)

    # -----------------------------------------------------------------------
    # Persistence (PRD §6)
    # -----------------------------------------------------------------------

    def checkpoint(self, path: str, mode: CheckpointMode = CheckpointMode.FULL) -> None:
        """Save state (PRD §8 checkpoint, §6).

        Args:
            path: File path (extension determines format: .json or .msgpack).
            mode: FULL, INCREMENTAL, or FORK.
        """
        if mode == CheckpointMode.FULL:
            data = self._serialize_full()
        elif mode == CheckpointMode.INCREMENTAL:
            data = self._serialize_incremental()
            # Clear dirty flags after incremental save
            self._dirty_nodes.clear()
            self._dirty_synapses.clear()
            self._dirty_hyperedges.clear()
        elif mode == CheckpointMode.FORK:
            data = self._serialize_full()
            data["_fork"] = True
        else:
            raise ValueError(f"Unknown checkpoint mode: {mode}")

        if path.endswith(".msgpack"):
            if msgpack is None:
                raise ImportError("msgpack required for .msgpack serialization")
            with open(path, "wb") as f:
                msgpack.pack(data, f, use_bin_type=True)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)

    def restore(self, path: str) -> None:
        """Load state from checkpoint (PRD §8 restore, §6)."""
        if path.endswith(".msgpack"):
            if msgpack is None:
                raise ImportError("msgpack required for .msgpack deserialization")
            with open(path, "rb") as f:
                data = msgpack.unpack(f, raw=False)
        else:
            with open(path, "r") as f:
                data = json.load(f)

        self._deserialize(data)

    def _serialize_node(self, node: Node) -> Dict[str, Any]:
        return {
            "node_id": node.node_id,
            "voltage": node.voltage,
            "threshold": node.threshold,
            "resting_potential": node.resting_potential,
            "refractory_remaining": node.refractory_remaining,
            "refractory_period": node.refractory_period,
            "last_spike_time": node.last_spike_time if not math.isinf(node.last_spike_time) else None,
            "spike_history": node.spike_history.to_list(),
            "spike_history_capacity": node.spike_history.capacity,
            "firing_rate_ema": node.firing_rate_ema,
            "intrinsic_excitability": node.intrinsic_excitability,
            "metadata": node.metadata,
            "is_inhibitory": node.is_inhibitory,
        }

    def _serialize_synapse(self, syn: Synapse) -> Dict[str, Any]:
        return {
            "synapse_id": syn.synapse_id,
            "pre_node_id": syn.pre_node_id,
            "post_node_id": syn.post_node_id,
            "weight": syn.weight,
            "max_weight": syn.max_weight,
            "delay": syn.delay,
            "last_update_time": syn.last_update_time,
            "eligibility_trace": syn.eligibility_trace,
            "creation_time": syn.creation_time,
            "synapse_type": syn.synapse_type.name,
            "peak_weight": syn.peak_weight,
            "low_weight_steps": syn.low_weight_steps,
            "inactive_steps": syn.inactive_steps,
            "metadata": syn.metadata,
        }

    def _serialize_hyperedge(self, he: Hyperedge) -> Dict[str, Any]:
        return {
            "hyperedge_id": he.hyperedge_id,
            "member_nodes": list(he.member_nodes),
            "member_weights": he.member_weights,
            "activation_threshold": he.activation_threshold,
            "activation_mode": he.activation_mode.name,
            "current_activation": he.current_activation,
            "output_targets": he.output_targets,
            "output_weight": he.output_weight,
            "metadata": he.metadata,
            "is_learnable": he.is_learnable,
            "refractory_period": he.refractory_period,
            "refractory_remaining": he.refractory_remaining,
            "activation_count": he.activation_count,
            "pattern_completion_strength": he.pattern_completion_strength,
            "child_hyperedges": list(he.child_hyperedges),
            "level": he.level,
            # Phase 2.5 fields
            "recent_activation_ema": he.recent_activation_ema,
            "is_archived": he.is_archived,
        }

    def _serialize_prediction(self, pred: Prediction) -> Dict[str, Any]:
        """Serialize a Phase 3 synapse-level Prediction."""
        return {
            "prediction_id": pred.prediction_id,
            "source_node_id": pred.source_node_id,
            "target_node_id": pred.target_node_id,
            "strength": pred.strength,
            "confidence": pred.confidence,
            "created_at": pred.created_at,
            "expires_at": pred.expires_at,
            "chain_depth": pred.chain_depth,
            "via_hyperedge": pred.via_hyperedge,
            "pre_charge_applied": pred.pre_charge_applied,
        }

    def _serialize_prediction_state(self, ps: PredictionState) -> Dict[str, Any]:
        """Serialize a Phase 2.5 hyperedge-level PredictionState."""
        return {
            "hyperedge_id": ps.hyperedge_id,
            "predicted_targets": list(ps.predicted_targets),
            "prediction_strength": ps.prediction_strength,
            "prediction_timestamp": ps.prediction_timestamp,
            "prediction_window": ps.prediction_window,
            "confirmed_targets": list(ps.confirmed_targets),
        }

    def _serialize_full(self) -> Dict[str, Any]:
        return {
            "version": "0.3.5",
            "timestep": self.timestep,
            "config": self.config,
            "nodes": {nid: self._serialize_node(n) for nid, n in self.nodes.items()},
            "synapses": {sid: self._serialize_synapse(s) for sid, s in self.synapses.items()},
            "hyperedges": {hid: self._serialize_hyperedge(h) for hid, h in self.hyperedges.items()},
            "archived_hyperedges": {
                hid: self._serialize_hyperedge(h)
                for hid, h in self._archived_hyperedges.items()
            },
            # Phase 3: Active synapse-level predictions
            "active_predictions": {
                pid: self._serialize_prediction(pred)
                for pid, pred in self.active_predictions.items()
            },
            # Phase 3: Recent prediction outcomes
            "prediction_outcomes": [
                {
                    "prediction": self._serialize_prediction(po.prediction),
                    "confirmed": po.confirmed,
                    "resolved_at": po.resolved_at,
                    "actual_firing_nodes": po.actual_firing_nodes,
                }
                for po in self._prediction_outcomes
            ],
            # Phase 3: Per-synapse confirmation history
            "synapse_confirmation_history": {
                syn_id: list(history)
                for syn_id, history in self._synapse_confirmation_history.items()
            },
            # Phase 3: Logs
            "novel_sequence_log": list(self._novel_sequence_log),
            "reward_history": list(self._reward_history),
            # Phase 2.5: Active HE-level predictions
            "he_active_predictions": {
                pid: self._serialize_prediction_state(ps)
                for pid, ps in self._active_predictions.items()
            },
            # Phase 2.5: Window-fired tracking
            "he_prediction_window_fired": {
                pid: list(nodes)
                for pid, nodes in self._prediction_window_fired.items()
            },
            # Phase 2.5: Counter for unique HE prediction IDs
            "he_prediction_counter": self._prediction_counter,
            "telemetry": {
                "total_pruned": self._total_pruned,
                "total_sprouted": self._total_sprouted,
                "total_he_discovered": self._total_he_discovered,
                "total_he_consolidated": self._total_he_consolidated,
                "total_predictions_made": self._total_predictions_made,
                "total_predictions_confirmed": self._total_predictions_confirmed,
                "total_predictions_errors": self._total_predictions_errors,
                "total_novel_sequences": self._total_novel_sequences,
                "total_rewards_injected": self._total_rewards_injected,
                # Phase 2.5 counters
                "he_total_predictions": self._total_predictions,
                "he_total_confirmed": self._total_confirmed,
                "he_total_surprised": self._total_surprised,
            },
        }

    def _serialize_incremental(self) -> Dict[str, Any]:
        return {
            "version": "0.1.0",
            "incremental": True,
            "timestep": self.timestep,
            "nodes": {
                nid: self._serialize_node(self.nodes[nid])
                for nid in self._dirty_nodes
                if nid in self.nodes
            },
            "synapses": {
                sid: self._serialize_synapse(self.synapses[sid])
                for sid in self._dirty_synapses
                if sid in self.synapses
            },
            "hyperedges": {
                hid: self._serialize_hyperedge(self.hyperedges[hid])
                for hid in self._dirty_hyperedges
                if hid in self.hyperedges
            },
        }

    def _deserialize(self, data: Dict[str, Any]) -> None:
        """Restore full graph state from serialized data."""
        self.config = {**DEFAULT_CONFIG, **data.get("config", {})}
        self.timestep = data.get("timestep", 0)

        # Clear existing state
        self.nodes.clear()
        self.synapses.clear()
        self.hyperedges.clear()
        self._outgoing.clear()
        self._incoming.clear()
        self._node_hyperedges.clear()
        self._recent_spikes.clear()
        self._delay_buffer.clear()
        self._active_predictions.clear()
        self._prediction_window_fired.clear()
        self._archived_hyperedges.clear()

        # Restore nodes
        for nid, nd in data.get("nodes", {}).items():
            lst = nd.get("last_spike_time")
            node = Node(
                node_id=nid,
                voltage=nd["voltage"],
                threshold=nd["threshold"],
                resting_potential=nd.get("resting_potential", 0.0),
                refractory_remaining=nd.get("refractory_remaining", 0),
                refractory_period=nd.get("refractory_period", 2),
                last_spike_time=lst if lst is not None else -math.inf,
                spike_history=RingBuffer.from_list(
                    nd.get("spike_history", []),
                    nd.get("spike_history_capacity", 100),
                ),
                firing_rate_ema=nd.get("firing_rate_ema", 0.0),
                intrinsic_excitability=nd.get("intrinsic_excitability", 1.0),
                metadata=nd.get("metadata", {}),
                is_inhibitory=nd.get("is_inhibitory", False),
            )
            self.nodes[nid] = node
            self._outgoing[nid] = set()
            self._incoming[nid] = set()
            self._node_hyperedges[nid] = set()
            self._recent_spikes[nid] = deque(maxlen=20)

        # Restore synapses
        for sid, sd in data.get("synapses", {}).items():
            syn = Synapse(
                synapse_id=sid,
                pre_node_id=sd["pre_node_id"],
                post_node_id=sd["post_node_id"],
                weight=sd["weight"],
                max_weight=sd.get("max_weight", 5.0),
                delay=sd.get("delay", 1),
                last_update_time=sd.get("last_update_time", 0.0),
                eligibility_trace=sd.get("eligibility_trace", 0.0),
                creation_time=sd.get("creation_time", 0.0),
                synapse_type=SynapseType[sd.get("synapse_type", "EXCITATORY")],
                peak_weight=sd.get("peak_weight", sd["weight"]),
                low_weight_steps=sd.get("low_weight_steps", 0),
                inactive_steps=sd.get("inactive_steps", 0),
                metadata=sd.get("metadata", {}),
            )
            self.synapses[sid] = syn
            self._outgoing.setdefault(syn.pre_node_id, set()).add(sid)
            self._incoming.setdefault(syn.post_node_id, set()).add(sid)

        # Restore hyperedges
        for hid, hd in data.get("hyperedges", {}).items():
            he = Hyperedge(
                hyperedge_id=hid,
                member_nodes=set(hd["member_nodes"]),
                member_weights=hd.get("member_weights", {}),
                activation_threshold=hd.get("activation_threshold", 0.6),
                activation_mode=ActivationMode[hd.get("activation_mode", "WEIGHTED_THRESHOLD")],
                current_activation=hd.get("current_activation", 0.0),
                output_targets=hd.get("output_targets", []),
                output_weight=hd.get("output_weight", 1.0),
                metadata=hd.get("metadata", {}),
                is_learnable=hd.get("is_learnable", True),
                refractory_period=hd.get("refractory_period", 2),
                refractory_remaining=hd.get("refractory_remaining", 0),
                activation_count=hd.get("activation_count", 0),
                pattern_completion_strength=hd.get("pattern_completion_strength", 0.3),
                child_hyperedges=set(hd.get("child_hyperedges", [])),
                level=hd.get("level", 0),
                recent_activation_ema=hd.get("recent_activation_ema", 0.0),
                is_archived=hd.get("is_archived", False),
            )
            self.hyperedges[hid] = he
            for nid in he.member_nodes:
                self._node_hyperedges.setdefault(nid, set()).add(hid)
            self._he_co_fire_counts[hid] = {}

        # Restore archived hyperedges (Phase 2.5)
        self._archived_hyperedges.clear()
        for hid, hd in data.get("archived_hyperedges", {}).items():
            he = Hyperedge(
                hyperedge_id=hid,
                member_nodes=set(hd["member_nodes"]),
                member_weights=hd.get("member_weights", {}),
                activation_threshold=hd.get("activation_threshold", 0.6),
                activation_mode=ActivationMode[hd.get("activation_mode", "WEIGHTED_THRESHOLD")],
                current_activation=hd.get("current_activation", 0.0),
                output_targets=hd.get("output_targets", []),
                output_weight=hd.get("output_weight", 1.0),
                metadata=hd.get("metadata", {}),
                is_learnable=hd.get("is_learnable", True),
                refractory_period=hd.get("refractory_period", 2),
                refractory_remaining=hd.get("refractory_remaining", 0),
                activation_count=hd.get("activation_count", 0),
                pattern_completion_strength=hd.get("pattern_completion_strength", 0.3),
                child_hyperedges=set(hd.get("child_hyperedges", [])),
                level=hd.get("level", 0),
                recent_activation_ema=hd.get("recent_activation_ema", 0.0),
                is_archived=hd.get("is_archived", True),
            )
            self._archived_hyperedges[hid] = he

        # Restore telemetry
        tel = data.get("telemetry", {})
        self._total_pruned = tel.get("total_pruned", 0)
        self._total_sprouted = tel.get("total_sprouted", 0)
        self._total_he_discovered = tel.get("total_he_discovered", 0)
        self._total_he_consolidated = tel.get("total_he_consolidated", 0)
        self._total_predictions_made = tel.get("total_predictions_made", 0)
        self._total_predictions_confirmed = tel.get("total_predictions_confirmed", 0)
        self._total_predictions_errors = tel.get("total_predictions_errors", 0)
        self._total_novel_sequences = tel.get("total_novel_sequences", 0)
        self._total_rewards_injected = tel.get("total_rewards_injected", 0)
        # Phase 2.5 counters
        self._total_predictions = tel.get("he_total_predictions", 0)
        self._total_confirmed = tel.get("he_total_confirmed", 0)
        self._total_surprised = tel.get("he_total_surprised", 0)

        # Restore Phase 3 active predictions with validation
        self.active_predictions.clear()
        node_ids = set(self.nodes.keys())
        for pid, pd in data.get("active_predictions", {}).items():
            src = pd.get("source_node_id", "")
            tgt = pd.get("target_node_id", "")
            # Validate: both source and target nodes must exist
            if src not in node_ids or tgt not in node_ids:
                continue
            # Validate: prediction must not already be expired
            if pd.get("expires_at", 0) <= self.timestep:
                continue
            pred = Prediction(
                prediction_id=pd.get("prediction_id", pid),
                source_node_id=src,
                target_node_id=tgt,
                strength=pd.get("strength", 0.0),
                confidence=pd.get("confidence", 0.0),
                created_at=pd.get("created_at", 0),
                expires_at=pd.get("expires_at", 0),
                chain_depth=pd.get("chain_depth", 0),
                via_hyperedge=pd.get("via_hyperedge"),
                pre_charge_applied=pd.get("pre_charge_applied", 0.0),
            )
            self.active_predictions[pid] = pred

        # Restore Phase 3 prediction outcomes
        self._prediction_outcomes.clear()
        for pod in data.get("prediction_outcomes", []):
            ppd = pod.get("prediction", {})
            inner_pred = Prediction(
                prediction_id=ppd.get("prediction_id", ""),
                source_node_id=ppd.get("source_node_id", ""),
                target_node_id=ppd.get("target_node_id", ""),
                strength=ppd.get("strength", 0.0),
                confidence=ppd.get("confidence", 0.0),
                created_at=ppd.get("created_at", 0),
                expires_at=ppd.get("expires_at", 0),
                chain_depth=ppd.get("chain_depth", 0),
                via_hyperedge=ppd.get("via_hyperedge"),
                pre_charge_applied=ppd.get("pre_charge_applied", 0.0),
            )
            outcome = PredictionOutcome(
                prediction=inner_pred,
                confirmed=pod.get("confirmed", False),
                resolved_at=pod.get("resolved_at", 0),
                actual_firing_nodes=pod.get("actual_firing_nodes", []),
            )
            self._prediction_outcomes.append(outcome)

        # Restore per-synapse confirmation history
        self._synapse_confirmation_history.clear()
        synapse_ids = set(self.synapses.keys())
        for syn_id, hist in data.get("synapse_confirmation_history", {}).items():
            if syn_id not in synapse_ids:
                continue  # Skip stale entries for deleted synapses
            d: Deque[bool] = deque(hist, maxlen=100)
            self._synapse_confirmation_history[syn_id] = d

        # Restore logs
        self._novel_sequence_log = list(data.get("novel_sequence_log", []))
        self._reward_history = list(data.get("reward_history", []))

        # Restore Phase 2.5 HE-level predictions with validation
        self._active_predictions.clear()
        self._prediction_window_fired.clear()
        he_ids = set(self.hyperedges.keys())
        for pid, psd in data.get("he_active_predictions", {}).items():
            he_id = psd.get("hyperedge_id", "")
            if he_id not in he_ids:
                continue  # HE was removed
            predicted = set(psd.get("predicted_targets", []))
            # Validate: at least some targets must still exist
            valid_targets = predicted & node_ids
            if not valid_targets:
                continue
            ps = PredictionState(
                hyperedge_id=he_id,
                predicted_targets=valid_targets,
                prediction_strength=psd.get("prediction_strength", 0.0),
                prediction_timestamp=psd.get("prediction_timestamp", 0),
                prediction_window=psd.get("prediction_window", 5),
                confirmed_targets=set(psd.get("confirmed_targets", [])) & node_ids,
            )
            # Validate: window must not already be expired
            if self.timestep - ps.prediction_timestamp >= ps.prediction_window:
                continue
            self._active_predictions[pid] = ps

        # Restore Phase 2.5 window-fired tracking
        for pid, fired_list in data.get("he_prediction_window_fired", {}).items():
            if pid in self._active_predictions:
                self._prediction_window_fired[pid] = set(fired_list)

        # Restore Phase 2.5 prediction counter
        self._prediction_counter = data.get("he_prediction_counter", 0)

        # Transient per-step state — always start fresh
        self._predicted_this_step.clear()

        # Re-init plasticity rules from config
        self._plasticity_rules = [
            STDPRule(
                tau_plus=self.config["tau_plus"],
                tau_minus=self.config["tau_minus"],
                A_plus=self.config["A_plus"],
                A_minus=self.config["A_minus"],
                learning_rate=self.config["learning_rate"],
            ),
            HomeostaticRule(
                target_firing_rate=self.config["target_firing_rate"],
                scaling_interval=self.config["scaling_interval"],
            ),
            HyperedgePlasticityRule(
                member_weight_lr=self.config["he_member_weight_lr"],
                threshold_lr=self.config["he_threshold_lr"],
                evolution_window=self.config["he_member_evolution_window"],
                evolution_min_co_fires=self.config["he_member_evolution_min_co_fires"],
                evolution_initial_weight=self.config["he_member_evolution_initial_weight"],
            ),
        ]

        self._dirty_nodes.clear()
        self._dirty_synapses.clear()
        self._dirty_hyperedges.clear()
        self._he_discovery_counts.clear()
        self._he_discovery_last_reset = self.timestep
