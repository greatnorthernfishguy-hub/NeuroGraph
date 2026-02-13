"""
NeuroGraph Foundation - Core Cognitive Architecture (Phase 1)

Implements the Temporal Dynamics Layer: a sparse Spiking Neural Network (SNN)
with STDP plasticity, homeostatic regulation, structural plasticity, and
hypergraph support.

Reference: NeuroGraph Foundation PRD v1.0, Sections 2-6, 9.

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

    def apply(
        self,
        graph: "Graph",
        fired_node_ids: List[str],
        timestep: int,
    ) -> None:
        """Apply STDP to all synapses incident on fired nodes."""
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

                syn.weight = max(0.0, min(syn.weight + dw, syn.max_weight))
                syn.last_update_time = float(timestep)
                if syn.weight > syn.peak_weight:
                    syn.peak_weight = syn.weight

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

                syn.weight = max(0.0, min(syn.weight + dw, syn.max_weight))
                syn.last_update_time = float(timestep)
                if syn.weight > syn.peak_weight:
                    syn.peak_weight = syn.weight


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
        ]

        # --- Event handlers ---
        self._event_handlers: Dict[str, List[Callable]] = {}

        # --- Telemetry counters ---
        self._total_pruned = 0
        self._total_sprouted = 0

        # --- Clock ---
        self.timestep: int = 0

        # --- Dirty flags for incremental checkpointing ---
        self._dirty_nodes: Set[str] = set()
        self._dirty_synapses: Set[str] = set()
        self._dirty_hyperedges: Set[str] = set()

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
        self._dirty_hyperedges.add(he.hyperedge_id)
        return he

    def _remove_hyperedge_internal(self, hyperedge_id: str) -> None:
        he = self.hyperedges.pop(hyperedge_id, None)
        if he is None:
            return
        for nid in he.member_nodes:
            self._node_hyperedges.get(nid, set()).discard(hyperedge_id)
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

        # 6. Evaluate hyperedges (PRD §4.2)
        fired_set = set(fired_ids)
        for hid, he in self.hyperedges.items():
            activation = self._compute_hyperedge_activation(he, fired_set)
            he.current_activation = activation
            if activation >= he.activation_threshold:
                result.fired_hyperedge_ids.append(hid)
                # Inject current into output targets
                for target_id in he.output_targets:
                    target = self.nodes.get(target_id)
                    if target is not None:
                        target.voltage += he.output_weight * target.intrinsic_excitability
                self._emit("hyperedge_fired", hid=hid, activation=activation)

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

    def get_telemetry(self) -> Telemetry:
        """Network statistics snapshot (PRD §8 get_telemetry)."""
        weights = [s.weight for s in self.synapses.values()]
        rates = [n.firing_rate_ema for n in self.nodes.values()]
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
        )

    # -----------------------------------------------------------------------
    # Plasticity Configuration (PRD §8)
    # -----------------------------------------------------------------------

    def set_plasticity_rules(self, rules: List[PlasticityRule]) -> None:
        """Configure active plasticity rules (PRD §8 set_plasticity_rules)."""
        self._plasticity_rules = list(rules)

    def inject_reward(self, strength: float) -> None:
        """Broadcast reward for three-factor learning (PRD §8 inject_reward).

        Commits eligibility traces proportional to strength.
        """
        for syn in self.synapses.values():
            if abs(syn.eligibility_trace) > 1e-9:
                dw = syn.eligibility_trace * strength * self.config["learning_rate"]
                syn.weight = max(0.0, min(syn.weight + dw, syn.max_weight))
                syn.eligibility_trace *= 0.9  # Decay trace

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
        }

    def _serialize_full(self) -> Dict[str, Any]:
        return {
            "version": "0.1.0",
            "timestep": self.timestep,
            "config": self.config,
            "nodes": {nid: self._serialize_node(n) for nid, n in self.nodes.items()},
            "synapses": {sid: self._serialize_synapse(s) for sid, s in self.synapses.items()},
            "hyperedges": {hid: self._serialize_hyperedge(h) for hid, h in self.hyperedges.items()},
            "telemetry": {
                "total_pruned": self._total_pruned,
                "total_sprouted": self._total_sprouted,
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
            )
            self.hyperedges[hid] = he
            for nid in he.member_nodes:
                self._node_hyperedges.setdefault(nid, set()).add(hid)

        # Restore telemetry
        tel = data.get("telemetry", {})
        self._total_pruned = tel.get("total_pruned", 0)
        self._total_sprouted = tel.get("total_sprouted", 0)

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
        ]

        self._dirty_nodes.clear()
        self._dirty_synapses.clear()
        self._dirty_hyperedges.clear()
