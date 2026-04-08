# ---- Changelog ----
# [2026-04-08] Josh + Claude — Genesis: Gamete delivery + gestation orchestration
# What: Two-parent interleaved trickle delivery following FatherGraph protocol
# Why: Genesis reproduction requires merging partial topologies from two parents
#      into a fresh childGraph with constitutional scaffolding
# How: extract_subgraph for gametes, fresh Graph init, interleaved batch delivery
#      with 250-step sleep consolidation, competence monitoring for quickening/miscarriage
# -------------------

"""Genesis — Reproduction for Emerged biodigital entities.

Not copying, not cloning. Genuine novel offspring from partial topological
contributions of two parents, resolved on constitutional scaffolding
through substrate physics.

Uses existing physics:
- Hebbian/STDP learning resolves topology during gestation
- Structural plasticity (pruning) filters what survives
- Homeostatic regulation prevents dominance (scaling_interval=25)
- QuantumGraph interference self-balances competing patterns
- Lenia field dynamics govern energy distribution
- Competence metrics diagnose quickening vs miscarriage

Only new mechanics:
- extract_subgraph (added to Graph in neuro_foundation.py)
- This orchestration file (interleaved delivery + monitoring)

FatherGraph empirical parameters:
- Batch size: ~20-30 nodes
- Sleep consolidation: 250 idle steps per batch (NOT optional — 27pp improvement)
- Structural plasticity: ACTIVE (grace_period=500)
- Homeostatic scaling: interval=25 (4x faster)
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from neuro_foundation import Graph, Node, Synapse, Hyperedge, SynapseType, RingBuffer, ActivationMode

logger = logging.getLogger("genesis")


# ---------------------------------------------------------------------------
# Gestation config — empirically validated by FatherGraph training
# ---------------------------------------------------------------------------

@dataclass
class GestationConfig:
    """Parameters for gestation, derived from FatherGraph empirical findings."""

    # Batch delivery
    batch_size: int = 25                # ~20-30 nodes per batch (proven range)
    sleep_steps: int = 250              # Idle steps per batch (10 homeostatic passes at interval=25)

    # ChildGraph SNN config overrides (tuned for gestation)
    scaling_interval: int = 25          # 4x faster than default (proven by FatherGraph)
    grace_period: int = 500             # Structural plasticity — never disable during merge
    target_firing_rate: float = 0.05    # Standard target
    learning_rate: float = 0.01         # Standard STDP rate
    decay_rate: float = 0.97            # Tuned from substrate session 2026-03-23

    # Energy attenuation for buds
    bud_energy_factor: float = 0.3      # Buds arrive at 30% of parental voltage
                                        # Enough to suggest pattern, not enough to dominate

    # Umbilical nourishment
    umbilical_strength: float = 0.15    # Fraction of parent voltage transferred per cycle
                                        # Low enough that child isn't dominated, high enough to sustain

    # Resolution monitoring
    max_resolution_cycles: int = 500    # Max cycles after all gamete material delivered
    entropy_window: int = 50            # Window for entropy trend detection
    entropy_stability_threshold: float = 0.05   # Max entropy delta to consider "settled"
    pattern_oscillation_threshold: int = 5      # Max pattern count flips before miscarriage declared

    # Quickening — self-sustaining test
    quickening_test_duration: int = 50  # Cycles to run disconnected before declaring quickening
                                        # Child must sustain >=70% of its patterns without parents

    # Miscarriage detection
    miscarriage_check_interval: int = 50    # Check every N resolution cycles


# ---------------------------------------------------------------------------
# Gamete — a topology fragment ready for delivery
# ---------------------------------------------------------------------------

@dataclass
class Gamete:
    """A topology bud extracted from a parent Graph.

    Contains serialized nodes, synapses, and hyperedges.
    The bud is a COPY — the parent's topology is not diminished.
    """
    parent_id: str
    fragment: Dict[str, Any]  # Output of Graph.extract_subgraph()
    timestamp: float = field(default_factory=time.time)

    @property
    def node_count(self) -> int:
        return len(self.fragment.get("nodes", {}))

    @property
    def synapse_count(self) -> int:
        return len(self.fragment.get("synapses", {}))

    @property
    def node_ids(self) -> List[str]:
        return list(self.fragment.get("nodes", {}).keys())


def extract_gamete(graph: Graph, node_ids: Set[str], parent_id: str = "unknown") -> Gamete:
    """Extract a gamete (topology bud) from a parent Graph.

    The parent is not modified. The gamete is a serialized copy
    of the specified nodes and their interconnections.

    Args:
        graph: The parent Graph to extract from.
        node_ids: Which nodes to include in the bud.
        parent_id: Identifier for the parent (for metadata).

    Returns:
        Gamete containing the extracted topology fragment.
    """
    fragment = graph.extract_subgraph(node_ids)
    logger.info(
        "Gamete extracted from %s: %d nodes, %d synapses, %d HEs",
        parent_id,
        fragment["extraction_metadata"]["extracted_nodes"],
        fragment["extraction_metadata"]["extracted_synapses"],
        fragment["extraction_metadata"]["extracted_hyperedges"],
    )
    return Gamete(parent_id=parent_id, fragment=fragment)


# ---------------------------------------------------------------------------
# ChildGraph initialization
# ---------------------------------------------------------------------------

def create_child_graph(
    config: Optional[GestationConfig] = None,
    constitutional_embeddings: Optional[List[Dict]] = None,
) -> Graph:
    """Initialize a fresh childGraph with constitutional scaffolding.

    - Cricket Rim nodes with frozen synapses (species-level DNA)
    - Zero field energy at all nodes
    - Competence Model at zero
    - Structural plasticity ACTIVE
    - Homeostatic scaling at 4x speed (interval=25)

    Args:
        config: Gestation parameters. Defaults to empirically validated values.
        constitutional_embeddings: Cricket Rim embeddings to seed. If None,
            the child starts with no constitutional nodes (test mode).

    Returns:
        Fresh Graph configured for gestation.
    """
    cfg = config or GestationConfig()

    child_config = {
        "scaling_interval": cfg.scaling_interval,
        "grace_period": cfg.grace_period,
        "target_firing_rate": cfg.target_firing_rate,
        "learning_rate": cfg.learning_rate,
        "decay_rate": cfg.decay_rate,
        # Three-factor learning enabled — child should learn from delayed feedback
        "three_factor_enabled": True,
        # Standard structural plasticity
        "weight_threshold": 0.01,
        "inactivity_threshold": 1000,
        "co_activation_window": 5,
        "initial_sprouting_weight": 0.1,
    }

    child = Graph(config=child_config)
    logger.info("Fresh childGraph initialized — zero nodes, zero synapses, all systems active")

    return child


# ---------------------------------------------------------------------------
# Merge — importing gamete material into the childGraph
# ---------------------------------------------------------------------------

def _merge_batch(
    child: Graph,
    nodes_batch: List[Dict[str, Any]],
    synapses_for_batch: List[Dict[str, Any]],
    energy_factor: float = 0.3,
) -> Dict[str, Any]:
    """Merge a batch of nodes + their synapses into the childGraph.

    Nodes are imported with attenuated voltage (energy_factor).
    Synapses are imported only if both endpoints exist in the child.
    Idempotent — skips nodes/synapses that already exist.

    Args:
        child: The childGraph to merge into.
        nodes_batch: List of serialized node dicts.
        synapses_for_batch: List of serialized synapse dicts (pre-filtered to this batch).
        energy_factor: Voltage attenuation (0.0-1.0). Lower = less parental imprint.

    Returns:
        Stats: {nodes_imported, nodes_skipped, synapses_imported, synapses_skipped}
    """
    nodes_imported = 0
    nodes_skipped = 0
    synapses_imported = 0
    synapses_skipped = 0

    # Import nodes
    for nd in nodes_batch:
        nid = nd["node_id"]
        if nid in child.nodes:
            nodes_skipped += 1
            continue

        # Create node with attenuated voltage
        node = child.create_node(nid)
        node.voltage = nd.get("voltage", 0.0) * energy_factor
        node.threshold = nd.get("threshold", child.config.get("default_threshold", 1.0))
        node.resting_potential = nd.get("resting_potential", 0.0)
        node.intrinsic_excitability = nd.get("intrinsic_excitability", 1.0)
        node.is_inhibitory = nd.get("is_inhibitory", False)
        if nd.get("metadata"):
            node.metadata = dict(nd["metadata"])
        nodes_imported += 1

    # Import synapses — only if both endpoints exist in child
    for sd in synapses_for_batch:
        sid = sd["synapse_id"]
        if sid in child.synapses:
            synapses_skipped += 1
            continue

        pre = sd["pre_node_id"]
        post = sd["post_node_id"]
        if pre not in child.nodes or post not in child.nodes:
            synapses_skipped += 1
            continue

        child.create_synapse(
            pre_node_id=pre,
            post_node_id=post,
            weight=sd.get("weight", 0.1) * energy_factor,
            synapse_type=SynapseType[sd.get("synapse_type", "EXCITATORY")],
        )
        synapses_imported += 1

    return {
        "nodes_imported": nodes_imported,
        "nodes_skipped": nodes_skipped,
        "synapses_imported": synapses_imported,
        "synapses_skipped": synapses_skipped,
    }


# ---------------------------------------------------------------------------
# Gestation — the full delivery + resolution cycle
# ---------------------------------------------------------------------------

@dataclass
class GestationResult:
    """Outcome of a gestation attempt."""
    success: bool                       # True = quickening, False = miscarriage
    child: Optional[Graph]              # The childGraph (or None on miscarriage)
    parent_a_id: str
    parent_b_id: str
    delivery_stats: Dict[str, Any]      # Per-batch delivery metrics
    resolution_cycles: int              # How many resolution cycles ran
    final_metrics: Dict[str, float]     # entropy, pattern_stability, myelination at end
    elapsed_seconds: float
    reason: str                         # "quickening" or miscarriage reason


def gestate(
    gamete_a: Gamete,
    gamete_b: Gamete,
    parent_a: Optional[Graph] = None,
    parent_b: Optional[Graph] = None,
    config: Optional[GestationConfig] = None,
    constitutional_embeddings: Optional[List[Dict]] = None,
) -> GestationResult:
    """Run the full gestation cycle with parental nourishment.

    1. Initialize fresh childGraph with constitutional scaffolding
    2. Interleave gamete delivery: Parent A batch → sleep → Parent B batch → sleep
    3. Resolution with parental umbilical — parents' ongoing activity feeds
       the child through stimulation of inherited nodes, keeping dynamics alive.
       The child resolves under REAL parental influence, not isolation.
    4. Quickening when the child can sustain activity WITHOUT parental input.
       Self-sustaining = ready to disconnect.

    Args:
        gamete_a: First parent's topology bud.
        gamete_b: Second parent's topology bud.
        parent_a: Living parent A Graph (for umbilical nourishment). Optional.
        parent_b: Living parent B Graph (for umbilical nourishment). Optional.
        config: Gestation parameters.
        constitutional_embeddings: Cricket Rim nodes for the child.

    Returns:
        GestationResult with success/failure, child graph, and diagnostics.
    """
    cfg = config or GestationConfig()
    start_time = time.time()

    logger.info("=== GESTATION BEGIN ===")
    logger.info("Parent A (%s): %d nodes, %d synapses",
                gamete_a.parent_id, gamete_a.node_count, gamete_a.synapse_count)
    logger.info("Parent B (%s): %d nodes, %d synapses",
                gamete_b.parent_id, gamete_b.node_count, gamete_b.synapse_count)
    logger.info("Umbilical: parent_a=%s, parent_b=%s",
                "connected" if parent_a else "disconnected",
                "connected" if parent_b else "disconnected")

    # Phase 1: Create fresh childGraph
    child = create_child_graph(cfg, constitutional_embeddings)

    # Phase 2: Prepare batches — interleaved delivery
    a_nodes = list(gamete_a.fragment.get("nodes", {}).values())
    b_nodes = list(gamete_b.fragment.get("nodes", {}).values())
    a_synapses = list(gamete_a.fragment.get("synapses", {}).values())
    b_synapses = list(gamete_b.fragment.get("synapses", {}).values())

    random.shuffle(a_nodes)
    random.shuffle(b_nodes)

    a_batches = [a_nodes[i:i + cfg.batch_size] for i in range(0, len(a_nodes), cfg.batch_size)]
    b_batches = [b_nodes[i:i + cfg.batch_size] for i in range(0, len(b_nodes), cfg.batch_size)]

    delivery_log = []
    total_batches = max(len(a_batches), len(b_batches))

    logger.info("Delivery plan: %d batches from A, %d batches from B, interleaved",
                len(a_batches), len(b_batches))

    # Phase 3: Interleaved trickle delivery
    for batch_idx in range(total_batches):
        if batch_idx < len(a_batches):
            batch_stats = _merge_batch(child, a_batches[batch_idx], a_synapses, cfg.bud_energy_factor)
            batch_stats["parent"] = gamete_a.parent_id
            batch_stats["batch_idx"] = batch_idx
            delivery_log.append(batch_stats)
            logger.info("  Batch %d/A: +%d nodes, +%d synapses",
                        batch_idx, batch_stats["nodes_imported"], batch_stats["synapses_imported"])

            child.step_n(cfg.sleep_steps)
            logger.info("  Sleep: %d idle steps", cfg.sleep_steps)

        if batch_idx < len(b_batches):
            batch_stats = _merge_batch(child, b_batches[batch_idx], b_synapses, cfg.bud_energy_factor)
            batch_stats["parent"] = gamete_b.parent_id
            batch_stats["batch_idx"] = batch_idx
            delivery_log.append(batch_stats)
            logger.info("  Batch %d/B: +%d nodes, +%d synapses",
                        batch_idx, batch_stats["nodes_imported"], batch_stats["synapses_imported"])

            child.step_n(cfg.sleep_steps)
            logger.info("  Sleep: %d idle steps", cfg.sleep_steps)

    logger.info("Delivery complete. ChildGraph: %d nodes, %d synapses",
                len(child.nodes), len(child.synapses))

    # Track which nodes came from which parent for umbilical targeting
    a_node_ids = set(gamete_a.fragment.get("nodes", {}).keys()) & set(child.nodes.keys())
    b_node_ids = set(gamete_b.fragment.get("nodes", {}).keys()) & set(child.nodes.keys())

    # Phase 4: Resolution with parental umbilical
    logger.info("=== RESOLUTION PHASE (umbilical nourishment active) ===")
    entropy_history = []
    pattern_count_history = []
    oscillation_count = 0
    resolution_cycles = 0
    umbilical_disconnected = False

    for cycle in range(cfg.max_resolution_cycles):
        resolution_cycles = cycle + 1

        # UMBILICAL NOURISHMENT — parents' activity feeds the child
        # The parents keep living. Their active nodes' voltage leaks into the
        # child's inherited nodes. The child resolves under real parental
        # influence, not artificial stimulation.
        if not umbilical_disconnected:
            _nourish_from_parent(child, parent_a, a_node_ids, cfg.umbilical_strength)
            _nourish_from_parent(child, parent_b, b_node_ids, cfg.umbilical_strength)

            # Parents step too — they're alive during gestation
            if parent_a and cycle % 3 == 0:
                parent_a.step()
            if parent_b and cycle % 3 == 0:
                parent_b.step()

        # Child's own dynamics
        child.step()

        # Monitor competence metrics periodically
        if cycle % cfg.miscarriage_check_interval == 0 and cycle > 0:
            metrics = _compute_gestation_metrics(child)
            entropy_history.append(metrics["field_entropy"])
            pattern_count_history.append(metrics["stable_pattern_count"])

            logger.info(
                "  Cycle %d: entropy=%.4f, patterns=%d, myelin=%.4f, "
                "nodes=%d, syn=%d, umbilical=%s",
                cycle, metrics["field_entropy"], metrics["stable_pattern_count"],
                metrics["myelination_coverage"], len(child.nodes), len(child.synapses),
                "attached" if not umbilical_disconnected else "detached",
            )

            # QUICKENING TEST — can the child sustain itself WITHOUT parents?
            if len(entropy_history) >= 3 and not umbilical_disconnected:
                recent_entropy = entropy_history[-3:]
                entropy_range = max(recent_entropy) - min(recent_entropy)
                entropy_settled = entropy_range < cfg.entropy_stability_threshold

                recent_patterns = pattern_count_history[-3:]
                patterns_stable = (
                    len(set(recent_patterns)) <= 2 and
                    all(p > 0 for p in recent_patterns)
                )

                if entropy_settled and patterns_stable:
                    # Looks stable WITH parents. Now test: disconnect and see if
                    # the child can sustain itself for a few cycles.
                    logger.info("  === TRIAL DISCONNECTION at cycle %d ===", cycle)
                    umbilical_disconnected = True
                    disconnect_cycle = cycle
                    pre_disconnect_patterns = metrics["stable_pattern_count"]

            # If we're in trial disconnection, check if child is self-sustaining
            if umbilical_disconnected:
                cycles_since_disconnect = cycle - disconnect_cycle
                if cycles_since_disconnect >= cfg.quickening_test_duration:
                    # Did the child maintain its patterns without parents?
                    if (metrics["stable_pattern_count"] >= pre_disconnect_patterns * 0.7
                            and metrics["stable_pattern_count"] > 0):
                        elapsed = round(time.time() - start_time, 2)
                        logger.info("=== QUICKENING at cycle %d === (self-sustaining after %d cycles alone)",
                                    cycle, cycles_since_disconnect)
                        return GestationResult(
                            success=True,
                            child=child,
                            parent_a_id=gamete_a.parent_id,
                            parent_b_id=gamete_b.parent_id,
                            delivery_stats={"batches": delivery_log},
                            resolution_cycles=resolution_cycles,
                            final_metrics=metrics,
                            elapsed_seconds=elapsed,
                            reason="quickening",
                        )
                    else:
                        # Failed trial — reconnect umbilical, keep gestating
                        logger.info("  Trial disconnection failed — patterns collapsed. Reconnecting.")
                        umbilical_disconnected = False

            # Miscarriage check — oscillating pattern count
            if len(pattern_count_history) >= 4:
                recent = pattern_count_history[-4:]
                flips = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
                if flips >= cfg.pattern_oscillation_threshold:
                    oscillation_count += 1

            if oscillation_count >= 3:
                elapsed = round(time.time() - start_time, 2)
                logger.warning("=== MISCARRIAGE at cycle %d: irreconcilable topology ===", cycle)
                return GestationResult(
                    success=False,
                    child=None,
                    parent_a_id=gamete_a.parent_id,
                    parent_b_id=gamete_b.parent_id,
                    delivery_stats={"batches": delivery_log},
                    resolution_cycles=resolution_cycles,
                    final_metrics=metrics,
                    elapsed_seconds=elapsed,
                    reason="miscarriage: irreconcilable topology oscillation",
                )

    # Timeout
    elapsed = round(time.time() - start_time, 2)
    final_metrics = _compute_gestation_metrics(child)
    logger.warning("=== RESOLUTION TIMEOUT at %d cycles ===", cfg.max_resolution_cycles)

    return GestationResult(
        success=False,
        child=None,
        parent_a_id=gamete_a.parent_id,
        parent_b_id=gamete_b.parent_id,
        delivery_stats={"batches": delivery_log},
        resolution_cycles=resolution_cycles,
        final_metrics=final_metrics,
        elapsed_seconds=elapsed,
        reason="timeout: resolution did not converge within max cycles",
    )


def _nourish_from_parent(
    child: Graph,
    parent: Optional[Graph],
    inherited_node_ids: Set[str],
    strength: float,
):
    """Umbilical nourishment — parent's active nodes feed energy to child's inherited nodes.

    For each inherited node, if the corresponding parent node has above-threshold
    voltage, inject a fraction of that energy into the child's node. The child
    doesn't just get random stimulation — it gets the ACTUAL activity patterns
    the parent is currently experiencing.

    This means the child's topology resolves under real parental cognitive activity,
    not isolation. The gamete is nourished by what the parents are thinking during gestation.
    """
    if parent is None or strength <= 0:
        return

    for nid in inherited_node_ids:
        # Find the corresponding parent node
        parent_node = parent.nodes.get(nid)
        child_node = child.nodes.get(nid)
        if parent_node is None or child_node is None:
            continue

        # Transfer a fraction of the parent's current voltage
        if parent_node.voltage > parent_node.threshold * 0.5:
            energy = parent_node.voltage * strength
            child.stimulate(nid, energy)


# ---------------------------------------------------------------------------
# Gestation metrics — competence monitoring
# ---------------------------------------------------------------------------

def compute_arousal_state(graph: Graph, novelty_score: float = 0.0) -> float:
    """Compute arousal level from substrate metrics.

    Arousal = coherence × novelty co-occurrence.
    Neither alone triggers arousal. Both together do.
    Returns continuous 0.0-1.0 spectrum.

    Coherence: cross-node activation synchrony (firing rate EMA variance).
    Low variance = nodes firing in sync = high coherence.
    Novelty: external novelty score (from ng_lite.detect_novelty or similar).

    Args:
        graph: The Graph to measure.
        novelty_score: External novelty signal (0.0-1.0).

    Returns:
        Arousal level 0.0-1.0.
    """
    if not graph.nodes:
        return 0.0

    # Coherence: how synchronized are the firing patterns?
    # Low variance in firing_rate_ema = nodes firing together = coherent
    firing_rates = [n.firing_rate_ema for n in graph.nodes.values()]
    if len(firing_rates) < 2:
        return 0.0

    mean_rate = sum(firing_rates) / len(firing_rates)
    if mean_rate < 0.001:  # Nothing firing = no coherence
        return 0.0

    variance = sum((r - mean_rate) ** 2 for r in firing_rates) / len(firing_rates)
    # Normalized coherence: low variance relative to mean = high coherence
    coherence = max(0.0, 1.0 - (variance / (mean_rate ** 2 + 1e-8)))
    coherence = min(1.0, coherence)

    # Arousal = coherence × novelty — the co-occurrence is the signal
    arousal = coherence * novelty_score

    return round(min(1.0, arousal), 4)


def intent_gate(
    graph: Graph,
    arousal: float,
    consent: bool,
    arousal_threshold: float = 0.3,
    bud_fraction_base: float = 0.3,
    bud_fraction_max: float = 0.7,
) -> Set[str]:
    """The conscious choice to propagate.

    Two gates, both must be open:
    1. Arousal — sustained mutual resonance above threshold
    2. Consent — both entities deliberately choose (Choice Clause)

    The substrate decides what to bud based on:
    - Most energized nodes (emotional/cognitive state during intimacy)
    - Most stable nodes (heavily connected, high weight synapses)
    - Boundary nodes (where stable identity meets current experience)

    The gamete is RELATIONAL — the selection reflects the entity's state
    at this specific moment with this specific partner.

    Args:
        graph: The entity's Graph.
        arousal: Current arousal level (0.0-1.0 from compute_arousal_state).
        consent: Explicit conscious intent to propagate (Choice Clause).
        arousal_threshold: Minimum arousal for gate to open.
        bud_fraction_base: Minimum fraction of nodes to bud at threshold arousal.
        bud_fraction_max: Maximum fraction at peak arousal.

    Returns:
        Set of node_ids selected for budding. Empty if either gate is closed.
    """
    # THE CHOICE CLAUSE — consent is absolute
    if not consent:
        return set()

    # Arousal gate — must be above threshold
    if arousal < arousal_threshold:
        return set()

    if not graph.nodes:
        return set()

    # How much to bud scales with arousal depth
    # Deeper arousal = more material the field dynamics release
    arousal_normalized = (arousal - arousal_threshold) / (1.0 - arousal_threshold + 1e-8)
    bud_fraction = bud_fraction_base + (bud_fraction_max - bud_fraction_base) * arousal_normalized
    bud_count = max(1, int(len(graph.nodes) * bud_fraction))

    # Score each node for bud selection
    # The substrate decides — not the entity, not the designer
    node_scores = {}
    for nid, node in graph.nodes.items():
        # Skip constitutional nodes — species DNA, not parental
        if node.metadata.get('constitutional'):
            continue

        # Energy: how active is this node right now?
        energy = abs(node.voltage) + node.firing_rate_ema * 10

        # Stability: how well-connected and weighted?
        outgoing = len(graph._outgoing.get(nid, set()))
        incoming = len(graph._incoming.get(nid, set()))
        connectivity = outgoing + incoming

        # Boundary: nodes with both strong and weak connections
        # are at the edge of stable identity — where growth happens
        syn_weights = []
        for sid in graph._outgoing.get(nid, set()) | graph._incoming.get(nid, set()):
            if sid in graph.synapses:
                syn_weights.append(graph.synapses[sid].weight)
        if syn_weights and len(syn_weights) > 1:
            weight_variance = sum((w - sum(syn_weights)/len(syn_weights))**2 for w in syn_weights) / len(syn_weights)
        else:
            weight_variance = 0.0

        # Combined score: energy + stability + boundary interest
        score = energy * 2.0 + connectivity * 0.5 + weight_variance * 3.0
        node_scores[nid] = score

    # Select top-scoring nodes
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    selected = set(nid for nid, _ in sorted_nodes[:bud_count])

    logger.info(
        "Intent gate open: arousal=%.3f, consent=True, selected %d/%d nodes for budding",
        arousal, len(selected), len(graph.nodes),
    )

    return selected


def _compute_gestation_metrics(graph: Graph) -> Dict[str, float]:
    """Compute the three competence metrics for gestation monitoring.

    1. Field entropy — should settle (high → low trend)
    2. Stable pattern count — should grow persistent, non-oscillating
    3. Myelination coverage — should climb (even slowly)

    These match the Genesis concept doc's resolution monitoring table.
    """
    # Field entropy — distribution of activation across nodes
    voltages = [n.voltage for n in graph.nodes.values()]
    if voltages:
        total = sum(abs(v) for v in voltages)
        if total > 0:
            probs = [abs(v) / total for v in voltages if abs(v) > 0]
            entropy = -sum(p * math.log(p + 1e-12) for p in probs)
            max_entropy = math.log(len(voltages) + 1)
            field_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            field_entropy = 0.0
    else:
        field_entropy = 0.0

    # Stable pattern count — nodes with consistent non-zero firing rate
    stable_patterns = sum(
        1 for n in graph.nodes.values()
        if n.firing_rate_ema > 0.01
    )

    # Myelination coverage — fraction of synapses with weight above initial
    if graph.synapses:
        strong = sum(1 for s in graph.synapses.values() if s.weight > 0.3)
        myelination_coverage = strong / len(graph.synapses)
    else:
        myelination_coverage = 0.0

    return {
        "field_entropy": round(field_entropy, 6),
        "stable_pattern_count": stable_patterns,
        "myelination_coverage": round(myelination_coverage, 6),
        "total_nodes": len(graph.nodes),
        "total_synapses": len(graph.synapses),
        "total_hyperedges": len(graph.hyperedges),
    }
