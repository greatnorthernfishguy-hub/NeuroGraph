"""Tests for Phase 2 Hypergraph Engine.

Covers PRD §4:
- Pattern completion (§4.2)
- Hyperedge plasticity: member weights, threshold learning, member evolution (§4.3)
- Hierarchical hyperedges (§4.4)
- Hyperedge discovery (§3.3.2 extended)
- Hyperedge consolidation (§4.3 extended)
- GRADED output scaling
- Activation count tracking
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuro_foundation import (
    Graph,
    Hyperedge,
    ActivationMode,
    HyperedgePlasticityRule,
)


# -----------------------------------------------------------------------
# Pattern Completion
# -----------------------------------------------------------------------

class TestPatternCompletion:
    """When a hyperedge fires from partial activation, inactive members
    get pre-charged (PRD §4.2)."""

    def test_inactive_members_pre_charged(self):
        """4 of 5 members fire → hyperedge fires → 5th member gets current."""
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "he_pattern_completion_strength": 0.3,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(5)]
        he = g.create_hyperedge(
            set(ids),
            activation_threshold=0.6,
        )
        he.pattern_completion_strength = 0.3

        # Fire 4 of 5 (80% > 60% threshold)
        for nid in ids[:4]:
            g.stimulate(nid, 2.0)

        inactive_node = g.nodes[ids[4]]
        v_before = inactive_node.voltage
        result = g.step()

        assert he.hyperedge_id in result.fired_hyperedge_ids
        # The inactive member should have received pattern completion current
        assert inactive_node.voltage > v_before, (
            f"Inactive member should be pre-charged: {v_before} → {inactive_node.voltage}"
        )

    def test_pattern_completion_respects_member_weight(self):
        """Higher-weight inactive members get more completion current."""
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(4)]
        weights = {ids[0]: 1.0, ids[1]: 1.0, ids[2]: 3.0, ids[3]: 1.0}
        he = g.create_hyperedge(
            set(ids),
            member_weights=weights,
            activation_threshold=0.3,  # Low enough that 2/6 weighted activates
        )
        he.pattern_completion_strength = 0.5

        # Fire ids[0] and ids[1] — ids[2] and ids[3] are inactive
        # Weighted activation = (1+1)/(1+1+3+1) = 2/6 = 0.33 > 0.3
        g.stimulate(ids[0], 2.0)
        g.stimulate(ids[1], 2.0)
        g.step()

        v_high = g.nodes[ids[2]].voltage  # weight 3.0
        v_low = g.nodes[ids[3]].voltage   # weight 1.0
        assert v_high > v_low, (
            f"Higher-weight member should get more current: {v_high} vs {v_low}"
        )

    def test_pattern_completion_can_cause_spike(self):
        """Enough completion current can push an inactive member over threshold."""
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(
            set(ids),
            activation_threshold=0.5,
        )
        # Very strong pattern completion
        he.pattern_completion_strength = 2.0

        # Pre-charge the inactive node close to threshold
        g.nodes[ids[2]].voltage = 0.5

        g.stimulate(ids[0], 2.0)
        g.stimulate(ids[1], 2.0)
        r1 = g.step()
        assert he.hyperedge_id in r1.fired_hyperedge_ids
        # ids[2] should be above threshold from completion + pre-charge
        # It will fire on the NEXT step (completion current added this step)
        r2 = g.step()
        # The node should have fired (voltage was 0.5 + 2.0*1.0 = 2.5 > 1.0)
        assert ids[2] in r1.fired_node_ids or ids[2] in r2.fired_node_ids or \
            g.nodes[ids[2]].last_spike_time > 0

    def test_pattern_completion_disabled_when_zero(self):
        """pattern_completion_strength=0 disables completion."""
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids), activation_threshold=0.5)
        he.pattern_completion_strength = 0.0

        g.stimulate(ids[0], 2.0)
        g.stimulate(ids[1], 2.0)
        g.step()

        # Inactive member should get NO completion current (only decay from 0)
        assert g.nodes[ids[2]].voltage == pytest.approx(0.0, abs=0.01)


# -----------------------------------------------------------------------
# GRADED Output Scaling
# -----------------------------------------------------------------------

class TestGradedOutputScaling:
    """GRADED mode scales output strength proportionally to activation level."""

    def test_graded_output_scales_with_activation(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(4)]
        out = g.create_node("out")
        he = g.create_hyperedge(
            set(ids),
            activation_threshold=0.25,
            activation_mode=ActivationMode.GRADED,
            output_targets=[out.node_id],
            output_weight=2.0,
        )

        # Fire 2 of 4 → activation = 0.5 → effective weight = 2.0 * 0.5 = 1.0
        g.stimulate(ids[0], 2.0)
        g.stimulate(ids[1], 2.0)
        v_before = out.voltage
        g.step()
        v_half = out.voltage
        boost_half = v_half - v_before

        # Now fire all 4 → activation = 1.0 → effective weight = 2.0
        # Need to wait for refractory to clear
        g.step_n(3)
        out2 = g.create_node("out2")
        he2 = g.create_hyperedge(
            set(ids),
            activation_threshold=0.25,
            activation_mode=ActivationMode.GRADED,
            output_targets=[out2.node_id],
            output_weight=2.0,
        )
        for nid in ids:
            g.stimulate(nid, 2.0)
        v2_before = out2.voltage
        g.step()
        boost_full = out2.voltage - v2_before

        # Full activation should give stronger output than half
        assert boost_full > boost_half, (
            f"Full activation boost ({boost_full}) should exceed half ({boost_half})"
        )


# -----------------------------------------------------------------------
# Hyperedge Plasticity: Member Weight Adaptation
# -----------------------------------------------------------------------

class TestMemberWeightAdaptation:
    """Consistently active members get higher weight; inactive get lower (§4.3)."""

    def test_active_member_weight_increases(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "refractory_period": 0,
            "he_member_weight_lr": 0.05,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids), activation_threshold=0.5)

        initial_w = he.member_weights[ids[0]]

        # Repeatedly fire only ids[0] and ids[1] (not ids[2])
        for _ in range(10):
            g.stimulate(ids[0], 2.0)
            g.stimulate(ids[1], 2.0)
            g.step()

        # ids[0] was active during firings → weight should increase
        assert he.member_weights[ids[0]] > initial_w

    def test_inactive_member_weight_decreases(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "refractory_period": 0,
            "he_member_weight_lr": 0.05,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids), activation_threshold=0.5)

        initial_w = he.member_weights[ids[2]]

        # Fire ids[0] and ids[1] — ids[2] is always inactive
        for _ in range(10):
            g.stimulate(ids[0], 2.0)
            g.stimulate(ids[1], 2.0)
            g.step()

        # ids[2] was inactive during firings → weight should decrease
        assert he.member_weights[ids[2]] < initial_w


# -----------------------------------------------------------------------
# Hyperedge Plasticity: Threshold Learning
# -----------------------------------------------------------------------

class TestThresholdLearning:
    """Reward → lower threshold; punishment → raise threshold (§4.3)."""

    def test_reward_lowers_threshold(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0, "he_threshold_lr": 0.05})
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids), activation_threshold=0.6)

        # Fire the hyperedge
        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()
        assert he.refractory_remaining > 0  # Just fired

        initial_t = he.activation_threshold
        g.inject_reward(1.0)  # Positive reward
        assert he.activation_threshold < initial_t, (
            f"Reward should lower threshold: {initial_t} → {he.activation_threshold}"
        )

    def test_punishment_raises_threshold(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0, "he_threshold_lr": 0.05})
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids), activation_threshold=0.6)

        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()

        initial_t = he.activation_threshold
        g.inject_reward(-1.0)  # Negative reward
        assert he.activation_threshold > initial_t, (
            f"Punishment should raise threshold: {initial_t} → {he.activation_threshold}"
        )

    def test_threshold_bounds(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0, "he_threshold_lr": 0.5})
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids), activation_threshold=0.6)

        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()

        # Lots of reward
        for _ in range(100):
            g.inject_reward(1.0)
        assert he.activation_threshold >= 0.1, "Threshold should not go below 0.1"

        # Lots of punishment
        he.activation_threshold = 0.6
        he.refractory_remaining = 1  # Pretend it just fired
        for _ in range(100):
            g.inject_reward(-1.0)
        assert he.activation_threshold <= 1.0, "Threshold should not go above 1.0"


# -----------------------------------------------------------------------
# Hyperedge Plasticity: Member Evolution
# -----------------------------------------------------------------------

class TestMemberEvolution:
    """Non-members that consistently co-fire get added as new members (§4.3)."""

    def test_co_firing_node_promoted(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "refractory_period": 0,
            "he_member_evolution_min_co_fires": 3,
            "he_member_evolution_initial_weight": 0.3,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        outsider = g.create_node("outsider")
        he = g.create_hyperedge(set(ids), activation_threshold=0.5)
        he.refractory_period = 0  # Allow firing every step

        assert outsider.node_id not in he.member_nodes

        # Fire all members + outsider repeatedly
        for _ in range(5):
            for nid in ids:
                g.stimulate(nid, 2.0)
            g.stimulate(outsider.node_id, 2.0)
            g.step()

        # Outsider should have been added as a member
        assert outsider.node_id in he.member_nodes, (
            "Outsider should be promoted to member after co-firing"
        )
        # Initial weight is 0.3, may be slightly adjusted by member weight
        # adaptation on the same step it was promoted
        assert 0.2 <= he.member_weights[outsider.node_id] <= 0.5

    def test_non_co_firing_node_not_promoted(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "refractory_period": 0,
            "he_member_evolution_min_co_fires": 5,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        outsider = g.create_node("outsider")
        he = g.create_hyperedge(set(ids), activation_threshold=0.5)

        # Fire members but NOT outsider
        for _ in range(10):
            for nid in ids:
                g.stimulate(nid, 2.0)
            g.step()

        assert outsider.node_id not in he.member_nodes


# -----------------------------------------------------------------------
# Hierarchical Hyperedges
# -----------------------------------------------------------------------

class TestHierarchicalHyperedges:
    """Hyperedges can contain other hyperedges (§4.4)."""

    def test_create_hierarchical(self):
        g = Graph()
        ids_a = [g.create_node(f"a{i}").node_id for i in range(3)]
        ids_b = [g.create_node(f"b{i}").node_id for i in range(3)]

        he_a = g.create_hyperedge(set(ids_a), metadata={"label": "syndrome_A"})
        he_b = g.create_hyperedge(set(ids_b), metadata={"label": "syndrome_B"})

        meta = g.create_hierarchical_hyperedge(
            {he_a.hyperedge_id, he_b.hyperedge_id},
            activation_threshold=0.5,
            metadata={"label": "diagnostic_category"},
        )

        assert meta.level == 1
        assert meta.child_hyperedges == {he_a.hyperedge_id, he_b.hyperedge_id}
        assert meta.member_nodes == set(ids_a) | set(ids_b)

    def test_hierarchical_fires_when_children_active(self):
        """Meta-hyperedge fires when enough of its (combined) members fire."""
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        ids_a = [g.create_node(f"a{i}").node_id for i in range(3)]
        ids_b = [g.create_node(f"b{i}").node_id for i in range(3)]

        he_a = g.create_hyperedge(set(ids_a), activation_threshold=0.5)
        he_b = g.create_hyperedge(set(ids_b), activation_threshold=0.5)

        out = g.create_node("out")
        meta = g.create_hierarchical_hyperedge(
            {he_a.hyperedge_id, he_b.hyperedge_id},
            activation_threshold=0.5,
            output_targets=[out.node_id],
        )

        # Fire all of group A and 2 of group B → 5/6 = 83% > 50%
        for nid in ids_a:
            g.stimulate(nid, 2.0)
        for nid in ids_b[:2]:
            g.stimulate(nid, 2.0)

        result = g.step()
        assert meta.hyperedge_id in result.fired_hyperedge_ids

    def test_multi_level_hierarchy(self):
        """Level 2 meta-hyperedge built from level 1 children."""
        g = Graph()
        nodes = [g.create_node(f"n{i}").node_id for i in range(9)]

        he_l0_a = g.create_hyperedge(set(nodes[0:3]))
        he_l0_b = g.create_hyperedge(set(nodes[3:6]))
        he_l0_c = g.create_hyperedge(set(nodes[6:9]))

        he_l1_ab = g.create_hierarchical_hyperedge(
            {he_l0_a.hyperedge_id, he_l0_b.hyperedge_id}
        )
        assert he_l1_ab.level == 1

        he_l2 = g.create_hierarchical_hyperedge(
            {he_l1_ab.hyperedge_id, he_l0_c.hyperedge_id}
        )
        assert he_l2.level == 2
        assert he_l2.member_nodes == set(nodes)

    def test_invalid_child_raises(self):
        g = Graph()
        with pytest.raises(KeyError):
            g.create_hierarchical_hyperedge({"nonexistent_id"})

    def test_levels_processed_bottom_up(self):
        """Level 0 hyperedges fire before level 1 in the same step."""
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(4)]
        he_l0 = g.create_hyperedge(set(ids[:2]), activation_threshold=0.5)
        he_l1 = g.create_hierarchical_hyperedge(
            {he_l0.hyperedge_id},
            activation_threshold=0.5,
        )
        # Add the other 2 nodes to level 1 only
        for nid in ids[2:]:
            he_l1.member_nodes.add(nid)
            he_l1.member_weights[nid] = 1.0

        # Fire all 4 nodes
        for nid in ids:
            g.stimulate(nid, 2.0)
        result = g.step()

        # Both should fire; level 0 first, then level 1
        assert he_l0.hyperedge_id in result.fired_hyperedge_ids
        assert he_l1.hyperedge_id in result.fired_hyperedge_ids


# -----------------------------------------------------------------------
# Hyperedge Discovery
# -----------------------------------------------------------------------

class TestHyperedgeDiscovery:
    """Automatic discovery from co-activation patterns (§3.3.2 extended)."""

    def test_repeated_co_activation_creates_hyperedge(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "refractory_period": 0,
            "he_discovery_min_co_fires": 3,
            "he_discovery_min_nodes": 3,
            "he_discovery_window": 100,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]

        initial_he_count = len(g.hyperedges)

        for _ in range(3):
            for nid in ids:
                g.stimulate(nid, 2.0)
            g.step()
            fired = [nid for nid in ids if g.nodes[nid].last_spike_time == g.timestep]
            g.discover_hyperedges(fired)

        # A new hyperedge should have been created
        assert len(g.hyperedges) > initial_he_count
        # The new hyperedge should contain all 3 nodes
        discovered = [
            he for he in g.hyperedges.values()
            if he.metadata.get("creation_mode") == "discovered"
        ]
        assert len(discovered) >= 1
        assert discovered[0].member_nodes == set(ids)

    def test_insufficient_co_fires_no_discovery(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "refractory_period": 0,
            "he_discovery_min_co_fires": 10,
            "he_discovery_min_nodes": 3,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]

        # Only 2 co-fires (below threshold of 10)
        for _ in range(2):
            for nid in ids:
                g.stimulate(nid, 2.0)
            g.step()
            g.discover_hyperedges(ids)

        discovered = [
            he for he in g.hyperedges.values()
            if he.metadata.get("creation_mode") == "discovered"
        ]
        assert len(discovered) == 0

    def test_too_few_nodes_no_discovery(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "refractory_period": 0,
            "he_discovery_min_co_fires": 2,
            "he_discovery_min_nodes": 3,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(2)]

        for _ in range(5):
            for nid in ids:
                g.stimulate(nid, 2.0)
            g.step()
            g.discover_hyperedges(ids)

        # Only 2 nodes — below min_nodes threshold
        discovered = [
            he for he in g.hyperedges.values()
            if he.metadata.get("creation_mode") == "discovered"
        ]
        assert len(discovered) == 0


# -----------------------------------------------------------------------
# Hyperedge Consolidation
# -----------------------------------------------------------------------

class TestHyperedgeConsolidation:
    """Merge highly overlapping hyperedges (§4.3)."""

    def test_merge_overlapping(self):
        g = Graph({"he_consolidation_overlap": 0.8})
        ids = [g.create_node(f"n{i}").node_id for i in range(5)]

        # Two hyperedges: same 4 members + 1 extra → Jaccard = 4/5 = 0.8
        he_a = g.create_hyperedge(set(ids[:4]))
        he_b = g.create_hyperedge(set(ids[:4]) | {ids[4]})

        initial_count = len(g.hyperedges)
        merged = g.consolidate_hyperedges()

        assert merged == 1
        assert len(g.hyperedges) == initial_count - 1

    def test_no_merge_below_threshold(self):
        g = Graph({"he_consolidation_overlap": 0.8})
        ids = [g.create_node(f"n{i}").node_id for i in range(6)]

        # Two hyperedges: {0,1,2} and {2,3,4} → intersection={2}, union={0,1,2,3,4}
        # Jaccard = 1/5 = 0.2 — well below 0.8
        he_a = g.create_hyperedge(set(ids[:3]))
        he_b = g.create_hyperedge(set(ids[2:5]))

        merged = g.consolidate_hyperedges()
        assert merged == 0

    def test_merged_hyperedge_has_union_members(self):
        """After merge, surviving hyperedge has the union of both member sets."""
        g = Graph({"he_consolidation_overlap": 0.6})
        ids = [g.create_node(f"n{i}").node_id for i in range(5)]

        # {0,1,2,3} and {1,2,3,4}: Jaccard = 3/5 = 0.6
        he_a = g.create_hyperedge(set(ids[:4]))
        he_b = g.create_hyperedge(set(ids[1:5]))

        merged = g.consolidate_hyperedges()
        assert merged == 1
        # Surviving hyperedge should have all 5 members
        remaining = list(g.hyperedges.values())[0]
        assert remaining.member_nodes == set(ids)

    def test_consolidation_keeps_lower_threshold(self):
        g = Graph({"he_consolidation_overlap": 0.8})
        ids = [g.create_node(f"n{i}").node_id for i in range(4)]

        he_a = g.create_hyperedge(set(ids), activation_threshold=0.7)
        he_b = g.create_hyperedge(set(ids), activation_threshold=0.5)

        g.consolidate_hyperedges()
        remaining = list(g.hyperedges.values())[0]
        assert remaining.activation_threshold == 0.5


# -----------------------------------------------------------------------
# Activation Count Tracking
# -----------------------------------------------------------------------

class TestActivationCount:
    """activation_count increments each time the hyperedge fires."""

    def test_count_increments(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "refractory_period": 0,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids), activation_threshold=0.5)
        he.refractory_period = 0

        assert he.activation_count == 0

        for _ in range(5):
            for nid in ids:
                g.stimulate(nid, 2.0)
            g.step()

        assert he.activation_count == 5


# -----------------------------------------------------------------------
# Serialization round-trip for Phase 2 fields
# -----------------------------------------------------------------------

class TestPhase2Serialization:
    """Phase 2 fields survive checkpoint/restore cycle."""

    def test_roundtrip_preserves_phase2_fields(self):
        import json
        import tempfile

        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(5)]
        he = g.create_hyperedge(
            set(ids[:3]),
            activation_threshold=0.5,
        )
        he.activation_count = 42
        he.pattern_completion_strength = 0.7
        he.level = 0

        # Create a hierarchical one
        he2 = g.create_hyperedge(set(ids[2:5]))
        meta = g.create_hierarchical_hyperedge(
            {he.hyperedge_id, he2.hyperedge_id},
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        import os
        try:
            g.checkpoint(path)
            g2 = Graph()
            g2.restore(path)

            he_r = g2.hyperedges[he.hyperedge_id]
            assert he_r.activation_count == 42
            assert he_r.pattern_completion_strength == pytest.approx(0.7)

            meta_r = g2.hyperedges[meta.hyperedge_id]
            assert meta_r.level == 1
            assert meta_r.child_hyperedges == {he.hyperedge_id, he2.hyperedge_id}

            tel = g2.get_telemetry()
            assert tel.total_hyperedges == 3
        finally:
            os.unlink(path)
