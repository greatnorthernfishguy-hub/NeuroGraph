"""Tests for Hyperedge activation dynamics.

Covers PRD §2.2.3 and §4 (activation modes, pattern completion basics).
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuro_foundation import Graph, Hyperedge, ActivationMode


class TestHyperedgeCreation:

    def test_create_hyperedge(self):
        g = Graph()
        ids = [g.create_node(f"n{i}").node_id for i in range(5)]
        he = g.create_hyperedge(set(ids), activation_threshold=0.6)
        assert he.hyperedge_id in g.hyperedges
        assert he.member_nodes == set(ids)

    def test_member_weights_default(self):
        g = Graph()
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids))
        for nid in ids:
            assert he.member_weights[nid] == 1.0

    def test_custom_member_weights(self):
        g = Graph()
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        weights = {ids[0]: 2.0, ids[1]: 1.0, ids[2]: 0.5}
        he = g.create_hyperedge(set(ids), member_weights=weights)
        assert he.member_weights[ids[0]] == 2.0

    def test_get_hyperedges_for_node(self):
        g = Graph()
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids))
        result = g.get_hyperedges(ids[0])
        assert len(result) == 1
        assert result[0].hyperedge_id == he.hyperedge_id

    def test_remove_hyperedge(self):
        g = Graph()
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids))
        hid = he.hyperedge_id
        g.remove_hyperedge(hid)
        assert hid not in g.hyperedges

    def test_invalid_member_raises(self):
        g = Graph()
        g.create_node("A")
        with pytest.raises(KeyError):
            g.create_hyperedge({"A", "nonexistent"})


class TestWeightedThresholdActivation:
    """WEIGHTED_THRESHOLD mode (PRD §4.2)."""

    def test_fires_when_enough_members_active(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(5)]
        output = g.create_node("out")
        he = g.create_hyperedge(
            set(ids),
            activation_threshold=0.6,
            output_targets=[output.node_id],
            output_weight=1.0,
        )

        # Fire 4 of 5 members (80% > 60% threshold)
        for nid in ids[:4]:
            g.stimulate(nid, 2.0)
        result = g.step()
        assert he.hyperedge_id in result.fired_hyperedge_ids

    def test_does_not_fire_below_threshold(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(5)]
        he = g.create_hyperedge(set(ids), activation_threshold=0.6)

        # Fire 2 of 5 (40% < 60%)
        for nid in ids[:2]:
            g.stimulate(nid, 2.0)
        result = g.step()
        assert he.hyperedge_id not in result.fired_hyperedge_ids

    def test_weighted_activation_respects_member_weights(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        # n0 has high weight, n1 and n2 have low
        weights = {ids[0]: 10.0, ids[1]: 1.0, ids[2]: 1.0}
        he = g.create_hyperedge(
            set(ids),
            member_weights=weights,
            activation_threshold=0.6,
        )

        # Only fire n0 (weight 10/12 = 83% > 60%)
        g.stimulate(ids[0], 2.0)
        result = g.step()
        assert he.hyperedge_id in result.fired_hyperedge_ids

    def test_output_injection(self):
        """Firing hyperedge should inject current into output targets."""
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        out_node = g.create_node("out")
        he = g.create_hyperedge(
            set(ids),
            activation_threshold=0.5,
            output_targets=[out_node.node_id],
            output_weight=2.0,
        )

        # Fire all members
        for nid in ids:
            g.stimulate(nid, 2.0)

        initial_v = out_node.voltage
        result = g.step()
        # Output node should have received current
        assert he.hyperedge_id in result.fired_hyperedge_ids
        assert out_node.voltage > initial_v or "out" in result.fired_node_ids


class TestKOfNActivation:

    def test_k_of_n_fires(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(5)]
        he = g.create_hyperedge(
            set(ids),
            activation_threshold=0.6,  # 3 of 5
            activation_mode=ActivationMode.K_OF_N,
        )

        # Fire 3 of 5
        for nid in ids[:3]:
            g.stimulate(nid, 2.0)
        result = g.step()
        assert he.hyperedge_id in result.fired_hyperedge_ids


class TestAllOrNoneActivation:

    def test_all_or_none_fires_when_all_active(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(
            set(ids),
            activation_mode=ActivationMode.ALL_OR_NONE,
        )

        for nid in ids:
            g.stimulate(nid, 2.0)
        result = g.step()
        assert he.hyperedge_id in result.fired_hyperedge_ids

    def test_all_or_none_does_not_fire_partial(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(
            set(ids),
            activation_mode=ActivationMode.ALL_OR_NONE,
        )

        # Only fire 2 of 3
        for nid in ids[:2]:
            g.stimulate(nid, 2.0)
        result = g.step()
        assert he.hyperedge_id not in result.fired_hyperedge_ids


class TestGradedActivation:

    def test_graded_proportional(self):
        g = Graph({"default_threshold": 1.0, "decay_rate": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(4)]
        he = g.create_hyperedge(
            set(ids),
            activation_threshold=0.5,
            activation_mode=ActivationMode.GRADED,
        )

        # Fire 2 of 4 → activation = 0.5
        for nid in ids[:2]:
            g.stimulate(nid, 2.0)
        result = g.step()
        assert he.current_activation == pytest.approx(0.5, abs=0.01)
        assert he.hyperedge_id in result.fired_hyperedge_ids


class TestNodeRemovalCascade:
    """Removing a node should update hyperedges."""

    def test_member_removed_from_hyperedge(self):
        g = Graph()
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids))
        g.remove_node(ids[0])
        assert ids[0] not in he.member_nodes
        assert len(he.member_nodes) == 2

    def test_empty_hyperedge_removed(self):
        g = Graph()
        n = g.create_node("solo")
        he = g.create_hyperedge({n.node_id})
        hid = he.hyperedge_id
        g.remove_node(n.node_id)
        assert hid not in g.hyperedges
