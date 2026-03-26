"""Tests for Phase 2.5b: Hyperedge Output Target Learning.

Covers:
- Output targets learned from consistent post-fire node activity
- Window expiry clears tracking
- Max targets cap respected
- Cleanup on HE removal
- Serialization roundtrip preserves learned output_targets
"""

# ---- Changelog ----
# [2026-03-23] Claude Code (Opus 4.6) — Initial test suite for output_target learning
#   What: 7 tests covering the output_target learning rule in step() §5b.
#   Why:  New feature needs regression coverage.
# -------------------

import sys
import os
import json
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuro_foundation import Graph, Hyperedge, ActivationMode


def _make_graph(**overrides):
    """Create a graph with output learning enabled and no decay."""
    config = {
        "default_threshold": 1.0,
        "decay_rate": 1.0,       # No decay — voltages persist
        "refractory_period": 0,  # No refractory — can fire every step
        "he_output_learning_window": 5,
        "he_output_min_co_fires": 3,
        "he_output_max_targets": 5,
        "grace_period": 99999,   # Prevent pruning during test
        "inactivity_threshold": 99999,
    }
    config.update(overrides)
    return Graph(config)


class TestOutputTargetLearning:
    """Nodes that consistently fire after a HE become output targets."""

    def test_output_target_learned_after_threshold(self):
        """A node firing 3 times within window after HE fires → output target."""
        g = _make_graph()
        # Create 3 member nodes and 1 candidate output node
        members = [g.create_node(f"m{i}").node_id for i in range(3)]
        output_node = g.create_node("out").node_id

        he = g.create_hyperedge(set(members), activation_threshold=0.6)
        assert he.output_targets == []

        # Fire all members so HE fires (step T)
        for nid in members:
            g.stimulate(nid, 2.0)
        result = g.step()
        assert he.hyperedge_id in result.fired_hyperedge_ids

        # Fire the candidate output node for 3 subsequent steps
        for _ in range(3):
            g.stimulate(output_node, 2.0)
            g.step()

        assert output_node in he.output_targets

    def test_output_not_learned_below_threshold(self):
        """A node firing only 2 times (below min_co_fires=3) → not learned."""
        g = _make_graph()
        members = [g.create_node(f"m{i}").node_id for i in range(3)]
        output_node = g.create_node("out").node_id

        he = g.create_hyperedge(set(members), activation_threshold=0.6)

        # Fire members so HE fires
        for nid in members:
            g.stimulate(nid, 2.0)
        g.step()

        # Fire candidate only 2 times (below threshold of 3)
        for _ in range(2):
            g.stimulate(output_node, 2.0)
            g.step()

        assert output_node not in he.output_targets

    def test_window_expiry_clears_tracking(self):
        """After window expires, partial counts are discarded."""
        g = _make_graph(he_output_learning_window=3)
        members = [g.create_node(f"m{i}").node_id for i in range(3)]
        output_node = g.create_node("out").node_id

        he = g.create_hyperedge(set(members), activation_threshold=0.6)

        # Fire members so HE fires
        for nid in members:
            g.stimulate(nid, 2.0)
        g.step()  # HE fires here

        # Fire candidate 2 times within window
        for _ in range(2):
            g.stimulate(output_node, 2.0)
            g.step()

        # Run past the window without firing candidate
        for _ in range(3):
            g.step()

        # Now fire candidate again — should not count toward previous HE fire
        g.stimulate(output_node, 2.0)
        g.step()

        assert output_node not in he.output_targets

    def test_max_targets_cap(self):
        """Output targets are capped at he_output_max_targets."""
        g = _make_graph(he_output_max_targets=2, refractory_period=0)
        members = [g.create_node(f"m{i}").node_id for i in range(3)]
        candidates = [g.create_node(f"c{i}").node_id for i in range(4)]

        he = g.create_hyperedge(set(members), activation_threshold=0.6)

        # Fire members so HE fires
        for nid in members:
            g.stimulate(nid, 2.0)
        g.step()

        # Fire all 4 candidates for 3 steps each
        for _ in range(3):
            for nid in candidates:
                g.stimulate(nid, 2.0)
            g.step()

        # Only 2 should be learned (cap)
        assert len(he.output_targets) == 2

    def test_members_excluded_from_output(self):
        """Member nodes are never learned as output targets."""
        g = _make_graph()
        members = [g.create_node(f"m{i}").node_id for i in range(3)]

        he = g.create_hyperedge(set(members), activation_threshold=0.6)

        # Fire all members → HE fires, then keep firing them
        for _ in range(5):
            for nid in members:
                g.stimulate(nid, 2.0)
            g.step()

        # No members should become output targets
        for nid in members:
            assert nid not in he.output_targets

    def test_cleanup_on_he_removal(self):
        """Removing a HE cleans up tracking dicts."""
        g = _make_graph()
        members = [g.create_node(f"m{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(members), activation_threshold=0.6)
        hid = he.hyperedge_id

        # Fire members so HE fires (populates tracking)
        for nid in members:
            g.stimulate(nid, 2.0)
        g.step()

        assert hid in g._he_last_fired_step

        g.remove_hyperedge(hid)

        assert hid not in g._he_last_fired_step
        assert hid not in g._he_output_candidates

    def test_serialization_preserves_learned_targets(self):
        """Learned output_targets survive save/restore cycle."""
        g = _make_graph()
        members = [g.create_node(f"m{i}").node_id for i in range(3)]
        output_node = g.create_node("out").node_id

        he = g.create_hyperedge(set(members), activation_threshold=0.6)
        hid = he.hyperedge_id

        # Learn an output target
        for nid in members:
            g.stimulate(nid, 2.0)
        g.step()
        for _ in range(3):
            g.stimulate(output_node, 2.0)
            g.step()

        assert output_node in he.output_targets

        # Save and restore
        with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as f:
            path = f.name
        try:
            g.checkpoint(path)
            g2 = Graph()
            g2.restore(path)
            he2 = g2.hyperedges[hid]
            assert output_node in he2.output_targets
        finally:
            os.unlink(path)

    def test_event_emitted_on_learning(self):
        """he_output_learned event is emitted when a target is learned."""
        g = _make_graph()
        members = [g.create_node(f"m{i}").node_id for i in range(3)]
        output_node = g.create_node("out").node_id

        he = g.create_hyperedge(set(members), activation_threshold=0.6)

        events = []
        g.register_event_handler("he_output_learned", lambda **kw: events.append(kw))

        # Fire members so HE fires
        for nid in members:
            g.stimulate(nid, 2.0)
        g.step()

        # Fire candidate 3 times
        for _ in range(3):
            g.stimulate(output_node, 2.0)
            g.step()

        assert len(events) == 1
        assert events[0]["target"] == output_node
        assert events[0]["hid"] == he.hyperedge_id
