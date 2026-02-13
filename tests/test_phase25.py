"""Tests for Phase 2.5 Predictive Infrastructure Enhancements.

Covers:
- Prediction error events: PredictionState lifecycle, SurpriseEvent emission
- Dynamic pattern completion: experience-scaled completion strength
- Cross-level consistency pruning: subsumption archival
- Telemetry: prediction_accuracy, surprise_rate, experience distribution
"""

import sys
import os
import json
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuro_foundation import (
    Graph,
    Hyperedge,
    ActivationMode,
    PredictionState,
    SurpriseEvent,
)


# -----------------------------------------------------------------------
# Prediction Error Events
# -----------------------------------------------------------------------

class TestPredictionCreation:
    """When a hyperedge with output_targets fires, a prediction is created."""

    def test_prediction_created_on_hyperedge_fire(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "prediction_window": 5,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        out = g.create_node("out")
        he = g.create_hyperedge(
            set(ids),
            activation_threshold=0.5,
            output_targets=[out.node_id],
        )

        # Fire the hyperedge
        for nid in ids:
            g.stimulate(nid, 2.0)
        result = g.step()

        assert he.hyperedge_id in result.fired_hyperedge_ids
        preds = g.get_active_predictions()
        assert len(preds) >= 1
        pred = list(preds.values())[0]
        assert pred.hyperedge_id == he.hyperedge_id
        assert out.node_id in pred.predicted_targets
        assert pred.prediction_strength > 0

    def test_no_prediction_without_output_targets(self):
        """Hyperedges with no output_targets don't create predictions."""
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids), activation_threshold=0.5)
        # No output_targets

        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()

        preds = g.get_active_predictions()
        assert len(preds) == 0

    def test_prediction_has_correct_window(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "prediction_window": 7,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        out = g.create_node("out")
        he = g.create_hyperedge(
            set(ids), activation_threshold=0.5,
            output_targets=[out.node_id],
        )

        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()

        pred = list(g.get_active_predictions().values())[0]
        assert pred.prediction_window == 7


class TestPredictionConfirmation:
    """Predictions are confirmed when targets fire within the window."""

    def test_target_fires_within_window_confirms(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "prediction_window": 5,
            "refractory_period": 0,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        out = g.create_node("out")
        he = g.create_hyperedge(
            set(ids), activation_threshold=0.5,
            output_targets=[out.node_id],
            output_weight=2.0,  # Strong enough to fire out
        )

        # Fire hyperedge → creates prediction for 'out'
        for nid in ids:
            g.stimulate(nid, 2.0)
        r1 = g.step()
        assert he.hyperedge_id in r1.fired_hyperedge_ids

        # Stimulate 'out' so it fires
        g.stimulate(out.node_id, 2.0)
        r2 = g.step()
        assert out.node_id in r2.fired_node_ids

        # Run until prediction window expires
        total_confirmed = 0
        for _ in range(10):
            r = g.step()
            total_confirmed += r.predictions_confirmed

        assert total_confirmed >= 1

    def test_confirmed_event_emitted(self):
        confirmed_events = []
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "prediction_window": 3,
            "refractory_period": 0,
        })
        g.register_event_handler("prediction_confirmed",
                                 lambda **kw: confirmed_events.append(kw))

        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        out = g.create_node("out")
        he = g.create_hyperedge(
            set(ids), activation_threshold=0.5,
            output_targets=[out.node_id],
        )

        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()

        g.stimulate(out.node_id, 2.0)
        g.step()

        # Let prediction window expire
        g.step_n(5)

        assert len(confirmed_events) >= 1
        assert confirmed_events[0]["target_node"] == out.node_id


class TestSurpriseDetection:
    """When a predicted target doesn't fire, a SurpriseEvent is emitted."""

    def test_surprise_when_target_doesnt_fire(self):
        surprise_events = []
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "prediction_window": 3,
            "refractory_period": 0,
        })
        g.register_event_handler("surprise",
                                 lambda **kw: surprise_events.append(kw))

        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        out = g.create_node("out")
        he = g.create_hyperedge(
            set(ids), activation_threshold=0.5,
            output_targets=[out.node_id],
            output_weight=0.01,  # Too weak to fire out
        )

        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()

        # Don't stimulate 'out' — let window expire
        g.step_n(5)

        assert len(surprise_events) >= 1
        se = surprise_events[0]["surprise"]
        assert isinstance(se, SurpriseEvent)
        assert se.expected_node == out.node_id
        assert se.hyperedge_id == he.hyperedge_id

    def test_surprise_count_in_step_result(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "prediction_window": 2,
            "refractory_period": 0,
        })

        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        out = g.create_node("out")
        he = g.create_hyperedge(
            set(ids), activation_threshold=0.5,
            output_targets=[out.node_id],
            output_weight=0.01,
        )

        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()

        # Run past window
        total_surprised = 0
        for _ in range(5):
            r = g.step()
            total_surprised += r.predictions_surprised

        assert total_surprised >= 1

    def test_surprise_event_contains_actual_nodes(self):
        """SurpriseEvent includes nodes that did fire during the window."""
        surprise_events = []
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "prediction_window": 3,
            "refractory_period": 0,
        })
        g.register_event_handler("surprise",
                                 lambda **kw: surprise_events.append(kw))

        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        out = g.create_node("out")
        other = g.create_node("other")
        he = g.create_hyperedge(
            set(ids), activation_threshold=0.5,
            output_targets=[out.node_id],
            output_weight=0.01,
        )

        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()

        # Fire 'other' (not 'out') during window
        g.stimulate(other.node_id, 2.0)
        g.step()

        g.step_n(5)

        assert len(surprise_events) >= 1
        se = surprise_events[0]["surprise"]
        assert other.node_id in se.actual_nodes


# -----------------------------------------------------------------------
# Dynamic Pattern Completion
# -----------------------------------------------------------------------

class TestDynamicPatternCompletion:
    """Pattern completion scales with hyperedge experience."""

    def test_new_hyperedge_weak_completion(self):
        """A brand-new hyperedge (activation_count=0) gives no completion."""
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "he_experience_threshold": 100,
            "he_pattern_completion_strength": 0.5,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids), activation_threshold=0.5)
        he.pattern_completion_strength = 0.5
        # activation_count is 0 → learning_factor = 0/100 = 0

        g.stimulate(ids[0], 2.0)
        g.stimulate(ids[1], 2.0)
        g.step()

        # Inactive member should get NO completion (factor = 0)
        v = g.nodes[ids[2]].voltage
        assert v == pytest.approx(0.0, abs=0.01), (
            f"New hyperedge should not complete: v={v}"
        )

    def test_experienced_hyperedge_full_completion(self):
        """An experienced hyperedge (activation_count >= threshold) completes fully."""
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "he_experience_threshold": 100,
            "he_pattern_completion_strength": 0.5,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids), activation_threshold=0.5)
        he.pattern_completion_strength = 0.5
        # Pre-set high activation count
        he.activation_count = 200

        g.stimulate(ids[0], 2.0)
        g.stimulate(ids[1], 2.0)
        g.step()

        # learning_factor = min(1.0, 200/100) = 1.0 → full strength 0.5
        v = g.nodes[ids[2]].voltage
        assert v > 0.3, (
            f"Experienced hyperedge should complete strongly: v={v}"
        )

    def test_partial_experience_scales_linearly(self):
        """activation_count=50 with threshold=100 gives 50% completion."""
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "he_experience_threshold": 100,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]

        # Create two hyperedges: one at 50% experience, one at 100%
        he_half = g.create_hyperedge(set(ids), activation_threshold=0.5)
        he_half.pattern_completion_strength = 0.5
        he_half.activation_count = 50

        # Fire and capture voltage for half-experienced
        g.stimulate(ids[0], 2.0)
        g.stimulate(ids[1], 2.0)
        g.step()
        v_half = g.nodes[ids[2]].voltage

        # Reset for full comparison
        g2 = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "he_experience_threshold": 100,
        })
        ids2 = [g2.create_node(f"n{i}").node_id for i in range(3)]
        he_full = g2.create_hyperedge(set(ids2), activation_threshold=0.5)
        he_full.pattern_completion_strength = 0.5
        he_full.activation_count = 100

        g2.stimulate(ids2[0], 2.0)
        g2.stimulate(ids2[1], 2.0)
        g2.step()
        v_full = g2.nodes[ids2[2]].voltage

        assert v_half < v_full, (
            f"Half experience ({v_half}) should give less completion than full ({v_full})"
        )
        # Half should be roughly 50% of full (within tolerance for decay etc.)
        if v_full > 0:
            ratio = v_half / v_full
            assert 0.3 <= ratio <= 0.7, (
                f"Half/full ratio should be ~0.5, got {ratio}"
            )


# -----------------------------------------------------------------------
# Cross-Level Consistency Pruning
# -----------------------------------------------------------------------

class TestSubsumptionPruning:
    """Redundant lower-level hyperedges are archived, not deleted."""

    def test_exact_match_archives_lower_level(self):
        """Level-0 HE with same members as level-1 gets archived."""
        g = Graph({"default_threshold": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]

        he_l0 = g.create_hyperedge(set(ids))
        # Create a hierarchical HE with exactly the same members
        he_l1 = g.create_hierarchical_hyperedge({he_l0.hyperedge_id})
        # Both have same member_nodes (all in ids)

        assert not he_l0.is_archived

        g.consolidate_hyperedges()

        assert he_l0.is_archived, "Lower-level HE should be archived"
        assert not he_l1.is_archived, "Higher-level HE should remain active"

    def test_archived_hyperedge_preserved_in_metadata(self):
        g = Graph({"default_threshold": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]

        he_l0 = g.create_hyperedge(set(ids))
        he_l1 = g.create_hierarchical_hyperedge({he_l0.hyperedge_id})

        g.consolidate_hyperedges()

        archived = g.get_archived_hyperedges()
        assert he_l0.hyperedge_id in archived
        assert archived[he_l0.hyperedge_id].member_nodes == set(ids)

    def test_archived_hyperedge_doesnt_fire(self):
        """Archived hyperedges are skipped during step evaluation."""
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]

        he_l0 = g.create_hyperedge(set(ids), activation_threshold=0.5)
        he_l1 = g.create_hierarchical_hyperedge({he_l0.hyperedge_id})

        g.consolidate_hyperedges()
        assert he_l0.is_archived

        # Fire all members
        for nid in ids:
            g.stimulate(nid, 2.0)
        result = g.step()

        # Archived HE should NOT be in fired list
        assert he_l0.hyperedge_id not in result.fired_hyperedge_ids
        # But the level-1 should fire
        assert he_l1.hyperedge_id in result.fired_hyperedge_ids

    def test_non_matching_not_archived(self):
        """Different members at different levels → no archival."""
        g = Graph({"default_threshold": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(6)]

        he_l0_a = g.create_hyperedge(set(ids[:3]))
        he_l0_b = g.create_hyperedge(set(ids[3:6]))
        he_l1 = g.create_hierarchical_hyperedge(
            {he_l0_a.hyperedge_id, he_l0_b.hyperedge_id}
        )

        g.consolidate_hyperedges()

        # Neither level-0 should be archived (they have 3 members, level-1 has 6)
        assert not he_l0_a.is_archived
        assert not he_l0_b.is_archived

    def test_child_subsumption(self):
        """A child of a hierarchical HE with subset members gets archived."""
        g = Graph({"default_threshold": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]

        he_l0 = g.create_hyperedge(set(ids))
        he_l1 = g.create_hierarchical_hyperedge({he_l0.hyperedge_id})
        # he_l1 members == he_l0 members (same set), and he_l0 is a child

        g.consolidate_hyperedges()

        assert he_l0.is_archived

    def test_consolidation_return_includes_archived(self):
        """Return value counts both merges and archivals."""
        g = Graph({"default_threshold": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]

        he_l0 = g.create_hyperedge(set(ids))
        he_l1 = g.create_hierarchical_hyperedge({he_l0.hyperedge_id})

        count = g.consolidate_hyperedges()
        assert count >= 1


# -----------------------------------------------------------------------
# Telemetry Updates
# -----------------------------------------------------------------------

class TestPredictionTelemetry:
    """Telemetry includes prediction accuracy and surprise rate."""

    def test_prediction_accuracy_after_confirmations(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "prediction_window": 3,
            "refractory_period": 0,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        out = g.create_node("out")
        he = g.create_hyperedge(
            set(ids), activation_threshold=0.5,
            output_targets=[out.node_id],
        )

        # Fire hyperedge + fire target
        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()
        g.stimulate(out.node_id, 2.0)
        g.step()

        # Let window expire
        g.step_n(5)

        tel = g.get_telemetry()
        assert tel.prediction_accuracy > 0, (
            f"Accuracy should be > 0 after confirmation: {tel.prediction_accuracy}"
        )

    def test_surprise_rate_after_failures(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "prediction_window": 2,
            "refractory_period": 0,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        out = g.create_node("out")
        he = g.create_hyperedge(
            set(ids), activation_threshold=0.5,
            output_targets=[out.node_id],
            output_weight=0.01,
        )

        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()

        g.step_n(5)

        tel = g.get_telemetry()
        assert tel.surprise_rate > 0

    def test_experience_distribution_buckets(self):
        g = Graph({"default_threshold": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(6)]

        he_0 = g.create_hyperedge(set(ids[:3]))  # count = 0
        he_50 = g.create_hyperedge(set(ids[:3]))
        he_50.activation_count = 50  # "10-99" bucket
        he_200 = g.create_hyperedge(set(ids[3:6]))
        he_200.activation_count = 200  # "100+" bucket

        tel = g.get_telemetry()
        dist = tel.hyperedge_experience_distribution
        assert dist["0"] >= 1
        assert dist["10-99"] >= 1
        assert dist["100+"] >= 1

    def test_telemetry_zero_predictions(self):
        """No predictions → accuracy=0, surprise_rate=0."""
        g = Graph({"default_threshold": 1.0})
        g.create_node("n0")
        tel = g.get_telemetry()
        assert tel.prediction_accuracy == 0.0
        assert tel.surprise_rate == 0.0


# -----------------------------------------------------------------------
# Serialization round-trip for Phase 2.5 fields
# -----------------------------------------------------------------------

class TestPhase25Serialization:
    """Phase 2.5 fields survive checkpoint/restore cycle."""

    def test_roundtrip_preserves_prediction_counters(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "prediction_window": 2,
            "refractory_period": 0,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        out = g.create_node("out")
        he = g.create_hyperedge(
            set(ids), activation_threshold=0.5,
            output_targets=[out.node_id],
            output_weight=0.01,
        )

        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()
        g.step_n(5)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            g.checkpoint(path)
            g2 = Graph()
            g2.restore(path)

            assert g2._total_predictions == g._total_predictions
            assert g2._total_confirmed == g._total_confirmed
            assert g2._total_surprised == g._total_surprised
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_archived_hyperedges(self):
        g = Graph({"default_threshold": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he_l0 = g.create_hyperedge(set(ids))
        he_l1 = g.create_hierarchical_hyperedge({he_l0.hyperedge_id})
        g.consolidate_hyperedges()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            g.checkpoint(path)
            g2 = Graph()
            g2.restore(path)

            archived = g2.get_archived_hyperedges()
            assert he_l0.hyperedge_id in archived

            # The original HE in hyperedges dict should have is_archived=True
            he_restored = g2.hyperedges[he_l0.hyperedge_id]
            assert he_restored.is_archived
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_new_hyperedge_fields(self):
        g = Graph({"default_threshold": 1.0})
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids))
        he.recent_activation_ema = 0.42
        he.is_archived = False

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            g.checkpoint(path)
            g2 = Graph()
            g2.restore(path)

            he_r = g2.hyperedges[he.hyperedge_id]
            assert he_r.recent_activation_ema == pytest.approx(0.42)
            assert not he_r.is_archived
        finally:
            os.unlink(path)


# -----------------------------------------------------------------------
# Activation EMA Tracking
# -----------------------------------------------------------------------

class TestActivationEMA:
    """Hyperedge recent_activation_ema tracks firing rate."""

    def test_ema_increases_with_firing(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "refractory_period": 0,
            "prediction_ema_alpha": 0.1,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids), activation_threshold=0.5)
        he.refractory_period = 0

        assert he.recent_activation_ema == 0.0

        # Fire repeatedly
        for _ in range(20):
            for nid in ids:
                g.stimulate(nid, 2.0)
            g.step()

        assert he.recent_activation_ema > 0.0

    def test_ema_decays_without_firing(self):
        g = Graph({
            "default_threshold": 1.0,
            "decay_rate": 1.0,
            "refractory_period": 0,
            "prediction_ema_alpha": 0.1,
        })
        ids = [g.create_node(f"n{i}").node_id for i in range(3)]
        he = g.create_hyperedge(set(ids), activation_threshold=0.5)
        he.refractory_period = 0

        # Fire to build up EMA
        for _ in range(20):
            for nid in ids:
                g.stimulate(nid, 2.0)
            g.step()
        ema_after_firing = he.recent_activation_ema

        # Stop firing
        for _ in range(50):
            g.step()
        ema_after_silence = he.recent_activation_ema

        assert ema_after_silence < ema_after_firing
