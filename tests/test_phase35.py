"""
Tests for Phase 3.5: Predictive State Persistence & Validation.

Covers:
    - Phase 3 active_predictions survive checkpoint/restore
    - Phase 2.5 HE-level _active_predictions survive checkpoint/restore
    - Prediction outcomes persist across serialization
    - Synapse confirmation history persists across serialization
    - Novel sequence log and reward history persist
    - Predictions continue working after restore (confirm/error)
    - Validation: expired predictions are dropped on restore
    - Validation: predictions referencing deleted nodes are dropped
    - Validation: predictions referencing deleted hyperedges are dropped
    - Validation: stale synapse confirmation history is dropped
    - Phase 2.5 prediction counter restored (no ID collisions)
    - HE prediction window-fired state restored
    - Version bump to 0.3.5
"""

import math
import tempfile
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_foundation import (
    Graph,
    Prediction,
    PredictionOutcome,
    PredictionState,
    SurpriseEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_predictive_graph(**extra_config) -> Graph:
    """Build a graph with a trained causal chain for prediction tests."""
    config = {
        "prediction_threshold": 3.0,
        "prediction_window": 10,
        "prediction_chain_decay": 0.7,
        "prediction_max_chain_depth": 3,
        "prediction_pre_charge_factor": 0.3,
        "prediction_confirm_bonus": 0.01,
        "prediction_error_penalty": 0.02,
        "prediction_max_active": 1000,
        "default_threshold": 1.0,
        "decay_rate": 1.0,
        "refractory_period": 0,
        "grace_period": 50000,
        "inactivity_threshold": 50000,
        "scaling_interval": 100000,
    }
    config.update(extra_config)
    return Graph(config=config)


def checkpoint_and_restore(g: Graph) -> Graph:
    """Serialize and deserialize a Graph through a temp file."""
    path = tempfile.mktemp(suffix=".json")
    try:
        g.checkpoint(path)
        g2 = Graph()
        g2.restore(path)
        return g2
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test: Phase 3 Active Predictions Persistence
# ---------------------------------------------------------------------------

class TestPhase3PredictionPersistence:
    """Phase 3 synapse-level active_predictions survive checkpoint/restore."""

    def test_active_predictions_restored(self):
        """Active predictions should be present after restore."""
        g = build_predictive_graph()
        g.create_node(node_id="A")
        g.create_node(node_id="B")
        g.create_synapse("A", "B", weight=4.0)

        # Fire A to generate prediction for B
        g.stimulate("A", 2.0)
        g.step()

        preds_before = g.get_predictions()
        assert len(preds_before) > 0, "Prediction should be generated"

        g2 = checkpoint_and_restore(g)

        preds_after = g2.get_predictions()
        assert len(preds_after) == len(preds_before)
        assert preds_after[0].source_node_id == "A"
        assert preds_after[0].target_node_id == "B"

    def test_prediction_fields_preserved(self):
        """All Prediction fields should roundtrip through serialization."""
        g = build_predictive_graph()
        g.create_node(node_id="X")
        g.create_node(node_id="Y")
        g.create_synapse("X", "Y", weight=4.0)

        g.stimulate("X", 2.0)
        g.step()

        orig = g.get_predictions()[0]
        g2 = checkpoint_and_restore(g)
        restored = g2.get_predictions()[0]

        assert restored.prediction_id == orig.prediction_id
        assert restored.strength == orig.strength
        assert restored.confidence == orig.confidence
        assert restored.created_at == orig.created_at
        assert restored.expires_at == orig.expires_at
        assert restored.chain_depth == orig.chain_depth
        assert restored.pre_charge_applied == orig.pre_charge_applied

    def test_predictions_still_confirmable_after_restore(self):
        """Restored predictions can still be confirmed when target fires."""
        g = build_predictive_graph()
        g.create_node(node_id="A")
        g.create_node(node_id="B")
        g.create_synapse("A", "B", weight=4.0)

        g.stimulate("A", 2.0)
        g.step()
        assert len(g.get_predictions()) > 0

        g2 = checkpoint_and_restore(g)

        # Fire B → should confirm the restored prediction
        confirmed_events = []
        g2.register_event_handler("prediction_confirmed",
                                  lambda **kw: confirmed_events.append(kw))
        g2.stimulate("B", 2.0)
        g2.step()

        assert len(confirmed_events) >= 1

    def test_predictions_still_error_after_restore(self):
        """Restored predictions can still expire and trigger error events."""
        g = build_predictive_graph(prediction_window=3)
        g.create_node(node_id="A")
        g.create_node(node_id="B")
        g.create_synapse("A", "B", weight=4.0)

        g.stimulate("A", 2.0)
        g.step()
        assert len(g.get_predictions()) > 0

        g2 = checkpoint_and_restore(g)

        error_events = []
        g2.register_event_handler("prediction_error",
                                  lambda **kw: error_events.append(kw))
        # Let the window expire without firing B
        g2.step_n(5)

        assert len(error_events) >= 1


# ---------------------------------------------------------------------------
# Test: Phase 2.5 HE-Level Predictions Persistence
# ---------------------------------------------------------------------------

class TestPhase25PredictionPersistence:
    """Phase 2.5 hyperedge-level _active_predictions survive checkpoint."""

    def test_he_predictions_restored(self):
        """HE-level predictions should be present after restore."""
        g = build_predictive_graph(prediction_window=10)
        ids = [g.create_node(f"m{i}").node_id for i in range(3)]
        out = g.create_node("out")
        he = g.create_hyperedge(
            set(ids), activation_threshold=0.5,
            output_targets=[out.node_id],
        )

        # Fire all members → HE fires → prediction created
        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()

        he_preds_before = g.get_active_predictions()
        assert len(he_preds_before) > 0

        g2 = checkpoint_and_restore(g)

        he_preds_after = g2.get_active_predictions()
        assert len(he_preds_after) == len(he_preds_before)
        pid = list(he_preds_after.keys())[0]
        assert he_preds_after[pid].hyperedge_id == he.hyperedge_id
        assert out.node_id in he_preds_after[pid].predicted_targets

    def test_he_predictions_confirmable_after_restore(self):
        """HE predictions can be confirmed after checkpoint/restore."""
        g = build_predictive_graph(prediction_window=10)
        ids = [g.create_node(f"m{i}").node_id for i in range(3)]
        out = g.create_node("out")
        g.create_hyperedge(
            set(ids), activation_threshold=0.5,
            output_targets=[out.node_id],
        )

        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()

        g2 = checkpoint_and_restore(g)

        # Fire the output target → should confirm HE prediction
        g2.stimulate(out.node_id, 2.0)
        r = g2.step()
        assert r.predictions_confirmed >= 1

    def test_he_prediction_counter_restored(self):
        """The HE prediction counter should not reset, preventing ID collisions."""
        g = build_predictive_graph(prediction_window=10, refractory_period=0)
        ids = [g.create_node(f"m{i}").node_id for i in range(3)]
        out = g.create_node("out")
        he = g.create_hyperedge(
            set(ids), activation_threshold=0.5,
            output_targets=[out.node_id],
        )
        he.refractory_period = 0

        # Fire twice to increment counter
        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()
        # Confirm first prediction
        g.stimulate(out.node_id, 2.0)
        g.step()

        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()

        counter_before = g._prediction_counter

        g2 = checkpoint_and_restore(g)
        assert g2._prediction_counter == counter_before

    def test_he_prediction_window_fired_restored(self):
        """Window-fired tracking should survive serialization."""
        g = build_predictive_graph(prediction_window=10)
        ids = [g.create_node(f"m{i}").node_id for i in range(3)]
        out = g.create_node("out")
        g.create_hyperedge(
            set(ids), activation_threshold=0.5,
            output_targets=[out.node_id],
            output_weight=0.01,  # Too weak to auto-fire out
        )

        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()

        # Prediction exists but out hasn't fired → window_fired is tracked
        assert len(g._prediction_window_fired) > 0

        g2 = checkpoint_and_restore(g)
        assert len(g2._prediction_window_fired) == len(g._prediction_window_fired)


# ---------------------------------------------------------------------------
# Test: Prediction Support State Persistence
# ---------------------------------------------------------------------------

class TestPredictionSupportStatePersistence:
    """Prediction outcomes, confirmation history, and logs persist."""

    def test_prediction_outcomes_restored(self):
        """Resolved prediction outcomes should survive serialization."""
        g = build_predictive_graph()
        g.create_node(node_id="A")
        g.create_node(node_id="B")
        g.create_synapse("A", "B", weight=4.0)

        g.stimulate("A", 2.0)
        g.step()

        # Confirm the prediction by firing B
        g.stimulate("B", 2.0)
        g.step()

        outcomes_before = len(g._prediction_outcomes)
        assert outcomes_before > 0

        g2 = checkpoint_and_restore(g)
        assert len(g2._prediction_outcomes) == outcomes_before
        assert g2._prediction_outcomes[0].confirmed is True

    def test_confirmation_history_restored(self):
        """Per-synapse confirmation history should survive serialization."""
        g = build_predictive_graph()
        g.create_node(node_id="A")
        g.create_node(node_id="B")
        syn = g.create_synapse("A", "B", weight=4.0)

        # Generate prediction and confirm it
        g.stimulate("A", 2.0)
        g.step()
        g.stimulate("B", 2.0)
        g.step()

        assert len(g._synapse_confirmation_history) > 0

        g2 = checkpoint_and_restore(g)
        assert len(g2._synapse_confirmation_history) > 0
        # Check the actual history has boolean values
        for syn_id, hist in g2._synapse_confirmation_history.items():
            assert all(isinstance(v, bool) for v in hist)

    def test_novel_sequence_log_restored(self):
        """Novel sequence log should survive serialization."""
        g = build_predictive_graph()
        g.create_node(node_id="A")
        g.create_node(node_id="B")
        g.create_synapse("A", "B", weight=4.0)
        # Manually add a log entry
        g._novel_sequence_log.append({
            "source": "A", "firing_nodes": ["C"], "timestep": 5,
        })
        g._total_novel_sequences = 1

        g2 = checkpoint_and_restore(g)
        assert len(g2._novel_sequence_log) == 1
        assert g2._novel_sequence_log[0]["source"] == "A"

    def test_reward_history_restored(self):
        """Reward history should survive serialization."""
        g = build_predictive_graph()
        g.create_node(node_id="A")
        g._reward_history.append({
            "strength": 1.0, "timestep": 10, "scope_size": None,
        })
        g._total_rewards_injected = 1

        g2 = checkpoint_and_restore(g)
        assert len(g2._reward_history) == 1
        assert g2._reward_history[0]["strength"] == 1.0


# ---------------------------------------------------------------------------
# Test: Validation on Restore
# ---------------------------------------------------------------------------

class TestPredictionValidation:
    """Predictions referencing stale state are dropped on restore."""

    def test_expired_phase3_predictions_dropped(self):
        """Phase 3 predictions past their expires_at are not restored."""
        g = build_predictive_graph(prediction_window=2)
        g.create_node(node_id="A")
        g.create_node(node_id="B")
        g.create_synapse("A", "B", weight=4.0)

        g.stimulate("A", 2.0)
        g.step()
        assert len(g.get_predictions()) > 0

        # Advance past the prediction window
        g.step_n(5)

        # Predictions should have expired by now
        g2 = checkpoint_and_restore(g)
        assert len(g2.get_predictions()) == 0

    def test_deleted_node_phase3_predictions_dropped(self):
        """Predictions referencing removed target nodes are not restored."""
        g = build_predictive_graph()
        g.create_node(node_id="A")
        g.create_node(node_id="B")
        g.create_synapse("A", "B", weight=4.0)

        g.stimulate("A", 2.0)
        g.step()
        assert len(g.get_predictions()) > 0

        # Save checkpoint data manually, then modify it to remove node B
        path = tempfile.mktemp(suffix=".json")
        try:
            g.checkpoint(path)

            import json
            with open(path, "r") as f:
                data = json.load(f)

            # Remove node B
            data["nodes"].pop("B", None)
            # Remove any synapses involving B
            data["synapses"] = {
                k: v for k, v in data["synapses"].items()
                if v.get("post_node_id") != "B" and v.get("pre_node_id") != "B"
            }

            with open(path, "w") as f:
                json.dump(data, f)

            g2 = Graph()
            g2.restore(path)

            # Predictions targeting deleted node B should be dropped
            assert len(g2.get_predictions()) == 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_expired_he_predictions_dropped(self):
        """HE predictions past their window are not restored."""
        g = build_predictive_graph(prediction_window=2)
        ids = [g.create_node(f"m{i}").node_id for i in range(3)]
        out = g.create_node("out")
        g.create_hyperedge(
            set(ids), activation_threshold=0.5,
            output_targets=[out.node_id],
        )

        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()
        assert len(g.get_active_predictions()) > 0

        # Advance past window
        g.step_n(5)

        g2 = checkpoint_and_restore(g)
        assert len(g2.get_active_predictions()) == 0

    def test_deleted_he_predictions_dropped(self):
        """HE predictions referencing removed hyperedges are dropped."""
        g = build_predictive_graph(prediction_window=10)
        ids = [g.create_node(f"m{i}").node_id for i in range(3)]
        out = g.create_node("out")
        he = g.create_hyperedge(
            set(ids), activation_threshold=0.5,
            output_targets=[out.node_id],
        )

        for nid in ids:
            g.stimulate(nid, 2.0)
        g.step()
        assert len(g.get_active_predictions()) > 0

        path = tempfile.mktemp(suffix=".json")
        try:
            g.checkpoint(path)

            import json
            with open(path, "r") as f:
                data = json.load(f)

            # Remove the hyperedge
            data["hyperedges"] = {}

            with open(path, "w") as f:
                json.dump(data, f)

            g2 = Graph()
            g2.restore(path)
            assert len(g2.get_active_predictions()) == 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_stale_confirmation_history_dropped(self):
        """Confirmation history for deleted synapses is dropped."""
        g = build_predictive_graph()
        g.create_node(node_id="A")
        g.create_node(node_id="B")
        syn = g.create_synapse("A", "B", weight=4.0)

        g.stimulate("A", 2.0)
        g.step()
        g.stimulate("B", 2.0)
        g.step()
        assert len(g._synapse_confirmation_history) > 0

        path = tempfile.mktemp(suffix=".json")
        try:
            g.checkpoint(path)

            import json
            with open(path, "r") as f:
                data = json.load(f)

            # Remove the synapse
            data["synapses"] = {}

            with open(path, "w") as f:
                json.dump(data, f)

            g2 = Graph()
            g2.restore(path)
            assert len(g2._synapse_confirmation_history) == 0
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ---------------------------------------------------------------------------
# Test: Version and Integration
# ---------------------------------------------------------------------------

class TestVersionAndIntegration:
    """Checkpoint version and end-to-end roundtrip tests."""

    def test_version_bump(self):
        """Checkpoint version should be 0.3.5."""
        g = build_predictive_graph()
        g.create_node(node_id="A")
        data = g._serialize_full()
        assert data["version"] == "0.3.5"

    def test_full_roundtrip_with_active_state(self):
        """Full scenario: generate predictions, save mid-flight, restore, continue."""
        g = build_predictive_graph(prediction_window=10)
        # Phase 3 prediction
        g.create_node(node_id="A")
        g.create_node(node_id="B")
        g.create_node(node_id="C")
        g.create_synapse("A", "B", weight=4.0)
        g.create_synapse("B", "C", weight=4.0)

        # Phase 2.5 HE prediction — weak output so out doesn't auto-fire
        he_members = [g.create_node(f"h{i}").node_id for i in range(3)]
        out = g.create_node("out")
        he = g.create_hyperedge(
            set(he_members), activation_threshold=0.5,
            output_targets=[out.node_id],
            output_weight=0.01,
        )
        he.refractory_period = 0

        # Fire A → prediction for B; fire HE members → prediction for out
        g.stimulate("A", 2.0)
        for nid in he_members:
            g.stimulate(nid, 2.0)
        g.step()

        # Verify active predictions exist
        p3_count = len(g.get_predictions())
        p25_count = len(g.get_active_predictions())
        assert p3_count > 0
        assert p25_count > 0

        # Checkpoint mid-flight
        g2 = checkpoint_and_restore(g)

        # Verify restored counts match
        assert len(g2.get_predictions()) == p3_count
        assert len(g2.get_active_predictions()) == p25_count

        # Confirm Phase 3 prediction by firing B
        g2.stimulate("B", 2.0)
        r = g2.step()
        assert g2._total_predictions_confirmed > 0

        # Confirm Phase 2.5 prediction by firing out
        g2.stimulate("out", 2.0)
        r2 = g2.step()
        assert r2.predictions_confirmed >= 1

    def test_backward_compatible_restore(self):
        """Restoring a checkpoint without prediction state should work cleanly."""
        g = build_predictive_graph()
        g.create_node(node_id="A")
        g.create_node(node_id="B")
        g.create_synapse("A", "B", weight=4.0)

        # Manually create a v0.2.5-style checkpoint (no prediction fields)
        path = tempfile.mktemp(suffix=".json")
        try:
            g.checkpoint(path)

            import json
            with open(path, "r") as f:
                data = json.load(f)

            # Strip all Phase 3.5 keys
            for key in [
                "active_predictions", "prediction_outcomes",
                "synapse_confirmation_history", "novel_sequence_log",
                "reward_history", "he_active_predictions",
                "he_prediction_window_fired", "he_prediction_counter",
            ]:
                data.pop(key, None)
            data["version"] = "0.2.5"

            with open(path, "w") as f:
                json.dump(data, f)

            g2 = Graph()
            g2.restore(path)

            # Should work cleanly with empty prediction state
            assert len(g2.get_predictions()) == 0
            assert len(g2.get_active_predictions()) == 0
            assert len(g2._prediction_outcomes) == 0
            assert len(g2._synapse_confirmation_history) == 0
            assert len(g2._novel_sequence_log) == 0
            assert len(g2._reward_history) == 0

            # System should still function
            g2.stimulate("A", 2.0)
            r = g2.step()
            assert len(r.fired_node_ids) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)
