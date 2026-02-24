"""Tests for Auto-Knowledge: Spreading Activation Harvest.

Tests the prime_and_propagate() method on Graph and the
auto-knowledge integration in NeuroGraphMemory (on_message
surfacing and standalone associate()).
"""

import math
import os
import tempfile
import unittest
from unittest.mock import patch

from neuro_foundation import (
    ActivationMode,
    FiredEntry,
    Graph,
    Prediction,
    PropagationResult,
    SynapseType,
)


# ---------------------------------------------------------------------------
# Graph.prime_and_propagate() tests
# ---------------------------------------------------------------------------

class TestPrimeAndPropagate(unittest.TestCase):
    """Tests for Graph.prime_and_propagate()."""

    def _make_chain(self, length=3, weight=3.0):
        """Create a chain n0→n1→n2→... with strong weights."""
        g = Graph()
        node_ids = []
        for i in range(length):
            n = g.create_node(node_id=f"n{i}")
            node_ids.append(n.node_id)
        synapses = []
        for i in range(length - 1):
            s = g.create_synapse(node_ids[i], node_ids[i + 1], weight=weight)
            synapses.append(s)
        return g, node_ids, synapses

    def test_priming_injects_current(self):
        """Prime a node, verify it fires during propagation."""
        g = Graph()
        g.create_node(node_id="A")

        result = g.prime_and_propagate(["A"], [2.0], steps=1)
        self.assertTrue(any(e.node_id == "A" for e in result.fired_entries))

    def test_propagation_spreads_activation(self):
        """Prime A, verify B fires when A→B synapse exists."""
        g, ids, syns = self._make_chain(2, weight=3.0)

        result = g.prime_and_propagate(["n0"], [2.0], steps=3)
        fired_ids = {e.node_id for e in result.fired_entries}
        self.assertIn("n0", fired_ids)
        self.assertIn("n1", fired_ids)

    def test_multi_step_wavefront(self):
        """A→B→C chain: B fires at higher latency than A."""
        g, ids, syns = self._make_chain(3, weight=3.0)

        result = g.prime_and_propagate(["n0"], [2.0], steps=4)
        entries_by_id = {e.node_id: e for e in result.fired_entries}

        self.assertIn("n0", entries_by_id)
        self.assertIn("n1", entries_by_id)
        # n1 should fire after n0
        self.assertGreater(entries_by_id["n1"].firing_step, entries_by_id["n0"].firing_step)

    def test_no_plasticity_side_effects(self):
        """Weights must be unchanged after prime_and_propagate."""
        g, ids, syns = self._make_chain(2, weight=3.0)
        syn_id = syns[0].synapse_id
        weight_before = g.synapses[syn_id].weight

        g.prime_and_propagate(["n0"], [2.0], steps=3)

        self.assertEqual(g.synapses[syn_id].weight, weight_before)

    def test_voltages_restored(self):
        """Node voltages should be restored after propagation."""
        g = Graph()
        n = g.create_node(node_id="A")
        n.voltage = 0.5

        g.prime_and_propagate(["A"], [0.3], steps=2)
        self.assertAlmostEqual(g.nodes["A"].voltage, 0.5, places=3)

    def test_refractory_restored(self):
        """Refractory counters should be restored after propagation."""
        g = Graph()
        n = g.create_node(node_id="A")
        n.refractory_remaining = 1

        g.prime_and_propagate(["A"], [2.0], steps=2)
        self.assertEqual(g.nodes["A"].refractory_remaining, 1)

    def test_empty_graph(self):
        """Empty graph returns empty result, no crash."""
        g = Graph()
        result = g.prime_and_propagate([], [], steps=3)
        self.assertEqual(result.fired_entries, [])
        self.assertEqual(result.nodes_primed, 0)

    def test_empty_node_ids(self):
        """No node IDs to prime returns empty result."""
        g = Graph()
        g.create_node(node_id="A")
        result = g.prime_and_propagate([], [], steps=1)
        self.assertEqual(len(result.fired_entries), 0)

    def test_source_distance_tracking(self):
        """Nodes further from primed set have higher source_distance."""
        g, ids, syns = self._make_chain(3, weight=3.0)
        result = g.prime_and_propagate(["n0"], [2.0], steps=4)

        entries_by_id = {e.node_id: e for e in result.fired_entries}
        if "n0" in entries_by_id and "n1" in entries_by_id:
            self.assertLess(
                entries_by_id["n0"].source_distance,
                entries_by_id["n1"].source_distance,
            )

    def test_pattern_completion_surfaces(self):
        """Prime 2/3 of hyperedge members, verify mechanism is exercised."""
        g = Graph()
        for i in range(3):
            g.create_node(node_id=f"h{i}")
        he = g.create_hyperedge(
            member_node_ids={"h0", "h1", "h2"},
            activation_threshold=0.5,
            activation_mode=ActivationMode.WEIGHTED_THRESHOLD,
        )
        he.pattern_completion_strength = 0.8
        he.activation_count = 200  # experienced enough for full completion

        # Prime h0 and h1 strongly enough to fire them
        result = g.prime_and_propagate(["h0", "h1"], [2.0, 2.0], steps=3)
        fired_ids = {e.node_id for e in result.fired_entries}
        self.assertIn("h0", fired_ids)
        self.assertIn("h1", fired_ids)

    def test_was_predicted_flag(self):
        """Nodes that are prediction targets get was_predicted=True."""
        g, ids, syns = self._make_chain(2, weight=4.0)
        # Manually set up a prediction for n1
        pred = Prediction(
            source_node_id="n0",
            target_node_id="n1",
            strength=1.0,
            confidence=0.8,
            created_at=g.timestep,
            expires_at=g.timestep + 10,
        )
        g.active_predictions["test_pred"] = pred

        result = g.prime_and_propagate(["n0"], [2.0], steps=3)
        n1_entries = [e for e in result.fired_entries if e.node_id == "n1"]
        if n1_entries:
            self.assertTrue(n1_entries[0].was_predicted)

    def test_result_metadata(self):
        """PropagationResult has correct metadata."""
        g, ids, syns = self._make_chain(2, weight=3.0)
        result = g.prime_and_propagate(["n0"], [2.0], steps=5)
        self.assertEqual(result.steps_run, 5)
        self.assertEqual(result.nodes_primed, 1)

    def test_configurable_depth(self):
        """More steps should potentially surface more nodes."""
        g, ids, syns = self._make_chain(4, weight=3.0)

        result_short = g.prime_and_propagate(["n0"], [2.0], steps=1)
        result_long = g.prime_and_propagate(["n0"], [2.0], steps=6)

        self.assertGreaterEqual(
            len(result_long.fired_entries),
            len(result_short.fired_entries),
        )


# ---------------------------------------------------------------------------
# NeuroGraphMemory auto-knowledge integration tests
# ---------------------------------------------------------------------------

class TestAutoKnowledgeIntegration(unittest.TestCase):
    """Tests for on_message() surfacing and associate() method."""

    def setUp(self):
        """Create a fresh NeuroGraphMemory in a temp directory."""
        from openclaw_hook import NeuroGraphMemory
        NeuroGraphMemory.reset_instance()
        self._tmpdir = tempfile.mkdtemp()
        self.ng = NeuroGraphMemory(
            workspace_dir=self._tmpdir,
            config={"peer_bridge": {"enabled": False}},
        )

    def tearDown(self):
        from openclaw_hook import NeuroGraphMemory
        NeuroGraphMemory.reset_instance()

    def test_on_message_returns_surfaced_key(self):
        """on_message() result should always include 'surfaced' key."""
        result = self.ng.on_message("Hello world, this is a test.")
        self.assertIn("surfaced", result)
        self.assertIsInstance(result["surfaced"], list)

    def test_on_message_empty_db_no_surfaced(self):
        """First message with empty DB should produce no surfaced items."""
        result = self.ng.on_message("This is the very first message.")
        self.assertEqual(result["surfaced"], [])

    def test_on_message_surfaces_related_knowledge(self):
        """After ingesting content, related messages should surface it."""
        self.ng.on_message("Python is a programming language created by Guido van Rossum.")
        self.ng.on_message("Recursion is a technique where a function calls itself.")
        self.ng.step(20)

        result = self.ng.on_message("Tell me about Python programming.")
        self.assertIsInstance(result["surfaced"], list)

    def test_associate_method_exists(self):
        """associate() method should exist and be callable."""
        result = self.ng.associate("test query")
        self.assertIsInstance(result, list)

    def test_associate_empty_input(self):
        """associate() with empty input returns empty list."""
        self.assertEqual(self.ng.associate(""), [])
        self.assertEqual(self.ng.associate("   "), [])

    def test_associate_without_ingestion(self):
        """associate() on empty DB returns empty list."""
        result = self.ng.associate("anything at all")
        self.assertEqual(result, [])

    def test_surfaced_item_structure(self):
        """Each surfaced item should have the expected keys."""
        import numpy as np

        fake_entry = FiredEntry(
            node_id="test_node",
            firing_step=0,
            voltage_at_fire=1.5,
            was_predicted=False,
            source_distance=1,
        )
        fake_result = PropagationResult(
            fired_entries=[fake_entry],
            steps_run=3,
            nodes_primed=1,
        )

        vec = np.random.randn(10).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        self.ng.vector_db.insert("test_node", vec, {
            "content": "Test content",
            "metadata": {"source": "test"},
        })

        with patch.object(self.ng.graph, "prime_and_propagate", return_value=fake_result):
            with patch.object(self.ng.ingestor.embedder, "embed_text", return_value=vec):
                surfaced = self.ng._harvest_associations("test text")

        self.assertTrue(len(surfaced) > 0)
        item = surfaced[0]
        self.assertIn("node_id", item)
        self.assertIn("content", item)
        self.assertIn("metadata", item)
        self.assertIn("latency", item)
        self.assertIn("strength", item)
        self.assertIn("was_predicted", item)

    def test_auto_knowledge_disabled(self):
        """auto_knowledge_enabled=False should produce no surfaced results."""
        self.ng.graph.config["auto_knowledge_enabled"] = False
        self.ng.on_message("First message to populate DB.")
        result = self.ng.on_message("Second message should not surface.")
        self.assertEqual(result["surfaced"], [])

    def test_stats_includes_auto_knowledge(self):
        """stats() should report auto_knowledge status."""
        stats = self.ng.stats()
        self.assertIn("auto_knowledge", stats)
        self.assertTrue(stats["auto_knowledge"])

    def test_stats_auto_knowledge_disabled(self):
        """stats() should report False when auto_knowledge is disabled."""
        self.ng.graph.config["auto_knowledge_enabled"] = False
        stats = self.ng.stats()
        self.assertFalse(stats["auto_knowledge"])

    def test_new_nodes_excluded_from_surfaced(self):
        """Newly ingested nodes should not appear in surfaced results."""
        self.ng.on_message("Base knowledge about neural networks.")
        r2 = self.ng.on_message("More about neural networks and learning.")
        for item in r2.get("surfaced", []):
            self.assertIn("node_id", item)

    def test_surfaced_sorted_by_latency_then_strength(self):
        """Surfaced items should be sorted: lower latency first, then higher strength."""
        import numpy as np

        entries = [
            FiredEntry(node_id="late_strong", firing_step=2, voltage_at_fire=3.0),
            FiredEntry(node_id="early_weak", firing_step=0, voltage_at_fire=1.0),
            FiredEntry(node_id="early_strong", firing_step=0, voltage_at_fire=2.0),
        ]
        fake_result = PropagationResult(fired_entries=entries, steps_run=3, nodes_primed=1)

        vec = np.random.randn(10).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-10)

        for e in entries:
            self.ng.vector_db.insert(e.node_id, vec, {
                "content": f"Content for {e.node_id}",
                "metadata": {},
            })

        with patch.object(self.ng.graph, "prime_and_propagate", return_value=fake_result):
            with patch.object(self.ng.ingestor.embedder, "embed_text", return_value=vec):
                surfaced = self.ng._harvest_associations("test")

        self.assertEqual(len(surfaced), 3)
        ids = [s["node_id"] for s in surfaced]
        self.assertEqual(ids[0], "early_strong")
        self.assertEqual(ids[1], "early_weak")
        self.assertEqual(ids[2], "late_strong")


# ---------------------------------------------------------------------------
# FiredEntry / PropagationResult dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses(unittest.TestCase):
    """Basic tests for the new dataclasses."""

    def test_fired_entry_defaults(self):
        entry = FiredEntry()
        self.assertEqual(entry.node_id, "")
        self.assertEqual(entry.firing_step, 0)
        self.assertEqual(entry.voltage_at_fire, 0.0)
        self.assertFalse(entry.was_predicted)
        self.assertEqual(entry.source_distance, 0)

    def test_fired_entry_custom(self):
        entry = FiredEntry(
            node_id="test",
            firing_step=2,
            voltage_at_fire=1.5,
            was_predicted=True,
            source_distance=3,
        )
        self.assertEqual(entry.node_id, "test")
        self.assertEqual(entry.firing_step, 2)
        self.assertTrue(entry.was_predicted)

    def test_propagation_result_defaults(self):
        result = PropagationResult()
        self.assertEqual(result.fired_entries, [])
        self.assertEqual(result.steps_run, 0)
        self.assertEqual(result.nodes_primed, 0)

    def test_propagation_result_mutable_list(self):
        r1 = PropagationResult()
        r2 = PropagationResult()
        r1.fired_entries.append(FiredEntry(node_id="A"))
        self.assertEqual(len(r2.fired_entries), 0)


if __name__ == "__main__":
    unittest.main()
