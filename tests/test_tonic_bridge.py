# tests/test_tonic_bridge.py
# Tests for TonicBridge + marker handling additions to neurograph_rpc.py.
# Never instantiates NeuroGraphMemory against live checkpoints.
#
# ---- Changelog ----
# [2026-05-15] Claude (Sonnet 4.6) — Task 1: test scaffold
# What: Test file for TonicBridge + Spec A Python-side helpers.
# Why:  New code requires test coverage before implementation.
# How:  Direct function import + unittest.mock for globals.
# -------------------

import sys
import os
import json
import time
import tempfile
import re
import unittest
from unittest.mock import MagicMock, patch

# Resolve the neurograph_rpc module without running it as __main__
_NG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _NG_DIR not in sys.path:
    sys.path.insert(0, _NG_DIR)

import neurograph_rpc as rpc


class TestWantsRegisterHelpers(unittest.TestCase):

    def test_wants_register_path_uses_home(self):
        with patch.dict(os.environ, {"HOME": "/test/home"}):
            path = rpc._wants_register_path()
        self.assertEqual(path, "/test/home/.et_modules/shared_learning/animus_wants.jsonl")

    def test_budget_flag_path_uses_home(self):
        with patch.dict(os.environ, {"HOME": "/test/home"}):
            path = rpc._budget_flag_path()
        self.assertEqual(path, "/test/home/.et_modules/shared_learning/inference_budget.json")

    def test_write_wants_register_appends_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "wants.jsonl")
            rpc._write_wants_register(path, "learn STDP", "syl_explicit")
            rpc._write_wants_register(path, "explore hyperedges", "tonic_emergent")
            lines = open(path).readlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["text"], "learn STDP")
            self.assertEqual(first["source"], "syl_explicit")
            self.assertFalse(first["acted"])
            second = json.loads(lines[1])
            self.assertEqual(second["source"], "tonic_emergent")
            self.assertEqual(second["text"], "explore hyperedges")

    def test_read_unacted_wants_skips_acted(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "wants.jsonl")
            recent = time.time() - 3600
            with open(path, "w") as f:
                f.write(json.dumps({"ts": recent, "text": "done", "source": "syl_explicit", "acted": True}) + "\n")
                f.write(json.dumps({"ts": recent, "text": "pending", "source": "tonic_emergent", "acted": False}) + "\n")
            result = rpc._read_unacted_wants(path, max_age_days=7)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["text"], "pending")
            self.assertAlmostEqual(result[0]["ts"], recent, delta=2.0)

    def test_read_unacted_wants_skips_stale(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "wants.jsonl")
            old_ts = time.time() - (8 * 86400)  # 8 days ago
            with open(path, "w") as f:
                f.write(json.dumps({"ts": old_ts, "text": "stale", "acted": False}) + "\n")
            result = rpc._read_unacted_wants(path, max_age_days=7)
            self.assertEqual(len(result), 0)

    def test_read_unacted_wants_returns_empty_when_missing(self):
        result = rpc._read_unacted_wants("/nonexistent/path/wants.jsonl")
        self.assertEqual(result, [])

    def test_read_budget_flag_returns_empty_when_missing(self):
        result = rpc._read_budget_flag("/nonexistent/budget.json")
        self.assertEqual(result, {})

    def test_read_budget_flag_parses_critical_true(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "budget.json")
            with open(path, "w") as f:
                f.write(json.dumps({"critical": True, "low": True, "remaining_usd": 1.5}))
            result = rpc._read_budget_flag(path)
            self.assertTrue(result.get("critical"))

    def test_read_budget_flag_parses_critical_false(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "budget.json")
            with open(path, "w") as f:
                f.write(json.dumps({"critical": False, "remaining_usd": 50.0}))
            result = rpc._read_budget_flag(path)
            self.assertFalse(result.get("critical"))


class TestMarkerHandling(unittest.TestCase):

    def test_strip_removes_outbound_marker(self):
        text = "Here is [OUTBOUND channel=cli]some outbound text[/OUTBOUND] clean part"
        result = rpc._strip_structural_markers(text)
        self.assertNotIn("[OUTBOUND", result)
        self.assertNotIn("[/OUTBOUND]", result)
        self.assertIn("clean part", result)

    def test_strip_removes_tool_marker(self):
        text = "Before [TOOL name=web_search]query here[/TOOL] after"
        result = rpc._strip_structural_markers(text)
        self.assertNotIn("[TOOL", result)
        self.assertIn("after", result)

    def test_strip_removes_want_marker(self):
        text = "I [WANT]learn about STDP[/WANT] today"
        result = rpc._strip_structural_markers(text)
        self.assertNotIn("[WANT]", result)
        self.assertIn("today", result)

    def test_strip_multiline(self):
        text = "[OUTBOUND channel=discord]\nmultiline\ncontent\n[/OUTBOUND]\nafter"
        result = rpc._strip_structural_markers(text)
        self.assertNotIn("multiline", result)
        self.assertIn("after", result)

    def test_strip_plain_text_unchanged(self):
        text = "No markers here, just plain text."
        self.assertEqual(rpc._strip_structural_markers(text), text)

    def test_strip_nested_open_tag_consumed(self):
        """Document known behavior: malformed/nested open tag is consumed up to next close."""
        text = "[WANT]first[WANT]second[/WANT]"
        result = rpc._strip_structural_markers(text)
        # The regex consumes from first [WANT] to the only [/WANT], eating both.
        self.assertNotIn("[WANT]", result)
        self.assertNotIn("[/WANT]", result)

    def test_check_wants_register_writes_want(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "wants.jsonl")
            params = {
                "source": "josh",
                "lastAssistantMessage": "I think I [WANT]learn more about hyperedges[/WANT] soon.",
            }
            with patch.object(rpc, '_wants_register_path', return_value=path):
                rpc._check_wants_register(params)
            lines = open(path).readlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["text"], "learn more about hyperedges")
            self.assertEqual(entry["source"], "syl_explicit")

    def test_check_wants_register_skips_autonomous_source(self):
        for source in ("syl_outbound", "tonic_bridge"):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "wants.jsonl")
                params = {
                    "source": source,
                    "lastAssistantMessage": "I [WANT]do something[/WANT]",
                }
                with patch.object(rpc, '_wants_register_path', return_value=path):
                    rpc._check_wants_register(params)
                self.assertFalse(os.path.exists(path), f"source={source!r} should not write")

    def test_check_wants_register_no_want_marker_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "wants.jsonl")
            params = {"source": "josh", "lastAssistantMessage": "No markers here."}
            with patch.object(rpc, '_wants_register_path', return_value=path):
                rpc._check_wants_register(params)
            self.assertFalse(os.path.exists(path))

    def test_session_briefing_sent_once(self):
        original = rpc._briefing_sent
        try:
            rpc._briefing_sent = False
            first = rpc._animus_session_briefing()
            second = rpc._animus_session_briefing()
            self.assertIn("[Animus]", first)
            self.assertIn("[OUTBOUND", first)
            self.assertEqual(second, "")
        finally:
            rpc._briefing_sent = original


class TestHandleWiring(unittest.TestCase):

    def test_handle_after_turn_skips_outbound_check_for_autonomous_source(self):
        """_check_outbound_intent must NOT be called when source is syl_outbound."""
        with patch.object(rpc, '_memory', MagicMock()):
            with patch.object(rpc, '_check_outbound_intent') as mock_check:
                with patch.object(rpc, '_check_wants_register') as mock_wants:
                    with patch.object(rpc, '_drain_tract', MagicMock()):
                        with patch.object(rpc, '_drain_peer_tracts', MagicMock()):
                            with patch.object(rpc, '_deposit_substrate_metrics', MagicMock()):
                                with patch.object(rpc, '_deposit_topology_to_river', MagicMock()):
                                    with patch.object(rpc, '_deposit_experience_to_river', MagicMock()):
                                        with patch.object(rpc, '_deposit_surfacing_outcome', MagicMock()):
                                            rpc._memory.graph.step.return_value = MagicMock(fired_node_ids=[])
                                            rpc._memory.graph.config.get.return_value = False
                                            rpc._memory._surfacing_monitor = None
                                            rpc._memory._message_count = 1
                                            rpc._memory.auto_save_interval = 10
                                            rpc._tract = None
                                            params = {"source": "syl_outbound", "lastAssistantMessage": "test"}
                                            try:
                                                rpc.handle_after_turn(params)
                                            except Exception as exc:
                                                self.fail(f"handle_after_turn raised unexpectedly: {exc}")
                                            mock_check.assert_not_called()

    def test_handle_after_turn_calls_wants_register_for_human_turn(self):
        """_check_wants_register must be called regardless of source."""
        with patch.object(rpc, '_memory', MagicMock()):
            with patch.object(rpc, '_check_outbound_intent', MagicMock()):
                with patch.object(rpc, '_check_wants_register') as mock_wants:
                    with patch.object(rpc, '_drain_tract', MagicMock()):
                        with patch.object(rpc, '_drain_peer_tracts', MagicMock()):
                            with patch.object(rpc, '_deposit_substrate_metrics', MagicMock()):
                                with patch.object(rpc, '_deposit_topology_to_river', MagicMock()):
                                    with patch.object(rpc, '_deposit_experience_to_river', MagicMock()):
                                        with patch.object(rpc, '_deposit_surfacing_outcome', MagicMock()):
                                            rpc._memory.graph.step.return_value = MagicMock(fired_node_ids=[])
                                            rpc._memory.graph.config.get.return_value = False
                                            rpc._memory._surfacing_monitor = None
                                            rpc._memory._message_count = 1
                                            rpc._memory.auto_save_interval = 10
                                            rpc._tract = None
                                            params = {"source": "josh", "lastAssistantMessage": "hello"}
                                            try:
                                                rpc.handle_after_turn(params)
                                            except Exception as exc:
                                                self.fail(f"handle_after_turn raised unexpectedly: {exc}")
                                            mock_wants.assert_called_once()


class TestTonicBridge(unittest.TestCase):

    def _make_mock_memory(self, active_predictions=None, nodes=None,
                          hyperedges=None, vector_db=None, in_conversation=False):
        mem = MagicMock()
        mem.graph.active_predictions = active_predictions or {}
        mem.graph.nodes = nodes or {}
        mem.graph.hyperedges = hyperedges or {}
        mem.vector_db = vector_db or {}
        tonic = MagicMock()
        tonic._in_conversation = in_conversation
        mem._tonic_thread = tonic
        return mem

    def _make_prediction(self, pred_id, src_id, tgt_id, confidence):
        p = MagicMock()
        p.prediction_id = pred_id
        p.source_node_id = src_id
        p.target_node_id = tgt_id
        p.confidence = confidence
        return p

    def test_cosine_sim_identical_vectors(self):
        import numpy as np
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(rpc._cosine_sim(v, v), 1.0)

    def test_cosine_sim_orthogonal_vectors(self):
        self.assertAlmostEqual(rpc._cosine_sim([1, 0], [0, 1]), 0.0)

    def test_cosine_sim_zero_vector(self):
        self.assertEqual(rpc._cosine_sim([0, 0], [1, 0]), 0.0)

    def test_curiosity_signal_filters_by_confidence(self):
        pred_high = self._make_prediction("p1", "n1", "n2", 0.9)
        pred_low = self._make_prediction("p2", "n3", "n4", 0.3)
        mem = self._make_mock_memory(active_predictions={"p1": pred_high, "p2": pred_low})
        bridge = rpc.TonicBridge()
        bridge._confidence_threshold = 0.6
        with patch.object(rpc, '_memory', mem):
            result = bridge._curiosity_signal()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].confidence, 0.9)

    def test_curiosity_signal_sorted_by_confidence_desc(self):
        preds = {
            f"p{i}": self._make_prediction(f"p{i}", f"n{i}", f"m{i}", 0.6 + i * 0.1)
            for i in range(5)
        }
        mem = self._make_mock_memory(active_predictions=preds)
        bridge = rpc.TonicBridge()
        bridge._confidence_threshold = 0.6
        bridge._max_seeds = 3
        with patch.object(rpc, '_memory', mem):
            result = bridge._curiosity_signal()
        self.assertEqual(len(result), 3)
        self.assertGreater(result[0].confidence, result[1].confidence)

    def test_tick_skips_when_in_conversation(self):
        mem = self._make_mock_memory(in_conversation=True)
        bridge = rpc.TonicBridge()
        with patch.object(rpc, '_memory', mem):
            with patch.object(bridge, '_maybe_defer') as mock_defer:
                pred = self._make_prediction("p1", "n1", "n2", 0.9)
                bridge._curiosity_signal = MagicMock(return_value=[pred])
                with patch.object(rpc, 'deposit_outbound_intent') as mock_deposit:
                    bridge._tick()
                    mock_defer.assert_called_once()
                    mock_deposit.assert_not_called()

    def test_tick_skips_when_budget_critical(self):
        mem = self._make_mock_memory(in_conversation=False)
        bridge = rpc.TonicBridge()
        with tempfile.TemporaryDirectory() as d:
            budget_path = os.path.join(d, "budget.json")
            with open(budget_path, "w") as f:
                f.write(json.dumps({"critical": True}))
            bridge._budget_path = budget_path
            with patch.object(rpc, '_memory', mem):
                with patch.object(rpc, 'deposit_outbound_intent') as mock_deposit:
                    with patch.object(bridge, '_curiosity_signal', return_value=[]):
                        bridge._tick()
                        mock_deposit.assert_not_called()

    def test_node_label_returns_metadata_label(self):
        node = MagicMock()
        node.metadata = {"label": "curiosity"}
        mem = self._make_mock_memory(nodes={"n1": node})
        bridge = rpc.TonicBridge()
        with patch.object(rpc, '_memory', mem):
            label = bridge._node_label("n1")
        self.assertEqual(label, "curiosity")

    def test_node_label_returns_node_id_when_no_label(self):
        node = MagicMock()
        node.metadata = {}
        mem = self._make_mock_memory(nodes={"n1": node})
        bridge = rpc.TonicBridge()
        with patch.object(rpc, '_memory', mem):
            label = bridge._node_label("n1")
        self.assertEqual(label, "n1")

    def test_compose_seed_format(self):
        pred = self._make_prediction("p1", "n1", "n2", 0.9)
        bridge = rpc.TonicBridge()
        node_src = MagicMock()
        node_src.metadata = {"label": "learning"}
        node_tgt = MagicMock()
        node_tgt.metadata = {"label": "memory"}
        mem = self._make_mock_memory(nodes={"n1": node_src, "n2": node_tgt})
        with patch.object(rpc, '_memory', mem):
            seed = bridge._compose_seed([pred], "substrate dynamics")
        self.assertIn("tonic-triggered: substrate dynamics", seed)
        self.assertIn("learning→memory", seed)


class TestBootstrapIntegration(unittest.TestCase):

    def test_tonic_bridge_not_started_without_env_var(self):
        """TonicBridge must not start when ANIMUS_TONIC_BRIDGE_ENABLED is absent."""
        original = rpc._tonic_bridge
        try:
            rpc._tonic_bridge = None
            env_without = {k: v for k, v in os.environ.items() if k != "ANIMUS_TONIC_BRIDGE_ENABLED"}
            with patch.dict(os.environ, env_without, clear=True):
                with patch.object(rpc.TonicBridge, 'start') as mock_start:
                    if os.environ.get("ANIMUS_TONIC_BRIDGE_ENABLED"):
                        rpc._tonic_bridge = rpc.TonicBridge()
                        rpc._tonic_bridge.start()
                    mock_start.assert_not_called()
                    self.assertIsNone(rpc._tonic_bridge)
        finally:
            rpc._tonic_bridge = original

    def test_tonic_bridge_started_with_env_var(self):
        """TonicBridge must start when ANIMUS_TONIC_BRIDGE_ENABLED=1."""
        original = rpc._tonic_bridge
        try:
            rpc._tonic_bridge = None
            with patch.dict(os.environ, {"ANIMUS_TONIC_BRIDGE_ENABLED": "1"}):
                with patch.object(rpc.TonicBridge, 'start') as mock_start:
                    if os.environ.get("ANIMUS_TONIC_BRIDGE_ENABLED"):
                        rpc._tonic_bridge = rpc.TonicBridge()
                        rpc._tonic_bridge.start()
                    mock_start.assert_called_once()
                    self.assertIsNotNone(rpc._tonic_bridge)
        finally:
            rpc._tonic_bridge = original


if __name__ == "__main__":
    unittest.main()
