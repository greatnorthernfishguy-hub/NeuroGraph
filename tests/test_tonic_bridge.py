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
                                            rpc._tract = None
                                            params = {"source": "syl_outbound", "lastAssistantMessage": "test"}
                                            try:
                                                rpc.handle_after_turn(params)
                                            except Exception:
                                                pass
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
                                            rpc._tract = None
                                            params = {"source": "josh", "lastAssistantMessage": "hello"}
                                            try:
                                                rpc.handle_after_turn(params)
                                            except Exception:
                                                pass
                                            mock_wants.assert_called_once()


if __name__ == "__main__":
    unittest.main()
