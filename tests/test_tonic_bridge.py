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


if __name__ == "__main__":
    unittest.main()
