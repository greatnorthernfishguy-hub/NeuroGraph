"""
Tests for openclaw_adapter.py — OpenClaw Adapter for E-T Systems Modules.

Tests cover:
  - OpenClawAdapter ABC enforcement (MODULE_ID required, _embed required)
  - on_message() lifecycle (embed, record, module hook, context, auto-save)
  - recall() context retrieval
  - stats() unified telemetry
  - Memory event logging (events.jsonl)
  - Hash embedding fallback
  - Skipping empty/blank messages
  - Auto-save interval
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw_adapter import OpenClawAdapter


# ---------------------------------------------------------------------------
# Test subclass — minimal implementation for testing
# ---------------------------------------------------------------------------

class MockHook(OpenClawAdapter):
    """Concrete subclass for testing.  Uses hash embedding."""

    MODULE_ID = "test_module"
    SKILL_NAME = "Test Module"
    WORKSPACE_ENV = "TEST_MODULE_WORKSPACE"
    DEFAULT_WORKSPACE = ""  # Set dynamically in each test

    def __init__(self, workspace_dir: str):
        # Set the default workspace before super().__init__
        MockHook.DEFAULT_WORKSPACE = workspace_dir
        super().__init__()
        self._module_on_message_calls = []
        self._custom_results = {}

    def _embed(self, text: str) -> np.ndarray:
        return self._hash_embed(text)

    def _module_on_message(self, text: str, embedding: np.ndarray) -> dict:
        self._module_on_message_calls.append(text)
        return self._custom_results

    def _module_stats(self) -> dict:
        return {"custom_calls": len(self._module_on_message_calls)}


class TestOpenClawAdapterABC(unittest.TestCase):
    """Tests for ABC enforcement."""

    def test_cannot_instantiate_without_module_id(self):
        """Subclass without MODULE_ID raises ValueError."""
        class NoID(OpenClawAdapter):
            def _embed(self, text):
                return np.zeros(384)

        with self.assertRaises(ValueError):
            NoID()

    def test_cannot_instantiate_without_embed(self):
        """Subclass without _embed raises TypeError."""
        with self.assertRaises(TypeError):
            class NoEmbed(OpenClawAdapter):
                MODULE_ID = "x"
                SKILL_NAME = "X"
                WORKSPACE_ENV = "X"
                DEFAULT_WORKSPACE = "/tmp/x"
            NoEmbed()


class TestOnMessage(unittest.TestCase):
    """Tests for on_message() lifecycle."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        # Clear any test_module singleton
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        os.environ.pop("TEST_MODULE_WORKSPACE", None)
        self.hook = MockHook(self._tmpdir)

    def tearDown(self):
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_on_message_returns_dict(self):
        result = self.hook.on_message("test input")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "ingested")

    def test_on_message_increments_count(self):
        self.assertEqual(self.hook._message_count, 0)
        self.hook.on_message("msg1")
        self.assertEqual(self.hook._message_count, 1)
        self.hook.on_message("msg2")
        self.assertEqual(self.hook._message_count, 2)

    def test_on_message_includes_tier(self):
        result = self.hook.on_message("test input")
        self.assertIn("tier", result)
        self.assertIn("tier_name", result)
        self.assertIsInstance(result["tier"], int)

    def test_on_message_includes_module_results(self):
        self.hook._custom_results = {"scan": "clean"}
        result = self.hook.on_message("test input")
        self.assertEqual(result["module_results"]["scan"], "clean")

    def test_on_message_calls_module_hook(self):
        self.hook.on_message("hello world")
        self.assertEqual(len(self.hook._module_on_message_calls), 1)
        self.assertEqual(self.hook._module_on_message_calls[0], "hello world")

    def test_on_message_includes_novelty(self):
        result = self.hook.on_message("test input")
        self.assertIn("novelty", result)
        self.assertIsInstance(result["novelty"], float)

    def test_on_message_includes_recommendations(self):
        result = self.hook.on_message("test input")
        self.assertIn("recommendations", result)


class TestSkipEmptyMessages(unittest.TestCase):
    """Tests for empty message handling."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        os.environ.pop("TEST_MODULE_WORKSPACE", None)
        self.hook = MockHook(self._tmpdir)

    def tearDown(self):
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_empty_string_skipped(self):
        result = self.hook.on_message("")
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(self.hook._message_count, 0)

    def test_none_skipped(self):
        result = self.hook.on_message(None)
        self.assertEqual(result["status"], "skipped")

    def test_whitespace_only_skipped(self):
        result = self.hook.on_message("   \n  ")
        self.assertEqual(result["status"], "skipped")


class TestRecall(unittest.TestCase):
    """Tests for recall() context retrieval."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        os.environ.pop("TEST_MODULE_WORKSPACE", None)
        self.hook = MockHook(self._tmpdir)

    def tearDown(self):
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_recall_returns_context_dict(self):
        ctx = self.hook.recall("test query")
        self.assertIsInstance(ctx, dict)
        self.assertIn("tier", ctx)
        self.assertIn("recommendations", ctx)
        self.assertIn("novelty", ctx)
        self.assertIn("ng_context", ctx)

    def test_recall_logs_event(self):
        self.hook.recall("test query")
        events_file = Path(self._tmpdir) / "memory" / "events.jsonl"
        self.assertTrue(events_file.exists())
        with open(events_file) as f:
            lines = f.readlines()
        # At least one recall event
        recall_events = [
            json.loads(l) for l in lines if json.loads(l).get("type") == "recall"
        ]
        self.assertGreater(len(recall_events), 0)


class TestStats(unittest.TestCase):
    """Tests for stats() telemetry."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        os.environ.pop("TEST_MODULE_WORKSPACE", None)
        self.hook = MockHook(self._tmpdir)

    def tearDown(self):
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_stats_returns_dict(self):
        s = self.hook.stats()
        self.assertIsInstance(s, dict)

    def test_stats_has_required_keys(self):
        s = self.hook.stats()
        self.assertIn("skill", s)
        self.assertIn("module_id", s)
        self.assertIn("uptime_seconds", s)
        self.assertIn("message_count", s)
        self.assertIn("workspace", s)
        self.assertIn("ecosystem", s)
        self.assertIn("module", s)

    def test_stats_skill_name(self):
        s = self.hook.stats()
        self.assertEqual(s["skill"], "Test Module")

    def test_stats_module_id(self):
        s = self.hook.stats()
        self.assertEqual(s["module_id"], "test_module")

    def test_stats_includes_module_stats(self):
        self.hook.on_message("msg1")
        self.hook.on_message("msg2")
        s = self.hook.stats()
        self.assertEqual(s["module"]["custom_calls"], 2)

    def test_stats_message_count(self):
        self.hook.on_message("msg1")
        s = self.hook.stats()
        self.assertEqual(s["message_count"], 1)

    def test_stats_ecosystem_present(self):
        s = self.hook.stats()
        self.assertIn("tier", s["ecosystem"])
        self.assertIn("ecosystem_version", s["ecosystem"])


class TestMemoryEventLogging(unittest.TestCase):
    """Tests for memory/events.jsonl logging."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        os.environ.pop("TEST_MODULE_WORKSPACE", None)
        self.hook = MockHook(self._tmpdir)

    def tearDown(self):
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_memory_dir_created(self):
        self.assertTrue((Path(self._tmpdir) / "memory").is_dir())

    def test_on_message_logs_event(self):
        self.hook.on_message("hello")
        events_file = Path(self._tmpdir) / "memory" / "events.jsonl"
        self.assertTrue(events_file.exists())

    def test_event_has_required_fields(self):
        self.hook.on_message("hello")
        events_file = Path(self._tmpdir) / "memory" / "events.jsonl"
        with open(events_file) as f:
            event = json.loads(f.readline())
        self.assertIn("ts", event)
        self.assertIn("type", event)
        self.assertIn("module", event)
        self.assertEqual(event["module"], "test_module")

    def test_multiple_events_logged(self):
        self.hook.on_message("msg1")
        self.hook.on_message("msg2")
        self.hook.recall("query")
        events_file = Path(self._tmpdir) / "memory" / "events.jsonl"
        with open(events_file) as f:
            lines = [l for l in f.readlines() if l.strip()]
        self.assertGreaterEqual(len(lines), 3)


class TestHashEmbedding(unittest.TestCase):
    """Tests for the _hash_embed() fallback."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        os.environ.pop("TEST_MODULE_WORKSPACE", None)
        self.hook = MockHook(self._tmpdir)

    def tearDown(self):
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_hash_embed_returns_ndarray(self):
        vec = self.hook._hash_embed("test")
        self.assertIsInstance(vec, np.ndarray)

    def test_hash_embed_correct_dims(self):
        vec = self.hook._hash_embed("test", dims=384)
        self.assertEqual(vec.shape, (384,))

    def test_hash_embed_normalized(self):
        vec = self.hook._hash_embed("test")
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_hash_embed_deterministic(self):
        vec1 = self.hook._hash_embed("same text")
        vec2 = self.hook._hash_embed("same text")
        np.testing.assert_array_equal(vec1, vec2)

    def test_hash_embed_different_texts(self):
        vec1 = self.hook._hash_embed("text A")
        vec2 = self.hook._hash_embed("text B")
        self.assertFalse(np.array_equal(vec1, vec2))


class TestAutoSave(unittest.TestCase):
    """Tests for auto-save interval."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        os.environ.pop("TEST_MODULE_WORKSPACE", None)
        self.hook = MockHook(self._tmpdir)
        self.hook.AUTO_SAVE_INTERVAL = 3  # Save every 3 messages

    def tearDown(self):
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_auto_save_triggers_at_interval(self):
        with patch.object(self.hook._eco, "save") as mock_save:
            self.hook.on_message("msg1")
            self.hook.on_message("msg2")
            mock_save.assert_not_called()
            self.hook.on_message("msg3")
            mock_save.assert_called_once()


class TestWorkspaceEnvVar(unittest.TestCase):
    """Tests for workspace directory configuration via env var."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._custom_ws = os.path.join(self._tmpdir, "custom_workspace")
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")

    def tearDown(self):
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        os.environ.pop("TEST_MODULE_WORKSPACE", None)
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_env_var_overrides_default(self):
        os.environ["TEST_MODULE_WORKSPACE"] = self._custom_ws
        hook = MockHook(os.path.join(self._tmpdir, "default_ws"))
        self.assertEqual(str(hook._workspace), self._custom_ws)
        self.assertTrue(Path(self._custom_ws).exists())


class TestEcosystemIntegration(unittest.TestCase):
    """Integration tests: OpenClawAdapter + NGEcosystem working together."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        os.environ.pop("TEST_MODULE_WORKSPACE", None)
        self.hook = MockHook(self._tmpdir)

    def tearDown(self):
        import ng_ecosystem
        ng_ecosystem.NGEcosystem.reset_instance("test_module")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_full_lifecycle(self):
        """Full message → recall → stats lifecycle."""
        # Message
        result = self.hook.on_message("The quick brown fox")
        self.assertEqual(result["status"], "ingested")

        # Recall
        ctx = self.hook.recall("brown fox")
        self.assertIn("tier", ctx)

        # Stats
        s = self.hook.stats()
        self.assertEqual(s["message_count"], 1)
        self.assertGreater(s["uptime_seconds"], 0)

    def test_ecosystem_tier_visible(self):
        s = self.hook.stats()
        eco = s["ecosystem"]
        self.assertIn("tier", eco)
        self.assertGreaterEqual(eco["tier"], 1)


if __name__ == "__main__":
    unittest.main()
