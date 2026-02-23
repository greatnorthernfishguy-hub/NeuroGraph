"""
Tests for ng_ecosystem.py — E-T Systems Module Integration Standard.

Tests cover:
  - NGEcosystem singleton lifecycle (get_instance, reset_instance)
  - Tier 1 initialization (NGLite)
  - Tier 2 initialization (NGPeerBridge)
  - Tier 3 auto-upgrade detection
  - Public API: record_outcome, get_recommendations, detect_novelty, get_context
  - State persistence (save/load)
  - Unified telemetry (stats)
  - Configuration deep merge
  - Module-level init() convenience function
  - Graceful degradation (missing deps)
  - NGEcosystemAdapter ABC enforcement
"""

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ng_ecosystem
from ng_ecosystem import (
    NGEcosystem,
    NGEcosystemAdapter,
    TIER_STANDALONE,
    TIER_PEER,
    TIER_FULL_SNN,
    TIER_NAMES,
    _deep_merge,
    __version__,
)


class TestDeepMerge(unittest.TestCase):
    """Tests for the _deep_merge() utility."""

    def test_flat_merge(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 3, "c": 4})
        self.assertEqual(base, {"a": 1, "b": 3, "c": 4})

    def test_nested_merge(self):
        base = {"outer": {"a": 1, "b": 2}}
        _deep_merge(base, {"outer": {"b": 3}})
        self.assertEqual(base, {"outer": {"a": 1, "b": 3}})

    def test_override_nested_with_scalar(self):
        base = {"a": {"nested": True}}
        _deep_merge(base, {"a": "flat"})
        self.assertEqual(base, {"a": "flat"})

    def test_empty_override(self):
        base = {"a": 1}
        _deep_merge(base, {})
        self.assertEqual(base, {"a": 1})


class TestConstants(unittest.TestCase):
    """Tests for module-level constants."""

    def test_tier_values(self):
        self.assertEqual(TIER_STANDALONE, 1)
        self.assertEqual(TIER_PEER, 2)
        self.assertEqual(TIER_FULL_SNN, 3)

    def test_tier_names(self):
        self.assertIn("Standalone", TIER_NAMES[TIER_STANDALONE])
        self.assertIn("Peer", TIER_NAMES[TIER_PEER])
        self.assertIn("SNN", TIER_NAMES[TIER_FULL_SNN])

    def test_version(self):
        self.assertEqual(__version__, "1.0.0")


class TestNGEcosystemSingleton(unittest.TestCase):
    """Tests for singleton lifecycle."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._state_path = os.path.join(self._tmpdir, "test_state.json")
        # Disable peer bridge and tier3 for these tests
        self._config = {
            "peer_bridge": {"enabled": False},
            "tier3_upgrade": {"enabled": False},
        }

    def tearDown(self):
        NGEcosystem.reset_instance("test_singleton")
        NGEcosystem.reset_instance("test_singleton_2")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_get_instance_creates_singleton(self):
        eco = NGEcosystem.get_instance(
            "test_singleton",
            state_path=self._state_path,
            config=self._config,
        )
        self.assertIsInstance(eco, NGEcosystem)
        self.assertEqual(eco.module_id, "test_singleton")

    def test_get_instance_returns_same_instance(self):
        eco1 = NGEcosystem.get_instance(
            "test_singleton",
            state_path=self._state_path,
            config=self._config,
        )
        eco2 = NGEcosystem.get_instance("test_singleton")
        self.assertIs(eco1, eco2)

    def test_different_module_ids_get_different_instances(self):
        path1 = os.path.join(self._tmpdir, "s1.json")
        path2 = os.path.join(self._tmpdir, "s2.json")
        eco1 = NGEcosystem.get_instance(
            "test_singleton", state_path=path1, config=self._config
        )
        eco2 = NGEcosystem.get_instance(
            "test_singleton_2", state_path=path2, config=self._config
        )
        self.assertIsNot(eco1, eco2)
        self.assertEqual(eco1.module_id, "test_singleton")
        self.assertEqual(eco2.module_id, "test_singleton_2")

    def test_reset_instance(self):
        eco1 = NGEcosystem.get_instance(
            "test_singleton",
            state_path=self._state_path,
            config=self._config,
        )
        NGEcosystem.reset_instance("test_singleton")
        eco2 = NGEcosystem.get_instance(
            "test_singleton",
            state_path=self._state_path,
            config=self._config,
        )
        self.assertIsNot(eco1, eco2)


class TestTier1Init(unittest.TestCase):
    """Tests for Tier 1 (NGLite) initialization."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._state_path = os.path.join(self._tmpdir, "ng_state.json")
        self._config = {
            "peer_bridge": {"enabled": False},
            "tier3_upgrade": {"enabled": False},
        }

    def tearDown(self):
        NGEcosystem.reset_instance("test_tier1")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_starts_at_tier1(self):
        eco = NGEcosystem.get_instance(
            "test_tier1",
            state_path=self._state_path,
            config=self._config,
        )
        self.assertEqual(eco.tier, TIER_STANDALONE)
        self.assertIn("Standalone", eco.tier_name)

    def test_ng_lite_initialized(self):
        eco = NGEcosystem.get_instance(
            "test_tier1",
            state_path=self._state_path,
            config=self._config,
        )
        self.assertIsNotNone(eco._ng)

    def test_state_path_created(self):
        eco = NGEcosystem.get_instance(
            "test_tier1",
            state_path=self._state_path,
            config=self._config,
        )
        self.assertTrue(Path(self._state_path).parent.exists())


class TestTier2Init(unittest.TestCase):
    """Tests for Tier 2 (NGPeerBridge) initialization."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._shared_dir = os.path.join(self._tmpdir, "shared_learning")
        os.makedirs(self._shared_dir, exist_ok=True)
        self._state_path = os.path.join(self._tmpdir, "ng_state.json")

    def tearDown(self):
        NGEcosystem.reset_instance("test_tier2")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_upgrades_to_tier2_when_peer_bridge_available(self):
        """When peer bridge is enabled and shared dir exists, should upgrade to Tier 2."""
        config = {
            "peer_bridge": {"enabled": True, "sync_interval": 100},
            "tier3_upgrade": {"enabled": False},
        }
        # Patch SHARED_LEARNING_DIR to use our temp dir
        with patch.object(ng_ecosystem, "SHARED_LEARNING_DIR", Path(self._shared_dir)):
            eco = NGEcosystem.get_instance(
                "test_tier2",
                state_path=self._state_path,
                config=config,
            )
        self.assertEqual(eco.tier, TIER_PEER)
        self.assertIsNotNone(eco._peer_bridge)

    def test_stays_tier1_when_peer_bridge_disabled(self):
        config = {
            "peer_bridge": {"enabled": False},
            "tier3_upgrade": {"enabled": False},
        }
        eco = NGEcosystem.get_instance(
            "test_tier2",
            state_path=self._state_path,
            config=config,
        )
        self.assertEqual(eco.tier, TIER_STANDALONE)
        self.assertIsNone(eco._peer_bridge)


class TestPublicAPI(unittest.TestCase):
    """Tests for the public API methods."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._state_path = os.path.join(self._tmpdir, "ng_state.json")
        self._config = {
            "peer_bridge": {"enabled": False},
            "tier3_upgrade": {"enabled": False},
        }
        self.eco = NGEcosystem.get_instance(
            "test_api",
            state_path=self._state_path,
            config=self._config,
        )

    def tearDown(self):
        NGEcosystem.reset_instance("test_api")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_embedding(self, seed=42):
        rng = np.random.RandomState(seed)
        vec = rng.randn(384).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def test_record_outcome_returns_dict(self):
        emb = self._make_embedding()
        result = self.eco.record_outcome(emb, "target:test", success=True)
        self.assertIsInstance(result, dict)
        self.assertIn("node_id", result)
        self.assertIn("weight_after", result)

    def test_record_outcome_with_metadata(self):
        emb = self._make_embedding()
        result = self.eco.record_outcome(
            emb, "target:test", success=True,
            metadata={"source": "unit_test"},
        )
        self.assertIsInstance(result, dict)

    def test_get_recommendations_empty(self):
        emb = self._make_embedding(seed=99)
        recs = self.eco.get_recommendations(emb)
        # Might be empty or None, both valid
        self.assertTrue(recs is None or isinstance(recs, list))

    def test_get_recommendations_after_learning(self):
        emb = self._make_embedding()
        for _ in range(5):
            self.eco.record_outcome(emb, "target:learned", success=True)
        recs = self.eco.get_recommendations(emb)
        self.assertIsNotNone(recs)
        self.assertGreater(len(recs), 0)
        self.assertEqual(recs[0][0], "target:learned")

    def test_detect_novelty_returns_float(self):
        emb = self._make_embedding()
        novelty = self.eco.detect_novelty(emb)
        self.assertIsInstance(novelty, float)
        self.assertGreaterEqual(novelty, 0.0)
        self.assertLessEqual(novelty, 1.0)

    def test_detect_novelty_known_vs_unknown(self):
        emb1 = self._make_embedding(seed=1)
        # First pass — should be highly novel
        novelty1 = self.eco.detect_novelty(emb1)
        # Record some outcomes so this pattern is known
        self.eco.record_outcome(emb1, "target:x", success=True)
        # Second pass — should be less novel
        novelty2 = self.eco.detect_novelty(emb1)
        self.assertLessEqual(novelty2, novelty1)

    def test_get_context_returns_dict(self):
        emb = self._make_embedding()
        ctx = self.eco.get_context(emb)
        self.assertIsInstance(ctx, dict)
        self.assertIn("tier", ctx)
        self.assertIn("tier_name", ctx)
        self.assertIn("recommendations", ctx)
        self.assertIn("novelty", ctx)
        self.assertIn("ng_context", ctx)
        self.assertEqual(ctx["tier"], TIER_STANDALONE)

    def test_get_context_ng_context_none_at_tier1(self):
        emb = self._make_embedding()
        ctx = self.eco.get_context(emb)
        self.assertIsNone(ctx["ng_context"])

    def test_record_outcome_none_when_ng_is_none(self):
        """When NGLite failed to init, API returns None gracefully."""
        eco = self.eco
        eco._ng = None
        emb = self._make_embedding()
        result = eco.record_outcome(emb, "target:x", success=True)
        self.assertIsNone(result)

    def test_detect_novelty_returns_1_when_ng_is_none(self):
        eco = self.eco
        eco._ng = None
        emb = self._make_embedding()
        novelty = eco.detect_novelty(emb)
        self.assertEqual(novelty, 1.0)


class TestStatePersistence(unittest.TestCase):
    """Tests for save/load persistence."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._state_path = os.path.join(self._tmpdir, "persist_state.json")
        self._config = {
            "peer_bridge": {"enabled": False},
            "tier3_upgrade": {"enabled": False},
        }

    def tearDown(self):
        NGEcosystem.reset_instance("test_persist")
        NGEcosystem.reset_instance("test_persist_load")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_embedding(self, seed=42):
        rng = np.random.RandomState(seed)
        vec = rng.randn(384).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def test_save_creates_file(self):
        eco = NGEcosystem.get_instance(
            "test_persist",
            state_path=self._state_path,
            config=self._config,
        )
        emb = self._make_embedding()
        eco.record_outcome(emb, "target:saved", success=True)
        eco.save()
        self.assertTrue(os.path.exists(self._state_path))

    def test_save_load_roundtrip(self):
        eco = NGEcosystem.get_instance(
            "test_persist",
            state_path=self._state_path,
            config=self._config,
        )
        emb = self._make_embedding()
        eco.record_outcome(emb, "target:saved", success=True)
        eco.save()
        NGEcosystem.reset_instance("test_persist")

        # Re-create and the state should be loaded
        eco2 = NGEcosystem.get_instance(
            "test_persist",
            state_path=self._state_path,
            config=self._config,
        )
        recs = eco2.get_recommendations(emb)
        self.assertIsNotNone(recs)
        self.assertGreater(len(recs), 0)

    def test_save_produces_valid_json(self):
        eco = NGEcosystem.get_instance(
            "test_persist",
            state_path=self._state_path,
            config=self._config,
        )
        eco.save()
        with open(self._state_path) as f:
            data = json.load(f)
        self.assertIn("version", data)
        self.assertIn("module_id", data)


class TestStats(unittest.TestCase):
    """Tests for unified telemetry."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._state_path = os.path.join(self._tmpdir, "stats_state.json")
        self._config = {
            "peer_bridge": {"enabled": False},
            "tier3_upgrade": {"enabled": False},
        }
        self.eco = NGEcosystem.get_instance(
            "test_stats",
            state_path=self._state_path,
            config=self._config,
        )

    def tearDown(self):
        NGEcosystem.reset_instance("test_stats")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_stats_returns_dict(self):
        s = self.eco.stats()
        self.assertIsInstance(s, dict)

    def test_stats_has_required_keys(self):
        s = self.eco.stats()
        self.assertIn("ecosystem_version", s)
        self.assertIn("module_id", s)
        self.assertIn("tier", s)
        self.assertIn("tier_name", s)
        self.assertIn("ng_lite", s)
        self.assertIn("state_path", s)

    def test_stats_module_id_matches(self):
        s = self.eco.stats()
        self.assertEqual(s["module_id"], "test_stats")

    def test_stats_tier_matches(self):
        s = self.eco.stats()
        self.assertEqual(s["tier"], TIER_STANDALONE)

    def test_stats_version(self):
        s = self.eco.stats()
        self.assertEqual(s["ecosystem_version"], "1.0.0")

    def test_stats_peer_bridge_none_when_disabled(self):
        s = self.eco.stats()
        self.assertIsNone(s["peer_bridge"])

    def test_stats_ng_memory_none_at_tier1(self):
        s = self.eco.stats()
        self.assertIsNone(s["ng_memory"])


class TestConvenienceInit(unittest.TestCase):
    """Tests for the module-level init() function."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._state_path = os.path.join(self._tmpdir, "init_state.json")

    def tearDown(self):
        NGEcosystem.reset_instance("test_init_func")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_init_returns_ecosystem(self):
        eco = ng_ecosystem.init(
            "test_init_func",
            state_path=self._state_path,
            config={
                "peer_bridge": {"enabled": False},
                "tier3_upgrade": {"enabled": False},
            },
        )
        self.assertIsInstance(eco, NGEcosystem)
        self.assertEqual(eco.module_id, "test_init_func")

    def test_init_returns_same_singleton(self):
        config = {
            "peer_bridge": {"enabled": False},
            "tier3_upgrade": {"enabled": False},
        }
        eco1 = ng_ecosystem.init(
            "test_init_func",
            state_path=self._state_path,
            config=config,
        )
        eco2 = ng_ecosystem.init("test_init_func")
        self.assertIs(eco1, eco2)


class TestNGEcosystemAdapter(unittest.TestCase):
    """Tests for the NGEcosystemAdapter ABC."""

    def test_cannot_instantiate_abc(self):
        with self.assertRaises(TypeError):
            NGEcosystemAdapter()

    def test_subclass_must_implement_all(self):
        class PartialAdapter(NGEcosystemAdapter):
            def on_message(self, text):
                return {}
            # Missing get_context and stats

        with self.assertRaises(TypeError):
            PartialAdapter()

    def test_complete_subclass_works(self):
        class TestAdapter(NGEcosystemAdapter):
            def on_message(self, text):
                return {"status": "ok"}
            def get_context(self, text):
                return {"tier": 1}
            def stats(self):
                return {"test": True}

        adapter = TestAdapter()
        self.assertEqual(adapter.on_message("hello"), {"status": "ok"})
        self.assertEqual(adapter.stats(), {"test": True})


class TestShutdown(unittest.TestCase):
    """Tests for graceful shutdown."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._state_path = os.path.join(self._tmpdir, "shutdown_state.json")
        self._config = {
            "peer_bridge": {"enabled": False},
            "tier3_upgrade": {"enabled": False},
        }

    def tearDown(self):
        NGEcosystem.reset_instance("test_shutdown")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_shutdown_saves_state(self):
        eco = NGEcosystem.get_instance(
            "test_shutdown",
            state_path=self._state_path,
            config=self._config,
        )
        rng = np.random.RandomState(42)
        emb = rng.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        eco.record_outcome(emb, "target:x", success=True)
        eco.shutdown()
        self.assertTrue(os.path.exists(self._state_path))


class TestConfigOverrides(unittest.TestCase):
    """Tests for configuration override behavior."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._state_path = os.path.join(self._tmpdir, "config_state.json")

    def tearDown(self):
        NGEcosystem.reset_instance("test_config")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_peer_bridge_interval_override(self):
        config = {
            "peer_bridge": {"enabled": False, "sync_interval": 50},
            "tier3_upgrade": {"enabled": False},
        }
        eco = NGEcosystem.get_instance(
            "test_config",
            state_path=self._state_path,
            config=config,
        )
        self.assertEqual(eco._config["peer_bridge"]["sync_interval"], 50)

    def test_tier3_upgrade_disabled(self):
        config = {
            "peer_bridge": {"enabled": False},
            "tier3_upgrade": {"enabled": False},
        }
        eco = NGEcosystem.get_instance(
            "test_config",
            state_path=self._state_path,
            config=config,
        )
        self.assertFalse(eco._config["tier3_upgrade"]["enabled"])

    def test_ng_lite_config_passthrough(self):
        config = {
            "peer_bridge": {"enabled": False},
            "tier3_upgrade": {"enabled": False},
            "ng_lite": {"max_nodes": 500},
        }
        eco = NGEcosystem.get_instance(
            "test_config",
            state_path=self._state_path,
            config=config,
        )
        self.assertEqual(eco._config["ng_lite"]["max_nodes"], 500)


class TestDefaultStatePath(unittest.TestCase):
    """Tests for default state path when none is provided."""

    def tearDown(self):
        NGEcosystem.reset_instance("test_default_path")

    def test_default_path_uses_et_modules_dir(self):
        config = {
            "peer_bridge": {"enabled": False},
            "tier3_upgrade": {"enabled": False},
        }
        eco = NGEcosystem.get_instance(
            "test_default_path",
            config=config,
        )
        expected_parent = Path.home() / ".et_modules" / "test_default_path"
        self.assertEqual(eco._state_path.parent, expected_parent)
        self.assertEqual(eco._state_path.name, "ng_lite_state.json")


class TestTier3Detection(unittest.TestCase):
    """Tests for Tier 3 NeuroGraph auto-detection."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._state_path = os.path.join(self._tmpdir, "tier3_state.json")

    def tearDown(self):
        NGEcosystem.reset_instance("test_tier3")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_registry_probe(self):
        """When registry has NeuroGraph entry, path is returned."""
        registry_path = os.path.join(self._tmpdir, "registry.json")
        registry = {
            "modules": {
                "neurograph": {"install_path": "/opt/neurograph"}
            }
        }
        with open(registry_path, "w") as f:
            json.dump(registry, f)

        config = {
            "peer_bridge": {"enabled": False},
            "tier3_upgrade": {"enabled": False},
        }
        eco = NGEcosystem.get_instance(
            "test_tier3",
            state_path=self._state_path,
            config=config,
        )
        with patch.object(ng_ecosystem, "REGISTRY_PATH", Path(registry_path)):
            path = eco._neurograph_path_from_registry()
        self.assertEqual(path, "/opt/neurograph")

    def test_registry_probe_missing_registry(self):
        config = {
            "peer_bridge": {"enabled": False},
            "tier3_upgrade": {"enabled": False},
        }
        eco = NGEcosystem.get_instance(
            "test_tier3",
            state_path=self._state_path,
            config=config,
        )
        fake_path = os.path.join(self._tmpdir, "nonexistent_registry.json")
        with patch.object(ng_ecosystem, "REGISTRY_PATH", Path(fake_path)):
            path = eco._neurograph_path_from_registry()
        self.assertIsNone(path)


class TestGracefulDegradation(unittest.TestCase):
    """Tests for graceful degradation when dependencies are unavailable."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._state_path = os.path.join(self._tmpdir, "degrade_state.json")

    def tearDown(self):
        NGEcosystem.reset_instance("test_degrade")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_works_without_peer_bridge_module(self):
        """Even if ng_peer_bridge import fails, ecosystem works at Tier 1."""
        config = {
            "peer_bridge": {"enabled": True},
            "tier3_upgrade": {"enabled": False},
        }
        with patch.dict(sys.modules, {"ng_peer_bridge": None}):
            # Force the import to fail
            eco = NGEcosystem(
                "test_degrade",
                state_path=self._state_path,
                config=config,
            )
        self.assertEqual(eco.tier, TIER_STANDALONE)


class TestThreadSafety(unittest.TestCase):
    """Basic thread safety tests."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._state_path = os.path.join(self._tmpdir, "thread_state.json")
        self._config = {
            "peer_bridge": {"enabled": False},
            "tier3_upgrade": {"enabled": False},
        }
        self.eco = NGEcosystem.get_instance(
            "test_thread",
            state_path=self._state_path,
            config=self._config,
        )

    def tearDown(self):
        NGEcosystem.reset_instance("test_thread")
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_concurrent_record_outcomes(self):
        """Multiple threads recording outcomes should not crash."""
        errors = []

        def record(seed):
            try:
                rng = np.random.RandomState(seed)
                emb = rng.randn(384).astype(np.float32)
                emb = emb / np.linalg.norm(emb)
                for _ in range(10):
                    self.eco.record_outcome(emb, f"target:{seed}", success=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")


if __name__ == "__main__":
    unittest.main()
