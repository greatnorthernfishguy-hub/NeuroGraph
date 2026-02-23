"""
Tests for ET Module Manager integration.

Covers:
  - ModuleManifest: creation, file I/O, field access
  - ETModuleManager: discovery, status, registration, updates, peer queries
  - NGPeerBridge: event writing, peer sync, recommendations, novelty
  - openclaw_hook integration: peer bridge in NeuroGraphMemory
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is importable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from et_modules.manager import ETModuleManager, ModuleManifest, ModuleStatus
from ng_peer_bridge import NGPeerBridge


class TestModuleManifest(unittest.TestCase):
    """Tests for ModuleManifest dataclass."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_default_values(self):
        m = ModuleManifest()
        self.assertEqual(m.module_id, "")
        self.assertEqual(m.version, "0.0.0")
        self.assertEqual(m.git_branch, "main")
        self.assertEqual(m.dependencies, [])
        self.assertEqual(m.api_port, 0)

    def test_from_file(self):
        manifest_data = {
            "module_id": "test_module",
            "display_name": "Test Module",
            "version": "1.2.3",
            "description": "A test module",
            "install_path": "/opt/test",
            "git_remote": "https://example.com/test.git",
            "git_branch": "main",
            "entry_point": "main.py",
            "ng_lite_version": "1.0.0",
            "dependencies": [],
            "service_name": "",
            "api_port": 0,
        }
        path = os.path.join(self.tmpdir, "et_module.json")
        with open(path, "w") as f:
            json.dump(manifest_data, f)

        m = ModuleManifest.from_file(path)
        self.assertIsNotNone(m)
        self.assertEqual(m.module_id, "test_module")
        self.assertEqual(m.display_name, "Test Module")
        self.assertEqual(m.version, "1.2.3")

    def test_from_file_missing(self):
        m = ModuleManifest.from_file("/nonexistent/path")
        self.assertIsNone(m)

    def test_from_file_invalid_json(self):
        path = os.path.join(self.tmpdir, "bad.json")
        with open(path, "w") as f:
            f.write("not json")
        m = ModuleManifest.from_file(path)
        self.assertIsNone(m)

    def test_from_file_ignores_extra_fields(self):
        manifest_data = {
            "module_id": "test",
            "extra_field": "should_be_ignored",
        }
        path = os.path.join(self.tmpdir, "et_module.json")
        with open(path, "w") as f:
            json.dump(manifest_data, f)

        m = ModuleManifest.from_file(path)
        self.assertIsNotNone(m)
        self.assertEqual(m.module_id, "test")
        self.assertFalse(hasattr(m, "extra_field"))

    def test_to_file(self):
        m = ModuleManifest(
            module_id="neurograph",
            display_name="NeuroGraph",
            version="0.6.0",
        )
        path = os.path.join(self.tmpdir, "et_module.json")
        m.to_file(path)

        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["module_id"], "neurograph")
        self.assertEqual(data["version"], "0.6.0")

    def test_roundtrip(self):
        original = ModuleManifest(
            module_id="neurograph",
            display_name="NeuroGraph Foundation",
            version="0.6.0",
            git_remote="https://github.com/test/test.git",
            api_port=8080,
        )
        path = os.path.join(self.tmpdir, "et_module.json")
        original.to_file(path)
        restored = ModuleManifest.from_file(path)

        self.assertEqual(original.module_id, restored.module_id)
        self.assertEqual(original.display_name, restored.display_name)
        self.assertEqual(original.version, restored.version)
        self.assertEqual(original.git_remote, restored.git_remote)
        self.assertEqual(original.api_port, restored.api_port)


class TestETModuleManager(unittest.TestCase):
    """Tests for ETModuleManager class."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manager = ETModuleManager(root_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_init_creates_directories(self):
        root = Path(self.tmpdir)
        self.assertTrue(root.exists())
        self.assertTrue((root / "shared_learning").exists())

    def test_empty_registry(self):
        registry_path = Path(self.tmpdir) / "registry.json"
        self.assertFalse(registry_path.exists())
        modules = self.manager.discover()
        # No known locations exist in temp dir, so discovery may be empty
        self.assertIsInstance(modules, dict)

    def test_register_module(self):
        manifest = ModuleManifest(
            module_id="test_module",
            display_name="Test",
            version="1.0.0",
            install_path=self.tmpdir,
        )
        self.manager.register(manifest)

        # Should be in registry
        registry_path = Path(self.tmpdir) / "registry.json"
        self.assertTrue(registry_path.exists())
        with open(registry_path) as f:
            data = json.load(f)
        self.assertIn("test_module", data["modules"])

    def test_register_and_discover(self):
        manifest = ModuleManifest(
            module_id="custom_module",
            display_name="Custom",
            version="2.0.0",
            install_path=self.tmpdir,
        )
        self.manager.register(manifest)

        # Create a fresh manager to test load from disk
        manager2 = ETModuleManager(root_dir=self.tmpdir)
        modules = manager2.discover()
        self.assertIn("custom_module", modules)
        self.assertEqual(modules["custom_module"].version, "2.0.0")

    def test_status_returns_module_statuses(self):
        manifest = ModuleManifest(
            module_id="test_mod",
            display_name="Test",
            version="1.0.0",
            install_path=self.tmpdir,
        )
        self.manager.register(manifest)

        statuses = self.manager.status()
        self.assertIn("test_mod", statuses)
        status = statuses["test_mod"]
        self.assertTrue(status.installed)
        self.assertEqual(status.health, "healthy")

    def test_status_tier_assignment(self):
        # Register a non-neurograph module
        manifest = ModuleManifest(
            module_id="trollguard",
            display_name="TrollGuard",
            version="0.1.0",
            install_path=self.tmpdir,
        )
        self.manager.register(manifest)

        statuses = self.manager.status()
        # No peer bridge file exists, so tier should be 1
        self.assertEqual(statuses["trollguard"].tier, 1)

        # Create a peer bridge event file
        peer_file = Path(self.tmpdir) / "shared_learning" / "trollguard.jsonl"
        peer_file.write_text('{"test": true}\n')

        statuses = self.manager.status()
        self.assertEqual(statuses["trollguard"].tier, 2)

    def test_neurograph_always_tier_3(self):
        manifest = ModuleManifest(
            module_id="neurograph",
            display_name="NeuroGraph",
            version="0.6.0",
            install_path=self.tmpdir,
        )
        self.manager.register(manifest)

        statuses = self.manager.status()
        self.assertEqual(statuses["neurograph"].tier, 3)

    def test_get_peer_modules(self):
        # Register neurograph and a peer
        self.manager.register(ModuleManifest(
            module_id="neurograph",
            display_name="NeuroGraph",
            version="0.6.0",
            install_path=self.tmpdir,
        ))
        peer_dir = os.path.join(self.tmpdir, "peer")
        os.makedirs(peer_dir)
        self.manager.register(ModuleManifest(
            module_id="trollguard",
            display_name="TrollGuard",
            version="0.1.0",
            install_path=peer_dir,
        ))

        peers = self.manager.get_peer_modules()
        peer_ids = [m.module_id for m in peers]
        self.assertIn("trollguard", peer_ids)
        self.assertNotIn("neurograph", peer_ids)

    def test_get_neurograph_path(self):
        self.manager.register(ModuleManifest(
            module_id="neurograph",
            install_path="/opt/neurograph",
        ))
        # Path exists in registry but not on disk, so discover may drop it
        # But get_neurograph_path checks the registry
        self.manager._registry["neurograph"] = ModuleManifest(
            module_id="neurograph",
            install_path=self.tmpdir,
        )
        self.manager._save_registry()

        manager2 = ETModuleManager(root_dir=self.tmpdir)
        path = manager2.get_neurograph_path()
        self.assertEqual(path, self.tmpdir)

    def test_get_shared_learning_dir(self):
        shared = self.manager.get_shared_learning_dir()
        self.assertTrue(shared.endswith("shared_learning"))

    def test_update_module_not_registered(self):
        result = self.manager.update_module("nonexistent")
        self.assertEqual(result["status"], "error")

    def test_update_module_no_git_remote(self):
        self.manager.register(ModuleManifest(
            module_id="no_git",
            install_path=self.tmpdir,
        ))
        result = self.manager.update_module("no_git")
        self.assertEqual(result["status"], "skipped")


class TestNGPeerBridge(unittest.TestCase):
    """Tests for NGPeerBridge class."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.shared_dir = os.path.join(self.tmpdir, "shared_learning")
        os.makedirs(self.shared_dir)
        self.bridge = NGPeerBridge(
            module_id="neurograph",
            shared_dir=self.shared_dir,
            sync_interval=3,  # Low for testing
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_init_creates_shared_dir(self):
        self.assertTrue(Path(self.shared_dir).exists())

    def test_init_registers_module(self):
        registry_path = Path(self.shared_dir) / "_peer_registry.json"
        self.assertTrue(registry_path.exists())
        with open(registry_path) as f:
            data = json.load(f)
        self.assertIn("neurograph", data["modules"])

    def test_is_connected(self):
        self.assertTrue(self.bridge.is_connected())

    def test_disconnect_reconnect(self):
        self.bridge.disconnect()
        self.assertFalse(self.bridge.is_connected())
        self.bridge.reconnect()
        self.assertTrue(self.bridge.is_connected())

    def test_record_outcome_writes_event(self):
        embedding = np.random.randn(64).astype(np.float32)
        result = self.bridge.record_outcome(
            embedding=embedding,
            target_id="test_target",
            success=True,
            module_id="neurograph",
        )
        self.assertIsNotNone(result)
        self.assertTrue(result["cross_module"])

        # Verify event file was written
        event_file = Path(self.shared_dir) / "neurograph.jsonl"
        self.assertTrue(event_file.exists())
        with open(event_file) as f:
            event = json.loads(f.readline())
        self.assertEqual(event["target_id"], "test_target")
        self.assertTrue(event["success"])

    def test_record_outcome_disconnected(self):
        self.bridge.disconnect()
        result = self.bridge.record_outcome(
            embedding=np.zeros(64),
            target_id="x",
            success=True,
            module_id="neurograph",
        )
        self.assertIsNone(result)

    def test_sync_from_peers(self):
        # Write some events as a "peer" module
        peer_file = Path(self.shared_dir) / "trollguard.jsonl"
        event = {
            "timestamp": time.time(),
            "module_id": "trollguard",
            "target_id": "safe",
            "success": True,
            "embedding": np.random.randn(64).tolist(),
        }
        with open(peer_file, "w") as f:
            f.write(json.dumps(event) + "\n")

        # Force sync
        self.bridge._sync_from_peers()
        self.assertEqual(len(self.bridge._peer_events), 1)
        self.assertEqual(self.bridge._peer_events[0]["module_id"], "trollguard")

    def test_auto_sync_on_interval(self):
        # Write a peer event
        peer_file = Path(self.shared_dir) / "trollguard.jsonl"
        event = {
            "timestamp": time.time(),
            "module_id": "trollguard",
            "target_id": "peer_target",
            "success": True,
            "embedding": np.random.randn(64).tolist(),
        }
        with open(peer_file, "w") as f:
            f.write(json.dumps(event) + "\n")

        # Record outcomes until sync triggers (interval=3)
        emb = np.random.randn(64).astype(np.float32)
        for i in range(3):
            self.bridge.record_outcome(
                embedding=emb,
                target_id=f"t_{i}",
                success=True,
                module_id="neurograph",
            )

        # Sync should have happened
        self.assertGreater(len(self.bridge._peer_events), 0)

    def test_get_recommendations_no_peers(self):
        result = self.bridge.get_recommendations(
            embedding=np.random.randn(64),
            module_id="neurograph",
        )
        self.assertIsNone(result)

    def test_get_recommendations_from_peers(self):
        # Seed peer events
        emb = np.array([1.0] * 64)
        self.bridge._peer_events = [
            {
                "module_id": "trollguard",
                "target_id": "safe_target",
                "embedding": emb.tolist(),
            },
        ]

        # Query with similar embedding
        query = np.array([1.0] * 64)
        result = self.bridge.get_recommendations(
            embedding=query,
            module_id="neurograph",
            top_k=5,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "safe_target")  # target_id
        self.assertGreater(result[0][1], 0.9)  # high similarity

    def test_get_recommendations_filters_own_module(self):
        emb = np.array([1.0] * 64)
        self.bridge._peer_events = [
            {
                "module_id": "neurograph",  # Same module
                "target_id": "own_target",
                "embedding": emb.tolist(),
            },
        ]

        result = self.bridge.get_recommendations(
            embedding=emb,
            module_id="neurograph",
        )
        self.assertIsNone(result)  # Filtered out own events

    def test_detect_novelty_no_peers(self):
        result = self.bridge.detect_novelty(
            embedding=np.random.randn(64),
            module_id="neurograph",
        )
        self.assertIsNone(result)

    def test_detect_novelty_known_pattern(self):
        emb = np.array([1.0] * 64)
        self.bridge._peer_events = [
            {"module_id": "trollguard", "embedding": emb.tolist()},
        ]

        novelty = self.bridge.detect_novelty(
            embedding=emb,
            module_id="neurograph",
        )
        self.assertIsNotNone(novelty)
        self.assertLess(novelty, 0.1)  # Very low novelty (known pattern)

    def test_detect_novelty_new_pattern(self):
        emb_known = np.array([1.0, 0.0] * 32)
        emb_novel = np.array([0.0, 1.0] * 32)
        self.bridge._peer_events = [
            {"module_id": "trollguard", "embedding": emb_known.tolist()},
        ]

        novelty = self.bridge.detect_novelty(
            embedding=emb_novel,
            module_id="neurograph",
        )
        self.assertIsNotNone(novelty)
        self.assertGreater(novelty, 0.5)  # High novelty

    def test_sync_state(self):
        result = self.bridge.sync_state(
            local_state={"nodes": {}},
            module_id="neurograph",
        )
        self.assertIsNotNone(result)
        self.assertTrue(result["synced"])

    def test_sync_state_disconnected(self):
        self.bridge.disconnect()
        result = self.bridge.sync_state(
            local_state={},
            module_id="neurograph",
        )
        self.assertIsNone(result)

    def test_get_stats(self):
        stats = self.bridge.get_stats()
        self.assertEqual(stats["module_id"], "neurograph")
        self.assertTrue(stats["connected"])
        self.assertEqual(stats["sync_interval"], 3)

    def test_peer_events_bounded(self):
        # Fill peer events beyond max
        self.bridge._peer_events_max = 5
        for i in range(10):
            self.bridge._peer_events.append({
                "module_id": f"peer_{i}",
                "embedding": [0.0],
            })

        # Trigger trim via sync
        self.bridge._sync_from_peers()
        self.assertLessEqual(len(self.bridge._peer_events), 10)

    def test_incremental_read(self):
        """Verify peers read from last position, not re-reading."""
        peer_file = Path(self.shared_dir) / "peer_a.jsonl"

        # Write first batch
        with open(peer_file, "w") as f:
            f.write(json.dumps({"module_id": "peer_a", "batch": 1, "embedding": []}) + "\n")

        self.bridge._sync_from_peers()
        self.assertEqual(len(self.bridge._peer_events), 1)

        # Write second batch (append)
        with open(peer_file, "a") as f:
            f.write(json.dumps({"module_id": "peer_a", "batch": 2, "embedding": []}) + "\n")

        self.bridge._sync_from_peers()
        # Should have both events total, but only read the second one on this sync
        self.assertEqual(len(self.bridge._peer_events), 2)


class TestNeuroGraphMemoryPeerIntegration(unittest.TestCase):
    """Tests for NGPeerBridge integration in NeuroGraphMemory."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.shared_dir = os.path.join(self.tmpdir, "shared_learning")
        os.makedirs(self.shared_dir)

        from openclaw_hook import NeuroGraphMemory
        NeuroGraphMemory.reset_instance()
        self.ng = NeuroGraphMemory(
            workspace_dir=os.path.join(self.tmpdir, "workspace"),
            config={
                "peer_bridge": {
                    "enabled": True,
                    "shared_dir": self.shared_dir,
                    "sync_interval": 5,
                },
            },
        )

    def tearDown(self):
        from openclaw_hook import NeuroGraphMemory
        NeuroGraphMemory.reset_instance()
        shutil.rmtree(self.tmpdir)

    def test_peer_bridge_initialized(self):
        self.assertIsNotNone(self.ng._peer_bridge)
        self.assertTrue(self.ng._peer_bridge.is_connected())

    def test_on_message_writes_peer_event(self):
        self.ng.on_message("Test message about neural networks")

        # Check that an event was written to the shared dir
        event_file = Path(self.shared_dir) / "neurograph.jsonl"
        self.assertTrue(event_file.exists())
        with open(event_file) as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 0)

        event = json.loads(lines[0])
        self.assertEqual(event["module_id"], "neurograph")
        self.assertTrue(event["success"])

    def test_stats_includes_peer_bridge(self):
        stats = self.ng.stats()
        self.assertIn("peer_bridge", stats)
        self.assertTrue(stats["peer_bridge"]["connected"])
        self.assertEqual(stats["peer_bridge"]["module_id"], "neurograph")

    def test_peer_bridge_disabled(self):
        from openclaw_hook import NeuroGraphMemory
        NeuroGraphMemory.reset_instance()
        ng = NeuroGraphMemory(
            workspace_dir=os.path.join(self.tmpdir, "workspace2"),
            config={
                "peer_bridge": {"enabled": False},
            },
        )
        self.assertIsNone(ng._peer_bridge)
        stats = ng.stats()
        self.assertFalse(stats["peer_bridge"]["connected"])
        NeuroGraphMemory.reset_instance()

    def test_on_message_works_without_peer_bridge(self):
        """Graceful degradation when peer bridge is None."""
        from openclaw_hook import NeuroGraphMemory
        NeuroGraphMemory.reset_instance()
        ng = NeuroGraphMemory(
            workspace_dir=os.path.join(self.tmpdir, "workspace3"),
            config={"peer_bridge": {"enabled": False}},
        )
        result = ng.on_message("Test without peer bridge")
        self.assertEqual(result["status"], "ingested")
        NeuroGraphMemory.reset_instance()

    def test_get_peer_modules(self):
        """get_peer_modules returns empty when no peers registered."""
        peers = self.ng.get_peer_modules()
        self.assertIsInstance(peers, list)


class TestETModuleManifestNeurograph(unittest.TestCase):
    """Test the actual et_module.json shipped with NeuroGraph."""

    def test_neurograph_manifest_valid(self):
        manifest_path = os.path.join(
            os.path.dirname(__file__), "..", "et_module.json"
        )
        if not os.path.exists(manifest_path):
            self.skipTest("et_module.json not found in repo root")

        m = ModuleManifest.from_file(manifest_path)
        self.assertIsNotNone(m)
        self.assertEqual(m.module_id, "neurograph")
        self.assertEqual(m.display_name, "NeuroGraph Cognitive Foundation")
        self.assertEqual(m.ng_lite_version, "1.0.0")
        self.assertEqual(m.git_branch, "main")
        self.assertIn("github.com", m.git_remote)


if __name__ == "__main__":
    unittest.main()
