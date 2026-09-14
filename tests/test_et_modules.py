# ---- Changelog ----
# [2026-09-08] Cursor Grok — Nightly audit area 2: delete dead peer-bridge tests
# What: Removed skipped TestNGPeerBridge class, NGPeerBridgeStub, and the four
#   skipped TestNeuroGraphMemoryPeerIntegration methods that only skip because
#   ng_peer_bridge.py is gone. Kept live graceful-degradation / peer-list tests.
# Why: Tests that only skip for a deleted file are inventory lag, not coverage.
# How: Delete. No rewrite of JSONL broadcast tests (that world is gone).
# -------------------

"""
Tests for ET Module Manager integration.

Covers:
  - ModuleManifest: creation, file I/O, field access
  - ETModuleManager: discovery, status, registration, updates, peer queries
  - NeuroGraphMemory still ingesting when the tract bridge is disabled
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

# Ensure project root is importable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from et_modules.manager import ETModuleManager, ModuleManifest, ModuleStatus


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


class TestNeuroGraphMemoryPeerIntegration(unittest.TestCase):
    """Live NeuroGraphMemory tests that do not depend on deleted NGPeerBridge."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        from openclaw_hook import NeuroGraphMemory
        NeuroGraphMemory.reset_instance()
        shutil.rmtree(self.tmpdir)

    def test_on_message_works_without_peer_bridge(self):
        """Graceful degradation when the tract/peer bridge is disabled."""
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
        from openclaw_hook import NeuroGraphMemory
        NeuroGraphMemory.reset_instance()
        ng = NeuroGraphMemory(
            workspace_dir=os.path.join(self.tmpdir, "workspace"),
            config={"peer_bridge": {"enabled": False}},
        )
        peers = ng.get_peer_modules()
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
        self.assertEqual(m.display_name, "NeuroGraph Foundation")
        self.assertEqual(m.ng_lite_version, "1.0.0")
        self.assertEqual(m.git_branch, "main")
        self.assertIn("github.com", m.git_remote)


if __name__ == "__main__":
    unittest.main()
