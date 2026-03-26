"""
Tests for NGTractBridge — per-pair directional tract bridge (punchlist #53 v0.3).

Tests verify the NGBridge interface contract, per-pair isolation, atomic
drain safety, legacy JSONL compatibility, bounded cache, and registry
isolation.  All tests use temporary directories — no live data touched.
"""

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

# Ensure NeuroGraph repo is importable
import sys
_ng_dir = os.path.expanduser("~/NeuroGraph")
if _ng_dir not in sys.path:
    sys.path.insert(0, _ng_dir)

from ng_tract_bridge import NGTractBridge
from ng_lite import NGBridge


class TractBridgeTestBase(unittest.TestCase):
    """Base class that creates isolated temp directories for each test."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="tract_test_")
        self.tracts_dir = os.path.join(self._tmpdir, "tracts")
        self.legacy_dir = os.path.join(self._tmpdir, "shared_learning")
        os.makedirs(self.tracts_dir)
        os.makedirs(self.legacy_dir)

        # Patch env vars so bridges use our temp dirs
        self._orig_tracts = os.environ.get("ET_TRACTS_DIR")
        self._orig_legacy = os.environ.get("ET_SHARED_LEARNING_DIR")
        os.environ["ET_TRACTS_DIR"] = self.tracts_dir
        os.environ["ET_SHARED_LEARNING_DIR"] = self.legacy_dir

    def tearDown(self):
        # Restore env vars
        if self._orig_tracts is None:
            os.environ.pop("ET_TRACTS_DIR", None)
        else:
            os.environ["ET_TRACTS_DIR"] = self._orig_tracts

        if self._orig_legacy is None:
            os.environ.pop("ET_SHARED_LEARNING_DIR", None)
        else:
            os.environ["ET_SHARED_LEARNING_DIR"] = self._orig_legacy

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_bridge(self, module_id, legacy_compat=True):
        return NGTractBridge(
            module_id=module_id,
            tracts_dir=self.tracts_dir,
            sync_interval=5,  # Low interval for testing
            relevance_threshold=0.3,
            legacy_compat=legacy_compat,
        )

    @staticmethod
    def _random_embedding(dim=768):
        emb = np.random.randn(dim).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-12)


class TestNGBridgeCompliance(TractBridgeTestBase):
    """Verify NGTractBridge implements the NGBridge interface correctly."""

    def test_is_subclass_of_ngbridge(self):
        self.assertTrue(issubclass(NGTractBridge, NGBridge))

    def test_is_connected(self):
        bridge = self._make_bridge("test_a")
        self.assertTrue(bridge.is_connected())
        bridge.disconnect()
        self.assertFalse(bridge.is_connected())
        bridge.reconnect()
        self.assertTrue(bridge.is_connected())

    def test_record_outcome_return_type(self):
        bridge_a = self._make_bridge("test_a")
        bridge_b = self._make_bridge("test_b")

        emb = self._random_embedding()
        result = bridge_a.record_outcome(
            embedding=emb,
            target_id="action:test",
            success=True,
            module_id="test_a",
        )
        self.assertIsInstance(result, dict)
        self.assertIn("cross_module", result)
        self.assertTrue(result["cross_module"])

    def test_get_recommendations_return_type(self):
        bridge_a = self._make_bridge("test_a")
        bridge_b = self._make_bridge("test_b")

        # Record some events from test_b, then drain into test_a
        emb = self._random_embedding()
        bridge_b.record_outcome(emb, "target:x", True, "test_b")
        bridge_a._drain_all()

        recs = bridge_a.get_recommendations(emb, "test_a", top_k=3)
        # May be None if similarity is below threshold, or a list of tuples
        if recs is not None:
            self.assertIsInstance(recs, list)
            for item in recs:
                self.assertIsInstance(item, tuple)
                self.assertEqual(len(item), 3)  # (target_id, confidence, reasoning)

    def test_detect_novelty_return_type(self):
        bridge_a = self._make_bridge("test_a")
        bridge_b = self._make_bridge("test_b")

        emb = self._random_embedding()
        bridge_b.record_outcome(emb, "target:x", True, "test_b")
        bridge_a._drain_all()

        novelty = bridge_a.detect_novelty(emb, "test_a")
        if novelty is not None:
            self.assertIsInstance(novelty, float)
            self.assertGreaterEqual(novelty, 0.0)
            self.assertLessEqual(novelty, 1.0)

    def test_sync_state_return_type(self):
        bridge = self._make_bridge("test_a")
        result = bridge.sync_state({"dummy": True}, "test_a")
        self.assertIsInstance(result, dict)
        self.assertIn("synced", result)
        self.assertTrue(result["synced"])

    def test_disconnected_returns_none(self):
        bridge = self._make_bridge("test_a")
        bridge.disconnect()

        emb = self._random_embedding()
        self.assertIsNone(bridge.record_outcome(emb, "t", True, "test_a"))
        self.assertIsNone(bridge.get_recommendations(emb, "test_a"))
        self.assertIsNone(bridge.detect_novelty(emb, "test_a"))
        self.assertIsNone(bridge.sync_state({}, "test_a"))


class TestDepositDrainCycle(TractBridgeTestBase):
    """Test the fundamental deposit → drain cycle."""

    def test_basic_deposit_and_drain(self):
        """Events deposited by A arrive at B via drain."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        emb = self._random_embedding()
        bridge_a.record_outcome(emb, "action:hello", True, "mod_a")

        # Tract file should exist: tracts/mod_a/mod_b.tract
        tract_path = Path(self.tracts_dir) / "mod_a" / "mod_b.tract"
        self.assertTrue(tract_path.exists(), f"Tract file not created: {tract_path}")

        # Drain from B's perspective
        bridge_b._drain_all()

        # B should now have the event cached
        self.assertEqual(len(bridge_b._peer_events), 1)
        self.assertEqual(bridge_b._peer_events[0]["module_id"], "mod_a")
        self.assertEqual(bridge_b._peer_events[0]["target_id"], "action:hello")

        # Tract file should be consumed (deleted after drain)
        self.assertFalse(tract_path.exists(), "Tract file should be consumed after drain")

    def test_multiple_events(self):
        """Multiple events accumulate and drain together."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        for i in range(10):
            emb = self._random_embedding()
            bridge_a.record_outcome(emb, f"action:{i}", True, "mod_a")

        bridge_b._drain_all()
        self.assertEqual(len(bridge_b._peer_events), 10)

    def test_drain_is_atomic_no_data_loss(self):
        """Deposits during drain go to a fresh file — no data lost."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        # Deposit some initial events
        for i in range(5):
            bridge_a.record_outcome(self._random_embedding(), f"pre:{i}", True, "mod_a")

        # Drain
        bridge_b._drain_all()
        self.assertEqual(len(bridge_b._peer_events), 5)

        # Deposit more events AFTER the drain
        for i in range(3):
            bridge_a.record_outcome(self._random_embedding(), f"post:{i}", True, "mod_a")

        # Second drain should get the new events
        bridge_b._drain_all()
        self.assertEqual(len(bridge_b._peer_events), 8)


class TestPerPairIsolation(TractBridgeTestBase):
    """Per-pair tracts ensure isolation between different peers."""

    def test_deposit_to_peer_a_not_visible_to_peer_c(self):
        """Events deposited by A for B are NOT visible to C (only B drains them)."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")
        bridge_c = self._make_bridge("mod_c")

        emb = self._random_embedding()
        bridge_a.record_outcome(emb, "action:secret", True, "mod_a")

        # A deposited to both B and C (fan-out to all peers)
        # But each peer only drains their OWN incoming tract
        tract_for_b = Path(self.tracts_dir) / "mod_a" / "mod_b.tract"
        tract_for_c = Path(self.tracts_dir) / "mod_a" / "mod_c.tract"
        self.assertTrue(tract_for_b.exists())
        self.assertTrue(tract_for_c.exists())

        # B drains — should get the event
        bridge_b._drain_all()
        self.assertEqual(len(bridge_b._peer_events), 1)

        # B's drain consumed its tract but NOT C's
        self.assertFalse(tract_for_b.exists())
        self.assertTrue(tract_for_c.exists())

        # C drains — gets the same event independently
        bridge_c._drain_all()
        self.assertEqual(len(bridge_c._peer_events), 1)
        self.assertFalse(tract_for_c.exists())

    def test_three_module_mesh(self):
        """Three modules depositing and draining form a complete mesh."""
        bridges = {
            mid: self._make_bridge(mid)
            for mid in ["alpha", "beta", "gamma"]
        }

        # Each module records one outcome
        for mid, bridge in bridges.items():
            emb = self._random_embedding()
            bridge.record_outcome(emb, f"from:{mid}", True, mid)

        # Each module drains — should get events from the other two
        for mid, bridge in bridges.items():
            bridge._drain_all()
            peer_ids = set(e["module_id"] for e in bridge._peer_events)
            expected = set(bridges.keys()) - {mid}
            self.assertEqual(peer_ids, expected,
                             f"{mid} should see events from {expected}, got {peer_ids}")


class TestBoundedCache(TractBridgeTestBase):
    """Verify peer_events cache stays bounded."""

    def test_cache_bounded_at_max(self):
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        # Deposit way more than the 500 max
        for i in range(600):
            bridge_a.record_outcome(self._random_embedding(), f"t:{i}", True, "mod_a")

        bridge_b._drain_all()
        self.assertLessEqual(len(bridge_b._peer_events), 500)


class TestRegistryIsolation(TractBridgeTestBase):
    """Only registered peers' tracts are drained."""

    def test_unregistered_peer_ignored(self):
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        # Manually create a tract directory for an unregistered module
        rogue_dir = Path(self.tracts_dir) / "rogue_module"
        rogue_dir.mkdir()
        rogue_tract = rogue_dir / "mod_b.tract"
        event = {
            "timestamp": time.time(),
            "module_id": "rogue_module",
            "target_id": "evil:payload",
            "success": True,
            "embedding": self._random_embedding().tolist(),
            "metadata": {},
        }
        rogue_tract.write_text(json.dumps(event) + "\n")

        # B drains — should NOT see the rogue event
        bridge_b._drain_all()
        rogue_events = [e for e in bridge_b._peer_events if e["module_id"] == "rogue_module"]
        self.assertEqual(len(rogue_events), 0, "Unregistered peer events should be ignored")


class TestLegacyCompat(TractBridgeTestBase):
    """Test dual-read and dual-write with legacy JSONL."""

    def test_dual_write_creates_jsonl(self):
        """With legacy_compat=True, events are also written to JSONL."""
        bridge = self._make_bridge("mod_a", legacy_compat=True)
        bridge.record_outcome(self._random_embedding(), "t:1", True, "mod_a")

        jsonl_path = Path(self.legacy_dir) / "mod_a.jsonl"
        self.assertTrue(jsonl_path.exists(), "Legacy JSONL should be written")

        with open(jsonl_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        self.assertEqual(len(lines), 1)

    def test_no_jsonl_when_legacy_disabled(self):
        """With legacy_compat=False, no JSONL is written."""
        bridge = self._make_bridge("mod_a", legacy_compat=False)
        bridge.record_outcome(self._random_embedding(), "t:1", True, "mod_a")

        jsonl_path = Path(self.legacy_dir) / "mod_a.jsonl"
        self.assertFalse(jsonl_path.exists(), "Legacy JSONL should NOT be written")

    def test_legacy_read_from_unupgraded_peer(self):
        """Tract bridge reads legacy JSONL from peers not on tracts."""
        bridge_b = self._make_bridge("mod_b", legacy_compat=True)

        # Simulate an unupgraded peer writing JSONL the old way
        legacy_peer_file = Path(self.legacy_dir) / "old_module.jsonl"
        event = {
            "timestamp": time.time(),
            "module_id": "old_module",
            "target_id": "legacy:event",
            "success": True,
            "embedding": self._random_embedding().tolist(),
            "metadata": {},
        }
        legacy_peer_file.write_text(json.dumps(event) + "\n")

        # Write a legacy registry so the peer is recognized
        registry_path = Path(self.legacy_dir) / "_peer_registry.json"
        registry = {
            "modules": {
                "old_module": {
                    "registered_at": time.time(),
                    "event_file": str(legacy_peer_file),
                    "pid": 99999,
                }
            }
        }
        registry_path.write_text(json.dumps(registry))

        # Drain — should pick up the legacy event
        bridge_b._drain_all()
        legacy_events = [e for e in bridge_b._peer_events if e["module_id"] == "old_module"]
        self.assertEqual(len(legacy_events), 1)
        self.assertEqual(legacy_events[0]["target_id"], "legacy:event")

    def test_no_double_count_tract_and_legacy(self):
        """Peers on tracts are NOT also read from legacy JSONL."""
        bridge_a = self._make_bridge("mod_a", legacy_compat=True)
        bridge_b = self._make_bridge("mod_b", legacy_compat=True)

        emb = self._random_embedding()
        bridge_a.record_outcome(emb, "t:1", True, "mod_a")

        # mod_a is in the tract registry AND has a JSONL file
        # Draining from B should only count the event once
        bridge_b._drain_all()
        events_from_a = [e for e in bridge_b._peer_events if e["module_id"] == "mod_a"]
        self.assertEqual(len(events_from_a), 1, "Should not double-count tract + JSONL")


class TestNoMetadataLeakage(TractBridgeTestBase):
    """Tract directories contain only .tract files — no metadata."""

    def test_module_dir_contains_only_tracts(self):
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")
        bridge_c = self._make_bridge("mod_c")

        bridge_a.record_outcome(self._random_embedding(), "t:1", True, "mod_a")

        mod_a_dir = Path(self.tracts_dir) / "mod_a"
        for f in mod_a_dir.iterdir():
            self.assertTrue(
                f.name.endswith(".tract"),
                f"Unexpected file in tract directory: {f.name}"
            )


class TestGetStats(TractBridgeTestBase):
    """Verify get_stats returns expected telemetry."""

    def test_stats_fields(self):
        bridge = self._make_bridge("mod_a")
        stats = bridge.get_stats()

        self.assertEqual(stats["module_id"], "mod_a")
        self.assertTrue(stats["connected"])
        self.assertIn("tracts_dir", stats)
        self.assertIn("drain_count", stats)
        self.assertIn("peer_events_cached", stats)
        self.assertIn("registered_peers", stats)
        self.assertIn("legacy_compat", stats)


class TestConcurrentDeposit(TractBridgeTestBase):
    """Concurrent deposits from multiple threads don't corrupt tract files."""

    def test_threaded_deposits(self):
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        errors = []
        deposit_count = 50

        def deposit_batch(start):
            try:
                for i in range(deposit_count):
                    bridge_a.record_outcome(
                        self._random_embedding(),
                        f"t:{start + i}",
                        True,
                        "mod_a",
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=deposit_batch, args=(i * deposit_count,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent deposit errors: {errors}")

        # Drain and verify all events arrived
        bridge_b._drain_all()
        self.assertEqual(len(bridge_b._peer_events), deposit_count * 4)


class TestMyelination(TractBridgeTestBase):
    """Test mmap-based myelinated tract transport."""

    def test_myelinate_and_demyelinate_cycle(self):
        """Tract upgrades to mmap, deposits/drains work, downgrades back."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        # Myelinate A→B tract
        self.assertTrue(bridge_a.myelinate_tract("mod_b"))
        self.assertTrue(bridge_a.is_myelinated("mod_b"))

        # Deposit through myelinated path
        emb = self._random_embedding()
        bridge_a.record_outcome(emb, "myelinated:event", True, "mod_a")

        # Drain from B — should get the event from mmap
        bridge_b._drain_all()
        self.assertGreaterEqual(len(bridge_b._peer_events), 1)
        myelinated_events = [
            e for e in bridge_b._peer_events
            if e.get("target_id") == "myelinated:event"
        ]
        self.assertEqual(len(myelinated_events), 1)

        # Demyelinate
        self.assertTrue(bridge_a.demyelinate_tract("mod_b"))
        self.assertFalse(bridge_a.is_myelinated("mod_b"))

        # Deposit through file path again
        bridge_a.record_outcome(self._random_embedding(), "file:event", True, "mod_a")
        bridge_b._drain_all()
        file_events = [
            e for e in bridge_b._peer_events
            if e.get("target_id") == "file:event"
        ]
        self.assertEqual(len(file_events), 1)

    def test_myelinated_deposit_drain_roundtrip(self):
        """Multiple events survive the mmap double-buffer."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        bridge_a.myelinate_tract("mod_b")

        for i in range(20):
            bridge_a.record_outcome(
                self._random_embedding(), f"mmap:{i}", True, "mod_a"
            )

        bridge_b._drain_all()
        mmap_events = [
            e for e in bridge_b._peer_events
            if e.get("target_id", "").startswith("mmap:")
        ]
        # Most should come through mmap (95% — explore rate is 5%)
        self.assertGreaterEqual(len(mmap_events), 15)

    def test_upgrade_preserves_pending_signals(self):
        """File-based signals are drained and preloaded into mmap on upgrade."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        # Deposit while unmyelinated
        for i in range(5):
            bridge_a.record_outcome(
                self._random_embedding(), f"pre:{i}", True, "mod_a"
            )

        # Myelinate — should drain file-based and preload into mmap
        bridge_a.myelinate_tract("mod_b")

        # Drain from B — should get the preloaded events
        bridge_b._drain_all()
        pre_events = [
            e for e in bridge_b._peer_events
            if e.get("target_id", "").startswith("pre:")
        ]
        self.assertEqual(len(pre_events), 5)

    def test_downgrade_preserves_pending_signals(self):
        """Mmap signals are drained and deposited to file on downgrade."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        bridge_a.myelinate_tract("mod_b")

        # Deposit while myelinated
        for i in range(5):
            bridge_a.record_outcome(
                self._random_embedding(), f"mmap_pre:{i}", True, "mod_a"
            )

        # Demyelinate — should drain mmap and deposit to file
        bridge_a.demyelinate_tract("mod_b")

        # Drain from B — should get the recovered events
        bridge_b._drain_all()
        recovered = [
            e for e in bridge_b._peer_events
            if e.get("target_id", "").startswith("mmap_pre:")
        ]
        self.assertEqual(len(recovered), 5)

    def test_myelination_state_is_runtime_only(self):
        """New bridge instance starts unmyelinated regardless of mmap files."""
        bridge_a = self._make_bridge("mod_a")
        _bridge_b = self._make_bridge("mod_b")

        bridge_a.myelinate_tract("mod_b")
        self.assertTrue(bridge_a.is_myelinated("mod_b"))

        # Create a fresh bridge for the same module
        bridge_a2 = self._make_bridge("mod_a")
        self.assertFalse(bridge_a2.is_myelinated("mod_b"))

    def test_double_myelinate_returns_false(self):
        """Myelinating an already-myelinated tract returns False."""
        bridge_a = self._make_bridge("mod_a")
        _bridge_b = self._make_bridge("mod_b")

        self.assertTrue(bridge_a.myelinate_tract("mod_b"))
        self.assertFalse(bridge_a.myelinate_tract("mod_b"))

    def test_demyelinate_unmyelinated_returns_false(self):
        """Demyelinating a non-myelinated tract returns False."""
        bridge_a = self._make_bridge("mod_a")
        self.assertFalse(bridge_a.demyelinate_tract("mod_b"))

    def test_explore_exploit_sends_some_to_file(self):
        """Myelinated tracts occasionally deposit via file (explore-exploit)."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        bridge_a.myelinate_tract("mod_b")

        # Deposit many events — some should go to file path
        for i in range(200):
            bridge_a.record_outcome(
                self._random_embedding(), f"ee:{i}", True, "mod_a"
            )

        # Check that the file-based tract exists (explore deposits landed there)
        file_tract = Path(self.tracts_dir) / "mod_a" / "mod_b.tract"
        self.assertTrue(
            file_tract.exists(),
            "Explore-exploit should produce some file-based deposits"
        )

    def test_get_stats_includes_myelination(self):
        """Stats report myelinated tract info."""
        bridge_a = self._make_bridge("mod_a")
        _bridge_b = self._make_bridge("mod_b")

        stats = bridge_a.get_stats()
        self.assertIn("myelinated_tracts", stats)
        self.assertEqual(stats["myelinated_count"], 0)

        bridge_a.myelinate_tract("mod_b")
        stats = bridge_a.get_stats()
        self.assertEqual(stats["myelinated_count"], 1)
        self.assertIn("mod_b", stats["myelinated_tracts"])


if __name__ == "__main__":
    unittest.main()
