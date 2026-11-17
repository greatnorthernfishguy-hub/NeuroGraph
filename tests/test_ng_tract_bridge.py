"""
Tests for NGTractBridge — per-pair directional tract bridge (punchlist #53 v0.3).

Tests verify the NGBridge interface contract, per-pair isolation, atomic
drain safety, legacy JSONL compatibility, bounded cache, and registry
isolation.  All tests use temporary directories — no live data touched.
"""

# ---- Changelog ----
# [2026-11-17] DudeMan CC — re-point 17 stale tests at the bridge's current contract (#761)
# What: tests write tracts with the real writers (ng_tract.deposit_outcome, MmapTract.deposit)
#   and read what _drain_all() returns. record_outcome is pinned as the no-op stub it is.
#   Removed three tests of behaviour that no longer exists anywhere: the bounded
#   _peer_events cache, the legacy JSONL dual-write, the myelinated explore/exploit split.
#   test_unregistered_peer_ignored now pins the rule that exists ("_"/"." directories).
# Why: f19dbc2 (2026-06-04, substrate-as-protocol PRD Phase 4) made record_outcome a no-op;
#   c59ecc7 (2026-04-20, #155) deleted the _peer_events cache. Both deliberate.
# How: _deposit / _pairs helpers on the base class.
# -------------------


import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

# Ensure THIS checkout is importable (2026-11-17: was a hard-coded ~/NeuroGraph inserted
# at the front of sys.path, so a worktree run tested the primary checkout's bridge, #1028).
import sys
_ng_dir = str(Path(__file__).resolve().parent.parent)
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

    # ---- deposit / drain helpers (2026-11-17) ----
    # record_outcome no longer writes tracts (f19dbc2) and the _peer_events
    # cache is gone (c59ecc7), so the tests write a tract with the real
    # writers and read what _drain_all() RETURNS.

    def _deposit(self, sender, peer_id, target_id):
        """One outcome on the sender -> peer tract (mmap if myelinated)."""
        emb = self._random_embedding()
        mt = sender._myelinated.get(peer_id)
        if mt is not None:
            line = json.dumps({
                "timestamp": time.time(), "module_id": sender.module_id,
                "target_id": target_id, "success": True,
                "embedding": emb.tolist(), "metadata": {},
            }) + "\n"
            self.assertTrue(mt.deposit(line.encode("utf-8")))
            return
        import ng_tract
        ng_tract.deposit_outcome(
            timestamp=time.time(), module_id=sender.module_id,
            target_id=target_id, success=True, embedding=emb,
            tract_paths=[str(sender._module_dir / f"{peer_id}.tract")],
        )

    @staticmethod
    def _pairs(events):
        """(module_id, target_id) of each drained event, dict or native."""
        return [(NGTractBridge._get_module_id(e), NGTractBridge._get_target_id(e))
                for e in events]


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
        """record_outcome is a no-op stub at the bridge (f19dbc2, PRD Phase 4):
        returns None, writes no tract and no legacy JSONL."""
        bridge_a = self._make_bridge("test_a")
        self._make_bridge("test_b")

        result = bridge_a.record_outcome(
            embedding=self._random_embedding(),
            target_id="action:test",
            success=True,
            module_id="test_a",
        )
        self.assertIsNone(result)
        self.assertFalse((Path(self.tracts_dir) / "test_a" / "test_b.tract").exists())
        self.assertFalse((Path(self.legacy_dir) / "test_a.jsonl").exists())
        self.assertFalse(hasattr(bridge_a, "_peer_events"))

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
        """An event on A's tract to B is returned by B's drain, once."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        self._deposit(bridge_a, "mod_b", "action:hello")
        tract_path = Path(self.tracts_dir) / "mod_a" / "mod_b.tract"
        self.assertTrue(tract_path.exists(), f"Tract file not created: {tract_path}")

        self.assertEqual(self._pairs(bridge_b._drain_all()), [("mod_a", "action:hello")])
        # The cursor has moved: a second drain returns nothing.
        self.assertEqual(bridge_b._drain_all(), [])

    def test_multiple_events(self):
        """Multiple events accumulate and drain together, in order."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        for i in range(10):
            self._deposit(bridge_a, "mod_b", f"action:{i}")

        self.assertEqual([t for _, t in self._pairs(bridge_b._drain_all())],
                         [f"action:{i}" for i in range(10)])

    def test_drain_is_atomic_no_data_loss(self):
        """Deposits after a drain are returned by the next drain — none lost, none repeated."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        for i in range(5):
            self._deposit(bridge_a, "mod_b", f"pre:{i}")
        self.assertEqual([t for _, t in self._pairs(bridge_b._drain_all())],
                         [f"pre:{i}" for i in range(5)])

        for i in range(3):
            self._deposit(bridge_a, "mod_b", f"post:{i}")
        self.assertEqual([t for _, t in self._pairs(bridge_b._drain_all())],
                         [f"post:{i}" for i in range(3)])


class TestPerPairIsolation(TractBridgeTestBase):
    """Per-pair tracts ensure isolation between different peers."""

    def test_deposit_to_peer_a_not_visible_to_peer_c(self):
        """An event on A's tract to B is NOT visible to C (only B drains it)."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")
        bridge_c = self._make_bridge("mod_c")

        self._deposit(bridge_a, "mod_b", "action:secret")

        self.assertEqual(bridge_c._drain_all(), [])
        self.assertEqual(self._pairs(bridge_b._drain_all()), [("mod_a", "action:secret")])
        self.assertEqual(bridge_c._drain_all(), [])

    def test_three_module_mesh(self):
        """Three modules, one tract per ordered pair: each drains the other two."""
        bridges = {
            mid: self._make_bridge(mid)
            for mid in ["alpha", "beta", "gamma"]
        }

        for mid, bridge in bridges.items():
            for peer in bridges:
                if peer != mid:
                    self._deposit(bridge, peer, f"from:{mid}")

        for mid, bridge in bridges.items():
            pairs = self._pairs(bridge._drain_all())
            expected = set(bridges.keys()) - {mid}
            self.assertEqual(set(m for m, _ in pairs), expected,
                             f"{mid} should see events from {expected}, got {pairs}")
            self.assertEqual(len(pairs), 2)


class TestRegistryIsolation(TractBridgeTestBase):
    """The tracts directory IS the registry (no registry file): every
    subdirectory is a peer, except names starting with "_" or "."."""

    def test_unregistered_peer_ignored(self):
        self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        import ng_tract
        for name in ("_archive", ".tract-archive"):
            d = Path(self.tracts_dir) / name
            d.mkdir()
            ng_tract.deposit_outcome(
                timestamp=time.time(), module_id=name, target_id="hidden:payload",
                success=True, embedding=self._random_embedding(),
                tract_paths=[str(d / "mod_b.tract")],
            )

        self.assertEqual(sorted(bridge_b._get_registered_peers()), ["mod_a"])
        self.assertEqual(bridge_b._drain_all(), [],
                         "Directories starting with _ or . are not peers")


class TestLegacyCompat(TractBridgeTestBase):
    """Test dual-read and dual-write with legacy JSONL."""

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

        # Drain — should pick up the legacy event, once
        self.assertEqual(self._pairs(bridge_b._drain_all()), [("old_module", "legacy:event")])
        self.assertEqual(bridge_b._drain_all(), [])

    def test_no_double_count_tract_and_legacy(self):
        """Peers on tracts are NOT also read from legacy JSONL."""
        bridge_a = self._make_bridge("mod_a", legacy_compat=True)
        bridge_b = self._make_bridge("mod_b", legacy_compat=True)

        self._deposit(bridge_a, "mod_b", "t:1")
        # mod_a is in the tract registry AND has a JSONL file with the same event
        (Path(self.legacy_dir) / "mod_a.jsonl").write_text(json.dumps({
            "timestamp": time.time(), "module_id": "mod_a", "target_id": "t:1",
            "success": True, "embedding": self._random_embedding().tolist(),
            "metadata": {},
        }) + "\n")

        self.assertEqual(self._pairs(bridge_b._drain_all()), [("mod_a", "t:1")],
                         "Should not double-count tract + JSONL")


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
                    self._deposit(bridge_a, "mod_b", f"t:{start + i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=deposit_batch, args=(i * deposit_count,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent deposit errors: {errors}")

        # Drain and verify every event arrived exactly once
        targets = [t for _, t in self._pairs(bridge_b._drain_all())]
        self.assertEqual(sorted(targets),
                         sorted(f"t:{i}" for i in range(deposit_count * 4)))


class TestMyelination(TractBridgeTestBase):
    """Test mmap-based myelinated tract transport."""

    def test_myelinate_and_demyelinate_cycle(self):
        """Tract upgrades to mmap, deposits/drains work, downgrades back."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        # Myelinate A→B tract
        self.assertTrue(bridge_a.myelinate_tract("mod_b"))
        self.assertTrue(bridge_a.is_myelinated("mod_b"))

        # Deposit through the mmap; B's drain returns it
        self._deposit(bridge_a, "mod_b", "myelinated:event")
        self.assertEqual(self._pairs(bridge_b._drain_all()),
                         [("mod_a", "myelinated:event")])

        # Demyelinate
        self.assertTrue(bridge_a.demyelinate_tract("mod_b"))
        self.assertFalse(bridge_a.is_myelinated("mod_b"))

        # Deposit through the file path again
        self._deposit(bridge_a, "mod_b", "file:event")
        self.assertEqual(self._pairs(bridge_b._drain_all()), [("mod_a", "file:event")])

    def test_myelinated_deposit_drain_roundtrip(self):
        """Multiple events survive the mmap double-buffer."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        bridge_a.myelinate_tract("mod_b")

        for i in range(20):
            self._deposit(bridge_a, "mod_b", f"mmap:{i}")

        self.assertEqual([t for _, t in self._pairs(bridge_b._drain_all())],
                         [f"mmap:{i}" for i in range(20)])

    def test_upgrade_preserves_pending_signals(self):
        """File-based signals are drained and preloaded into mmap on upgrade."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        # Deposit while unmyelinated
        for i in range(5):
            self._deposit(bridge_a, "mod_b", f"pre:{i}")

        # Myelinate — should drain file-based and preload into mmap
        bridge_a.myelinate_tract("mod_b")

        # Drain from B — the preloaded events, each exactly once
        targets = [t for _, t in self._pairs(bridge_b._drain_all())]
        self.assertEqual(sorted(targets), [f"pre:{i}" for i in range(5)])

    def test_downgrade_preserves_pending_signals(self):
        """Mmap signals are drained and deposited to file on downgrade."""
        bridge_a = self._make_bridge("mod_a")
        bridge_b = self._make_bridge("mod_b")

        bridge_a.myelinate_tract("mod_b")

        # Deposit while myelinated
        for i in range(5):
            self._deposit(bridge_a, "mod_b", f"mmap_pre:{i}")

        # Demyelinate — should drain mmap and deposit to file
        bridge_a.demyelinate_tract("mod_b")

        # Drain from B — the recovered events, each exactly once
        targets = [t for _, t in self._pairs(bridge_b._drain_all())]
        self.assertEqual(sorted(targets), [f"mmap_pre:{i}" for i in range(5)])

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
