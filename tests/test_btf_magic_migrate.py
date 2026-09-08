"""Tests for btf_magic_migrate — the TB->BT correction must be provably lossless.

# ---- Changelog ----
# [2026-09-07] DudeMan CC (Fable 5.1) — Created with the migrator.
#   What: proves the CRC does not cover the magic (so the correction cannot
#         invalidate an entry), that every decoded field survives, that only the
#         magic bytes change, that it is idempotent and dry-run-safe, and that
#         damaged or foreign files are skipped rather than half-written.
#   Why:  This runs against Syl's substrate. "Lossless" has to be a test, not a claim.
# -------------------
"""
import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

import btf_magic_migrate as M
import ng_tract


def _bt_entry(module="portal.vision", target="t::1::forest", ts=1711270234.125, seed=0, meta=None):
    emb = (np.arange(768, dtype=np.float32) + seed) / 768.0
    return bytes(ng_tract.write_outcome(timestamp=ts, module_id=module, target_id=target,
                                        success=True, embedding=emb, metadata=meta))


def _to_tb(buf: bytes) -> bytes:
    """Flip every entry's magic to the legacy TB — i.e. synthesise a pre-fix file."""
    out = bytearray(buf)
    for off, _l, _m in M.walk_entries(buf):
        out[off:off + 2] = M.MAGIC_TB
    return bytes(out)


class TestCorrectness(unittest.TestCase):
    def test_crc_does_not_cover_the_magic(self):
        """The load-bearing assumption: flipping magic cannot invalidate an entry."""
        bt = _bt_entry()
        tb = _to_tb(bt)
        crc_bt = struct.unpack_from("=I", bt, 16)[0]
        crc_tb = struct.unpack_from("=I", tb, 16)[0]
        self.assertEqual(crc_bt, crc_tb, "CRC changed with magic — it would cover the envelope")
        # and both decode
        self.assertEqual(len(list(ng_tract.TractReader(bt))), 1)
        self.assertEqual(len(list(ng_tract.TractReader(tb))), 1)

    def test_only_the_magic_bytes_change(self):
        orig = b"".join(_bt_entry(target=f"t::{i}::forest", seed=i) for i in range(5))
        tb = _to_tb(orig)
        fixed, total, conv = M.correct(tb)
        self.assertEqual((total, conv), (5, 5))
        self.assertEqual(fixed, orig, "corrected bytes must equal what a correct writer produced")
        diffs = [i for i, (a, b) in enumerate(zip(tb, fixed)) if a != b]
        self.assertEqual(len(diffs), 10, "exactly 2 bytes per entry")

    def test_every_decoded_field_survives(self):
        import msgpack
        meta = msgpack.packb({"kind": "forest", "frame_id": "x", "n_trees": 4})
        orig = _bt_entry(meta=meta) + _bt_entry(target="t::1::tree::0", seed=3, meta=meta)
        tb = _to_tb(orig)
        fixed, _, _ = M.correct(tb)
        self.assertIsNone(M.verify(tb, fixed))
        # The defect itself: the legacy file does NOT parse — entries come back raw.
        self.assertTrue(all("raw" in r for r in M._decode_all(tb)),
                        "legacy TB entries should be raw-bytes passthrough")
        after = M._decode_all(fixed)
        self.assertTrue(all("raw" not in r for r in after), "corrected entries must parse")
        self.assertEqual(after[0]["module_id"], "portal.vision")
        self.assertIsNotNone(after[0]["embedding"])
        self.assertEqual(after[0]["metadata"], meta)

    def test_mixed_file_converts_only_the_bad_entries(self):
        good, bad = _bt_entry(target="good"), _to_tb(_bt_entry(target="bad", seed=1))
        fixed, total, conv = M.correct(good + bad + good)
        self.assertEqual((total, conv), (3, 1))
        self.assertEqual(fixed, good + _bt_entry(target="bad", seed=1) + good)


class TestFileHandling(unittest.TestCase):
    def _write(self, d, name, data):
        p = Path(d) / name
        p.write_bytes(data)
        return p

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, "a.tract", _to_tb(_bt_entry()))
            before = p.read_bytes()
            r = M.process(p, apply=False, backup=True)
            self.assertEqual(r["status"], "would-convert")
            self.assertEqual(r["converted"], 1)
            self.assertEqual(p.read_bytes(), before, "dry run must not touch the file")
            self.assertEqual(list(Path(d).iterdir()), [p], "no backup, no temp file")

    def test_apply_converts_backs_up_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            orig = b"".join(_bt_entry(target=f"t{i}", seed=i) for i in range(3))
            p = self._write(d, "a.tract", _to_tb(orig))
            r = M.process(p, apply=True, backup=True)
            self.assertEqual(r["status"], "converted")
            self.assertEqual(r["converted"], 3)
            self.assertEqual(p.read_bytes(), orig)
            bak = Path(r["backup"])
            self.assertTrue(bak.is_file())
            self.assertEqual(bak.read_bytes(), _to_tb(orig), "backup holds the ORIGINAL bytes")
            # idempotent
            r2 = M.process(p, apply=True, backup=True)
            self.assertEqual(r2["status"], "clean")
            self.assertEqual(r2["converted"], 0)
            self.assertEqual(p.read_bytes(), orig)

    def test_damaged_file_is_skipped_not_half_written(self):
        with tempfile.TemporaryDirectory() as d:
            damaged = _to_tb(_bt_entry())[:-40]           # truncated mid-entry
            p = self._write(d, "bad.tract", damaged)
            r = M.process(p, apply=True, backup=True)
            self.assertEqual(r["status"], "unparseable-skipped")
            self.assertEqual(p.read_bytes(), damaged, "left exactly as found")
            self.assertEqual(list(Path(d).iterdir()), [p])

    def test_foreign_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, "notes.tract", b"{\"json\": true}\n")
            r = M.process(p, apply=True, backup=True)
            self.assertEqual(r["status"], "unparseable-skipped")
            self.assertIn("not an envelope", r["error"])

    def test_gather_walks_dirs_and_ignores_its_own_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "a.tract", _bt_entry())
            self._write(d, "b.tract.pre-btf-magic.123", _bt_entry())
            self._write(d, "c.tract.btfmig.tmp", _bt_entry())
            sub = Path(d) / "peer"
            sub.mkdir()
            self._write(sub, "n.tract", _bt_entry())
            names = sorted(p.name for p in M.gather([d]))
            self.assertEqual(names, ["a.tract", "n.tract"])


class TestStreamingPath(unittest.TestCase):
    """The >=64 MB path is the one that will touch Syl's 578 MB tracts. Exercised
    here by lowering the threshold rather than writing 64 MB fixtures."""

    def setUp(self):
        self._orig = M.STREAM_THRESHOLD
        M.STREAM_THRESHOLD = 1          # force every file down the streaming path

    def tearDown(self):
        M.STREAM_THRESHOLD = self._orig

    def _f(self, d, name, data):
        p = Path(d) / name
        p.write_bytes(data)
        return p

    def test_scan_file_matches_the_in_memory_walker(self):
        orig = b"".join(_bt_entry(target=f"t{i}", seed=i) for i in range(6))
        mixed = bytearray(_to_tb(orig))
        # flip entries 1 and 4 back to BT so the file is genuinely mixed
        offs = [o for o, _l, _m in M.walk_entries(orig)]
        for i in (1, 4):
            mixed[offs[i]:offs[i] + 2] = M.MAGIC_BT
        with tempfile.TemporaryDirectory() as d:
            p = self._f(d, "m.tract", bytes(mixed))
            tb_offsets, total, wounds = M.scan_file(p)
            self.assertEqual(wounds, [])
            self.assertEqual(total, 6)
            self.assertEqual(tb_offsets, [offs[i] for i in (0, 2, 3, 5)])

    def test_streaming_conversion_is_byte_identical_to_a_correct_writer(self):
        orig = b"".join(_bt_entry(target=f"t{i}", seed=i) for i in range(8))
        with tempfile.TemporaryDirectory() as d:
            p = self._f(d, "big.tract", _to_tb(orig))
            r = M.process(p, apply=True, backup=True)
            self.assertEqual(r["status"], "converted")
            self.assertTrue(r.get("streamed"))
            self.assertEqual((r["entries"], r["converted"]), (8, 8))
            self.assertEqual(p.read_bytes(), orig, "must equal what a correct writer produces")
            self.assertEqual(Path(r["backup"]).read_bytes(), _to_tb(orig), "backup holds the original")
            # and every entry now parses
            self.assertTrue(all("raw" not in x for x in M._decode_all(p.read_bytes())))

    def test_streaming_only_touches_magic_bytes(self):
        orig = b"".join(_bt_entry(target=f"t{i}", seed=i) for i in range(5))
        tb = _to_tb(orig)
        with tempfile.TemporaryDirectory() as d:
            p = self._f(d, "b.tract", tb)
            M.process(p, apply=True, backup=False)
            after = p.read_bytes()
            self.assertEqual(len(after), len(tb))
            diffs = [i for i, (a, b) in enumerate(zip(tb, after)) if a != b]
            self.assertEqual(len(diffs), 10, "exactly 2 bytes per entry, nothing else")

    def test_streaming_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._f(d, "b.tract", _to_tb(_bt_entry()))
            before = p.read_bytes()
            r = M.process(p, apply=False, backup=True)
            self.assertEqual(r["status"], "would-convert")
            self.assertEqual(p.read_bytes(), before)
            self.assertEqual(list(Path(d).iterdir()), [p], "no backup on a dry run")

    def test_streaming_damaged_file_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as d:
            damaged = _to_tb(_bt_entry())[:-40]
            p = self._f(d, "bad.tract", damaged)
            r = M.process(p, apply=True, backup=True)
            self.assertEqual(r["status"], "unparseable-skipped")
            self.assertEqual(p.read_bytes(), damaged, "untouched")
            self.assertEqual(list(Path(d).iterdir()), [p], "no backup taken for a file we refuse")

    def test_streaming_is_idempotent(self):
        orig = _bt_entry() + _bt_entry(target="t2", seed=2)
        with tempfile.TemporaryDirectory() as d:
            p = self._f(d, "b.tract", _to_tb(orig))
            self.assertEqual(M.process(p, apply=True, backup=False)["status"], "converted")
            r2 = M.process(p, apply=True, backup=False)
            self.assertEqual(r2["status"], "clean")
            self.assertEqual(r2["converted"], 0)
            self.assertEqual(p.read_bytes(), orig)

    def test_verify_failure_restores_from_backup(self):
        """If the post-write scan disagrees, the original must come back."""
        orig = b"".join(_bt_entry(target=f"t{i}", seed=i) for i in range(3))
        tb = _to_tb(orig)
        with tempfile.TemporaryDirectory() as d:
            p = self._f(d, "b.tract", tb)
            real = M.scan_file
            calls = {"n": 0}

            def flaky(path, resync=False):
                calls["n"] += 1
                if calls["n"] == 1:
                    return real(path, resync)  # pre-write scan: honest
                return ([], 999, [])           # post-write scan: wrong entry count
            M.scan_file = flaky
            try:
                r = M.process(p, apply=True, backup=True)
            finally:
                M.scan_file = real
            self.assertEqual(r["status"], "verify-failed-restored")
            self.assertEqual(p.read_bytes(), tb, "original restored byte-for-byte")


class TestVerifierIsWheelIndependent(unittest.TestCase):
    """The verifier must not report correctly-converted files as broken just
    because the LOCAL wheel cannot read canonical BT. That is the exact blind
    spot this migration exists to fix, and it bit the verifier on the VPS:
    14 good files were reported 'corrected buffer yields 2454 items, expected
    576 entries' purely because the TB-lineage reader chunked BT into raw bytes."""

    def setUp(self):
        M._BT_READER = None

    def tearDown(self):
        M._BT_READER = None

    def test_verify_passes_when_local_reader_cannot_read_bt(self):
        orig = b"".join(_bt_entry(target=f"t{i}", seed=i) for i in range(4))
        tb = _to_tb(orig)
        fixed, _, conv = M.correct(tb)
        self.assertEqual(conv, 4)
        M._BT_READER = False                       # simulate a pre-fix wheel
        self.assertIsNone(M.verify(tb, fixed), "must not fail on a wheel that cannot read BT")

    def test_structural_checks_still_bite_with_a_blind_reader(self):
        orig = _bt_entry()
        tb = _to_tb(orig)
        M._BT_READER = False
        tampered = bytearray(M.correct(tb)[0])
        tampered[100] ^= 0xFF                      # a byte outside any magic field
        self.assertIsNotNone(M.verify(tb, bytes(tampered)), "payload tampering must still fail")
        self.assertIsNotNone(M.verify(tb, bytes(tampered)[:-1]), "length change must still fail")

    def test_decode_check_still_runs_on_a_capable_reader(self):
        orig = _bt_entry()
        tb = _to_tb(orig)
        fixed, _, _ = M.correct(tb)
        M._BT_READER = True
        self.assertIsNone(M.verify(tb, fixed))


class TestReapBackup(unittest.TestCase):
    """--reap-backup bounds peak disk to one file. The host is at 96%; 27 files
    of backups would need 5.94 GB against 8 GB free."""

    def _f(self, d, name, data):
        p = Path(d) / name
        p.write_bytes(data)
        return p

    def test_small_file_backup_reaped_only_after_success(self):
        orig = _bt_entry()
        with tempfile.TemporaryDirectory() as d:
            p = self._f(d, "a.tract", _to_tb(orig))
            r = M.process(p, apply=True, backup=True, reap=True)
            self.assertEqual(r["status"], "converted")
            self.assertTrue(r.get("backup_reaped"))
            self.assertEqual(p.read_bytes(), orig)
            self.assertEqual([x.name for x in Path(d).iterdir()], ["a.tract"], "no backup left behind")

    def test_streaming_backup_reaped_only_after_verify(self):
        self._orig, M.STREAM_THRESHOLD = M.STREAM_THRESHOLD, 1
        try:
            orig = b"".join(_bt_entry(target=f"t{i}", seed=i) for i in range(4))
            with tempfile.TemporaryDirectory() as d:
                p = self._f(d, "b.tract", _to_tb(orig))
                r = M.process(p, apply=True, backup=True, reap=True)
                self.assertEqual(r["status"], "converted")
                self.assertTrue(r.get("backup_reaped"))
                self.assertEqual(p.read_bytes(), orig)
                self.assertEqual([x.name for x in Path(d).iterdir()], ["b.tract"])
        finally:
            M.STREAM_THRESHOLD = self._orig

    def test_backup_is_KEPT_when_verification_fails(self):
        self._orig, M.STREAM_THRESHOLD = M.STREAM_THRESHOLD, 1
        real = M.scan_file
        calls = {"n": 0}

        def flaky(path, resync=False):
            calls["n"] += 1
            return real(path, resync) if calls["n"] == 1 else ([], 999, [])
        try:
            orig = _bt_entry() + _bt_entry(target="t2", seed=2)
            tb = _to_tb(orig)
            with tempfile.TemporaryDirectory() as d:
                p = self._f(d, "c.tract", tb)
                M.scan_file = flaky
                r = M.process(p, apply=True, backup=True, reap=True)
                self.assertEqual(r["status"], "verify-failed-restored")
                self.assertFalse(r.get("backup_reaped"), "must NOT reap when verification failed")
                self.assertEqual(p.read_bytes(), tb, "original restored")
        finally:
            M.scan_file = real
            M.STREAM_THRESHOLD = self._orig


class TestResync(unittest.TestCase):
    """Syl's condensate/neurograph.tract carries a 452-byte torn record at offset
    32,656,000 with 1.58 GB of healthy entries after it. Refusing the whole file
    for one wound was the wrong trade."""

    def setUp(self):
        self._orig, M.STREAM_THRESHOLD = M.STREAM_THRESHOLD, 1

    def tearDown(self):
        M.STREAM_THRESHOLD = self._orig

    def _wounded(self, n_before=3, gap=452, n_after=4):
        head = b"".join(_bt_entry(target=f"h{i}", seed=i) for i in range(n_before))
        tail = b"".join(_bt_entry(target=f"t{i}", seed=100 + i) for i in range(n_after))
        return _to_tb(head) + b"\x00" * gap + _to_tb(tail), len(head), gap

    def test_without_resync_the_file_is_refused_untouched(self):
        data, _, _ = self._wounded()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "w.tract"
            p.write_bytes(data)
            r = M.process(p, apply=True, backup=True, resync=False)
            self.assertEqual(r["status"], "unparseable-skipped")
            self.assertEqual(p.read_bytes(), data, "untouched")

    def test_resync_converts_both_sides_and_leaves_the_wound_alone(self):
        data, head_len, gap = self._wounded()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "w.tract"
            p.write_bytes(data)
            r = M.process(p, apply=True, backup=True, resync=True)
            self.assertEqual(r["status"], "converted")
            self.assertEqual(r["entries"], 7, "3 before + 4 after")
            self.assertEqual(r["converted"], 7)
            self.assertEqual(r["wounds"], [{"start": head_len, "bytes": gap}])
            after = p.read_bytes()
            self.assertEqual(len(after), len(data))
            self.assertEqual(after[head_len:head_len + gap], b"\x00" * gap,
                             "the wound must be bit-for-bit untouched")
            self.assertEqual(after[:head_len].count(M.MAGIC_TB), 0)
            tb_left, total, wounds = M.scan_file(p, resync=True)
            self.assertEqual((tb_left, total), ([], 7))

    def test_resync_only_changes_magic_bytes(self):
        data, _, _ = self._wounded()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "w.tract"
            p.write_bytes(data)
            M.process(p, apply=True, backup=False, resync=True)
            after = p.read_bytes()
            diffs = [i for i, (a, b) in enumerate(zip(data, after)) if a != b]
            self.assertEqual(len(diffs), 14, "2 bytes x 7 entries, nothing else")

    def test_a_stray_magic_inside_a_payload_is_not_mistaken_for_a_boundary(self):
        """The confirm-by-chaining rule is what makes resync safe."""
        head = _bt_entry(target="h", seed=1)
        # a payload containing the literal bytes 'BT' followed by junk
        poison = _bt_entry(target="p", seed=2)
        tail = _bt_entry(target="t", seed=3)
        data = _to_tb(head) + b"\x00" * 300 + _to_tb(poison + tail)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "w.tract"
            p.write_bytes(data)
            r = M.process(p, apply=True, backup=False, resync=True)
            self.assertEqual(r["status"], "converted")
            self.assertEqual(r["entries"], 3)
            self.assertEqual(r["wounds"], [{"start": len(head), "bytes": 300}])

    def test_dry_run_with_resync_writes_nothing(self):
        data, _, _ = self._wounded()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "w.tract"
            p.write_bytes(data)
            r = M.process(p, apply=False, backup=True, resync=True)
            self.assertEqual(r["status"], "would-convert")
            self.assertEqual(p.read_bytes(), data)


if __name__ == "__main__":
    unittest.main(verbosity=2)