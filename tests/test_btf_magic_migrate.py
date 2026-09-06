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


if __name__ == "__main__":
    unittest.main(verbosity=2)
