# ---- Changelog ----
# [2026-10-25] Claude Code (Opus 5.5) — #1189 resumable, damage-tolerant tract drain
# What: Synthetic-tract tests for NGTractBridge._drain_with_cursor: zeroed-page damage
#       mid-file and at the tail, byte-exact <tract>.damaged records, bounded batches
#       with per-batch cursor commits, a real crash (os._exit) mid-pass and resume,
#       cursor compatibility and inode safety, undecodable / JSONL units, and a
#       concurrent locked writer racing the drainer's compaction.
# Why:  Josh 2026-10-24 "Do it, DudeMan" (Stage 1 lane A): skip a damaged frame and
#       carry on, read in chunks, catch up a large backlog in batches; LAW 7 keeps
#       the damaged bytes.
# How:  Frames are made by the installed ng_tract wheel's own encoder (write_outcome),
#       written into temp dirs; nothing live is touched.
# -------------------
"""#1189 drain tests. Temp dirs only; no live tract, cursor or checkpoint is read."""

import fcntl
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent  # the checkout under test, never ~/NeuroGraph
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import ng_tract  # noqa: E402
import ng_tract_bridge  # noqa: E402
from ng_tract_bridge import NGTractBridge  # noqa: E402

PAGE = 4096
_DMG_RE = re.compile(rb"#ng-tract-damaged v1 offset=(\d+) length=(\d+) reason=(\w+) ts=[0-9.]+\n")


def frame(i: int, dim: int = 64) -> bytes:
    emb = np.full(dim, (i % 997) / 997.0, dtype=np.float32)
    return bytes(ng_tract.write_outcome(1.0 + i, "src", "t:%d" % i, True, emb, None))


def parse_damaged(path: Path):
    """[(offset, bytes, reason)] from a .damaged side file; asserts the framing is exact."""
    data = path.read_bytes() if path.exists() else b""
    out, i = [], 0
    while i < len(data):
        m = _DMG_RE.match(data, i)
        assert m, "bad .damaged record header at %d" % i
        off, n, reason = int(m.group(1)), int(m.group(2)), m.group(3).decode()
        j = m.end()
        out.append((off, data[j:j + n], reason))
        assert data[j + n:j + n + 1] == b"\n"
        i = j + n + 1
    return out


def locked_append(path: str, data: bytes) -> None:
    """The writer side of the #1176 design (as ng_tract >= #1189 and miniTID do it):
    open O_APPEND, flock, re-check the path is still this inode, one write."""
    while True:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                if os.stat(path).st_ino != os.fstat(fd).st_ino:
                    continue
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                return
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tract1189_")
        self.tracts = os.path.join(self.tmp, "tracts")
        os.makedirs(os.path.join(self.tracts, "peer"))
        self.tract = Path(self.tracts) / "peer" / "me.tract"
        self._env = {k: os.environ.get(k) for k in (
            "NG_TRACT_DRAIN_CHUNK_BYTES", "NG_TRACT_DRAIN_MAX_ENTRIES", "NG_TRACT_DRAIN_MAX_BYTES")}
        self.bridge = NGTractBridge("me", tracts_dir=self.tracts)

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def drain(self):
        return self.bridge._drain_with_cursor(self.tract, "peer")

    def drain_all_passes(self, limit=10_000):
        got = []
        cp = self.bridge._cursor_path(self.tract)
        for _ in range(limit):
            before = cp.read_text() if cp.exists() else None
            batch = self.drain()
            got.extend(batch)
            after = cp.read_text() if cp.exists() else None
            if not batch and (after is None or before is not None
                              and json.loads(after)["offset"] == json.loads(before)["offset"]):
                return got
        raise AssertionError("drain never settled")

    def write_frames(self, n, start=0):
        frames = [frame(i) for i in range(start, start + n)]
        offsets, o = [], 0
        for f in frames:
            offsets.append(o)
            o += len(f)
        self.tract.write_bytes(b"".join(frames))
        return frames, offsets

    def cursor(self):
        return json.loads(self.bridge._cursor_path(self.tract).read_text())

    @staticmethod
    def ids(entries):
        return [int(e.target_id.split(":")[1]) for e in entries]


class TestCleanAndCompat(Base):
    def test_clean_tract_delivers_everything_once(self):
        frames, _ = self.write_frames(200)
        got = self.drain_all_passes()
        self.assertEqual(self.ids(got), list(range(200)))
        c = self.cursor()
        self.assertEqual(c["offset"], self.tract.stat().st_size)
        self.assertEqual(c["entries"], 200)
        self.assertEqual(c["ino"], self.tract.stat().st_ino)
        self.assertFalse(self.bridge._damaged_path(self.tract).exists())
        self.assertEqual(self.drain(), [])

    def test_legacy_cursor_without_ino_is_honoured(self):
        frames, offs = self.write_frames(10)
        self.bridge._cursor_path(self.tract).write_text(
            json.dumps({"offset": offs[4], "ts": 0.0, "entries": 4}))
        self.assertEqual(self.ids(self.drain()), list(range(4, 10)))
        c = self.cursor()
        self.assertEqual(set(c) >= {"offset", "ts", "entries"}, True)
        self.assertEqual(c["entries"], 10)

    def test_cursor_from_another_inode_restarts_at_zero(self):
        self.write_frames(5)
        self.bridge._cursor_path(self.tract).write_text(
            json.dumps({"offset": 10_000_000, "ts": 0.0, "entries": 3, "ino": 1}))
        self.assertEqual(self.ids(self.drain()), list(range(5)))

    def test_compaction_crash_before_rename_uses_prev_offset(self):
        frames, offs = self.write_frames(8)
        ino = self.tract.stat().st_ino
        self.bridge._cursor_path(self.tract).write_text(json.dumps(
            {"offset": 0, "ts": 0.0, "entries": 6, "ino": ino + 12345,
             "prev_ino": ino, "prev_offset": offs[6]}))
        self.assertEqual(self.ids(self.drain()), [6, 7])

    def test_jsonl_residual_lines_still_delivered(self):
        f0, f1 = frame(0), frame(1)
        self.tract.write_bytes(f0 + b'{"module_id": "x", "target_id": "j"}\n' + f1)
        got = self.drain()
        self.assertEqual(len(got), 3)
        self.assertEqual(got[1], {"module_id": "x", "target_id": "j"})


class TestDamage(Base):
    def _zero_to_page(self, data: bytearray, start: int) -> int:
        """Zero from `start` to the next 4 KB page boundary (an unwritten page)."""
        end = (start // PAGE + 1) * PAGE
        data[start:end] = b"\0" * (end - start)
        return end

    def test_zeroed_page_mid_file_skips_only_the_region_and_keeps_bytes(self):
        frames, offs = self.write_frames(300)
        original = bytearray(self.tract.read_bytes())
        data = bytearray(original)
        k = 120
        zend = self._zero_to_page(data, offs[k] + 40)  # header intact, body zeroed to page end
        self.tract.write_bytes(bytes(data))
        lost = [i for i in range(300) if offs[i] + len(frames[i]) > offs[k] and offs[i] < zend]
        nxt = next(o for o in offs if o >= zend)
        with self.assertLogs("ng_tract_bridge", level="ERROR") as cm:
            got = self.drain_all_passes()
        self.assertEqual(self.ids(got), [i for i in range(300) if i not in lost])
        recs = parse_damaged(self.bridge._damaged_path(self.tract))
        self.assertEqual(len(recs), 1)
        off, raw, reason = recs[0]
        self.assertEqual(off, offs[k])
        self.assertEqual(raw, bytes(data[offs[k]:nxt]))  # byte-exact, as on disk
        self.assertEqual(reason, "crc")
        self.assertTrue(any("SKIPPED 1 damaged region" in m for m in cm.output))

    def test_many_regions_one_pass_one_loud_line(self):
        os.environ["NG_TRACT_DRAIN_MAX_ENTRIES"] = "100000"
        frames, offs = self.write_frames(2000)
        data = bytearray(self.tract.read_bytes())
        hits = [100, 400, 777, 1200, 1900]
        for k in hits:
            self._zero_to_page(data, offs[k] + 30)
        self.tract.write_bytes(bytes(data))
        with self.assertLogs("ng_tract_bridge", level="ERROR") as cm:
            got = self.drain()
        lines = [m for m in cm.output if "SKIPPED" in m]
        self.assertEqual(len(lines), 1)
        self.assertIn("SKIPPED 5 damaged region(s)", lines[0])
        recs = parse_damaged(self.bridge._damaged_path(self.tract))
        self.assertEqual([r[0] for r in recs], [offs[k] for k in hits])
        for off, raw, _ in recs:
            self.assertEqual(raw, bytes(data[off:off + len(raw)]))
        self.assertTrue(set(hits).isdisjoint(self.ids(got)))
        self.assertEqual(len(got) + sum(1 for i in range(2000) if any(
            offs[i] < r[0] + len(r[1]) and offs[i] + len(frames[i]) > r[0] for r in recs)), 2000)

    def test_damage_at_window_edge_and_long_zero_run(self):
        os.environ["NG_TRACT_DRAIN_CHUNK_BYTES"] = str(16 * 1024)
        os.environ["NG_TRACT_DRAIN_MAX_ENTRIES"] = "100000"
        os.environ["NG_TRACT_DRAIN_MAX_BYTES"] = str(1 << 30)
        frames, offs = self.write_frames(400)
        data = bytearray(self.tract.read_bytes())
        # a 40 KB zero run: longer than two windows
        a, b = offs[50] + 24, offs[50] + 24 + 40 * 1024
        data[a:b] = b"\0" * (b - a)
        self.tract.write_bytes(bytes(data))
        got = self.drain_all_passes()
        recs = parse_damaged(self.bridge._damaged_path(self.tract))
        joined = b"".join(r[1] for r in recs)
        first, last = recs[0][0], recs[-1][0] + len(recs[-1][1])
        self.assertEqual(first, offs[50])
        self.assertEqual(joined, bytes(data[first:last]))  # contiguous pieces, byte-exact
        self.assertIn(last, offs)                           # resumes on a real frame
        survivors = [i for i in range(400) if offs[i] >= last or offs[i] + len(frames[i]) <= first]
        self.assertEqual(self.ids(got), survivors)

    def test_torn_tail_is_held_then_skipped_when_frames_follow(self):
        frames, offs = self.write_frames(50)
        data = self.tract.read_bytes()
        torn = data[:offs[49] + 100]          # last frame cut mid-body (a crash)
        self.tract.write_bytes(torn)
        self.assertEqual(self.ids(self.drain_all_passes()), list(range(49)))
        self.assertEqual(self.cursor()["offset"], offs[49])   # held, not skipped
        self.assertFalse(self.bridge._damaged_path(self.tract).exists())
        # the writer restarts and appends (no repair): the torn frame is now mid-file
        new = b"".join(frame(i) for i in range(100, 110))
        with open(self.tract, "ab") as fh:
            fh.write(new)
        got = self.drain_all_passes()
        self.assertEqual(self.ids(got), list(range(100, 110)))
        recs = parse_damaged(self.bridge._damaged_path(self.tract))
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0][0], offs[49])
        self.assertEqual(recs[0][1], torn[offs[49]:])
        self.assertIn(recs[0][2], ("crc", "torn"))  # crc once the appends cover its claimed length

    def test_zeroed_tail_page_is_held_not_skipped(self):
        frames, offs = self.write_frames(30)
        data = bytearray(self.tract.read_bytes())
        data[offs[29] + 10:] = b"\0" * (len(data) - offs[29] - 10)
        data += b"\0" * 300
        self.tract.write_bytes(bytes(data))
        self.assertEqual(self.ids(self.drain_all_passes()), list(range(29)))
        self.assertEqual(self.cursor()["offset"], offs[29])
        self.assertFalse(self.bridge._damaged_path(self.tract).exists())

    def test_crc_valid_but_undecodable_frame_is_kept(self):
        import struct
        import zlib
        payload = b"\xff" * 40
        env = (b"BT" + bytes([1, 1]) + struct.pack("<I", 24 + len(payload)) + struct.pack("<d", 1.0)
               + struct.pack("<I", zlib.crc32(payload)) + bytes([1, 0, 0, 0]))
        bad = env + payload
        self.tract.write_bytes(frame(0) + bad + frame(1))
        got = self.drain()
        self.assertEqual(self.ids(got), [0, 1])
        recs = parse_damaged(self.bridge._damaged_path(self.tract))
        self.assertEqual(recs, [(len(frame(0)), bad, "undecodable")])

    def test_preserve_failure_never_skips(self):
        frames, offs = self.write_frames(20)
        data = bytearray(self.tract.read_bytes())
        self._zero_to_page(data, offs[5] + 30)
        self.tract.write_bytes(bytes(data))
        orig = self.bridge._preserve_damaged
        self.bridge._preserve_damaged = lambda *a, **k: False
        self.assertEqual(self.drain(), [])            # the window holding damage is not delivered
        self.assertFalse(self.bridge._cursor_path(self.tract).exists())
        self.bridge._preserve_damaged = orig
        self.assertTrue(len(self.drain_all_passes()) > 0)
        self.assertEqual(parse_damaged(self.bridge._damaged_path(self.tract))[0][0], offs[5])


class TestBatches(Base):
    def test_bounded_batches_no_redelivery(self):
        os.environ["NG_TRACT_DRAIN_MAX_ENTRIES"] = "7"
        self.write_frames(50)
        passes = []
        while True:
            b = self.drain()
            if not b:
                break
            passes.append(self.ids(b))
        self.assertTrue(all(len(p) <= 7 for p in passes))
        self.assertEqual([i for p in passes for i in p], list(range(50)))

    def test_cursor_committed_after_each_window(self):
        os.environ["NG_TRACT_DRAIN_CHUNK_BYTES"] = str(8 * 1024)
        os.environ["NG_TRACT_DRAIN_MAX_ENTRIES"] = "100000"
        os.environ["NG_TRACT_DRAIN_MAX_BYTES"] = str(1 << 30)
        self.write_frames(100)
        seen = []
        orig = self.bridge._write_cursor

        def spy(tp, off, tot, **kw):
            seen.append(off)
            return orig(tp, off, tot, **kw)
        self.bridge._write_cursor = spy
        self.assertEqual(self.ids(self.drain()), list(range(100)))
        self.assertGreater(len(seen), 3)
        self.assertEqual(seen, sorted(seen))

    def test_real_crash_mid_pass_resumes_at_last_committed_batch(self):
        """A child process drains with 8 KB windows and dies (os._exit) on its 3rd cursor
        commit; the parent resumes from the 2nd commit and gets exactly the rest."""
        frames, offs = self.write_frames(120)
        child = textwrap.dedent("""
            import os, sys
            sys.path.insert(0, %r)
            os.environ["NG_TRACT_DRAIN_CHUNK_BYTES"] = "8192"
            os.environ["NG_TRACT_DRAIN_MAX_ENTRIES"] = "100000"
            from pathlib import Path
            from ng_tract_bridge import NGTractBridge
            b = NGTractBridge("me", tracts_dir=%r)
            orig = b._write_cursor
            n = [0]
            def w(*a, **k):
                n[0] += 1
                if n[0] == 3:
                    os._exit(17)          # killed after two committed batches
                return orig(*a, **k)
            b._write_cursor = w
            b._drain_with_cursor(Path(%r), "peer")
            os._exit(0)
        """) % (str(_HERE), self.tracts, str(self.tract))
        r = subprocess.run([sys.executable, "-c", child], capture_output=True)
        self.assertEqual(r.returncode, 17, r.stderr.decode()[-2000:])
        committed = self.cursor()["offset"]
        self.assertIn(committed, offs)
        k = offs.index(committed)
        self.assertGreater(k, 0)
        self.assertEqual(self.ids(self.drain_all_passes()), list(range(k, 120)))


class TestConcurrentWriterAndCompaction(Base):
    def test_locked_writer_races_drain_and_compaction(self):
        ng_tract_bridge._COMPACT_THRESHOLD_BYTES, saved = 20_000, ng_tract_bridge._COMPACT_THRESHOLD_BYTES
        os.environ["NG_TRACT_DRAIN_MAX_ENTRIES"] = "50"
        try:
            N = 1500
            stop = threading.Event()

            def writer():
                for i in range(N):
                    locked_append(str(self.tract), frame(i))
                    if i % 50 == 0:
                        time.sleep(0.001)
                stop.set()

            t = threading.Thread(target=writer)
            t.start()
            got = []
            while not stop.is_set():
                got.extend(self.drain())
            t.join()
            got.extend(self.drain_all_passes())
            self.assertEqual(sorted(self.ids(got)), list(range(N)))
            self.assertEqual(len(got), N)  # each exactly once
            self.assertFalse(self.bridge._damaged_path(self.tract).exists())
            self.assertLess(self.tract.stat().st_size, 20_000 + 10 * len(frame(0)))
        finally:
            ng_tract_bridge._COMPACT_THRESHOLD_BYTES = saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
