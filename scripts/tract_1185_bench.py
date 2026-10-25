#!/usr/bin/env python3
# ---- Changelog ----
# [2026-10-25] Claude Code (Opus 5.5) — #1185 drain / durability measurements
# What: Three read-only-or-scratch measurements for the #1185 tract fixes:
#       `synth`  builds a ~3 GB synthetic tract with injected zeroed-page regions in a
#                scratch dir and drains it to the end with the new reader (peak RSS,
#                time, frames delivered, regions skipped, .damaged byte-exactness);
#       `drain`  drains a COPY of a real tract the same way (Syl's condensate tracts);
#       `write`  measures deposit throughput under a given fsync cadence (run once per
#                setting: the cadence knobs are read once per process).
# Why:  Josh 2026-10-24 "Do it, DudeMan" (Stage 1 lane A): prove the drain fix and the
#       fsync cost before carrying them to Syl.
# How:  NGTractBridge._drain_with_cursor in a pass loop (entries counted, then dropped,
#       as a consumer would after absorbing them); resource.getrusage for peak RSS.
#       Never point it at a live tract: it writes a cursor and <tract>.damaged next
#       to the file it drains.
# -------------------
"""Usage:
  tract_1185_bench.py synth <scratch_dir> [--gb 3] [--regions 200]
  tract_1185_bench.py drain <copy_of_tract>
  tract_1185_bench.py write <scratch_dir> [--frames 20000]
"""
import argparse
import bisect
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import ng_tract  # noqa: E402
from ng_tract_bridge import NGTractBridge  # noqa: E402

_DMG_RE = re.compile(rb"#ng-tract-damaged v1 offset=(\d+) length=(\d+) reason=(\S+) ts=[0-9.]+\n")
PAGE = 4096


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def damaged_records(path: Path):
    """Yield (offset, length, reason, data_offset_in_side_file) without loading the file."""
    if not path.exists():
        return
    with open(path, "rb") as f:
        pos = 0
        while True:
            f.seek(pos)
            head = f.readline()
            if not head:
                return
            m = _DMG_RE.match(head)
            assert m, "bad .damaged header at %d" % pos
            off, n = int(m.group(1)), int(m.group(2))
            yield off, n, m.group(3).decode(), pos + len(head)
            pos += len(head) + n + 1


def merged_regions(recs):
    out = []
    for off, n, _r, _d in sorted(recs):
        if out and out[-1][0] + out[-1][1] == off:
            out[-1][1] += n
        else:
            out.append([off, n])
    return out


def drain_to_end(tract: Path):
    tracts_dir = tract.parent.parent
    peer = tract.parent.name
    bridge = NGTractBridge(tract.stem, tracts_dir=str(tracts_dir))
    t0 = time.time()
    delivered, passes, types = 0, 0, {}
    while True:
        batch = bridge._drain_with_cursor(tract, peer)
        passes += 1
        if not batch:
            break
        delivered += len(batch)
        for e in batch:
            k = type(e).__name__
            types[k] = types.get(k, 0) + 1
        del batch
    dt = time.time() - t0
    cur = json.loads(bridge._cursor_path(tract).read_text()) if bridge._cursor_path(tract).exists() else {}
    recs = list(damaged_records(bridge._damaged_path(tract)))
    regions = merged_regions(recs)
    reasons = {}
    for _o, _n, r, _d in recs:
        reasons[r] = reasons.get(r, 0) + 1
    return {
        "file_bytes": tract.stat().st_size,
        "delivered": delivered,
        "entry_types": types,
        "passes": passes,
        "seconds": round(dt, 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "cursor_offset": cur.get("offset"),
        "held_tail_bytes": tract.stat().st_size - (cur.get("offset") or 0),
        "damaged_records": len(recs),
        "damaged_regions_merged": len(regions),
        "damaged_bytes": sum(n for _, n in regions),
        "damaged_reasons": reasons,
        "first_regions": regions[:5],
    }, recs


def cmd_synth(a):
    scratch = Path(a.scratch)
    tract = scratch / "tracts" / "peer" / "me.tract"
    tract.parent.mkdir(parents=True, exist_ok=True)
    for p in tract.parent.iterdir():
        p.unlink()
    rng = np.random.default_rng(1185)
    # one 64 MiB block of real outcome frames (768-dim, like the condensate observer's)
    frames, offs, o = [], [], 0
    i = 0
    while o < 64 * 1024 * 1024:
        emb = rng.standard_normal(768).astype(np.float32)
        fr = bytes(ng_tract.write_outcome(1.0 + i, "condensate", "t:%d" % i, True, emb, None))
        frames.append(fr)
        offs.append(o)
        o += len(fr)
        i += 1
    block = b"".join(frames)
    nblocks = max(1, int(a.gb * (1 << 30) // len(block)))
    t0 = time.time()
    with open(tract, "wb") as f:
        for _ in range(nblocks):
            f.write(block)
    size = tract.stat().st_size
    all_offs = [b * len(block) + x for b in range(nblocks) for x in offs]
    # inject zeroed-page regions: header intact, body zeroed up to the next 4 KB boundary
    picks = sorted(rng.choice(len(all_offs) - 10, size=a.regions, replace=False).tolist())
    lost = set()
    with open(tract, "r+b") as f:
        for k in picks:
            start = all_offs[k] + 40
            end = (start // PAGE + 1) * PAGE
            f.seek(start)
            f.write(b"\0" * (end - start))
            j = k
            while j < len(all_offs) and all_offs[j] < end:
                lost.add(j)
                j += 1
    # regions that touch merge into one (adjacent picks)
    expect_regions = 0
    last_end = -1
    for k in picks:
        if all_offs[k] >= last_end:
            expect_regions += 1
        e = (all_offs[k] + 40) // PAGE * PAGE + PAGE
        last_end = all_offs[bisect.bisect_left(all_offs, e)] if e < size else size
    gen_s = time.time() - t0
    res, recs = drain_to_end(tract)
    # byte-exactness of every .damaged record against the tract
    exact = True
    with open(tract, "rb") as tf, open(tract.with_name(tract.name + ".damaged"), "rb") as df:
        for off, n, _r, doff in recs:
            tf.seek(off)
            df.seek(doff)
            if tf.read(n) != df.read(n):
                exact = False
                break
    res.update({
        "frames_written": len(all_offs),
        "frames_lost_to_injection": len(lost),
        "expected_delivered": len(all_offs) - len(lost),
        "injected_regions": a.regions,
        "expected_regions_merged": expect_regions,
        "damaged_byte_exact": exact,
        "generate_seconds": round(gen_s, 1),
        "knobs": {k: os.environ.get(k) for k in (
            "NG_TRACT_DRAIN_CHUNK_BYTES", "NG_TRACT_DRAIN_MAX_ENTRIES", "NG_TRACT_DRAIN_MAX_BYTES")},
    })
    print(json.dumps(res, indent=1))


def cmd_drain(a):
    tract = Path(a.tract)
    res, _ = drain_to_end(tract)
    res["knobs"] = {k: os.environ.get(k) for k in (
        "NG_TRACT_DRAIN_CHUNK_BYTES", "NG_TRACT_DRAIN_MAX_ENTRIES", "NG_TRACT_DRAIN_MAX_BYTES")}
    print(json.dumps(res, indent=1))


def cmd_write(a):
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / ("w_%d.tract" % os.getpid())
    if path.exists():
        path.unlink()
    emb = np.random.default_rng(0).standard_normal(768).astype(np.float32)
    t0 = time.perf_counter()
    for i in range(a.frames):
        ng_tract.deposit_outcome(1.0 * i, "bench", "t:%d" % i, True, emb, [str(path)])
    sync = getattr(ng_tract, "sync_tracts", None)
    if sync:
        sync()
    dt = time.perf_counter() - t0
    size = path.stat().st_size
    print(json.dumps({
        "frames": a.frames, "bytes": size, "seconds": round(dt, 3),
        "frames_per_s": round(a.frames / dt), "us_per_frame": round(dt / a.frames * 1e6, 1),
        "sync_tracts": bool(sync),
        "knobs": {k: os.environ.get(k) for k in ("NG_TRACT_FSYNC_EVERY_FRAMES", "NG_TRACT_FSYNC_INTERVAL_MS")},
    }))
    path.unlink()


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("synth")
    s.add_argument("scratch")
    s.add_argument("--gb", type=float, default=3.0)
    s.add_argument("--regions", type=int, default=200)
    d = sp.add_parser("drain")
    d.add_argument("tract")
    w = sp.add_parser("write")
    w.add_argument("scratch")
    w.add_argument("--frames", type=int, default=20000)
    a = ap.parse_args()
    {"synth": cmd_synth, "drain": cmd_drain, "write": cmd_write}[a.cmd](a)


if __name__ == "__main__":
    main()
