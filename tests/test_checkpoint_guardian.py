# ---- Changelog ----
# [2026-07-09] Claude Code (Fable 5 design / Haiku implementation) — #373 guardian unit tests
# What: manifests (roundtrip, unreadable), atomic_file_write (suffix preservation,
#   failure leaves final untouched + tmp cleaned), SaveGate (permissive fresh-install,
#   ratio refusal, provisional on failed/skipped restore with existing file, operator
#   clear), quarantine pruning, generation ring (hardlink inodes, GFS retention).
# Why: #373 — the empty-writer clobber destroyed state 3x (2026-06-14/-26, 2026-07-08);
#   these pin the gate that makes it impossible and the atomicity that survives
#   mid-write power death.
# How: pure tmp_path fixtures, real files, no mocks; env-dependent knobs exercised
#   via the functions' keyword overrides, not by mutating os.environ.
# -------------------
"""Unit tests for checkpoint_guardian (#373)."""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from checkpoint_guardian import (
    SaveGate,
    atomic_file_write,
    best_effort_git_hash,
    manifest_path_for,
    read_manifest,
    write_manifest,
    quarantine_save,
    rotate_generations,
    _parse_stamp,
)


# ---- manifests ----

def test_manifest_roundtrip(tmp_path):
    ckpt = tmp_path / "main.msgpack"
    p = write_manifest(ckpt, {"nodes": 1800, "synapses": 23000, "timestep": 16000,
                              "vdb_count": 2158})
    assert p == manifest_path_for(ckpt)
    m = read_manifest(ckpt)
    assert m["nodes"] == 1800
    assert m["version"] == 1
    assert "saved_at" in m


def test_manifest_absent_returns_none(tmp_path):
    assert read_manifest(tmp_path / "main.msgpack") is None


def test_manifest_unreadable_returns_none(tmp_path):
    ckpt = tmp_path / "main.msgpack"
    manifest_path_for(ckpt).write_text("{not json")
    assert read_manifest(ckpt) is None


def test_git_hash_best_effort(tmp_path):
    # Not a git repo -> None, never raises.
    assert best_effort_git_hash(str(tmp_path)) is None


# ---- atomic writes ----

def test_atomic_write_replaces_final(tmp_path):
    final = tmp_path / "main.msgpack"
    final.write_bytes(b"OLD")
    out = atomic_file_write(str(final), lambda p: open(p, "wb").write(b"NEW"))
    assert final.read_bytes() == b"NEW"
    assert out == 3  # write_fn's return value passes through


def test_atomic_write_preserves_suffix(tmp_path):
    # graph.checkpoint() REFUSES non-.msgpack paths (#325); vdb.save() silently
    # switches to JSON on non-.msgpack (#356 trap). The tmp path MUST end .msgpack.
    final = tmp_path / "main.msgpack"
    seen = {}
    atomic_file_write(str(final), lambda p: (seen.__setitem__("p", p),
                                             open(p, "wb").write(b"x"))[1])
    assert seen["p"].endswith(".msgpack")
    assert seen["p"] != str(final)


def test_atomic_write_failure_leaves_final_untouched(tmp_path):
    final = tmp_path / "main.msgpack"
    final.write_bytes(b"GOOD")

    def bad_write(p):
        open(p, "wb").write(b"partial")
        raise RuntimeError("simulated crash mid-serialize")

    with pytest.raises(RuntimeError):
        atomic_file_write(str(final), bad_write)
    assert final.read_bytes() == b"GOOD"
    leftovers = [f for f in os.listdir(tmp_path) if "tmp-" in f]
    assert leftovers == [], "tmp file must be cleaned up on failure"


# ---- SaveGate ----

def test_gate_fresh_install_is_permissive(tmp_path):
    gate = SaveGate(tmp_path / "main.msgpack")
    gate.record_restore("no_file", 0)
    ok, _ = gate.permit(0)
    assert ok


def test_gate_permits_normal_growth(tmp_path):
    ckpt = tmp_path / "main.msgpack"
    ckpt.write_bytes(b"x")
    write_manifest(ckpt, {"nodes": 1800})
    gate = SaveGate(ckpt)
    gate.record_restore("ok", 1800)
    ok, _ = gate.permit(1810)
    assert ok


def test_gate_refuses_collapsed_state(tmp_path):
    ckpt = tmp_path / "main.msgpack"
    ckpt.write_bytes(b"x")
    write_manifest(ckpt, {"nodes": 1800})
    gate = SaveGate(ckpt)
    gate.record_restore("ok", 1800)
    ok, reason = gate.permit(6)  # the 2026-07-08 shape: 1800 -> 4-6 nodes
    assert not ok
    assert "1800" in reason


def test_gate_small_reference_is_permissive(tmp_path):
    # Below the floor, shrink is legitimate (fresh graphs churn).
    ckpt = tmp_path / "main.msgpack"
    ckpt.write_bytes(b"x")
    write_manifest(ckpt, {"nodes": 40})
    gate = SaveGate(ckpt)
    gate.record_restore("ok", 40)
    ok, _ = gate.permit(2)
    assert ok


def test_gate_provisional_on_failed_restore_with_existing_file(tmp_path):
    ckpt = tmp_path / "main.msgpack"
    ckpt.write_bytes(b"real checkpoint bytes")
    write_manifest(ckpt, {"nodes": 1800})
    gate = SaveGate(ckpt)
    gate.record_restore("failed", 0)
    assert gate.provisional
    ok, reason = gate.permit(0)
    assert not ok and "provisional" in reason


def test_gate_provisional_on_skipped_unstable_with_existing_file(tmp_path):
    # The _wait_for_stable_checkpoint timeout branch boots an EMPTY graph while a
    # real checkpoint sits on disk — a clobber-in-waiting. Must be provisional.
    ckpt = tmp_path / "main.msgpack"
    ckpt.write_bytes(b"real checkpoint bytes")
    gate = SaveGate(ckpt)
    gate.record_restore("skipped_unstable", 0)
    assert gate.provisional


def test_gate_no_provisional_without_file(tmp_path):
    gate = SaveGate(tmp_path / "main.msgpack")
    gate.record_restore("failed", 0)  # file doesn't exist -> nothing to protect
    assert not gate.provisional


def test_gate_clear_provisional(tmp_path):
    ckpt = tmp_path / "main.msgpack"
    ckpt.write_bytes(b"x")
    gate = SaveGate(ckpt)
    gate.record_restore("failed", 0)
    assert gate.provisional
    gate.clear_provisional()
    assert not gate.provisional
    ok, _ = gate.permit(0)
    assert ok  # no reference either — permissive after operator clear


# ---- quarantine ----

def test_quarantine_writes_and_prunes(tmp_path):
    paths = []
    for i in range(5):
        p = quarantine_save(tmp_path, "main",
                            lambda t, i=i: open(t, "wb").write(b"q%d" % i),
                            keep=3)
        paths.append(p)
    qdir = tmp_path / "quarantine"
    kept = list(qdir.glob("main.*.msgpack"))
    assert len(kept) == 3
    assert os.path.exists(paths[-1]), "newest quarantine entry must survive"
    assert not os.path.exists(paths[0]), "oldest must be pruned"


def test_quarantine_write_is_atomic_suffix(tmp_path):
    seen = {}
    quarantine_save(tmp_path, "vectors",
                    lambda t: (seen.__setitem__("p", t), open(t, "wb").write(b"v"))[1])
    assert seen["p"].endswith(".msgpack")


# ---- generation ring ----

def _mk_set(tmp_path, content=b"DATA"):
    main = tmp_path / "main.msgpack"
    vec = tmp_path / "vectors.msgpack"
    man = tmp_path / "main.msgpack.manifest.json"
    main.write_bytes(content)
    vec.write_bytes(content + b"v")
    man.write_text("{}")
    return [str(main), str(vec), str(man)]


def test_rotation_hardlinks_frozen_set(tmp_path):
    files = _mk_set(tmp_path)
    gen = rotate_generations(tmp_path, files)
    gen_main = Path(gen) / "main.msgpack"
    assert gen_main.exists()
    # hardlink: same inode as the current primary...
    assert os.stat(gen_main).st_ino == os.stat(files[0]).st_ino
    # ...until an atomic replace swaps the primary's inode — generation keeps old bytes.
    atomic_file_write(files[0], lambda p: open(p, "wb").write(b"NEWER"))
    assert gen_main.read_bytes() == b"DATA"
    assert Path(files[0]).read_bytes() == b"NEWER"


def test_rotation_skips_missing_set_members(tmp_path):
    files = _mk_set(tmp_path)
    files.append(str(tmp_path / "main.msgpack.activations.json"))  # not present
    gen = rotate_generations(tmp_path, files)
    assert not (Path(gen) / "main.msgpack.activations.json").exists()
    assert (Path(gen) / "main.msgpack").exists()


def test_rotation_stamp_collision_gets_suffix(tmp_path):
    files = _mk_set(tmp_path)
    now = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)
    g1 = rotate_generations(tmp_path, files, now=now)
    g2 = rotate_generations(tmp_path, files, now=now)
    assert g1 != g2
    assert Path(g2).name.startswith(Path(g1).name)


def test_parse_stamp():
    assert _parse_stamp("20260709T120000Z") == datetime(2026, 7, 9, 12, 0, 0,
                                                        tzinfo=timezone.utc)
    assert _parse_stamp("20260709T120000Z-2") == datetime(2026, 7, 9, 12, 0, 0,
                                                          tzinfo=timezone.utc)
    assert _parse_stamp("not-a-stamp") is None


def test_gfs_retention(tmp_path):
    files = _mk_set(tmp_path)
    now = datetime(2026, 7, 9, 12, 0, 0, tzinfo=timezone.utc)
    ages = [
        timedelta(minutes=0), timedelta(minutes=5), timedelta(minutes=10),   # recent 3
        timedelta(minutes=90), timedelta(minutes=95),                        # same hour: keep 1
        timedelta(hours=3),                                                  # another hour
        timedelta(days=2), timedelta(days=2, hours=1),                       # same day: keep 1
        timedelta(days=5),                                                   # another day
        timedelta(days=30),                                                  # too old: drop
    ]
    gen_root = tmp_path / "generations"
    gen_root.mkdir()
    for age in ages:
        stamp = (now - age).strftime("%Y%m%dT%H%M%SZ")
        d = gen_root / stamp
        d.mkdir(exist_ok=True)
        (d / "main.msgpack").write_bytes(b"x")
    # trigger a rotation at `now`, which also prunes
    rotate_generations(tmp_path, files, recent=3, hourly=6, daily=7, now=now)
    remaining = sorted(d.name for d in gen_root.iterdir())
    # dropped: the 30-day-old; the same-hour and same-day duplicates
    assert (now - timedelta(days=30)).strftime("%Y%m%dT%H%M%SZ") not in remaining
    hour_bucket = [(now - timedelta(minutes=90)).strftime("%Y%m%dT%H%M%SZ"),
                   (now - timedelta(minutes=95)).strftime("%Y%m%dT%H%M%SZ")]
    assert sum(1 for n in hour_bucket if n in remaining) == 1
    day_bucket = [(now - timedelta(days=2)).strftime("%Y%m%dT%H%M%SZ"),
                  (now - timedelta(days=2, hours=1)).strftime("%Y%m%dT%H%M%SZ")]
    assert sum(1 for n in day_bucket if n in remaining) == 1
    # the newest 3 all survive (plus the rotation just made)
    for age in ages[:3]:
        assert (now - age).strftime("%Y%m%dT%H%M%SZ") in remaining


def test_unknown_generation_dirs_never_deleted(tmp_path):
    files = _mk_set(tmp_path)
    gen_root = tmp_path / "generations"
    gen_root.mkdir()
    (gen_root / "keep-me-manual-backup").mkdir()
    rotate_generations(tmp_path, files)
    assert (gen_root / "keep-me-manual-backup").exists()
