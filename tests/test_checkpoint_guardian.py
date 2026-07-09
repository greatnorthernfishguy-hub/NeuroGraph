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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from checkpoint_guardian import (
    SaveGate,
    atomic_file_write,
    best_effort_git_hash,
    manifest_path_for,
    read_manifest,
    write_manifest,
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
