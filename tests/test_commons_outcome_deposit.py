"""
Commons Track-2 — outcome-throttle retirement (the LAST of the three NG _deposit_*_to_river throttles).

# ---- Changelog ----
# [2026-06-24] Claude Code (Opus 4.8) — retire _deposit_outcome_to_river throttle
# What: Proves _deposit_outcome_to_river deposits a raw outcome (precomputed embedding + target_id +
#       success + metadata) into a sandbox Commons as a single deposit, fail-soft. Mirrors the
#       topology + experience throttle tests. Completes the deposit-side substrate-as-protocol work.
# Why: LAW 7 — raw outcome into the one pool; consumers classify at their bucket. Single deposit
#       (no content text → not dual-pass; it's a raw broadcast).
# How: patch commons.get_commons → sandbox Commons; drive the real rpc function; assert the deposit
#      landed with the target_id + metadata in synapse last_context.
# -------------------
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neurograph_rpc as rpc
import commons as commons_mod

assert getattr(rpc, "_memory", None) is None, "test must not run against a live NeuroGraphMemory"


def _emb(seed=1):
    rng = np.random.RandomState(seed)
    v = rng.randn(768).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _drive(embedding, target_id, success=True, metadata=None):
    commons = commons_mod.Commons()
    orig = commons_mod.get_commons
    commons_mod.get_commons = lambda: commons
    try:
        rpc._deposit_outcome_to_river(embedding, target_id, success, metadata)
    finally:
        commons_mod.get_commons = orig
    return commons


def _outcome_synapses(commons, target_id):
    return [s for s in commons._ng.synapses.values() if getattr(s, "target_id", "") == target_id]


def test_outcome_deposits_to_commons():
    commons = _drive(_emb(1), "wire:absorb:abc123", True, {"kind": "wire_outcome", "src": "expansion"})
    syns = _outcome_synapses(commons, "wire:absorb:abc123")
    assert len(syns) == 1, f"one outcome deposit expected; got {len(syns)}"
    ctx = syns[0].metadata.get("last_context", {})
    assert ctx.get("kind") == "wire_outcome" and ctx.get("src") == "expansion"


def test_none_embedding_no_deposit():
    """No embedding → nothing deposited, never raises."""
    commons = _drive(None, "wire:absorb:none", True, {})
    assert _outcome_synapses(commons, "wire:absorb:none") == []


def test_no_commons_is_graceful():
    orig = commons_mod.get_commons
    commons_mod.get_commons = lambda: None
    try:
        rpc._deposit_outcome_to_river(_emb(1), "wire:absorb:x", True, {})  # must not raise
    finally:
        commons_mod.get_commons = orig


def test_deposit_failsoft_on_error():
    """A deposit failure must never break the caller — fail-soft."""
    class _BoomCommons:
        def deposit(self, *a, **k):
            raise RuntimeError("commons down")
    orig = commons_mod.get_commons
    commons_mod.get_commons = lambda: _BoomCommons()
    try:
        rpc._deposit_outcome_to_river(_emb(1), "wire:absorb:boom", True, {})  # must not raise
    finally:
        commons_mod.get_commons = orig


if __name__ == "__main__":
    test_outcome_deposits_to_commons(); print("PASS outcome deposits to Commons (target_id + metadata)")
    test_none_embedding_no_deposit();   print("PASS None embedding → no deposit, no raise")
    test_no_commons_is_graceful();      print("PASS no Commons → graceful no-op")
    test_deposit_failsoft_on_error();   print("PASS deposit error → fail-soft, caller unbroken")
    print("\nCommons outcome deposit (last throttle): ALL PASS — raw outcome flows to the Commons, fail-soft")
