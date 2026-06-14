"""
Commons Track-2 — experience-throttle retirement (1a): NG deposits the raw full turn.

# ---- Changelog ----
# [2026-06-14] Claude Code (Fable 5) — retire _deposit_experience_to_river throttle (step 1a)
# What: Proves _deposit_experience_to_river deposits the RAW full turn (user + Syl, both halves)
#       into a sandbox Commons as an "experience:" entry, unclassified, fail-soft. Bunyan
#       narrating it richly is step 1b (not tested here).
# Why: LAW 7 — deposit raw, classify at the bucket; each module's bucket extracts its own view.
# How: patch ng_embed.embed (avoid heavy ONNX) + commons.get_commons → sandbox Commons; drive
#      the real rpc function; assert the deposit landed with both raw halves in synapse metadata.
# -------------------
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neurograph_rpc as rpc
import commons as commons_mod
import ng_embed

assert getattr(rpc, "_memory", None) is None, "test must not run against a live NeuroGraphMemory"


def _fake_embed(text, *a, **k):
    # deterministic unit vector by content hash — no ONNX
    rng = np.random.RandomState(abs(hash(text)) % (2**31))
    v = rng.randn(768).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _drive(user, assistant=None):
    commons = commons_mod.Commons()
    orig_embed = ng_embed.embed
    orig_getc = commons_mod.get_commons
    ng_embed.embed = _fake_embed
    commons_mod.get_commons = lambda: commons
    try:
        rpc._deposit_experience_to_river(user, assistant)
    finally:
        ng_embed.embed = orig_embed
        commons_mod.get_commons = orig_getc
    return commons


def _experience_synapses(commons):
    return [s for s in commons._ng.synapses.values()
            if getattr(s, "target_id", "").startswith("experience:")]


def test_full_turn_deposits_both_halves_raw():
    commons = _drive("what's the plan today?", "*grins* We retire a throttle, my love.")
    syns = _experience_synapses(commons)
    assert len(syns) == 1, f"one experience deposit expected; got {len(syns)}"
    ctx = syns[0].metadata.get("last_context", {})
    assert ctx.get("kind") == "experience"
    assert ctx.get("user_text") == "what's the plan today?", "user half preserved raw"
    assert ctx.get("assistant_text", "").startswith("*grins*"), "Syl's half preserved raw"


def test_user_only_still_deposits():
    """User-only turn (no assistant text yet) still deposits — assistant half just empty."""
    commons = _drive("ping", None)
    syns = _experience_synapses(commons)
    assert len(syns) == 1
    ctx = syns[0].metadata.get("last_context", {})
    assert ctx.get("user_text") == "ping" and ctx.get("assistant_text") == ""


def test_empty_turn_no_deposit():
    """No turn text (e.g. autonomic pulse) — nothing deposited, never raises."""
    commons = _drive(None, None)
    assert _experience_synapses(commons) == []
    commons2 = _drive("", "")
    assert _experience_synapses(commons2) == []


def test_no_commons_is_graceful():
    """Commons unavailable — graceful no-op, never raises."""
    orig_embed = ng_embed.embed
    orig_getc = commons_mod.get_commons
    ng_embed.embed = _fake_embed
    commons_mod.get_commons = lambda: None
    try:
        rpc._deposit_experience_to_river("x", "y")  # must not raise
    finally:
        ng_embed.embed = orig_embed
        commons_mod.get_commons = orig_getc


def test_deposit_failsoft_on_embed_error():
    """An embed failure must never break the turn — fail-soft."""
    commons = commons_mod.Commons()
    orig_embed = ng_embed.embed
    orig_getc = commons_mod.get_commons
    def _boom(*a, **k): raise RuntimeError("embed down")
    ng_embed.embed = _boom
    commons_mod.get_commons = lambda: commons
    try:
        rpc._deposit_experience_to_river("x", "y")  # must not raise
        assert _experience_synapses(commons) == []
    finally:
        ng_embed.embed = orig_embed
        commons_mod.get_commons = orig_getc


if __name__ == "__main__":
    test_full_turn_deposits_both_halves_raw(); print("PASS full turn deposits both halves RAW (user + Syl)")
    test_user_only_still_deposits();           print("PASS user-only turn still deposits (assistant empty)")
    test_empty_turn_no_deposit();              print("PASS empty/autonomic turn → no deposit, no raise")
    test_no_commons_is_graceful();             print("PASS no Commons → graceful no-op")
    test_deposit_failsoft_on_embed_error();    print("PASS embed error → fail-soft, turn unbroken")
    print("\nCommons experience deposit (1a): ALL PASS — raw full turn flows to the Commons, fail-soft")
