"""
#328 Step 2 — Commons.read_arousal (the vagus-nerve bucket).

# ---- Changelog ----
# [2026-06-22] Claude Code (Opus 4.8) — read_arousal test
# What: Proves Commons.read_arousal returns the latest autonomic:arousal deposit's state, defaults
#       PARASYMPATHETIC (fresh-assess), latest-wins, and — critically — RELIABLY finds the arousal
#       deposit even when buried under many later deposits (design subtlety #2: the vagus is never
#       missed; a recency-window scan would lose it, the direct lookup does not).
# Why: #328 — readers bucket arousal from the Commons instead of the shared ng_autonomic file.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import commons as commons_mod


def _emb(seed, dim=768):
    r = np.random.RandomState(seed); v = r.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def test_default_parasympathetic_when_no_deposit():
    assert commons_mod.Commons().read_arousal() == "PARASYMPATHETIC"


def test_returns_deposited_state():
    c = commons_mod.Commons()
    c.deposit(_emb(1), "autonomic:arousal", metadata={"state": "SYMPATHETIC", "threat_level": "critical"})
    assert c.read_arousal() == "SYMPATHETIC"


def test_latest_wins():
    c = commons_mod.Commons()
    c.deposit(_emb(1), "autonomic:arousal", metadata={"state": "SYMPATHETIC"})
    c.deposit(_emb(2), "autonomic:arousal", metadata={"state": "PARASYMPATHETIC"})
    assert c.read_arousal() == "PARASYMPATHETIC"


def test_reliable_under_deposit_load():
    """Subtlety #2: arousal is low-frequency — it must NEVER be missed when buried by later deposits.
    A recency-window scan (limit=50) would lose it; the direct lookup must not."""
    c = commons_mod.Commons()
    c.deposit(_emb(1), "autonomic:arousal", metadata={"state": "SYMPATHETIC"})
    for i in range(200):
        c.deposit(_emb(1000 + i), f"metrics:neurograph:x:{i}", metadata={"state": "ignore"})
    assert c.read_arousal() == "SYMPATHETIC", "arousal must survive burial under 200 later deposits"


def test_custom_default():
    assert commons_mod.Commons().read_arousal(default="PARASYMPATHETIC") == "PARASYMPATHETIC"


def test_arousal_full_dict_carries_threat_level():
    """arousal() returns the full latest deposit metadata (state + threat_level) for callers
    that need modulation intensity (e.g. Elmer engine); read_arousal() is the state-only view."""
    c = commons_mod.Commons()
    assert c.arousal()["state"] == "PARASYMPATHETIC" and c.arousal()["threat_level"] == "none"
    c.deposit(_emb(1), "autonomic:arousal", metadata={"state": "SYMPATHETIC", "threat_level": "critical"})
    a = c.arousal()
    assert a["state"] == "SYMPATHETIC" and a["threat_level"] == "critical"
    assert c.read_arousal() == "SYMPATHETIC"  # delegation intact


if __name__ == "__main__":
    test_default_parasympathetic_when_no_deposit(); print("PASS default PARASYMPATHETIC (fresh-assess)")
    test_returns_deposited_state();                 print("PASS returns deposited arousal state")
    test_latest_wins();                             print("PASS latest deposit wins")
    test_reliable_under_deposit_load();             print("PASS reliable under deposit load (vagus never missed)")
    test_custom_default();                          print("PASS custom default")
    test_arousal_full_dict_carries_threat_level();  print("PASS arousal() full dict carries threat_level")
    print("\n#328 Step 2 (Commons.read_arousal): ALL PASS — the vagus-nerve bucket")
