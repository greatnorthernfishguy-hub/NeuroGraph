# ---- Changelog ----
# [2026-10-14] Claude Code (Opus 5.5, lane atomic-saves) — created with the PROTECTED activation_persistence.py commit
# What: SIGKILL proof for ActivationPersistence.write_state(): a forked child rewrites a 40,000-entry sidecar in a loop;
#       the parent SIGKILLs it at random points; after every kill the sidecar parses with all entries, and complete
#       writes do replace it; and the whole NeuroGraphMemory.save() set incl. the sidecar. (The in-place control lives in test_atomic_saves.py and tears.)
# Why:  #1095 class; the sidecar is Syl's / the CC's cross-session warmth.
# How:  the _kill_rounds harness from test_atomic_saves.py.
# -------------------
"""Atomic-save proof for the activation sidecar (protected file activation_persistence.py)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_atomic_saves import _clean_tmps, _kill_rounds, _ngmemory_kill_proof  # noqa: E402


def test_sigkill_activation_sidecar_never_tears(tmp_path):
    from activation_persistence import ActivationPersistence
    ap = ActivationPersistence.__new__(ActivationPersistence)
    ap._sidecar_path_for = lambda cp: cp + ".activations.json"
    ckpt = str(tmp_path / "main.msgpack")
    big = {f"n{i}": {"voltage": i * 0.001, "last_fired": i, "excitability": 1.0} for i in range(40000)}

    def loop(r):
        k = 0
        while True:
            k += 1
            ap.write_state(ckpt, {"version": "1.0", "saved_at": float(r * 100000 + k), "timestep": k,
                                  "entries": big})

    seen = []

    def verify():
        d = json.load(open(ckpt + ".activations.json"))
        assert len(d["entries"]) == 40000
        seen.append(d["saved_at"])
        _clean_tmps(tmp_path)

    ap.write_state(ckpt, {"version": "1.0", "saved_at": 0.0, "timestep": 0, "entries": big})   # the previous save
    v = _kill_rounds(loop, verify, rounds=20, seed=1, lo=0.05, hi=1.5)
    assert False not in v and v.count(True) >= 15, v
    assert any(x > 0 for x in seen), seen           # complete writes did replace the previous sidecar


def test_sigkill_neurographmemory_save_with_sidecar_never_tears(tmp_path):
    """The whole checkpoint set incl. main.msgpack.activations.json (CES activation persistence on by default)."""
    _ngmemory_kill_proof(tmp_path, check_sidecar=True)
