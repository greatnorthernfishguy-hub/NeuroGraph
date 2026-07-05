"""
Commons persist/restore lifecycle (#332) — round-trip survives restart; arousal fresh-assesses.

# ---- Changelog ----
# [2026-07-05] Claude Code (Sonnet 5) — #332 persist/restore wired into neurograph_rpc.py
# What: Proves Commons.persist()/restore() round-trip experience/topology/metrics/repair
#       synapses through disk, and that restore() drops autonomic:* synapses so arousal
#       always fresh-assesses on restart (#328 Decision #2) instead of resurrecting a
#       possibly-stale SYMPATHETIC/PARASYMPATHETIC verdict.
# Why: persist()/restore() were defined with zero callers — Commons was wiped every gateway
#      restart. Wiring them in (neurograph_rpc.py bootstrap + auto-save + shutdown) needed
#      test coverage that (a) the round-trip actually works and (b) the one deliberate
#      exception (autonomic) behaves correctly rather than silently reintroducing a stale
#      arousal read the #328 design explicitly rejected.
# How: deposit into a sandbox Commons, persist to a tmp file, restore into a FRESH Commons
#      instance, assert survivors + the autonomic drop.
"""

import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commons as commons_mod


def _emb(seed, dim=768):
    r = np.random.RandomState(seed)
    v = r.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _count(commons, prefix):
    return sum(1 for s in commons._ng.synapses.values()
               if getattr(s, "target_id", "").startswith(prefix))


def test_persist_restore_round_trip():
    """Experience/topology/metrics/repair synapses survive a persist -> restore cycle."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "commons_test.msgpack")

        c1 = commons_mod.Commons()
        c1.deposit(_emb(1), "experience:hello", metadata={"kind": "experience"})
        c1.deposit(_emb(2), "topology:n1", metadata={"kind": "topology_delta"})
        c1.deposit(_emb(3), "metrics:neurograph:nominal:h1:1.0:1", metadata={"kind": "metrics"})
        c1.deposit(_emb(4), "repair:process_restart", metadata={"kind": "outcome"})
        c1.persist(path)

        c2 = commons_mod.Commons()
        assert _count(c2, "experience:") == 0, "fresh Commons starts empty before restore"
        c2.restore(path)

        assert _count(c2, "experience:") == 1
        assert _count(c2, "topology:") == 1
        assert _count(c2, "metrics:neurograph:nominal:") == 1
        assert _count(c2, "repair:") == 1


def test_restore_drops_autonomic_synapses():
    """restore() removes autonomic:* synapses — arousal fresh-assesses, never resurrected (#328)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "commons_test.msgpack")

        c1 = commons_mod.Commons()
        c1.deposit(_emb(1), "autonomic:arousal",
                   metadata={"state": "SYMPATHETIC", "threat_level": "critical",
                             "triggered_by": "immunis", "reason": "test", "ts": 123.0})
        c1.deposit(_emb(2), "experience:hello", metadata={"kind": "experience"})
        c1.persist(path)

        c2 = commons_mod.Commons()
        c2.restore(path)

        assert _count(c2, "autonomic:") == 0, "autonomic:* must be dropped on restore"
        assert _count(c2, "experience:") == 1, "non-autonomic synapses are unaffected"
        # the vagus bucket falls back to its safe default, not a stale SYMPATHETIC
        assert c2.read_arousal() == "PARASYMPATHETIC"


def test_restore_missing_file_is_noop():
    """restore() on a nonexistent path doesn't raise (callers guard with os.path.exists,
    but the method itself should not explode if called directly)."""
    commons = commons_mod.Commons()
    try:
        commons.restore("/nonexistent/path/commons_test.msgpack")
    except Exception as exc:
        assert False, f"restore() on a missing file must not raise: {exc}"


if __name__ == "__main__":
    test_persist_restore_round_trip();          print("PASS persist/restore round-trip survives")
    test_restore_drops_autonomic_synapses();    print("PASS autonomic:* dropped on restore (#328 fresh-assess)")
    test_restore_missing_file_is_noop();        print("PASS restore on missing file is a no-op")
    print("\nCommons persist/restore (#332): ALL PASS")
