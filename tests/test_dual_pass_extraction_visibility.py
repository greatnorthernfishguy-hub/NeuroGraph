"""
Dual-pass extraction-failure VISIBILITY — no silent failures (Josh, 2026-07-15).

ng_embed.dual_record_outcome() runs Pass 1 (forest) + Pass 2 (TID concept extraction → trees).
When TID breaks (down / timeout / malformed), the tree half is lost. This MUST NOT be silent:
_extract_concepts returns None (distinct from a legitimate empty []), dual_record_outcome sets
result["extraction_failed"]=True, logs a warning, and signals the Commons operational-logger (#330)
via ecosystem.signal_error. A legitimately-empty extraction ([]) is NOT a failure and stays quiet.

These tests are deterministic (the failure/empty are simulated) — they do NOT depend on a live TID,
unlike test_ng_commons_eco.py::test_dual_record_outcome_forest_and_trees_live.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ng_embed import NGEmbed


class _FakeEco:
    """Minimal ecosystem stand-in: records Pass-1 outcomes and captures signal_error calls."""
    def __init__(self):
        self.errors = []
    def record_outcome_broadcast(self, *a, **k):
        return {"ok": True}
    def record_outcome(self, *a, **k):
        return {"ok": True}
    def signal_error(self, exc, context=None):
        self.errors.append((str(exc), context))


def _embed_instance(failures=0):
    # Bypass __init__ (no ONNX model load needed) — only _failures + the patched _extract_concepts
    # are touched on the failure/empty paths.
    emb = NGEmbed.__new__(NGEmbed)
    emb._failures = failures
    return emb


def test_extraction_failure_is_surfaced_not_silent():
    emb = _embed_instance(failures=3)
    emb._extract_concepts = lambda content: None            # simulate a TID failure
    eco = _FakeEco()
    res = emb.dual_record_outcome(eco, "SSH brute force", np.zeros(768, dtype=np.float32),
                                  "threat:probe", True)
    assert res["pass2_attempted"] is True
    assert res["extraction_failed"] is True, "a broken TID extraction must be surfaced, not silent"
    assert res["tree_ids"] == [], "no trees deposited on failure (forest-only degradation)"


def test_extraction_failure_is_signalled_to_the_commons():
    emb = _embed_instance()
    emb._extract_concepts = lambda content: None
    eco = _FakeEco()
    emb.dual_record_outcome(eco, "content", np.zeros(768, dtype=np.float32), "threat:x", True)
    assert len(eco.errors) == 1, "failure must be signalled via signal_error (#330 operational-logger)"
    _msg, ctx = eco.errors[0]
    assert ctx["stage"] == "pass2_trees" and ctx["target_id"] == "threat:x"


def test_legitimate_empty_extraction_is_not_a_failure():
    emb = _embed_instance()
    emb._extract_concepts = lambda content: []              # TID ran fine, found no concepts
    eco = _FakeEco()
    res = emb.dual_record_outcome(eco, "content", np.zeros(768, dtype=np.float32), "threat:y", True)
    assert res["extraction_failed"] is False, "an empty result is not a failure"
    assert eco.errors == [], "a legitimate empty extraction must NOT be signalled as an error"
    assert res["tree_ids"] == []


def test_signalling_never_breaks_the_deposit():
    """If signal_error itself throws, the deposit result is still returned (fail-soft)."""
    emb = _embed_instance()
    emb._extract_concepts = lambda content: None
    class _BadEco(_FakeEco):
        def signal_error(self, exc, context=None):
            raise RuntimeError("signal channel down")
    res = emb.dual_record_outcome(_BadEco(), "content", np.zeros(768, dtype=np.float32), "threat:z", True)
    assert res["extraction_failed"] is True  # still surfaced in the result even if signalling failed


if __name__ == "__main__":
    test_extraction_failure_is_surfaced_not_silent();  print("PASS failure surfaced (extraction_failed)")
    test_extraction_failure_is_signalled_to_the_commons(); print("PASS failure signalled (#330)")
    test_legitimate_empty_extraction_is_not_a_failure(); print("PASS legitimate empty ≠ failure")
    test_signalling_never_breaks_the_deposit(); print("PASS signalling fail-soft")
    print("\ndual-pass extraction visibility: ALL PASS")
