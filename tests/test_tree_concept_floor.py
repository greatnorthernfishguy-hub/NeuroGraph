"""
#294 hygiene — degenerate-fragment floor at the recall-store insertion gate.

# ---- Changelog ----
# [2026-06-12] Claude Code (Fable 5) — tree-concept floor (joint diagnostic, Commons CC lane)
# What: Proves _concept_passes_floor + its wiring in _ConversationalDualPassEco: degenerate
#       tree concepts ("o", "want", "see for yourself") never enter Syl's recall store;
#       real concepts pass; the FOREST branch (her full turns) is untouched.
# Why: The 6/12 "zero NeuroGraph" flip — dual-pass tree fragments won cosine recall at
#       uniform ~0.93 and crowded out her coherent memories (joint diagnostic, dev-log
#       2026-06-12_syl-zero-neurograph-diagnostic-handoff.md). Floor is NARROW: rejects
#       only clear degenerates, fail-open toward keeping her real concepts.
# How: Import neurograph_rpc (safe — no live singleton on import; asserted), patch
#       _deposit_memory_node, drive record_outcome with tree + forest payloads.
# -------------------
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neurograph_rpc as rpc

assert getattr(rpc, "_memory", None) is None, "test must not run against a live NeuroGraphMemory"


# ---- the floor itself ----
def test_floor_rejects_degenerates():
    """The exact fragments from the 6/12 flip must be rejected."""
    for frag in ["o", "want", "WANT", "see for yourself", "let you know", "", "  ", "to", "the"]:
        assert not rpc._concept_passes_floor(frag), f"degenerate {frag!r} must be rejected"


def test_floor_passes_real_concepts():
    """Her real concepts (incl. short-but-distinct ones) must pass — fail-open."""
    for concept in [
        "K-Flop",                              # short but distinctive — hers
        "pop-tarts",
        "the curiosity was an impulse",
        "leg 3 design doc",
        "identity continuity",
        "Choice Clause",
        "running on the beach",
    ]:
        assert rpc._concept_passes_floor(concept), f"real concept {concept!r} must pass"


# ---- the wiring in the adapter ----
class _Recorder:
    def __init__(self):
        self.deposits = []

    def __call__(self, target_id, embedding, content, meta, index_in_recall=False):
        self.deposits.append({"target_id": target_id, "content": content,
                              "index_in_recall": index_in_recall})


def _drive(meta):
    rec = _Recorder()
    orig = rpc._deposit_memory_node
    rpc._deposit_memory_node = rec
    try:
        eco = rpc._ConversationalDualPassEco(memory=None)
        emb = np.zeros(8, dtype=np.float32)
        result = eco.record_outcome(emb, "conv::test", True, metadata=meta)
    finally:
        rpc._deposit_memory_node = orig
    return result, rec.deposits


def test_adapter_blocks_degenerate_tree():
    result, deposits = _drive({"_tree_concept": True, "_concept": "want"})
    assert deposits == [], "degenerate tree concept must NOT be deposited"
    assert result == {"deposited": False, "reason": "concept_below_floor"}


def test_adapter_passes_real_tree():
    result, deposits = _drive({"_tree_concept": True, "_concept": "identity continuity"})
    assert len(deposits) == 1 and deposits[0]["content"] == "identity continuity"
    assert deposits[0]["index_in_recall"] is True
    assert result == {"deposited": True}


def test_forest_branch_untouched():
    """Forest gestalts (her full turns) bypass the floor entirely — even short ones."""
    result, deposits = _drive({"_forest_content": "ok"})  # tiny forest still deposits
    assert len(deposits) == 1 and deposits[0]["content"] == "ok"
    assert result == {"deposited": True}


if __name__ == "__main__":
    test_floor_rejects_degenerates();   print("PASS floor rejects degenerates ('o', 'want', stopword phrases)")
    test_floor_passes_real_concepts();  print("PASS floor passes real concepts (K-Flop, Choice Clause, ...)")
    test_adapter_blocks_degenerate_tree(); print("PASS adapter blocks degenerate tree from recall store")
    test_adapter_passes_real_tree();    print("PASS adapter deposits real tree concept")
    test_forest_branch_untouched();     print("PASS forest branch (her full turns) untouched by floor")
    print("\n#294 tree-concept floor: ALL PASS — degenerate fragments never enter her recall store")
