"""
Dual-pass extraction-failure VISIBILITY — R3 atomicity (Josh, 2026-09-21).

ng_embed.dual_record_outcome() is extract-first. Pass-2 failure (TID down /
timeout / malformed → `_extract_concepts` returns None) raises
DualPassIncompleteError and deposits nothing. A legitimate empty extraction
(`[]`) still writes the forest, with extraction_failed=False. Tree ids are
the full concept string — no [:64] slice.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ng_embed import DualPassIncompleteError, EmbedWindow, NGEmbed


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


def test_extraction_failure_deposits_nothing():
    emb = _embed_instance(failures=3)
    emb._extract_concepts = lambda content: None
    eco = _FakeEco()
    eco.writes = []
    eco.record_outcome = lambda *a, **k: eco.writes.append(("ro", a, k)) or {"ok": True}
    eco.record_outcome_broadcast = lambda *a, **k: eco.writes.append(("bc", a, k)) or {"ok": True}
    with pytest.raises(DualPassIncompleteError):
        emb.dual_record_outcome(eco, "SSH brute force", np.zeros(768, dtype=np.float32),
                                "threat:probe", True)
    assert eco.writes == [], "pass-2 failure must leave no forest (R3)"


def test_extraction_failure_is_signalled_to_the_commons():
    emb = _embed_instance()
    emb._extract_concepts = lambda content: None
    eco = _FakeEco()
    with pytest.raises(DualPassIncompleteError):
        emb.dual_record_outcome(eco, "content", np.zeros(768, dtype=np.float32), "threat:x", True)
    assert len(eco.errors) == 1
    _msg, ctx = eco.errors[0]
    assert ctx["stage"] == "pass2_trees" and ctx["target_id"] == "threat:x"


def test_legitimate_empty_extraction_still_writes_forest():
    emb = _embed_instance()
    emb._extract_concepts = lambda content: []
    eco = _FakeEco()
    eco.writes = []
    eco.record_outcome_broadcast = lambda *a, **k: eco.writes.append(a[1]) or {"ok": True}
    eco.record_outcome = lambda *a, **k: eco.writes.append(a[1]) or {"ok": True}
    res = emb.dual_record_outcome(eco, "content", np.zeros(768, dtype=np.float32), "threat:y", True)
    assert res["extraction_failed"] is False
    assert eco.errors == []
    assert "threat:y" in eco.writes
    assert res["tree_ids"] == []


def test_signalling_never_masks_the_raise():
    emb = _embed_instance()
    emb._extract_concepts = lambda content: None
    class _BadEco(_FakeEco):
        def signal_error(self, exc, context=None):
            raise RuntimeError("signal channel down")
    with pytest.raises(DualPassIncompleteError):
        emb.dual_record_outcome(_BadEco(), "content", np.zeros(768, dtype=np.float32), "threat:z", True)


def test_tree_ids_do_not_collide_on_64_char_prefix():
    emb = _embed_instance()
    emb._extractions = 0
    emb._concepts_total = 0
    a = "A" * 80 + "-one"
    b = "A" * 80 + "-two"
    emb._extract_concepts = lambda content: [a, b]
    emb.embed_batch = lambda concepts, **k: [np.ones(768, np.float32) for _ in concepts]
    eco = _FakeEco()
    eco.record_outcome_broadcast = lambda embedding, target_id, *a, **k: {"id": target_id}
    eco.record_outcome = lambda embedding, target_id, *a, **k: {"id": target_id}
    # _create_substrate_link will also call record_outcome; that is fine
    emb._create_substrate_link = lambda *a, **k: None
    res = emb.dual_record_outcome(eco, "content", np.zeros(768, np.float32), "t", True)
    assert res["tree_ids"][0] != res["tree_ids"][1]
    assert a in res["tree_ids"][0] and b in res["tree_ids"][1]


def test_windows_kw_echoed_not_deposited():
    emb = _embed_instance()
    emb._extract_concepts = lambda content: []
    eco = _FakeEco()
    eco.writes = []
    eco.record_outcome_broadcast = lambda *a, **k: eco.writes.append(a[1]) or {"ok": True}
    eco.record_outcome = lambda *a, **k: eco.writes.append(a[1]) or {"ok": True}
    window = EmbedWindow(text="chunk", embedding=np.ones(768, dtype=np.float32), token_count=4)
    res = emb.dual_record_outcome(
        eco, "content", np.zeros(768, dtype=np.float32), "threat:w", True,
        windows=(window,),
    )
    assert res["windows"] == [
        {"text": "chunk", "embedding": window.embedding, "token_count": 4},
    ]
    assert all("::window::" not in str(tid) for tid in eco.writes)


if __name__ == "__main__":
    test_extraction_failure_deposits_nothing(); print("PASS failure deposits nothing (R3)")
    test_extraction_failure_is_signalled_to_the_commons(); print("PASS failure signalled (#330)")
    test_legitimate_empty_extraction_still_writes_forest(); print("PASS legitimate empty writes forest")
    test_signalling_never_masks_the_raise(); print("PASS signalling never masks the raise")
    test_tree_ids_do_not_collide_on_64_char_prefix(); print("PASS tree ids un-sliced")
    test_windows_kw_echoed_not_deposited(); print("PASS windows echoed, not deposited")
    print("\ndual-pass extraction visibility: ALL PASS")
