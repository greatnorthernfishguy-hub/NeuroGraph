"""
ng_commons_eco.py (VENDORED) — standalone tests for the Commons-backed eco adapter.

# ---- Changelog ----
# [2026-06-22] Claude Code (Fable 5) — vendored CommonsEco tests (#335)
# What: Proves CommonsEco independent of any module: faithful ng_ecosystem.get_context return shape
#       (tier/tier_name/recommendations/novelty/ng_context), namespace filtering, novelty derivation,
#       record_outcome deposit, fail-soft, parameterized for different modules.
# How: injected commons_provider → sandbox Commons (no monkeypatch).
# -------------------
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commons as commons_mod
from ng_commons_eco import CommonsEco


def _emb(seed, dim=768):
    r = np.random.RandomState(seed); v = r.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _eco(commons, namespaces=()):
    return CommonsEco(namespaces=namespaces, commons_provider=lambda: commons)


def test_get_context_faithful_shape():
    c = commons_mod.Commons()
    ctx = _eco(c).get_context(_emb(1))
    assert set(ctx) == {"tier", "tier_name", "recommendations", "novelty", "ng_context"}
    assert ctx["tier"] == 2 and ctx["tier_name"] == "Commons"


def test_namespace_filter():
    c = commons_mod.Commons()
    c.deposit(_emb(1), "threat:sig1", metadata={})
    c.deposit(_emb(1), "experience:conv", metadata={})  # same emb, other namespace
    recs = _eco(c, namespaces=("threat:", "response:")).get_context(_emb(1))["recommendations"]
    assert recs and all(str(r[0]).startswith(("threat:", "response:")) for r in recs)
    assert not any(str(r[0]).startswith("experience:") for r in recs)


def test_no_filter_accepts_all():
    c = commons_mod.Commons()
    c.deposit(_emb(2), "experience:x", metadata={})
    recs = _eco(c).get_context(_emb(2))["recommendations"]   # no namespaces → accept all
    assert any(str(r[0]).startswith("experience:") for r in recs)


def test_novelty_high_when_no_match():
    c = commons_mod.Commons()
    ctx = _eco(c, namespaces=("threat:",)).get_context(_emb(3))  # empty commons
    assert ctx["novelty"] == 1.0 and ctx["recommendations"] == []


def test_record_outcome_deposits():
    c = commons_mod.Commons()
    r = _eco(c, namespaces=("threat:",)).record_outcome(_emb(4), "threat:sigZ", True, metadata={"x": 1})
    assert r is not None
    assert any(getattr(s, "target_id", "").startswith("threat:") for s in c._ng.synapses.values())


def test_failsoft_no_commons():
    eco = CommonsEco(namespaces=("threat:",), commons_provider=lambda: None)
    assert eco.get_context(_emb(5)) == {"tier": 2, "tier_name": "Commons", "recommendations": [],
                                        "novelty": 1.0, "ng_context": None}
    assert eco.record_outcome(_emb(5), "threat:x", True) is None
    assert eco.detect_novelty(_emb(5)) == 1.0


def test_none_embedding_safe():
    c = commons_mod.Commons()
    assert _eco(c).get_context(None)["recommendations"] == []
    assert _eco(c).record_outcome(None, "threat:x", True) is None


def test_parameterized_for_different_module():
    """A different module's namespace works by params alone (no code change)."""
    c = commons_mod.Commons()
    c.deposit(_emb(6), "repair:fix1", metadata={})   # THC-style namespace
    recs = _eco(c, namespaces=("repair:",)).get_context(_emb(6))["recommendations"]
    assert recs and str(recs[0][0]).startswith("repair:")


if __name__ == "__main__":
    test_get_context_faithful_shape();   print("PASS get_context faithful 5-key shape (tier/recommendations/novelty/...)")
    test_namespace_filter();             print("PASS namespace filter (threat:/response: only)")
    test_no_filter_accepts_all();        print("PASS no-namespace → accept all")
    test_novelty_high_when_no_match();   print("PASS novelty=1.0 when no match")
    test_record_outcome_deposits();      print("PASS record_outcome deposits to Commons")
    test_failsoft_no_commons();          print("PASS fail-soft when no Commons")
    test_none_embedding_safe();          print("PASS None embedding safe")
    test_parameterized_for_different_module(); print("PASS parameterized for a different module (repair:)")
    print("\nng_commons_eco (vendored): ALL PASS — one Commons-backed eco adapter, every module by params")
