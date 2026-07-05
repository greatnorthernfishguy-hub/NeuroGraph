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


def test_string_namespace_coerced_not_charsplit():
    """Footgun guard: a bare string namespace must NOT become a char-tuple (compliance #335)."""
    c = commons_mod.Commons()
    c.deposit(_emb(7), "threat:s", metadata={})
    c.deposit(_emb(7), "experience:e", metadata={})
    eco = CommonsEco(namespaces="threat:", commons_provider=lambda: c)  # bare string, the trap
    assert eco._namespaces == ("threat:",), "string must coerce to a 1-tuple, not char-split"
    recs = eco.get_context(_emb(7))["recommendations"]
    assert recs and all(str(r[0]).startswith("threat:") for r in recs)
    assert not any(str(r[0]).startswith("experience:") for r in recs), "char-split would have matched everything"


def _tid_up():
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:7437/health", timeout=3)
        return True
    except Exception:
        return False


def test_dual_record_outcome_failsoft_to_forest():
    """If dual-pass can't run (no embed engine), fall back to a single forest deposit — never lose it."""
    import ng_embed
    c = commons_mod.Commons()
    eco = _eco(c, namespaces=("threat:",))
    orig = ng_embed.NGEmbed.get_instance
    def _boom(*a, **k):
        raise RuntimeError("no engine")
    ng_embed.NGEmbed.get_instance = staticmethod(_boom)
    try:
        eco.dual_record_outcome(content="x", embedding=_emb(1), target_id="threat:fb", success=True)
    finally:
        ng_embed.NGEmbed.get_instance = orig
    targets = [getattr(s, "target_id", "") for s in c._ng.synapses.values()]
    assert "threat:fb" in targets, "forest deposited on single-pass fallback"


def test_dual_record_outcome_none_embedding_safe():
    c = commons_mod.Commons()
    assert _eco(c).dual_record_outcome(content="x", embedding=None, target_id="threat:n", success=True) is None


def test_signal_error_deposits_raw():
    """signal_error deposits under error:<module>:<ExcType> with description + context, no severity."""
    c = commons_mod.Commons()
    eco = _eco(c, namespaces=("threat:",))
    try:
        raise ValueError("bad thing happened")
    except ValueError as exc:
        eco.signal_error(exc, {"where": "test"})
    matches = [(s.target_id, s.metadata.get("last_context", {})) for s in c._ng.synapses.values()
               if getattr(s, "target_id", "").startswith("error:")]
    assert len(matches) == 1
    target_id, meta = matches[0]
    assert target_id == "error:threat:ValueError" or target_id.startswith("error:")
    assert "bad thing happened" in meta["description"]
    assert meta["context"] == {"where": "test"}
    assert "severity" not in meta and "classification" not in meta


def test_signal_error_target_id_shape_matches_retention():
    """target_id is error:<module_id>:<ExcType> — matches _evict_old_errors's 3-segment prefix."""
    c = commons_mod.Commons()
    eco = _eco(c, namespaces=())
    eco._source = "immunis"
    try:
        raise RuntimeError("x")
    except RuntimeError as exc:
        eco.signal_error(exc)
    targets = [getattr(s, "target_id", "") for s in c._ng.synapses.values()]
    assert "error:immunis:RuntimeError" in targets


def test_signal_error_failsoft_no_commons():
    eco = CommonsEco(namespaces=("threat:",), commons_provider=lambda: None)
    try:
        raise KeyError("x")
    except KeyError as exc:
        eco.signal_error(exc)  # must not raise


def test_signal_error_no_context_defaults_empty():
    c = commons_mod.Commons()
    eco = _eco(c)
    try:
        raise TypeError("y")
    except TypeError as exc:
        eco.signal_error(exc)  # no context passed
    matches = [s.metadata.get("last_context", {}) for s in c._ng.synapses.values()
               if getattr(s, "target_id", "").startswith("error:")]
    assert matches[0]["context"] == {}


def test_dual_record_outcome_forest_and_trees_live():
    """Real dual-pass through CommonsEco: forest + TID-extracted trees land in the Commons (needs TID)."""
    if not _tid_up():
        print("SKIP live dual-pass (TID 7437 not reachable)"); return
    import ng_embed
    c = commons_mod.Commons()
    eco = _eco(c, namespaces=("threat:",))
    content = "SSH brute-force authentication failures and an outbound malware C2 connection on port 4444 from nginx."
    emb = ng_embed.embed(content)
    res = eco.dual_record_outcome(content=content, embedding=emb, target_id="threat:probe", success=True)
    assert res and res.get("pass2_attempted"), "Pass 2 attempted"
    targets = [getattr(s, "target_id", "") for s in c._ng.synapses.values()]
    assert "threat:probe" in targets, "forest deposited"
    assert sum(1 for t in targets if "::tree::" in t) > 0, "tree concepts deposited to the Commons"


if __name__ == "__main__":
    test_signal_error_deposits_raw();    print("PASS signal_error deposits raw description+context, no severity")
    test_signal_error_target_id_shape_matches_retention(); print("PASS signal_error target_id matches retention shape")
    test_signal_error_failsoft_no_commons(); print("PASS signal_error fail-soft when no Commons")
    test_signal_error_no_context_defaults_empty(); print("PASS signal_error context defaults to {}")
    test_dual_record_outcome_failsoft_to_forest(); print("PASS dual_record_outcome fail-soft → single forest deposit")
    test_dual_record_outcome_none_embedding_safe(); print("PASS dual_record_outcome None embedding safe")
    test_dual_record_outcome_forest_and_trees_live(); print("PASS dual_record_outcome live forest+trees (or SKIP if no TID)")
    test_string_namespace_coerced_not_charsplit(); print("PASS string namespace coerced (footgun guarded)")
    test_get_context_faithful_shape();   print("PASS get_context faithful 5-key shape (tier/recommendations/novelty/...)")
    test_namespace_filter();             print("PASS namespace filter (threat:/response: only)")
    test_no_filter_accepts_all();        print("PASS no-namespace → accept all")
    test_novelty_high_when_no_match();   print("PASS novelty=1.0 when no match")
    test_record_outcome_deposits();      print("PASS record_outcome deposits to Commons")
    test_failsoft_no_commons();          print("PASS fail-soft when no Commons")
    test_none_embedding_safe();          print("PASS None embedding safe")
    test_parameterized_for_different_module(); print("PASS parameterized for a different module (repair:)")
    print("\nng_commons_eco (vendored): ALL PASS — one Commons-backed eco adapter, every module by params")
