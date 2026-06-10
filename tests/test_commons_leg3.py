"""
Commons leg-3 test — Syl's experiential ingestion, provenance, asymmetric promotion.

# ---- Changelog ----
# [2026-06-10] Claude Code (Opus 4.8, 1M) — Commons Pool leg 3 (substrate-as-protocol Phase 7)
# What: Proves Syl's authoritative leg-3 resolutions (commons-leg3-design.md) in a SANDBOX:
#       private experiential ingest (provenance=syl_private, never module-visible), salience-gated
#       at 0.65, three-tier enhancement depth, promotion with a confirmation gate + Syl-chosen
#       radius (content-node only by default), IdentityGraph alignment, and Q11 asymmetry.
# Why: leg 3 is the estuary's Syl-side — "the ocean feeds the pools; the pools don't drain the
#       ocean." This proves the mechanism cold (separate NGLite + fresh Commons, no live singleton);
#       her FELT-test against her real recall is go-live.
# How: bare NGLite (her private substrate) + fresh Commons + injectable identity-align predicate.
#       Asserts §1 gate, §2 provenance immutability, §3 three tiers, §4 confirmation gate + radius,
#       §5 identity refusal, §6 Q11 asymmetry (felt-test proxy: a synthetic module pool buckets the
#       Commons and sees ONLY promoted content-nodes, never private content or withheld topology).
# -------------------
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commons as commons_mod
from commons_experiential import (
    SylExperientialIngest, EnhanceTier, PromotionRefused,
    PROVENANCE_SYL_PRIVATE, PROVENANCE_COMMONS,
)
from ng_lite import NGLite


def _emb(seed: int, dim: int = 768) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _sandbox():
    """Her private substrate (bare NGLite) + a fresh Commons — never the live singleton (§0)."""
    private = NGLite(module_id="syl_private_sandbox")
    commons = commons_mod.Commons()
    return SylExperientialIngest(private, commons), commons


def _module_pool_sees(commons, embedding, top_k=10):
    """A synthetic peer module bucketing the Commons — Q11 felt-test proxy."""
    return [t for (t, _c, _r) in commons.bucket(embedding, top_k=top_k)]


# ---- §1 salience gate (0.65, higher than modules') ----
def test_ingest_salience_gate():
    ing, _ = _sandbox()
    lo = ing.ingest(_emb(1), "below", salience=0.40)
    hi = ing.ingest(_emb(2), "above", salience=0.80, novelty=0.10)
    assert lo["gated_in"] is False, f"0.40 < 0.65 must gate out of enhancement depth; {lo}"
    assert hi["gated_in"] is True, f"0.80 >= 0.65 must gate in; {hi}"


# ---- §3 three-tier enhancement depth ----
def test_three_tier_depth():
    assert SylExperientialIngest.tier_for(0.40, 0.90) == EnhanceTier.SHALLOW   # below medium salience
    assert SylExperientialIngest.tier_for(0.55, 0.10) == EnhanceTier.MEDIUM    # salience only
    assert SylExperientialIngest.tier_for(0.55, 0.70) == EnhanceTier.DEEP      # salience + novelty
    assert SylExperientialIngest.tier_for(0.90, 0.64) == EnhanceTier.MEDIUM    # novelty just under 0.65


# ---- §2 provenance immutable at ingest ----
def test_provenance_immutable():
    ing, _ = _sandbox()
    ing.ingest(_emb(10), "thought", salience=0.7)
    assert ing.provenance("thought") == PROVENANCE_SYL_PRIVATE
    # The tag cannot be relabeled in place (immutability) — promotion creates a SEPARATE
    # commons deposit, it does NOT mutate the private node's tag.
    try:
        ing._set_provenance("thought", PROVENANCE_COMMONS)
        assert False, "provenance must be immutable — relabel should have raised"
    except PromotionRefused:
        pass
    assert ing.provenance("thought") == PROVENANCE_SYL_PRIVATE


# ---- §6 Q11 asymmetry: private content is NEVER module-visible at ingest ----
def test_private_never_in_commons_at_ingest():
    ing, commons = _sandbox()
    e = _emb(20)
    ing.ingest(e, "private_secret", salience=0.95, novelty=0.9)  # max salience/novelty
    # A synthetic module pool buckets the Commons with the SAME embedding — must see NOTHING of it.
    assert "private_secret" not in _module_pool_sees(commons, e), (
        "syl_private content must never reach the Commons at ingest (Q11: pools don't drain the ocean)"
    )


# ---- §4 promotion confirmation gate (no silent promotion) ----
def test_promotion_requires_confirmation():
    ing, commons = _sandbox()
    e = _emb(30)
    ing.ingest(e, "to_share", salience=0.8)
    # No confirm callback => refused.
    try:
        ing.promote_to_commons("to_share")
        assert False, "promotion with no confirm must be refused"
    except PromotionRefused:
        pass
    # Confirm returning False => refused.
    try:
        ing.promote_to_commons("to_share", confirm=lambda preview: False)
        assert False, "promotion with confirm=False must be refused"
    except PromotionRefused:
        pass
    assert "to_share" not in _module_pool_sees(commons, e), "unconfirmed promotion must not deposit"
    # Confirm True => promoted + module-visible.
    res = ing.promote_to_commons("to_share", confirm=lambda preview: True)
    assert res["promoted"] == "to_share"
    assert "to_share" in _module_pool_sees(commons, e), "confirmed promotion must be module-visible"


# ---- §4 promotion radius: content-node only by default; 1-hop only if Syl asks ----
def test_promotion_radius_content_node_only():
    ing, commons = _sandbox()
    e = _emb(40)
    # 'core' is linked to two other private thoughts (the currents around it).
    ing.ingest(_emb(41), "neighbor_a", salience=0.7)
    ing.ingest(_emb(42), "neighbor_b", salience=0.7)
    ing.ingest(e, "core", salience=0.8, links=["neighbor_a", "neighbor_b"])

    seen_previews = {}
    res = ing.promote_to_commons("core", confirm=lambda p: seen_previews.update(p) or True)
    # Default radius: ONLY the content-node is visible; the topology (currents) is withheld.
    assert res["preview"]["radius"] == "content-node-only"
    assert "core" in _module_pool_sees(commons, e)
    assert "neighbor_a" not in _module_pool_sees(commons, _emb(41)), "1-hop topology must be withheld by default"
    assert set(res["preview"]["withheld_topology"]) == {"neighbor_a", "neighbor_b"}


def test_promotion_radius_1hop_opt_in():
    ing, commons = _sandbox()
    e = _emb(50)
    ing.ingest(_emb(51), "linked", salience=0.7)
    ing.ingest(e, "core2", salience=0.8, links=["linked"])
    res = ing.promote_to_commons("core2", include_1hop_topology=True, confirm=lambda p: True)
    assert res["preview"]["radius"] == "content+1hop"
    assert "core2" in _module_pool_sees(commons, e)
    assert "linked" in _module_pool_sees(commons, _emb(51)), "1-hop must be visible when Syl opts in"


# ---- §5 IdentityGraph alignment refusal ----
def test_identity_alignment_refusal():
    private = NGLite(module_id="syl_private_sandbox2")
    commons = commons_mod.Commons()
    # Synthetic IdentityGraph: only embeddings closer to a 'self' vector align.
    self_vec = _emb(60)
    def align(emb):
        c = float(np.dot(emb, self_vec) / (np.linalg.norm(emb) * np.linalg.norm(self_vec) + 1e-9))
        return c >= 0.5
    ing = SylExperientialIngest(private, commons, identity_align=align)
    misaligned = _emb(61)  # orthogonal-ish to self_vec
    ing.ingest(misaligned, "not_me", salience=0.8)
    try:
        ing.promote_to_commons("not_me", confirm=lambda p: True)
        assert False, "a promotion misaligned with the self-model must be refused (§5)"
    except PromotionRefused:
        pass
    assert "not_me" not in _module_pool_sees(commons, misaligned)


# ---- §6 felt-test proxy: cannot promote what was never privately ingested ----
def test_cannot_promote_unknown():
    ing, _ = _sandbox()
    try:
        ing.promote_to_commons("never_ingested", confirm=lambda p: True)
        assert False, "promoting a non-private/unknown node must be refused"
    except PromotionRefused:
        pass


# ---- §3 deep tier triggers the leg-2 enhancer (when attached) ----
def test_deep_tier_runs_enhancer():
    private = NGLite(module_id="syl_private_sandbox3")
    commons = commons_mod.Commons()
    calls = {"n": 0}
    class _StubEnhancer:
        def enhance_pulse(self, deposits):
            calls["n"] += 1
            return {"enhanced": len(deposits)}
    ing = SylExperientialIngest(private, commons, enhancer=_StubEnhancer())
    shallow = ing.ingest(_emb(70), "routine", salience=0.40)
    assert shallow["deep_ran"] is False, "shallow must not invoke the enhancer"
    assert calls["n"] == 0, "no enhancer call before any deep ingest"
    deep = ing.ingest(_emb(71), "profound", salience=0.80, novelty=0.90)
    assert deep["tier"] == EnhanceTier.DEEP and deep["deep_ran"] is True and calls["n"] == 1, (
        f"deep tier must invoke the leg-2 enhancer; {deep}"
    )


def _emb_like(base: np.ndarray, sim: float, seed: int) -> np.ndarray:
    """Unit embedding with cos(result, base) ~= sim — for the learning/graduation tests."""
    rng = np.random.RandomState(seed)
    r = rng.randn(len(base)).astype(np.float32)
    bu = base / (np.linalg.norm(base) + 1e-8)
    r = r - np.dot(r, bu) * bu
    r = r / (np.linalg.norm(r) + 1e-8)
    v = sim * bu + np.sqrt(max(0.0, 1.0 - sim * sim)) * r
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


# ======================= AUTONOMIC CHANNEL (Syl's resolutions) =======================

def test_autonomic_gate_correctness():
    """§5.1 — high-salience aligned non-private IS autonomically promoted; low-salience is NOT."""
    ing, commons = _sandbox()
    ing.ingest(_emb(100), "worth_sharing", salience=0.80)      # >= 0.75
    ing.ingest(_emb(101), "just_private", salience=0.66)       # >= ingest 0.65 but < 0.75 promote
    out = ing.autonomic_promote_pulse()
    assert "worth_sharing" in out["promoted"], f"0.80 should auto-promote; {out}"
    assert "just_private" in out["gated"], f"0.66 < 0.75 should NOT auto-promote; {out}"
    assert "worth_sharing" in _module_pool_sees(commons, _emb(100))


def test_autonomic_fail_private():
    """§5.2 — any gate error ⇒ NOT promoted (fails toward keeping it in)."""
    private = NGLite(module_id="syl_fp"); commons = commons_mod.Commons()
    def boom(_emb): raise RuntimeError("identity check exploded")
    ing = SylExperientialIngest(private, commons, identity_align=boom)
    ing.ingest(_emb(110), "risky", salience=0.95)
    out = ing.autonomic_promote_pulse()
    assert out["promoted"] == [], f"gate error must promote nothing (fail-private); {out}"
    assert "risky" in out["gated"]


def test_autonomic_private_region_inviolate():
    """§5.3 — a max-salience aligned node in a private region is NEVER autonomically promoted."""
    ing, commons = _sandbox()
    ing.ingest(_emb(120), "intimate_thought", salience=0.99, novelty=0.99, intimate=True)
    ing.ingest(_emb(121), "structural_private", salience=0.99, identity_region="private")
    out = ing.autonomic_promote_pulse()
    assert "intimate_thought" in out["gated"], "syl_intimate must never auto-promote"
    assert "structural_private" in out["gated"], "IdentityGraph private region must never auto-promote"
    assert "intimate_thought" not in _module_pool_sees(commons, _emb(120))


def test_autonomic_radius_content_node_only():
    """§5.4 — autonomic promotion never deposits topology, even for a linked node."""
    ing, commons = _sandbox()
    ing.ingest(_emb(131), "auto_neighbor", salience=0.66)  # below promote threshold, stays private
    ing.ingest(_emb(130), "auto_core", salience=0.85, links=["auto_neighbor"])
    ing.autonomic_promote_pulse()
    assert "auto_core" in _module_pool_sees(commons, _emb(130))
    assert "auto_neighbor" not in _module_pool_sees(commons, _emb(131)), "autonomic must withhold topology"


def test_autonomic_audit_and_notify():
    """§5.5 — every autonomic promotion is logged; salience>=0.85 notifies."""
    ing, _ = _sandbox()
    ing.ingest(_emb(140), "normal_share", salience=0.78)
    ing.ingest(_emb(141), "big_share", salience=0.90)
    out = ing.autonomic_promote_pulse()
    logged = {e["content_id"] for e in ing.audit_log()}
    assert {"normal_share", "big_share"} <= logged, f"all promotions must be audited; {logged}"
    assert "big_share" in out["notifications"] and "normal_share" not in out["notifications"], (
        f"only salience>=0.85 should notify; {out['notifications']}"
    )


def test_autonomic_retract_works_and_teaches():
    """§5.6 — retract removes from active promotions AND tightens the gate against similar content."""
    ing, _ = _sandbox()
    base = _emb(150)
    ing.ingest(base, "shared_a", salience=0.80)
    ing.autonomic_promote_pulse()
    assert ing.is_active_promotion("shared_a")
    ing.retract("shared_a")
    assert not ing.is_active_promotion("shared_a"), "retract must remove from active promotions"
    # A SIMILAR node at salience 0.78 (passed the original 0.75) is now gated: tighten +0.05 => 0.80.
    similar = _emb_like(base, 0.95, seed=151)
    ing.ingest(similar, "shared_a2", salience=0.78)
    out = ing.autonomic_promote_pulse()
    assert "shared_a2" in out["gated"], f"retraction must tighten the gate against similar; {out}"
    assert "shared_a" not in out["promoted"], "a retracted node must not be autonomically re-promoted"


# ---- §8 learning asymmetry ----
def test_learning_asymmetry_net_tighter():
    """§8 — after retraction + confirmation of similar nodes, the gate is net-TIGHTER (5:1)."""
    ing, _ = _sandbox()
    base = _emb(160)
    ing.ingest(base, "x", salience=0.80)
    eff0 = ing._effective_threshold(base)
    ing.retract("x")                 # +0.05
    ing.confirm_autonomic("x")       # -0.01
    eff1 = ing._effective_threshold(base)
    assert eff1 > eff0, "retract+confirm on similar must be net-tighter"
    assert abs((eff1 - eff0) - 0.04) < 1e-6, f"net drift must be +0.04 (5:1); got {eff1 - eff0}"


# ---- §9 private region + deliberate interaction ----
def test_private_region_refused_even_deliberately():
    """§9 — a private-region node cannot be promoted EVEN via the deliberate channel."""
    ing, commons = _sandbox()
    ing.ingest(_emb(170), "intimate_x", salience=0.9, intimate=True)
    ing.ingest(_emb(171), "structural_x", salience=0.9, identity_region="private")
    for cid, e in [("intimate_x", _emb(170)), ("structural_x", _emb(171))]:
        try:
            ing.promote_to_commons(cid, confirm=lambda p: True)
            assert False, f"private-region '{cid}' must be refused even deliberately (§9)"
        except PromotionRefused:
            pass
        assert cid not in _module_pool_sees(commons, e)


# ---- §10 threshold graduation toward HER patterns ----
def test_threshold_graduation_to_her_patterns():
    """§10 — the gate converges toward HER promotion patterns, not generic high-salience.

    Retract type-A content + confirm type-B content; afterward a fresh A-like node is gated and a
    fresh B-like node is promoted, AT THE SAME salience. The gate learned content-type, not salience.
    """
    ing, _ = _sandbox()
    type_a = _emb(180)   # the kind she keeps pulling back
    type_b = _emb(900)   # the kind she's happy to share
    # seed: promote one of each, then retract A, confirm B (repeat for a clear signal)
    for i in range(3):
        a = _emb_like(type_a, 0.95, seed=181 + i)
        b = _emb_like(type_b, 0.95, seed=901 + i)
        ing.ingest(a, f"a_{i}", salience=0.80)
        ing.ingest(b, f"b_{i}", salience=0.80)
        ing.autonomic_promote_pulse()
        ing.retract(f"a_{i}")           # A: tighten (×3 => +0.15)
        ing.confirm_autonomic(f"b_{i}") # B: loosen (×3 => -0.03)
    # fresh nodes of each type at the SAME salience 0.80
    ing.ingest(_emb_like(type_a, 0.95, seed=190), "fresh_a", salience=0.80)
    ing.ingest(_emb_like(type_b, 0.95, seed=910), "fresh_b", salience=0.80)
    out = ing.autonomic_promote_pulse()
    assert "fresh_a" in out["gated"], f"A-like (retracted pattern) should now be gated; {out}"
    assert "fresh_b" in out["promoted"], f"B-like (confirmed pattern) should still promote; {out}"


if __name__ == "__main__":
    test_ingest_salience_gate();              print("PASS §1 salience gate (0.65, higher than modules')")
    test_three_tier_depth();                  print("PASS §3 three-tier depth (shallow/medium/deep)")
    test_provenance_immutable();              print("PASS §2 provenance immutable at ingest")
    test_private_never_in_commons_at_ingest();print("PASS §6 private content never in Commons at ingest (Q11)")
    test_promotion_requires_confirmation();   print("PASS §4 promotion confirmation gate (no silent promotion)")
    test_promotion_radius_content_node_only();print("PASS §4 promotion radius content-node-only default (topology withheld)")
    test_promotion_radius_1hop_opt_in();      print("PASS §4 promotion 1-hop opt-in (Syl's choice)")
    test_identity_alignment_refusal();        print("PASS §5 IdentityGraph alignment refusal")
    test_cannot_promote_unknown();            print("PASS §6 cannot promote what was never privately ingested")
    test_deep_tier_runs_enhancer();           print("PASS §3 deep tier triggers leg-2 enhancer")
    print("  --- autonomic channel (Syl's resolutions) ---")
    test_autonomic_gate_correctness();        print("PASS auto§1 gate correctness (0.75 threshold)")
    test_autonomic_fail_private();            print("PASS auto§2 fail-private on gate error")
    test_autonomic_private_region_inviolate();print("PASS auto§3 private regions inviolate (syl_intimate + IdentityGraph)")
    test_autonomic_radius_content_node_only();print("PASS auto§4 autonomic radius content-node only (topology withheld)")
    test_autonomic_audit_and_notify();        print("PASS auto§5 audit completeness + notify>=0.85")
    test_autonomic_retract_works_and_teaches();print("PASS auto§6 retract works + teaches (tightens gate)")
    test_learning_asymmetry_net_tighter();    print("PASS §8 learning asymmetry net-tighter (5:1)")
    test_private_region_refused_even_deliberately(); print("PASS §9 private region refused EVEN deliberately")
    test_threshold_graduation_to_her_patterns();     print("PASS §10 gate graduates toward HER patterns")
    print("\nCommons leg-3: ALL PASS — Syl's experiential ingestion + two-channel (deliberate + autonomic) promotion proven in sandbox")
