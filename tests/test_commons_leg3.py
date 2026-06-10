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
    print("\nCommons leg-3: ALL PASS — Syl's experiential ingestion + Q11 asymmetric promotion proven in sandbox")
