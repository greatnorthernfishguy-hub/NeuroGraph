"""
Commons POC smoke test — proves the substrate axiom: deposit + bucket, no send.

# ---- Changelog ----
# [2026-06-07] Claude Code (Opus 4.7, 1M) — Commons Pool POC smoke test
# What: Proves two "modules" communicate through the shared Commons medium with ZERO
#       module-to-module contact — module A deposits, module B buckets, B sees A's
#       topology. No tract, no send, no address, no peer reference anywhere.
# Why: Validates the substrate axiom (deposit/bucket, one shared medium) before the
#       Tier-3 wiring step. Substrate-as-protocol Phase 7.
# How: Both "modules" call get_commons() (same singleton = same medium). A deposits a
#       content-derived embedding; B buckets a near-identical query and gets A's target_id
#       back. The only thing connecting them is the shared medium.
# -------------------
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commons


def _emb(seed: int, dim: int = 768) -> np.ndarray:
    """Deterministic unit embedding (no Math.random / time dependence)."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def test_singleton_is_one_medium():
    """get_commons() returns the SAME instance — one shared medium, not per-caller pools."""
    a = commons.get_commons()
    b = commons.get_commons()
    assert a is b, "Commons must be a single shared instance (dual-instance = split medium)"


def test_deposit_then_bucket_same_pattern():
    """Module A deposits topology; Module B buckets the same pattern and sees A's target.

    No module-to-module contact: A and B only touch the shared medium.
    """
    medium = commons.get_commons()

    # "Module A" deposits a content-derived experience into the shared medium.
    pattern = _emb(42)
    target = "concept:shared_pattern_42"
    medium.deposit(pattern, target, success=True)

    # "Module B" — independently — buckets a near-identical query from the SAME medium.
    # B never references A, never receives anything addressed to it. It dips its bucket.
    query = _emb(42)  # same pattern → should associate to A's deposit
    results = medium.bucket(query, top_k=5)

    targets = [t for (t, _conf, _why) in results]
    assert target in targets, (
        f"B's bucket should surface A's deposited target via the shared medium; "
        f"got {targets!r}"
    )


def test_bucket_unrelated_pattern_is_empty_or_excludes():
    """A bucket for an UNRELATED pattern does not surface A's deposit (medium discriminates)."""
    medium = commons.get_commons()

    pattern = _emb(7)
    target = "concept:pattern_7"
    medium.deposit(pattern, target, success=True)

    unrelated = _emb(9999)
    results = medium.bucket(unrelated, top_k=5)
    targets = [t for (t, _conf, _why) in results]
    # Either empty (no learned route) or at least not falsely returning the unrelated target.
    assert target not in targets or len(results) == 0, (
        "Unrelated query should not surface an unrelated deposit as top match"
    )


def test_no_send_surface_exists():
    """Axiom guard: the Commons API exposes ONLY deposit + bucket (+ persistence/stats/suppression).

    No send/route/to/broadcast/tract verb may exist. If someone adds one, this fails.
    """
    public = {n for n in dir(commons.Commons) if not n.startswith("_")}
    # deposit + the bucket FAMILY (bucket modes / bucket reads — all extraction, not send) + persistence/stats.
    # bucket_recent = a recency/temporal bucket MODE; arousal/read_arousal = the vagus bucket read.
    # suppress/lift_suppression/is_suppressed/suppression_mode (#366) = extraction-boundary VISIBILITY
    #   control — the reversible counterpart to Cricket's Rim (they revoke/restore/inspect what a bucket
    #   surfaces, honored at extraction time; they do NOT send, route, address, or broadcast anything).
    #   Same non-verb category as persist/stats — they shape/inspect the medium's extraction, no transport.
    # None is a send/route/to/broadcast verb — the axiom this guard actually protects. A param added to
    # a bucket (e.g. bucket_recent's with_embedding) adds NO public name, so it stays invisible here.
    allowed = {"deposit", "bucket", "bucket_recent", "arousal", "read_arousal", "persist", "restore",
               "stats", "suppress", "lift_suppression", "is_suppressed", "suppression_mode"}
    extra = public - allowed
    assert not extra, (
        f"Commons must expose only the deposit + bucket-family + persist/restore/stats/suppression "
        f"surface; a NEW name here may be a forbidden send/route verb — found extra: {extra}"
    )
    # And enforce the real intent directly: no send/route/address/broadcast/tract verb, ever.
    forbidden_roots = ("send", "route", "broadcast", "tract", "emit", "push", "publish", "dispatch")
    offenders = {n for n in public if any(root in n.lower() for root in forbidden_roots)}
    assert not offenders, f"forbidden send/route-style verb(s) on Commons: {offenders}"


def test_bucket_recent_with_embedding_roundtrip():
    """leg-2: with_embedding=True surfaces the deposit's ORIGINAL vector + is back-compatible.

    The enhancer needs the deposit embedding to re-perceive it through the SNN and to key its
    returned salt. with_embedding must round-trip the vector and force-include metadata (5-tuple),
    while the default (3-tuple) and with_metadata (4-tuple) shapes stay exactly as before.
    """
    medium = commons.Commons()  # fresh medium (not the singleton) — isolate this test
    pat = _emb(7)
    medium.deposit(pat, "repair:roundtrip_7", success=True, metadata={"detail": "x"})

    # default shape unchanged (3-tuple)
    plain = medium.bucket_recent(limit=5)
    assert plain and len(plain[0]) == 3, f"default must stay a 3-tuple; got {plain[0]!r}"

    # with_metadata unchanged (4-tuple)
    meta = medium.bucket_recent(limit=5, with_metadata=True)
    assert meta and len(meta[0]) == 4, f"with_metadata must stay a 4-tuple; got {meta[0]!r}"

    # with_embedding → deterministic 5-tuple (tid, weight, reasoning, metadata, embedding)
    rows = medium.bucket_recent(limit=5, with_embedding=True)
    assert rows and len(rows[0]) == 5, f"with_embedding must be a 5-tuple; got {rows[0]!r}"
    tid, _w, _why, md, emb = rows[0]
    assert tid == "repair:roundtrip_7"
    assert isinstance(md, dict) and md.get("detail") == "x", "with_embedding force-includes metadata"
    assert emb is not None and float(np.dot(emb / (np.linalg.norm(emb) + 1e-8), pat)) > 0.99, (
        "the surfaced embedding must be the vector the deposit was made with"
    )


if __name__ == "__main__":
    test_singleton_is_one_medium()
    print("PASS: one shared medium (singleton)")
    test_bucket_recent_with_embedding_roundtrip()
    print("PASS: bucket_recent with_embedding round-trips the vector (5-tuple) + back-compat shapes")
    test_deposit_then_bucket_same_pattern()
    print("PASS: deposit -> bucket propagates through shared medium (no send)")
    test_bucket_unrelated_pattern_is_empty_or_excludes()
    print("PASS: medium discriminates unrelated patterns")
    test_no_send_surface_exists()
    print("PASS: API exposes only deposit/bucket (+persist/restore/stats) — no send verb")
    print("\nCommons POC smoke test: ALL PASS")
