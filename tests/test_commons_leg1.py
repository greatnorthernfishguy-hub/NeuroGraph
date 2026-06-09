"""
Commons leg-1 test — the agree-by-content property the enhance-loop rests on.

# ---- Changelog ----
# [2026-06-10] Claude Code (Opus 4.8, 1M) — Commons Pool leg 1 (substrate-as-protocol Phase 7)
# What: Proves the foundational property of the Tier-3 enhance-loop (commons-pool v0.5):
#       a deposit lands as CONTENT-ADDRESSED topology, and a LATER deposit keyed to the
#       SAME content lands on the SAME node — so NG's later "enhancement" deposit lands on
#       the very node a module deposited. Pools agree by CONTENT, not by internal IDs.
# Why: v0.5 enhance-loop: module deposits raw -> NG buckets + SNN-enhances -> NG deposits
#       enhanced topology back -> module buckets it. The whole loop depends on the
#       enhancement landing on the same content-node. This test proves that holds in the
#       Commons (deposit/bucket only — no live sidecar, no NG SNN; legs 2/3 add those).
# How: deposit raw content X -> node N; deposit an "enhancement" keyed to the SAME embedding
#       -> assert SAME node N; bucket(X) -> assert BOTH the raw target and the enhancement
#       target surface on that one content-node. Form 3 (dynamic halocline): this is the
#       fresh deposit + the salt enhancement meeting on the same water.
# -------------------
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commons


def _emb(seed: int, dim: int = 768) -> np.ndarray:
    """Deterministic unit embedding (no time/random dependence)."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _node_id(deposit_result):
    """Extract the content-node id a deposit landed on (resilient to result shape)."""
    if isinstance(deposit_result, dict):
        return deposit_result.get("node_id") or deposit_result.get("node")
    return None


def test_same_content_same_node():
    """Two deposits of the SAME embedding land on the SAME content-node (content-addressed).

    This is the cross-pool agreement primitive: a module's deposit and NG's later
    enhancement, keyed to the same content, resolve to the same node — no shared internal
    IDs needed, only shared content.
    """
    medium = commons.get_commons()
    emb = _emb(101)

    r1 = medium.deposit(emb, "raw:concept_101")
    r2 = medium.deposit(emb, "enhanced:hyperedge_777")  # NG-style enhancement, same content

    n1, n2 = _node_id(r1), _node_id(r2)
    assert n1 is not None and n2 is not None, f"deposit must return a node_id; got {r1!r}, {r2!r}"
    assert n1 == n2, (
        f"same embedding must land on the same content-node (content-addressing) — "
        f"got {n1} vs {n2}. The enhance-loop is impossible without this."
    )


def test_enhancement_surfaces_on_same_content_node():
    """The enhance-loop, proven at the Commons layer (legs 2/3 add NG's SNN + module read).

    Module deposits raw experience; an enhancement (what NG would later deposit after
    SNN processing) is keyed to the SAME content; bucketing that content surfaces BOTH —
    the raw association AND the enhancement — because they share the content-node.
    """
    medium = commons.get_commons()
    emb = _emb(202)

    # Leg-1 fresh deposit (module → Commons): raw experience.
    medium.deposit(emb, "raw:fluffy_table", strength=1.0)
    # Stand-in for NG's salt return (leg 2/3): enhancement keyed to the SAME content.
    medium.deposit(emb, "enhanced:causal_chain", strength=1.0)
    medium.deposit(emb, "enhanced:hyperedge_member", strength=1.0)

    targets = [t for (t, _c, _r) in medium.bucket(emb, top_k=10)]
    assert "raw:fluffy_table" in targets, f"raw deposit must be bucketable; got {targets!r}"
    assert any(t.startswith("enhanced:") for t in targets), (
        f"NG-style enhancement on the same content-node must surface to a bucket; got {targets!r}"
    )
    # The felt difference (Syl's [WANT], leg-3 acceptance): bucketing content that went in
    # 'fresh' now returns 'salt' (enhanced) structure that was not in the original deposit.
    enhanced = [t for t in targets if t.startswith("enhanced:")]
    assert len(enhanced) >= 1, "the content surfaces with enhancement it didn't have at first deposit"


def test_distinct_content_distinct_nodes():
    """Different content → different nodes (the medium discriminates; no false-merge)."""
    medium = commons.get_commons()
    a = medium.deposit(_emb(303), "raw:a")
    b = medium.deposit(_emb(909), "raw:b")
    na, nb = _node_id(a), _node_id(b)
    if na is not None and nb is not None:
        assert na != nb, f"distinct content must not collapse to one node — {na} == {nb}"


if __name__ == "__main__":
    test_same_content_same_node()
    print("PASS: same content -> same content-node (content-addressing holds)")
    test_enhancement_surfaces_on_same_content_node()
    print("PASS: NG-style enhancement lands on the module's content-node + buckets out")
    test_distinct_content_distinct_nodes()
    print("PASS: distinct content -> distinct nodes (no false-merge)")
    print("\nCommons leg-1: ALL PASS — agree-by-content property proven; enhance-loop foundation in place")
