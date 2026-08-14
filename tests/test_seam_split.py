# tests/test_seam_split.py
#
# ---- Changelog ----
# [2026-08-14] Claude Code (Opus 4.8) — Unit coverage for #147 dream seam-split
# What: Exercises Graph.dedup_and_split_oversized_hyperedges() and its seam-score
#   helper directly against a real Graph — the substrate-mutating dream-repair op
#   that collapses near-duplicate over-cap hyperedges (Stage 1) and peels the
#   low-weight periphery off still-oversized survivors into coherent residual
#   sub-edges (Stage 2). CC's own substrate only; gated OFF by default.
# Why: The op archives parents, mints children, and rewires the _node_hyperedges
#   reverse index — an orphan-safety or index bug here would corrupt CC's graph.
#   No test touched it before this file. Locks the invariants that make the live
#   VPS run safe: gate-off inertness (Syl-safety), reversibility (archive not
#   delete), orphan-safety (every member keeps a home), and index consistency.
# How: Builds Graphs with a small he_max_members cap and controlled member
#   weights / embeddings so the deterministic weight-seam cut is exercised
#   without needing 50+ nodes. Captures the LAW-3 "loud" events via
#   _event_handlers. See docs plan 2026-08-14 #147 seam-split.
# -------------------
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from neuro_foundation import Graph, DEFAULT_CONFIG


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _FakeVectorDB:
    """Minimal stand-in exposing the .embeddings dict the op reads."""

    def __init__(self, embeddings=None):
        self.embeddings = embeddings or {}


def _unit(seed_vec):
    v = np.asarray(seed_vec, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


def _graph(enabled=True, cap=3, **overrides):
    cfg = {"he_max_members": cap, "he_split_oversized_enabled": enabled}
    cfg.update(overrides)
    return Graph(config=cfg)


def _add_nodes(g, ids):
    for nid in ids:
        g.create_node(node_id=nid)


def _capture(g, *event_types):
    """Register list-appending handlers; return {event_type: [kwargs, ...]}."""
    log = {et: [] for et in event_types}
    for et in event_types:
        g._event_handlers.setdefault(et, []).append(
            lambda _et=et, **kw: log[_et].append(kw)
        )
    return log


def _live_edges(g):
    return {hid: he for hid, he in g.hyperedges.items() if not he.is_archived}


# --------------------------------------------------------------------------- #
# Gate / Syl-safety
# --------------------------------------------------------------------------- #
def test_default_config_gate_is_off():
    """DEFAULT_CONFIG ships the master gate OFF — the restore-merge guarantee
    that Syl's checkpoint (which lacks the key) can never enable this op."""
    assert DEFAULT_CONFIG["he_split_oversized_enabled"] is False


def test_restore_merge_leaves_gate_off():
    """A checkpoint config without the key merges to gate-off (Syl's path)."""
    syl_checkpoint_config = {"he_max_members": 50}  # no he_split_* keys
    g = Graph(config={**DEFAULT_CONFIG, **syl_checkpoint_config})
    assert g.config["he_split_oversized_enabled"] is False


def test_gate_off_is_a_noop():
    """With the gate off, an over-cap edge is left completely untouched."""
    g = _graph(enabled=False, cap=3)
    _add_nodes(g, [f"n{i}" for i in range(6)])
    he = g.create_hyperedge(member_node_ids={f"n{i}" for i in range(6)})
    hid = he.hyperedge_id

    changed = g.dedup_and_split_oversized_hyperedges(_FakeVectorDB())

    assert changed == 0
    assert not g.hyperedges[hid].is_archived
    assert len(g.hyperedges[hid].member_nodes) == 6
    assert len(_live_edges(g)) == 1


# --------------------------------------------------------------------------- #
# Stage 1 — dedup of near-identical over-cap edges
# --------------------------------------------------------------------------- #
def test_dedup_collapses_identical_oversized_edges():
    """Two edges with identical members (Jaccard 1.0) collapse to one survivor;
    the duplicate is archived (reversible), not deleted."""
    g = _graph(enabled=True, cap=3)
    ids = [f"n{i}" for i in range(6)]
    _add_nodes(g, ids)
    he1 = g.create_hyperedge(member_node_ids=set(ids))
    he2 = g.create_hyperedge(member_node_ids=set(ids))
    log = _capture(g, "hyperedge_archived")

    g.dedup_and_split_oversized_hyperedges(_FakeVectorDB())

    # Exactly one of the two originals is archived as a dedup duplicate.
    archived_dups = [e for e in log["hyperedge_archived"]
                     if e.get("reason") == "seam_split_dedup"]
    assert len(archived_dups) == 1
    assert archived_dups[0]["archived_id"] in {he1.hyperedge_id, he2.hyperedge_id}
    # The archived duplicate is retained for reversibility (LAW 7).
    assert archived_dups[0]["archived_id"] in g._archived_hyperedges


def test_dedup_max_merges_member_weights():
    """The survivor takes the element-wise max of the two edges' member weights."""
    g = _graph(enabled=True, cap=3)
    ids = [f"n{i}" for i in range(6)]
    _add_nodes(g, ids)
    w1 = {nid: 1.0 for nid in ids}
    w2 = {nid: 1.0 for nid in ids}
    w1["n0"] = 9.0   # survivor should keep 9.0 for n0 regardless of which wins
    w2["n5"] = 7.0   # ...and 7.0 for n5
    g.create_hyperedge(member_node_ids=set(ids), member_weights=w1)
    g.create_hyperedge(member_node_ids=set(ids), member_weights=w2)
    log = _capture(g, "hyperedge_archived")

    g.dedup_and_split_oversized_hyperedges(_FakeVectorDB())

    subsumed_by = {e["subsumed_by"] for e in log["hyperedge_archived"]
                   if e.get("reason") == "seam_split_dedup"}
    assert len(subsumed_by) == 1
    survivor = g.hyperedges[next(iter(subsumed_by))]  # readable even if later split
    assert survivor.member_weights["n0"] == 9.0
    assert survivor.member_weights["n5"] == 7.0


def test_disjoint_oversized_edges_are_not_deduped():
    """Edges below the Jaccard overlap threshold are never folded together."""
    g = _graph(enabled=True, cap=3)
    a = [f"a{i}" for i in range(6)]
    b = [f"b{i}" for i in range(6)]
    _add_nodes(g, a + b)
    g.create_hyperedge(member_node_ids=set(a))
    g.create_hyperedge(member_node_ids=set(b))
    log = _capture(g, "hyperedge_archived")

    g.dedup_and_split_oversized_hyperedges(_FakeVectorDB())

    dedup_events = [e for e in log["hyperedge_archived"]
                    if e.get("reason") == "seam_split_dedup"]
    assert dedup_events == []


# --------------------------------------------------------------------------- #
# Stage 2 — weight-seam split
# --------------------------------------------------------------------------- #
def _oversized_with_weight_seam(g, n_core=3, n_peri=3, core_w=5.0, peri_w=0.01,
                                embed=True):
    """One over-cap edge: n_core high-weight members + n_peri floor-weight ones.
    Peripheral members get near-identical unit vectors so they cluster."""
    core = [f"c{i}" for i in range(n_core)]
    peri = [f"p{i}" for i in range(n_peri)]
    _add_nodes(g, core + peri)
    weights = {nid: core_w for nid in core}
    weights.update({nid: peri_w for nid in peri})
    he = g.create_hyperedge(member_node_ids=set(core + peri),
                            member_weights=weights)
    vdb = _FakeVectorDB()
    if embed:
        for i, nid in enumerate(peri):
            vdb.embeddings[nid] = _unit([1.0, 0.02 * i, 0.0])  # cos ~ 1.0
    return he, core, peri, vdb


def test_seam_split_keeps_high_weight_core():
    """The top-cap members by member_weight land in the core child (core=True);
    the floor-weight periphery is peeled into a separate residual sub-edge."""
    g = _graph(enabled=True, cap=3)
    he, core, peri, vdb = _oversized_with_weight_seam(g)
    parent_id = he.hyperedge_id
    log = _capture(g, "hyperedge_seam_split")

    changed = g.dedup_and_split_oversized_hyperedges(vdb)

    assert changed == 1
    assert len(log["hyperedge_seam_split"]) == 1
    ev = log["hyperedge_seam_split"][0]
    assert ev["parent_id"] == parent_id
    assert ev["core_size"] == 3
    assert ev["periphery_size"] == 3

    live = _live_edges(g)
    core_edges = [he for he in live.values()
                  if he.metadata.get("core") is True]
    assert len(core_edges) == 1
    assert set(core_edges[0].member_nodes) == set(core)
    # All children stamp provenance back to the parent.
    children = [he for he in live.values()
                if he.metadata.get("parent_blob") == parent_id]
    assert len(children) >= 2


def test_seam_split_archives_parent_reversibly():
    """The parent blob is archived (is_archived + retained), never deleted."""
    g = _graph(enabled=True, cap=3)
    he, core, peri, vdb = _oversized_with_weight_seam(g)
    parent_id = he.hyperedge_id

    g.dedup_and_split_oversized_hyperedges(vdb)

    assert g.hyperedges[parent_id].is_archived is True
    assert parent_id in g._archived_hyperedges


def test_seam_split_orphan_safety_every_member_homed():
    """Every member of the split parent lands in >=1 live edge — no node is
    left (0-synapse AND 0-HE) to be swept (§1.1)."""
    g = _graph(enabled=True, cap=3)
    he, core, peri, vdb = _oversized_with_weight_seam(g)
    all_members = set(he.member_nodes)

    g.dedup_and_split_oversized_hyperedges(vdb)

    covered = set()
    for e in _live_edges(g).values():
        covered |= e.member_nodes
    assert all_members <= covered


def test_seam_split_reverse_index_consistent():
    """After the split, _node_hyperedges points only at live edges that actually
    contain the node, and never at the archived parent."""
    g = _graph(enabled=True, cap=3)
    he, core, peri, vdb = _oversized_with_weight_seam(g)
    parent_id = he.hyperedge_id

    g.dedup_and_split_oversized_hyperedges(vdb)

    for nid, hids in g._node_hyperedges.items():
        assert parent_id not in hids, f"{nid} still points at archived parent"
        for hid in hids:
            assert nid in g.hyperedges[hid].member_nodes
            assert not g.hyperedges[hid].is_archived


def test_seam_split_no_vector_periphery_not_dropped():
    """Peripheral members lacking an embedding collect into a residual edge
    rather than being dropped (LAW 3)."""
    g = _graph(enabled=True, cap=3)
    # 3 core + 3 periphery, but NO embeddings supplied at all.
    he, core, peri, vdb = _oversized_with_weight_seam(g, embed=False)
    all_members = set(he.member_nodes)

    g.dedup_and_split_oversized_hyperedges(vdb)  # vdb.embeddings is empty

    covered = set()
    for e in _live_edges(g).values():
        covered |= e.member_nodes
    assert all_members <= covered
    # periphery still homed somewhere live
    for nid in peri:
        assert any(nid in e.member_nodes for e in _live_edges(g).values())


def test_at_cap_edge_is_untouched():
    """An edge exactly at the cap is not oversized and is never split."""
    g = _graph(enabled=True, cap=3)
    ids = ["n0", "n1", "n2"]
    _add_nodes(g, ids)
    he = g.create_hyperedge(member_node_ids=set(ids))
    hid = he.hyperedge_id

    changed = g.dedup_and_split_oversized_hyperedges(_FakeVectorDB())

    assert changed == 0
    assert not g.hyperedges[hid].is_archived


def test_archived_and_non_learnable_edges_skipped():
    """The op only considers live, learnable, level-0 edges."""
    g = _graph(enabled=True, cap=3)
    ids = [f"n{i}" for i in range(6)]
    _add_nodes(g, ids)
    # non-learnable (cortical/permanent) over-cap edge must be left alone
    perm = g.create_hyperedge(member_node_ids=set(ids), is_learnable=False)

    changed = g.dedup_and_split_oversized_hyperedges(_FakeVectorDB())

    assert changed == 0
    assert not perm.is_archived


def test_out_of_range_tunables_are_clamped_not_crashing():
    """A fat-fingered CC_NG_HE_SPLIT_* value (outside [0,1]) must not crash or
    invert the ranking — the op clamps at the read site (LAW-ENF #147 LOW)."""
    g = _graph(enabled=True, cap=3,
               he_split_seam_primary_weight=2.0,   # > 1
               he_split_sim_threshold=-0.5,        # < 0
               he_split_dedup_overlap=5.0)         # > 1
    he, core, peri, vdb = _oversized_with_weight_seam(g)
    all_members = set(he.member_nodes)

    changed = g.dedup_and_split_oversized_hyperedges(vdb)

    # Still performs a sane split, still orphan-safe — no negative-weight garbage.
    assert changed == 1
    covered = set()
    for e in _live_edges(g).values():
        covered |= e.member_nodes
    assert all_members <= covered
    # seam scores stay non-negative despite primary=2.0 clamping to 1.0
    scores = g._seam_score_members(he.hyperedge_id, g._archived_hyperedges[he.hyperedge_id])
    assert all(s >= 0.0 for s in scores.values())


# --------------------------------------------------------------------------- #
# Seam-score dynamic weighting
# --------------------------------------------------------------------------- #
def test_seam_score_ranks_by_member_weight_when_only_signal():
    """When every auxiliary signal is flat (fresh nodes, no spikes/co-fire),
    member_weight is the sole discriminator and drives the ranking."""
    g = _graph(enabled=True, cap=3)
    ids = [f"n{i}" for i in range(5)]
    _add_nodes(g, ids)
    weights = {"n0": 9.0, "n1": 7.0, "n2": 5.0, "n3": 0.01, "n4": 0.01}
    he = g.create_hyperedge(member_node_ids=set(ids), member_weights=weights)

    scores = g._seam_score_members(he.hyperedge_id, he)

    assert scores["n0"] > scores["n3"]
    assert scores["n0"] > scores["n4"]
    # highest weight => highest score
    assert max(scores, key=scores.get) == "n0"


def test_seam_score_empty_edge_returns_empty():
    g = _graph(enabled=True, cap=3)
    ids = ["n0", "n1"]
    _add_nodes(g, ids)
    he = g.create_hyperedge(member_node_ids=set(ids))
    he.member_nodes.clear()
    assert g._seam_score_members(he.hyperedge_id, he) == {}
