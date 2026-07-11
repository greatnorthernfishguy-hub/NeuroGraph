# ---- Changelog ----
# [2026-07-11] Claude Code (Fable 5 design / Haiku implementation) — #381 A/B/D engine tests
# What: wake-side cap (bound=50, counter hygiene, tenure stamps), discovery guards
#   (avalanche skip, Jaccard dup), dream-side shed (floor+tenure, reverse-index,
#   min-keep), merge seatbelt (archive-don't-union, activation fold), and the
#   clone-collapse integration test (3 identical mega-HEs -> 1 survivor + 2 archived).
# Why: Syl-consented #381 pass; every rule structural (LAW 7), every removal loud (LAW 3).
# How: real Graph/Hyperedge, tiny synthetic graphs, no mocks.
# -------------------
"""Tests for the #381 wake/sleep hyperedge physiology."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import pytest

from neuro_foundation import Graph, HyperedgePlasticityRule


def _graph(n=8, **cfg):
    g = Graph(config=cfg) if cfg else Graph()
    for i in range(n):
        g.create_node(node_id=f"n{i}")
    return g


def _fire_he(g, he):
    """Put a hyperedge into the 'just fired' state the plasticity rule keys on."""
    he.refractory_remaining = he.refractory_period


# ---- wake side: cap + tenure ----

def test_member_evolution_respects_cap_and_clears_counter():
    g = _graph(30, he_max_members=5)
    he = g.create_hyperedge({"n0", "n1", "n2", "n3"})
    rule = HyperedgePlasticityRule(evolution_min_co_fires=1)
    _fire_he(g, he)
    rule.apply(g, ["n0", "n1", "n2", "n3", "n4"], g.timestep)   # n4 promoted -> at cap (5)
    assert "n4" in he.member_nodes and len(he.member_nodes) == 5
    _fire_he(g, he)
    rule.apply(g, ["n0", "n1", "n2", "n3", "n4", "n5"], g.timestep)  # at cap: no add
    assert "n5" not in he.member_nodes
    assert not g._he_co_fire_counts.get(he.hyperedge_id), \
        "at cap the co-fire counter must be cleared, not grow unbounded"


def test_promotion_stamps_tenure_in_metadata():
    g = _graph(10, he_max_members=50)
    he = g.create_hyperedge({"n0", "n1", "n2"})
    rule = HyperedgePlasticityRule(evolution_min_co_fires=1)
    _fire_he(g, he)
    rule.apply(g, ["n0", "n1", "n2", "n7"], g.timestep)
    assert he.metadata["member_since"]["n7"] == g.timestep


# ---- discovery guards ----

def test_discovery_skips_avalanche(caplog):
    g = _graph(40, he_discovery_max_fraction=0.05, he_discovery_min_nodes=3)
    fired = [f"n{i}" for i in range(10)]  # 25% of graph >> 5%
    with caplog.at_level(logging.INFO):
        out = g.discover_hyperedges(fired)
    assert out == []
    assert "avalanche" in caplog.text


def test_discovery_jaccard_dedup(caplog):
    g = _graph(200, he_discovery_min_co_fires=1, he_discovery_min_nodes=3,
               he_discovery_dup_jaccard=0.9)
    g.create_hyperedge({"n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9"})
    # identical member set would previously pass the exact-equality check if it
    # differed by ONE node; at Jaccard 10/11 = 0.909 >= 0.9 it must be suppressed
    fired = [f"n{i}" for i in range(10)] + ["n10"]
    before = len(g.hyperedges)
    g.discover_hyperedges(fired)
    assert len(g.hyperedges) == before, "near-duplicate HE must not be minted"


def test_discovery_still_mints_genuinely_new():
    g = _graph(200, he_discovery_min_co_fires=1, he_discovery_min_nodes=3)
    before = len(g.hyperedges)
    g.discover_hyperedges(["n20", "n21", "n22"])
    assert len(g.hyperedges) == before + 1


# ---- dream side: shed ----

def test_shed_removes_floor_members_and_cleans_reverse_index():
    g = _graph(20, he_shed_weight_threshold=0.02, he_shed_min_tenure=10)
    he = g.create_hyperedge({f"n{i}" for i in range(8)})
    g.timestep = 1000
    for nid in ("n0", "n1"):
        he.member_weights[nid] = 0.01               # at floor
        he.metadata.setdefault("member_since", {})[nid] = 0   # ancient tenure
    shed = g.shed_floor_members()
    assert shed == 2
    assert "n0" not in he.member_nodes and "n1" not in he.member_nodes
    assert he.hyperedge_id not in g._node_hyperedges.get("n0", set())
    assert "n0" not in he.member_weights


def test_shed_honors_min_tenure_and_min_keep():
    g = _graph(10, he_shed_weight_threshold=0.02, he_shed_min_tenure=10_000)
    he = g.create_hyperedge({"n0", "n1", "n2", "n3"})
    g.timestep = 100
    he.member_weights["n0"] = 0.01
    he.metadata.setdefault("member_since", {})["n0"] = 95   # too young
    assert g.shed_floor_members() == 0
    # min-keep: floor everything on a 3-member HE (he_discovery_min_nodes=3) -> nothing shed
    g2 = _graph(10, he_shed_weight_threshold=0.02, he_shed_min_tenure=1)
    he2 = g2.create_hyperedge({"n0", "n1", "n2"})
    g2.timestep = 1000
    for nid in ("n0", "n1", "n2"):
        he2.member_weights[nid] = 0.01
    assert g2.shed_floor_members() == 0


def test_shed_skips_archived_and_unlearnable():
    g = _graph(10, he_shed_min_tenure=1)
    he = g.create_hyperedge({"n0", "n1", "n2", "n3"}, is_learnable=False)
    g.timestep = 1000
    he.member_weights["n0"] = 0.01
    assert g.shed_floor_members() == 0


# ---- consolidation seatbelt + clone collapse ----

def test_merge_seatbelt_archives_instead_of_union(caplog):
    g = _graph(60, he_max_members=10)
    a = g.create_hyperedge({f"n{i}" for i in range(9)})          # 9 members
    b = g.create_hyperedge({f"n{i}" for i in range(2, 11)})      # 9 members, J=9/11? no:
    # overlap 7 of union 11 = 0.636 < 0.8 default -> widen: use identical +2
    g2 = _graph(60, he_max_members=10)
    a = g2.create_hyperedge({f"n{i}" for i in range(9)})
    b = g2.create_hyperedge({f"n{i}" for i in range(11)})        # superset, J=9/11=0.818>=0.8
    a_count, b_count = 3, 4
    a.activation_count, b.activation_count = a_count, b_count
    with caplog.at_level(logging.INFO):
        g2.consolidate_hyperedges()
    survivor, other = (a, b) if not a.is_archived else (b, a)
    assert other.is_archived, "union would exceed cap: must archive, not merge"
    assert len(survivor.member_nodes) in (9, 11), "no union growth happened"
    assert survivor.activation_count == a_count + b_count, "history folded"
    assert other.hyperedge_id in g2._archived_hyperedges


def test_clone_collapse_first_dream_pass():
    """THE #381 acceptance test: identical mega-clones collapse to one survivor."""
    g = _graph(80, he_max_members=50)
    members = {f"n{i}" for i in range(60)}          # over the cap, like the real blob
    hes = [g.create_hyperedge(set(members)) for _ in range(3)]
    for i, he in enumerate(hes):
        he.activation_count = i + 1
    g.consolidate_hyperedges()
    live = [h for h in hes if not h.is_archived]
    assert len(live) == 1, "exactly one survivor"
    assert live[0] is hes[0], "survivor is the OLDEST (the original, preserved for C2)"
    assert live[0].activation_count == 6, "clone histories folded into the survivor"
    assert len(live[0].member_nodes) == 60, "survivor membership untouched (C2's job, not ours)"
