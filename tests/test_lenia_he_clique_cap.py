# ---- Changelog ----
# [2026-07-10] Claude Code (Fable 5 design / Haiku implementation) — #381/#380 cap tests
# What: mega-HE cliques are skipped above NG_LENIA_HE_CLIQUE_CAP; normal HEs still
#   contribute pairs; cap=0 disables; the skip is logged loudly.
# Why: the guard that unblocks #380 must not silently eat legitimate hyperedge structure.
# How: real Graph/NeuroGraphSubstrate/DistanceCache (no-mocks convention); env via
#   monkeypatch.setenv (read at populate-call time by design). [Controller replaced an
#   implementer-authored MagicMock version — mocks violate the suite convention and
#   cannot catch real Hyperedge/substrate interaction breaks.]
# -------------------
"""Tests for the Lenia hyperedge clique cap (#381/#380)."""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lenia.graph_substrate import NeuroGraphSubstrate
from lenia.kernel import DistanceCache
from neuro_foundation import Graph


def _build_graph():
    """Chain of 25 + one giant HE over the chain + a small HE over three
    OFF-chain isolated nodes (their pairs can only come from the HE)."""
    g = Graph()
    for i in range(25):
        g.create_node(node_id=f"n{i}")
    for i in range(24):
        g.create_synapse(f"n{i}", f"n{i + 1}", weight=0.5)
    for z in ("z0", "z1", "z2"):
        g.create_node(node_id=z)
    g.create_hyperedge({f"n{i}" for i in range(25)})   # giant: 25 members
    g.create_hyperedge({"z0", "z1", "z2"})             # small: 3 members
    return g


def _populated(monkeypatch, cap, caplog=None):
    monkeypatch.setenv("NG_LENIA_HE_CLIQUE_CAP", str(cap))
    g = _build_graph()
    sub = NeuroGraphSubstrate(g, None)
    cache = DistanceCache(sub.entity_count(), entity_ids=sub.entities())
    if caplog is not None:
        with caplog.at_level(logging.INFO):
            cache.populate(sub)
    else:
        cache.populate(sub)
    idx = {e: i for i, e in enumerate(sub.entities())}
    return cache, idx


def _has_pair(cache, idx, a, b):
    return bool(np.any(np.abs(cache.get_distance_vector(idx[a], idx[b])) > 1e-15))


def test_cap_skips_giant_he_but_keeps_small(monkeypatch, caplog):
    cache, idx = _populated(monkeypatch, cap=10, caplog=caplog)
    # small HE (3 members, isolated nodes): pair exists only via the HE
    assert _has_pair(cache, idx, "z0", "z2")
    # giant-HE-only pair (chain-distant, >2 hops apart): must be absent
    assert not _has_pair(cache, idx, "n0", "n20")
    # direct synapse pairs unaffected
    assert _has_pair(cache, idx, "n0", "n1")
    assert "skipped 1 hyperedge clique" in caplog.text
    assert "largest: 25 members" in caplog.text


def test_cap_zero_disables_guard(monkeypatch):
    cache, idx = _populated(monkeypatch, cap=0)
    assert _has_pair(cache, idx, "n0", "n20")  # giant clique fully expanded


def test_default_cap_leaves_normal_hes_alone(monkeypatch):
    # Default 100: both HEs here are under it — everything expands.
    monkeypatch.delenv("NG_LENIA_HE_CLIQUE_CAP", raising=False)
    g = _build_graph()
    sub = NeuroGraphSubstrate(g, None)
    cache = DistanceCache(sub.entity_count(), entity_ids=sub.entities())
    cache.populate(sub)
    idx = {e: i for i, e in enumerate(sub.entities())}
    assert _has_pair(cache, idx, "n0", "n20")
    assert _has_pair(cache, idx, "z0", "z2")
