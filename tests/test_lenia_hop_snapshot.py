# ---- Changelog ----
# [2026-11-11] DudeMan CC (Claude Fable 5.2, laptop; plan docs/superpowers/plans/2026-11-11-lenia-hop-distances-rust.md;
#   Josh 2026-11-11 "OK, begin steps 1-5 ... you got my 'go'") -- CREATE
# What: the Lenia distance cache's hop distances are native (ng_tract.HopSnapshot, one search per run of pairs): the
#   values, the substrate's cache and populate()'s whole result equal the pre-port Python's exactly.
# Why:  #1218 / #1084 / #1104: the per-pair Python BFS ran ~30 h after a 6,833-node growth on the live CC daemon.
#   Parity against Python is legitimate here: it is existing Python being replaced (the S18 / DiffPc plans' ruling).
# How:  the oracle is lenia/graph_substrate.py + lenia/kernel.py at the pre-port commit (PRE_REV), loaded from git;
#   real Graph / NeuroGraphSubstrate / DistanceCache (this suite's no-mocks convention); random graphs with several
#   components, isolated nodes, duplicate edges and hyperedges; full, incremental and resumed populate().
# -------------------
"""Native hop distances for the Lenia distance cache: exact parity with the pre-port Python."""

import itertools
import os
import random
import subprocess
import sys
import types

import numpy as np
import pytest

from lenia import graph_substrate as GS
from lenia import kernel as KN
from neuro_foundation import Graph

PRE_REV = "182caba"   # the line before the port
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_pre(path, name):
    src = subprocess.run(["git", "-C", _REPO, "show", f"{PRE_REV}:{path}"], check=True,
                         capture_output=True, text=True).stdout
    mod = types.ModuleType(name)
    mod.__file__ = f"<{PRE_REV}:{path}>"
    sys.modules[name] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


PRE_GS = _load_pre("lenia/graph_substrate.py", "lenia_gs_preport")
PRE_KN = _load_pre("lenia/kernel.py", "lenia_kn_preport")


def _graph(seed, n=60, edges=90, hyperedges=6, isolated=5):
    """Several components (edges only inside thirds), isolated nodes, repeated and reversed edges, hyperedges."""
    rng = random.Random(seed)
    g = Graph()
    ids = [f"n{i:03d}" for i in range(n + isolated)]
    for nid in ids:
        g.create_node(node_id=nid)
    third = n // 3
    made = set()
    for _ in range(edges):
        blk = rng.randrange(3)
        a, b = rng.sample(ids[blk * third:(blk + 1) * third], 2)
        if (a, b) in made:
            continue
        made.add((a, b))
        g.create_synapse(a, b, weight=rng.uniform(0.1, 0.9))
        if rng.random() < 0.15 and (b, a) not in made:   # both directions between the same two nodes
            made.add((b, a))
            g.create_synapse(b, a, weight=rng.uniform(0.1, 0.9))
    for _ in range(hyperedges):
        g.create_hyperedge(set(rng.sample(ids[:n], rng.randrange(3, 7))))
    return g, ids


def _grow(g, ids, seed, new=12, edges=30):
    rng = random.Random(seed)
    fresh = [f"z{i:03d}" for i in range(new)]   # sort after every n*: appended, as on a live restart
    for nid in fresh:
        g.create_node(node_id=nid)
    for _ in range(edges):
        a = rng.choice(fresh)
        b = rng.choice(ids + fresh)
        if a != b:
            try:
                g.create_synapse(a, b, weight=rng.uniform(0.1, 0.9))
            except Exception:
                pass
    g.create_hyperedge(set(rng.sample(fresh, 3) + rng.sample(ids[:20], 2)))
    return fresh


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_every_pair_equals_the_preport_python_bfs(seed):
    g, ids = _graph(seed)
    new, old = GS.NeuroGraphSubstrate(g, None), PRE_GS.NeuroGraphSubstrate(g, None)
    reach = 0
    for a, b in itertools.product(ids, ids):
        want = old._topology_distance(a, b)
        got = new._topology_distance(a, b)
        assert got == want and type(got) is float, (a, b, got, want)
        reach += want not in (0.0, float(len(ids)))
    assert reach > 200, "the graph must have real multi-hop pairs"
    assert new._topology_distance("ghost", ids[0]) == old._topology_distance("ghost", ids[0]) == float(len(ids))
    assert new._topo_cache == old._topo_cache, "the same keys and values are memoised"


def test_prefetch_fills_the_cache_exactly_as_the_pairs_would():
    g, ids = _graph(7)
    new, old = GS.NeuroGraphSubstrate(g, None), PRE_GS.NeuroGraphSubstrate(g, None)
    src = ids[4]
    targets = ids[:30] + [src, "ghost"]
    new.prefetch_topology(src, targets)
    for t in targets:
        old._topology_distance(src, t)
    assert new._topo_cache == old._topo_cache
    calls = []
    real = new._hops
    new._hops = types.SimpleNamespace(hops_from=lambda *a: calls.append(a) or real.hops_from(*a))
    new.prefetch_topology(src, targets)
    for t in targets:
        new._topology_distance(src, t)
    assert calls == [], "a prefetched group costs no further search"


def _components(cache):
    return [c.toarray() for c in cache._components_csr]


def _same(a, b):
    assert a._watermark == b._watermark
    for c, (x, y) in enumerate(zip(_components(a), _components(b))):
        assert x.dtype == y.dtype and np.array_equal(x, y), f"component {c} differs"
    assert np.array_equal(a._magnitude_csr.toarray(), b._magnitude_csr.toarray())


@pytest.mark.parametrize("seed", [0, 5])
def test_populate_full_incremental_and_resume_equal_the_preport(seed):
    g, ids = _graph(seed)
    subs = GS.NeuroGraphSubstrate(g, None), PRE_GS.NeuroGraphSubstrate(g, None)
    caches = (KN.DistanceCache(subs[0].entity_count(), entity_ids=subs[0].entities()),
              PRE_KN.DistanceCache(subs[1].entity_count(), entity_ids=subs[1].entities()))
    for c, s in zip(caches, subs):
        c.populate(s)
    _same(*caches)
    assert caches[0]._components_csr[0].nnz > 100

    # growth: new entities appended, incremental populate from the old count
    old_n = len(subs[0].entities())
    order = list(subs[0].entities())
    _grow(g, ids, seed + 100)
    subs = (GS.NeuroGraphSubstrate(g, None, known_entity_order=order),
            PRE_GS.NeuroGraphSubstrate(g, None, known_entity_order=order))
    assert subs[0].entities() == subs[1].entities() and subs[0].entities()[:old_n] == order
    for c, s in zip(caches, subs):
        c.resize(s.entity_count(), new_entity_ids=s.entities())
        c.populate(s, start_index=old_n)
    _same(*caches)

    # resume: a fresh cache carrying an interrupted run's watermark covers the rest
    n = subs[0].entity_count()
    wm = (n // 3, n // 2)
    fresh = (KN.DistanceCache(n, entity_ids=subs[0].entities()), PRE_KN.DistanceCache(n, entity_ids=subs[1].entities()))
    for c, s in zip(fresh, subs):
        c.populate(s, resume_watermark=wm)
    _same(*fresh)


def test_one_search_per_run_of_pairs():
    g, _ids = _graph(3)
    sub = GS.NeuroGraphSubstrate(g, None)
    sub._build_adjacency()
    real, calls = sub._hops, []
    sub._hops = types.SimpleNamespace(hops_from=lambda s, t, u: calls.append((s, len(t))) or real.hops_from(s, t, u))
    cache = KN.DistanceCache(sub.entity_count(), entity_ids=sub.entities())
    cache.populate(sub)
    pairs = cache._components_csr[0].nnz // 2
    assert len(calls) <= sub.entity_count(), "at most one search per max endpoint"
    assert len(calls) < pairs / 3, (len(calls), pairs)
    assert len({s for s, _n in calls}) == len(calls), "never two searches from the same node"


def test_a_substrate_without_prefetch_populates_as_before():
    g, _ids = _graph(2)
    sub = PRE_GS.NeuroGraphSubstrate(g, None)   # the pre-port substrate has no prefetch_topology
    assert not hasattr(sub, "prefetch_topology")
    a = KN.DistanceCache(sub.entity_count(), entity_ids=sub.entities())
    b = PRE_KN.DistanceCache(sub.entity_count(), entity_ids=sub.entities())
    a.populate(sub)
    b.populate(PRE_GS.NeuroGraphSubstrate(g, None))
    _same(a, b)


def test_without_the_native_snapshot_the_adjacency_build_raises(monkeypatch):
    import ng_tract
    monkeypatch.delattr(ng_tract, "HopSnapshot")
    g, ids = _graph(1)
    sub = GS.NeuroGraphSubstrate(g, None)
    with pytest.raises(RuntimeError, match="HopSnapshot"):
        sub._topology_distance(ids[0], ids[1])


def test_topology_change_drops_the_snapshot_and_the_cache():
    g, ids = _graph(4)
    sub = GS.NeuroGraphSubstrate(g, None)
    far = [b for b in ids if sub._topology_distance(ids[0], b) == float(len(ids))][0]
    g.create_synapse(ids[0], far, weight=0.3)
    assert sub._topology_distance(ids[0], far) == float(len(ids)), "the snapshot is frozen until a topology change"
    sub.on_topology_change()
    assert sub._hops is None and sub._topo_cache == {}
    assert sub._topology_distance(ids[0], far) == 1.0
