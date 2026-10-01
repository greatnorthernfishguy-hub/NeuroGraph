#!/usr/bin/env python3
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane ack-bound-918, dispatch #13352) — #918 tests: the ack means "I hold it BOUND"
# What: a REAL neuro_foundation.Graph + real SimpleVectorDB + the REAL exporter (export_cc_topology_frame) feeding the REAL
#   merge (merge_cc_topology), for:
#   (1) cc_current_membership = CC nodes MINUS the #905 sweep-eligible-unbound set (bound IN, unprotected unbound OUT,
#       protected unbound IN, non-CC OUT, byte-identical when nothing is unbound, a PURE query, it CALLS the imported
#       `_unbound_nodes`, no age term);
#   (2) the merge LANDS a re-offered node's hyperedge / synapses onto the already-present node (bound after ONE merge call; the
#       ack written by that call includes it, the previous call's did not);
#   (3) the end-to-end sender -> merge -> ack -> next-frame LOOP with the new membership vs the old (the #909 deadlock);
#   (4) the bounded, loud, redacted, in-memory receiver-side re-offer counter (`_track_reoffers`, env
#       CC_TOPOLOGY_REOFFER_WARN_STREAK) and the stats keys.
# Why: Exec Packet 496 via Chief-003 (#918, RULED ADOPT, S4b-GATING). The #110 ack counted nodes the receiver holds UNBOUND, so
#   the 147 unbound cohort nodes were never re-offered: the #897/#905 clock hold forbids the cull, the ack forbade the re-send.
# How: Z12_MERGE_UNDER_TEST (a path) loads a different cc_topology_merge.py as the module under test, so this SAME file runs
#   unchanged against the BASE file (the #905 tip's copy) and against each scratch MUTANT to show which cases fail. P379/#770:
#   the printed preamble names every module under test. The synthetic-loop pattern is the #909 analysis script's
#   (analysis-909-sender-candidate-predicate.py): fake ids and embeddings only.
# -------------------
import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Set

import numpy as np
import pytest

_WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKTREE))

from neuro_foundation import Graph  # noqa: E402
from universal_ingestor import SimpleVectorDB  # noqa: E402
import cc_topology_export as tex  # noqa: E402


def _load_by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import cc_ng_organism as cno  # noqa: E402  (first: the merge imports it lazily by NAME)

_MERGE_UNDER_TEST = os.environ.get("Z12_MERGE_UNDER_TEST")
if _MERGE_UNDER_TEST:
    tmg = _load_by_path(Path(_MERGE_UNDER_TEST).resolve(), "cc_topology_merge")
else:
    import cc_topology_merge as tmg  # noqa: E402

print("[P379/#770 preamble] worktree root        ->", _WORKTREE)
print("[P379/#770 preamble] merge under test     ->", Path(tmg.__file__).resolve())
for _m in (sys.modules["neuro_foundation"], tex, cno):
    print("[P379/#770 preamble] %-18s ->" % _m.__name__, Path(_m.__file__).resolve())

_ENV = "CC_TOPOLOGY_REOFFER_WARN_STREAK"


def test_modules_under_test_resolve_inside_the_worktree():
    if _MERGE_UNDER_TEST:
        pytest.skip("the merge is deliberately overridden (%s)" % _MERGE_UNDER_TEST)
    for mod in (tmg, cno, tex, sys.modules["neuro_foundation"]):
        assert _WORKTREE in Path(mod.__file__).resolve().parents, f"{mod.__name__} resolves outside {_WORKTREE}"


@pytest.fixture(autouse=True)
def _fresh_counter_state(monkeypatch):
    """The re-offer table is process-lifetime state: every test starts and ends with it empty and the env unset."""
    monkeypatch.delenv(_ENV, raising=False)
    if hasattr(tmg, "_reoffer_streaks"):
        tmg._reoffer_streaks = {}
    yield
    if hasattr(tmg, "_reoffer_streaks"):
        tmg._reoffer_streaks = {}


# ------------------------------------------------------------------ rig

DIM = 768
OLD = 100           # laptop timestep: a node created at 0 is 100 steps old; orphan grace is 25
FRAME = 6           # small sender frame; the overflow factor stays the real default (3) -> hard cap 18
BIG = 200           # receiver budget: generous, so no member is ever deferred
_SEED = [0]


def _emb():
    _SEED[0] += 1
    return np.random.default_rng(_SEED[0]).normal(size=DIM).astype(np.float32)


def _meta(**extra):
    m = {"cc": True, "creation_mode": "conversational"}
    m.update(extra)
    return m


def _vps(g, v, nid, t, **extra):
    """A node on the synthetic VPS (sender) graph, with an embedding in its vector db."""
    meta = _meta(**extra)
    g.create_node(node_id=nid, metadata=dict(meta)).creation_time = t
    v.insert(id=nid, embedding=_emb(), content="content for " + nid, metadata=dict(meta))


def _held(g, v, nid, **extra):
    """A node the laptop (receiver) already holds, age OLD (older than orphan grace); unbound until something wires it."""
    meta = _meta(**extra)
    g.create_node(node_id=nid, metadata=dict(meta)).creation_time = 0
    v.insert(id=nid, embedding=_emb(), content="content for " + nid, metadata=dict(meta))


def _he(g, members):
    g.create_hyperedge(member_node_ids=set(members), metadata={"creation_mode": "conversational", "cc": True})


def _receiver():
    rg, rv = Graph(), SimpleVectorDB()
    rg.timestep = OLD
    return rg, rv


def _all_unbound(g):
    return tmg._unbound_nodes(g, set(g.nodes))


def _export_frame(vg, vv, out, exclude_ids, frame_size=FRAME):
    return tex.export_cc_topology_frame(vg, vv, str(out), machine_id="vps", frame_size=frame_size,
                                        exclude_ids=set(exclude_ids), embedding_model="test-model",
                                        skip_resource_gate=True)


def _merge(rg, rv, conduit, ack, **kw):
    kw.setdefault("local_machine_id", "laptop")
    kw.setdefault("expected_embedding_model", "test-model")
    kw.setdefault("membership_path", str(ack))
    kw.setdefault("idle_steps", 0)
    kw.setdefault("max_nodes_per_call", BIG)
    return tmg.merge_cc_topology(rg, rv, str(conduit), **kw)


def _ack(path) -> Set[str]:
    return set(Path(path).read_text().split())


def _bound_pair_conduit(tmp_path, tag="unrelated"):
    """A conduit from the real exporter carrying one bound pair that has nothing to do with the held nodes."""
    sg, sv = Graph(), SimpleVectorDB()
    f, t = f"cc:conv::{tag}", f"cc:conv::{tag}::tree::x"
    _vps(sg, sv, f, 1)
    _vps(sg, sv, t, 2)
    sg.create_synapse(f, t, weight=0.4, delay=2)
    out = tmg_out = tmp_path / f"{tag}.conduit"
    tex.export_cc_topology(sg, sv, str(out), machine_id="vps", embedding_model="test-model")
    return out


# ============================================================== (1) cc_current_membership

def _membership_graph():
    g = Graph()
    g.timestep = OLD
    for nid, meta in (
        ("cc:conv::b1", {}), ("cc:conv::b1::tree::x", {"creation_mode": "tree"}),          # bound by a synapse
        ("cc:conv::h1", {}), ("cc:conv::h2", {}), ("cc:conv::h3", {}),                       # bound by a hyperedge only
        ("cc:conv::fresh-unbound", {}),                                                      # unprotected unbound, age 0
        ("cc:conv::old-unbound", {}),                                                        # unprotected unbound, age 100
        ("cc:want::emergent", {"provenance": "cc_emergent"}),                                # '*_emergent': prunable, unbound
        ("cc:want::authored", {"kind": "want", "provenance": "cc_authored"}),                # PROTECTED unbound
        ("cc:conv::constitutional", {"constitutional": True}),                               # PROTECTED unbound
        ("syl:bound-a", {}), ("syl:bound-b", {}), ("syl:unbound", {}),                       # not CC
    ):
        g.create_node(node_id=nid, metadata=_meta(**meta)).creation_time = 0
    g.nodes["cc:conv::fresh-unbound"].creation_time = OLD                                    # age 0 at the moment of decision
    g.create_synapse("cc:conv::b1", "cc:conv::b1::tree::x", weight=0.4)
    g.create_synapse("syl:bound-a", "syl:bound-b", weight=0.4)
    _he(g, ["cc:conv::h1", "cc:conv::h2", "cc:conv::h3"])
    return g


_MEMBERS = {"cc:conv::b1", "cc:conv::b1::tree::x", "cc:conv::h1", "cc:conv::h2", "cc:conv::h3",
            "cc:want::authored", "cc:conv::constitutional"}


def test_membership_bound_in_unprotected_unbound_out_protected_unbound_in_non_cc_out():
    g = _membership_graph()
    m = tmg.cc_current_membership(g)
    assert m == _MEMBERS
    # the two unprotected unbound ones are OUT, including the one that is age 0 (NO age term, by design)
    for nid in ("cc:conv::fresh-unbound", "cc:conv::old-unbound", "cc:want::emergent"):
        assert nid not in m
        assert nid in _all_unbound(g)
    # the protected unbound ones ARE structurally unbound -- only the protection leg keeps them in
    for nid in ("cc:want::authored", "cc:conv::constitutional"):
        assert nid in m and nid not in _all_unbound(g) and not g._outgoing.get(nid) and not g._incoming.get(nid)
    assert not any(n.startswith("syl:") for n in m)


def test_membership_is_byte_identical_to_the_base_on_a_graph_with_no_unbound_node(tmp_path):
    g = Graph()
    g.timestep = OLD
    for nid in ("cc:conv::a", "cc:conv::b", "cc:conv::c", "cc:conv::d", "syl:s"):
        g.create_node(node_id=nid, metadata=_meta()).creation_time = 0
    g.create_synapse("cc:conv::a", "cc:conv::b", weight=0.4)
    _he(g, ["cc:conv::c", "cc:conv::d"])
    g.create_node(node_id="cc:want::authored", metadata=_meta(provenance="cc_authored"))     # protected unbound: still IN
    base = {nid for nid, node in g.nodes.items() if tex.is_cc_provenance(nid, node.metadata)}   # the base definition
    assert tmg.cc_current_membership(g) == base
    p_new, p_base = tmg_path = tmp_path / "new.txt", tmp_path / "base.txt"
    tmg._write_membership(str(p_new), tmg.cc_current_membership(g))
    tmg._write_membership(str(p_base), base)
    assert p_new.read_bytes() == p_base.read_bytes()


def test_membership_is_a_pure_query_graph_counters_and_logs_unchanged(caplog):
    g = _membership_graph()
    caplog.set_level(logging.DEBUG)
    before = (set(g.nodes), len(g.synapses), len(g.hyperedges), g.timestep,
              {k: set(v) for k, v in g._outgoing.items()}, {k: set(v) for k, v in g._incoming.items()},
              {k: set(v) for k, v in g._node_hyperedges.items()})
    streaks_before = dict(getattr(tmg, "_reoffer_streaks", {}))
    first = tmg.cc_current_membership(g)
    second = tmg.cc_current_membership(g)
    after = (set(g.nodes), len(g.synapses), len(g.hyperedges), g.timestep,
             {k: set(v) for k, v in g._outgoing.items()}, {k: set(v) for k, v in g._incoming.items()},
             {k: set(v) for k, v in g._node_hyperedges.items()})
    assert first == second and isinstance(first, set)
    assert before == after
    assert dict(getattr(tmg, "_reoffer_streaks", {})) == streaks_before     # no counter touched from the query
    assert [r for r in caplog.records if r.name == tmg.logger.name] == []   # no logging side effect


def test_membership_signature_and_return_type_are_unchanged():
    sig = inspect.signature(tmg.cc_current_membership)
    assert list(sig.parameters) == ["graph"]
    assert sig.return_annotation == Set[str]


def test_membership_USES_the_imported_unbound_predicate_not_a_copy(monkeypatch):
    """LAW 3: ONE definition. A spy standing in for `_unbound_nodes` is OBEYED, and is handed the CC node set."""
    g = _membership_graph()
    calls = []

    def spy(graph, node_ids):
        calls.append((graph, set(node_ids)))
        return {"cc:conv::b1"}                       # a (wrong) 'unbound' set only the real predicate would not give

    monkeypatch.setattr(tmg, "_unbound_nodes", spy)
    m = tmg.cc_current_membership(g)
    assert len(calls) == 1 and calls[0][0] is g
    assert calls[0][1] == {n for n in g.nodes if not n.startswith("syl:")}        # exactly the CC-provenance node set
    assert "cc:conv::b1" not in m                    # the spy's answer decided
    assert "cc:conv::old-unbound" in m               # ... and nothing else did (a copy would have excluded this)


def test_membership_a_double_without_is_identity_protected_is_not_exempted_unbound_nodes_stay_out():
    from types import SimpleNamespace
    import threading
    dbl = SimpleNamespace(_step_lock=threading.RLock(),
                          nodes={"cc:conv::a": SimpleNamespace(metadata={}), "cc:conv::b": SimpleNamespace(metadata={})},
                          _outgoing={"cc:conv::b": {"s"}}, _incoming={}, _node_hyperedges={})
    assert tmg.cc_current_membership(dbl) == {"cc:conv::b"}


def test_the_ack_file_the_merge_writes_is_the_bound_membership(tmp_path):
    rg, rv = _receiver()
    _held(rg, rv, "cc:conv::held-unbound")
    ack = tmp_path / "ack.txt"
    st = _merge(rg, rv, _bound_pair_conduit(tmp_path), ack)
    assert st["absorbed_nodes"] == 2
    assert _ack(ack) == tmg.cc_current_membership(rg) == {"cc:conv::unrelated", "cc:conv::unrelated::tree::x"}
    assert "cc:conv::held-unbound" in rg.nodes and "cc:conv::held-unbound" not in _ack(ack)


# ============================================================== (2) the merge lands edges onto a re-offered, present node

def test_one_merge_binds_a_held_unbound_node_via_a_reoffered_hyperedge_and_the_ack_follows(tmp_path):
    X, C1, C2 = "cc:conv::x-held", "cc:conv::c1", "cc:conv::c2"
    vg, vv = Graph(), SimpleVectorDB()
    for i, n in enumerate((C1, C2)):
        _vps(vg, vv, n, 10 + i)
    _vps(vg, vv, X, 200)
    _he(vg, [X, C1, C2])
    rg, rv = _receiver()
    _held(rg, rv, X)
    ack = tmp_path / "ack.txt"

    prev = _merge(rg, rv, _bound_pair_conduit(tmp_path), ack)          # the PREVIOUS merge call: leaves X unbound
    assert prev["absorbed_nodes"] == 2 and X in _all_unbound(rg)
    assert X not in _ack(ack), "the previous call's ack must NOT include the held-unbound node"

    out = tmp_path / "frame.conduit"
    st = _export_frame(vg, vv, out, _ack(ack))
    assert X in st["frame_node_ids"], "an unacked held-unbound node must be a sender candidate again"
    stats = _merge(rg, rv, out, ack)

    assert stats["skipped_present"] == 1                                # X: already held, usable as an endpoint
    assert stats["absorbed_nodes"] == 2 and stats["absorbed_hyperedges"] == 1
    assert stats["skipped_hyperedges"] == 0
    assert X not in _all_unbound(rg)                                    # BOUND after ONE merge call
    assert any(he.member_nodes == {X, C1, C2} for he in rg.hyperedges.values())
    assert X in _ack(ack)                                               # the ack of THAT call includes it (it is bound)
    assert _ack(ack) == tmg.cc_current_membership(rg)


def test_one_merge_binds_a_held_unbound_node_via_a_reoffered_synapse_only(tmp_path):
    X, C = "cc:conv::x-held", "cc:conv::c-cand"
    vg, vv = Graph(), SimpleVectorDB()
    _vps(vg, vv, C, 10)
    _vps(vg, vv, X, 200)
    vg.create_synapse(C, X, weight=0.3, delay=1)
    rg, rv = _receiver()
    _held(rg, rv, X)
    ack = tmp_path / "ack.txt"
    _merge(rg, rv, _bound_pair_conduit(tmp_path), ack)
    assert X in _all_unbound(rg) and X not in _ack(ack)

    out = tmp_path / "frame.conduit"
    st = _export_frame(vg, vv, out, _ack(ack))
    assert set(st["frame_node_ids"]) == {C, X}
    stats = _merge(rg, rv, out, ack)

    assert stats["skipped_present"] == 1 and stats["absorbed_synapses"] == 1 and stats["absorbed_hyperedges"] == 0
    assert tmg._synapse_exists(rg, C, X)
    assert X not in _all_unbound(rg)                                    # bound by a SYNAPSE only, after ONE merge call
    assert X in _ack(ack)


def test_the_skipped_present_path_does_not_skip_the_nodes_edges_even_when_every_node_is_present(tmp_path):
    """Both ends already held (all `skipped_present`): the frame's synapse and hyperedge still land, so nothing is lost."""
    A, B, Cn = "cc:conv::p-a", "cc:conv::p-b", "cc:conv::p-c"
    vg, vv = Graph(), SimpleVectorDB()
    for i, n in enumerate((A, B, Cn)):
        _vps(vg, vv, n, 10 + i)
    vg.create_synapse(A, B, weight=0.3, delay=1)
    _he(vg, [A, B, Cn])
    rg, rv = _receiver()
    for n in (A, B, Cn):
        _held(rg, rv, n)
    ack = tmp_path / "ack.txt"
    out = tmp_path / "frame.conduit"
    _export_frame(vg, vv, out, set())
    stats = _merge(rg, rv, out, ack)
    assert stats["skipped_present"] == 3 and stats["absorbed_nodes"] == 0
    assert stats["absorbed_synapses"] == 1 and stats["absorbed_hyperedges"] == 1
    assert _all_unbound(rg) == set()
    assert _ack(ack) == {A, B, Cn}


def test_membership_stale_readmitted_counts_a_culled_BOUND_node_not_a_culled_unbound_one(tmp_path):
    """#918 narrows `membership_stale_readmitted` (the Tier-1 `nid in held` branch): `held` is the last ack, which no longer
    lists unbound nodes, so a node culled while UNBOUND is re-absorbed as a plain new node. The same node acked BOUND and then
    culled still counts (the #106/#110 case the stat exists for). On the base the first call below reports 1, not 0."""
    U, C = "cc:conv::u-held", "cc:conv::c"
    vg, vv = Graph(), SimpleVectorDB()
    _vps(vg, vv, C, 10)
    _vps(vg, vv, U, 200)
    _he(vg, [U, C])
    rg, rv = _receiver()
    _held(rg, rv, U)
    ack = tmp_path / "ack.txt"
    _merge(rg, rv, _bound_pair_conduit(tmp_path), ack)                  # U stays unbound -> NOT in the ack
    assert U in _all_unbound(rg) and U not in _ack(ack)
    rg.remove_node(U)                                                   # culled while unbound
    out = tmp_path / "frame.conduit"
    _export_frame(vg, vv, out, _ack(ack))
    st = _merge(rg, rv, out, ack)
    assert st["absorbed_nodes"] == 2 and st["membership_stale_readmitted"] == 0
    assert U not in _all_unbound(rg) and U in _ack(ack)                 # bound by the hyperedge, so now acked
    rg.remove_node(U)                                                   # culled again, this time acked BOUND
    st = _merge(rg, rv, out, ack)
    assert st["membership_stale_readmitted"] == 1


# ============================================================== (3) the end-to-end loop

def _world():
    """A synthetic VPS (sender) graph and a laptop (receiver) graph. Returns (vg, vv, lg, lv, ids)."""
    vg, vv = Graph(), SimpleVectorDB()
    lg, lv = _receiver()
    ids = {}
    # (i) un-held nodes bound by a new VPS hyperedge: oldest, so they fill tick 1
    ids["new"] = [f"cc:conv::new-{k}" for k in range(6)]
    for k, n in enumerate(ids["new"]):
        _vps(vg, vv, n, 1 + k)
    _he(vg, ids["new"])
    # COHORT: a hyperedge whose members the laptop ALL holds unbound (the #909 deadlock shape: no un-acked member to trigger it)
    ids["cohort"] = [f"cc:conv::coh-{k}" for k in range(3)]
    for k, n in enumerate(ids["cohort"]):
        _vps(vg, vv, n, 200 + k)
        _held(lg, lv, n)
    _he(vg, ids["cohort"])
    # PAIR: forest + tree (id embeds the user's words), both held unbound, bound on the VPS by a hyperedge
    ids["pair"] = ["cc:conv::pair-f", "cc:conv::pair-f::tree::words-the-user-typed"]
    for k, n in enumerate(ids["pair"]):
        _vps(vg, vv, n, 210 + k)
        _held(lg, lv, n)
    _he(vg, ids["pair"])
    # (iii) two held-unbound nodes in a VPS hyperedge whose OTHER member (A3) the laptop holds BOUND (acked)
    ids["iii"] = ["cc:conv::iii-0", "cc:conv::iii-1"]
    a3, b3 = "cc:conv::iii-acked", "cc:conv::iii-acked-b"
    for k, n in enumerate(ids["iii"] + [a3]):
        _vps(vg, vv, n, 220 + k)
    for n in ids["iii"]:
        _held(lg, lv, n)
    for n in (a3, b3):
        _held(lg, lv, n)
    lg.create_synapse(a3, b3, weight=0.4, delay=1)
    lg.create_synapse(b3, a3, weight=0.4, delay=1)
    _vps(vg, vv, b3, 230)
    _he(vg, ids["iii"] + [a3])
    # (v) held-unbound X5 with a VPS SYNAPSE to an un-held candidate C5
    ids["v"] = ["cc:conv::v-held"]
    _vps(vg, vv, "cc:conv::v-cand", 10)
    _vps(vg, vv, ids["v"][0], 240)
    vg.create_synapse("cc:conv::v-cand", ids["v"][0], weight=0.3, delay=1)
    _held(lg, lv, ids["v"][0])
    # (iv) held-unbound, but the VPS holds it with NO edge: never a candidate;  (vi) the VPS does not hold it at all
    ids["iv"] = ["cc:conv::iv-held"]
    _vps(vg, vv, ids["iv"][0], 250)
    _held(lg, lv, ids["iv"][0])
    ids["vi"] = ["cc:conv::vi-held"]
    _held(lg, lv, ids["vi"][0])
    ids["acked_bound"] = [a3, b3]
    return vg, vv, lg, lv, ids


def _run_loop(tmp_path, ticks=8):
    vg, vv, lg, lv, ids = _world()
    ack, out = tmp_path / "laptop_cc_membership.json", tmp_path / "vps_topology.conduit"
    tmg._write_membership(str(ack), tmg.cc_current_membership(lg))        # the ack a PRIOR merge call left on the laptop
    frames, acks, exhausted = [], [], False
    for _ in range(ticks):
        st = _export_frame(vg, vv, out, tmg._load_membership(str(ack)))  # exactly the sender's read: _load_membership -> exclude_ids
        frames.append(set(st.get("frame_node_ids") or ()))
        exhausted = bool(st.get("exhausted"))
        _merge(lg, lv, out, ack)
        acks.append(_ack(ack))
        if exhausted:
            break
    return lg, ids, frames, acks, exhausted


def _old_membership(graph):
    """The BASE definition (#110, before #918): every CC-provenance node, bound or not."""
    with graph._step_lock:
        return {nid for nid, node in graph.nodes.items()
                if tex.is_cc_provenance(nid, getattr(node, "metadata", None) or {})}


def test_the_loop_re_offers_held_unbound_nodes_with_their_hyperedge_and_binds_them(tmp_path):
    lg, ids, frames, acks, exhausted = _run_loop(tmp_path)
    unbound = _all_unbound(lg)
    for case in ("cohort", "pair", "iii", "v", "new"):
        assert not (set(ids[case]) & unbound), f"{case}: still unbound after the loop"
        assert set(ids[case]) <= set().union(*frames), f"{case}: never offered"
    # the cohort/pair/(iii) were offered AS NODES (they are held): they arrived whole and now carry their hyperedge
    for case in ("cohort", "pair", "iii"):
        members = set(ids[case]) | ({"cc:conv::iii-acked"} if case == "iii" else set())
        assert any(he.member_nodes == members for he in lg.hyperedges.values()), case
    # never offered: not connected on the VPS (iv), not held on the VPS (vi) -- the laptop-only forest cannot starve the loop
    for case in ("iv", "vi"):
        assert not (set(ids[case]) & set().union(*frames))
        assert set(ids[case]) <= unbound and not (set(ids[case]) & acks[-1])
    assert exhausted, "the loop must reach 'nothing left to send' (no permanent candidate)"


def test_the_ack_after_every_tick_contains_only_bound_nodes(tmp_path):
    lg, ids, frames, acks, _ = _run_loop(tmp_path)
    assert acks
    final_unbound = _all_unbound(lg)
    for tick, ack in enumerate(acks):
        assert ack <= tmg.cc_current_membership(lg) | ack          # (shape guard)
        assert not any(n in ack for n in ids["vi"] + ids["iv"]), tick
    assert not (acks[-1] & final_unbound)
    assert acks[-1] == tmg.cc_current_membership(lg)
    # and the ack GROWS only as nodes bind: every held-unbound case joins it only after it is bound
    for case in ("cohort", "pair", "iii", "v"):
        assert set(ids[case]) <= acks[-1]


def test_CONTROL_with_the_old_ack_the_same_loop_never_re_offers_the_held_unbound_cohort(tmp_path, monkeypatch):
    """The #909 deadlock, reproduced in-file: patch the membership back to 'every CC node'. The cohort, the pair and the (iii)
    nodes are held UNBOUND, in the ack, so never a candidate; they stay unbound forever. (This control passes on the base AND
    the new file: it replaces the function under test with the old definition.)"""
    monkeypatch.setattr(tmg, "cc_current_membership", _old_membership)
    lg, ids, frames, acks, exhausted = _run_loop(tmp_path)
    unbound = _all_unbound(lg)
    offered = set().union(*frames)
    for case in ("cohort", "pair", "iii"):
        assert set(ids[case]) <= unbound, f"{case}: bound without ever being re-offered?"
        assert not (set(ids[case]) & offered), f"{case}: re-offered under the old ack"
    assert set(ids["new"]) <= offered and not (set(ids["new"]) & unbound)    # the loop is not dead -- only the cohort is
    assert not (set(ids["v"]) & unbound)       # bound by the ARRIVING candidate's synapse to the acked node (works either way)
    assert exhausted


# ============================================================== (4) the starvation counter

def _stuck_world(tmp_path, nodes=("cc:conv::stk-a::tree::the-users-secret-concept", "cc:conv::stk-b")):
    """Held-unbound nodes in a VPS hyperedge, a real frame from the real exporter, and a laptop whose hyperedge creation is
    faulted: the nodes are re-offered every call and STAY unbound."""
    vg, vv = Graph(), SimpleVectorDB()
    rg, rv = _receiver()
    for k, n in enumerate(nodes):
        _vps(vg, vv, n, 10 + k)
        _held(rg, rv, n)
    _he(vg, list(nodes))
    out = tmp_path / "stuck.conduit"
    _export_frame(vg, vv, out, set(), frame_size=max(FRAME, len(nodes)))

    def boom(*a, **kw):
        raise RuntimeError("synthetic: the hyperedge cannot be created")

    rg.create_hyperedge = boom
    return rg, rv, out, list(nodes)


def _warns(caplog):
    return [r for r in caplog.records if r.name == tmg.logger.name and r.levelno == logging.WARNING
            and "re-offered" in r.getMessage()]


def test_one_warning_per_node_exactly_when_the_streak_reaches_N_none_before_none_after(tmp_path, caplog):
    rg, rv, out, nodes = _stuck_world(tmp_path)
    ack = tmp_path / "ack.txt"
    N = tmg._REOFFER_WARN_STREAK_DEFAULT
    assert N == 5
    per_call = []
    for call in range(1, N + 3):
        caplog.clear()
        st = _merge(rg, rv, out, ack)
        per_call.append((call, len(_warns(caplog)), st["reoffer_streak_warnings"], st["reoffered_unbound"],
                         st["reoffer_streak_nodes_at_or_over"]))
        assert set(nodes) <= _all_unbound(rg)                           # they STAYED unbound
    for call, logged, in_stats, still, over in per_call:
        assert still == 2
        if call < N:
            assert (logged, in_stats, over) == (0, 0, 0), per_call
        elif call == N:
            assert (logged, in_stats, over) == (2, 2, 2), per_call       # one per node, both reach N together
        else:
            assert (logged, in_stats, over) == (0, 0, 2), per_call       # never repeated; still counted as at/over N


def test_the_warning_carries_only_redacted_forms_the_streak_and_the_over_count_no_raw_id_no_words(tmp_path, caplog):
    rg, rv, out, nodes = _stuck_world(tmp_path)
    ack = tmp_path / "ack.txt"
    caplog.set_level(logging.DEBUG)
    for _ in range(tmg._REOFFER_WARN_STREAK_DEFAULT):
        _merge(rg, rv, out, ack)
    recs = _warns(caplog)
    assert len(recs) == 2
    keys = {tmg.redact_node_id(n) for n in nodes}
    seen = set()
    for r in recs:
        msg = r.getMessage()
        hit = [k for k in keys if k in msg]
        assert len(hit) == 1                                            # exactly ONE redacted id per record
        seen.add(hit[0])
        assert "5 consecutive" in msg and "N=5" in msg and "2 node(s) now at/over it" in msg
    assert seen == keys
    # LAW 7: nothing the user typed, no raw id, no id substring, in ANY record of the module logger
    for r in caplog.records:
        if r.name != tmg.logger.name:
            continue
        text = r.getMessage() + " " + str(r.args)
        for forbidden in ("the-users-secret-concept", "secret", "words", "stk-a", "stk-b", "cc:conv::", "::tree::"):
            assert forbidden not in text, (forbidden, text)
    # the in-memory table is keyed by the redacted form too, never the raw id
    assert set(tmg._reoffer_streaks) == keys
    assert not any(n in tmg._reoffer_streaks for n in nodes)


def test_the_counter_uses_the_one_redaction_helper(tmp_path, monkeypatch):
    rg, rv, out, nodes = _stuck_world(tmp_path)
    seen = []
    real = tmg.redact_node_id
    monkeypatch.setattr(tmg, "redact_node_id", lambda nid: seen.append(nid) or real(nid))
    _merge(rg, rv, out, tmp_path / "ack.txt")
    assert set(nodes) <= set(seen)


def test_the_streak_resets_when_the_node_is_bound(tmp_path, caplog):
    rg, rv, out, nodes = _stuck_world(tmp_path)
    ack = tmp_path / "ack.txt"
    for _ in range(3):
        _merge(rg, rv, out, ack)
    assert set(tmg._reoffer_streaks.values()) == {3}
    del rg.create_hyperedge                                             # the fault clears: the real method is back
    st = _merge(rg, rv, out, ack)
    assert not (set(nodes) & _all_unbound(rg))                          # bound
    assert st["reoffered_unbound"] == 0 and tmg._reoffer_streaks == {}
    caplog.clear()
    for _ in range(tmg._REOFFER_WARN_STREAK_DEFAULT + 1):
        _merge(rg, rv, out, ack)
    assert _warns(caplog) == []


def test_the_streak_resets_on_a_gap_a_call_that_did_not_re_send_it(tmp_path, caplog):
    rg, rv, out, nodes = _stuck_world(tmp_path)
    ack = tmp_path / "ack.txt"
    N = tmg._REOFFER_WARN_STREAK_DEFAULT
    for _ in range(N - 1):
        _merge(rg, rv, out, ack)
    assert set(tmg._reoffer_streaks.values()) == {N - 1}
    st = _merge(rg, rv, _bound_pair_conduit(tmp_path), ack)            # a call whose frames do NOT carry them
    assert tmg._reoffer_streaks == {} and st["reoffered_unbound"] == 0
    caplog.clear()
    for _ in range(N - 1):                                              # resumes at 1: N-1 more calls is still short of N
        _merge(rg, rv, out, ack)
    assert _warns(caplog) == []
    caplog.clear()
    _merge(rg, rv, out, ack)                                            # streak N again -> the warning is allowed again
    assert len(_warns(caplog)) == 2


def test_a_call_with_no_conduit_resets_the_streaks_too(tmp_path):
    rg, rv, out, nodes = _stuck_world(tmp_path)
    _merge(rg, rv, out, tmp_path / "ack.txt")
    assert tmg._reoffer_streaks
    assert _merge(rg, rv, tmp_path / "absent.conduit", tmp_path / "ack.txt")["status"] == "no_conduit"
    assert tmg._reoffer_streaks == {}


def test_the_threshold_is_overridable_by_the_one_env_variable_and_bad_values_fall_back(tmp_path, caplog, monkeypatch):
    assert tmg._REOFFER_WARN_STREAK_ENV == _ENV
    monkeypatch.setenv(_ENV, "2")
    rg, rv, out, nodes = _stuck_world(tmp_path)
    ack = tmp_path / "ack.txt"
    _merge(rg, rv, out, ack)
    assert _warns(caplog) == []
    _merge(rg, rv, out, ack)
    assert len(_warns(caplog)) == 2 and "N=2" in _warns(caplog)[0].getMessage()
    for bad, expect in (("not-a-number", 5), ("", 5), ("0", 1), ("-3", 1)):
        monkeypatch.setenv(_ENV, bad)
        tmg._reoffer_streaks = {}
        caplog.clear()
        for _ in range(expect):
            _merge(rg, rv, out, ack)
        assert len(_warns(caplog)) == 2, (bad, expect)


def test_the_table_is_hard_capped_and_keeps_the_longest_streaks(tmp_path, monkeypatch):
    monkeypatch.setattr(tmg, "_REOFFER_TABLE_CAP", 3)
    nodes = tuple(f"cc:conv::cap-{k}" for k in range(6))
    vg, vv = Graph(), SimpleVectorDB()
    rg, rv = _receiver()
    for k in range(0, 6, 2):
        _vps(vg, vv, nodes[k], 10 + k)
        _vps(vg, vv, nodes[k + 1], 11 + k)
        _held(rg, rv, nodes[k])
        _held(rg, rv, nodes[k + 1])
        _he(vg, [nodes[k], nodes[k + 1]])
    out = tmp_path / "cap.conduit"
    _export_frame(vg, vv, out, set(), frame_size=6)

    def boom(*a, **kw):
        raise RuntimeError("synthetic")

    rg.create_hyperedge = boom
    for call in range(1, 5):
        st = _merge(rg, rv, out, tmp_path / "ack.txt")
        assert st["reoffered_unbound"] == 6                              # the COUNT is not capped, only the table
        assert len(tmg._reoffer_streaks) <= 3, call
    assert len(tmg._reoffer_streaks) == 3
    assert set(tmg._reoffer_streaks.values()) == {4}                     # the retained entries kept climbing


def test_a_protected_unbound_node_re_sent_is_not_sweep_eligible_so_it_never_builds_a_streak():
    g = Graph()
    g.timestep = OLD
    g.create_node(node_id="cc:want::authored", metadata=_meta(provenance="cc_authored")).creation_time = 0
    g.create_node(node_id="cc:conv::plain", metadata=_meta()).creation_time = 0
    for _ in range(tmg._REOFFER_WARN_STREAK_DEFAULT + 2):
        r = tmg._track_reoffers(g, {"cc:want::authored", "cc:conv::plain"})
    assert r["still_unbound"] == 1                                       # only the unprotected one counts
    assert set(tmg._reoffer_streaks) == {tmg.redact_node_id("cc:conv::plain")}


def test_the_stats_keys_exist_on_every_ok_call_and_the_counter_is_inert_when_nothing_is_reoffered(tmp_path):
    rg, rv = _receiver()
    st = _merge(rg, rv, _bound_pair_conduit(tmp_path), tmp_path / "ack.txt")
    for key in ("reoffered_unbound", "reoffer_streak_warnings", "reoffer_streak_nodes_at_or_over"):
        assert st[key] == 0
    assert tmg._reoffer_streaks == {}


def test_the_counter_never_touches_the_ack_or_any_file(tmp_path):
    rg, rv, out, nodes = _stuck_world(tmp_path)
    ack = tmp_path / "ack.txt"
    before = set(os.listdir(tmp_path))
    for _ in range(6):
        _merge(rg, rv, out, ack)
    assert set(os.listdir(tmp_path)) - before <= {"ack.txt"}            # nothing persisted for the counter
    assert _ack(ack) == tmg.cc_current_membership(rg)
    assert not any(tmg.redact_node_id(n) in ack.read_text() for n in nodes)


def test_the_streak_table_is_a_replaced_not_mutated_dict():
    """Copy-on-write: a concurrent reader holding the previous table never sees it change under it."""
    g = Graph()
    g.timestep = OLD
    g.create_node(node_id="cc:conv::z", metadata=_meta()).creation_time = 0
    tmg._track_reoffers(g, {"cc:conv::z"})
    held_ref = tmg._reoffer_streaks
    snapshot = dict(held_ref)
    tmg._track_reoffers(g, {"cc:conv::z"})
    assert held_ref == snapshot and tmg._reoffer_streaks is not held_ref
