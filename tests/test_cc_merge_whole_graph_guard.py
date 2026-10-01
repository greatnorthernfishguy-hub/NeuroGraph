#!/usr/bin/env python3
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane ack-bound-918, dispatch #13352) — #918 expectation updates (spy scope only)
# What: test_d_idle_steps_zero_never_evaluates_the_guard... and test_d_the_guard_is_asked_about_every_node...: their spy on
#   `tmg._unbound_nodes` now counts only the GUARD's asks (callers `merge_cc_topology` = the batch-end check, `_guard` = the
#   per-slice guard). #918 makes two MORE callers of the ONE predicate -- `cc_current_membership` (the ack = CC nodes minus the
#   sweep-eligible-unbound set) and `_track_reoffers` (the re-offer counter) -- and they run after the batches, even at
#   idle_steps=0. The second test also pins those two other callers (names, and that both read under _step_lock).
# Why: #918 (Exec Packet 496). Nothing the tests assert about the GUARD changed (still 0 asks at idle_steps=0; still exactly 2
#   whole-graph asks under the lock, steps unlocked); the spy used to see every call to the predicate, and there are now more.
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane emergent-want-bound-905, dispatch #13138) — #905 expectation updates
# What: only where #905 changes a thing these tests pinned: (1) the per-batch ERROR sample is REDACTED (Part C: kind:sha256-12,
#   never the id) -- test_a_preexisting..., test_c_an_arrival..., test_c_arrival_scope..., test_c_mixed..., test_c_the_id_sample...;
#   (2) the merge now passes guard=/progress= to _cc_callosum_consolidate and the guard predicate is also called once per slice
#   (Part D) -- test_d_the_guard_is_asked_about_every_node... (spy signature; 2 asks, both under _step_lock, steps still unlocked);
#   (3) one new stats key, consolidation_held_midpass -- test_d_every_existing_stat_key... (set difference).
#   NO test here pinned 'protection is not consulted': none was changed for Part B.
# [2026-09-30] Claude Sonnet 5.5 (Z12 builder, lane merge-guard-897, dispatch #12973) — #897 whole-graph consolidation guard tests
# What: drives the REAL cc_topology_merge.merge_cc_topology (real conduit frames written by the real exporter, real
#   cc_ng_organism._cc_callosum_consolidate, real neuro_foundation.Graph.step() and orphan sweep) into a REAL in-memory
#   Graph + SimpleVectorDB. Cases: (a) the defect, (b) the cohort binding mid-merge, (c) arrival-scoped behaviour kept,
#   (d) the no-unbound baseline + idle_steps==0 never evaluating the guard + lock discipline, (e) the sweep consequence.
# Why: Exec Packet 484 Finding 1 / Chief-003 GO (#897): the merge's own guard was merge-scoped, so the first Leg 2 tick whose
#   arrivals were all bound would have run 250 steps and the orphan sweep would have reaped the laptop CC's 147 pre-existing
#   unbound conversational nodes. The guard must ask the same question as the daemon's #896 rule 1: the whole graph.
# How: no fakes for the graph, the sweep, the exporter or the consolidation. Z12_MERGE_UNDER_TEST (a path) loads a different
#   cc_topology_merge.py as the module under test, so this SAME file runs unchanged against the BASE file and against a
#   mutant (the guard put back to merge_landed) to show which cases fail. Z12_MERGE_BASE_REF (a path) enables the
#   existing-keys-identical-to-base comparison. P379/#770: the printed preamble names every module under test.
# -------------------
"""Which cases FAIL on the BASE (merge-scoped) guard and on the merge_landed mutant: see the return file; the defect, the
cohort-binds-mid-merge and the mixed case encode the new behaviour, (c) and (d) pin what must not change."""
import importlib.util
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKTREE))
# #918: the merge's GUARD asks `_unbound_nodes` from these two frames only (the batch-end check in merge_cc_topology and the
# per-slice closure from whole_graph_guard); cc_current_membership and _track_reoffers are the other two callers.
_GUARD_CALLERS = ("merge_cc_topology", "_guard")

from neuro_foundation import Graph  # noqa: E402
from universal_ingestor import SimpleVectorDB  # noqa: E402
import cc_ng_organism as cno  # noqa: E402
import cc_topology_export as tex  # noqa: E402


def _load_by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_UNDER_TEST = os.environ.get("Z12_MERGE_UNDER_TEST")
if _UNDER_TEST:
    tmg = _load_by_path(Path(_UNDER_TEST).resolve(), "cc_topology_merge_under_test")
else:
    import cc_topology_merge as tmg  # noqa: E402

print("[P379/#770 preamble] worktree root          ->", _WORKTREE)
print("[P379/#770 preamble] merge under test       ->", Path(tmg.__file__).resolve())
for _m in (cno, tex):
    print("[P379/#770 preamble] %-20s ->" % _m.__name__, Path(_m.__file__).resolve())

DIM = 768
IDLE = 5            # a few real steps: the sweep runs inside every step(), so 5 is enough to reap an aged orphan
OLD = 100           # receiver timestep: a node created at 0 is 100 steps old, grace is 25
ORPHAN = "cc:conv::old-orphan"


def test_modules_under_test_resolve_inside_the_worktree():
    if _UNDER_TEST:
        pytest.skip("Z12_MERGE_UNDER_TEST is set: the module under test is deliberately %s" % _UNDER_TEST)
    for mod in (tmg, cno, tex):
        assert _WORKTREE in Path(mod.__file__).resolve().parents, f"{mod.__name__} resolves outside {_WORKTREE}"


# ---------------------------------------------------------------- rig

_SEED = [0]


def _emb():
    _SEED[0] += 1
    return np.random.default_rng(_SEED[0]).normal(size=DIM).astype(np.float32)


def _meta(mode):
    return {"cc": True, "creation_mode": mode, "_forest_content": "TEXT-MUST-NEVER-BE-LOGGED"}


def _pair(g, v, tag, t0):
    """A forest + tree joined by one synapse: a bound pair. creation_time orders the export."""
    f, t = f"cc:conv::{tag}", f"cc:conv::{tag}::tree::x"
    for i, (nid, mode) in enumerate(((f, "conversational"), (t, "tree"))):
        meta = _meta(mode)
        g.create_node(node_id=nid, metadata=meta).creation_time = t0 + i
        v.insert(id=nid, embedding=_emb(), content="content for " + nid, metadata=meta)
    g.create_synapse(f, t, weight=0.4, delay=2)
    return f, t


def _lone(g, v, tag, t0):
    """A cc node with no synapse and no hyperedge: an arrival that cannot bind."""
    nid = f"cc:conv::{tag}"
    meta = _meta("conversational")
    g.create_node(node_id=nid, metadata=meta).creation_time = t0
    v.insert(id=nid, embedding=_emb(), content="content for " + nid, metadata=meta)
    return nid


def _export(g, v, tmp_path, name="topo.conduit", **kw):
    out = str(tmp_path / name)
    kw.setdefault("machine_id", "vps")
    kw.setdefault("embedding_model", "test-model")
    tex.export_cc_topology(g, v, out, **kw)
    return out


def _merge(rg, rv, path, tmp_path, **kw):
    kw.setdefault("local_machine_id", "laptop")
    kw.setdefault("expected_embedding_model", "test-model")
    kw.setdefault("membership_path", str(tmp_path / "membership.txt"))
    kw.setdefault("idle_steps", IDLE)
    return tmg.merge_cc_topology(rg, rv, path, **kw)


def _receiver_with_orphan(grace_old=True):
    """The laptop CC graph's shape: a PRE-EXISTING unbound conversational node, older than orphan grace."""
    rg, rv = Graph(), SimpleVectorDB()
    rg.create_node(node_id=ORPHAN, metadata=_meta("conversational"))     # creation_time 0, no synapse, no hyperedge
    if grace_old:
        rg.timestep = OLD
        assert rg.timestep - rg.nodes[ORPHAN].creation_time > rg.config["orphan_node_grace_period"]
    return rg, rv


def _multi_batch_conduit_binding_orphan(tmp_path, n_pairs, bind_in_batch):
    """n_pairs bound pairs, one pair per batch (real exporter, batch_size=2), plus ONE synapse ORPHAN->tree injected
    into batch `bind_in_batch` (1-based). That is the real Leg 2 shape: the paced frame exporter ships a synapse whose
    other end the receiver already holds ("edges-to-acked survive", cc_topology_export.py export_cc_topology_frame);
    the orphan itself is NOT in the conduit's nodes, so it is not one of this merge's arrivals."""
    sg, sv = Graph(), SimpleVectorDB()
    trees = []
    for i in range(n_pairs):
        trees.append(_pair(sg, sv, f"p{i}", t0=10 * i)[1])
    path = _export(sg, sv, tmp_path, "src.conduit", batch_size=2)
    frames = list(tex.read_topology_frames(open(path, "rb").read()))
    assert [f["kind"] for f in frames] == ["header"] + ["batch"] * n_pairs
    frames[bind_in_batch]["synapses"].append(
        {"pre": ORPHAN, "post": trees[bind_in_batch - 1], "weight": 0.3, "delay": 1,
         "synapse_type": "excitatory", "max_weight": 5.0})
    out = str(tmp_path / "bound-by-later-frame.conduit")
    with open(out, "wb") as fh:
        for fr in frames:
            fh.write(tex._frame(fr))
    return out


def _errors(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR and r.name == tmg.logger.name]


def _blocked_records(caplog):
    return [m for m in _errors(caplog) if "skipping" in m and "consolidation step" in m]


# ---------------------------------------------------------------- (a) the defect

def test_a_preexisting_unbound_node_blocks_the_steps_even_when_every_arrival_binds(tmp_path, caplog):
    sg, sv = Graph(), SimpleVectorDB()
    _pair(sg, sv, "A", 0)
    path = _export(sg, sv, tmp_path)
    rg, rv = _receiver_with_orphan()

    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, path, tmp_path)

    assert st["absorbed_nodes"] == 2 and st["absorbed_synapses"] == 1      # Leg 2 frames still merge and bind
    assert rg.timestep == OLD                                              # the clock did NOT move
    assert ORPHAN in rg.nodes                                              # the 147-node stand-in survives
    assert st["consolidation_passes"] == 0 and st["consolidation_steps"] == 0
    assert st["consolidation_blocked_batches"] == 1
    assert st["consolidation_skipped_unbound_preexisting"] == 1
    assert st["consolidation_skipped_unbound_arrivals"] == 0               # no arrival is unbound
    recs = _blocked_records(caplog)
    assert len(recs) == 1
    msg = recs[0]
    assert "after batch 1" in msg and "skipping %d consolidation" % IDLE in msg
    assert "1 node(s) in the graph are still unbound" in msg
    assert "1 pre-existing (not landed by this merge), 0 from this merge" in msg
    assert tmg.redact_node_id(ORPHAN) in msg and ORPHAN not in msg         # the sample is REDACTED (#905 C), never the id
    assert "TEXT-MUST-NEVER-BE-LOGGED" not in msg and "content for" not in msg     # ids only, never node text
    assert "orphan grace" in msg and "CC-CALLOSUM-TRUTH" in msg           # says WHY the clock is held
    assert [r.levelno for r in caplog.records if "consolidation step" in r.getMessage()] == [logging.ERROR]


def test_a_binding_the_orphan_releases_the_steps_for_the_next_frame_by_exactly_idle_steps(tmp_path, caplog):
    sg, sv = Graph(), SimpleVectorDB()
    _pair(sg, sv, "A", 0)
    path1 = _export(sg, sv, tmp_path, "one.conduit")
    rg, rv = _receiver_with_orphan()
    _merge(rg, rv, path1, tmp_path)
    assert rg.timestep == OLD and ORPHAN in rg.nodes                       # blocked, as in the defect test

    # Leg 2 (here: the local graph) binds the orphan: one synapse to a partner
    rg.create_node(node_id="cc:conv::partner", metadata=_meta("tree"))
    rg.create_synapse(ORPHAN, "cc:conv::partner", weight=0.2)
    assert tmg._unbound_nodes(rg, set(rg.nodes)) == set()

    sg2, sv2 = Graph(), SimpleVectorDB()
    _pair(sg2, sv2, "B", 0)
    path2 = _export(sg2, sv2, tmp_path, "two.conduit")
    caplog.clear()                                                         # the first merge's (legitimate) ERROR is not this one's
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, path2, tmp_path)

    assert st["consolidation_passes"] == 1 and st["consolidation_steps"] == IDLE
    assert st["consolidation_blocked_batches"] == 0
    assert st["consolidation_skipped_unbound_preexisting"] == 0 and st["consolidation_skipped_unbound_arrivals"] == 0
    assert rg.timestep == OLD + IDLE                                       # advanced by EXACTLY idle_steps
    assert ORPHAN in rg.nodes and "cc:conv::B" in rg.nodes                 # bound, so the sweep spares them
    assert _blocked_records(caplog) == []


# ---------------------------------------------------------------- (b) the cohort binds mid-merge

@pytest.mark.parametrize("n_pairs,bind_in_batch", [(2, 2), (3, 2), (3, 3)])
def test_b_consolidation_runs_at_the_batch_where_the_graph_first_has_no_unbound_node(
        tmp_path, caplog, n_pairs, bind_in_batch):
    path = _multi_batch_conduit_binding_orphan(tmp_path, n_pairs, bind_in_batch)
    rg, rv = _receiver_with_orphan()

    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, path, tmp_path)

    blocked = bind_in_batch - 1                  # every batch before the binding one is skipped
    ran = n_pairs - blocked                      # the binding batch and every later one consolidate
    assert st["batches_read"] == n_pairs and st["absorbed_nodes"] == 2 * n_pairs
    assert st["consolidation_blocked_batches"] == blocked
    assert st["consolidation_skipped_unbound_preexisting"] == blocked      # the one orphan, once per blocked batch
    assert st["consolidation_skipped_unbound_arrivals"] == 0
    assert st["consolidation_passes"] == ran and st["consolidation_steps"] == IDLE * ran
    assert rg.timestep == OLD + IDLE * ran
    assert ORPHAN in rg.nodes and rg._outgoing.get(ORPHAN)                 # bound by the later frame, and alive
    recs = _blocked_records(caplog)
    assert len(recs) == blocked                                            # one record per blocked batch, none after
    for i, m in enumerate(recs, start=1):
        assert "after batch %d" % i in m


# ---------------------------------------------------------------- (c) arrival-scoped behaviour preserved + the mixed case

def test_c_an_arrival_that_does_not_bind_still_blocks_and_keeps_its_original_stat_meaning(tmp_path, caplog):
    sg, sv = Graph(), SimpleVectorDB()
    _pair(sg, sv, "A", 0)
    lone = _lone(sg, sv, "lone", -1)
    path = _export(sg, sv, tmp_path)
    rg, rv = Graph(), SimpleVectorDB()                                     # an EMPTY receiver: nothing pre-existing

    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, path, tmp_path)

    assert st["consolidation_passes"] == 0 and rg.timestep == 0 and lone in rg.nodes
    assert st["consolidation_skipped_unbound_arrivals"] == 1               # THIS merge's arrival, as before
    assert st["consolidation_skipped_unbound_preexisting"] == 0
    assert st["consolidation_blocked_batches"] == 1
    recs = _blocked_records(caplog)
    assert len(recs) == 1 and "0 pre-existing (not landed by this merge), 1 from this merge" in recs[0]
    assert tmg.redact_node_id(lone) in recs[0] and lone not in recs[0]     # redacted sample (#905 C)


def test_c_arrival_scope_holds_across_batches_and_the_blocked_batch_count_is_per_batch(tmp_path, caplog):
    """The cross-batch shape of the #108 guard test: an unbound arrival in batch 1 keeps every later batch blocked."""
    sg, sv = Graph(), SimpleVectorDB()
    lone = _lone(sg, sv, "lone", -1)
    _pair(sg, sv, "A", 0)
    _pair(sg, sv, "B", 10)
    path = _export(sg, sv, tmp_path, batch_size=1)
    rg, rv = Graph(), SimpleVectorDB()

    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, path, tmp_path)

    assert st["batches_read"] == 5 and st["consolidation_passes"] == 0 and rg.timestep == 0 and lone in rg.nodes
    assert st["consolidation_blocked_batches"] == 5
    # batch_size=1 puts each pair's two endpoints in different batches, so the pair's forest is ALSO transiently unbound
    # at the batch that lands only it: unbound arrivals per batch = lone | lone,A | lone | lone,B | lone = 1+2+1+2+1
    assert st["consolidation_skipped_unbound_arrivals"] == 7
    assert st["consolidation_skipped_unbound_preexisting"] == 0
    recs = _blocked_records(caplog)
    assert len(recs) == 5                                                  # the lone arrival is in every record, redacted (#905 C)
    assert all(tmg.redact_node_id(lone) in m and lone not in m for m in recs)


def test_c_mixed_a_preexisting_and_an_arrival_both_unbound_are_split_in_the_stats_and_the_record(tmp_path, caplog):
    sg, sv = Graph(), SimpleVectorDB()
    _pair(sg, sv, "A", 0)
    lone = _lone(sg, sv, "lone", -1)
    path = _export(sg, sv, tmp_path)
    rg, rv = _receiver_with_orphan()

    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, path, tmp_path)

    assert rg.timestep == OLD and ORPHAN in rg.nodes and lone in rg.nodes
    assert st["consolidation_blocked_batches"] == 1
    assert st["consolidation_skipped_unbound_arrivals"] == 1
    assert st["consolidation_skipped_unbound_preexisting"] == 1
    (msg,) = _blocked_records(caplog)
    assert "2 node(s) in the graph are still unbound" in msg
    assert "1 pre-existing (not landed by this merge), 1 from this merge" in msg
    # a sample sorted on the ids, printed REDACTED (#905 C)
    assert "Sample: ['%s', '%s']" % tuple(tmg.redact_node_id(n) for n in sorted((ORPHAN, lone))) in msg
    assert ORPHAN not in msg and lone not in msg


def test_c_the_id_sample_is_bounded_to_three_sorted_ids(tmp_path, caplog):
    sg, sv = Graph(), SimpleVectorDB()
    _pair(sg, sv, "A", 0)
    path = _export(sg, sv, tmp_path)
    rg, rv = Graph(), SimpleVectorDB()
    names = ["cc:conv::o%d" % i for i in (5, 3, 9, 1, 7)]
    for n in names:
        rg.create_node(node_id=n, metadata=_meta("conversational"))

    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, path, tmp_path)

    assert st["consolidation_skipped_unbound_preexisting"] == 5
    (msg,) = _blocked_records(caplog)
    assert "Sample: %s." % [tmg.redact_node_id(n) for n in sorted(names)[:3]] in msg   # sorted on ids, printed redacted
    assert not any(n in msg for n in names)                                # no raw id at all (#905 C)
    assert tmg.redact_node_id("cc:conv::o7") not in msg and tmg.redact_node_id("cc:conv::o9") not in msg   # 4th/5th not sampled


# ---------------------------------------------------------------- (d) baseline: nothing unbound

_EXISTING_KEYS_EXPECTED = {
    "absorbed_nodes": 2, "absorbed_synapses": 1, "absorbed_hyperedges": 0, "skipped_present": 0,
    "membership_stale_readmitted": 0, "skipped_not_cc": 0, "skipped_identity": 0, "bad_embedding": 0,
    "absorbed_without_embedding_DEFECT": 0, "skipped_synapses": 0, "skipped_hyperedges": 0,
    "hyperedge_id_reminted": 0, "batches_read": 1, "deferred_by_budget": 0,
    "consolidation_passes": 1, "consolidation_steps": IDLE, "consolidation_skipped_unbound_arrivals": 0,
}


def test_d_no_unbound_node_consolidates_exactly_as_before(tmp_path, caplog):
    sg, sv = Graph(), SimpleVectorDB()
    _pair(sg, sv, "A", 0)
    path = _export(sg, sv, tmp_path)
    rg, rv = Graph(), SimpleVectorDB()

    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, path, tmp_path)

    for k, want in _EXISTING_KEYS_EXPECTED.items():
        assert st[k] == want, (k, st[k], want)
    assert st["status"] == "ok" and st["completed"] is True
    assert st["consolidation_blocked_batches"] == 0 and st["consolidation_skipped_unbound_preexisting"] == 0
    assert rg.timestep == IDLE
    assert _errors(caplog) == []


def test_d_a_preexisting_BOUND_node_does_not_block(tmp_path):
    sg, sv = Graph(), SimpleVectorDB()
    _pair(sg, sv, "A", 0)
    path = _export(sg, sv, tmp_path)
    rg, rv = Graph(), SimpleVectorDB()
    _pair(rg, rv, "already-here", 0)                                       # pre-existing, but bound
    st = _merge(rg, rv, path, tmp_path)
    assert st["consolidation_passes"] == 1 and st["consolidation_blocked_batches"] == 0 and rg.timestep == IDLE


def test_d_idle_steps_zero_never_evaluates_the_guard_and_never_blocks(tmp_path, monkeypatch, caplog):
    sg, sv = Graph(), SimpleVectorDB()
    _pair(sg, sv, "A", 0)
    path = _export(sg, sv, tmp_path)
    rg, rv = _receiver_with_orphan()
    calls = []
    real = tmg._unbound_nodes

    def spy(g, ids):
        if sys._getframe(1).f_code.co_name in _GUARD_CALLERS:              # #918: only the GUARD's asks
            calls.append(set(ids))
        return real(g, ids)

    monkeypatch.setattr(tmg, "_unbound_nodes", spy)

    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, path, tmp_path, idle_steps=0)

    assert calls == []                                                     # the guard is not even asked
    assert st["consolidation_passes"] == 0 and st["consolidation_blocked_batches"] == 0
    assert st["consolidation_skipped_unbound_preexisting"] == 0 and st["consolidation_skipped_unbound_arrivals"] == 0
    assert rg.timestep == OLD and ORPHAN in rg.nodes
    assert _blocked_records(caplog) == []


def test_d_the_guard_is_asked_about_every_node_in_the_graph_under_the_step_lock(tmp_path, monkeypatch):
    """Pins the predicate (whole graph, the daemon's #896 rule 1 call) and the lock discipline: evaluated while
    graph._step_lock is held, and the consolidation steps run with it RELEASED (no lock held across the steps)."""
    sg, sv = Graph(), SimpleVectorDB()
    _pair(sg, sv, "A", 0)
    path = _export(sg, sv, tmp_path)
    rg, rv = Graph(), SimpleVectorDB()
    _pair(rg, rv, "already-here", 0)
    asked, owned_at_guard, owned_at_steps, others = [], [], [], []
    real = tmg._unbound_nodes

    def spy_guard(g, ids):
        caller = sys._getframe(1).f_code.co_name
        if caller in _GUARD_CALLERS:
            asked.append((set(ids), set(g.nodes)))
            owned_at_guard.append(g._step_lock._is_owned())
        else:                                                              # #918: the other two callers of the one predicate
            others.append((caller, g._step_lock._is_owned()))
        return real(g, ids)

    real_cons = cno._cc_callosum_consolidate

    def spy_cons(g, n, **kw):                                              # #905 D: the merge now passes guard=/progress=
        owned_at_steps.append(g._step_lock._is_owned())
        return real_cons(g, n, **kw)

    monkeypatch.setattr(tmg, "_unbound_nodes", spy_guard)
    monkeypatch.setattr(cno, "_cc_callosum_consolidate", spy_cons)

    st = _merge(rg, rv, path, tmp_path)

    assert st["consolidation_passes"] == 1
    # #905 D: TWO asks now -- the OUTER batch-end check, then the per-slice guard before the one slice (IDLE < 25)
    assert len(asked) == 2 and all(a[0] == a[1] == set(rg.nodes) for a in asked)   # ALL nodes, not merge_landed
    assert len(asked[0][0]) == 4                                           # 2 pre-existing + 2 arrivals
    assert owned_at_guard == [True, True] and owned_at_steps == [False]    # both reads under _step_lock; steps unlocked
    # #918: the ack and the re-offer counter each ask once, after the batches, also under _step_lock
    assert sorted(others) == [("_track_reoffers", True), ("cc_current_membership", True)]


def test_d_idle_steps_from_the_env_default_path_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("CC_NG_IDLE_STEPS", "7")
    sg, sv = Graph(), SimpleVectorDB()
    _pair(sg, sv, "A", 0)
    path = _export(sg, sv, tmp_path)
    rg, rv = Graph(), SimpleVectorDB()
    st = tmg.merge_cc_topology(rg, rv, path, local_machine_id="laptop", expected_embedding_model="test-model",
                               membership_path=str(tmp_path / "m.txt"))
    assert st["consolidation_steps"] == 7 and rg.timestep == 7


_BASE_REF = os.environ.get("Z12_MERGE_BASE_REF")


@pytest.mark.skipif(not _BASE_REF, reason="Z12_MERGE_BASE_REF (a path to e4ebf982:cc_topology_merge.py) is not set")
@pytest.mark.parametrize("shape", ["no_unbound", "arrival_unbound"])
def test_d_every_existing_stat_key_and_the_timestep_are_identical_to_the_base_module(tmp_path, shape):
    """The base and the module under test, on identical input where the arrival-scoped and the whole-graph guard
    agree (nothing pre-existing is unbound): every key the base returns has the same value, the graph clock the same."""
    base = _load_by_path(Path(_BASE_REF).resolve(), "cc_topology_merge_base_ref")
    results = []
    for mod, sub in ((base, "base"), (tmg, "new")):
        d = tmp_path / sub
        d.mkdir()
        sg, sv = Graph(), SimpleVectorDB()
        _pair(sg, sv, "A", 0)
        if shape == "arrival_unbound":
            _lone(sg, sv, "lone", -1)
        path = _export(sg, sv, d)
        rg, rv = Graph(), SimpleVectorDB()
        st = mod.merge_cc_topology(rg, rv, path, local_machine_id="laptop", expected_embedding_model="test-model",
                                   membership_path=str(d / "m.txt"), idle_steps=IDLE)
        st = {k: v for k, v in st.items() if k != "path"}
        results.append((st, rg.timestep))
    (b_st, b_ts), (n_st, n_ts) = results
    assert b_ts == n_ts
    for k, v in b_st.items():
        assert n_st[k] == v, (k, v, n_st[k])
    assert set(n_st) - set(b_st) == {"consolidation_skipped_unbound_preexisting", "consolidation_blocked_batches",
                                     "consolidation_held_midpass"}


# ---------------------------------------------------------------- (e) the real consequence (proves the tests are not vacuous)

def test_e_control_when_the_clock_does_run_the_sweep_really_reaps_the_aged_unbound_node():
    """No merge, no guard: advance the real clock over the aged orphan with the real consolidation function. It is
    reaped. This is what the guard exists to prevent, and why case (a)'s `ORPHAN in rg.nodes` is not vacuous."""
    rg, _ = _receiver_with_orphan()
    assert ORPHAN in rg.nodes
    assert cno._cc_callosum_consolidate(rg, IDLE) is True
    assert rg.timestep == OLD + IDLE
    assert ORPHAN not in rg.nodes


def test_e_the_old_predicate_is_blind_to_the_orphan_and_the_new_one_sees_it(tmp_path):
    """Executable statement of the defect, independent of which guard the module under test has: evaluate BOTH
    predicates on case (a)'s state. The merge-scoped one is blind to the orphan; the whole-graph one sees it. (The mutant
    is killed by case (a), which runs the real merge; this pins why.)"""
    sg, sv = Graph(), SimpleVectorDB()
    _pair(sg, sv, "A", 0)
    path = _export(sg, sv, tmp_path)
    rg, rv = _receiver_with_orphan()
    arrivals = {"cc:conv::A", "cc:conv::A::tree::x"}
    # reproduce the merge's landing without any consolidation (idle_steps=0), then ask both questions
    _merge(rg, rv, path, tmp_path, idle_steps=0)
    assert set(rg.nodes) == arrivals | {ORPHAN}
    assert tmg._unbound_nodes(rg, arrivals) == set()                       # the OLD guard: "all clear" -> 250 steps
    assert tmg._unbound_nodes(rg, set(rg.nodes)) == {ORPHAN}               # the NEW guard: blocked
