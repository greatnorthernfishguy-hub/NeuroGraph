#!/usr/bin/env python3
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane emergent-want-bound-905, dispatch #13138) — #905 tests (parts A, B, C, D)
# What: a REAL neuro_foundation.Graph (real step(), real orphan sweep, real _is_identity_protected) + the REAL
#   cc_topology_merge.merge_cc_topology driven by conduit frames from the REAL exporter, for:
#   (A) generate_emergent_want is BORN BOUND (synapse seed->want, weight 0.3, same lock; rollback + one WARNING when no seed
#       binds; one WARNING on a shortfall; success dict golden; reinforced/idempotent branches unchanged);
#   (B) _unbound_nodes = unbound AND sweep-eligible (protected does not block, unprotected does, NO age term, a double without
#       _is_identity_protected is counted), at the predicate level and through the real merge guard;
#   (C) redact_node_id (kind:sha256-12, never the id) and the redacted #897 ERROR sample for a TREE id;
#   (D) the per-slice guard inside the shared _cc_callosum_consolidate (guard=/progress= contract, whole_graph_guard, the merge
#       passing it and reporting consolidation_held_midpass).
# Why: Exec Packets 488 / 489 / 490 (Chief-003), #905.
# How: Z12_ORGANISM_UNDER_TEST / Z12_MERGE_UNDER_TEST (paths) load a different cc_ng_organism.py / cc_topology_merge.py as the
#   module under test, so this SAME file runs unchanged against the BASE files and against each scratch MUTANT to show which
#   cases fail (the mutants are killed by at least one case). P379/#770: the printed preamble names every module under test.
# -------------------
import hashlib
import importlib.util
import inspect
import logging
import os
import re
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKTREE))

from neuro_foundation import Graph, Prediction  # noqa: E402
from universal_ingestor import SimpleVectorDB  # noqa: E402
import cc_topology_export as tex  # noqa: E402


def _load_by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The organism first: the merge imports it lazily by NAME, so a mutant must own sys.modules["cc_ng_organism"].
_ORG_UNDER_TEST = os.environ.get("Z12_ORGANISM_UNDER_TEST")
_MERGE_UNDER_TEST = os.environ.get("Z12_MERGE_UNDER_TEST")
if _ORG_UNDER_TEST:
    cno = _load_by_path(Path(_ORG_UNDER_TEST).resolve(), "cc_ng_organism")
else:
    import cc_ng_organism as cno  # noqa: E402
if _MERGE_UNDER_TEST:
    tmg = _load_by_path(Path(_MERGE_UNDER_TEST).resolve(), "cc_topology_merge")
else:
    import cc_topology_merge as tmg  # noqa: E402

print("[P379/#770 preamble] worktree root        ->", _WORKTREE)
print("[P379/#770 preamble] organism under test  ->", Path(cno.__file__).resolve())
print("[P379/#770 preamble] merge under test     ->", Path(tmg.__file__).resolve())
for _m in (sys.modules["neuro_foundation"], tex):
    print("[P379/#770 preamble] %-18s ->" % _m.__name__, Path(_m.__file__).resolve())


def test_modules_under_test_resolve_inside_the_worktree():
    if _ORG_UNDER_TEST or _MERGE_UNDER_TEST:
        pytest.skip("a module under test is deliberately overridden (%s / %s)" % (_ORG_UNDER_TEST, _MERGE_UNDER_TEST))
    for mod in (tmg, cno, tex, sys.modules["neuro_foundation"]):
        assert _WORKTREE in Path(mod.__file__).resolve().parents, f"{mod.__name__} resolves outside {_WORKTREE}"


# ------------------------------------------------------------------ rig

DIM = 768
OLD = 100           # receiver/graph timestep: a node created at 0 is 100 steps old; orphan grace is 25
IDLE = 5
_SEED = [0]


def _emb():
    _SEED[0] += 1
    return np.random.default_rng(_SEED[0]).normal(size=DIM).astype(np.float32)


def _meta(mode="conversational", **extra):
    m = {"cc": True, "creation_mode": mode, "_forest_content": "TEXT-MUST-NEVER-BE-LOGGED"}
    m.update(extra)
    return m


def _node(g, nid, **meta):
    return g.create_node(node_id=nid, metadata=dict(meta))


def _pair(g, v, tag, t0):
    """A forest + tree joined by one synapse: a bound pair (the #897 rig)."""
    f, t = f"cc:conv::{tag}", f"cc:conv::{tag}::tree::x"
    for i, (nid, mode) in enumerate(((f, "conversational"), (t, "tree"))):
        meta = _meta(mode)
        g.create_node(node_id=nid, metadata=meta).creation_time = t0 + i
        v.insert(id=nid, embedding=_emb(), content="content for " + nid, metadata=meta)
    g.create_synapse(f, t, weight=0.4, delay=2)
    return f, t


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


def _recs(caplog, logger_name, min_level=logging.DEBUG):
    return [r for r in caplog.records if r.name == logger_name and r.levelno >= min_level]


def _blocked_records(caplog):
    return [r.getMessage() for r in _recs(caplog, tmg.logger.name, logging.ERROR)
            if "skipping" in r.getMessage() and "consolidation step" in r.getMessage()]


class _SpyLock:
    """Stands in for the host-attached graph._concurrent_lock (a real Graph has none). Counts slice exits, tracks hold
    depth, and runs `after_exit(n)` once the lock is RELEASED -- i.e. 'between slices', where a hook deposit lands."""

    def __init__(self, after_exit=None):
        self.inner = threading.RLock()
        self.after_exit, self.exits, self.depth = after_exit, 0, 0

    def __enter__(self):
        self.depth += 1
        return self.inner.__enter__()

    def __exit__(self, *a):
        r = self.inner.__exit__(*a)
        self.depth -= 1
        self.exits += 1
        if self.after_exit:
            self.after_exit(self.exits)
        return r


# ============================================================== (A) born bound

def _seeded_graph(seed_ids=("s1", "s2"), t=OLD):
    g = Graph()
    g.timestep = t
    for sid in seed_ids:
        _node(g, sid, label=sid)
    return g


def _predict(g, *pairs, conf=0.9):
    for i, (src, tgt) in enumerate(pairs):
        g.active_predictions["p%d" % i] = Prediction(source_node_id=src, target_node_id=tgt, confidence=conf - 0.01 * i)


def _want_text(*pairs):
    return "tonic-triggered: (unknown) -- open questions: " + ", ".join("%s→%s" % p for p in pairs)


def _want_id(text):
    return "cc:want::" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _degree(g, nid):
    return len(g._outgoing.get(nid) or ()) + len(g._incoming.get(nid) or ())


def test_a_the_want_is_born_bound_counted_bound_and_survives_the_real_sweep():
    g = _seeded_graph()
    _predict(g, ("s1", "t1"), ("s2", "t2"))
    res = cno.generate_emergent_want(g, None)

    text = _want_text(("s1", "t1"), ("s2", "t2"))
    assert res is not None and res["id"] == _want_id(text)
    wid = res["id"]
    assert wid in g.nodes and _degree(g, wid) >= 1                          # born bound
    assert not g._is_identity_protected(wid)                                # '*_emergent' is NOT protected: only the binding can save it
    assert tmg._unbound_nodes(g, set(g.nodes)) == set()                     # the whole-graph guards do not count it
    for _ in range(OLD // 2):                                               # >= 30 real steps, well past orphan grace 25
        g.step()
    assert g.timestep >= OLD + 30
    assert wid in g.nodes and "s1" in g.nodes and "s2" in g.nodes           # the real sweep did NOT reap it


def test_a_control_the_unbound_want_the_base_minted_is_counted_unbound_and_reaped():
    """Executable statement of the defect (the base created the want with NO synapse): counted unbound, then really reaped.
    Run against the BASE organism, test_a_the_want_is_born_bound... fails for exactly this reason."""
    g = _seeded_graph()
    wid = _want_id(_want_text(("s1", "t1")))
    _node(g, wid, kind="want", want_state="open", provenance="cc_emergent", creation_mode="emergent")
    assert wid in tmg._unbound_nodes(g, set(g.nodes))
    for _ in range(40):
        g.step()
    assert wid not in g.nodes


def test_a_one_synapse_per_existing_deduplicated_seed_weight_03_source_to_want():
    g = _seeded_graph(("s1", "s2"))
    _predict(g, ("s1", "t1"), ("s1", "t2"), ("s2", "t3"))                   # s1 twice: seed_ids == [s1, s1, s2]
    res = cno.generate_emergent_want(g, None)
    wid = res["id"]
    got = sorted((s.pre_node_id, s.post_node_id, round(float(s.weight), 6)) for s in g.synapses.values())
    assert got == [("s1", wid, 0.3), ("s2", wid, 0.3)]                      # de-duplicated; direction seed -> want
    assert not any(s.post_node_id == s.pre_node_id for s in g.synapses.values())


def _race_remove(g, *gone):
    """A seed reaped between prime_and_propagate and the want's lock block (the only way a real graph reaches the bind loop
    with a missing seed: the real prime raises KeyError on an id that is absent when it runs)."""
    real = g.prime_and_propagate

    def spy(*a, **kw):
        out = real(*a, **kw)
        for sid in gone:
            g.remove_node(sid)
        return out
    g.prime_and_propagate = spy


def test_a_a_missing_seed_is_skipped_and_counted_and_the_want_stays_bound_by_the_others(caplog):
    g = _seeded_graph(("s1", "s2"))
    _predict(g, ("s1", "t1"), ("s2", "t2"))
    _race_remove(g, "s2")
    with caplog.at_level(logging.DEBUG, logger="cc_ng_organism"):
        res = cno.generate_emergent_want(g, None)
    assert res is not None and res["id"] in g.nodes
    assert [(s.pre_node_id, s.post_node_id) for s in g.synapses.values()] == [("s1", res["id"])]
    warns = _recs(caplog, "cc_ng_organism", logging.WARNING)
    assert len(warns) == 1
    m = warns[0].getMessage()
    assert "seeds=2" in m and "bound=1" in m and "missing=1" in m and "failed=0" in m


def test_a_all_seeds_missing_leaves_no_node_returns_none_with_one_warning(caplog):
    g = _seeded_graph(("s1", "s2"))
    _predict(g, ("s1", "t1"), ("s2", "t2"))
    _race_remove(g, "s1", "s2")
    with caplog.at_level(logging.DEBUG, logger="cc_ng_organism"):
        res = cno.generate_emergent_want(g, None)
    assert res is None
    assert len(g.nodes) == 0                                                # both seeds were reaped by the race; the want was NOT left behind
    assert not any(n.startswith("cc:want::") for n in g._dirty_nodes)
    assert not g.synapses
    warns = _recs(caplog, "cc_ng_organism", logging.WARNING)
    assert len(warns) == 1 and warns[0].levelno == logging.WARNING
    m = warns[0].getMessage()
    assert "no_bindable_seed" in m and "seeds=2" in m and "missing=2" in m and "bound=0" in m and "rolled_back=True" in m


def test_a_a_create_synapse_that_raises_for_one_seed_keeps_the_want_bound_by_the_others_one_warning(caplog):
    g = _seeded_graph(("s1", "s2"))
    _predict(g, ("s1", "t1"), ("s2", "t2"))
    real = g.create_synapse

    def spy(pre, post, *a, **kw):
        if pre == "s1":
            raise RuntimeError("SECRET-USER-WORDS in an exception message")
        return real(pre, post, *a, **kw)
    g.create_synapse = spy
    with caplog.at_level(logging.DEBUG, logger="cc_ng_organism"):
        res = cno.generate_emergent_want(g, None)
    assert res is not None and _degree(g, res["id"]) == 1
    warns = _recs(caplog, "cc_ng_organism", logging.WARNING)
    assert len(warns) == 1
    m = warns[0].getMessage()
    assert "bound=1" in m and "failed=1" in m and "RuntimeError" in m
    assert "SECRET" not in " ".join(r.getMessage() for r in caplog.records)     # never str(exc), at any level


def test_a_no_seed_binds_rolls_the_node_back_inside_the_lock_returns_none_one_warning(caplog):
    g = _seeded_graph(("s1", "s2"))
    _predict(g, ("s1", "t1"), ("s2", "t2"))
    n_before = len(g.nodes)

    def boom(*a, **kw):
        raise RuntimeError("SECRET-USER-WORDS")
    g.create_synapse = boom
    real_remove, owned = g.remove_node, []

    def spy_remove(nid):
        owned.append(g._step_lock._is_owned())
        return real_remove(nid)
    g.remove_node = spy_remove
    with caplog.at_level(logging.DEBUG, logger="cc_ng_organism"):
        res = cno.generate_emergent_want(g, None)
    assert res is None
    assert owned == [True]                                                  # rolled back INSIDE the same lock block
    assert len(g.nodes) == n_before and not g.synapses
    assert not any(n.startswith("cc:want::") for n in g._dirty_nodes)       # no stale dirty entry for the removed node
    warns = _recs(caplog, "cc_ng_organism", logging.WARNING)
    assert len(warns) == 1
    m = warns[0].getMessage()
    assert "no_bindable_seed" in m and "failed=2" in m and "RuntimeError" in m and "rolled_back=True" in m
    assert "SECRET" not in " ".join(r.getMessage() for r in caplog.records)


def test_a_a_failed_rollback_is_loud_not_silent(caplog):
    g = _seeded_graph(("s1",))
    _predict(g, ("s1", "t1"))

    def boom(*a, **kw):
        raise RuntimeError("x")
    g.create_synapse = boom

    def boom_remove(nid):
        raise KeyError("SECRET-USER-WORDS")
    g.remove_node = boom_remove
    with caplog.at_level(logging.DEBUG, logger="cc_ng_organism"):
        res = cno.generate_emergent_want(g, None)
    assert res is None
    errs = _recs(caplog, "cc_ng_organism", logging.WARNING)
    assert len(errs) == 1 and errs[0].levelno == logging.ERROR
    assert "rollback_failed" in errs[0].getMessage() and "rolled_back=False" in errs[0].getMessage()
    assert "SECRET" not in errs[0].getMessage()


def test_a_the_success_return_dict_is_byte_identical_to_the_base():
    g = _seeded_graph(("s1",))
    _predict(g, ("s1", "t1"))
    text = _want_text(("s1", "t1"))
    assert cno.generate_emergent_want(g, None) == {
        "id": _want_id(text), "text": text, "provenance": "cc_emergent", "state": "open"}
    assert g.nodes[_want_id(text)].metadata == {
        "kind": "want", "want_text": text, "want_state": "open", "provenance": "cc_emergent",
        "creation_mode": "emergent", "concept_key": None}                  # the node's metadata is unchanged too


def test_a_idempotent_branch_unchanged_second_call_returns_none_and_adds_no_synapse():
    g = _seeded_graph(("s1",))
    _predict(g, ("s1", "t1"))
    first = cno.generate_emergent_want(g, None)
    n_syn = len(g.synapses)
    assert first is not None and n_syn == 1
    assert cno.generate_emergent_want(g, None) is None                      # want_id in graph.nodes -> None, as before
    assert len(g.synapses) == n_syn


def _resolving_graph():
    """A graph whose curiosity RESOLVES a concept label (the KISS-dedup / reinforced branch)."""
    g = _seeded_graph(("s1",))
    g.nodes["s1"].metadata["label"] = "alpha"
    vdb = SimpleVectorDB()
    vdb.insert(id="s1", embedding=np.eye(DIM, dtype=np.float32)[0], content="c", metadata={})
    g.prime_and_propagate = lambda **kw: SimpleNamespace(fired_entries=[SimpleNamespace(node_id="s1")])
    _predict(g, ("s1", "t1"))
    return g, vdb, "cc:want::" + hashlib.sha1(b"tonic-concept::alpha").hexdigest()[:16]


def test_a_reinforced_branch_unchanged_and_writes_no_new_synapse():
    g, vdb, wid = _resolving_graph()
    first = cno.generate_emergent_want(g, vdb)
    assert first["id"] == wid and "reinforced" not in first and _degree(g, wid) == 1   # created BOUND
    n_syn = len(g.synapses)
    second = cno.generate_emergent_want(g, vdb)
    assert second["id"] == wid and second["reinforced"] is True
    assert g.nodes[wid].metadata["kiss_reinforcement_count"] == 1
    assert len(g.synapses) == n_syn


def test_a_reinforcing_an_old_degree_zero_emergent_want_does_not_bind_it_REPORTED_unchanged_behaviour():
    """REPORT (not decided here): a want minted by the OLD code has degree 0. The reinforced branch only bumps metadata, as
    before, so it stays unbound and the (unprotected) guard keeps counting it. Pinned so the behaviour is a visible fact."""
    g, vdb, wid = _resolving_graph()
    _node(g, wid, kind="want", want_state="open", provenance="cc_emergent", creation_mode="emergent",
          concept_key="tonic-concept::alpha")
    res = cno.generate_emergent_want(g, vdb)
    assert res["reinforced"] is True
    assert _degree(g, wid) == 0 and wid in tmg._unbound_nodes(g, set(g.nodes))


# ============================================================== (B) unbound AND sweep-eligible

def _mixed_graph(t=OLD):
    g = Graph()
    g.timestep = t
    _node(g, "constitutional-core", constitutional=True)
    _node(g, "cc:want::authored", kind="want", provenance="cc_authored")
    _node(g, "syl:want::authored", kind="want", provenance="syl_authored")
    _node(g, "cc:want::emergent", kind="want", provenance="cc_emergent")
    _node(g, "plain")
    _node(g, "bound-a")
    _node(g, "bound-b")
    g.create_synapse("bound-a", "bound-b", weight=0.4)
    return g


def test_b_a_protected_unbound_node_does_not_block_an_unprotected_one_does():
    g = _mixed_graph()
    assert tmg._unbound_nodes(g, set(g.nodes)) == {"cc:want::emergent", "plain"}
    # the protected ones are unbound structurally -- only the protection leg spares them
    for nid in ("constitutional-core", "cc:want::authored", "syl:want::authored"):
        assert not g._outgoing.get(nid) and not g._incoming.get(nid) and not g._node_hyperedges.get(nid)
        assert g._is_identity_protected(nid) and nid not in tmg._unbound_nodes(g, {nid})
    assert tmg._unbound_nodes(g, {"cc:want::emergent"}) == {"cc:want::emergent"}


def test_b_the_age_term_is_excluded_a_fresh_unprotected_unbound_node_still_counts():
    g = Graph()
    g.timestep = OLD
    _node(g, "fresh")                                                       # age 0 at the moment the guard decides
    assert g.timestep - g.nodes["fresh"].creation_time == 0
    assert tmg._unbound_nodes(g, set(g.nodes)) == {"fresh"}                 # the sweep would NOT reap it yet; the guard must hold
    _node(g, "old").creation_time = 0
    assert tmg._unbound_nodes(g, set(g.nodes)) == {"fresh", "old"}


def test_b_a_node_with_a_hyperedge_but_no_synapse_is_bound():
    g = Graph()
    for n in ("h1", "h2", "h3"):
        _node(g, n)
    g.create_hyperedge({"h1", "h2", "h3"})
    assert tmg._unbound_nodes(g, set(g.nodes)) == set()


def test_b_a_double_without_is_identity_protected_is_counted_not_exempted():
    dbl = SimpleNamespace(_outgoing={"a": set(), "b": {"s"}}, _incoming={"a": set(), "b": set()}, _node_hyperedges={})
    assert not hasattr(dbl, "_is_identity_protected")
    assert tmg._unbound_nodes(dbl, {"a", "b"}) == {"a"}


def test_b_the_signature_is_unchanged_graph_node_ids():
    sig = inspect.signature(tmg._unbound_nodes)
    assert list(sig.parameters) == ["graph", "node_ids"]
    assert all(p.default is inspect.Parameter.empty for p in sig.parameters.values())


def test_b_the_protection_leg_is_the_graphs_own_test_not_a_copy():
    """LAW 3: the predicate CALLS graph._is_identity_protected. A graph whose own test says 'protected' is obeyed."""
    g = Graph()
    _node(g, "plain")
    assert tmg._unbound_nodes(g, {"plain"}) == {"plain"}
    g._is_identity_protected = lambda nid: nid == "plain"
    assert tmg._unbound_nodes(g, {"plain"}) == set()


def _receiver_with(meta, nid="the-unbound"):
    rg, rv = Graph(), SimpleVectorDB()
    rg.create_node(node_id=nid, metadata=dict(meta))                        # creation_time 0 ...
    rg.timestep = OLD                                                       # ... THEN age the clock: 100 > orphan grace 25
    assert rg.timestep - rg.nodes[nid].creation_time > rg.config["orphan_node_grace_period"]   # else 'survives' is vacuous
    return rg, rv, nid


def _all_bound_conduit(tmp_path):
    sg, sv = Graph(), SimpleVectorDB()
    _pair(sg, sv, "A", 0)
    return _export(sg, sv, tmp_path)


@pytest.mark.parametrize("meta", [
    {"constitutional": True},
    {"kind": "want", "provenance": "cc_authored"},
    {"kind": "want", "provenance": "syl_authored"},
], ids=["constitutional", "cc_authored", "syl_authored"])
def test_b_merge_a_protected_unbound_preexisting_node_does_not_block_the_steps_and_survives_the_real_sweep(
        tmp_path, caplog, meta):
    path = _all_bound_conduit(tmp_path)
    rg, rv, nid = _receiver_with(meta)
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, path, tmp_path)
    assert st["consolidation_passes"] == 1 and st["consolidation_steps"] == IDLE      # the steps RAN
    assert st["consolidation_blocked_batches"] == 0 and st["consolidation_skipped_unbound_preexisting"] == 0
    assert st["consolidation_held_midpass"] == 0
    assert rg.timestep == OLD + IDLE
    assert nid in rg.nodes                                                  # the real sweep spared it (it ran: 105 - 0 > 25)
    assert _blocked_records(caplog) == []


@pytest.mark.parametrize("meta", [
    {"kind": "want", "provenance": "cc_emergent"},
    {"creation_mode": "conversational"},
    {},
], ids=["cc_emergent_want", "conversational", "plain"])
def test_b_merge_an_unprotected_unbound_preexisting_node_blocks_the_steps_and_survives_with_one_error(
        tmp_path, caplog, meta):
    path = _all_bound_conduit(tmp_path)
    rg, rv, nid = _receiver_with(meta)
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, path, tmp_path)
    assert st["consolidation_passes"] == 0 and rg.timestep == OLD           # the clock did not move
    assert nid in rg.nodes
    assert st["consolidation_blocked_batches"] == 1 and st["consolidation_skipped_unbound_preexisting"] == 1
    assert len(_blocked_records(caplog)) == 1


def test_b_merge_a_FRESH_unprotected_unbound_arrival_blocks_no_age_term(tmp_path, caplog):
    """The age term would let the clock run on a node that is age 0 when the guard decides: an arrival that cannot bind."""
    sg, sv = Graph(), SimpleVectorDB()
    _pair(sg, sv, "A", 0)
    meta = _meta()
    sg.create_node(node_id="cc:conv::lone", metadata=meta).creation_time = -1
    sv.insert(id="cc:conv::lone", embedding=_emb(), content="c", metadata=meta)
    path = _export(sg, sv, tmp_path)
    rg, rv = Graph(), SimpleVectorDB()                                      # timestep 0: every arrival is age 0
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, path, tmp_path)
    assert rg.timestep == 0 and st["consolidation_passes"] == 0
    assert st["consolidation_skipped_unbound_arrivals"] == 1 and len(_blocked_records(caplog)) == 1


def test_b_the_unbound_unprotected_node_survives_the_real_sweep_because_the_clock_was_held(tmp_path):
    path = _all_bound_conduit(tmp_path)
    rg, rv, nid = _receiver_with({"kind": "want", "provenance": "cc_emergent"})
    _merge(rg, rv, path, tmp_path)
    assert nid in rg.nodes
    # control: with the clock run anyway (no guard), the same node is reaped -- the survival above is not vacuous
    assert cno._cc_callosum_consolidate(rg, IDLE) is True
    assert nid not in rg.nodes


# ============================================================== (C) redaction

_FOREST = "cc:conv::" + "ab12" * 10


@pytest.mark.parametrize("nid,kind", [
    (_FOREST, "forest"),
    (_FOREST + "::tree::", "tree"),
    ("cc:want::0123456789abcdef", "want"),
    ("want::anything", "want"),
    ("cc:conv::" + "ab12" * 10 + "::window::3", "window"),
    ("cc:conv::not-a-sha", "node"),
    ("cc:conv::" + "ab12" * 10 + "x", "node"),
    ("cc:conv::" + "AB12" * 10, "node"),
    ("old-orphan", "node"),
    ("", "node"),
])
def test_c_redact_node_id_shapes(nid, kind):
    out = tmg.redact_node_id(nid)
    assert out == "%s:%s" % (kind, hashlib.sha256(nid.encode("utf-8")).hexdigest()[:12])
    assert re.fullmatch(r"(tree|window|want|forest|node):[0-9a-f]{12}", out)


def test_c_a_tree_id_never_yields_any_of_the_users_words():
    words = ["my", "secret", "grandmother's", "recipe", "SECRET", "Password123"]
    nid = "cc:conv::%s::tree::%s" % ("ab12" * 10, " ".join(words))
    out = tmg.redact_node_id(nid)
    assert out.startswith("tree:") and len(out) == len("tree:") + 12
    for w in words:
        assert w not in out and w.lower() not in out.lower()
    assert nid not in out and "cc:conv" not in out and "::" not in out
    assert out[5:] not in nid                                               # the 12 hex is a hash, not a substring of the id


def test_c_deterministic_and_collision_free_on_distinct_ids():
    a, b = _FOREST + "::tree::one", _FOREST + "::tree::two"
    assert tmg.redact_node_id(a) == tmg.redact_node_id(a)
    assert tmg.redact_node_id(a) != tmg.redact_node_id(b)


def test_c_the_kind_comes_only_from_structural_markers_never_from_free_text():
    adversarial = ["SECRET words here", "evil:payload", "tree", "forest:abc", "secret::x", "ünïcode wörds", "a" * 500,
                   "cc:conv::SECRET", "kind=tree", "lone\ud800surrogate"]
    for nid in adversarial:
        out = tmg.redact_node_id(nid)
        assert re.fullmatch(r"(tree|window|want|forest|node):[0-9a-f]{12}", out), (nid, out)
        assert out.startswith("node:")


def test_c_the_helper_does_not_raise_on_a_non_string_id():
    assert tmg.redact_node_id(12345).startswith("node:")


def test_c_the_897_error_for_an_unbound_TREE_node_prints_only_redacted_forms(tmp_path, caplog):
    path = _all_bound_conduit(tmp_path)
    words = "my secret grandmother recipe zebra"
    tree_id = "cc:conv::%s::tree::%s" % ("cd34" * 10, words)
    forest_id = "cc:conv::" + "ef56" * 10
    rg, rv = Graph(), SimpleVectorDB()
    for nid in (tree_id, forest_id, "cc:want::feedfacefeedface"):
        rg.create_node(node_id=nid, metadata=_meta("tree"))
    rg.timestep = OLD
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, path, tmp_path)
    (msg,) = _blocked_records(caplog)
    for forbidden in ("secret", "grandmother", "recipe", "zebra", tree_id, forest_id, "cc:conv::cd34", "feedface"):
        assert forbidden not in msg, forbidden
    sample = re.search(r"Sample: \[(.*?)\]\.", msg)
    assert sample, msg
    items = [x.strip().strip("'") for x in sample.group(1).split(",")]
    assert len(items) == 3
    assert all(re.fullmatch(r"(tree|window|want|forest|node):[0-9a-f]{12}", i) for i in items), items
    assert "tree:" + hashlib.sha256(tree_id.encode()).hexdigest()[:12] in items
    # the counts / split / batch are still named
    assert "after batch 1" in msg and "3 node(s) in the graph are still unbound" in msg
    assert "3 pre-existing (not landed by this merge), 0 from this merge" in msg
    assert st["consolidation_skipped_unbound_preexisting"] == 3


# ============================================================== (D) the per-slice guard

def _bound_graph(n_pairs=2, t=OLD):
    g, v = Graph(), SimpleVectorDB()
    for i in range(n_pairs):
        _pair(g, v, "p%d" % i, 0)
    g.timestep = t
    return g


def _hook_node(g):
    return g.create_node(node_id="cc:conv::hook-arrival-SECRETWORDS", metadata=_meta())


@pytest.fixture
def slice25(monkeypatch):
    monkeypatch.setenv("CC_CALLOSUM_LOCK_SLICE_STEPS", "25")


@pytest.mark.parametrize("which", ["no_kwargs", "guard_none", "guard_empty", "whole_graph_guard"])
def test_d_an_all_bound_pass_runs_exactly_idle_steps_and_is_identical_to_the_positional_call(slice25, which):
    base_g = _bound_graph()
    assert cno._cc_callosum_consolidate(base_g, 60) is True                 # the positional call, exactly as before
    g, prog = _bound_graph(), {}
    kw = {"no_kwargs": {}, "guard_none": {"guard": None, "progress": prog},
          "guard_empty": {"guard": lambda: set(), "progress": prog},
          "whole_graph_guard": {"guard": tmg.whole_graph_guard(g), "progress": prog}}[which]
    assert cno._cc_callosum_consolidate(g, 60, **kw) is True
    assert g.timestep == base_g.timestep == OLD + 60
    assert set(g.nodes) == set(base_g.nodes) and len(g.synapses) == len(base_g.synapses)
    if which != "no_kwargs":
        assert prog == {"done": 60, "remaining": 0, "held": False, "failed": False}


def test_d_the_guard_is_called_before_EACH_slice_including_the_first_and_never_under_a_held_lock(slice25):
    g, calls, depth, owned = _bound_graph(), [], [], []
    spy = _SpyLock()
    g._concurrent_lock = spy
    real = tmg.whole_graph_guard(g)

    def guard():
        calls.append(spy.exits)                                             # slices completed so far
        depth.append(spy.depth)
        owned.append(g._step_lock._is_owned())
        return real()
    assert cno._cc_callosum_consolidate(g, 60, guard=guard, progress={}) is True
    assert calls == [0, 1, 2]                                               # 3 slices (25+25+10): asked before each
    assert depth == [0, 0, 0] and owned == [False, False, False]            # _concurrent_lock and _step_lock NOT held across the ask
    assert spy.exits == 3 and g.timestep == OLD + 60


def test_d_a_node_that_becomes_unbound_between_slices_stops_the_pass_at_the_slice_boundary(slice25, caplog):
    g, hook = _bound_graph(), {}

    def after_exit(n):
        if n == 1:                                                          # a hook deposit lands between slice 1 and 2
            hook["node"] = _hook_node(g)
    g._concurrent_lock = _SpyLock(after_exit)
    prog = {}
    with caplog.at_level(logging.DEBUG, logger="cc_ng_organism"):
        ok = cno._cc_callosum_consolidate(g, 100, guard=tmg.whole_graph_guard(g), progress=prog)
    node = hook["node"]
    assert ok is False
    assert g.timestep == OLD + 25                                           # exactly ONE slice ran
    assert g.timestep - node.creation_time == 0                             # the unbound node's age did not move
    assert node.node_id in g.nodes                                          # and it SURVIVES (the sweep never ran past it)
    assert prog == {"done": 25, "remaining": 75, "held": True, "failed": False}
    errs = _recs(caplog, "cc_ng_organism", logging.ERROR)
    assert len(errs) == 1
    m = errs[0].getMessage()
    assert "unbound_nodes_present" in m and "blocking_nodes=1" in m and "steps_done=25" in m and "steps_remaining=75" in m
    text = " ".join(r.getMessage() + " " + str(r.args) for r in caplog.records)
    assert "SECRETWORDS" not in text and "hook-arrival" not in text and "cc:conv" not in text    # NO ids, any level


def test_d_control_without_the_guard_the_same_between_slice_node_really_is_reaped(slice25):
    """Non-vacuity: guard=None is today's behaviour -- the node added after slice 1 ages 75 steps against grace 25 and dies."""
    g, hook = _bound_graph(), {}

    def after_exit(n):
        if n == 1:
            hook["node"] = _hook_node(g)
    g._concurrent_lock = _SpyLock(after_exit)
    assert cno._cc_callosum_consolidate(g, 100) is True
    assert g.timestep == OLD + 100 and hook["node"].node_id not in g.nodes


def test_d_a_guard_that_is_non_empty_at_the_start_runs_zero_steps(slice25, caplog):
    g, prog, calls = _bound_graph(), {}, []
    with caplog.at_level(logging.DEBUG, logger="cc_ng_organism"):
        ok = cno._cc_callosum_consolidate(g, 100, guard=lambda: calls.append(1) or {"x", "y"}, progress=prog)
    assert ok is False and g.timestep == OLD and calls == [1]
    assert prog == {"done": 0, "remaining": 100, "held": True, "failed": False}
    (rec,) = _recs(caplog, "cc_ng_organism", logging.ERROR)
    assert "blocking_nodes=2" in rec.getMessage() and "steps_done=0" in rec.getMessage()
    assert "'x'" not in rec.getMessage() and "{" not in rec.getMessage()


def test_d_a_raising_guard_fails_closed_class_name_only(slice25, caplog):
    g, prog = _bound_graph(), {}

    def bad():
        raise RuntimeError("SECRET-USER-WORDS")
    with caplog.at_level(logging.DEBUG, logger="cc_ng_organism"):
        ok = cno._cc_callosum_consolidate(g, 100, guard=bad, progress=prog)
    assert ok is False and g.timestep == OLD                                # stopped, NOT run: it did not fail open
    assert prog == {"done": 0, "remaining": 100, "held": True, "failed": True}
    (rec,) = _recs(caplog, "cc_ng_organism", logging.ERROR)
    assert "guard_raised" in rec.getMessage() and "RuntimeError" in rec.getMessage()
    assert "SECRET" not in " ".join(r.getMessage() for r in caplog.records)


def test_d_a_guard_that_raises_after_slice_one_stops_closed_at_that_boundary(slice25):
    g, prog, n = _bound_graph(), {}, [0]

    def flaky():
        n[0] += 1
        if n[0] == 2:
            raise ValueError("boom")
        return set()
    assert cno._cc_callosum_consolidate(g, 100, guard=flaky, progress=prog) is False
    assert g.timestep == OLD + 25
    assert prog == {"done": 25, "remaining": 75, "held": True, "failed": True}


def test_d_a_failing_step_keeps_the_existing_record_and_reports_failed_not_held(slice25):
    g, prog = _bound_graph(), {}

    def boom():
        raise ValueError("step blew up")
    g.step = boom
    assert cno._cc_callosum_consolidate(g, 100, guard=lambda: set(), progress=prog) is False
    assert prog == {"done": 0, "remaining": 100, "held": False, "failed": True}


def test_d_progress_is_filled_on_every_return_path():
    for g, idle in ((_bound_graph(), 0), (None, 50), (_bound_graph(), -3)):
        prog = {}
        assert cno._cc_callosum_consolidate(g, idle, progress=prog) is False
        assert prog == {"done": 0, "remaining": max(0, idle), "held": False, "failed": False}


def test_d_the_signature_is_the_contract_keyword_only_guard_and_progress():
    sig = inspect.signature(cno._cc_callosum_consolidate)
    assert list(sig.parameters) == ["graph", "idle_steps", "guard", "progress"]
    assert sig.parameters["guard"].kind is inspect.Parameter.KEYWORD_ONLY and sig.parameters["guard"].default is None
    assert sig.parameters["progress"].kind is inspect.Parameter.KEYWORD_ONLY and sig.parameters["progress"].default is None


def test_d_whole_graph_guard_is_zero_arg_reads_the_whole_graph_under_the_step_lock_and_skips_protected(monkeypatch):
    g = _mixed_graph()
    asked, owned = [], []
    real = tmg._unbound_nodes
    monkeypatch.setattr(tmg, "_unbound_nodes",
                        lambda gr, ids: asked.append(set(ids)) or owned.append(gr._step_lock._is_owned()) or real(gr, ids))
    guard = tmg.whole_graph_guard(g)
    assert not inspect.signature(guard).parameters
    assert guard() == {"cc:want::emergent", "plain"}                        # whole graph, protected excluded
    assert asked == [set(g.nodes)] and owned == [True]
    assert not g._step_lock._is_owned()                                     # released after the read


def test_d_the_merge_passes_the_real_guard_and_a_progress_dict_to_the_consolidation(tmp_path, monkeypatch):
    path = _all_bound_conduit(tmp_path)
    rg, rv = Graph(), SimpleVectorDB()
    seen = {}
    real = cno._cc_callosum_consolidate

    def spy(g, n, **kw):
        seen["kw"] = dict(kw)
        seen["empty_at_start"] = kw["guard"]()
        _node(g, "late-unbound")
        seen["after_unbound"] = kw["guard"]()
        g.remove_node("late-unbound")
        return real(g, n, **kw)
    monkeypatch.setattr(cno, "_cc_callosum_consolidate", spy)
    st = _merge(rg, rv, path, tmp_path)
    assert set(seen["kw"]) == {"guard", "progress"} and isinstance(seen["kw"]["progress"], dict)
    assert seen["empty_at_start"] == set() and seen["after_unbound"] == {"late-unbound"}
    assert st["consolidation_passes"] == 1


def _two_batch_conduit(tmp_path):
    sg, sv = Graph(), SimpleVectorDB()
    for i in range(2):
        _pair(sg, sv, "p%d" % i, 10 * i)
    path = _export(sg, sv, tmp_path, "src.conduit", batch_size=2)
    assert [f["kind"] for f in tex.read_topology_frames(open(path, "rb").read())] == ["header", "batch", "batch"]
    return path


def test_d_merge_a_pass_held_midway_is_reported_not_counted_the_next_batch_proceeds_and_the_outer_guard_still_blocks(
        tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("CC_CALLOSUM_LOCK_SLICE_STEPS", "2")                 # IDLE=5 -> slices of 2, 2, 1
    path = _two_batch_conduit(tmp_path)
    rg, rv = Graph(), SimpleVectorDB()

    def after_exit(n):
        if n == 1:                                                          # a hook deposit between slice 1 and 2 of batch 1's pass
            rg.create_node(node_id="cc:conv::hook", metadata=_meta())
    rg._concurrent_lock = _SpyLock(after_exit)
    with caplog.at_level(logging.DEBUG):
        st = _merge(rg, rv, path, tmp_path)

    assert st["absorbed_nodes"] == 4 and st["batches_read"] == 2            # the next batch proceeded
    assert st["consolidation_held_midpass"] == 1                            # the held pass is REPORTED
    assert st["consolidation_passes"] == 0 and st["consolidation_steps"] == 0   # ...and NOT counted as consolidated
    assert rg.timestep == 2                                                 # exactly one slice ran
    assert st["consolidation_blocked_batches"] == 1                         # batch 2: the OUTER batch-end guard blocked a pass that never started
    assert st["consolidation_skipped_unbound_preexisting"] == 1 and "cc:conv::hook" in rg.nodes
    held = [r for r in _recs(caplog, "cc_ng_organism", logging.ERROR) if "HELD" in r.getMessage()]
    assert len(held) == 1 and "unbound_nodes_present" in held[0].getMessage()
    assert len(_blocked_records(caplog)) == 1


def test_d_merge_a_failed_pass_is_not_counted_and_not_reported_held(tmp_path, monkeypatch):
    path = _all_bound_conduit(tmp_path)
    rg, rv = Graph(), SimpleVectorDB()
    monkeypatch.setattr(cno, "_cc_callosum_consolidate",
                        lambda g, n, *, guard=None, progress=None: progress.update(done=0, remaining=n, held=False, failed=True) or False)
    st = _merge(rg, rv, path, tmp_path)
    assert st["consolidation_passes"] == 0 and st["consolidation_steps"] == 0 and st["consolidation_held_midpass"] == 0


def test_d_idle_steps_zero_never_builds_or_asks_a_guard(tmp_path, monkeypatch):
    path = _all_bound_conduit(tmp_path)
    rg, rv = Graph(), SimpleVectorDB()
    monkeypatch.setattr(tmg, "whole_graph_guard", lambda g: pytest.fail("guard built with idle_steps == 0"))
    st = _merge(rg, rv, path, tmp_path, idle_steps=0)
    assert st["consolidation_passes"] == 0 and st["consolidation_held_midpass"] == 0 and rg.timestep == 0
