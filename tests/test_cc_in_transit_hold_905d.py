#!/usr/bin/env python3
# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane in-transit-hold-905d, dispatch #14460/#14474) — #905-DELTA tests: the whole-graph hold covers ONLY nodes whose binding is IN TRANSIT
# What: tests for the NEW cc_topology_merge.held_unbound_nodes + the once-read, fail-closed in-transit id set (CC_NG_IN_TRANSIT_IDS_PATH),
#   and for the two NG consumers routed through it (the merge's batch-end check, whole_graph_guard(graph, merge_landed=None)).
#   Cases (a)-(j) of the brief: (a) a laptop-own unbound node is NOT held (both guards; the clock runs; NO claim about its fate); (b) a cohort node and a
#   merge_landed arrival ARE held (and the result is the INTERSECTION, never a union); (c) every corrupt-file class holds EVERYTHING with
#   ONE loud ERROR naming the class, UNSET is byte-identical to _unbound_nodes with no I/O and no log; (d) a laptop-own node that co-fires
#   and sprouts through the engine's OWN dynamics survives; (e) a cohort node that binds drops out with the file unchanged and the clock
#   runs; (f) read-once; (g) the INFO line carries sha256 + count and no id; (h) `_unbound_nodes` is byte-identical (source-hash pin);
#   (i) merge_cc_topology end to end with the VALID set and with UNSET; (j) whole_graph_guard(graph) without merge_landed still works.
# Why: Exec P547/P548 (Josh's ruling; Chief-003 ruling B). The #897/#905 hold counted EVERY sweep-eligible unbound node, so the laptop-own
#   forest:2dfa2d637643 (no binding in transit) froze the clock. The hold now covers only nodes whose binding is IN TRANSIT
#   (CC-CALLOSUM-TRUTH §8.12). EXEC P550 (follow-up, dispatch #14536): this change only narrows what HOLDS THE CLOCK and decides NOTHING about
#   a laptop-own node's fate, so NO test here asserts that such a node is, or is not, culled; its protected window is a separate, pending sweep
#   change (neuro_foundation.py is protected), closing on the AUTONOMIC clock (cc_update_probation / the daemon _autosave_loop).
# How: REAL neuro_foundation.Graph (import only), REAL step(), REAL exporter + merge for (i); scratch files in tmp_path only
#   (the real in-transit-146.jsonl is NEVER read). Z12_MERGE_UNDER_TEST (a path) loads a different cc_topology_merge.py as the module under
#   test so this SAME file runs unchanged against a scratch MUTANT copy of the committed file. P379/#770: the printed preamble names it.
#   Follow-up in the same commit (review): the file is read through ONE O_NONBLOCK fd + fstat (a FIFO swapped in cannot block a caller
#   holding graph._step_lock; the regular-file/size checks cannot be raced), so the no-I/O patches cover os.open/os.fstat/os.read and a
#   FIFO case is added to the corrupt matrix.
# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane in-transit-hold-905d, dispatch #14776) — #976 shape-guard tests
#   (1) 146 scratch ids SHAPED like the real ones (132 tree + 14 forest; NEVER the real file) are VALID; (2) an all-redacted-shape file is corrupt
#   (class redacted_id_shape, no id/line in caplog, in_transit_ids() None); (3) a mixed file (146 good + ONE redacted id first/middle/last) is corrupt;
#   (4) boundary shapes that must NOT trip it; (5) the empty-intersection case stays quiet (guard 1); (6) 1 id and 500 ids are both VALID (guard 2:
#   no count pinned); (7) reported through in_transit_ids / held_unbound_nodes / whole_graph_guard like bad_id (one read, one ERROR, cached).
# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane in-transit-hold-905d, dispatch #14946) — DL-1 (#987) tests: padded redacted ids trip the guard
#   The two #976 boundary params the pinned old behaviour (a trailing-newline id and a leading-space id 'do NOT trip') are FLIPPED: they are the
#   padded redacted form and now trip `redacted_id_shape`; padded variants (space, \n, \r\n, tab, NBSP; first/middle/last line of 146 good ids) are
#   added; and the NEGATIVE side: raw tree ids with internal spaces/colons and padded concept parts stay VALID and are stored UNSTRIPPED.
# -------------------
"""#905-DELTA: the hold covers only nodes whose binding is in transit."""
import builtins
import hashlib
import importlib.util
import inspect
import io
import json
import logging
import os
import random
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

_WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKTREE))

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

ENV = "CC_NG_IN_TRANSIT_IDS_PATH"
GRACE = 25
IDLE = 5
OLD = 100
DIM = 768

# `_unbound_nodes` at the reviewed #905 tip f685a3a7 (computed from inspect.getsource at that commit).
_UNBOUND_NODES_SRC_SHA256 = "231c2380c15cd0f12583479b017f1d8d6e7737983c38cb0183daac3f937594a1"
_UNBOUND_NODES_SIGNATURE = "(graph: Any, node_ids: Set[str]) -> Set[str]"

# Ids deliberately carry distinctive words, so a leak into a log line is a plain substring check.
OWN = "cc:conv::SECRETWORDS-laptop-own"
OTHER = "cc:conv::SECRETWORDS-laptop-other"
COH_A = "cc:conv::SECRETWORDS-cohort-a::tree::concept-a"
COH_B = "cc:conv::SECRETWORDS-cohort-b::tree::concept-b"
ARR = "cc:conv::SECRETWORDS-arrival"
ORPHAN = "cc:conv::old-orphan"


def test_modules_under_test_resolve_inside_the_worktree():
    if _UNDER_TEST:
        pytest.skip("Z12_MERGE_UNDER_TEST is set: the module under test is deliberately %s" % _UNDER_TEST)
    for mod in (tmg, cno, tex):
        assert _WORKTREE in Path(mod.__file__).resolve().parents, f"{mod.__name__} resolves outside {_WORKTREE}"


# ---------------------------------------------------------------- rig

@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """Every test starts UNSET with an empty process-wide cache; the reset helper is TEST-ONLY."""
    monkeypatch.delenv(ENV, raising=False)
    tmg._reset_in_transit_cache_for_tests()
    yield
    tmg._reset_in_transit_cache_for_tests()


def _rows(ids):
    """The census row shape of in-transit-146.jsonl (census keys), one JSON object per line."""
    return "".join(json.dumps({
        "age": 3, "age_gt_grace": False, "creation_mode": "tree", "has_text": True, "id": i,
        "identity_protected": False, "in_vdb": True, "kind": "tree", "source": "SECRETSOURCE"}) + "\n" for i in ids)


def _cohort_file(tmp_path, ids, name="in-transit.jsonl"):
    p = tmp_path / name
    p.write_text(_rows(ids))
    return p


def _use(monkeypatch, path):
    monkeypatch.setenv(ENV, str(path))
    tmg._reset_in_transit_cache_for_tests()


def _node(g, nid, mode="conversational"):
    return g.create_node(node_id=nid, metadata={"cc": True, "creation_mode": mode})


def _bind(g, a, b):
    g.create_synapse(a, b, weight=0.4, delay=2)


def _tmg_records(caplog, level=None):
    return [r for r in caplog.records if r.name == tmg.logger.name and (level is None or r.levelno == level)]


def _held(g, ids=None, merge_landed=None):
    return tmg.held_unbound_nodes(g, set(g.nodes) if ids is None else ids, merge_landed)


# ---------------------------------------------------------------- (h) _unbound_nodes is byte-identical

def test_h_unbound_nodes_is_byte_identical_and_its_signature_pinned():
    src = inspect.getsource(tmg._unbound_nodes)
    assert hashlib.sha256(src.encode()).hexdigest() == _UNBOUND_NODES_SRC_SHA256
    assert str(inspect.signature(tmg._unbound_nodes)) == _UNBOUND_NODES_SIGNATURE


def test_h_the_new_rule_is_a_separate_function_with_the_contracted_signature():
    assert tmg.held_unbound_nodes is not tmg._unbound_nodes
    assert list(inspect.signature(tmg.held_unbound_nodes).parameters) == ["graph", "node_ids", "merge_landed"]
    assert inspect.signature(tmg.held_unbound_nodes).parameters["merge_landed"].default is None
    sig = inspect.signature(tmg.whole_graph_guard)
    assert list(sig.parameters) == ["graph", "merge_landed"] and sig.parameters["merge_landed"].default is None


def test_h_the_test_only_reset_helper_is_never_called_by_production_code():
    """Production never calls the reset: no other root module names it, and cc_topology_merge.py only DEFINES it."""
    import ast
    offenders = []
    for p in sorted(_WORKTREE.glob("*.py")):
        text = p.read_text(errors="replace")
        if "_reset_in_transit_cache_for_tests" not in text:
            continue
        if p.name != "cc_topology_merge.py":
            offenders.append(p.name)
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "_reset_in_transit_cache_for_tests":
                offenders.append("cc_topology_merge.py:%d (calls it)" % node.lineno)
    assert offenders == []


# ---------------------------------------------------------------- (a) a laptop-own node is NOT held (the hold narrowing ONLY: no claim about its fate)

def test_a_a_laptop_own_unbound_node_does_not_hold_either_guard(tmp_path, monkeypatch):
    g = Graph()
    _node(g, OWN)
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A]))
    assert tmg._unbound_nodes(g, set(g.nodes)) == {OWN}                 # sweep-eligible and unbound: what _unbound_nodes answers
    assert _held(g) == set()                                            # ...but its binding is not in transit: not held (batch-end path)
    assert _held(g, merge_landed={ARR}) == set()
    assert tmg.whole_graph_guard(g)() == set()                          # per-slice guard
    assert tmg.whole_graph_guard(g, merge_landed=set())() == set()


def test_a_the_clock_runs_when_the_only_unbound_node_is_laptop_own(tmp_path, monkeypatch):
    """The hold narrowing ONLY: with a laptop-own unbound node present (and no cohort/arrival node), the per-slice guard is clear, the
    pass runs every step, and nothing reports held. NO assertion about whether that node is or is not culled: not decided by this change."""
    g = Graph()
    _node(g, OWN)
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A]))
    progress = {}
    ran = cno._cc_callosum_consolidate(g, IDLE, guard=tmg.whole_graph_guard(g), progress=progress)
    assert ran is True and progress["held"] is False and progress["failed"] is False
    assert progress["done"] == IDLE and g.timestep == IDLE


def test_a_contrast_with_the_variable_unset_the_same_node_still_freezes_the_clock(monkeypatch):
    """UNSET = today's hold-everything behaviour: the defect this build removes, kept for every other consumer."""
    g = Graph()
    _node(g, OWN)
    progress = {}
    assert cno._cc_callosum_consolidate(g, GRACE + 5, guard=tmg.whole_graph_guard(g), progress=progress) is False
    assert progress["held"] is True and g.timestep == 0


# ---------------------------------------------------------------- (b) cohort / merge_landed hold; intersection, not union

def test_b_a_cohort_node_holds_both_guards_and_the_clock_stays_put(tmp_path, monkeypatch):
    g = Graph()
    _node(g, COH_A)
    _node(g, OWN)
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A, COH_B]))
    assert _held(g) == {COH_A}
    assert _held(g, merge_landed={ARR}) == {COH_A}
    assert tmg.whole_graph_guard(g)() == {COH_A}
    progress = {}
    assert cno._cc_callosum_consolidate(g, 10, guard=tmg.whole_graph_guard(g), progress=progress) is False
    assert progress["held"] is True and g.timestep == 0 and {COH_A, OWN} <= set(g.nodes)


def test_b_a_merge_landed_arrival_holds_even_when_it_is_not_in_the_file(tmp_path, monkeypatch):
    g = Graph()
    _node(g, ARR)
    _node(g, OWN)
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A]))
    assert _held(g, merge_landed={ARR}) == {ARR}
    assert _held(g) == set()                                            # no merge in flight: an arrival is held only INSIDE its merge
    assert tmg.whole_graph_guard(g, merge_landed={ARR})() == {ARR}      # the per-slice guard sees the merge's arrivals too
    assert tmg.whole_graph_guard(g, merge_landed={"cc:conv::ghost"})() == set()


def test_b_cohort_and_arrival_together_are_both_held_and_own_is_not(tmp_path, monkeypatch):
    g = Graph()
    for n in (COH_A, ARR, OWN):
        _node(g, n)
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A]))
    assert _held(g, merge_landed={ARR}) == {COH_A, ARR}
    assert tmg.whole_graph_guard(g, merge_landed={ARR})() == {COH_A, ARR}


def test_b_the_result_is_the_intersection_with_the_unbound_set_never_a_union(tmp_path, monkeypatch):
    g = Graph()
    _node(g, COH_A)                    # cohort, BOUND below
    _node(g, ARR)                      # arrival, BOUND below
    _node(g, OWN)                      # unbound, neither
    _node(g, COH_B)                    # cohort, unbound, but NOT in the node_ids asked about
    _bind(g, COH_A, ARR)
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A, COH_B, "cc:conv::not-in-the-graph"]))
    landed = {ARR, "cc:conv::ghost-arrival"}
    assert tmg._unbound_nodes(g, {COH_A, ARR, OWN}) == {OWN}
    assert _held(g, ids={COH_A, ARR, OWN}, merge_landed=landed) == set()    # bound cohort + bound arrival + own: nothing held
    assert _held(g, ids={OWN}) == set()                                     # COH_B is unbound but not in node_ids: not returned
    assert _held(g, ids={COH_B}) == {COH_B}


# ---------------------------------------------------------------- (c) corrupt / missing / empty -> hold EVERYTHING, one loud ERROR

def _w(tmp_path, content, name="bad.jsonl", mode="w"):
    p = tmp_path / name
    with open(p, mode) as fh:
        fh.write(content)
    return p


_GOOD = _rows([COH_A])

_CORRUPT = [
    # id, class named in the ERROR, builder(tmp_path, monkeypatch) -> value for the env variable
    ("missing", "not_found", lambda tp, mp: str(tp / "nope.jsonl")),
    ("a_directory", "not_regular_file", lambda tp, mp: str(tp)),
    ("path_is_the_empty_string", "not_found", lambda tp, mp: ""),
    ("a_fifo_never_blocks", "not_regular_file", lambda tp, mp: (os.mkfifo(str(tp / "fifo")), str(tp / "fifo"))[1]),
    ("oversize", "oversize", lambda tp, mp: (mp.setattr(tmg, "_IN_TRANSIT_MAX_BYTES", 256),
                                              str(_w(tp, _rows(["cc:conv::pad-%d" % i for i in range(20)]))))[1]),
    ("one_malformed_line", "malformed_line", lambda tp, mp: str(_w(tp, _GOOD + "SECRETWORDS-not-json{\n" + _rows([COH_B])))),
    ("non_object_list", "not_an_object", lambda tp, mp: str(_w(tp, _GOOD + '["SECRETWORDS"]\n'))),
    ("non_object_string", "not_an_object", lambda tp, mp: str(_w(tp, _GOOD + '"SECRETWORDS"\n'))),
    ("non_object_number", "not_an_object", lambda tp, mp: str(_w(tp, _GOOD + "5\n"))),
    ("non_object_null", "not_an_object", lambda tp, mp: str(_w(tp, _GOOD + "null\n"))),
    ("id_missing", "bad_id", lambda tp, mp: str(_w(tp, _GOOD + '{"kind": "tree", "source": "SECRETWORDS"}\n'))),
    ("id_empty", "bad_id", lambda tp, mp: str(_w(tp, _GOOD + '{"id": ""}\n'))),
    ("id_not_a_string", "bad_id", lambda tp, mp: str(_w(tp, _GOOD + '{"id": 7}\n'))),
    ("id_null", "bad_id", lambda tp, mp: str(_w(tp, _GOOD + '{"id": null}\n'))),
    ("zero_ids_blank_lines", "zero_ids", lambda tp, mp: str(_w(tp, "\n   \n\n\t\n"))),
    ("empty_file", "zero_ids", lambda tp, mp: str(_w(tp, ""))),
    ("not_utf8", "not_utf8", lambda tp, mp: str(_w(tp, b"\xff\xfe" + _GOOD.encode(), mode="wb"))),
]


@pytest.mark.parametrize("cls,builder", [(c, b) for _i, c, b in _CORRUPT], ids=[i for i, _c, _b in _CORRUPT])
def test_c_every_corrupt_state_holds_everything_with_one_loud_error_naming_the_class(
        tmp_path, monkeypatch, caplog, cls, builder):
    value = builder(tmp_path, monkeypatch)
    monkeypatch.setenv(ENV, value)
    tmg._reset_in_transit_cache_for_tests()
    g = Graph()
    _node(g, OWN)
    _node(g, COH_A)
    _node(g, COH_B)
    ids = set(g.nodes)
    base = tmg._unbound_nodes(g, ids)
    assert base == {OWN, COH_A, COH_B}

    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        got = tmg.held_unbound_nodes(g, ids)
        got_landed = tmg.held_unbound_nodes(g, ids, merge_landed={ARR})
        got_guard = tmg.whole_graph_guard(g)()
        got_guard2 = tmg.whole_graph_guard(g, merge_landed={ARR})()
        got_again = tmg.held_unbound_nodes(g, ids)

    assert got == got_landed == got_guard == got_guard2 == got_again == base       # NEVER fails open: holds ALL sweep-eligible unbound
    errs = _tmg_records(caplog, logging.ERROR)
    assert len(errs) == 1, [r.getMessage() for r in errs]                           # ONE loud ERROR, at the first read; cached after
    msg = errs[0].getMessage()
    assert "class=%s" % cls in msg and ENV in msg
    assert "SECRETWORDS" not in caplog.text and "SECRETSOURCE" not in caplog.text   # never a line of the file, never an id
    for frag in ("Expecting", "JSONDecodeError", "column", "Errno", "Permission denied", "No such file",
                 "Is a directory", "codec", "Traceback"):
        assert frag not in caplog.text, frag                                        # the class, never str(exc)
    assert [r for r in _tmg_records(caplog, logging.INFO)] == []                    # a corrupt file never logs the INFO "valid" line


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file")
def test_c_an_unreadable_file_holds_everything_with_one_error(tmp_path, monkeypatch, caplog):
    p = _cohort_file(tmp_path, [COH_A])
    os.chmod(p, 0)
    try:
        _use(monkeypatch, p)
        g = Graph()
        _node(g, OWN)
        with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
            assert _held(g) == {OWN} == tmg._unbound_nodes(g, set(g.nodes))
        errs = _tmg_records(caplog, logging.ERROR)
        assert len(errs) == 1 and "class=unreadable" in errs[0].getMessage()
        assert "Permission denied" not in caplog.text and "Errno" not in caplog.text
    finally:
        os.chmod(p, 0o600)


def test_c_a_corrupt_file_is_cached_a_later_fix_is_not_read_until_restart(tmp_path, monkeypatch, caplog):
    p = tmp_path / "later.jsonl"
    _use(monkeypatch, p)                                                # does not exist yet
    g = Graph()
    _node(g, OWN)
    _node(g, COH_A)
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        assert _held(g) == {OWN, COH_A}
        p.write_text(_rows([COH_A]))                                    # the file appears and is valid...
        assert _held(g) == {OWN, COH_A}                                 # ...but the failed state is cached: no re-read
        tmg._reset_in_transit_cache_for_tests()                         # "a restart"
        assert _held(g) == {COH_A}
    assert len(_tmg_records(caplog, logging.ERROR)) == 1


@pytest.mark.parametrize("seed", range(12))
def test_c_unset_is_byte_identical_to_unbound_nodes_with_no_io_and_no_log(seed, monkeypatch, caplog):
    rnd = random.Random(seed)

    class _FakeGraph:
        def __init__(self):
            self.nodes = {}
            self._outgoing, self._incoming, self._node_hyperedges = {}, {}, {}
            self.protected = set()

        def _is_identity_protected(self, nid):
            return nid in self.protected

    g = _FakeGraph()
    names = ["n%02d" % i for i in range(40)]
    for n in names:
        g.nodes[n] = object()
        if rnd.random() < 0.4:
            g._outgoing[n] = {"s%d" % rnd.randrange(9)}
        if rnd.random() < 0.3:
            g._incoming[n] = {"s%d" % rnd.randrange(9)}
        if rnd.random() < 0.3:
            g._node_hyperedges[n] = {"h%d" % rnd.randrange(5)}
        if rnd.random() < 0.15:
            g.protected.add(n)
    ids = set(rnd.sample(names, rnd.randrange(0, 41)))
    landed = set(rnd.sample(names, rnd.randrange(0, 20)))
    base = tmg._unbound_nodes(g, ids)

    def _boom(*a, **k):
        raise AssertionError("UNSET must do NO file I/O")

    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        with monkeypatch.context() as m:
            for target, fn in ((builtins, "open"), (io, "open"), (os, "open"), (os, "stat"), (os, "lstat"),
                               (os, "fstat"), (os, "read")):
                m.setattr(target, fn, _boom)
            got = tmg.held_unbound_nodes(g, ids)
            got_landed = tmg.held_unbound_nodes(g, ids, merge_landed=landed)
            got_set_arg = tmg.held_unbound_nodes(g, ids, set())
    assert got == got_landed == got_set_arg == base and type(got) is type(base)    # EXACTLY _unbound_nodes' answer
    assert _tmg_records(caplog) == []                                              # no new log line of any level


def test_c_unset_whole_graph_guard_is_exactly_the_old_guard(monkeypatch):
    g = Graph()
    for n in (OWN, COH_A):
        _node(g, n)
    assert tmg.whole_graph_guard(g)() == tmg._unbound_nodes(g, set(g.nodes)) == {OWN, COH_A}
    assert tmg.whole_graph_guard(g, merge_landed={ARR})() == {OWN, COH_A}


# ---------------------------------------------------------------- (d) a laptop-own node that co-fires and sprouts SURVIVES (engine's own dynamics)

def test_d_a_laptop_own_node_that_cofires_and_sprouts_survives_through_the_engines_own_dynamics(tmp_path, monkeypatch):
    """No injected synapse, no injected hyperedge. The ONLY stimulus is the input a node receives (voltage); the binding is the
    engine's own _sprout_synapses (co-firing within co_activation_window). The delay it draws uses random.randint, so seed it."""
    random.seed(1234)
    g = Graph()
    own, other = _node(g, OWN), _node(g, OTHER)
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A]))
    assert len(g.synapses) == 0 and len(g.hyperedges) == 0               # nothing is pre-wired
    assert _held(g) == set()                                             # not held: the clock may run

    own.voltage = 1.0                                                    # input to the node -> it fires on the next step
    r1 = g.step()
    assert r1.fired_node_ids == [OWN] and r1.synapses_sprouted == 0
    other.voltage = 1.0                                                  # its co-firing partner fires within the co-activation window
    r2 = g.step()
    assert r2.fired_node_ids == [OTHER] and r2.synapses_sprouted == 1    # the engine itself sprouted the binding
    assert {(s.pre_node_id, s.post_node_id) for s in g.synapses.values()} == {(OTHER, OWN)}
    assert tmg._unbound_nodes(g, set(g.nodes)) == set()                  # both are bound now

    progress = {}
    assert cno._cc_callosum_consolidate(g, 40, guard=tmg.whole_graph_guard(g), progress=progress) is True
    assert g.timestep == 42 > GRACE + 2 and progress["held"] is False
    assert OWN in g.nodes and OTHER in g.nodes                           # bound by the engine's own dynamics: they survive (true regardless of any sweep change)


# ---------------------------------------------------------------- (e) a cohort node that binds drops out while the file is unchanged

def test_e_a_cohort_node_that_binds_drops_out_of_the_held_set_and_the_clock_runs(tmp_path, monkeypatch):
    g = Graph()
    _node(g, COH_A)
    _node(g, OTHER)
    p = _cohort_file(tmp_path, [COH_A])
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    _use(monkeypatch, p)
    assert _held(g) == {COH_A}
    progress = {}
    assert cno._cc_callosum_consolidate(g, 10, guard=tmg.whole_graph_guard(g), progress=progress) is False
    assert progress["held"] is True and g.timestep == 0

    _bind(g, COH_A, OTHER)                                               # Leg 2 delivers the binding
    assert hashlib.sha256(p.read_bytes()).hexdigest() == sha             # the file is static
    assert _held(g) == set()                                             # the INTERSECTION empties the cohort term
    assert tmg.whole_graph_guard(g)() == set()
    assert cno._cc_callosum_consolidate(g, 10, guard=tmg.whole_graph_guard(g)) is True
    assert g.timestep == 10 and COH_A in g.nodes


# ---------------------------------------------------------------- (f) read ONCE

def test_f_a_valid_set_is_read_once_later_changes_and_deletion_are_not_seen(tmp_path, monkeypatch, caplog):
    g = Graph()
    _node(g, COH_A)
    _node(g, COH_B)
    p = _cohort_file(tmp_path, [COH_A])
    _use(monkeypatch, p)
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        assert _held(g) == {COH_A}
        n_logs = len(_tmg_records(caplog))
        p.write_text(_rows([COH_B]))                                    # change...
        assert _held(g) == {COH_A}
        p.unlink()                                                      # ...then delete
        with monkeypatch.context() as m:
            def _boom(*a, **k):
                raise AssertionError("re-read after the first call")
            for target, fn in ((builtins, "open"), (os, "open"), (os, "stat"), (os, "fstat"), (os, "read")):
                m.setattr(target, fn, _boom)
            assert _held(g) == {COH_A}
            assert tmg.whole_graph_guard(g)() == {COH_A}
        assert len(_tmg_records(caplog)) == n_logs                      # no new log of any kind


def test_f_the_first_read_is_made_once_across_threads(tmp_path, monkeypatch, caplog):
    g = Graph()
    _node(g, COH_A)
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A]))
    out, barrier = [], threading.Barrier(8)

    def _go():
        barrier.wait()
        out.append(frozenset(_held(g)))

    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        ts = [threading.Thread(target=_go) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    assert set(out) == {frozenset({COH_A})}
    assert len(_tmg_records(caplog, logging.INFO)) == 1                 # the lock makes the read happen exactly once


# ---------------------------------------------------------------- (g) the INFO line: sha256 + count, NO id

def test_g_the_info_line_carries_the_sha256_and_the_count_and_no_id_anywhere(tmp_path, monkeypatch, caplog):
    ids = [COH_A, COH_B, "cc:conv::SECRETWORDS-" + "ab" * 20, COH_A]    # one duplicate: the count is of DISTINCT ids
    p = _cohort_file(tmp_path, ids)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    _use(monkeypatch, p)
    g = Graph()
    for n in (COH_A, OWN, ARR):
        _node(g, n)
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        assert _held(g, merge_landed={ARR}) == {COH_A, ARR}
        tmg.whole_graph_guard(g, merge_landed={ARR})()
    infos = _tmg_records(caplog, logging.INFO)
    assert len(infos) == 1
    msg = infos[0].getMessage()
    assert "sha256=%s" % sha in msg and "ids=3" in msg
    text = caplog.text
    for i in set(ids) | {OWN, ARR}:
        assert i not in text and i[-12:] not in text                    # never an id, never a substring of one
    assert "SECRETWORDS" not in text and "SECRETSOURCE" not in text
    assert _tmg_records(caplog, logging.ERROR) == [] and _tmg_records(caplog, logging.WARNING) == []


# ---------------------------------------------------------------- (i) merge_cc_topology end to end

_SEED = [0]


def _emb():
    _SEED[0] += 1
    return np.random.default_rng(_SEED[0]).normal(size=DIM).astype(np.float32)


def _meta(mode):
    return {"cc": True, "creation_mode": mode, "_forest_content": "TEXT-MUST-NEVER-BE-LOGGED"}


def _pair(g, v, tag, t0):
    f, t = f"cc:conv::{tag}", f"cc:conv::{tag}::tree::x"
    for i, (nid, mode) in enumerate(((f, "conversational"), (t, "tree"))):
        meta = _meta(mode)
        g.create_node(node_id=nid, metadata=meta).creation_time = t0 + i
        v.insert(id=nid, embedding=_emb(), content="content for " + nid, metadata=meta)
    g.create_synapse(f, t, weight=0.4, delay=2)
    return f, t


def _lone(g, v, nid, t0):
    meta = _meta("conversational")
    g.create_node(node_id=nid, metadata=meta).creation_time = t0
    v.insert(id=nid, embedding=_emb(), content="content for " + nid, metadata=meta)
    return nid


def _export(g, v, tmp_path, name="topo.conduit"):
    out = str(tmp_path / name)
    tex.export_cc_topology(g, v, out, machine_id="vps", embedding_model="test-model")
    return out


def _merge(rg, rv, path, tmp_path, **kw):
    kw.setdefault("local_machine_id", "laptop")
    kw.setdefault("expected_embedding_model", "test-model")
    kw.setdefault("membership_path", str(tmp_path / "membership.txt"))
    kw.setdefault("idle_steps", IDLE)
    return tmg.merge_cc_topology(rg, rv, path, **kw)


def _receiver_with_orphan():
    """The laptop CC graph's shape: a PRE-EXISTING unbound conversational node, older than orphan grace."""
    rg, rv = Graph(), SimpleVectorDB()
    rg.create_node(node_id=ORPHAN, metadata=_meta("conversational"))
    rg.timestep = OLD
    assert rg.timestep - rg.nodes[ORPHAN].creation_time > rg.config["orphan_node_grace_period"]
    return rg, rv


def _blocked(caplog):
    return [r.getMessage() for r in _tmg_records(caplog, logging.ERROR) if "consolidation step" in r.getMessage()]


def _bound_pair_conduit(tmp_path, tag="A", extra_lone=None):
    sg, sv = Graph(), SimpleVectorDB()
    _pair(sg, sv, tag, 0)
    if extra_lone:
        _lone(sg, sv, extra_lone, -1)
    return _export(sg, sv, tmp_path, "conduit-%s.conduit" % tag)


def test_i_valid_a_laptop_own_orphan_does_not_block_the_clock(tmp_path, monkeypatch, caplog):
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A]))                  # ORPHAN is NOT in transit
    rg, rv = _receiver_with_orphan()
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, _bound_pair_conduit(tmp_path), tmp_path)
    assert st["absorbed_nodes"] == 2 and st["absorbed_synapses"] == 1
    assert st["consolidation_blocked_batches"] == 0 and st["consolidation_skipped_unbound_preexisting"] == 0
    assert st["consolidation_passes"] == 1 and st["consolidation_steps"] == IDLE and st["consolidation_held_midpass"] == 0
    assert rg.timestep == OLD + IDLE                                    # the clock ran (NO claim about the orphan's fate)
    assert "cc:conv::A" in rg.nodes and "cc:conv::A::tree::x" in rg.nodes
    assert _blocked(caplog) == []


def test_i_valid_a_cohort_node_still_blocks_exactly_like_the_whole_graph_hold(tmp_path, monkeypatch, caplog):
    _use(monkeypatch, _cohort_file(tmp_path, [ORPHAN]))                 # ORPHAN's binding IS in transit
    rg, rv = _receiver_with_orphan()
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, _bound_pair_conduit(tmp_path), tmp_path)
    assert rg.timestep == OLD and ORPHAN in rg.nodes
    assert st["consolidation_passes"] == 0 and st["consolidation_blocked_batches"] == 1
    assert st["consolidation_skipped_unbound_preexisting"] == 1 and st["consolidation_skipped_unbound_arrivals"] == 0
    msgs = _blocked(caplog)
    assert len(msgs) == 1 and "1 node(s) in the graph are still unbound" in msgs[0]
    assert tmg.redact_node_id(ORPHAN) in msgs[0] and ORPHAN not in msgs[0]


def test_i_valid_an_arrival_that_does_not_bind_is_held_even_though_it_is_not_in_the_file(tmp_path, monkeypatch, caplog):
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A]))
    rg, rv = _receiver_with_orphan()                                    # a laptop-own orphan is also present, and is NOT counted
    path = _bound_pair_conduit(tmp_path, extra_lone="cc:conv::lone-arrival")
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, path, tmp_path)
    assert rg.timestep == OLD and ORPHAN in rg.nodes
    assert st["consolidation_passes"] == 0 and st["consolidation_blocked_batches"] == 1
    assert st["consolidation_skipped_unbound_arrivals"] == 1            # the arrival: held via merge_landed
    assert st["consolidation_skipped_unbound_preexisting"] == 0         # the laptop-own orphan: not held, so not counted
    assert len(_blocked(caplog)) == 1


def test_i_valid_binding_the_held_cohort_node_releases_the_next_merge(tmp_path, monkeypatch, caplog):
    _use(monkeypatch, _cohort_file(tmp_path, [ORPHAN]))
    rg, rv = _receiver_with_orphan()
    _merge(rg, rv, _bound_pair_conduit(tmp_path, "A"), tmp_path)
    assert rg.timestep == OLD and ORPHAN in rg.nodes                    # blocked
    rg.create_node(node_id="cc:conv::partner", metadata=_meta("tree"))
    rg.create_synapse(ORPHAN, "cc:conv::partner", weight=0.2)           # Leg 2 binds it
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, _bound_pair_conduit(tmp_path, "B"), tmp_path)
    assert st["consolidation_passes"] == 1 and rg.timestep == OLD + IDLE
    assert ORPHAN in rg.nodes and _blocked(caplog) == []


def test_i_unset_the_merge_is_unchanged_a_laptop_own_orphan_still_blocks(tmp_path, caplog):
    rg, rv = _receiver_with_orphan()
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        st = _merge(rg, rv, _bound_pair_conduit(tmp_path), tmp_path)
    assert rg.timestep == OLD and ORPHAN in rg.nodes
    assert st["consolidation_passes"] == 0 and st["consolidation_blocked_batches"] == 1
    assert st["consolidation_skipped_unbound_preexisting"] == 1 and st["consolidation_skipped_unbound_arrivals"] == 0
    msgs = _blocked(caplog)
    assert len(msgs) == 1 and "1 pre-existing (not landed by this merge), 0 from this merge" in msgs[0]
    assert all("in-transit" not in r.getMessage().lower() for r in _tmg_records(caplog))   # no in-transit line of any kind
    assert [r.getMessage()[:22] for r in _tmg_records(caplog) if r.levelno != logging.ERROR] == [
        "CC topology merge from"]                                                           # only the merge's own pre-existing summary line


def test_i_the_merge_passes_its_own_merge_landed_to_the_per_slice_guard(tmp_path, monkeypatch):
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A]))
    real, calls = tmg.whole_graph_guard, []

    def _spy(graph, merge_landed=None):
        calls.append(None if merge_landed is None else set(merge_landed))
        return real(graph, merge_landed=merge_landed)

    monkeypatch.setattr(tmg, "whole_graph_guard", _spy)
    rg, rv = Graph(), SimpleVectorDB()
    st = _merge(rg, rv, _bound_pair_conduit(tmp_path), tmp_path)
    assert st["consolidation_passes"] == 1
    assert calls == [{"cc:conv::A", "cc:conv::A::tree::x"}]             # the merge's own arrivals, exactly


# ---------------------------------------------------------------- (j) whole_graph_guard(graph) without merge_landed still works

def test_j_whole_graph_guard_called_the_way_the_daemon_calls_it_still_works(tmp_path, monkeypatch):
    g = Graph()
    for n in (COH_A, OWN):
        _node(g, n)
    guard = tmg.whole_graph_guard(g)                                    # the daemon's existing one-argument call
    assert callable(guard)
    assert guard() == {COH_A, OWN}                                      # UNSET: hold everything
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A]))
    assert tmg.whole_graph_guard(g)() == {COH_A}                        # VALID, no merge_landed: the cohort only
    assert cno._cc_callosum_consolidate(g, 3, guard=tmg.whole_graph_guard(g)) is False


# ---------------------------------------------------------------- ADDENDUM 1: the read-only accessor in_transit_ids()

def test_accessor_valid_returns_the_cached_frozenset(tmp_path, monkeypatch):
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A, COH_B, COH_A]))
    got = tmg.in_transit_ids()
    assert isinstance(got, frozenset) and got == frozenset({COH_A, COH_B})
    assert tmg.in_transit_ids() is got                                   # the same cached object, never rebuilt


def test_accessor_unset_is_none_with_no_file_io_and_no_log(monkeypatch, caplog):
    def _boom(*a, **k):
        raise AssertionError("UNSET must do NO file I/O")
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        with monkeypatch.context() as m:
            for target, fn in ((builtins, "open"), (io, "open"), (os, "open"), (os, "stat"), (os, "lstat"), (os, "fstat"), (os, "read")):
                m.setattr(target, fn, _boom)
            assert tmg.in_transit_ids() is None
            assert tmg.in_transit_ids() is None
    assert _tmg_records(caplog) == []


@pytest.mark.parametrize("content", ["", "SECRETWORDS-not-json{\n", '{"id": ""}\n', '["x"]\n'], ids=["empty", "malformed", "empty_id", "non_object"])
def test_accessor_corrupt_is_none_and_the_first_call_logs_one_error_never_a_line(tmp_path, monkeypatch, caplog, content):
    p = tmp_path / "bad.jsonl"
    p.write_text(content)
    _use(monkeypatch, p)
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        assert tmg.in_transit_ids() is None
        assert tmg.in_transit_ids() is None                              # later calls: no re-read, no log
    assert len(_tmg_records(caplog, logging.ERROR)) == 1 and len(_tmg_records(caplog)) == 1
    assert "SECRETWORDS" not in caplog.text


def test_accessor_missing_file_is_none(tmp_path, monkeypatch):
    _use(monkeypatch, tmp_path / "nope.jsonl")
    assert tmg.in_transit_ids() is None


def test_accessor_and_held_unbound_nodes_share_the_one_cache_a_spy_proves_one_read(tmp_path, monkeypatch, caplog):
    real, reads = tmg._read_in_transit_ids, []

    def _spy(path):
        reads.append(path)
        return real(path)

    monkeypatch.setattr(tmg, "_read_in_transit_ids", _spy)
    g = Graph()
    _node(g, COH_A)
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A]))
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        assert tmg.in_transit_ids() == frozenset({COH_A})                # first reader: the accessor
        assert _held(g) == {COH_A}                                       # second reader: held_unbound_nodes
        assert tmg.whole_graph_guard(g)() == {COH_A}
        assert tmg.in_transit_ids() == frozenset({COH_A})
    assert len(reads) == 1                                               # ONE read for all of them (no second reader, no second parse)
    assert len(_tmg_records(caplog, logging.INFO)) == 1                  # and ONE INFO line


def test_accessor_held_first_then_accessor_is_also_one_read(tmp_path, monkeypatch):
    real, reads = tmg._read_in_transit_ids, []
    monkeypatch.setattr(tmg, "_read_in_transit_ids", lambda path: (reads.append(path), real(path))[1])
    g = Graph()
    _node(g, COH_A)
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A]))
    assert _held(g) == {COH_A}
    assert tmg.in_transit_ids() == frozenset({COH_A})
    assert len(reads) == 1


def test_accessor_has_no_parameters_and_reads_no_graph():
    assert list(inspect.signature(tmg.in_transit_ids).parameters) == []


# ---------------------------------------------------------------- #976: the redacted-id SHAPE guard

def _raw_ids(n_trees=132, n_forests=14, seed=976):
    """Scratch ids SHAPED like the real in-transit ones (never the real file): `cc:conv::<40 hex>::tree::<concept words>` (the concept
    carries spaces, colons and non-ASCII) and `cc:conv::<40 hex>`; seeded random hex."""
    rnd = random.Random(seed)
    hexs = lambda n: "".join(rnd.choice("0123456789abcdef") for _ in range(n))
    words = ["quarterly salary: negotiation", "naïve café plan", "日本語 のメモ", "x y  z", "colon:inside:word", "a/b\\c"]
    trees = ["cc:conv::%s::tree::%s %d" % (hexs(40), rnd.choice(words), i) for i in range(n_trees)]
    forests = ["cc:conv::%s" % hexs(40) for _ in range(n_forests)]
    return trees + forests


REDACTED = ["forest:2dfa2d637643", "tree:0123456789ab", "want:abcdef012345", "window:ffffffffffff", "node:000000000000"]


def test_976_1_the_real_raw_id_shapes_pass_and_hold_as_before(tmp_path, monkeypatch, caplog):
    ids = _raw_ids()
    assert len(ids) == 146 and sum("::tree::" in i for i in ids) == 132
    g = Graph()
    for i in ids[:3]:
        _node(g, i)
    _node(g, OWN)
    _use(monkeypatch, _cohort_file(tmp_path, ids))
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        got = tmg.in_transit_ids()
        held = _held(g)
    assert got == frozenset(ids) and len(got) == 146                      # VALID: the whole cohort of 146
    assert held == set(ids[:3])                                           # holds as before: the cohort's unbound nodes, not OWN
    assert _tmg_records(caplog, logging.ERROR) == [] and len(_tmg_records(caplog, logging.INFO)) == 1


def _corrupt_976_assertions(g, caplog, secret_fragments):
    ids = set(g.nodes)
    base = tmg._unbound_nodes(g, ids)
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        assert tmg.in_transit_ids() is None
        assert tmg.held_unbound_nodes(g, ids) == base                     # hold ALL, fail CLOSED
        assert tmg.held_unbound_nodes(g, ids, merge_landed={ARR}) == base
        assert tmg.whole_graph_guard(g)() == base
        assert tmg.in_transit_ids() is None
    errs = _tmg_records(caplog, logging.ERROR)
    assert len(errs) == 1 and "class=redacted_id_shape" in errs[0].getMessage()      # ONE loud ERROR naming the class
    assert _tmg_records(caplog, logging.INFO) == []
    for frag in secret_fragments:
        assert frag not in caplog.text                                    # never the id, never a line


def test_976_2_an_all_redacted_shape_file_is_corrupt(tmp_path, monkeypatch, caplog):
    g = Graph()
    for i in (OWN, COH_A, "forest:2dfa2d637643"):
        _node(g, i)
    _use(monkeypatch, _cohort_file(tmp_path, REDACTED))
    _corrupt_976_assertions(g, caplog, ["2dfa2d637643", "0123456789ab", "abcdef012345", "ffffffffffff", "000000000000", "SECRETSOURCE"])
    assert tmg._unbound_nodes(g, set(g.nodes)) == {OWN, COH_A, "forest:2dfa2d637643"}


@pytest.mark.parametrize("where", ["first", "middle", "last"])
def test_976_3_one_redacted_id_poisons_a_file_of_146_good_ones(tmp_path, monkeypatch, caplog, where):
    good = _raw_ids()
    ids = {"first": ["forest:2dfa2d637643"] + good, "middle": good[:73] + ["forest:2dfa2d637643"] + good[73:],
           "last": good + ["forest:2dfa2d637643"]}[where]
    g = Graph()
    for i in good[:2] + [OWN]:
        _node(g, i)
    _use(monkeypatch, _cohort_file(tmp_path, ids))
    _corrupt_976_assertions(g, caplog, ["2dfa2d637643", "SECRETSOURCE"])
    assert tmg._unbound_nodes(g, set(g.nodes)) == set(g.nodes)            # every sweep-eligible unbound node held (the whole-graph hold)


@pytest.mark.parametrize("nid", [
    "forest:2dfa2d63764",          # 11 hex
    "forest:2dfa2d6376431",        # 13 hex
    "forest:2DFA2D637643",         # uppercase hex
    "for3st:2dfa2d637643",         # a digit in the kind
    "for_est:2dfa2d637643",        # an underscore in the kind
    "cc:want::2dfa2d637643",       # a raw want id with `::`
    "cc:conv::2dfa2d637643",       # a raw-looking forest prefix with 12 hex
    "forest:2dfa2d637643x",        # a trailing char
    "forest: 2dfa2d637643",        # an INTERNAL space (strip() does not touch it; not the shape)
], ids=["hex11", "hex13", "upper", "digit_kind", "underscore_kind", "want_double_colon", "conv_prefix", "trailing_char", "internal_space"])
def test_976_4_boundary_shapes_do_not_trip_the_guard(tmp_path, monkeypatch, caplog, nid):
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A, nid]))
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        assert tmg.in_transit_ids() == frozenset({COH_A, nid})
    assert _tmg_records(caplog, logging.ERROR) == []


def test_976_5_an_empty_intersection_is_not_corrupt_the_hold_goes_quiet_and_the_clock_runs(tmp_path, monkeypatch, caplog):
    """GUARD 1: every cohort id bound or absent => held empty, NO error, the pass runs every step."""
    ids = _raw_ids(5, 1)
    g = Graph()
    for i in ids[:3]:
        _node(g, i)
    _bind(g, ids[0], ids[1])
    _bind(g, ids[1], ids[2])                                              # 3 present AND bound; the other 3 are absent from the graph
    _use(monkeypatch, _cohort_file(tmp_path, ids))
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        assert _held(g) == set() and tmg.whole_graph_guard(g)() == set()
        progress = {}
        assert cno._cc_callosum_consolidate(g, IDLE, guard=tmg.whole_graph_guard(g), progress=progress) is True
    assert progress["held"] is False and g.timestep == IDLE
    assert _tmg_records(caplog, logging.ERROR) == []                      # quiet
    assert tmg.in_transit_ids() == frozenset(ids)                         # still VALID, the file is static
    g2 = Graph()                                                          # a graph holding NONE of the cohort: intersection empty too
    assert tmg.held_unbound_nodes(g2, set(g2.nodes)) == set() and _tmg_records(caplog, logging.ERROR) == []


@pytest.mark.parametrize("n", [1, 2, 500])
def test_976_6_no_count_is_pinned_one_id_and_five_hundred_are_both_valid(tmp_path, monkeypatch, caplog, n):
    ids = _raw_ids(n, 0)
    _use(monkeypatch, _cohort_file(tmp_path, ids))
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        got = tmg.in_transit_ids()
    assert got == frozenset(ids) and len(got) == n != 146
    assert _tmg_records(caplog, logging.ERROR) == []


def test_976_7_reported_like_bad_id_one_read_one_error_cached(tmp_path, monkeypatch, caplog):
    real, reads = tmg._read_in_transit_ids, []
    monkeypatch.setattr(tmg, "_read_in_transit_ids", lambda path: (reads.append(path), real(path))[1])
    g = Graph()
    _node(g, OWN)
    _use(monkeypatch, _cohort_file(tmp_path, _raw_ids(3, 1) + ["tree:0123456789ab"]))
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        assert tmg.in_transit_ids() is None
        assert _held(g) == {OWN} == tmg._unbound_nodes(g, set(g.nodes))
        assert tmg.whole_graph_guard(g, merge_landed={ARR})() == {OWN}
        (tmp_path / "in-transit.jsonl").write_text(_rows(_raw_ids(3, 1)))   # fixed later: NOT re-read (cached until a restart)
        assert tmg.in_transit_ids() is None
    assert len(reads) == 1 and len(_tmg_records(caplog, logging.ERROR)) == 1 and len(_tmg_records(caplog)) == 1


def test_976_the_constant_is_the_shape_redact_node_id_produces():
    """The guard's shape is exactly redact_node_id's output shape (one rule): every redacted form it can make matches, no raw form does."""
    for raw in ("cc:conv::" + "ab" * 20, "cc:conv::" + "ab" * 20 + "::tree::some words", "cc:want::x", "x::window::y", "plain"):
        red = tmg.redact_node_id(raw)
        assert tmg._IN_TRANSIT_REDACTED_SHAPE.fullmatch(red), red
        assert not tmg._IN_TRANSIT_REDACTED_SHAPE.fullmatch(raw), raw
    assert tmg._IN_TRANSIT_REDACTED_SHAPE.pattern == r"^[a-z]+:[0-9a-f]{12}$"


# ---------------------------------------------------------------- DL-1 (#987): padded redacted ids

_PADDED = [" forest:2dfa2d637643", "forest:2dfa2d637643 ", "forest:2dfa2d637643\n", "forest:2dfa2d637643\r\n", "\tforest:2dfa2d637643",
           "forest:2dfa2d637643\t", "\xa0forest:2dfa2d637643", "forest:2dfa2d637643\xa0", "  \ttree:0123456789ab \r\n"]
_PADDED_IDS = ["lead_space", "trail_space", "trail_newline", "trail_crlf", "lead_tab", "trail_tab", "lead_nbsp", "trail_nbsp", "mixed"]


@pytest.mark.parametrize("nid", _PADDED, ids=_PADDED_IDS)
def test_987_a_padded_redacted_id_trips_the_guard(tmp_path, monkeypatch, caplog, nid):
    g = Graph()
    for i in (OWN, COH_A):
        _node(g, i)
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A, nid]))
    _corrupt_976_assertions(g, caplog, ["2dfa2d637643", "0123456789ab", "SECRETSOURCE"])        # class, hold ALL, no id/line, cached


@pytest.mark.parametrize("where", ["first", "middle", "last"])
def test_987_a_padded_redacted_id_poisons_146_good_ones_at_any_position(tmp_path, monkeypatch, caplog, where):
    good = _raw_ids()
    bad = " forest:2dfa2d637643\r\n"
    ids = {"first": [bad] + good, "middle": good[:73] + [bad] + good[73:], "last": good + [bad]}[where]
    g = Graph()
    for i in good[:2] + [OWN]:
        _node(g, i)
    _use(monkeypatch, _cohort_file(tmp_path, ids))
    _corrupt_976_assertions(g, caplog, ["2dfa2d637643", "SECRETSOURCE"])
    assert tmg._unbound_nodes(g, set(g.nodes)) == set(g.nodes)


def test_987_the_two_formerly_pinned_boundary_ids_now_trip_the_guard(tmp_path, monkeypatch, caplog):
    """The two #976 boundary params le-061 named ('forest:...\\n' and ' forest:...') were pinned as 'does NOT trip': that pinned the bug."""
    for nid in ("forest:2dfa2d637643\n", " forest:2dfa2d637643"):
        tmg._reset_in_transit_cache_for_tests()
        caplog.clear()
        _use(monkeypatch, _cohort_file(tmp_path, [COH_A, nid], name="p-%d.jsonl" % len(nid)))
        with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
            assert tmg.in_transit_ids() is None
        assert any("class=redacted_id_shape" in r.getMessage() for r in _tmg_records(caplog, logging.ERROR))


_RAW_NEGATIVES = [
    "cc:conv::%s::tree::two  words : and colons" % ("ab" * 20),
    "cc:conv::%s::tree:: padded concept " % ("cd" * 20),                        # leading/trailing spaces in the CONCEPT part
    "cc:conv::%s::tree::\tconcept\twith tabs\n" % ("ef" * 20),                  # tabs and a trailing newline INSIDE the id
    " cc:conv::%s" % ("01" * 20),                                               # a padded RAW forest id is still not the redacted shape
    "cc:conv::%s " % ("23" * 20),
]


@pytest.mark.parametrize("nid", _RAW_NEGATIVES, ids=["internal_spaces_colons", "padded_concept", "tabs_newline_in_concept", "lead_pad_forest", "trail_pad_forest"])
def test_987_raw_ids_with_spaces_stay_valid_and_are_stored_unstripped(tmp_path, monkeypatch, caplog, nid):
    _use(monkeypatch, _cohort_file(tmp_path, [COH_A, nid]))
    with caplog.at_level(logging.DEBUG, logger=tmg.logger.name):
        got = tmg.in_transit_ids()
    assert got == frozenset({COH_A, nid})                                     # the cohort holds the ORIGINAL string...
    if nid.strip() != nid:
        assert nid.strip() not in got                                         # ...and NOT its stripped form
    assert _tmg_records(caplog, logging.ERROR) == []
