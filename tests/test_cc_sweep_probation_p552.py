# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 build worker, lane sweep-probation-p552, NG-4 / round 2) — Josh's ruling (Exec P550 / P552; Exec P561; Exec P562
#   Addendum 1); CC-CALLOSUM-TRUTH §8.13
# What: the orphan-sweep tests re-written for the PERMANENT, host-agnostic body: `getattr(self, "_fair_chance_window_open", None)`; spare iff
#   `pred is not None and pred(node)`; a raising `pred` => not spared + ONE WARNING with a count; the body parses NOTHING. (a) STRUCTURAL tests with
#   stub predicates: None registration = today's sweep (identical to the base `_collect_orphan_nodes`, ast-extracted from `git show b5e47686`, over a
#   seeded family); True spares / False sweeps / a truthy non-bool spares; a raising predicate => swept + ONE WARNING with a count and class names only
#   (never the exception text); the predicate is consulted LAST (never for bound, young or identity-protected nodes); a node with ANY metadata is spared
#   iff the stub says so (so the body reads no `probation_*` field); grace unchanged; the OLD attribute name is not consulted (no alias). (b) ONE
#   real-integration group: the REAL `cc_ng_organism.fair_chance_window_open` registered on a real `Graph`, driven by the real `cc_update_probation` and
#   the real `step()`: held clock, stepped pulses closing the window, a node that wires during it, legacy seeding (the P550 case), `ingested` swept, a
#   real-step() run past grace, the heartbeat going stale (ONE WARNING per episode, recovery re-arms), a poisoned advancer (the L3 stall) and an unarmed
#   host. (c) STATIC: ONE function differs from the base (87 functions, 179 module-level non-def statements identical), no `probation` string / attribute
#   / name and no `cc_` identifier and no numeric-parsing construct in the function's EXECUTABLE code, and no leftover old predicate name anywhere.
#   The round-1 value matrix (`5`, `0.5`, `np.float64`, `"5"`, `-1`, `0`, NaN, `inf`, `True`, ...) now lives in tests/test_cc_fair_chance_window_p561.py
#   against the predicate that owns it. Every round-1 test here is either kept (re-keyed to the new attribute, meaning preserved) or moved: see the return.
# Why: Exec P562 Addendum 1: all window logic moves into the HOST predicate and this is the LAST protected edit.
# How: scratch Graphs only; no checkpoint, no NeuroGraphMemory, no embedder. The sweep builds `orphans` by iterating self.nodes (a dict: insertion order), so
#   nothing here depends on PYTHONHASHSEED (the 0..31 sweep is in the return).
# [2026-10-02] (NG-3, round 2) re-keyed only; (NG-2, round 1) the original 65 tests.
# -------------------
import ast
import logging
import os
import random
import subprocess
import sys
import tokenize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest

import cc_ng_organism as org
from neuro_foundation import Graph

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_REV = "b5e476863cc069a29ec482959b4f9465f2ea4ccf"
GRACE = 25        # DEFAULT_CONFIG["orphan_node_grace_period"]
PERIOD = 3        # the integration STEP-window size: set by monkeypatching `_CC_PROBATION_STEP_WINDOW` (its own knob, Chief-003 Addendum 2)
GRAD = 5          # the GRADUATION period `_CC_CONV_PROBATION_PERIOD`, deliberately a DIFFERENT number: the sweep must not care about it
ATTR = "_fair_chance_window_open"
_SECRET = "secret-detail-must-never-be-logged-xyz"


@pytest.fixture(autouse=True)
def host(monkeypatch):
    """The host layer's patchable parts: the heartbeat clock (ONE indirection), the TWO distinct knobs, and a clean unarmed heartbeat."""
    now = [1000.0]
    monkeypatch.setattr(org, "_probation_clock", lambda: now[0])
    monkeypatch.setattr(org, "_CC_PROBATION_STEP_WINDOW", PERIOD)
    monkeypatch.setattr(org, "_CC_CONV_PROBATION_PERIOD", GRAD)
    saved = dict(org._PROBATION_HEARTBEAT)
    org._PROBATION_HEARTBEAT.update(armed=False, max_age_s=None, stamp=None, stale_logged=False)
    yield now
    org._PROBATION_HEARTBEAT.clear()
    org._PROBATION_HEARTBEAT.update(saved)


_base_cache = {}


def _base_collect():
    """Today's sweep: the BASE `_collect_orphan_nodes` ast-extracted from `git show b5e47686:neuro_foundation.py` (never re-derived)."""
    if "ast" not in _base_cache:
        out = subprocess.run(["git", "-C", _REPO, "show", "%s:neuro_foundation.py" % BASE_REV], capture_output=True)
        assert out.returncode == 0, "cannot read the base neuro_foundation.py at %s in %s: this must FAIL, never skip" % (BASE_REV[:8], _REPO)
        tree = ast.parse(out.stdout.decode())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Graph")
        _base_cache["ast"] = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_collect_orphan_nodes")
    ns = {"logger": logging.getLogger("neuro_foundation.p562_base")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[_base_cache["ast"]], type_ignores=[])), "<base _collect_orphan_nodes @ b5e47686>",
                 "exec"), ns)
    return ns["_collect_orphan_nodes"]


def _graph(timestep=10_000, pred=None):
    g = Graph()
    g.timestep = timestep
    if pred is not None:
        setattr(g, ATTR, pred)
    return g


def _node(g, nid, meta=None, age=GRACE + 100):
    n = g.create_node(node_id=nid, metadata=dict(meta) if meta is not None else {})
    n.creation_time = g.timestep - age
    return n


def _snap(g):
    return (sorted(g.nodes), len(g.synapses), len(g.hyperedges))


def _warnings(caplog):
    return [r for r in caplog.records if r.name == "neuro_foundation" and r.levelno == logging.WARNING]


# ===========================================================================
# (a) STRUCTURAL: stub predicates. The body parses NOTHING.
# ===========================================================================

def test_the_graph_accepts_a_plain_attribute_and_has_none_by_default():
    assert not hasattr(Graph(), ATTR)
    assert "__slots__" not in Graph.__dict__
    assert not isinstance(getattr(Graph, ATTR, None), property)
    g = Graph()
    f = lambda n: True      # noqa: E731
    setattr(g, ATTR, f)
    assert getattr(g, ATTR) is f                                        # the function object ITSELF


def test_true_spares_false_sweeps():
    g = _graph(pred=lambda n: n.metadata.get("keep"))
    _node(g, "keep", {"keep": True})
    _node(g, "drop", {"keep": False})
    _node(g, "none", {})
    assert g._collect_orphan_nodes() == 2 and set(g.nodes) == {"keep"}


@pytest.mark.parametrize("truthy", ["yes", 1, 2.5, [0], (0,), {"a": 1}, object(), -1])
def test_a_truthy_non_bool_spares(truthy):
    g = _graph(pred=lambda n: truthy)
    _node(g, "n")
    g._collect_orphan_nodes()
    assert "n" in g.nodes


@pytest.mark.parametrize("falsy", [None, 0, 0.0, "", [], (), {}, False])
def test_a_falsy_return_sweeps(falsy):
    g = _graph(pred=lambda n: falsy)
    _node(g, "n")
    g._collect_orphan_nodes()
    assert "n" not in g.nodes


_ODD = [_SECRET, None, 0, 0.0, -1, -0.5, float("nan"), float("inf"), -float("inf"), True, False, "5", "", [], [3], 3, 0.5, 10 ** 12, 2 ** 70,
        np.float64(5.0), np.int64(5)]


@pytest.mark.parametrize("value", _ODD, ids=[repr(v)[:14] for v in _ODD])
@pytest.mark.parametrize("field", ["probation_remaining", "probation_steps_remaining", "probation_last_timestep"])
def test_a_node_with_ANY_metadata_is_spared_iff_the_stub_says_so_the_body_parses_nothing(field, value):
    meta = {"creation_mode": "conversational", field: value}
    for verdict in (True, False):
        g = _graph(pred=lambda n, v=verdict: v)
        _node(g, "n", meta)
        g._collect_orphan_nodes()
        assert ("n" in g.nodes) is verdict, "the sweep must not look at %s=%r" % (field, value)


@pytest.mark.parametrize("meta", [None, [], "", 0])
def test_falsy_non_dict_metadata_is_decided_by_the_stub_alone(meta):
    for verdict in (True, False):
        g = _graph(pred=lambda n, v=verdict: v)
        n = _node(g, "n", {})
        n.metadata = meta
        g._collect_orphan_nodes()
        assert ("n" in g.nodes) is verdict


def test_truthy_non_dict_metadata_fails_exactly_as_the_base_does_before_the_term():
    """A truthy non-dict metadata already raises inside _is_identity_protected (a pre-existing shape this change never touches): the
    new function raises the SAME way and never reaches the predicate."""
    base = _base_collect()
    calls = []
    gb, gn = _graph(), _graph(pred=lambda n: calls.append(n) or True)
    for g in (gb, gn):
        n = _node(g, "bad", {})
        n.metadata = "not-a-dict"
    with pytest.raises(AttributeError):
        base(gb)
    with pytest.raises(AttributeError):
        gn._collect_orphan_nodes()
    assert calls == []


def test_the_predicate_is_consulted_only_for_unbound_old_unprotected_orphans():
    seen = []
    g = _graph(pred=lambda n: seen.append(n) or True)
    _node(g, "bound_a")
    _node(g, "bound_b")
    g.create_synapse("bound_a", "bound_b", weight=0.2)                  # structurally bound
    _node(g, "hyper_a")
    _node(g, "hyper_b")
    g.create_hyperedge({"hyper_a", "hyper_b"})                          # structurally bound
    _node(g, "young", age=3)                                            # inside grace
    _node(g, "edge", age=GRACE)                                         # age == grace is still inside it
    _node(g, "const", {"constitutional": True})                         # identity-protected
    _node(g, "want", {"provenance": "cc_authored"})                     # identity-protected
    _node(g, "target")                                                  # the ONLY candidate
    g._collect_orphan_nodes()
    assert len(seen) == 1 and seen[0] is g.nodes["target"]
    assert set(g.nodes) == {"bound_a", "bound_b", "hyper_a", "hyper_b", "young", "edge", "const", "want", "target"}


@pytest.mark.parametrize("age,candidate", [(0, False), (GRACE - 1, False), (GRACE, False), (GRACE + 1, True), (400, True)])
def test_grace_is_unchanged_and_the_stub_is_reached_only_past_it(age, candidate):
    seen = []
    g = _graph(pred=lambda n: seen.append(n) or False)
    _node(g, "n", age=age)
    g._collect_orphan_nodes()
    assert bool(seen) is candidate
    assert ("n" not in g.nodes) is candidate                            # a stub that says False: swept exactly when past grace


def test_a_bound_node_and_an_identity_protected_one_are_unaffected_by_any_stub():
    for verdict in (True, False):
        g = _graph(pred=lambda n, v=verdict: v)
        _node(g, "a", {"probation_remaining": 0})
        _node(g, "b", {"probation_remaining": -3})
        g.create_synapse("a", "b", weight=0.2)
        _node(g, "const", {"constitutional": True, "probation_remaining": 0})
        g._collect_orphan_nodes()
        assert {"a", "b", "const"} <= set(g.nodes)


_PROBS_ALL = [0, 1, 3, -1, 2.5, None, "4", float("nan"), True, False, float("inf"), -float("inf"), 0.0]


def _family(seed, with_attr, verdict=None):
    rng = random.Random(seed)
    g = _graph(timestep=500)
    if with_attr:
        setattr(g, ATTR, verdict)
    ids = []
    for i in range(30):
        nid = "n%02d" % i
        ids.append(nid)
        meta = {}
        if rng.random() < 0.3:
            meta["creation_mode"] = rng.choice(["ingested", "conversational", "emergent"])
        if rng.random() < 0.7:
            meta["probation_remaining"] = rng.choice(_PROBS_ALL)
        if rng.random() < 0.7:
            meta["probation_steps_remaining"] = rng.choice(_PROBS_ALL)
        r = rng.random()
        if r < 0.08:
            meta["constitutional"] = True
        elif r < 0.16:
            meta["provenance"] = rng.choice(["syl_authored", "cc_authored", "cc_emergent"])
        _node(g, nid, meta, age=rng.choice([0, 5, GRACE, GRACE + 1, 100, 400]))
    seen = set()
    for _ in range(rng.randint(3, 10)):
        a, b = rng.sample(ids, 2)
        if (a, b) not in seen:
            seen.add((a, b))
            g.create_synapse(a, b, weight=0.2)
    for _ in range(rng.randint(0, 3)):
        g.create_hyperedge(set(rng.sample(ids, 3)))
    return g


def test_no_registration_is_exactly_todays_sweep_over_a_seeded_family():
    base = _base_collect()
    removed = 0
    for seed in range(120):
        gb, gn = _family(seed, False), _family(seed, False)
        assert _snap(gb) == _snap(gn)
        rb, rn = base(gb), gn._collect_orphan_nodes()
        assert rb == rn and _snap(gb) == _snap(gn), "seed %d" % seed
        removed += rb
    assert removed > 100                                                # the family really sweeps (not vacuous)


def test_an_explicit_none_registration_is_exactly_todays_sweep():
    base = _base_collect()
    for seed in range(40):
        gb, gn = _family(seed, False), _family(seed, True, None)
        assert base(gb) == gn._collect_orphan_nodes() and _snap(gb) == _snap(gn)


def test_a_predicate_that_says_False_for_everything_is_exactly_todays_sweep():
    base = _base_collect()
    for seed in range(40):
        gb, gn = _family(seed, False), _family(seed, True, lambda n: False)
        assert base(gb) == gn._collect_orphan_nodes() and _snap(gb) == _snap(gn)


def test_the_exemption_only_ever_spares_and_does_something_when_a_predicate_is_registered():
    base = _base_collect()
    spared_total = 0
    for seed in range(120):
        gb = _family(seed, False)
        gn = _family(seed, True, lambda n: int(n.node_id[1:]) % 3 == 0)         # a deterministic stub that spares a third of the nodes
        base(gb)
        gn._collect_orphan_nodes()
        assert set(gb.nodes) <= set(gn.nodes), "seed %d: the exemption removed a node today's sweep keeps" % seed
        spared_total += len(set(gn.nodes) - set(gb.nodes))
    assert spared_total > 20


def test_the_old_attribute_name_is_not_consulted_there_is_no_alias():
    g = _graph()
    g._probation_advances = lambda n: True                              # the round-1 attribute
    _node(g, "n", {"probation_remaining": 5, "probation_steps_remaining": 5, "probation_last_timestep": 0})
    g._collect_orphan_nodes()
    assert "n" not in g.nodes                                           # swept: only `_fair_chance_window_open` is read
    assert not hasattr(org, "probation_advances")


# --- the raising predicate -------------------------------------------------

def test_a_raising_predicate_node_not_spared_ONE_warning_with_a_count_and_class_names_only(caplog):
    def boom(node):
        raise RuntimeError(_SECRET)

    g = _graph(pred=boom)
    for i in range(3):
        _node(g, "c%d" % i)
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        removed = g._collect_orphan_nodes()
    assert removed == 3 and not g.nodes                                 # fail toward today: all swept
    ws = _warnings(caplog)
    assert len(ws) == 1                                                 # ONE per sweep, not one per node
    msg = ws[0].getMessage()
    assert "3" in msg and "RuntimeError" in msg
    assert _SECRET not in msg and _SECRET not in str(ws[0].args)        # never the exception text


def test_one_warning_per_sweep_not_per_process(caplog):
    g = _graph(pred=lambda n: (_ for _ in ()).throw(ValueError(_SECRET)))
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        _node(g, "a")
        g._collect_orphan_nodes()
        _node(g, "b")
        g._collect_orphan_nodes()
    assert len(_warnings(caplog)) == 2


def test_mixed_raise_and_true_spares_the_true_one_and_counts_only_the_raise(caplog):
    def sometimes(node):
        if node.metadata.get("who") == "bad":
            raise KeyError(_SECRET)
        return True

    g = _graph(pred=sometimes)
    _node(g, "good", {"who": "good"})
    _node(g, "bad", {"who": "bad"})
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        g._collect_orphan_nodes()
    assert "good" in g.nodes and "bad" not in g.nodes
    ws = _warnings(caplog)
    assert len(ws) == 1 and "1 node" in ws[0].getMessage() and "KeyError" in ws[0].getMessage()
    assert _SECRET not in ws[0].getMessage()


def test_no_warning_when_the_predicate_does_not_raise(caplog):
    g = _graph(pred=lambda n: n.metadata.get("keep", False))
    _node(g, "kept", {"keep": True})
    _node(g, "dropped", {})
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        g._collect_orphan_nodes()
    assert _warnings(caplog) == []


def test_a_non_callable_registration_fails_toward_todays_sweep_with_a_warning(caplog):
    g = _graph(pred=7)                                                  # a host bug
    _node(g, "n")
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        g._collect_orphan_nodes()
    assert "n" not in g.nodes
    assert len(_warnings(caplog)) == 1 and "TypeError" in _warnings(caplog)[0].getMessage()


# ===========================================================================
# (b) ONE real-integration group: the REAL predicate, a real Graph, the real advancer and the real step()
# ===========================================================================

def _live(timestep=0):
    """A real Graph with the REAL host predicate registered (what the daemon's init_ng does)."""
    g = Graph()
    g.timestep = timestep
    setattr(g, ATTR, org.fair_chance_window_open)
    return g


def _deposit(g, nid="n", mode="conversational", age=GRACE + 100):
    """The REAL deposit path; the node is then made old (past grace) and left UNBOUND."""
    n = org._cc_deposit_memory_node(g, None, nid, np.ones(8, dtype=np.float32), "text",
                                    {"source": "cc_gateway", "creation_mode": mode}, index_in_recall=False)
    n.creation_time = g.timestep - age
    return n


def _pulse(g, steps=1):
    """One autosave pulse: some real steps (each runs the real sweep) then the real advancer."""
    for _ in range(steps):
        g.step()
    return org.cc_update_probation(g)


def _hold(g, ms=1):
    """The clock is HELD: sweeps run (as they do from the Tonic / `on_message`) but graph.timestep does not advance."""
    for _ in range(ms):
        g._collect_orphan_nodes()


def test_int_held_clock_100_pulses_zero_steps_the_node_survives_every_sweep():
    g = _live(timestep=500)
    _deposit(g)
    for _ in range(100):
        org.cc_update_probation(g)                                      # the autonomic pulse; NO step() anywhere
        _hold(g)
        assert "n" in g.nodes
    assert g.timestep == 500 and g.nodes["n"].metadata["probation_steps_remaining"] == PERIOD


def test_int_exactly_N_stepped_pulses_close_the_window_then_the_next_sweep_takes_the_unwired_node():
    g = _live()
    g.config["orphan_node_grace_period"] = 0                            # sweeps run on every real step(); no grace between us and the window
    _deposit(g, "n")
    for k in range(PERIOD):
        assert "n" in g.nodes
        _pulse(g)
    assert g.nodes["n"].metadata["probation_steps_remaining"] == 0 if "n" in g.nodes else True
    g.step()
    assert "n" not in g.nodes


def test_int_a_node_that_wires_during_its_window_survives_after_it():
    g = _live()
    g.config["orphan_node_grace_period"] = 0
    for nid in ("x", "y", "w"):
        _deposit(g, nid)
    g.create_hyperedge({"x", "y"})                                      # a real membership, mid-window
    for _ in range(PERIOD + 2):
        _pulse(g)
    assert g.nodes["x"].metadata["probation_steps_remaining"] == 0
    assert {"x", "y"} <= set(g.nodes)                                   # bound: survives after the window
    assert "w" not in g.nodes                                           # the unwired twin is swept at expiry


def test_int_a_node_wired_by_the_engines_own_cofiring_survives_after_its_window():
    g = _live()
    g.config["orphan_node_grace_period"] = 0
    for nid in ("a", "b", "w"):
        _deposit(g, nid)
    g.stimulate("a", 5.0)
    g.step()
    g.stimulate("b", 5.0)
    g.step()
    assert g._find_synapse("a", "b") is not None or g._find_synapse("b", "a") is not None, "precondition: real co-firing sprouted a synapse"
    for _ in range(PERIOD + 3):
        _pulse(g)
    assert g.nodes["a"].metadata["probation_steps_remaining"] == 0
    assert {"a", "b"} <= set(g.nodes) and "w" not in g.nodes


def test_int_legacy_node_with_a_closed_old_window_is_seeded_and_survives_the_P550_case_then_is_bounded():
    g = _live(timestep=900)
    n = g.create_node(node_id="forest", metadata={"creation_mode": "conversational", "probation_remaining": 0, "probation_total": PERIOD,
                                                  "graduated": True})                                  # predates this build: no step fields
    n.creation_time = 0                                                 # past grace
    org.cc_update_probation(g)                                          # the first pulse seeds it
    _hold(g)
    assert "forest" in g.nodes and g.nodes["forest"].metadata["probation_steps_remaining"] == PERIOD
    for _ in range(PERIOD):
        _pulse(g)
    g.step()
    assert "forest" not in g.nodes                                      # one-time and BOUNDED: <= N stepped pulses, then today's sweep


def test_int_without_any_pulse_a_legacy_node_is_culled_as_today():
    """The control for the seeding: no advancer pass ever ran, so nothing seeded it, and the real sweep takes it."""
    g = _live(timestep=900)
    n = g.create_node(node_id="forest", metadata={"creation_mode": "conversational", "probation_remaining": 0})
    n.creation_time = 0
    g._collect_orphan_nodes()
    assert "forest" not in g.nodes


def test_int_an_ingested_node_with_an_open_window_is_swept():
    g = _live()
    _deposit(g, "ing", mode="ingested")
    g.nodes["ing"].metadata.update(probation_steps_remaining=PERIOD, probation_last_timestep=0)
    g._collect_orphan_nodes()
    assert "ing" not in g.nodes


def test_int_real_step_calls_past_grace_spare_an_open_window_and_cull_the_keyless_control():
    g = _live(timestep=0)
    _deposit(g, "conv", age=0)
    n = g.create_node(node_id="ctrl", metadata={"creation_mode": "conversational"})     # no window fields: the control
    n.creation_time = 0
    for _ in range(GRACE + 40):
        g.step()                                                        # real sweeps, no advancer pass: the clock moves, the window does not
    assert g.timestep > GRACE + 30
    assert "ctrl" not in g.nodes                                        # the real step() reached the sweep and took the control
    assert "conv" in g.nodes                                            # an open step-window protected it across every sweep


def test_int_arrivals_with_an_embedding_get_a_window_and_the_structural_install_gets_none():
    """Who is covered is unchanged: a node deposited WITH an embedding (cc_topology_merge -> _cc_deposit_memory_node) is stamped; the
    no-embedding structural install (create_node only) has no key and is swept as today."""
    g = _live(timestep=300)
    _deposit(g, "with_embedding")
    n = g.create_node(node_id="no_embedding", metadata={"creation_mode": "conversational"})
    n.creation_time = 0
    g._collect_orphan_nodes()
    assert "with_embedding" in g.nodes and "no_embedding" not in g.nodes


def test_int_heartbeat_stale_culls_with_ONE_warning_per_episode_and_a_completed_pass_recovers(host, caplog):
    g = _live()
    _deposit(g, "a")
    _deposit(g, "b")
    org.probation_heartbeat_arm(300)
    with caplog.at_level(logging.DEBUG):
        _hold(g)
        assert {"a", "b"} <= set(g.nodes)                               # fresh: the exemption is on
        host[0] += 301                                                  # the advancer has not completed a cycle for longer than the limit
        _hold(g, ms=3)
        assert "a" not in g.nodes and "b" not in g.nodes                # stale: the REAL sweep took them (today's sweep)
        hb = [r for r in caplog.records if r.name == "cc_ng_organism" and r.levelno == logging.WARNING]
        assert len(hb) == 1                                             # ONE per stale episode
        assert [r for r in caplog.records if r.name == "neuro_foundation" and r.levelno == logging.WARNING] == []   # the predicate did not raise
        _deposit(g, "c")                                                # a new arrival lands while stale ...
        _hold(g)
        assert "c" not in g.nodes                                       # ... and gets no exemption
        org.cc_update_probation(g)                                      # a completed pass: recovery, the latch re-arms
        _deposit(g, "d")
        _hold(g)
        assert "d" in g.nodes                                           # the exemption is back on
        host[0] += 301
        _hold(g, ms=2)
        assert "d" not in g.nodes
        assert len([r for r in caplog.records if r.name == "cc_ng_organism" and r.levelno == logging.WARNING]) == 2


def test_int_L3_a_poisoned_advancer_never_completes_so_after_the_limit_the_exemption_closes(host):
    """The `str` poison on a BOUND node aborts the real cc_update_probation every pulse: the heartbeat is never stamped."""
    g = _live()
    g.create_node(node_id="p1", metadata={"creation_mode": "conversational", "probation_remaining": "7"})     # the poison
    g.create_node(node_id="p2", metadata={"creation_mode": "conversational"})
    g.create_synapse("p1", "p2", weight=0.2)                            # BOUND, so the sweep never removes the poison
    g.nodes["p1"].creation_time = g.nodes["p2"].creation_time = 0
    _deposit(g, "late")                                                 # deposited AFTER it in graph order: never reached by the aborted pass
    org.probation_heartbeat_arm(300)
    for _ in range(6):                                                  # K=5 cycles of exactly 60 s is age == limit (staleness is a strict `>`):
        host[0] += 60                                                   # the exemption closes on the cycle AFTER K when cycles are exactly 60 s
        with pytest.raises(TypeError):                                  # (real cycles run a little over 60 s); every one aborts here
            org.cc_update_probation(g)
    _hold(g)
    assert "late" not in g.nodes and {"p1", "p2"} <= set(g.nodes)       # the unbound node is swept as today; the bound poison stays


def test_int_a_completing_advancer_keeps_the_exemption_on_across_many_pulses(host):
    g = _live()
    _deposit(g, "n")
    org.probation_heartbeat_arm(300)
    for _ in range(40):
        host[0] += 60                                                   # 40 minutes of 60 s pulses ...
        org.cc_update_probation(g)                                      # ... each one completes
        _hold(g)
        assert "n" in g.nodes                                           # held clock: no step, so the window never closes
    assert g.nodes["n"].metadata["probation_steps_remaining"] == PERIOD


def test_int_an_unarmed_host_ignores_the_heartbeat_entirely(host):
    g = _live()
    _deposit(g, "n")
    host[0] += 10 ** 9
    _hold(g)
    assert "n" in g.nodes                                               # not armed (the VPS host, any host that does not arm): not enforced


# ===========================================================================
# (c) STATIC: one function, nothing parsed, no leftover name
# ===========================================================================

def _read_base():
    out = subprocess.run(["git", "-C", _REPO, "show", "%s:neuro_foundation.py" % BASE_REV], capture_output=True)
    assert out.returncode == 0, "cannot read the base neuro_foundation.py at %s in %s: this must FAIL, never skip" % (BASE_REV[:8], _REPO)
    return ast.parse(out.stdout.decode())


def _funcs(tree):
    out = {}

    def walk(body, prefix):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[prefix + n.name] = ast.dump(n)
                walk(n.body, prefix + n.name + ".")
            elif isinstance(n, ast.ClassDef):
                out[prefix + n.name + "#class"] = ast.dump(ast.ClassDef(name=n.name, bases=n.bases, keywords=n.keywords, body=[], decorator_list=n.decorator_list))
                walk(n.body, prefix + n.name + ".")
    walk(tree.body, "")
    return out


def test_static_exactly_one_function_differs_from_the_base_and_no_module_level_statement_does(capsys):
    new = ast.parse(open(os.path.join(_REPO, "neuro_foundation.py"), encoding="utf-8").read())
    old = _read_base()
    fn_new, fn_old = _funcs(new), _funcs(old)
    assert set(fn_new) == set(fn_old), (set(fn_new) ^ set(fn_old))          # none added, none removed
    differing = sorted(k for k in fn_new if fn_new[k] != fn_old[k])
    assert differing == ["Graph._collect_orphan_nodes"], differing
    top_new = [ast.dump(n) for n in new.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    top_old = [ast.dump(n) for n in old.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert top_new == top_old                                               # no module-level addition, change or import
    n_funcs = len([k for k in fn_new if not k.endswith("#class")])
    print("[ast-proof] functions=%d differing=%s module-level-non-def-statements=%d identical=True" % (n_funcs, differing, len(top_new)))
    assert n_funcs > 50 and len(top_new) > 5          # non-vacuous; the proof is the EQUALITY above (these counts are this test's own method)


def _sweep_fn():
    tree = ast.parse(open(os.path.join(_REPO, "neuro_foundation.py"), encoding="utf-8").read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Graph")
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_collect_orphan_nodes")


def test_static_the_executable_body_is_host_agnostic():
    fn = _sweep_fn()
    body = fn.body[1:]                                                      # [0] is the docstring (it may name the host and the window)
    reads = []
    for stmt in body:
        for c in ast.walk(stmt):
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "getattr":
                reads.append([a.value if isinstance(a, ast.Constant) else None for a in c.args[1:]])
            if isinstance(c, ast.Constant) and isinstance(c.value, str):
                assert "probation" not in c.value.lower() and "heartbeat" not in c.value.lower() and c.value != "ingested", c.value
                assert not c.value.startswith("cc_")
            if isinstance(c, ast.Attribute):
                assert "probation" not in c.attr.lower() and not c.attr.startswith("cc_") and c.attr != "metadata", c.attr
            if isinstance(c, ast.Name):
                assert "probation" not in c.id.lower() and not c.id.startswith("cc_"), c.id
                assert c.id not in ("isinstance", "float", "int", "bool", "math", "inf"), "numeric parsing in the body: %s" % c.id
            assert not isinstance(c, (ast.Import, ast.ImportFrom))
    assert reads == [[ATTR, None]], reads                                   # ONE getattr, of exactly the one attribute, defaulting to None


def test_static_the_docstring_states_the_permanent_shape():
    doc = ast.get_docstring(_sweep_fn())
    for needle in ("HOST-AGNOSTIC", "LAST PROTECTED EDIT", "_fair_chance_window_open", "P562", "P561", "P552", "§8.13", "graph STEPS"):
        assert needle in doc, needle


def _code_tokens(path):
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.NAME, tokenize.STRING):
                yield tok.string


_SKIP_DIRS = {".git", "tests", "data", "Defunct-Historical", "__pycache__", "docs", "handoffs"}


@pytest.mark.parametrize("old", ["probation_advances", "_probation_advances"])
def test_static_no_leftover_old_predicate_name_anywhere_in_the_tree_outside_comments_and_tests(old):
    hits = []
    for root, dirs, files in os.walk(_REPO):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                hits += [(os.path.relpath(path, _REPO), s[:40]) for s in _code_tokens(path) if old in s]
    assert hits == [], hits[:3]
