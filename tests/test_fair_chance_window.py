# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 F1 builder; Exec P571 (b) F1 + F3 / Exec P574)
# What: re-pin the stale-episode tests on the NEW site. `_fair_chance_heartbeat_fresh` is now a PURE query and `_note_fair_chance_stale` (called once per
#   sweep by `_collect_orphan_nodes`) owns the latch + the ONE WARNING per episode: the episode / registering-again tests now drive the SWEEP; a new test pins
#   that the query is silent for every node shape; a new test pins that a raising clock never escapes the sweep; a new static test pins purity of the
#   query and once-per-sweep, sweep-only calling of the note; `_note_fair_chance_stale` joins NEW_METHODS.
# Why: LAW 4 (a function named for its query does what its name says). Same approved behaviour; the pinned site moved.
# How: same helpers and the same miniature host; the old assertions that the QUERY warned are inverted (the query must NOT warn or latch).
# [2026-10-02] Claude Sonnet 5.5 (Z12 build worker, lane sweep-probation-p552, NG-4' / round 2) — Josh's ruling (Exec P550 / P552; Exec P561 P1 + P2; Exec P563;
#   Chief-003 Addenda 3-4); CC-CALLOSUM-TRUTH §8.13
# FRAMING (Josh): the fair-chance window is SHARED MACHINERY being TESTED FIRST on the CC, not CC-specific code: the pioneer implementation of canonical §8.13
#   arrival protection; rollout to other NeuroGraphs (Syl's) is Josh's call, LAW 8 gate per host.
# What: tests of the CANONICAL, HOST-NEUTRAL window in neuro_foundation.Graph (enable_fair_chance_window / fair_chance_stamp / fair_chance_advance /
#   fair_chance_heartbeat_stamp / _in_fair_chance_window / the sweep). REPLACES tests/test_cc_sweep_probation_p552.py (the Addendum 1 "predicate-only body" tests, which pin the
#   SUPERSEDED shape). Nothing here knows the CC organism: a miniature HOST is written in this file (it registers, stamps, advances, stamps the heartbeat). Covers: the
#   registration contract (validation is atomic: nothing is registered unless everything held); the value matrix of the window test; the STEP-keyed rules (held clock, N
#   stepped pulses, many steps count once, timestep regression, legacy seeding (A), odd shapes, an unreadable clock); the completion HEARTBEAT (unarmed, strict `>`, stale
#   for EVERY node, ONE WARNING per episode, recovery re-opens the latch); the sweep (an UNREGISTERED graph sweeps EXACTLY as the base does, over a seeded family; the window is
#   consulted LAST; a raising check is swept + ONE WARNING with a count and no exception text; grace and identity protection unchanged; a real step() / real wiring); and the
#   STATIC proof: exactly which functions differ from the base and nothing else, no module-level statement added, no environment read, no `cc`/`CC` token in executable code.
# Why: Exec P563 re-framed the shape: the window logic is canonical and host-neutral; the host keeps only the switch and the heartbeat stamp.
# How: real Graph and real step(); scratch only; patched clocks are plain callables handed to the registration. Node order is dict insertion order, so nothing depends on
#   PYTHONHASHSEED (the 0..31 sweep is in the return).
# -------------------
import ast
import logging
import os
import random
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest

import neuro_foundation as nf
from neuro_foundation import Graph

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_REV = "b5e476863cc069a29ec482959b4f9465f2ea4ccf"
GRACE = 25                       # DEFAULT_CONFIG["orphan_node_grace_period"]
WINDOW = 3                       # a test window size: passed to the registration (the canonical code has no default and reads no environment)
STEPS, LAST = "fair_chance_steps_remaining", "fair_chance_last_timestep"
_SECRET = "secret-detail-must-never-be-logged-xyz"


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _graph(timestep=10_000, register=True, window=WINDOW, max_age=None, clock=None, excluded=("ingested",)):
    g = Graph()
    g.timestep = timestep
    g.config["orphan_node_grace_period"] = GRACE
    if register:
        g.enable_fair_chance_window(window, heartbeat_max_age_s=max_age, excluded_creation_modes=excluded,
                                    clock=clock if clock is not None else (Clock() if max_age is not None else None))
    return g


def _node(g, nid, meta=None, age=GRACE + 100):
    n = g.create_node(node_id=nid, metadata=dict(meta) if meta is not None else {})
    n.creation_time = g.timestep - age
    return n


def _deposit(g, nid, mode="conversational", age=GRACE + 100):
    """A host's deposit: create the node, then open its window through the canonical helper; left unbound and old."""
    n = g.create_node(node_id=nid, metadata={"creation_mode": mode})
    g.fair_chance_stamp(n)
    n.creation_time = g.timestep - age
    return n


def _pulse(g, steps=1):
    """A host's autonomic pulse: some real steps (each runs the real sweep), then its advancer visits every node and stamps the heartbeat."""
    for _ in range(steps):
        g.step()
    for node in list(g.nodes.values()):
        g.fair_chance_advance(node)
    g.fair_chance_heartbeat_stamp()


def _quiet(g):
    """Keep the real sweep out of the way while a FIELD is under test."""
    g.config["orphan_node_grace_period"] = 2 ** 20


def _warnings(caplog):
    return [r for r in caplog.records if r.name == "neuro_foundation" and r.levelno == logging.WARNING and "fair-chance window" in r.getMessage()]


def _steps(g, nid="n"):
    return g.nodes[nid].metadata.get(STEPS)


_base_cache = {}


def _base_collect():
    """Today's sweep: the BASE `_collect_orphan_nodes` ast-extracted from `git show b5e47686:neuro_foundation.py` (never re-derived)."""
    if "ast" not in _base_cache:
        out = subprocess.run(["git", "-C", _REPO, "show", "%s:neuro_foundation.py" % BASE_REV], capture_output=True)
        assert out.returncode == 0, "cannot read the base neuro_foundation.py at %s in %s: this must FAIL, never skip" % (BASE_REV[:8], _REPO)
        tree = ast.parse(out.stdout.decode())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Graph")
        _base_cache["ast"] = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_collect_orphan_nodes")
    ns = {"logger": logging.getLogger("neuro_foundation.p563_base")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[_base_cache["ast"]], type_ignores=[])), "<base _collect_orphan_nodes @ b5e47686>", "exec"), ns)
    return ns["_collect_orphan_nodes"]


# ===========================================================================
# the REGISTRATION contract: one switch the host sets; validation is atomic
# ===========================================================================

def test_nothing_is_registered_by_default_and_no_config_key_or_slot_is_involved():
    g = Graph()
    assert not hasattr(g, "_fair_chance_cfg")
    assert "__slots__" not in Graph.__dict__
    assert not [k for k in nf.DEFAULT_CONFIG if "fair_chance" in k] and not [k for k in g.config if "fair_chance" in k]


@pytest.mark.parametrize("bad", [0, -1, True, False, 2.5, "3", None, float("nan"), [3]])
def test_a_bad_window_size_raises_and_registers_nothing(bad):
    g = Graph()
    with pytest.raises(ValueError):
        g.enable_fair_chance_window(bad)
    assert not hasattr(g, "_fair_chance_cfg")


@pytest.mark.parametrize("bad", [0, -1, -0.5, float("nan"), float("inf"), -float("inf"), True, False, "5", [5]])
def test_a_bad_heartbeat_age_raises_and_registers_nothing(bad):
    g = Graph()
    with pytest.raises(ValueError):
        g.enable_fair_chance_window(WINDOW, heartbeat_max_age_s=bad, clock=Clock())
    assert not hasattr(g, "_fair_chance_cfg")


@pytest.mark.parametrize("clock", [None, 5, "t", object()])
def test_a_heartbeat_needs_a_callable_clock(clock):
    g = Graph()
    with pytest.raises(ValueError):
        g.enable_fair_chance_window(WINDOW, heartbeat_max_age_s=60.0, clock=clock)
    assert not hasattr(g, "_fair_chance_cfg")


@pytest.mark.parametrize("modes", ["ingested", b"ingested"])
def test_excluded_modes_must_be_a_collection_not_a_string(modes):
    g = Graph()
    with pytest.raises(ValueError):
        g.enable_fair_chance_window(WINDOW, excluded_creation_modes=modes)
    assert not hasattr(g, "_fair_chance_cfg")


def test_a_clock_that_raises_registers_nothing():
    def boom():
        raise RuntimeError(_SECRET)
    g = Graph()
    with pytest.raises(RuntimeError):
        g.enable_fair_chance_window(WINDOW, heartbeat_max_age_s=60.0, clock=boom)
    assert not hasattr(g, "_fair_chance_cfg")                       # validated and stamped FIRST, attached LAST


def test_a_valid_registration_attaches_everything_at_once_and_arms_the_heartbeat_now():
    clock = Clock(500.0)
    g = Graph()
    g.enable_fair_chance_window(WINDOW, heartbeat_max_age_s=90, excluded_creation_modes=["ingested", "x"], clock=clock)
    cfg = g._fair_chance_cfg
    assert cfg["window_steps"] == WINDOW and cfg["max_age_s"] == 90.0 and cfg["excluded"] == ("ingested", "x")
    assert cfg["stamp"] == 500.0 and cfg["stale_logged"] is False
    g.enable_fair_chance_window(WINDOW + 1)                           # a later registration REPLACES the earlier one
    assert g._fair_chance_cfg["window_steps"] == WINDOW + 1 and g._fair_chance_cfg["max_age_s"] is None


def test_a_checkpoint_carries_no_registration():
    g = _graph()
    _deposit(g, "n")
    assert "fair_chance" not in repr(sorted(g._serialize_full().keys()))     # the registration is an instance attribute, never serialized


# ===========================================================================
# the WINDOW TEST: a value matrix (the sweep parses nothing; this does)
# ===========================================================================

def _open(steps, last=0, **extra):
    meta = {"creation_mode": "conversational", STEPS: steps, LAST: last}
    meta.update(extra)
    return meta


def _check(g, meta):
    n = _node(g, "probe", meta)
    try:
        return g._in_fair_chance_window(n)
    finally:
        del g.nodes["probe"]


@pytest.mark.parametrize("steps", [1, 2, 3, 0.5, 2.5, 2 ** 40, 2 ** 70, np.float64(5.0), np.float64(0.25)])
def test_the_window_is_open_for_a_finite_number_greater_than_zero(steps):
    assert _check(_graph(), _open(steps)) is True


@pytest.mark.parametrize("steps", [None, "5", "", "abc", [], [3], {}, -1, -0.5, 0, 0.0, False, True, float("nan"), float("inf"), -float("inf"),
                                   np.int64(5), np.float32(5.0)])
def test_the_window_is_closed_for_every_other_shape_and_never_raises(steps):
    assert _check(_graph(), _open(steps)) is False


def test_the_window_is_closed_when_the_counter_is_absent():
    assert _check(_graph(), {"creation_mode": "conversational", LAST: 0}) is False


@pytest.mark.parametrize("last", [None, "7", [], True, False, float("nan"), float("inf"), -float("inf")])
def test_the_window_is_closed_when_last_is_not_a_finite_number(last):
    """A count > 0 with no usable `last` could NEVER be decremented: it reads as closed, never as forever-open."""
    assert _check(_graph(), _open(3, last=last)) is False
    meta = _open(3)
    del meta[LAST]
    assert _check(_graph(), meta) is False


def test_an_excluded_creation_mode_is_never_in_the_window_even_with_an_open_counter():
    g = _graph()
    assert _check(g, _open(5, creation_mode="ingested")) is False
    assert _check(g, _open(5, creation_mode="conversational")) is True


def test_the_exclusion_is_whatever_the_host_set_not_a_baked_in_value():
    g = _graph(excluded=("tool",))
    assert _check(g, _open(5, creation_mode="tool")) is False
    assert _check(g, _open(5, creation_mode="ingested")) is True       # not excluded by THIS host
    g2 = _graph(excluded=())
    assert _check(g2, _open(5, creation_mode="ingested")) is True


def test_a_creation_mode_that_is_unhashable_or_odd_does_not_raise():
    g = _graph()
    for mode in (["a"], {"a": 1}, None, 7, float("nan")):
        assert _check(g, _open(5, creation_mode=mode)) is True


@pytest.mark.parametrize("meta", [None, [], "", 0, "not-a-dict", ["x"], 7])
def test_none_or_non_dict_metadata_is_closed_without_raising(meta):
    g = _graph()
    n = _node(g, "n", {})
    n.metadata = meta
    assert g._in_fair_chance_window(n) is False


def test_an_unregistered_graph_never_has_an_open_window():
    g = _graph(register=False)
    assert _check(g, _open(5)) is False


# ===========================================================================
# the STEP-keyed rules (the unit is graph STEPS)
# ===========================================================================

def test_held_clock_100_pulses_zero_steps_leave_the_counter_untouched_and_the_node_survives():
    g = _graph(timestep=50)
    _deposit(g, "n")
    assert _steps(g) == WINDOW and g.nodes["n"].metadata[LAST] == 50
    for _ in range(100):
        _pulse(g, steps=0)
        g._collect_orphan_nodes()
        assert "n" in g.nodes
    assert g.timestep == 50 and _steps(g) == WINDOW


def test_exactly_N_stepped_pulses_close_the_window_then_the_next_sweep_takes_the_unwired_node():
    g = _graph()
    g.config["orphan_node_grace_period"] = 0
    _deposit(g, "n")
    for k in range(1, WINDOW + 1):
        assert "n" in g.nodes
        _pulse(g)
        if "n" in g.nodes:
            assert _steps(g) == WINDOW - k
    g.step()
    assert "n" not in g.nodes


def test_many_steps_in_one_pulse_count_once():
    g = _graph()
    _quiet(g)
    _deposit(g, "n")
    _pulse(g, steps=40)
    assert _steps(g) == WINDOW - 1
    _pulse(g, steps=0)                                               # a second pulse with NO new steps: still once
    assert _steps(g) == WINDOW - 1


def test_timestep_regression_resets_last_and_never_decrements_or_goes_negative():
    g = _graph(timestep=40)
    _quiet(g)
    _deposit(g, "n")
    _pulse(g)
    assert _steps(g) == WINDOW - 1 and g.nodes["n"].metadata[LAST] == 41
    g.timestep = 7                                                   # a restore from an older checkpoint
    _pulse(g, steps=0)
    md = g.nodes["n"].metadata
    assert md[STEPS] == WINDOW - 1 and md[LAST] == 7
    _pulse(g)
    assert _steps(g) == WINDOW - 2
    for _ in range(3 * WINDOW):
        _pulse(g)
    assert _steps(g) == 0


def test_legacy_node_is_seeded_once_and_its_window_is_bounded():
    g = _graph(timestep=500)
    _quiet(g)
    g.create_node(node_id="legacy", metadata={"creation_mode": "conversational", "probation_remaining": 0})
    g.fair_chance_advance(g.nodes["legacy"])
    md = g.nodes["legacy"].metadata
    assert md[STEPS] == WINDOW and md[LAST] == 500
    g.fair_chance_advance(g.nodes["legacy"])                         # NOT reseeded on every visit
    assert g.nodes["legacy"].metadata[STEPS] == WINDOW
    for _ in range(WINDOW):
        _pulse(g)
    assert g.nodes["legacy"].metadata[STEPS] == 0 and g._in_fair_chance_window(g.nodes["legacy"]) is False


def test_seeding_scope_is_every_non_excluded_node_that_lacks_the_counter_and_never_an_excluded_one():
    """Design call (A), as written: EVERY node lacking the counter is seeded; an excluded creation_mode never is."""
    g = _graph(timestep=9)
    g.create_node(node_id="a", metadata={"creation_mode": "emergent"})
    g.create_node(node_id="b", metadata={})
    g.create_node(node_id="ing", metadata={"creation_mode": "ingested"})
    for nid in list(g.nodes):
        g.fair_chance_advance(g.nodes[nid])
    for nid in ("a", "b"):
        assert g.nodes[nid].metadata[STEPS] == WINDOW
    assert STEPS not in g.nodes["ing"].metadata and LAST not in g.nodes["ing"].metadata


@pytest.mark.parametrize("steps", [None, "3", "", [], float("nan"), float("inf"), -float("inf"), True, False, -1, 0, 0.0])
@pytest.mark.parametrize("last", [0, None, "x", float("nan"), True])
def test_odd_shapes_of_the_fields_are_left_untouched_and_never_raise(steps, last):
    g = _graph(timestep=20)
    g.create_node(node_id="n", metadata={"creation_mode": "conversational", STEPS: steps, LAST: last})
    g.fair_chance_advance(g.nodes["n"])
    md = g.nodes["n"].metadata
    assert repr(md[STEPS]) == repr(steps) and repr(md[LAST]) == repr(last)
    assert g._in_fair_chance_window(g.nodes["n"]) is False           # today's sweep


@pytest.mark.parametrize("t", [None, "5", float("nan"), float("inf"), True, [1]])
def test_an_unreadable_clock_makes_advance_a_no_op_and_stamp_falls_back_to_zero(t):
    g = _graph()
    n = g.create_node(node_id="n", metadata={"creation_mode": "conversational"})
    g.timestep = t
    g.fair_chance_advance(n)                                         # must not raise, must not seed
    assert STEPS not in n.metadata
    g.fair_chance_stamp(n)                                           # a stamp never raises either
    assert n.metadata[STEPS] == WINDOW and n.metadata[LAST] == 0


def test_stamp_resets_both_counters_like_an_exact_repeat_does():
    g = _graph(timestep=11)
    _quiet(g)
    n = _deposit(g, "n")
    for _ in range(2):
        _pulse(g)
    assert _steps(g) == WINDOW - 2
    g.fair_chance_stamp(n)
    assert n.metadata[STEPS] == WINDOW and n.metadata[LAST] == g.timestep


def test_the_helpers_are_no_ops_on_an_unregistered_graph_and_touch_nothing_else():
    g = _graph(register=False)
    n = g.create_node(node_id="n", metadata={"creation_mode": "conversational", "probation_remaining": 5, "novelty_dampening": 0.3})
    before = dict(n.metadata)
    g.fair_chance_stamp(n)
    g.fair_chance_advance(n)
    g.fair_chance_heartbeat_stamp()
    assert n.metadata == before                                       # not one key added


def test_the_helpers_touch_only_the_two_window_fields():
    g = _graph(timestep=3)
    n = g.create_node(node_id="n", metadata={"creation_mode": "conversational", "probation_remaining": 7, "probation_total": 7, "other": [1, 2]})
    before = dict(n.metadata)
    g.fair_chance_advance(n)
    g.step()
    g.fair_chance_advance(n)
    changed = {k for k in set(n.metadata) | set(before) if n.metadata.get(k) != before.get(k)}
    assert changed == {STEPS, LAST}


# ===========================================================================
# the completion HEARTBEAT
# ===========================================================================

def test_unarmed_means_the_heartbeat_is_not_enforced_however_old():
    clock = Clock()
    g = _graph(clock=clock)
    _deposit(g, "n")
    clock.t += 2 ** 40
    g.fair_chance_heartbeat_stamp()                                  # a no-op, not an error
    assert g._in_fair_chance_window(g.nodes["n"]) is True


def test_fresh_means_the_window_applies_and_staleness_is_a_strict_greater_than():
    clock = Clock(1000.0)
    g = _graph(max_age=300, clock=clock)
    _deposit(g, "n")
    clock.t += 300                                                   # age == max_age: still fresh (strict >)
    assert g._in_fair_chance_window(g.nodes["n"]) is True
    clock.t += 0.001
    assert g._in_fair_chance_window(g.nodes["n"]) is False


def test_stale_closes_the_window_for_EVERY_node_with_one_warning_per_episode_and_a_completed_pass_recovers(caplog):
    # P571 F1 (LAW 4): the window QUERY is pure; the stale latch + the ONE WARNING per episode belong to the sweep (`_note_fair_chance_stale`)
    clock = Clock(1000.0)
    g = _graph(max_age=300, clock=clock)
    for nid in ("a", "b", "c"):
        _deposit(g, nid)
    clock.t += 301
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        for _ in range(5):
            for nid in ("a", "b", "c"):
                assert g._in_fair_chance_window(g.nodes[nid]) is False       # stale closes the exemption for EVERY node
        assert _warnings(caplog) == [] and g._fair_chance_cfg["stale_logged"] is False   # ...and the query alone writes and logs NOTHING
        for rnd in range(5):                                         # the SWEEP owns the episode: 5 sweeps, ONE WARNING in all
            if rnd:
                for nid in ("a", "b", "c"):
                    _deposit(g, "%s%d" % (nid, rnd))
            g._collect_orphan_nodes()
            assert not g.nodes                                       # stale: every old unwired node went, none was spared
        ws = _warnings(caplog)
        assert len(ws) == 1 and g._fair_chance_cfg["stale_logged"] is True   # ONE per stale EPISODE, not per sweep
        msg = ws[0].getMessage()
        assert "301" in msg and "300" in msg and "OFF" in msg
        g.fair_chance_heartbeat_stamp()                              # the host's advancer completes a cycle: recovery
        infos = [r for r in caplog.records if r.name == "neuro_foundation" and r.levelno == logging.INFO and "back on" in r.getMessage()]
        assert len(infos) == 1 and g._fair_chance_cfg["stale_logged"] is False
        for nid in ("a", "b", "c"):
            _deposit(g, nid)
        g._collect_orphan_nodes()
        assert {"a", "b", "c"} <= set(g.nodes)                       # the exemption is back on
        clock.t += 301                                               # a SECOND episode warns again (the latch re-opened)
        g._collect_orphan_nodes()
        assert not g.nodes
        assert len(_warnings(caplog)) == 2


def test_the_window_query_is_pure_for_every_node_shape_and_the_sweep_is_what_warns(caplog):
    clock = Clock()
    g = _graph(max_age=20, clock=clock)
    clock.t += 21
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        assert _check(g, _open(5, creation_mode="ingested")) is False        # excluded
        assert _check(g, {"creation_mode": "conversational"}) is False       # no window, stale heartbeat
        assert _check(g, _open(5)) is False                                  # an open counter, stale heartbeat
        assert _warnings(caplog) == [] and g._fair_chance_cfg["stale_logged"] is False
        _node(g, "x", {"creation_mode": "conversational"})
        g._collect_orphan_nodes()
        assert "x" not in g.nodes and len(_warnings(caplog)) == 1


def test_a_clock_that_raises_never_escapes_the_sweep_and_the_node_is_swept_with_one_raise_warning(caplog):
    state = {"boom": False}

    def clock():
        if state["boom"]:
            raise RuntimeError(_SECRET)
        return 1000.0
    g = _graph(max_age=300, clock=clock)
    _deposit(g, "a")
    state["boom"] = True
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        removed = g._collect_orphan_nodes()                          # the note's clock read must not escape the sweep
    assert removed == 1 and "a" not in g.nodes                       # fail toward today's sweep
    ws = [r for r in caplog.records if r.name == "neuro_foundation" and r.levelno == logging.WARNING]
    assert len(ws) == 1                                              # the per-node check reports it, ONE per sweep
    msg = ws[0].getMessage()
    assert "1 node" in msg and "RuntimeError" in msg and _SECRET not in msg


def test_the_stamp_is_the_hosts_completion_signal_and_a_pass_that_never_happens_leaves_the_old_stamp():
    clock = Clock(1000.0)
    g = _graph(max_age=300, clock=clock)
    t0 = g._fair_chance_cfg["stamp"]
    clock.t += 1000
    assert g._fair_chance_cfg["stamp"] == t0
    assert g._in_fair_chance_window(_node(g, "x", _open(3))) is False
    g.fair_chance_heartbeat_stamp()
    assert g._fair_chance_cfg["stamp"] == clock.t != t0


def test_registering_again_resets_the_stale_latch(caplog):
    clock = Clock()
    g = _graph(max_age=20, clock=clock)
    clock.t += 21
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        _node(g, "x", _open(3))
        g._collect_orphan_nodes()
        assert len(_warnings(caplog)) == 1
        g.enable_fair_chance_window(WINDOW, heartbeat_max_age_s=20, clock=clock)
        clock.t += 21
        _node(g, "y", _open(3))
        g._collect_orphan_nodes()
        assert len(_warnings(caplog)) == 2


# ===========================================================================
# the SWEEP
# ===========================================================================

_PROBS = [0, 1, 3, -1, 2.5, None, "4", float("nan"), True, False, float("inf"), -float("inf"), 0.0]


def _family(seed, register, with_fields=True):
    rng = random.Random(seed)
    g = _graph(timestep=500, register=False)
    if register:
        g.enable_fair_chance_window(WINDOW, excluded_creation_modes=("ingested",))
    ids = []
    for i in range(30):
        nid = "n%02d" % i
        ids.append(nid)
        meta = {}
        if rng.random() < 0.3:
            meta["creation_mode"] = rng.choice(["ingested", "conversational", "emergent"])
        if rng.random() < 0.7:
            meta["probation_remaining"] = rng.choice(_PROBS)
        if with_fields and rng.random() < 0.7:
            meta[STEPS] = rng.choice(_PROBS)
            meta[LAST] = rng.choice([0, 5, "x", None, 499])
        r = rng.random()
        if r < 0.08:
            meta["constitutional"] = True
        elif r < 0.16:
            meta["provenance"] = rng.choice(["syl_authored", "mind_authored", "mind_emergent"])
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


def _snap(g):
    return (sorted(g.nodes), len(g.synapses), len(g.hyperedges))


def test_an_UNREGISTERED_graph_sweeps_EXACTLY_as_the_base_does_over_a_seeded_family_even_when_nodes_carry_window_fields():
    base = _base_collect()
    removed = 0
    for seed in range(150):
        gb, gn = _family(seed, False), _family(seed, False)
        assert _snap(gb) == _snap(gn)
        rb, rn = base(gb), gn._collect_orphan_nodes()
        assert rb == rn and _snap(gb) == _snap(gn), "seed %d" % seed
        removed += rb
    assert removed > 100                                             # the family really sweeps (not vacuous)


def test_the_unregistered_sweep_also_emits_the_same_events_as_the_base():
    base = _base_collect()
    for seed in range(20):
        gb, gn = _family(seed, False), _family(seed, False)
        seen = {"b": [], "n": []}
        gb.register_event_handler("nodes_collected", lambda **kw: seen["b"].append(kw.get("count")))
        gn.register_event_handler("nodes_collected", lambda **kw: seen["n"].append(kw.get("count")))
        base(gb)
        gn._collect_orphan_nodes()
        assert seen["b"] == seen["n"]


def test_a_registered_graph_only_ever_spares_and_does_something():
    base = _base_collect()
    spared_total = 0
    for seed in range(150):
        gb, gn = _family(seed, False), _family(seed, True)
        base(gb)
        gn._collect_orphan_nodes()
        assert set(gb.nodes) <= set(gn.nodes), "seed %d: the window removed a node today's sweep keeps" % seed
        spared_total += len(set(gn.nodes) - set(gb.nodes))
    assert spared_total > 20


def test_the_window_is_consulted_only_for_unbound_old_unprotected_orphans():
    seen = []
    g = _graph()
    g._in_fair_chance_window = lambda node: seen.append(node) or True
    for nid in ("bound_a", "bound_b", "hyper_a", "hyper_b", "young", "edge", "const", "want", "target"):
        _node(g, nid, age=3 if nid == "young" else GRACE if nid == "edge" else GRACE + 100,
              meta={"constitutional": True} if nid == "const" else {"provenance": "x_authored"} if nid == "want" else {})
    g.create_synapse("bound_a", "bound_b", weight=0.2)
    g.create_hyperedge({"hyper_a", "hyper_b"})
    g._collect_orphan_nodes()
    assert len(seen) == 1 and seen[0] is g.nodes["target"]
    assert set(g.nodes) == {"bound_a", "bound_b", "hyper_a", "hyper_b", "young", "edge", "const", "want", "target"}


def test_a_check_that_raises_node_not_spared_ONE_warning_with_a_count_and_class_names_only(caplog):
    def boom(node):
        raise RuntimeError(_SECRET)
    g = _graph()
    g._in_fair_chance_window = boom
    for i in range(3):
        _node(g, "c%d" % i)
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        removed = g._collect_orphan_nodes()
    assert removed == 3 and not g.nodes                              # fail toward today: all swept
    ws = [r for r in caplog.records if r.name == "neuro_foundation" and r.levelno == logging.WARNING]
    assert len(ws) == 1
    msg = ws[0].getMessage()
    assert "3" in msg and "RuntimeError" in msg
    assert _SECRET not in msg and _SECRET not in str(ws[0].args)


def test_one_warning_per_sweep_and_a_mixed_raise_spares_the_good_node(caplog):
    def sometimes(node):
        if node.metadata.get("who") == "bad":
            raise KeyError(_SECRET)
        return True
    g = _graph()
    g._in_fair_chance_window = sometimes
    _node(g, "good", {"who": "good"})
    _node(g, "bad", {"who": "bad"})
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        g._collect_orphan_nodes()
        assert "good" in g.nodes and "bad" not in g.nodes
        _node(g, "bad2", {"who": "bad"})
        g._collect_orphan_nodes()
    ws = [r for r in caplog.records if r.name == "neuro_foundation" and r.levelno == logging.WARNING]
    assert len(ws) == 2 and "1 node" in ws[0].getMessage() and "KeyError" in ws[0].getMessage()
    assert _SECRET not in "".join(r.getMessage() for r in ws)


def test_no_warning_when_the_check_does_not_raise(caplog):
    g = _graph()
    _deposit(g, "kept")
    _node(g, "dropped", {})
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        g._collect_orphan_nodes()
    assert "kept" in g.nodes and "dropped" not in g.nodes
    assert [r for r in caplog.records if r.name == "neuro_foundation" and r.levelno == logging.WARNING] == []


@pytest.mark.parametrize("age,candidate", [(0, False), (GRACE - 1, False), (GRACE, False), (GRACE + 1, True), (400, True)])
def test_grace_is_unchanged_and_the_window_is_reached_only_past_it(age, candidate):
    seen = []
    g = _graph()
    g._in_fair_chance_window = lambda node: seen.append(node) or False
    _node(g, "n", age=age)
    g._collect_orphan_nodes()
    assert bool(seen) is candidate and (("n" not in g.nodes) is candidate)


def test_an_identity_protected_or_bound_node_is_unaffected_by_any_window_state():
    g = _graph()
    _node(g, "a", {STEPS: 0, LAST: 0})
    _node(g, "b", {STEPS: -3})
    g.create_synapse("a", "b", weight=0.2)
    _node(g, "const", {"constitutional": True, STEPS: 0})
    g._collect_orphan_nodes()
    assert {"a", "b", "const"} <= set(g.nodes)


def test_a_window_open_node_survives_real_steps_past_grace_and_the_keyless_control_is_culled():
    g = _graph(timestep=0)
    g.config["orphan_node_grace_period"] = GRACE
    _deposit(g, "conv", age=0)
    n = g.create_node(node_id="ctrl", metadata={"creation_mode": "conversational"})
    n.creation_time = 0
    for _ in range(GRACE + 40):
        g.step()                                                     # real sweeps, the host's advancer never ran: the clock moves, the window does not
    assert g.timestep > GRACE + 30
    assert "ctrl" not in g.nodes and "conv" in g.nodes


def test_a_node_that_wires_during_its_window_survives_after_it_by_a_real_hyperedge():
    g = _graph()
    g.config["orphan_node_grace_period"] = 0
    for nid in ("x", "y", "w"):
        _deposit(g, nid)
    g.create_hyperedge({"x", "y"})
    for _ in range(WINDOW + 2):
        _pulse(g)
    assert g.nodes["x"].metadata[STEPS] == 0 and {"x", "y"} <= set(g.nodes) and "w" not in g.nodes


def test_a_node_wired_by_the_engines_own_cofiring_survives_after_its_window():
    g = _graph()
    g.config["orphan_node_grace_period"] = 0
    for nid in ("a", "b", "w"):
        _deposit(g, nid)
    g.stimulate("a", 5.0)
    g.step()
    g.stimulate("b", 5.0)
    g.step()
    assert g._find_synapse("a", "b") is not None or g._find_synapse("b", "a") is not None, "precondition: real co-firing sprouted a synapse"
    for _ in range(WINDOW + 3):
        _pulse(g)
    assert g.nodes["a"].metadata[STEPS] == 0 and {"a", "b"} <= set(g.nodes) and "w" not in g.nodes


def test_an_excluded_node_with_an_open_counter_is_swept():
    g = _graph()
    n = _node(g, "ing", {"creation_mode": "ingested", STEPS: WINDOW, LAST: 0})
    g._collect_orphan_nodes()
    assert "ing" not in g.nodes


def test_a_stale_heartbeat_makes_the_REAL_sweep_cull_as_today_and_a_completed_pass_reopens_it(caplog):
    clock = Clock()
    g = _graph(max_age=300, clock=clock)
    _deposit(g, "a")
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        g._collect_orphan_nodes()
        assert "a" in g.nodes                                        # fresh: spared
        clock.t += 301
        g._collect_orphan_nodes()
        assert "a" not in g.nodes                                    # stale: the REAL sweep took it
        assert len(_warnings(caplog)) == 1
        _deposit(g, "c")
        g._collect_orphan_nodes()
        assert "c" not in g.nodes                                    # no exemption while stale
        g.fair_chance_heartbeat_stamp()
        _deposit(g, "d")
        g._collect_orphan_nodes()
        assert "d" in g.nodes                                        # the exemption is back on


# ===========================================================================
# STATIC: exactly which functions differ, nothing else, nothing read from the environment, nothing CC-named
# ===========================================================================

NEW_METHODS = ["Graph._in_fair_chance_window", "Graph._fair_chance_heartbeat_fresh", "Graph._is_finite_number", "Graph.enable_fair_chance_window",
               "Graph._note_fair_chance_stale", "Graph.fair_chance_advance", "Graph.fair_chance_heartbeat_stamp", "Graph.fair_chance_stamp"]
CHANGED = ["Graph._collect_orphan_nodes"]


def _read_base():
    out = subprocess.run(["git", "-C", _REPO, "show", "%s:neuro_foundation.py" % BASE_REV], capture_output=True)
    assert out.returncode == 0, "cannot read the base neuro_foundation.py at %s in %s: this must FAIL, never skip" % (BASE_REV[:8], _REPO)
    return ast.parse(out.stdout.decode())


def _funcs(tree):
    out = {}

    def walk(body, prefix):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[prefix + n.name] = n
                walk(n.body, prefix + n.name + ".")
            elif isinstance(n, ast.ClassDef):
                out[prefix + n.name + "#class"] = n
                walk(n.body, prefix + n.name + ".")
    walk(tree.body, "")
    return out


def _new_tree():
    return ast.parse(open(os.path.join(_REPO, "neuro_foundation.py"), encoding="utf-8").read())


def test_static_EXACTLY_these_functions_differ_from_the_base_and_nothing_else(capsys):
    new, old = _new_tree(), _read_base()
    fn_new, fn_old = _funcs(new), _funcs(old)
    removed = sorted(set(fn_old) - set(fn_new))
    added = sorted(k for k in set(fn_new) - set(fn_old) if not k.endswith("#class"))
    classes_added = sorted(k for k in set(fn_new) - set(fn_old) if k.endswith("#class"))
    differing = sorted(k for k in set(fn_new) & set(fn_old)
                       if not k.endswith("#class") and ast.dump(fn_new[k]) != ast.dump(fn_old[k]))
    # a class statement itself differs only by its body, which is covered function by function; compare headers
    for k in set(fn_new) & set(fn_old):
        if k.endswith("#class"):
            a, b = fn_new[k], fn_old[k]
            assert ast.dump(ast.ClassDef(a.name, a.bases, a.keywords, [], a.decorator_list)) == ast.dump(ast.ClassDef(b.name, b.bases, b.keywords, [], b.decorator_list))
    assert removed == [] and classes_added == []
    assert added == sorted(NEW_METHODS), added
    assert differing == sorted(CHANGED), differing
    top_new = [ast.dump(n) for n in new.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    top_old = [ast.dump(n) for n in old.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert top_new == top_old                                        # no module-level addition, change or import
    names_new = {t.name for t in new.body if isinstance(t, (ast.FunctionDef, ast.ClassDef))}
    assert names_new == {t.name for t in old.body if isinstance(t, (ast.FunctionDef, ast.ClassDef))}   # no module-level def/class added either
    n_old = len([k for k in fn_old if not k.endswith("#class")])
    print("[ast-proof] base-functions=%d added=%d %s differing=%d %s module-level-non-def-statements=%d identical=True module-level-defs-added=0"
          % (n_old, len(added), added, len(differing), differing, len(top_new)))
    assert n_old > 50 and len(top_new) > 5


def _executable_nodes():
    """Every AST node inside the NEW and CHANGED functions, minus their docstrings."""
    new = _new_tree()
    fns = _funcs(new)
    out = []
    for key in NEW_METHODS + CHANGED:
        fn = fns[key]
        body = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant)
                               and isinstance(fn.body[0].value.value, str)) else fn.body
        out.append((key, fn.name, [a.arg for a in fn.args.args + fn.args.kwonlyargs], body))
    return out


_CC = re.compile(r"(^|[^A-Za-z])cc([^A-Za-z]|$)", re.I)


def test_static_no_cc_token_in_the_executable_code_of_the_new_and_changed_functions():
    for key, name, args, body in _executable_nodes():
        assert not _CC.search(name) and not any(_CC.search(a) for a in args), key
        for stmt in body:
            for c in ast.walk(stmt):
                for ident in (getattr(c, "id", None), getattr(c, "attr", None), getattr(c, "arg", None)):
                    assert not (ident and _CC.search(ident)), (key, ident)
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    assert not _CC.search(c.value), (key, c.value[:40])


def test_static_the_canonical_helper_reads_no_environment_anywhere_in_its_executable_code():
    for key, name, args, body in _executable_nodes():
        for stmt in body:
            for c in ast.walk(stmt):
                ident = getattr(c, "id", None) or getattr(c, "attr", None)
                assert ident not in ("environ", "getenv", "putenv", "os", "sys", "subprocess", "open"), (key, ident)
                assert not isinstance(c, (ast.Import, ast.ImportFrom)), key
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    assert "NG_" not in c.value and "$" not in c.value, (key, c.value[:40])


def test_static_the_diff_adds_no_environment_read_even_in_prose_code_lines():
    """The same rule applied to the raw ADDED lines (computed in-process against the base text, so it works in any tree), minus comments and
    docstrings: no `environ` / `getenv` / `os.` there."""
    import difflib
    base_text = subprocess.run(["git", "-C", _REPO, "show", "%s:neuro_foundation.py" % BASE_REV], capture_output=True, text=True)
    assert base_text.returncode == 0, "cannot read the base: this must FAIL, never skip"
    base_lines = base_text.stdout.splitlines()
    new_text = open(os.path.join(_REPO, "neuro_foundation.py"), encoding="utf-8").read()
    new_lines = new_text.splitlines()
    new = ast.parse(new_text)
    doc_lines = set()
    for n in ast.walk(new):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr) \
                and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str):
            doc_lines.update(range(n.body[0].lineno, n.body[0].end_lineno + 1))
    added = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, base_lines, new_lines, autojunk=False).get_opcodes():
        if tag in ("insert", "replace"):
            added.extend((j + 1, new_lines[j]) for j in range(j1, j2))
    assert len(added) > 100                                          # the diff really was computed (the canonical block is ~190 lines)
    for lineno, code in added:
        if lineno not in doc_lines and not code.strip().startswith("#"):
            assert not re.search(r"environ|getenv|\bos\.", code), (lineno, code)


def test_static_the_heartbeat_query_is_pure_and_the_note_is_called_once_per_sweep_from_the_sweep_only():
    """P571 F1 (LAW 4): `_fair_chance_heartbeat_fresh` writes nothing and logs nothing; `_in_fair_chance_window` never calls the note; the sweep
    body calls `_note_fair_chance_stale` exactly once, outside any loop; no other function in the module calls it."""
    fns = _funcs(_new_tree())
    for c in ast.walk(fns["Graph._fair_chance_heartbeat_fresh"]):
        assert not isinstance(c, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)), ast.dump(c)[:80]
        assert (getattr(c, "id", None) or getattr(c, "attr", None)) not in ("logger", "warning", "info", "stale_logged", "_note_fair_chance_stale")
        assert not (isinstance(c, ast.Constant) and c.value == "stale_logged")
    assert not any(getattr(c, "attr", None) == "_note_fair_chance_stale" for c in ast.walk(fns["Graph._in_fair_chance_window"]))
    sweep = fns["Graph._collect_orphan_nodes"]
    calls = [c for c in ast.walk(sweep) if isinstance(c, ast.Call) and getattr(c.func, "attr", None) == "_note_fair_chance_stale"]
    assert len(calls) == 1
    for loop in (n for n in ast.walk(sweep) if isinstance(n, (ast.For, ast.While, ast.ListComp, ast.comprehension))):
        assert not any(c is calls[0] for c in ast.walk(loop))        # once per SWEEP, never per node
    elsewhere = [k for k, f in fns.items() if not k.endswith("#class") and k != "Graph._collect_orphan_nodes"
                 and any(isinstance(c, ast.Attribute) and c.attr == "_note_fair_chance_stale" for c in ast.walk(f))]
    assert elsewhere == [], elsewhere


def test_static_the_docstrings_carry_the_framing_and_the_host_contract():
    fns = _funcs(_new_tree())
    sweep = ast.get_docstring(fns["Graph._collect_orphan_nodes"])
    for needle in ("TESTED FIRST", "pioneer", "Josh's call", "LAW 8", "HOST-NEUTRAL", "reads NO environment", "P563", "§8.13", "graph STEPS"):
        assert needle in sweep, needle
    reg = ast.get_docstring(fns["Graph.enable_fair_chance_window"])
    for needle in ("AUTONOMIC", "LAW 8", "reads", "ValueError", "LAST"):
        assert needle in reg, needle
    src = open(os.path.join(_REPO, "neuro_foundation.py"), encoding="utf-8").read()
    assert "SHARED MACHINERY being TESTED FIRST on the CC" in src          # the framing sentence, in the code comment block too
