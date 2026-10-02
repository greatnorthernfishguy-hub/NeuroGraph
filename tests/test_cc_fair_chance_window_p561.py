# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 build worker, lane sweep-probation-p552, NG-3b / round 2) — Chief-003 Addendum 2 (the step window is a DEDICATED knob)
# What: the fixture now patches the STEP window (`_CC_PROBATION_STEP_WINDOW`) and the GRADUATION period (`_CC_CONV_PROBATION_PERIOD`) to two DIFFERENT
#   values (so any coupling shows up in every test); the graduation-count assertions follow the graduation value. NEW: the two knobs are pinned INDEPENDENT
#   in BOTH directions and under a SWAP (the step window follows `CC_PROBATION_STEP_WINDOW` at stamp / seed / exact-repeat reset while probation_total /
#   probation_remaining / the ramp's default total follow `CC_CONV_PROBATION_PERIOD`, and vice versa); the env reader's matrix (absent / valid / zero /
#   negative / non-int / empty / bool-ish); IMPORT-TIME subprocess checks that an invalid value logs exactly ONE WARNING naming the variable and the failed
#   rule and that both knobs set independently reach their constants; static checks that no step-window stamp reads the graduation constant and no
#   graduation line reads the step constant, and that the single `10` literal in the new code is the default constant. No literal 10 in this file.
# Why: P1 decouples the window (graph STEPS) from graduation (pulses); sharing one variable would re-couple them at the tuning level.
# How: monkeypatch the module constants (or set the env in a subprocess); values are derived or small distinct numbers, never today's default.
# [2026-10-02] Claude Sonnet 5.5 (Z12 build worker, lane sweep-probation-p552, NG-3 / round 2) — Josh's ruling (Exec P550 / P552; Exec P561 P1 + P2;
#   Exec P562 Addendum 1); CC-CALLOSUM-TRUTH §8.13
# What: replaces tests/test_cc_probation_advances_p552.py (the round-1 NG-1 file; renamed: its subject `probation_advances` is now
#   `probation_population` + `fair_chance_window_open`, and the rename rule is ONE name everywhere, no alias). Covers: the population
#   (round-1 unit tests, re-keyed, meaning preserved); the predicate's value matrix (direct tests; the sweep no longer parses anything, NG-4);
#   the STEP-keyed window on a REAL `Graph` with the real `step()` and the real `cc_update_probation` (held clock, N stepped pulses,
#   many steps in one pulse count once, timestep regression, legacy seeding (Z12 design call A), odd shapes of the new fields); the
#   GRADUATION differential: the BASE `cc_update_probation` (`git show b5e47686:cc_ng_organism.py`, `ast`-extracted, exec'd in the new
#   module's namespace) against the new function over 1000 seeded graphs, every old field / the `graduated` list / the exception class
#   identical (the two NEW fields are the only allowed difference, compared separately); the completion HEARTBEAT (unarmed / armed /
#   stale / strict `>` / one WARNING per episode / recovery / stamp only on a non-raising completion); `_BANNED_META` and a wire round-trip;
#   the deposit stamps; the window SIZE coming from the existing `CC_CONV_PROBATION_PERIOD` (no new literal); the static (`ast`/tokenize) checks.
# Why: P1 (the unit is graph steps) + P2 (a completion heartbeat) + Addendum 1 (all window logic in the host predicate).
# How: real `Graph`, real functions, a patched clock through the ONE module-level indirection `_probation_clock`; no checkpoint, no
#   NeuroGraphMemory, no embedder. Node order is dict insertion order, so nothing depends on PYTHONHASHSEED (the 0..31 sweep is in the return).
# -------------------
import ast
import io
import logging
import math
import os
import random
import subprocess
import sys
import tokenize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest

import cc_ng_organism as org
import cc_topology_export as tex
from neuro_foundation import Graph

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_REV = "b5e476863cc069a29ec482959b4f9465f2ea4ccf"
NEW_FIELDS = ("probation_steps_remaining", "probation_last_timestep")
PERIOD = 3     # the STEP-window size under test: set by monkeypatching `_CC_PROBATION_STEP_WINDOW`
GRAD = 5       # the GRADUATION period under test: `_CC_CONV_PROBATION_PERIOD`, deliberately a DIFFERENT number


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    """The heartbeat's ONE clock, patched; the heartbeat state is reset before and after every test."""
    now = [1000.0]
    monkeypatch.setattr(org, "_probation_clock", lambda: now[0])
    monkeypatch.setattr(org, "_CC_PROBATION_STEP_WINDOW", PERIOD)
    monkeypatch.setattr(org, "_CC_CONV_PROBATION_PERIOD", GRAD)
    saved = dict(org._PROBATION_HEARTBEAT)
    org._PROBATION_HEARTBEAT.update(armed=False, max_age_s=None, stamp=None, stale_logged=False)
    yield now
    org._PROBATION_HEARTBEAT.clear()
    org._PROBATION_HEARTBEAT.update(saved)


class _N:
    """A bare node-like (the predicate / population read only .metadata)."""

    def __init__(self, metadata):
        self.metadata = metadata


def _graph(timestep=0):
    g = Graph()
    g.timestep = timestep
    # These tests are about the FIELDS. A real step() runs the orphan sweep, which (with no predicate registered, or in NG-4's integration
    # group with one) is the sweep's own business: keep grace out of the way so no node is culled while a field is under test.
    g.config["orphan_node_grace_period"] = 2 ** 20
    return g


def _deposit(g, nid="n", mode="conversational", **extra):
    """The REAL deposit path (so the stamps under test are the ones production writes)."""
    meta = {"source": "cc_gateway", "creation_mode": mode}
    meta.update(extra)
    return org._cc_deposit_memory_node(g, None, nid, np.ones(8, dtype=np.float32), "text", meta, index_in_recall=False)


def _pulse(g, steps=1):
    for _ in range(steps):
        g.step()
    return org.cc_update_probation(g)


def _steps(g, nid="n"):
    return g.nodes[nid].metadata.get("probation_steps_remaining")


# ---------------------------------------------------------------------------
# the population (round-1 NG-1 unit tests, re-keyed: `probation_advances` -> `probation_population`)
# ---------------------------------------------------------------------------

def test_population_false_for_ingested():
    assert org.probation_population(_N({"creation_mode": "ingested"})) is False


@pytest.mark.parametrize("meta", [
    {"creation_mode": "conversational"}, {"creation_mode": "emergent"}, {"creation_mode": ""}, {"creation_mode": None},
    {"creation_mode": "Ingested"},   # exact-match semantics, as the base skip had
    {},                              # no creation_mode: older checkpoints / seeds are still advanced
    {"probation_remaining": 5},
    None,                            # the `(node.metadata or {})` guard
])
def test_population_true_for_everything_else(meta):
    assert org.probation_population(_N(meta)) is True


def test_population_raises_like_the_base_expression_on_non_dict_metadata():
    with pytest.raises(AttributeError):
        org.probation_population(_N("not-a-dict"))


def test_cc_update_probation_consults_probation_population_and_never_the_heartbeat_aware_predicate(monkeypatch):
    """The advancer's skip is the pure population, looked up at call time, and never the heartbeat-aware predicate:
    otherwise a stale heartbeat would stop the advancer from advancing, so it could never recover (a deadlock)."""
    g = _graph()
    _deposit(g, "conv")
    _deposit(g, "ing", mode="ingested")
    org.cc_update_probation(g)
    assert g.nodes["conv"].metadata["probation_remaining"] == GRAD - 1
    assert g.nodes["ing"].metadata["probation_remaining"] == GRAD                 # ingested: not advanced
    monkeypatch.setattr(org, "probation_population", lambda node: True)
    org.cc_update_probation(g)
    assert g.nodes["ing"].metadata["probation_remaining"] == GRAD - 1             # population True => advanced
    monkeypatch.setattr(org, "probation_population", lambda node: False)
    org.cc_update_probation(g)
    assert g.nodes["conv"].metadata["probation_remaining"] == GRAD - 2            # population False: skipped, so unchanged
    # ... and a predicate that says False for everything (a stale heartbeat) changes NOTHING about the advancer:
    monkeypatch.setattr(org, "probation_population", lambda node: True)
    monkeypatch.setattr(org, "fair_chance_window_open", lambda node: False)
    monkeypatch.setattr(org, "_probation_heartbeat_fresh", lambda: False)
    before = g.nodes["conv"].metadata["probation_remaining"]
    org.cc_update_probation(g)
    assert g.nodes["conv"].metadata["probation_remaining"] == before - 1


# ---------------------------------------------------------------------------
# fair_chance_window_open: the value matrix (it owns the parsing now; the sweep parses nothing)
# ---------------------------------------------------------------------------

def _open_meta(steps, **extra):
    meta = {"creation_mode": "conversational", "probation_steps_remaining": steps, "probation_last_timestep": 0}
    meta.update(extra)
    return meta


@pytest.mark.parametrize("steps", [1, 2, 3, 0.5, 2.5, 2 ** 40, 2 ** 70, np.float64(5.0), np.float64(0.25)])
def test_window_open_for_a_finite_number_greater_than_zero(steps):
    assert org.fair_chance_window_open(_N(_open_meta(steps))) is True


@pytest.mark.parametrize("steps", [None, "5", "", "abc", [], [3], {}, -1, -0.5, 0, 0.0, False, True, float("nan"),
                                   float("inf"), -float("inf"), np.int64(5), np.float32(5.0)])
def test_window_closed_for_every_other_shape_and_it_never_raises(steps):
    assert org.fair_chance_window_open(_N(_open_meta(steps))) is False


def test_window_closed_when_the_key_is_absent():
    assert org.fair_chance_window_open(_N({"creation_mode": "conversational", "probation_last_timestep": 0})) is False


@pytest.mark.parametrize("last", [None, "7", [], True, False, float("nan"), float("inf"), -float("inf")])
def test_window_closed_when_last_timestep_is_not_a_finite_number(last):
    """A count > 0 with no usable `last` could never be decremented: it must read as closed, never as forever-open."""
    assert org.fair_chance_window_open(_N(_open_meta(3, probation_last_timestep=last))) is False
    meta = _open_meta(3)
    del meta["probation_last_timestep"]
    assert org.fair_chance_window_open(_N(meta)) is False


def test_window_closed_for_ingested_even_with_an_open_count():
    assert org.fair_chance_window_open(_N(_open_meta(5, creation_mode="ingested"))) is False


@pytest.mark.parametrize("meta", [None, [], "", 0, "not-a-dict", ["x"], 7])
def test_window_closed_for_none_or_non_dict_metadata_without_raising(meta):
    assert org.fair_chance_window_open(_N(meta)) is False


def test_window_open_needs_no_creation_mode():
    assert org.fair_chance_window_open(_N({"probation_steps_remaining": 2, "probation_last_timestep": 0})) is True


# ---------------------------------------------------------------------------
# P1: the window's unit is GRAPH STEPS
# ---------------------------------------------------------------------------

def test_1_held_clock_100_pulses_zero_steps_leave_the_step_window_untouched_and_the_old_count_on_its_own_rule():
    g = _graph(timestep=50)
    _deposit(g)
    assert _steps(g) == PERIOD and g.nodes["n"].metadata["probation_last_timestep"] == 50
    for _ in range(100):
        org.cc_update_probation(g)                                   # NO step() anywhere: the clock is held
    assert g.timestep == 50
    assert _steps(g) == PERIOD                                       # the step window never moved
    assert org.fair_chance_window_open(g.nodes["n"]) is True
    md = g.nodes["n"].metadata
    assert md["probation_remaining"] == 0 and md["graduated"] in (True, False)   # the OLD per-pulse rule ran to its end


def test_2_exactly_N_stepped_pulses_close_the_window():
    g = _graph()
    _deposit(g)
    for k in range(1, PERIOD + 1):
        _pulse(g)
        assert _steps(g) == PERIOD - k
        assert org.fair_chance_window_open(g.nodes["n"]) is (k < PERIOD)
    _pulse(g)
    assert _steps(g) == 0                                            # never below zero


def test_3_many_steps_in_one_pulse_count_once():
    g = _graph()
    _deposit(g)
    _pulse(g, steps=40)
    assert _steps(g) == PERIOD - 1
    org.cc_update_probation(g)                                       # a second pulse with NO new steps: still once
    assert _steps(g) == PERIOD - 1


def test_4_timestep_regression_resets_last_and_never_decrements_or_goes_negative():
    g = _graph(timestep=40)
    _deposit(g)
    _pulse(g)
    assert _steps(g) == PERIOD - 1 and g.nodes["n"].metadata["probation_last_timestep"] == 41
    g.timestep = 7                                                   # a restore from an older checkpoint
    org.cc_update_probation(g)
    md = g.nodes["n"].metadata
    assert md["probation_steps_remaining"] == PERIOD - 1 and md["probation_last_timestep"] == 7
    _pulse(g)                                                        # forward again from the new baseline
    assert _steps(g) == PERIOD - 2
    for _ in range(3 * PERIOD):
        _pulse(g)
    assert _steps(g) == 0


def test_5_legacy_node_is_seeded_and_its_window_is_bounded():
    g = _graph(timestep=500)
    g.create_node(node_id="legacy", metadata={"creation_mode": "conversational", "probation_remaining": 0,
                                              "probation_total": PERIOD, "graduated": True})
    assert "probation_steps_remaining" not in g.nodes["legacy"].metadata
    org.cc_update_probation(g)
    md = g.nodes["legacy"].metadata
    assert md["probation_steps_remaining"] == PERIOD and md["probation_last_timestep"] == 500   # seeded fresh, once
    assert org.fair_chance_window_open(g.nodes["legacy"]) is True
    org.cc_update_probation(g)                                       # NOT reseeded on every visit
    assert g.nodes["legacy"].metadata["probation_steps_remaining"] == PERIOD
    for _ in range(PERIOD):
        _pulse(g)
    assert g.nodes["legacy"].metadata["probation_steps_remaining"] == 0                         # bounded: <= N stepped pulses
    assert org.fair_chance_window_open(g.nodes["legacy"]) is False


def test_5b_seeding_scope_is_every_population_node_that_lacks_the_key_and_never_ingested():
    """Z12 design call (A), as written: EVERY population node lacking the key is seeded (even one that never had a
    probation_remaining); `ingested` is outside the population and is never seeded."""
    g = _graph(timestep=9)
    g.create_node(node_id="no_old_key", metadata={"creation_mode": "emergent"})
    g.create_node(node_id="no_mode", metadata={})
    g.create_node(node_id="ing", metadata={"creation_mode": "ingested", "probation_remaining": 5})
    org.cc_update_probation(g)
    for nid in ("no_old_key", "no_mode"):
        assert g.nodes[nid].metadata["probation_steps_remaining"] == PERIOD
    assert "probation_steps_remaining" not in g.nodes["ing"].metadata
    assert "probation_last_timestep" not in g.nodes["ing"].metadata


@pytest.mark.parametrize("steps", [None, "3", "", [], float("nan"), float("inf"), -float("inf"), True, False, -1, 0, 0.0])
@pytest.mark.parametrize("last", [0, None, "x", float("nan"), True])
def test_6_odd_shapes_of_the_new_fields_are_left_untouched_and_never_raise(steps, last):
    g = _graph(timestep=20)
    g.create_node(node_id="n", metadata={"creation_mode": "conversational", "probation_remaining": PERIOD,
                                         "probation_total": PERIOD, "probation_steps_remaining": steps,
                                         "probation_last_timestep": last})
    org.cc_update_probation(g)
    md = g.nodes["n"].metadata
    for k, v in (("probation_steps_remaining", steps), ("probation_last_timestep", last)):
        assert repr(md[k]) == repr(v), "an odd shape must be left exactly as found"
    assert org.fair_chance_window_open(g.nodes["n"]) is False        # today's sweep


def test_6b_a_graph_with_no_readable_clock_gets_no_step_progress_and_nothing_raises(monkeypatch):
    monkeypatch.setattr(org, "_CC_CONV_PROBATION_REQUIRE_SPIKE", False)   # a bare node has no spike_history
    class NoClock:
        def __init__(self):
            self._step_lock = __import__("threading").RLock()
            self.config = {}
            self.nodes = {}
    g = NoClock()
    from types import SimpleNamespace
    g.nodes["a"] = SimpleNamespace(metadata={"creation_mode": "conversational", "probation_remaining": 1, "probation_total": 4},
                                   threshold=3, intrinsic_excitability=.5)
    assert org.cc_update_probation(g) == ["a"]                       # graduation does not need the clock
    assert "probation_steps_remaining" not in g.nodes["a"].metadata
    assert not hasattr(g, "timestep")                                # and the advancer never creates one


def test_the_deposit_stamps_both_fields_beside_the_old_three_and_an_exact_repeat_resets_both():
    g = _graph(timestep=11)
    n = _deposit(g)
    assert n.metadata["probation_steps_remaining"] == PERIOD and n.metadata["probation_last_timestep"] == 11
    assert n.metadata["probation_remaining"] == GRAD and n.metadata["probation_total"] == GRAD
    for _ in range(2):
        _pulse(g)
    assert _steps(g) == PERIOD - 2
    _deposit(g)                                                      # the exact-repeat path
    md = g.nodes["n"].metadata
    assert md["probation_steps_remaining"] == PERIOD and md["probation_last_timestep"] == g.timestep
    assert md["probation_remaining"] == GRAD


def test_the_deposit_never_raises_on_a_graph_with_no_clock():
    class NoClock:
        def __init__(self):
            self._step_lock = __import__("threading").RLock()
            self.config = {}
            self.nodes = {}

        def create_node(self, node_id, metadata):
            from types import SimpleNamespace
            n = SimpleNamespace(metadata=metadata, threshold=0, intrinsic_excitability=0)
            self.nodes[node_id] = n
            return n
    g = NoClock()
    n = org._cc_deposit_memory_node(g, None, "x", np.ones(4, dtype=np.float32), "t", {}, index_in_recall=False)
    assert n.metadata["probation_last_timestep"] == 0 and n.metadata["probation_steps_remaining"] == PERIOD


def test_the_window_size_is_the_dedicated_knob_and_the_only_literal_is_its_default(monkeypatch):
    monkeypatch.setattr(org, "_CC_PROBATION_STEP_WINDOW", PERIOD + 4)
    g = _graph(timestep=4)
    _deposit(g, "dep")
    g.create_node(node_id="seeded", metadata={"creation_mode": "conversational"})
    org.cc_update_probation(g)
    assert g.nodes["dep"].metadata["probation_steps_remaining"] == PERIOD + 4
    assert g.nodes["seeded"].metadata["probation_steps_remaining"] == PERIOD + 4
    tree = ast.parse(open(os.path.join(_REPO, "cc_ng_organism.py"), encoding="utf-8").read())
    new_fns = {"_probation_clock", "_probation_finite_number", "probation_population", "probation_heartbeat_arm",
               "_probation_heartbeat_stamp", "_probation_heartbeat_fresh", "fair_chance_window_open",
               "_probation_step_window_tick", "_read_probation_step_window_env"}
    found = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert new_fns <= found
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in new_fns:
            for c in ast.walk(n):
                assert not (isinstance(c, ast.Constant) and type(c.value) is int and c.value == 10), \
                    "an int literal 10 in %s: the window size is the env knob CC_PROBATION_STEP_WINDOW" % n.name
    tens = [t.targets[0].id for t in tree.body if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name)
            and isinstance(t.value, ast.Constant) and type(t.value.value) is int and t.value.value == 10]
    assert tens == ["_CC_PROBATION_STEP_WINDOW_DEFAULT"], tens             # the ONE int-10 literal in the new module-level code


# ---------------------------------------------------------------------------
# THE GRADUATION DIFFERENTIAL (the Exec's condition): the BASE function vs the new one
# ---------------------------------------------------------------------------

_base_cache = {}


def _base_cc_update_probation():
    if "ast" not in _base_cache:
        out = subprocess.run(["git", "-C", _REPO, "show", "%s:cc_ng_organism.py" % BASE_REV], capture_output=True)
        assert out.returncode == 0, ("cannot read the base cc_ng_organism.py at %s in %s: the differential must FAIL, never "
                                     "skip" % (BASE_REV[:8], _REPO))
        tree = ast.parse(out.stdout.decode())
        _base_cache["ast"] = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cc_update_probation")
    ns = dict(vars(org))              # the base resolves _cc_mutation_lock / constants / _cc_has_ever_fired in the NEW module, fresh each call
    exec(compile(ast.fix_missing_locations(ast.Module(body=[_base_cache["ast"]], type_ignores=[])),
                 "<base cc_update_probation @ b5e47686>", "exec"), ns)
    return ns["cc_update_probation"]


_ABSENT = object()
_MODES = ["ingested", "conversational", None, "emergent", "weird", ""]            # None => key absent
_PROBS_CLEAN = [_ABSENT, None, 0, 1, 2, 3, 5, 7, -1, 2.5]
_PROBS_ODD_SAFE = _PROBS_CLEAN + [float("nan"), True, float("inf"), 0.0, -float("inf"), False]      # odd, but they do not raise
_PROBS_ODD = _PROBS_ODD_SAFE + ["3", "x"]                                                              # the str shapes raise at `prob <= 0`
_TOTALS_SAFE = [_ABSENT, 3, 5, 7, 0, float("nan")]
_TOTALS = _TOTALS_SAFE + ["x", None]
_DAMPS = [_ABSENT, 0.1, 0.3, 0.5, "bad"]
_N_NODES = 24
_PULSES = 14


def _build(seed, odd):
    rng = random.Random(seed)
    g = _graph(timestep=rng.randint(0, 60))
    poison = odd and rng.random() < 0.5          # half the odd graphs carry the RAISING shapes, half only the non-raising odd ones
    probs = _PROBS_ODD if poison else (_PROBS_ODD_SAFE if odd else _PROBS_CLEAN)
    for i in range(_N_NODES):
        meta = {}
        mode = rng.choice(_MODES)
        if mode is not None:
            meta["creation_mode"] = mode
        elif rng.random() < 0.3:
            meta["creation_mode"] = None                                          # the None creation_mode shape
        prob = rng.choice(probs)
        if prob is not _ABSENT:
            meta["probation_remaining"] = prob
        total = rng.choice(_TOTALS if poison else (_TOTALS_SAFE if odd else [_ABSENT, 3, 5, 7]))
        if total is not _ABSENT:
            meta["probation_total"] = total
        damp = rng.choice(_DAMPS if poison else [_ABSENT, 0.1, 0.3, 0.5])
        if damp is not _ABSENT:
            meta["novelty_dampening"] = damp
        if rng.random() < 0.15:
            meta["probation_expired_unfired"] = True
        if rng.random() < 0.1:
            meta["graduated"] = rng.choice([True, False])
        if rng.random() < 0.3:                                                    # PRE-EXISTING new-field shapes (compared apart)
            meta["probation_steps_remaining"] = rng.choice([_ABSENT, 3, 1, 0, "x", None, float("nan"), True])
            if meta["probation_steps_remaining"] is _ABSENT:
                del meta["probation_steps_remaining"]
            if rng.random() < 0.7:
                meta["probation_last_timestep"] = rng.choice([0, 5, 30, "x", None])
        node = g.create_node(node_id="n%02d" % i, metadata=meta)
        if rng.random() < 0.5:
            node.spike_history.append(float(rng.randint(1, 50)))
        node.intrinsic_excitability = rng.choice([0.3, 0.5, 1.0])
        node.threshold = rng.choice([0.85, 1.05, 1.2])
    if poison:
        for nid in rng.sample(sorted(g.nodes), 2):
            if rng.random() < 0.5:
                g.nodes[nid].metadata = None                                      # the None-metadata shape
    return g


def _snapshot(g, drop_new):
    out = []
    for nid, node in g.nodes.items():
        md = node.metadata
        if md is None:
            mdr = "None"
        else:
            items = [(k, v) for k, v in md.items() if not (drop_new and k in NEW_FIELDS)]
            mdr = repr(sorted(items, key=lambda kv: kv[0]))
        out.append((nid, mdr, repr(node.intrinsic_excitability), repr(node.threshold)))
    return out


def _new_fields(g):
    out = {}
    for nid, node in g.nodes.items():
        if isinstance(node.metadata, dict):
            out[nid] = tuple(repr(node.metadata.get(k, "<absent>")) for k in NEW_FIELDS)
    return out


def _run(fn, g):
    try:
        return ("ok", fn(g))
    except Exception as exc:   # noqa: BLE001 -- the exception CLASS is part of "identical"
        return ("raised", type(exc).__name__)


def _differential(seeds, odd, clock):
    base = _base_cc_update_probation()
    raised = graduated = 0
    for seed in seeds:
        gb, gn = _build(seed, odd), _build(seed, odd)
        assert _snapshot(gb, drop_new=False) == _snapshot(gn, drop_new=False), "builder is not deterministic (seed %d)" % seed
        rng = random.Random(seed * 7919 + 1)
        for pulse in range(_PULSES):
            dt = rng.choice([0, 0, 1, 2, 5, -3])         # the clock: held, stepping, or going BACKWARDS; the base ignores it
            gb.timestep += dt
            gn.timestep += dt
            stamp_before = org._PROBATION_HEARTBEAT["stamp"]
            clock[0] += 1.0
            rb, rn = _run(base, gb), _run(org.cc_update_probation, gn)
            assert rb == rn, "seed %d pulse %d: result %r != %r" % (seed, pulse, rb, rn)
            assert _snapshot(gb, drop_new=True) == _snapshot(gn, drop_new=True), "seed %d pulse %d: OLD fields diverged" % (seed, pulse)
            if rb[0] == "raised":
                raised += 1
                assert org._PROBATION_HEARTBEAT["stamp"] == stamp_before, "a raising pass must NOT stamp the heartbeat"
                break
            assert org._PROBATION_HEARTBEAT["stamp"] == clock[0], "a non-raising pass must stamp the heartbeat"
            graduated += len(rb[1])
    print("[differential] graphs=%d odd=%s old-fields-identical-every-pulse=True graphs-that-raised-identically=%d graduated-nodes=%d"
          % (len(list(seeds)), odd, raised, graduated))
    return raised, graduated


def test_graduation_is_byte_identical_to_the_base_over_500_clean_graphs(clock):
    raised, graduated = _differential(range(500), odd=False, clock=clock)
    assert raised == 0
    assert graduated > 300                       # the family really exercises graduation (not vacuous)


def test_graduation_is_byte_identical_to_the_base_over_500_odd_graphs_including_the_raising_shapes(clock):
    raised, graduated = _differential(range(500), odd=True, clock=clock)
    assert raised > 100                          # str probation / None metadata / bad total really occur
    assert graduated > 50


def test_the_new_fields_obey_the_step_rules_over_the_same_family(clock):
    """The new fields are the only allowed difference, so they get their own oracle: for every node a NON-raising pass
    visits, the pair moves exactly as the rules say (seed / decrement-once-per-advance / regress-without-decrement /
    untouched; an `ingested` node is never touched). The family includes PRE-EXISTING new-field shapes."""
    visited = 0
    for seed in range(300):
        g = _build(seed, odd=False)
        rng = random.Random(seed)
        for _ in range(8):
            g.timestep += rng.choice([0, 1, 3, -2])
            before = {nid: dict(n.metadata) for nid, n in g.nodes.items()}
            t = g.timestep
            result = _run(org.cc_update_probation, g)
            assert result[0] == "ok", result                 # the clean family has no raising shape
            for nid, node in g.nodes.items():
                md0, md = before[nid], node.metadata
                s0, l0 = md0.get(NEW_FIELDS[0], _ABSENT), md0.get(NEW_FIELDS[1], _ABSENT)
                s1, l1 = md.get(NEW_FIELDS[0], _ABSENT), md.get(NEW_FIELDS[1], _ABSENT)
                visited += 1
                if md.get("creation_mode") == "ingested":
                    assert (repr(s1), repr(l1)) == (repr(s0), repr(l0))
                elif s0 is _ABSENT:
                    assert s1 == PERIOD and l1 == t
                elif org._probation_finite_number(s0) and s0 > 0 and org._probation_finite_number(l0):
                    if t > l0:
                        assert s1 == max(0, s0 - 1) and l1 == t
                    elif t == l0:
                        assert (s1, l1) == (s0, l0)
                    else:
                        assert s1 == s0 and l1 == t
                else:
                    assert (repr(s1), repr(l1)) == (repr(s0), repr(l0))
    assert visited > 50000


# ---------------------------------------------------------------------------
# P2: the completion HEARTBEAT
# ---------------------------------------------------------------------------

def _warnings(caplog, needle="fair-chance window"):
    return [r for r in caplog.records if r.name == "cc_ng_organism" and r.levelno == logging.WARNING and needle in r.getMessage()]


@pytest.mark.parametrize("bad", [0, -1, -0.5, float("nan"), float("inf"), True, False, None, "5", [], 0.0])
def test_arm_rejects_anything_that_is_not_a_finite_number_above_zero(bad):
    with pytest.raises(ValueError):
        org.probation_heartbeat_arm(bad)
    assert org._PROBATION_HEARTBEAT["armed"] is False                # a refused arm arms nothing


def test_arm_stamps_now_so_a_fresh_arm_is_fresh(clock):
    org.probation_heartbeat_arm(300)
    assert org._PROBATION_HEARTBEAT["armed"] is True and org._PROBATION_HEARTBEAT["stamp"] == clock[0]
    assert org.fair_chance_window_open(_N(_open_meta(2))) is True


def test_11_not_armed_means_the_heartbeat_is_not_enforced_however_old(clock):
    g = _graph()
    _deposit(g)
    clock[0] += 2 ** 30                                              # a billion seconds: irrelevant while unarmed
    assert org.fair_chance_window_open(g.nodes["n"]) is True
    assert org._PROBATION_HEARTBEAT["armed"] is False


def test_9_fresh_means_exemption_as_before_and_staleness_is_a_strict_greater_than(clock):
    g = _graph()
    _deposit(g)
    org.probation_heartbeat_arm(300)
    clock[0] += 300                                                  # age == max_age: still fresh (strict >)
    assert org.fair_chance_window_open(g.nodes["n"]) is True
    clock[0] += 0.001                                                # one tick past: stale
    assert org.fair_chance_window_open(g.nodes["n"]) is False


def test_9b_stale_closes_the_exemption_for_EVERY_node_with_one_warning_per_episode_and_recovers(clock, caplog):
    g = _graph()
    for nid in ("a", "b", "c"):
        _deposit(g, nid)
    org.probation_heartbeat_arm(300)
    clock[0] += 301
    with caplog.at_level(logging.DEBUG, logger="cc_ng_organism"):
        for _ in range(5):
            for nid in ("a", "b", "c"):
                assert org.fair_chance_window_open(g.nodes[nid]) is False
        ws = _warnings(caplog)
        assert len(ws) == 1                                          # ONE per stale EPISODE, not per call
        msg = ws[0].getMessage()
        assert "301" in msg and "300" in msg and "OFF" in msg        # the age, the limit, and what it means
        org.cc_update_probation(g)                                   # the advancer completes a cycle: recovery
        infos = [r for r in caplog.records if r.name == "cc_ng_organism" and r.levelno == logging.INFO and "back on" in r.getMessage()]
        assert len(infos) == 1
        for nid in ("a", "b", "c"):
            assert org.fair_chance_window_open(g.nodes[nid]) is True
        clock[0] += 301                                              # a SECOND episode warns again (the latch re-armed)
        assert org.fair_chance_window_open(g.nodes["a"]) is False
        assert len(_warnings(caplog)) == 2


def test_stale_is_false_for_a_node_with_no_window_too_and_ingested_never_reaches_the_heartbeat(clock, caplog):
    org.probation_heartbeat_arm(20)
    clock[0] += 21
    with caplog.at_level(logging.DEBUG, logger="cc_ng_organism"):
        assert org.fair_chance_window_open(_N(_open_meta(5, creation_mode="ingested"))) is False
        assert _warnings(caplog) == []                               # not in the population: the heartbeat was not consulted
        assert org.fair_chance_window_open(_N({"creation_mode": "conversational"})) is False
        assert len(_warnings(caplog)) == 1


def test_the_stamp_is_written_only_by_a_non_raising_completion_never_before_the_loop_never_in_a_finally(clock):
    g = _graph()
    _deposit(g, "ok")
    org.probation_heartbeat_arm(300)
    t0 = org._PROBATION_HEARTBEAT["stamp"]
    clock[0] += 5
    org.cc_update_probation(g)
    assert org._PROBATION_HEARTBEAT["stamp"] == clock[0] != t0       # completed: stamped
    g.create_node(node_id="poison", metadata={"creation_mode": "conversational", "probation_remaining": "7"})
    t1 = org._PROBATION_HEARTBEAT["stamp"]
    clock[0] += 5
    with pytest.raises(TypeError):
        org.cc_update_probation(g)                                   # the str poison aborts the pass mid-loop
    assert org._PROBATION_HEARTBEAT["stamp"] == t1                   # NOT stamped: the staleness IS the signal


def test_a_pass_that_is_never_called_leaves_the_old_stamp(clock):
    org.probation_heartbeat_arm(300)
    t0 = org._PROBATION_HEARTBEAT["stamp"]
    clock[0] += 1000
    assert org._PROBATION_HEARTBEAT["stamp"] == t0
    assert org.fair_chance_window_open(_N(_open_meta(3))) is False


def test_arm_again_resets_the_stale_latch(clock, caplog):
    org.probation_heartbeat_arm(20)
    clock[0] += 21
    with caplog.at_level(logging.DEBUG, logger="cc_ng_organism"):
        org.fair_chance_window_open(_N(_open_meta(3)))
        assert len(_warnings(caplog)) == 1
        org.probation_heartbeat_arm(20)
        clock[0] += 21
        org.fair_chance_window_open(_N(_open_meta(3)))
        assert len(_warnings(caplog)) == 2


# ---------------------------------------------------------------------------
# the wire: _BANNED_META and a round-trip
# ---------------------------------------------------------------------------

def test_8_banned_meta_carries_both_new_fields():
    assert set(NEW_FIELDS) <= set(tex._BANNED_META)
    assert {"probation_remaining", "probation_total", "creation_time"} <= set(tex._BANNED_META)     # unchanged neighbours
    out = tex._portable_metadata({k: 5 for k in NEW_FIELDS} | {"creation_mode": "conversational"})
    assert out == {"creation_mode": "conversational"}


def test_8b_a_wire_round_trip_never_carries_them_and_the_receiver_restamps_with_its_own_clock(tmp_path):
    import importlib
    tc = importlib.import_module("tests.test_cc_topology_callosum")
    sg, sv, ids = tc._build_sender()
    for n in sg.nodes.values():
        n.metadata.update(probation_steps_remaining=2, probation_last_timestep=999, probation_remaining=2)
    path, _ = tc._export(sg, sv, tmp_path)
    for frame in tex.read_topology_frames(open(path, "rb").read()):
        for rec in frame.get("nodes") or ():
            for k in NEW_FIELDS + ("probation_remaining", "probation_total"):
                assert k not in rec["metadata"], "%s leaked onto the wire" % k
    rg, rv = tc._receiver()
    rg.timestep = 31
    tc._merge(rg, rv, path, tmp_path)
    assert set(rg.nodes) == set(ids.values())
    for n in rg.nodes.values():                                      # arrivals WITH an embedding: stamped by the receiver's deposit
        md = n.metadata
        assert md["probation_steps_remaining"] == PERIOD
        assert md["probation_last_timestep"] == 31 != 999           # the RECEIVER's clock, never the sender's


# ---------------------------------------------------------------------------
# Chief-003 Addendum 2: the TWO knobs are INDEPENDENT (step window vs graduation period), in BOTH directions
# ---------------------------------------------------------------------------

STEP_X, GRAD_Y = 4, 6          # distinct, neither is today's default; the second parametrization SWAPS them


@pytest.mark.parametrize("step,grad", [(STEP_X, GRAD_Y), (GRAD_Y, STEP_X)], ids=["step=X,grad=Y", "SWAPPED step=Y,grad=X"])
def test_the_two_knobs_are_independent_in_both_directions_and_under_a_swap(monkeypatch, step, grad):
    monkeypatch.setattr(org, "_CC_PROBATION_STEP_WINDOW", step)
    monkeypatch.setattr(org, "_CC_CONV_PROBATION_PERIOD", grad)
    g = _graph(timestep=3)
    # STAMP: the step window follows the STEP knob, probation_total / probation_remaining follow the GRADUATION knob
    md = _deposit(g, "dep").metadata
    assert md["probation_steps_remaining"] == step
    assert md["probation_remaining"] == grad and md["probation_total"] == grad
    # SEED: a legacy node's step window follows the step knob; it has no graduation fields and gets none
    g.create_node(node_id="legacy", metadata={"creation_mode": "conversational"})
    # the RAMP's default total (a node with probation_remaining and NO probation_total) follows the graduation knob
    g.create_node(node_id="ramp", metadata={"creation_mode": "conversational", "probation_remaining": grad, "novelty_dampening": 0.3})
    g.nodes["ramp"].intrinsic_excitability = 0.3
    _pulse(g)                                                          # one stepped pulse: every counter moves exactly once
    assert g.nodes["legacy"].metadata["probation_steps_remaining"] == step
    assert "probation_remaining" not in g.nodes["legacy"].metadata
    assert g.nodes["dep"].metadata["probation_steps_remaining"] == step - 1
    assert g.nodes["dep"].metadata["probation_remaining"] == grad - 1
    frac = max(0.0, min(1.0, 1.0 - (grad - 1) / grad))                 # total defaults to the GRADUATION period
    assert g.nodes["ramp"].intrinsic_excitability == pytest.approx(0.3 + 0.7 * frac)
    # EXACT-REPEAT RESET: both counters return to THEIR OWN knob
    _pulse(g)
    _deposit(g, "dep")
    md = g.nodes["dep"].metadata
    assert md["probation_steps_remaining"] == step and md["probation_remaining"] == grad and md["probation_total"] == grad
    # and the windows CLOSE on their own schedules: `step` stepped pulses close the step window, `grad` pulses graduate
    g2 = _graph(timestep=0)
    _deposit(g2, "n")
    for k in range(1, max(step, grad) + 1):
        _pulse(g2)
        assert g2.nodes["n"].metadata["probation_steps_remaining"] == max(0, step - k)
        assert g2.nodes["n"].metadata["probation_remaining"] == max(0, grad - k)


def test_the_step_window_follows_its_knob_even_when_the_graduation_period_is_not_a_clean_multiple(monkeypatch):
    monkeypatch.setattr(org, "_CC_PROBATION_STEP_WINDOW", STEP_X + 1)
    monkeypatch.setattr(org, "_CC_CONV_PROBATION_PERIOD", STEP_X + 100)
    md = _deposit(_graph(), "n").metadata
    assert md["probation_steps_remaining"] == STEP_X + 1 and md["probation_remaining"] == STEP_X + 100


# --- the env reader (pure) and the IMPORT-TIME behaviour ---------------------

_ENV = "CC_PROBATION_STEP_WINDOW"


def test_the_default_equals_todays_effective_window_derived_from_the_graduation_knobs_own_default():
    tree = ast.parse(open(os.path.join(_REPO, "cc_ng_organism.py"), encoding="utf-8").read())
    grad_default = None
    for t in tree.body:
        if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name) and t.targets[0].id == "_CC_CONV_PROBATION_PERIOD":
            call = t.value.args[0]                                      # int(os.environ.get("CC_CONV_PROBATION_PERIOD", "<default>"))
            grad_default = int(call.args[1].value)
    assert grad_default is not None and org._CC_PROBATION_STEP_WINDOW_DEFAULT == grad_default


def test_the_reader_absent_means_the_default_silently():
    assert org._read_probation_step_window_env({}) == (org._CC_PROBATION_STEP_WINDOW_DEFAULT, None)
    assert org._read_probation_step_window_env({"CC_CONV_PROBATION_PERIOD": str(GRAD_Y)}) == (org._CC_PROBATION_STEP_WINDOW_DEFAULT, None)


@pytest.mark.parametrize("raw,value", [("7", 7), (" 7 ", 7), ("007", 7), ("+3", 3), ("1", 1), ("100000", 100000)])
def test_the_reader_accepts_a_positive_integer(raw, value):
    assert org._read_probation_step_window_env({_ENV: raw}) == (value, None)


@pytest.mark.parametrize("raw,rule", [
    ("0", "<= 0"), ("-3", "<= 0"), ("-0", "<= 0"),
    ("abc", "not an integer"), ("", "not an integer"), ("   ", "not an integer"), ("3.5", "not an integer"), ("1e1", "not an integer"), ("7x", "not an integer"),
    ("true", "bool-ish"), ("False", "bool-ish"), ("YES", "bool-ish"), ("no", "bool-ish"), ("on", "bool-ish"), ("OFF", "bool-ish"),
])
def test_the_reader_rejects_everything_else_with_the_default_and_names_which_rule_failed(raw, rule):
    value, problem = org._read_probation_step_window_env({_ENV: raw})
    assert value == org._CC_PROBATION_STEP_WINDOW_DEFAULT
    assert problem and rule in problem


def _import_organism(env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CC_")}
    env.update({"PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"})
    env.update(env_extra)
    code = ("import logging; logging.basicConfig(level=logging.INFO, format='%(name)s|%(levelname)s|%(message)s'); "
            "import cc_ng_organism as o; print('RESULT', o._CC_PROBATION_STEP_WINDOW, o._CC_CONV_PROBATION_PERIOD)")
    out = subprocess.run([sys.executable, "-c", code], cwd=_REPO, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-600:]
    result = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")]
    assert len(result) == 1
    _, step, grad = result[0].split()
    warns = [ln for ln in (out.stderr + out.stdout).splitlines() if "|WARNING|" in ln and "CC_PROBATION_STEP_WINDOW" in ln]
    return int(step), int(grad), warns


def test_import_time_absent_and_valid_values_log_nothing_and_each_knob_reaches_its_own_constant():
    step, grad, warns = _import_organism({})
    assert step == org._CC_PROBATION_STEP_WINDOW_DEFAULT and warns == []
    step, grad, warns = _import_organism({_ENV: str(STEP_X), "CC_CONV_PROBATION_PERIOD": str(GRAD_Y)})
    assert (step, grad) == (STEP_X, GRAD_Y) and warns == []
    step, grad, warns = _import_organism({_ENV: str(GRAD_Y), "CC_CONV_PROBATION_PERIOD": str(STEP_X)})   # SWAPPED at the process boundary too
    assert (step, grad) == (GRAD_Y, STEP_X) and warns == []
    step, grad, warns = _import_organism({"CC_CONV_PROBATION_PERIOD": str(GRAD_Y)})                       # only graduation set: the step window is untouched
    assert step == org._CC_PROBATION_STEP_WINDOW_DEFAULT and grad == GRAD_Y and warns == []


@pytest.mark.parametrize("raw,rule", [("abc", "not an integer"), ("0", "<= 0"), ("-4", "<= 0"), ("true", "bool-ish"), ("", "not an integer")])
def test_import_time_an_invalid_value_gives_the_default_and_exactly_one_warning_naming_the_variable_and_the_rule(raw, rule):
    step, grad, warns = _import_organism({_ENV: raw})
    assert step == org._CC_PROBATION_STEP_WINDOW_DEFAULT
    assert len(warns) == 1 and rule in warns[0]
    assert str(org._CC_PROBATION_STEP_WINDOW_DEFAULT) in warns[0]       # it names the default it fell back to


def test_static_no_step_window_stamp_reads_the_graduation_constant_and_no_graduation_line_reads_the_step_constant():
    tree = ast.parse(open(os.path.join(_REPO, "cc_ng_organism.py"), encoding="utf-8").read())
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def stamps(fn, key):
        out = []
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript) \
                    and isinstance(n.targets[0].slice, ast.Constant) and n.targets[0].slice.value == key:
                if isinstance(n.value, ast.Name):          # a stamp of a CONSTANT; the decrement (`max(0, steps - 1)`) is not one
                    out.append(n.value.id)
        return out
    dep = fns["_cc_deposit_memory_node"]
    assert stamps(dep, "probation_steps_remaining") == ["_CC_PROBATION_STEP_WINDOW"]
    assert stamps(dep, "probation_remaining") == ["_CC_CONV_PROBATION_PERIOD"]
    assert stamps(dep, "probation_total") == ["_CC_CONV_PROBATION_PERIOD"]
    assert stamps(fns["_probation_step_window_tick"], "probation_steps_remaining") == ["_CC_PROBATION_STEP_WINDOW"]
    tick_names = {n.id for n in ast.walk(fns["_probation_step_window_tick"]) if isinstance(n, ast.Name)}
    assert "_CC_CONV_PROBATION_PERIOD" not in tick_names
    adv_names = {n.id for n in ast.walk(fns["cc_update_probation"]) if isinstance(n, ast.Name)}
    assert "_CC_PROBATION_STEP_WINDOW" not in adv_names and "_CC_CONV_PROBATION_PERIOD" in adv_names   # graduation stays graduation-only


# ---------------------------------------------------------------------------
# static checks (ast / tokenize)
# ---------------------------------------------------------------------------

_SKIP_DIRS = {".git", "tests", "data", "Defunct-Historical", "__pycache__", "docs", "handoffs"}


def _repo_py_files():
    for root, dirs, files in os.walk(_REPO):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


def _parse(path):
    with open(path, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=path)


@pytest.mark.parametrize("name", ["probation_population", "fair_chance_window_open", "probation_heartbeat_arm"])
def test_each_public_name_is_defined_exactly_once_in_the_whole_tree_and_in_cc_ng_organism(name):
    defs = []
    for path in _repo_py_files():
        for n in ast.walk(_parse(path)):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
                defs.append((os.path.relpath(path, _REPO), n.lineno))
    assert len(defs) == 1 and defs[0][0] == "cc_ng_organism.py", defs
    tops = [n for n in _parse(os.path.join(_REPO, "cc_ng_organism.py")).body
            if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(tops) == 1


def test_cc_update_probation_uses_the_population_only_and_holds_no_literal():
    tree = _parse(os.path.join(_REPO, "cc_ng_organism.py"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cc_update_probation")
    called = [c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
    assert called.count("probation_population") == 1
    for forbidden in ("fair_chance_window_open", "_probation_heartbeat_fresh"):
        assert forbidden not in called and forbidden not in {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert called.count("_probation_step_window_tick") == 1 and called.count("_probation_heartbeat_stamp") == 1
    assert [c for c in ast.walk(fn) if isinstance(c, ast.Constant) and c.value == "ingested"] == []


def test_the_stamp_call_is_the_last_statement_before_return_outside_any_try_or_finally():
    tree = _parse(os.path.join(_REPO, "cc_ng_organism.py"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cc_update_probation")
    with_stmt = next(n for n in fn.body if isinstance(n, ast.With))
    body = with_stmt.body
    assert isinstance(body[-1], ast.Return)
    last_expr = body[-2]
    assert isinstance(last_expr, ast.Expr) and isinstance(last_expr.value, ast.Call) \
        and last_expr.value.func.id == "_probation_heartbeat_stamp"
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.Try)]   # no try/finally anywhere in the advancer


def test_the_ingested_literal_lives_only_in_probation_population_within_cc_ng_organism():
    holders = []
    for top in _parse(os.path.join(_REPO, "cc_ng_organism.py")).body:
        for c in ast.walk(top):
            if isinstance(c, ast.Constant) and c.value == "ingested":
                holders.append(getattr(top, "name", "<module-level>"))
    assert holders == ["probation_population"], holders


def test_neuro_foundation_is_unaware_of_the_heartbeat_and_imports_no_cc_module():
    tree = _parse(os.path.join(_REPO, "neuro_foundation.py"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            assert "heartbeat" not in n.id.lower() and not n.id.startswith("cc_")
        elif isinstance(n, ast.Attribute):
            assert "heartbeat" not in n.attr.lower() and not n.attr.startswith("cc_")
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            assert "heartbeat" not in n.name.lower()
        elif isinstance(n, ast.Import):
            assert not any(a.name.startswith("cc_") for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            assert not (n.module or "").startswith("cc_")
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 60:
            assert "heartbeat" not in n.value.lower() and n.value != "ingested"


def _code_tokens(path):
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.NAME, tokenize.STRING):
                yield tok.string


@pytest.mark.parametrize("old", ["probation_advances", "_probation_advances"])
def test_no_leftover_old_predicate_name_in_the_organism_or_the_export_outside_comments(old):
    """The rename is complete in the files this commit owns: no NAME or STRING token (so no docstring either) carries the old
    name; only the changelog COMMENTS keep it, as history. (The protected sweep body is re-keyed in NG-4.)"""
    for f in ("cc_ng_organism.py", "cc_topology_export.py"):
        hits = [s for s in _code_tokens(os.path.join(_REPO, f)) if old in s]
        assert hits == [], (f, hits[:2])
