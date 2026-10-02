# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 build worker, lane sweep-probation-p552, NG-3c / round 2) — Josh's ruling (Exec P550 / P552; Exec P561 P1 + P2; Exec P563;
#   Chief-003 Addenda 3-4); CC-CALLOSUM-TRUTH §8.13
# FRAMING (Josh): the fair-chance window is SHARED MACHINERY being TESTED FIRST on the CC, not CC-specific code: the pioneer implementation of canonical §8.13 arrival
#   protection; rollout to other NeuroGraphs (Syl's) is Josh's call, LAW 8 gate per host.
# What: REPLACES tests/test_cc_fair_chance_window_p561.py (the organism-side predicate / heartbeat / step-tick tests, which pin logic that is now CANONICAL and is tested in
#   tests/test_fair_chance_window.py). This file tests only the HOST'S share: `probation_population` over the single constant `PROBATION_UNADVANCED_CREATION_MODES`; the deposit
#   and the advancer calling the CANONICAL helpers (and being exactly the round-1 functions on a graph that has none, or an unregistered one); THE GRADUATION DIFFERENTIAL: the BASE
#   `cc_update_probation` (`git show b5e47686:cc_ng_organism.py`, `ast`-extracted, exec'd in the NEW module's namespace) against the new function over 1500 seeded graphs: on an
#   UNREGISTERED graph EVERYTHING is identical (not one key added), and on a REGISTERED one every old field, the `graduated` list and the exception CLASS are identical with the two
#   window fields the only allowed difference (they get their own oracle); an integration group (the real organism advancer + the real canonical sweep + a real Graph: held clock,
#   stepped pulses, wiring, legacy seeding, `ingested` excluded, heartbeat stale / recovery, the poisoned advancer, an unarmed host); the wire; graduation and the window size INDEPENDENT
#   in BOTH directions and under a swap; and the static checks (no duplicated step / heartbeat logic and no leftover old name anywhere in the tree).
# Why: Exec P563 moved the window logic into canonical neuro_foundation; LAW 4: the host calls the canonical helpers and holds no copy.
# How: real Graph and real functions; scratch only; clocks are plain callables handed to the registration. Node order is dict insertion order, so nothing depends on PYTHONHASHSEED
#   (the 0..31 sweep is in the return).
# -------------------
import ast
import importlib
import io
import logging
import msgpack
import os
import random
import subprocess
import struct
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
STEPS, LAST = "fair_chance_steps_remaining", "fair_chance_last_timestep"
NEW_FIELDS = (STEPS, LAST)
WINDOW = 3      # the window size handed to the canonical registration (the host's choice; the organism reads no environment for it)
GRAD = 5        # the graduation period `_CC_CONV_PROBATION_PERIOD`, deliberately a DIFFERENT number


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture(autouse=True)
def grad_period(monkeypatch):
    monkeypatch.setattr(org, "_CC_CONV_PROBATION_PERIOD", GRAD)


def _graph(timestep=0, register=True, window=WINDOW, max_age=None, clock=None, grace=None):
    g = Graph()
    g.timestep = timestep
    if grace is not None:
        g.config["orphan_node_grace_period"] = grace
    if register:
        g.enable_fair_chance_window(window, heartbeat_max_age_s=max_age, excluded_creation_modes=org.PROBATION_UNADVANCED_CREATION_MODES,
                                    clock=clock if clock is not None else (Clock() if max_age is not None else None))
    return g


def _deposit(g, nid="n", mode="conversational", age=125):
    n = org._cc_deposit_memory_node(g, None, nid, np.ones(8, dtype=np.float32), "text", {"source": "cc_gateway", "creation_mode": mode},
                                    index_in_recall=False)
    n.creation_time = g.timestep - age
    return n


def _pulse(g, steps=1):
    for _ in range(steps):
        g.step()
    return org.cc_update_probation(g)


class _N:
    def __init__(self, metadata):
        self.metadata = metadata


# ---------------------------------------------------------------------------
# the population (the host's definition; ONE constant)
# ---------------------------------------------------------------------------

def test_population_false_for_ingested_true_for_everything_else():
    assert org.probation_population(_N({"creation_mode": "ingested"})) is False
    for meta in ({"creation_mode": "conversational"}, {"creation_mode": "emergent"}, {"creation_mode": ""}, {"creation_mode": None},
                 {"creation_mode": "Ingested"}, {}, {"probation_remaining": 5}, None):
        assert org.probation_population(_N(meta)) is True


def test_population_raises_like_the_round_1_expression_on_non_dict_metadata():
    with pytest.raises(AttributeError):
        org.probation_population(_N("not-a-dict"))


@pytest.mark.parametrize("mode", ["ingested", "conversational", "Ingested", "", None, ["ingested"], {"a": 1}, 7, float("nan"), ("ingested",)])
def test_population_is_byte_identical_to_the_round_1_skip_over_odd_values(mode):
    node = _N({"creation_mode": mode})
    assert org.probation_population(node) is (((node.metadata or {}).get("creation_mode") != "ingested"))


def test_the_excluded_modes_are_ONE_constant_used_by_the_population_and_handed_to_the_registration():
    assert org.PROBATION_UNADVANCED_CREATION_MODES == ("ingested",)
    tree = ast.parse(open(os.path.join(_REPO, "cc_ng_organism.py"), encoding="utf-8").read())
    holders = []
    for top in tree.body:
        for c in ast.walk(top):
            if isinstance(c, ast.Constant) and c.value == "ingested":
                holders.append(getattr(top, "name", None) or [t.id for t in top.targets if isinstance(t, ast.Name)][0])
    assert holders == ["PROBATION_UNADVANCED_CREATION_MODES"], holders      # the literal lives in the constant and nowhere else
    g = _graph()
    assert g._fair_chance_cfg["excluded"] == org.PROBATION_UNADVANCED_CREATION_MODES


# ---------------------------------------------------------------------------
# the deposit calls the canonical helper
# ---------------------------------------------------------------------------

def test_a_registered_deposit_opens_the_window_beside_the_graduation_fields_and_an_exact_repeat_reopens_it():
    g = _graph(timestep=11, grace=2 ** 20)
    md = _deposit(g, "n").metadata
    assert md[STEPS] == WINDOW and md[LAST] == 11
    assert md["probation_remaining"] == GRAD and md["probation_total"] == GRAD       # graduation follows ITS knob
    for _ in range(2):
        _pulse(g)
    assert g.nodes["n"].metadata[STEPS] == WINDOW - 2 and g.nodes["n"].metadata["probation_remaining"] == GRAD - 2
    _deposit(g, "n")                                                                   # the exact-repeat path
    md = g.nodes["n"].metadata
    assert md[STEPS] == WINDOW and md[LAST] == g.timestep and md["probation_remaining"] == GRAD


def test_an_unregistered_deposit_adds_no_window_field_at_all():
    g = _graph(register=False)
    md = _deposit(g, "n").metadata
    assert STEPS not in md and LAST not in md and md["probation_remaining"] == GRAD


def test_an_excluded_creation_mode_is_never_stamped():
    g = _graph()
    md = _deposit(g, "ing", mode="ingested").metadata
    assert STEPS not in md and LAST not in md and md["probation_remaining"] == GRAD   # graduation is stamped as before; the window is not opened


def test_the_deposit_never_raises_on_a_graph_that_predates_the_helpers():
    class Old:
        def __init__(self):
            self._step_lock = __import__("threading").RLock()
            self.config = {}
            self.nodes = {}

        def create_node(self, node_id, metadata):
            from types import SimpleNamespace
            n = SimpleNamespace(metadata=metadata, threshold=0, intrinsic_excitability=0)
            self.nodes[node_id] = n
            return n
    n = org._cc_deposit_memory_node(Old(), None, "x", np.ones(4, dtype=np.float32), "t", {}, index_in_recall=False)
    assert STEPS not in n.metadata and n.metadata["probation_remaining"] == GRAD


# ---------------------------------------------------------------------------
# the advancer calls the canonical helpers
# ---------------------------------------------------------------------------

def test_the_advancer_calls_the_canonical_advance_once_per_population_node_before_any_continue_and_never_for_an_excluded_one():
    g = _graph(grace=2 ** 20)
    seen = []
    real = g.fair_chance_advance
    g.fair_chance_advance = lambda node: (seen.append(node.node_id), real(node))[1]
    g.create_node(node_id="conv", metadata={"creation_mode": "conversational", "probation_remaining": 3, "probation_total": 3})
    g.create_node(node_id="expired", metadata={"creation_mode": "conversational", "probation_remaining": 0})     # an early `continue` for the old logic
    g.create_node(node_id="nokey", metadata={"creation_mode": "emergent"})                                          # no probation_remaining at all
    g.create_node(node_id="ing", metadata={"creation_mode": "ingested", "probation_remaining": 3})
    org.cc_update_probation(g)
    assert sorted(seen) == ["conv", "expired", "nokey"]                              # once each; the excluded mode never reaches it
    for nid in ("conv", "expired", "nokey"):
        assert g.nodes[nid].metadata[STEPS] == WINDOW                                # seeded (design call A)


def test_the_completion_heartbeat_is_stamped_only_by_a_non_raising_pass_never_before_the_loop_never_in_a_finally():
    clock = Clock(1000.0)
    g = _graph(max_age=300, clock=clock, grace=2 ** 20)
    _deposit(g, "ok")
    t0 = g._fair_chance_cfg["stamp"]
    clock.t += 5
    org.cc_update_probation(g)
    assert g._fair_chance_cfg["stamp"] == clock.t != t0                              # completed: stamped
    g.create_node(node_id="poison", metadata={"creation_mode": "conversational", "probation_remaining": "7"})
    t1 = g._fair_chance_cfg["stamp"]
    clock.t += 5
    with pytest.raises(TypeError):
        org.cc_update_probation(g)                                                    # the str poison aborts the pass mid-loop
    assert g._fair_chance_cfg["stamp"] == t1                                          # NOT stamped: the staleness IS the signal


def test_a_pass_that_never_happens_leaves_the_old_stamp_and_the_window_closes_for_every_node():
    clock = Clock(1000.0)
    g = _graph(max_age=300, clock=clock)
    _deposit(g, "n")
    clock.t += 1000
    assert g._in_fair_chance_window(g.nodes["n"]) is False


def test_the_advancer_is_exactly_the_round_1_function_when_the_graph_has_no_helpers(monkeypatch):
    g = _graph(register=False, grace=2 ** 20)
    for name in ("fair_chance_advance", "fair_chance_heartbeat_stamp", "fair_chance_stamp"):
        monkeypatch.setattr(Graph, name, None, raising=False)                         # a graph that PREDATES the helpers (getattr finds None)
    _deposit(g, "n")
    assert org.cc_update_probation(g) == []
    assert STEPS not in g.nodes["n"].metadata


# ---------------------------------------------------------------------------
# THE GRADUATION DIFFERENTIAL (the Exec's condition): the BASE function vs the new one
# ---------------------------------------------------------------------------

_base_cache = {}


def _base_cc_update_probation():
    if "ast" not in _base_cache:
        out = subprocess.run(["git", "-C", _REPO, "show", "%s:cc_ng_organism.py" % BASE_REV], capture_output=True)
        assert out.returncode == 0, ("cannot read the base cc_ng_organism.py at %s in %s: the differential must FAIL, never skip" % (BASE_REV[:8], _REPO))
        tree = ast.parse(out.stdout.decode())
        _base_cache["ast"] = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cc_update_probation")
    ns = dict(vars(org))     # the base resolves _cc_mutation_lock / constants / _cc_has_ever_fired in the NEW module, fresh each call
    exec(compile(ast.fix_missing_locations(ast.Module(body=[_base_cache["ast"]], type_ignores=[])), "<base cc_update_probation @ b5e47686>", "exec"), ns)
    return ns["cc_update_probation"]


_ABSENT = object()
_MODES = ["ingested", "conversational", None, "emergent", "weird", ""]
_PROBS_CLEAN = [_ABSENT, None, 0, 1, 2, 3, 5, 7, -1, 2.5]
_PROBS_ODD_SAFE = _PROBS_CLEAN + [float("nan"), True, float("inf"), 0.0, -float("inf"), False]
_PROBS_ODD = _PROBS_ODD_SAFE + ["3", "x"]
_TOTALS_SAFE = [_ABSENT, 3, 5, 7, 0, float("nan")]
_TOTALS = _TOTALS_SAFE + ["x", None]
_DAMPS = [_ABSENT, 0.1, 0.3, 0.5, "bad"]
_N_NODES = 24
_PULSES = 14


def _build(seed, odd, register):
    rng = random.Random(seed)
    g = _graph(timestep=rng.randint(0, 60), register=register, grace=2 ** 20)
    poison = odd and rng.random() < 0.5
    probs = _PROBS_ODD if poison else (_PROBS_ODD_SAFE if odd else _PROBS_CLEAN)
    for i in range(_N_NODES):
        meta = {}
        mode = rng.choice(_MODES)
        if mode is not None:
            meta["creation_mode"] = mode
        elif rng.random() < 0.3:
            meta["creation_mode"] = None
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
        if rng.random() < 0.3:                                                   # PRE-EXISTING window-field shapes (compared apart when registered)
            meta[STEPS] = rng.choice([3, 1, 0, "x", None, float("nan"), True])
            if rng.random() < 0.7:
                meta[LAST] = rng.choice([0, 5, 30, "x", None])
        node = g.create_node(node_id="n%02d" % i, metadata=meta)
        if rng.random() < 0.5:
            node.spike_history.append(float(rng.randint(1, 50)))
        node.intrinsic_excitability = rng.choice([0.3, 0.5, 1.0])
        node.threshold = rng.choice([0.85, 1.05, 1.2])
    if poison:
        for nid in rng.sample(sorted(g.nodes), 2):
            if rng.random() < 0.5:
                g.nodes[nid].metadata = None
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


def _run(fn, g):
    try:
        return ("ok", fn(g))
    except Exception as exc:   # noqa: BLE001 -- the exception CLASS is part of "identical"
        return ("raised", type(exc).__name__)


def _differential(seeds, odd, register):
    base = _base_cc_update_probation()
    raised = graduated = 0
    for seed in seeds:
        gb, gn = _build(seed, odd, False), _build(seed, odd, register)
        assert _snapshot(gb, drop_new=False) == _snapshot(gn, drop_new=False), "builder is not deterministic (seed %d)" % seed
        rng = random.Random(seed * 7919 + 1)
        for pulse in range(_PULSES):
            dt = rng.choice([0, 0, 1, 2, 5, -3])         # the clock: held, stepping, or going BACKWARDS; the base ignores it
            gb.timestep += dt
            gn.timestep += dt
            rb, rn = _run(base, gb), _run(org.cc_update_probation, gn)
            assert rb == rn, "seed %d pulse %d: result %r != %r" % (seed, pulse, rb, rn)
            assert _snapshot(gb, drop_new=register) == _snapshot(gn, drop_new=register), "seed %d pulse %d: OLD fields diverged" % (seed, pulse)
            if rb[0] == "raised":
                raised += 1
                break
            graduated += len(rb[1])
    print("[differential] graphs=%d odd=%s registered=%s old-fields-identical-every-pulse=True graphs-that-raised-identically=%d graduated-nodes=%d"
          % (len(list(seeds)), odd, register, raised, graduated))
    return raised, graduated


def test_UNREGISTERED_graduation_is_byte_identical_to_the_base_NOT_ONE_KEY_ADDED_clean():
    raised, graduated = _differential(range(250), odd=False, register=False)
    assert raised == 0 and graduated > 150


def test_UNREGISTERED_graduation_is_byte_identical_to_the_base_NOT_ONE_KEY_ADDED_odd_including_the_raising_shapes():
    raised, graduated = _differential(range(250), odd=True, register=False)
    assert raised > 50 and graduated > 25


def test_REGISTERED_graduation_is_byte_identical_to_the_base_over_500_clean_graphs():
    raised, graduated = _differential(range(500), odd=False, register=True)
    assert raised == 0 and graduated > 300


def test_REGISTERED_graduation_is_byte_identical_to_the_base_over_500_odd_graphs_including_the_raising_shapes():
    raised, graduated = _differential(range(500), odd=True, register=True)
    assert raised > 100 and graduated > 50


def test_the_window_fields_obey_the_canonical_step_rules_when_driven_by_the_real_advancer():
    """The two window fields are the only allowed difference, so they get their own oracle: for every node a NON-raising pass visits."""
    visited = 0
    for seed in range(300):
        g = _build(seed, odd=False, register=True)
        rng = random.Random(seed)
        for _ in range(8):
            g.timestep += rng.choice([0, 1, 3, -2])
            before = {nid: dict(n.metadata) for nid, n in g.nodes.items()}
            t = g.timestep
            result = _run(org.cc_update_probation, g)
            assert result[0] == "ok", result
            for nid, node in g.nodes.items():
                md0, md = before[nid], node.metadata
                s0, l0 = md0.get(STEPS, _ABSENT), md0.get(LAST, _ABSENT)
                s1, l1 = md.get(STEPS, _ABSENT), md.get(LAST, _ABSENT)
                visited += 1
                if md.get("creation_mode") == "ingested":
                    assert (repr(s1), repr(l1)) == (repr(s0), repr(l0))
                elif s0 is _ABSENT:
                    assert s1 == WINDOW and l1 == t
                elif g._is_finite_number(s0) and s0 > 0 and g._is_finite_number(l0):
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
# graduation and the window size are INDEPENDENT, in BOTH directions and under a swap
# ---------------------------------------------------------------------------

STEP_X, GRAD_Y = 4, 6


@pytest.mark.parametrize("step,grad", [(STEP_X, GRAD_Y), (GRAD_Y, STEP_X)], ids=["window=X,graduation=Y", "SWAPPED window=Y,graduation=X"])
def test_graduation_and_the_window_size_are_independent_in_both_directions_and_under_a_swap(monkeypatch, step, grad):
    monkeypatch.setattr(org, "_CC_CONV_PROBATION_PERIOD", grad)
    g = _graph(timestep=3, window=step)
    md = _deposit(g, "dep").metadata
    assert md[STEPS] == step                                                          # the window follows what the HOST registered
    assert md["probation_remaining"] == grad and md["probation_total"] == grad         # graduation follows ITS knob
    g.create_node(node_id="legacy", metadata={"creation_mode": "conversational"})
    g.create_node(node_id="ramp", metadata={"creation_mode": "conversational", "probation_remaining": grad, "novelty_dampening": 0.3})
    g.nodes["ramp"].intrinsic_excitability = 0.3
    g.config["orphan_node_grace_period"] = 2 ** 20
    _pulse(g)                                                                          # one stepped pulse: every counter moves exactly once
    assert g.nodes["legacy"].metadata[STEPS] == step and "probation_remaining" not in g.nodes["legacy"].metadata
    assert g.nodes["dep"].metadata[STEPS] == step - 1 and g.nodes["dep"].metadata["probation_remaining"] == grad - 1
    frac = max(0.0, min(1.0, 1.0 - (grad - 1) / grad))                                 # the ramp's default total is the GRADUATION period
    assert g.nodes["ramp"].intrinsic_excitability == pytest.approx(0.3 + 0.7 * frac)
    _pulse(g)
    _deposit(g, "dep")                                                                 # the exact repeat: each counter returns to ITS OWN value
    md = g.nodes["dep"].metadata
    assert md[STEPS] == step and md["probation_remaining"] == grad and md["probation_total"] == grad
    g2 = _graph(timestep=0, window=step, grace=2 ** 20)
    _deposit(g2, "n")
    for k in range(1, max(step, grad) + 1):                                            # and they CLOSE on their own schedules
        _pulse(g2)
        assert g2.nodes["n"].metadata[STEPS] == max(0, step - k)
        assert g2.nodes["n"].metadata["probation_remaining"] == max(0, grad - k)


# ---------------------------------------------------------------------------
# the INTEGRATION group: the real organism advancer + the real canonical sweep + a real Graph
# ---------------------------------------------------------------------------

def _hold(g, ms=1):
    for _ in range(ms):
        g._collect_orphan_nodes()


def test_int_held_clock_100_pulses_zero_steps_the_node_survives_every_sweep():
    g = _graph(timestep=500)
    _deposit(g)
    for _ in range(100):
        org.cc_update_probation(g)
        _hold(g)
        assert "n" in g.nodes
    assert g.timestep == 500 and g.nodes["n"].metadata[STEPS] == WINDOW
    assert g.nodes["n"].metadata["probation_remaining"] == 0                          # the OLD per-pulse rule ran to its end regardless


def test_int_exactly_N_stepped_pulses_close_the_window_then_the_next_sweep_takes_the_unwired_node():
    g = _graph(grace=0)
    _deposit(g, "n")
    for _ in range(WINDOW):
        assert "n" in g.nodes
        _pulse(g)
    g.step()
    assert "n" not in g.nodes


def test_int_a_node_that_wires_during_its_window_survives_after_it():
    g = _graph(grace=0)
    for nid in ("x", "y", "w"):
        _deposit(g, nid)
    g.create_hyperedge({"x", "y"})
    for _ in range(WINDOW + 2):
        _pulse(g)
    assert g.nodes["x"].metadata[STEPS] == 0 and {"x", "y"} <= set(g.nodes) and "w" not in g.nodes


def test_int_legacy_node_with_a_closed_old_window_is_seeded_survives_the_P550_case_then_is_bounded():
    g = _graph(timestep=900)
    n = g.create_node(node_id="forest", metadata={"creation_mode": "conversational", "probation_remaining": 0, "probation_total": GRAD, "graduated": True})
    n.creation_time = 0
    org.cc_update_probation(g)                                                          # the first pulse seeds it
    _hold(g)
    assert "forest" in g.nodes and g.nodes["forest"].metadata[STEPS] == WINDOW
    for _ in range(WINDOW):
        _pulse(g)
    g.step()
    assert "forest" not in g.nodes                                                      # one-time and BOUNDED


def test_int_without_any_pass_a_legacy_node_is_culled_as_today():
    g = _graph(timestep=900)
    n = g.create_node(node_id="forest", metadata={"creation_mode": "conversational", "probation_remaining": 0})
    n.creation_time = 0
    g._collect_orphan_nodes()
    assert "forest" not in g.nodes


def test_int_an_ingested_node_is_swept_even_with_an_open_graduation_count_and_fields():
    g = _graph()
    n = _deposit(g, "ing", mode="ingested")
    n.metadata.update({STEPS: WINDOW, LAST: 0})
    g._collect_orphan_nodes()
    assert "ing" not in g.nodes


def test_int_arrivals_with_an_embedding_get_a_window_and_the_structural_install_gets_none():
    """Who is covered is unchanged by the DEPOSIT: a node deposited WITH an embedding (cc_topology_merge -> _cc_deposit_memory_node) is stamped;
    the no-embedding structural install (create_node only) has no fields and is swept as today."""
    g = _graph(timestep=300)
    _deposit(g, "with_embedding")
    n = g.create_node(node_id="no_embedding", metadata={"creation_mode": "conversational"})
    n.creation_time = 0
    g._collect_orphan_nodes()
    assert "with_embedding" in g.nodes and "no_embedding" not in g.nodes


def test_int_real_steps_past_grace_spare_an_open_window_and_cull_the_keyless_control():
    g = _graph(timestep=0, grace=25)
    _deposit(g, "conv", age=0)
    n = g.create_node(node_id="ctrl", metadata={"creation_mode": "conversational"})
    n.creation_time = 0
    for _ in range(25 + 40):
        g.step()
    assert "ctrl" not in g.nodes and "conv" in g.nodes


def test_int_heartbeat_stale_culls_with_ONE_warning_per_episode_and_a_completed_pass_recovers(caplog):
    clock = Clock()
    g = _graph(max_age=300, clock=clock)
    _deposit(g, "a")
    _deposit(g, "b")
    with caplog.at_level(logging.DEBUG, logger="neuro_foundation"):
        _hold(g)
        assert {"a", "b"} <= set(g.nodes)
        clock.t += 301
        _hold(g, ms=3)
        assert "a" not in g.nodes and "b" not in g.nodes
        ws = [r for r in caplog.records if r.name == "neuro_foundation" and r.levelno == logging.WARNING and "fair-chance window" in r.getMessage()]
        assert len(ws) == 1
        _deposit(g, "c")
        _hold(g)
        assert "c" not in g.nodes
        org.cc_update_probation(g)                                                      # a completed pass: recovery
        _deposit(g, "d")
        _hold(g)
        assert "d" in g.nodes
        clock.t += 301
        _hold(g, ms=2)
        assert "d" not in g.nodes
        assert len([r for r in caplog.records if r.name == "neuro_foundation" and r.levelno == logging.WARNING and "fair-chance window" in r.getMessage()]) == 2


def test_int_L3_a_poisoned_advancer_never_completes_so_after_the_limit_the_exemption_closes():
    clock = Clock()
    g = _graph(max_age=300, clock=clock)
    g.create_node(node_id="p1", metadata={"creation_mode": "conversational", "probation_remaining": "7"})
    g.create_node(node_id="p2", metadata={"creation_mode": "conversational"})
    g.create_synapse("p1", "p2", weight=0.2)                                            # BOUND, so the sweep never removes the poison
    g.nodes["p1"].creation_time = g.nodes["p2"].creation_time = 0
    _deposit(g, "late")
    for _ in range(6):
        clock.t += 61
        with pytest.raises(TypeError):
            org.cc_update_probation(g)
    _hold(g)
    assert "late" not in g.nodes and {"p1", "p2"} <= set(g.nodes)


def test_int_a_completing_advancer_keeps_the_exemption_on_across_many_pulses():
    clock = Clock()
    g = _graph(max_age=300, clock=clock)
    _deposit(g, "n")
    for _ in range(40):
        clock.t += 60
        org.cc_update_probation(g)
        _hold(g)
        assert "n" in g.nodes
    assert g.nodes["n"].metadata[STEPS] == WINDOW


def test_int_an_unarmed_host_ignores_the_heartbeat_entirely():
    clock = Clock()
    g = _graph(clock=clock)
    _deposit(g, "n")
    clock.t += 2 ** 40
    _hold(g)
    assert "n" in g.nodes


def test_int_an_UNREGISTERED_graph_driven_by_the_same_host_calls_sweeps_exactly_as_today():
    g = _graph(register=False)
    _deposit(g, "n")
    for _ in range(5):
        org.cc_update_probation(g)
    g._collect_orphan_nodes()
    assert "n" not in g.nodes                                                           # Syl's case: no registration, no exemption, no fields


# ---------------------------------------------------------------------------
# the wire
# ---------------------------------------------------------------------------

def test_banned_meta_carries_both_final_field_names_and_no_old_name():
    assert set(NEW_FIELDS) <= set(tex._BANNED_META)
    assert {"probation_remaining", "probation_total", "creation_time"} <= set(tex._BANNED_META)
    assert not [k for k in tex._BANNED_META if k.startswith("probation_steps") or k.startswith("probation_last")]
    assert tex._portable_metadata({k: 5 for k in NEW_FIELDS} | {"creation_mode": "conversational"}) == {"creation_mode": "conversational"}


def test_a_wire_round_trip_never_carries_the_fields_and_a_registered_receiver_restamps_with_its_own_clock(tmp_path):
    tc = importlib.import_module("tests.test_cc_topology_callosum")
    sg, sv, ids = tc._build_sender()
    for n in sg.nodes.values():
        n.metadata.update({STEPS: 2, LAST: 999, "probation_remaining": 2})
    path, _ = tc._export(sg, sv, tmp_path)
    for frame in tex.read_topology_frames(open(path, "rb").read()):
        for rec in frame.get("nodes") or ():
            for k in NEW_FIELDS + ("probation_remaining", "probation_total"):
                assert k not in rec["metadata"], "%s leaked onto the wire" % k
    rg, rv = tc._receiver()
    rg.timestep = 31
    rg.enable_fair_chance_window(WINDOW, excluded_creation_modes=org.PROBATION_UNADVANCED_CREATION_MODES)
    tc._merge(rg, rv, path, tmp_path)
    assert set(rg.nodes) == set(ids.values())
    for n in rg.nodes.values():
        assert n.metadata[STEPS] == WINDOW and n.metadata[LAST] == 31 != 999            # the RECEIVER's clock, never the sender's
    rg2, rv2 = tc._receiver()                                                           # an UNREGISTERED receiver: no window field at all
    tc._merge(rg2, rv2, path, tmp_path)
    assert all(STEPS not in n.metadata and LAST not in n.metadata for n in rg2.nodes.values())


def test_receiver_drops_forged_fair_chance_meta_on_structural_no_embedding_landing(tmp_path):
    tc = importlib.import_module("tests.test_cc_topology_callosum")
    forged_id = "cc:conv::forged-structural-window"
    payload = [
        {"kind": "header", "version": 1, "machine_id": "vps",
         "embedding_model": "test-model", "created": 0.0, "node_count": 1},
        {"kind": "batch", "seq": 1, "synapses": [], "hyperedges": [], "nodes": [
            {"id": forged_id, "content": "forged local-only state", "metadata": {
                "cc": True,
                "creation_mode": "conversational",
                STEPS: 1e308,
                LAST: 7,
            }},
        ]},
    ]
    path = str(tmp_path / "forged-no-embedding.conduit")
    with open(path, "wb") as fh:
        for frame in payload:
            body = msgpack.packb(frame, use_bin_type=True)
            fh.write(struct.pack(">I", len(body)) + body)

    rg, rv = tc._receiver()
    rg.enable_fair_chance_window(WINDOW, excluded_creation_modes=org.PROBATION_UNADVANCED_CREATION_MODES)
    stats = tc._merge(rg, rv, path, tmp_path, idle_steps=0)

    node = rg.nodes[forged_id]
    assert stats["absorbed_without_embedding_DEFECT"] == 1
    assert stats["banned_meta_dropped"] >= 1
    assert STEPS not in node.metadata and LAST not in node.metadata
    assert rg._in_fair_chance_window(node) is False

    node.creation_time = 0
    rg.timestep = rg.config.get("orphan_node_grace_period", 0) + 1
    rg._collect_orphan_nodes()
    assert forged_id not in rg.nodes


# ---------------------------------------------------------------------------
# static: no duplicated logic and no leftover name
# ---------------------------------------------------------------------------

_SKIP_DIRS = {".git", "tests", "data", "Defunct-Historical", "__pycache__", "docs", "handoffs"}
_REMOVED = ["fair_chance_window_open", "_fair_chance_window_open", "probation_advances", "_probation_advances", "probation_heartbeat_arm",
            "_probation_heartbeat_stamp", "_probation_heartbeat_fresh", "_probation_step_window_tick", "_probation_clock", "_probation_finite_number",
            "_PROBATION_HEARTBEAT", "_PROBATION_HEARTBEAT_LOCK", "probation_steps_remaining", "probation_last_timestep", "CC_PROBATION_STEP_WINDOW",
            "_CC_PROBATION_STEP_WINDOW", "CC_PROBATION_HEARTBEAT_CYCLES", "_read_probation_step_window_env"]


def _repo_py_files():
    for root, dirs, files in os.walk(_REPO):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


def _code_tokens(path):
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.NAME, tokenize.STRING):
                yield tok.string


def test_static_no_removed_name_survives_anywhere_in_the_tree_outside_comments_and_tests():
    hits = []
    for path in _repo_py_files():
        toks = list(_code_tokens(path))
        for name in _REMOVED:
            hits += [(os.path.relpath(path, _REPO), name) for s in toks if name in s]
    assert hits == [], hits[:4]            # (only changelog COMMENTS keep a superseded name, as history)


def test_static_the_window_field_names_live_only_in_the_canonical_module_and_the_export_list():
    owners = set()
    for path in _repo_py_files():
        if any(any(n in s for n in NEW_FIELDS) for s in _code_tokens(path)):
            owners.add(os.path.relpath(path, _REPO))
    assert owners == {"neuro_foundation.py", "cc_topology_export.py"}, owners          # not one copy of the step logic in the host


def test_static_the_organism_holds_no_window_logic_only_the_three_canonical_calls():
    tree = ast.parse(open(os.path.join(_REPO, "cc_ng_organism.py"), encoding="utf-8").read())
    top = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert not [n for n in top if "heartbeat" in n or "fair_chance" in n or "step_window" in n], "a window / heartbeat function lives in the host"
    adv = top["cc_update_probation"]
    getattrs = sorted(c.args[1].value for c in ast.walk(adv) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "getattr"
                      and len(c.args) >= 2 and isinstance(c.args[1], ast.Constant))
    assert getattrs == ["fair_chance_advance", "fair_chance_heartbeat_stamp"]
    dep = top["_cc_deposit_memory_node"]
    assert sorted(c.args[1].value for c in ast.walk(dep) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "getattr"
                  and len(c.args) >= 2 and isinstance(c.args[1], ast.Constant) and c.args[1].value.startswith("fair_chance")) == ["fair_chance_stamp"]
    called = [c.func.id for c in ast.walk(adv) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
    assert called.count("probation_population") == 1 and "_advance" in called and "_heartbeat" in called


def test_static_the_heartbeat_stamp_is_the_last_statement_before_return_outside_any_try_or_finally():
    tree = ast.parse(open(os.path.join(_REPO, "cc_ng_organism.py"), encoding="utf-8").read())
    adv = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cc_update_probation")
    body = next(n for n in adv.body if isinstance(n, ast.With)).body
    assert isinstance(body[-1], ast.Return)
    last = body[-2]
    assert isinstance(last, ast.If) and isinstance(last.test, ast.Compare)                      # `if _heartbeat is not None: _heartbeat()`
    assert isinstance(last.body[0], ast.Expr) and last.body[0].value.func.id == "_heartbeat"
    assert not [n for n in ast.walk(adv) if isinstance(n, ast.Try)]


def test_static_the_organism_reads_no_environment_for_the_window_and_names_no_window_knob():
    src = open(os.path.join(_REPO, "cc_ng_organism.py"), encoding="utf-8").read()
    toks = list(_code_tokens(os.path.join(_REPO, "cc_ng_organism.py")))
    assert not [s for s in toks if "NG_FAIR_CHANCE" in s]                                       # the HOST (the daemon) reads it; not this module
    assert "CC_CONV_PROBATION_PERIOD" in src                                                    # graduation keeps ITS knob here, untouched
