# ---- Changelog ----
# [2026-11-05] DudeMan CC (#1212; Josh 2026-11-05 "write it up ... You have my go") — CREATE
# What: the surprise slot (config surprise_slot_enabled): (1) off / keys absent = e2f93d4 exactly (whole-run checkpoint
#       bytes, tally, stats); (2) on: the slot, reach and top-k rules on hand-built graphs -- a node outside the slot or
#       outside reach never appears, the rarer of two eligible nodes wins, an uncovered slot gives none, the result goes
#       to the tally as today; (3) on, a whole run: deterministic, fewer surprise touches than off, the counters move;
#       (4) on without the native stores raises.
# Why:  spec docs superpowers/specs/2026-11-05-surprise-alternatives-design.md §4 (readiness bar, tests).
# How:  the test_sprout_1050 / test_sleep_p1 builders and workload; the base engine is loaded from git (e2f93d4).
# -------------------
import importlib.util
import os
import subprocess
import sys
import tempfile

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import neuro_foundation as NF  # noqa: E402  (the branch)
from tests import test_sleep_p1 as P1  # noqa: E402
from tests import test_sprout_1050 as T  # noqa: E402

BASE_REV = os.environ.get("NG_SURPRISE_SLOT_BASE_REV", "e2f93d4")
SLOT = hasattr(NF.ng_tract.SynapseStore, "surprise_alternatives")
needs_slot = pytest.mark.skipif(not SLOT, reason="installed ng_tract has no surprise_alternatives")
ON = {"surprise_slot_enabled": True, "surprise_slot_tol": 1, "surprise_topk": 4}


def _load_base():
    src = subprocess.run(["git", "-C", _REPO, "show", f"{BASE_REV}:neuro_foundation.py"],
                         check=True, capture_output=True, text=True).stdout
    d = tempfile.mkdtemp(prefix="surprise_slot_base_")
    p = os.path.join(d, "nf_base_surprise_slot.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nf_base_surprise_slot", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nf_base_surprise_slot"] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base()


def _restore(mod, mode, b):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        with open(p, "wb") as f:
            f.write(b)
        g = mod.Graph(native_node_store=(mode == "on"))
        g.restore(p)
        return g


def _run(mod, mode, start, act, seed, cfg):
    g = _restore(mod, mode, start)
    g.config.update(cfg)
    out = P1.seeded(seed, lambda rng: act(g, rng), uuid_stream=1)
    return P1.ckpt_bytes(g), out, g


CFG = dict(T.TALLY, **T.DISUSE, prediction_threshold=0.5)   # the workload's weights predict


# ---------------------------------------------------------------------------
# (1) off / absent: e2f93d4 exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", P1.SEEDS[:2])
@pytest.mark.parametrize("mode", ["off", "on"])
@pytest.mark.parametrize("cfg", [CFG, dict(CFG, surprise_slot_enabled=False, surprise_topk=1)])
def test_off_is_the_base_exactly(mode, seed, cfg):
    start = P1.built_bytes(seed)
    act = T.tally_workload(30)
    a, (la, _), ga = _run(NF, mode, start, act, seed, cfg)
    b, (lb, _), gb = _run(BASE, mode, start, act, seed, CFG)
    if "surprise_slot_enabled" not in cfg:
        P1.compare(la, lb, [], [], a, b)     # keys absent: checkpoint bytes too (an explicit key is saved in config)
    assert ga._total_predictions_errors == gb._total_predictions_errors > 0
    assert ga._sprout_tally_stats == gb._sprout_tally_stats
    assert T.tstate(ga) == T.tstate(gb)
    assert sorted((s.pre_node_id, s.post_node_id) for s in ga.synapses.values()) == \
        sorted((s.pre_node_id, s.post_node_id) for s in gb.synapses.values())
    assert ga._novel_sequence_log == gb._novel_sequence_log


# ---------------------------------------------------------------------------
# (2) on: the rules
# ---------------------------------------------------------------------------

def _g(**cfg):
    g = NF.Graph(dict(T.TALLY, **ON, **cfg), native_node_store=True)
    for nid in ("A", "B", "X", "X2", "C1", "C2", "H", "Y", "Z"):
        g.create_node(node_id=nid)
    for pre, post in (("A", "X"), ("X", "C1"), ("X", "H"), ("X2", "B"), ("X2", "Y"), ("Z", "C2"), ("A", "B")):
        g.create_synapse(pre, post, weight=0.5)
    return g


def _pred(created_at, depth=0):
    return NF.Prediction(source_node_id="A", target_node_id="B", strength=0.8, confidence=0.5,
                         created_at=created_at, expires_at=created_at + 10, chain_depth=depth)


def _ring(g, rows, keep=11):
    for t, ids in rows:
        g.synapses.surprise_record_step(t, ids, keep)


@needs_slot
def test_only_step_fired_nodes_at_the_due_moment_within_reach():
    g = _g()
    for nid in ("C1", "C2", "H", "Y"):
        g.nodes[nid].firing_rate_ema = 0.01
    # created at 100: B due at 101 (+-1, never before 101). C2 fired at the due moment but is out of reach; Y (a
    # sibling of B) and C1 (A -> X -> C1) are in reach; H fired only at 104 (outside the slot).
    _ring(g, [(100, ["C1"]), (101, ["C1", "C2", "Y"]), (102, []), (103, []), (104, ["H"])])
    assert g._surprise_slot_alternatives(_pred(100)) == {"C1", "Y"}
    # a depth-2 chain prediction is due at 103 (102..104): only H
    assert g._surprise_slot_alternatives(_pred(100, depth=2)) == {"H"}


@needs_slot
def test_the_rarer_eligible_node_wins_the_top_k():
    g = _g(surprise_topk=1)
    g.nodes["C1"].firing_rate_ema = 0.02
    g.nodes["H"].firing_rate_ema = 0.40      # a hub that fires all the time
    g.nodes["Y"].firing_rate_ema = 0.005
    _ring(g, [(100, []), (101, ["C1", "H", "Y"]), (102, [])])
    assert g._surprise_slot_alternatives(_pred(100)) == {"Y"}
    g.nodes["Y"].firing_rate_ema = 0.9
    assert g._surprise_slot_alternatives(_pred(100)) == {"C1"}


@needs_slot
def test_an_uncovered_slot_gives_none_never_the_whole_window():
    g = _g()
    _ring(g, [(105, ["C1", "Y"])])                      # e.g. right after a restore: the ring starts at 105
    assert g._surprise_slot_alternatives(_pred(100)) == set()
    st = g.synapses.surprise_stats()
    assert st["uncovered"] == 1 and st["calls"] == 1


@needs_slot
def test_the_ring_keeps_only_the_window_and_clear_empties_it():
    g = _g()
    _ring(g, [(t, ["C1"]) for t in range(100, 130)], keep=11)
    assert g.synapses.surprise_stats()["ring_steps"] == 11
    g.synapses.surprise_ring_clear()
    assert g.synapses.surprise_stats()["ring_steps"] == 0


@needs_slot
def test_the_alternatives_feed_the_tally_as_today():
    g = _g()
    for nid in ("C1", "Y"):
        g.nodes[nid].firing_rate_ema = 0.01
    _ring(g, [(100, []), (101, ["C1", "Y", "C2"]), (102, [])])
    g.timestep = 111
    g._surprise_exploration(_pred(100), {"C1", "C2", "Y", "H", "Z"})   # recent_fired is ignored under the slot
    held = dict(g.cofire_tally_state())
    assert set(held) == {"C1", "Y"} and held["C1"][0][0] == "A"
    assert g._sprout_tally_stats["surprise_calls"] == 1


# ---------------------------------------------------------------------------
# (3) on: a whole run
# ---------------------------------------------------------------------------

@needs_slot
@pytest.mark.parametrize("seed", P1.SEEDS[:2])
def test_whole_run_on_is_deterministic_and_touches_less(seed):
    start = P1.built_bytes(seed)
    act = T.tally_workload(30)
    a, (la, _), ga = _run(NF, "on", start, act, seed, dict(CFG, **ON))
    b, (lb, _), gb = _run(NF, "on", start, act, seed, dict(CFG, **ON))
    P1.compare(la, lb, [], [], a, b)
    c, (lc, _), gc = _run(NF, "on", start, act, seed, CFG)
    touches_on = sum(1 for _ in ga.cofire_tally_state())
    touches_off = sum(1 for _ in gc.cofire_tally_state())
    st = ga.synapses.surprise_stats()
    assert st["calls"] > 0 and st["ring_steps"] > 0
    assert gc.synapses.surprise_stats()["calls"] == 0                  # off never asks the slot
    assert touches_on < touches_off


# ---------------------------------------------------------------------------
# (4) on without the native stores
# ---------------------------------------------------------------------------

def test_on_without_the_native_node_store_raises():
    g = NF.Graph(dict(T.TALLY, **ON), native_node_store=False)
    g.create_node(node_id="A")
    with pytest.raises(RuntimeError, match="native NodeStore"):
        g._surprise_slot_store()
