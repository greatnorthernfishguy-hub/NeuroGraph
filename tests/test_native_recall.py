# ---- Changelog ----
# [2026-11-21] DudeMan CC (Claude Fable 5.2, laptop) -- read-mode prime_and_propagate as one native call
#   (config native_recall_enabled; ng_tract.recall_sweep; SQ10, punch #1221 / #1201 option B).
# What: on ONE graph at one moment, the same recall twice: the existing Python sweep (key absent), then the native
#       call (key on). Both leave the mind as it was, so the second starts from the same state. Compared: the same
#       nodes fire at the same sweep step, the voltage at fire to rounding, was_predicted and the hop distance equal;
#       and the checkpoint bytes are unchanged by either. Run plain, with the stability arm, and with the recall
#       brakes on; a graph that is not native logs once and runs the Python sweep; write mode never takes the call.
# Why:  docs superpowers/specs/2026-11-20-recall-without-the-step-lock-design.md §3a / §5 bar 1. Declared difference
#       (Josh 2026-11-20 "Sure, sounds legit"): row order instead of Python set order, so the bar is "same nodes at
#       the same sweep step, voltages to rounding, the mind byte-unchanged", not the order of the entries.
# How:  tests.test_hyperedge_store_h2 built() start graphs, driven a few steps first so that voltages, refractory
#       counts, hyperedges and open predictions are live when the recall runs.
# -------------------
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import neuro_foundation as NF  # noqa: E402
from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s18 as S18        # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(NF.ng_tract, "recall_sweep"), reason="installed ng_tract has no recall_sweep")

KEY = "native_recall_enabled"


def _warm(seed, cfg, steps=12):
    g = S.restore_as(NF, "native", H2.built(seed))
    g.config.update(cfg)
    side = H2.Side(g, seed)
    for i in range(steps):
        def op(g, i=i):
            rng = random.Random(seed * 100 + i)
            ids = sorted(g.nodes.keys())
            for nid in rng.sample(ids, min(12, len(ids))):
                g.nodes[nid].voltage += 1.5
            return g.step()
        side(op)
    return g


def _rows(res):
    return sorted((e.firing_step, e.node_id, e.voltage_at_fire, e.was_predicted, e.source_distance)
                  for e in res.fired_entries)


def _both(g, ids, currents, steps, monkeypatch):
    """(python rows, native rows); asserts the native call was taken and that neither changed the checkpoint."""
    calls = []
    orig = NF.ng_tract.recall_sweep
    monkeypatch.setattr(NF.ng_tract, "recall_sweep", lambda *a: (calls.append(1), orig(*a))[1])
    before = S18.ckpt(g)
    g.config.pop(KEY, None)
    py = g.prime_and_propagate(ids, currents, steps=steps)
    assert not calls and S18.ckpt(g) == before
    g.config[KEY] = True
    rs = g.prime_and_propagate(ids, currents, steps=steps)
    g.config.pop(KEY)
    assert calls == [1] and S18.ckpt(g) == before
    assert (py.steps_run, py.nodes_primed) == (rs.steps_run, rs.nodes_primed)
    return _rows(py), _rows(rs)


def _same(py, rs, rel=1e-9):
    assert [(k, n) for k, n, *_ in py] == [(k, n) for k, n, *_ in rs]
    for (k, n, va, pa, da), (_, _, vb, pb, db) in zip(py, rs):
        assert abs(va - vb) <= rel * max(1.0, abs(va), abs(vb)), (k, n, va, vb)
        assert (pa, da) == (pb, db), (k, n)


ARMS = {"plain": {},
        "arm": dict(S18.ARM),
        "brakes": dict(S18.ARM, recall_brakes_enabled=True),
        "receiver": dict(S18.ARM, **S18.RPE, **S18.ON, recall_brakes_enabled=True)}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_the_same_nodes_fire_at_the_same_sweep_step(seed, arm, monkeypatch):
    g = _warm(seed, ARMS[arm])
    rng = random.Random(seed)
    ids_all = sorted(g.nodes.keys())
    fired = later = 0
    for trial in range(6):
        ids = rng.sample(ids_all, min(3 + trial, len(ids_all)))
        cur = [rng.uniform(0.8, 2.5) for _ in ids]
        py, rs = _both(g, ids, cur, 3 + trial % 3, monkeypatch)
        _same(py, rs)
        fired += len(rs)
        later += sum(1 for k, *_ in rs if k > 0)
    assert fired > 0 and later > 0          # not vacuous: something fired, and not only the primes


def test_the_brakes_arm_does_brake():
    g = _warm(1, ARMS["brakes"])
    assert g._recall_brakes(g._stability_params()) is not None


@pytest.mark.parametrize("steps", [0, 1])
def test_few_steps_and_an_unknown_prime(steps, monkeypatch):
    g = _warm(0, ARMS["arm"])
    ids = sorted(g.nodes.keys())[:4]
    py, rs = _both(g, ids, [2.0] * 4, steps, monkeypatch)
    _same(py, rs)
    assert (len(rs) == 0) == (steps == 0)
    g.config[KEY] = True                    # a prime that is not a node is skipped by the native call
    res = g.prime_and_propagate(ids + ["no-such-node"], [2.0] * 5, steps=2)
    assert {e.node_id for e in res.fired_entries} >= set(ids) and "no-such-node" not in {e.node_id for e in res.fired_entries}


def test_no_primes_and_write_mode_do_not_take_the_call(monkeypatch):
    g = _warm(0, ARMS["arm"])
    g.config[KEY] = True
    calls = []
    orig = NF.ng_tract.recall_sweep
    monkeypatch.setattr(NF.ng_tract, "recall_sweep", lambda *a: (calls.append(1), orig(*a))[1])
    assert g.prime_and_propagate([], [], steps=3).nodes_primed == 0
    g.prime_and_propagate(sorted(g.nodes.keys())[:3], [2.0] * 3, steps=2, write_mode=True)
    assert not calls


@pytest.mark.parametrize("kind", ["dict", "nodes"])
def test_a_graph_that_is_not_native_logs_once_and_runs_the_python_sweep(kind, caplog):
    g = S.restore_as(NF, kind, H2.built(0))
    g.config[KEY] = True
    ids = sorted(g.nodes.keys())[:3]
    with caplog.at_level("ERROR"):
        a = g.prime_and_propagate(ids, [2.0] * 3, steps=2)
        b = g.prime_and_propagate(ids, [2.0] * 3, steps=2)
    assert _rows(a) == _rows(b) and len(a.fired_entries) > 0
    assert sum(KEY in r.message for r in caplog.records) == 1
