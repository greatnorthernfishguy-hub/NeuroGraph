# ---- Changelog ----
# [2026-11-20] DudeMan CC (Claude Fable 5.2, laptop) -- native tick M2 (punch #1221): the receiver's registration and
#   chains in the crate (ng_tract.receiver_step), against the engine before that change.
# What: twin runs with the S18 receiver on: the engine at bb440b8 (registration and the chain hop loop in Python, loaded
#       from git) beside this branch's engine (one native call), same wheel, same seeded id / random streams. Compared
#       after every operation: results, every event (prediction_generated included), the whole state; at the end the
#       receiver columns and the checkpoint bytes. Cases: the plain run, with the corrected DiffPC (S19), with a small
#       table cap (drops), and a long run without the sleep (chains and merges build up).
# Why:  go-live register 2026-11-19 / 11-20 (the receiver pass is 11 % of a live-size step, 74 % of it in the two Python
#       loops); spec 2026-11-12-lightweight-substrate-design §3 parity bar: bit-identical against the pre-port engine
#       loaded from git, phase by phase.
# How:  tests.test_stability_s.Pair / drive (the H2 workload), as tests/test_native_tick_m1.py case (1).
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

HAVE_STEP = hasattr(NF.ng_tract, "receiver_step")
pytestmark = pytest.mark.skipif(not HAVE_STEP, reason="installed ng_tract has no receiver_step")

PREPORT_REV = os.environ.get("NG_NATIVE_TICK_M2_PREPORT_REV", "bb440b8")
PRE = H2._load(PREPORT_REV, "nf_native_tick_m2_preport") if HAVE_STEP else None


def _pair(seed, extra=None, heard=False):
    start = H2.built(seed)
    t = S.Pair(S18.restore_as(PRE, "native", start), S18.restore_as(NF, "native", start), seed)
    if heard:   # the Pair compares both sides' event logs after every operation
        for side in (t.a, t.b):
            side.g.register_event_handler("prediction_generated", lambda _log=side.log, **kw: _log.append(
                ("prediction_generated", tuple(sorted(kw.items())))))
    t.op("keys", lambda g: g.config.update(dict(S18.ARM, **S18.RPE, **S18.ON, **(extra or {}))))
    return t


def _same_end(t):
    a, b = t.a.g, t.b.g
    assert a._recv_stats["registered"] == b._recv_stats["registered"] > 0
    for k in ("merged", "dropped", "chains", "read_skipped"):
        assert a._recv_stats[k] == b._recv_stats[k], k
    assert a._total_predictions_made == b._total_predictions_made
    assert b._total_predictions_confirmed + b._total_predictions_errors > 0          # not vacuous
    assert S18._state_rows(a) == S18._state_rows(b)
    assert [(r.prediction_id, r.source_node_id, r.target_node_id, r.strength, r.confidence, r.chain_depth,
             r.pre_charge_applied, r.contributors) for r in a.active_predictions.values()] == \
           [(r.prediction_id, r.source_node_id, r.target_node_id, r.strength, r.confidence, r.chain_depth,
             r.pre_charge_applied, r.contributors) for r in b.active_predictions.values()]
    assert S18.ckpt(a) == S18.ckpt(b)


def test_the_oracle_is_the_python_registration_and_the_candidate_is_not():
    assert hasattr(PRE.Graph, "_receiver_chains") and not hasattr(NF.Graph, "_receiver_chains")


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("s19", [False, True])
def test_receiver_whole_run_equals_the_python_registration(seed, s19):
    t = _pair(seed, {"diffpc_receiver_enabled": True} if s19 else None)
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    _same_end(t)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("cap", [1, 2, 3])
def test_a_full_table_drops_the_same_predictions(seed, cap):
    t = _pair(seed, {"prediction_receiver_max_active": cap})
    S.drive(t, random.Random(seed), 25)
    assert t.b.g._recv_stats["dropped"] > 0
    _same_end(t)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_a_long_run_with_merges_and_chains(seed):
    t = _pair(seed, {"prediction_receiver_chain_floor": 0.0, "prediction_receiver_rho0": 0.12,
                     "prediction_receiver_rho_min": 0.1})
    S.drive(t, random.Random(seed), 80)
    _same_end(t)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_listeners_hear_the_same_predictions_in_the_same_order(seed):
    """`prediction_generated` is now emitted after the pass instead of inside it: same rows, same order per step."""
    t = _pair(seed, {"prediction_receiver_chain_floor": 0.0, "prediction_receiver_rho0": 0.12,
                     "prediction_receiver_rho_min": 0.1}, heard=True)
    heard = []
    t.b.g.register_event_handler("prediction_generated", lambda **kw: heard.append(kw["chain_depth"]))
    S.drive(t, random.Random(seed), 40)
    assert heard, "no prediction was generated"
    _same_end(t)


def test_the_cases_above_reach_merges_and_chains():
    """The parity cases are only worth something if the merge and chain paths ran: one run that must show both."""
    tot = {"merged": 0, "chains": 0}
    for seed in range(5):
        t = _pair(seed, {"prediction_receiver_chain_floor": 0.0, "prediction_receiver_rho0": 0.12,
                         "prediction_receiver_rho_min": 0.1})
        S.drive(t, random.Random(seed), 80)
        for k in tot:
            tot[k] += t.b.g._recv_stats[k]
    assert tot["merged"] > 0, tot
    assert tot["chains"] > 0, tot


def test_the_pass_is_one_native_call(monkeypatch):
    g = S18._native_graph(1)
    calls = []
    orig = NF.ng_tract.receiver_step
    monkeypatch.setattr(NF.ng_tract, "receiver_step", lambda *a: (calls.append(1), orig(*a))[1])
    for name in ("receiver_candidates", "receiver_chain_hop"):
        monkeypatch.setattr(NF.ng_tract, name, lambda *a, _n=name: pytest.fail(f"{_n} called by the engine"))
    g._receiver_predictions(list(g.nodes.keys())[:30], g._stab_receiver())
    assert calls == [1]


def test_a_wheel_without_receiver_step_keeps_the_receiver_off(monkeypatch, caplog):
    g = S18._native_graph(1)
    monkeypatch.delattr(NF.ng_tract, "receiver_step")
    with caplog.at_level("ERROR"):
        assert g._stab_receiver() is None
    assert any("receiver_step" in r.getMessage() for r in caplog.records)
