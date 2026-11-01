# ---- Changelog ----
# [2026-11-01] DudeMan CC (Claude Opus 5.5, laptop; #1202 receiver-side predictions; Josh 2026-11-01 "Yeah, the design
#   looks good") — S18: a neuron is predicted when pending input charges it partway but cannot fire it alone
# What: (1) the switch absent / explicitly False == the synced copy line (1d0a1a0's module) exactly (final checkpoint
#       BYTES); keys not in DEFAULT_CONFIG; bad params log ERROR once and change nothing. (2) the predictive state:
#       partway-not-over, graded, inhibition nets, one state per neuron, capacity by strength, pre-charge never fires the
#       target, each neuron's own rate. (3) credit by share, one-branch gated chains, checkpoints the base loader reads.
# Why:  docs/superpowers/specs/2026-11-01-receiver-side-prediction-design.md; plan 2026-11-01-receiver-side-prediction.md.
# How:  oracle via `git show 1d0a1a0:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S7b helpers;
#       small hand-built graphs for the mechanism.
# -------------------
"""#1202 round S18: receiver-side, graded predictions."""
import logging
import os
import random

import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s7b as S7B        # noqa: E402

NF = H2.NF
BASE_REV = os.environ.get("NG_S18_BASE_REV", "1d0a1a0")
CMOD = H2._load(BASE_REV, "nf_s17_commit_s18")
S18_SWITCHES = ("prediction_receiver_enabled",)
S18_PARAMS = ("prediction_receiver_rho0", "prediction_receiver_k", "prediction_receiver_eta",
              "prediction_receiver_rho_min", "prediction_receiver_ema_alpha", "prediction_receiver_chain_floor",
              "prediction_receiver_margin")
ARM = S7B.S7B_ARM
RPE = {"three_factor_enabled": True, "reward_rpe_enabled": True}
ON = {"prediction_receiver_enabled": True}
ckpt = S.ckpt
restore_as = S.restore_as


# ---------------------------------------------------------------------------
# (1) the switch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_base_branch_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(CMOD, kind, start), restore_as(NF, kind, start), seed)
    t.op("keys", lambda g: g.config.update(dict(ARM, **RPE) if kind == "native" else RPE))
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in S18_SWITCHES})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S18_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


def test_keys_not_in_default_config():
    for k in S18_SWITCHES + S18_PARAMS:
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(CMOD.DEFAULT_CONFIG.items())


def test_params_default_when_on():
    g = NF.Graph(dict(ON))
    assert g._stab_receiver() == (0.3, 1.0, 0.002, 0.1, 0.01, 0.05, 0.05)
    assert NF.Graph()._stab_receiver() is None


@pytest.mark.parametrize("bad", [dict(prediction_receiver_rho0=1.5), dict(prediction_receiver_rho_min=0.5),
                                 dict(prediction_receiver_k="many"), dict(prediction_receiver_eta=0)])
def test_bad_params_error_once_and_change_nothing(bad, caplog):
    start = H2.built(0)
    t = S.Pair(restore_as(CMOD, "native", start), restore_as(NF, "native", start), 0)
    t.op("keys", lambda g: g.config.update(dict(ARM, **RPE)))
    t.b.g.config.update(dict(ON, **bad))
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 10, sleep_every=True)
    errs = [r for r in caplog.records if r.levelno == logging.ERROR and "prediction_receiver_enabled" in r.getMessage()]
    assert len(errs) == 1, [r.getMessage() for r in errs]
    for k in list(ON) + list(bad):
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)


# ---------------------------------------------------------------------------
# (2) the predictive state
# ---------------------------------------------------------------------------

def _rg(n_src=3, w=0.1, theta=0.8, extra=None, **cfg):
    """sources s0.. -> T (weight w each); T's threshold theta; returns (graph, params)."""
    g = NF.Graph(dict(ON, **cfg))
    g.create_node(node_id="T")
    g.nodes["T"].threshold = theta
    for i in range(n_src):
        g.create_node(node_id=f"s{i}")
        g.create_synapse(f"s{i}", "T", weight=w)
    for nid, kw in (extra or {}).items():
        g.create_node(node_id=nid, **kw)
    return g, g._stab_receiver()


def _pend(g, target, current, dt=1):
    g._delay_buffer.setdefault(g.timestep + dt, []).append((target, current))


def _fire(g, params, srcs, currents):
    for s, (tgt, c) in zip(srcs, currents):
        _pend(g, tgt, c)
    g._receiver_predictions(list(srcs), params)


def _preds_on(g, target):
    return [p for p in g.active_predictions.values() if p.target_node_id == target]


def test_partway_drive_registers_graded_prediction():
    g, p = _rg()
    _fire(g, p, ["s0", "s1", "s2"], [("T", 0.1)] * 3)
    (pred,) = _preds_on(g, "T")
    assert pred.strength == pytest.approx(0.3 / 0.8)
    assert pred.chain_depth == 0 and pred.drive == pytest.approx(0.3) and pred.rho == pytest.approx(0.3, abs=0.003)   # rho moves by eta at every visit, the first too
    assert sum(s for _, s in pred.contributors) == pytest.approx(1.0) and len(pred.contributors) == 3
    assert pred.source_node_id in {"s0", "s1", "s2"}
    assert pred.expires_at == g.timestep + g.config["prediction_window"]


def test_drive_that_fires_alone_registers_nothing():
    g, p = _rg(n_src=1, w=4.6)
    _fire(g, p, ["s0"], [("T", 4.6)])
    assert _preds_on(g, "T") == []


def test_below_rho_registers_nothing():
    g, p = _rg(n_src=1, w=0.05)
    _fire(g, p, ["s0"], [("T", 0.05)])   # 0.0625 of the way < rho0 0.3
    assert _preds_on(g, "T") == []


def test_inhibitory_only_drive_registers_nothing():
    g, p = _rg(n_src=0, extra={"i0": {"is_inhibitory": True}})
    g.create_synapse("i0", "T", weight=0.5)
    _fire(g, p, ["i0"], [("T", -0.5)])
    assert _preds_on(g, "T") == []


def test_inhibition_nets_the_drive():
    g, p = _rg(n_src=3, w=0.1, theta=0.6, extra={"i0": {"is_inhibitory": True}})   # net 0.2 of 0.6 > rho
    g.create_synapse("i0", "T", weight=0.1)
    _fire(g, p, ["s0", "s1", "s2", "i0"], [("T", 0.1)] * 3 + [("T", -0.1)])
    (pred,) = _preds_on(g, "T")
    assert pred.drive == pytest.approx(0.2)
    assert {s for s, _ in pred.contributors} and all(sh > 0 for _, sh in pred.contributors)


def test_refractory_target_registers_nothing():
    g, p = _rg()
    g.nodes["T"].refractory_remaining = 2
    _fire(g, p, ["s0", "s1", "s2"], [("T", 0.1)] * 3)
    assert _preds_on(g, "T") == []


def test_one_state_per_neuron_merges():
    g, p = _rg(n_src=6)
    _fire(g, p, ["s0", "s1", "s2"], [("T", 0.1)] * 3)
    (first,) = _preds_on(g, "T")
    exp, s0 = first.expires_at, first.strength
    g.timestep += 1
    g._delay_buffer.pop(g.timestep, None)   # step 1's input delivered (not pending any more)
    g.nodes["T"].voltage = 0.1
    _fire(g, p, ["s3", "s4", "s5", "s3"], [("T", 0.1)] * 4)
    (pred,) = _preds_on(g, "T")
    assert pred.expires_at == exp and pred.strength >= s0
    assert g._recv_stats["merged"] == 1
    assert len({sid for sid, _ in pred.contributors}) == 6


def test_capacity_keeps_strongest():
    g, p = _rg(n_src=0, prediction_max_active=2)
    for i, w in enumerate((0.3, 0.5, 0.4)):
        g.create_node(node_id=f"T{i}")
        g.nodes[f"T{i}"].threshold = 0.8
        g.create_node(node_id=f"a{i}")
        g.create_synapse(f"a{i}", f"T{i}", weight=w)
    _fire(g, p, ["a0", "a1", "a2"], [("T0", 0.3), ("T1", 0.5), ("T2", 0.4)])
    assert {pr.target_node_id for pr in g.active_predictions.values()} == {"T1", "T2"}
    assert g._recv_stats["dropped"] == 1


def test_precharge_never_fires_target():
    rng = random.Random(7)
    for _ in range(50):
        theta, d, v = rng.uniform(0.5, 1.2), rng.uniform(0.0, 1.0), rng.uniform(-0.2, 0.4)
        g, p = _rg(n_src=1, w=d, theta=theta, prediction_pre_charge_factor=1.0)
        g.nodes["T"].voltage = v
        _fire(g, p, ["s0"], [("T", d)])
        if _preds_on(g, "T"):
            assert g.nodes["T"].voltage + d < theta


def test_rho_rises_when_overpredicted_falls_when_under():
    g, p = _rg()
    g.nodes["T"].firing_rate_ema = 0.0
    st = g._receiver_state()
    for _ in range(40):
        g.active_predictions.clear()
        g.timestep += 1
        _fire(g, p, ["s0", "s1", "s2"], [("T", 0.1)] * 3)
    rho_hi = float(st["rho"][st["idx"]["T"]])
    assert rho_hi > 0.3
    g.nodes["T"].firing_rate_ema = 0.5
    for _ in range(400):
        g.active_predictions.clear()
        g.timestep += 1
        _fire(g, p, ["s0"], [("T", 0.01)])   # candidate, never predicted
    assert float(st["rho"][st["idx"]["T"]]) == pytest.approx(0.1)   # down to rho_min, not below
