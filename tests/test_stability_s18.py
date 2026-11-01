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
    g, p = _rg(n_src=0, prediction_receiver_max_active=2)
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


# ---------------------------------------------------------------------------
# (3) credit by share, chains, checkpoints, a whole-graph run
# ---------------------------------------------------------------------------

def _weights(g, srcs, tgt="T"):
    return [g._find_synapse(s, tgt).weight for s in srcs]


def test_confirm_splits_bonus_by_share():
    g, p = _rg(n_src=0, three_factor_enabled=False, prediction_confirm_bonus=0.1)
    for i, w in enumerate((0.2, 0.1, 0.1)):
        g.create_node(node_id=f"s{i}")
        g.create_synapse(f"s{i}", "T", weight=w)
    _fire(g, p, ["s0", "s1", "s2"], [("T", 0.2), ("T", 0.1), ("T", 0.1)])
    (pred,) = _preds_on(g, "T")
    shares = dict(pred.contributors)
    assert sorted(shares.values()) == pytest.approx([0.25, 0.25, 0.5])
    before = _weights(g, ["s0", "s1", "s2"])
    g._evaluate_predictions({"T"})
    after = _weights(g, ["s0", "s1", "s2"])
    want = [b + 0.1 * pred.confidence * sh for b, sh in zip(before, (0.5, 0.25, 0.25))]
    assert after == pytest.approx(want)
    assert _preds_on(g, "T") == []


def test_error_splits_penalty_by_share_and_updates_own_record():
    g, p = _rg(three_factor_enabled=False, prediction_error_penalty=0.05)
    _fire(g, p, ["s0", "s1", "s2"], [("T", 0.1)] * 3)
    (pred,) = _preds_on(g, "T")
    st = g._receiver_state()
    c0 = float(st["conf_ema"][st["idx"]["T"]])
    g.timestep = pred.expires_at + 1
    g._cleanup_predictions()
    assert _weights(g, ["s0", "s1", "s2"]) == pytest.approx([0.1 - 0.05 * pred.confidence / 3] * 3)
    assert float(st["conf_ema"][st["idx"]["T"]]) < c0


def test_shares_sum_to_one_excitatory_only():
    g, p = _rg(n_src=2, w=0.2, theta=0.6, extra={"i0": {"is_inhibitory": True}})
    g.create_synapse("i0", "T", weight=0.1)
    _fire(g, p, ["s0", "s1", "i0"], [("T", 0.2), ("T", 0.2), ("T", -0.1)])
    (pred,) = _preds_on(g, "T")
    pre = {g.synapses.get(sid).pre_node_id for sid, _ in pred.contributors}
    assert pre == {"s0", "s1"} and sum(s for _, s in pred.contributors) == pytest.approx(1.0)


def test_contributor_synapse_pruned_midflight():
    g, p = _rg(three_factor_enabled=False)
    _fire(g, p, ["s0", "s1", "s2"], [("T", 0.1)] * 3)
    (pred,) = _preds_on(g, "T")
    g.remove_synapse(pred.contributors[0][0])
    g.timestep = pred.expires_at + 1
    g._cleanup_predictions()   # no exception; the other two still get the penalty
    assert _preds_on(g, "T") == []


def _chain_graph(w1=2.0, w2=1.0, **cfg):
    g, p = _rg(**cfg)
    for u in ("U1", "U2"):
        g.create_node(node_id=u)
        g.nodes[u].threshold = 0.8
    g.create_synapse("T", "U1", weight=w1)
    g.create_synapse("T", "U2", weight=w2)
    return g, g._stab_receiver()


def test_chain_follows_strongest_branch_only():
    g, p = _chain_graph()
    _fire(g, p, ["s0", "s1", "s2"], [("T", 0.1)] * 3)
    deep = [pr for pr in g.active_predictions.values() if pr.chain_depth == 1]
    assert [pr.target_node_id for pr in deep] == ["U1"]
    (head,) = _preds_on(g, "T")
    assert deep[0].source_node_id == head.source_node_id   # credited to the chain's origin
    assert g._recv_stats["chains"] == 1


def test_chain_stops_below_floor():
    g, p = _chain_graph(prediction_receiver_chain_floor=0.9)
    _fire(g, p, ["s0", "s1", "s2"], [("T", 0.1)] * 3)
    assert [pr for pr in g.active_predictions.values() if pr.chain_depth > 0] == []


@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
def test_receiver_on_runs_the_drive_harness(kind):
    start = H2.built(0)
    t = S.Pair(restore_as(NF, kind, start), restore_as(NF, kind, start), 0)
    t.op("keys", lambda g: g.config.update(dict(ARM, **RPE, **ON) if kind == "native" else dict(RPE, **ON)))
    S.drive(t, random.Random(0), 25, sleep_every=True)
    assert t.b.g._recv_stats["registered"] > 0
    assert ckpt(t.a.g) == ckpt(t.b.g)   # deterministic: two identical runs agree


@pytest.mark.parametrize("kind", ["dict", "native"])
def test_checkpoint_roundtrip_base_loader(kind):
    start = H2.built(1)
    t = S.Pair(restore_as(NF, kind, start), restore_as(NF, kind, start), 1)
    t.op("keys", lambda g: g.config.update(dict(ARM, **RPE, **ON) if kind == "native" else dict(RPE, **ON)))
    S.drive(t, random.Random(1), 20)
    g = t.b.g
    sid, syn = next(iter(g.synapses.items()))
    pr = NF.Prediction(target_node_id=syn.post_node_id, strength=0.4, confidence=0.5, created_at=g.timestep,
                       expires_at=g.timestep + 10, drive=0.3, rho=0.3)
    NF.Graph._receiver_set_contributors(pr, {sid: (syn.pre_node_id, syn.weight)})
    g.active_predictions[pr.prediction_id] = pr   # a pending receiver prediction is in the checkpoint
    b = ckpt(t.b.g)
    restore_as(CMOD, kind, b)                       # the base loader reads it
    g2 = restore_as(NF, kind, b)
    g2.config.update(ON)
    for _ in range(5):
        g2.step()


# ---------------------------------------------------------------------------
# (4) final-review fixes (Grok 4.7, 2026-11-01) + the S18short findings
# ---------------------------------------------------------------------------

def test_mixed_sign_ticks_that_fire_alone_register_nothing():
    """Review C1: +0.55 at t+1, -0.20 at t+3: the net 0.35 is partway, but the first tick alone fires T (0.97*0.30+0.55)."""
    g, p = _rg(n_src=1, w=0.55, extra={"i0": {"is_inhibitory": True}})
    g.create_synapse("i0", "T", weight=0.2)
    g.nodes["T"].voltage = 0.30
    _pend(g, "T", 0.55, dt=1)
    _pend(g, "T", -0.20, dt=3)
    g._receiver_predictions(["s0", "i0"], p)
    assert _preds_on(g, "T") == []


def test_spread_ticks_that_cannot_fire_alone_register():
    g, p = _rg(n_src=2, w=0.2)
    _pend(g, "T", 0.2, dt=1)
    _pend(g, "T", 0.2, dt=2)
    g._receiver_predictions(["s0", "s1"], p)
    assert len(_preds_on(g, "T")) == 1


def test_precharge_with_mixed_ticks_never_fires():
    rng = random.Random(11)
    decay = NF.DEFAULT_CONFIG["decay_rate"]
    for _ in range(60):
        g, p = _rg(n_src=1, w=0.5, theta=rng.uniform(0.5, 1.0), prediction_pre_charge_factor=1.0,
                   extra={"i0": {"is_inhibitory": True}})
        g.create_synapse("i0", "T", weight=0.3)
        g.nodes["T"].voltage = rng.uniform(0.0, 0.3)
        ticks = {1: rng.uniform(0.0, 0.5), 2: -rng.uniform(0.0, 0.3), 3: rng.uniform(0.0, 0.4)}
        for dt, c in ticks.items():
            _pend(g, "T", c, dt=dt)
        g._receiver_predictions(["s0", "i0"], p)
        if _preds_on(g, "T"):
            v = g.nodes["T"].voltage
            for k in (1, 2, 3, 4, 5):
                v = v * decay + ticks.get(k, 0.0)
                assert v < g.nodes["T"].threshold, (k, ticks)


def test_converging_parents_do_not_grow_second_branch():
    """Review I2: P1 and P2 both lead to U (their strongest); P2 also to a weaker W. U is registered once, W never."""
    g, p = _rg(n_src=0)
    for n in ("P1", "P2", "U", "W", "a1", "a2"):
        g.create_node(node_id=n)
        g.nodes[n].threshold = 0.8
    g.create_synapse("a1", "P1", weight=0.3)
    g.create_synapse("a2", "P2", weight=0.3)
    g.create_synapse("P1", "U", weight=1.0)
    g.create_synapse("P2", "U", weight=1.0)
    g.create_synapse("P2", "W", weight=0.95)   # imagined 0.375*0.7*0.95 = 0.249 >= rho*theta 0.24: W passes alone
    _fire(g, p, ["a1", "a2"], [("P1", 0.3), ("P2", 0.3)])
    deep = sorted(pr.target_node_id for pr in g.active_predictions.values() if pr.chain_depth == 1)
    assert deep == ["U"]


def test_chain_episode_decays_pred_ema():
    """Review I3: a chain registration decays the target's episode EMA over the gap before adding alpha."""
    g, p = _chain_graph()
    st = g._receiver_state()
    i = g._receiver_row(st, "U1")
    st["pred_ema"][i], st["last_t"][i] = 1.0, g.timestep
    g.timestep += 100
    _fire(g, p, ["s0", "s1", "s2"], [("T", 0.1)] * 3)
    assert [pr.target_node_id for pr in g.active_predictions.values() if pr.chain_depth == 1] == ["U1"]
    alpha = p[4]
    assert float(st["pred_ema"][i]) == pytest.approx((1 - alpha) ** 100 + alpha)
    assert int(st["last_t"][i]) == g.timestep


def test_passing_but_dropped_still_counts_as_an_episode():
    """S18short: the table cap is bookkeeping; a neuron that reached the predictive state counts toward its own rate."""
    g, p = _rg(n_src=0, prediction_max_active=0, prediction_receiver_max_active=0)
    g.create_node(node_id="a0")
    g.create_synapse("a0", "T", weight=0.3)
    _fire(g, p, ["a0"], [("T", 0.3)])
    st = g._receiver_state()
    assert g._recv_stats["dropped"] == 1
    assert float(st["pred_ema"][st["idx"]["T"]]) == pytest.approx(p[4])


def test_receiver_table_capacity_is_its_own_knob():
    g, p = _rg(n_src=0, prediction_max_active=1, prediction_receiver_max_active=3)
    for i in range(3):
        g.create_node(node_id=f"T{i}")
        g.nodes[f"T{i}"].threshold = 0.8
        g.create_node(node_id=f"a{i}")
        g.create_synapse(f"a{i}", f"T{i}", weight=0.4)
    _fire(g, p, ["a0", "a1", "a2"], [(f"T{i}", 0.4) for i in range(3)])
    assert len(g.active_predictions) == 3 and g._recv_stats["dropped"] == 0


def test_receiver_default_capacity_is_10000():
    g, _ = _rg()
    assert g._receiver_max_active() == 10000
