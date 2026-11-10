# ---- Changelog ----
# [2026-11-10] DudeMan CC (Claude Opus 5.5, laptop; #1136 S18 Rust-only, plan 2026-11-10-s18-receiver-rust-only) — the
#   receiver is Rust-only: _rg builds the native stores; state is read / seeded through NodeStore.receiver_state /
#   receiver_set_state (the `_recv` dict is gone); the native-vs-Python-path tests become PARITY against the pre-port
#   engine (NG 4409133): one pass exactly, field for field, and twin whole runs (receiver on, with and without S19) to
#   equal checkpoint bytes; + the receiver needs the native stores (ERROR once, off); the effective-threshold check
#   wraps the native scans. (`_receiver_max_active` / capacity tests unchanged.) The chain-episode decay is checked EXACTLY
#   against CPython's float ** int (Grok review 2026-11-10).
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
from collections import deque

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
    g = NF.Graph(dict(ON), native_node_store=True, native_hyperedge_store=True)   # [2026-11-10] Rust-only
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
    g = NF.Graph(dict(ON, **cfg), native_node_store=True, native_hyperedge_store=True)   # [2026-11-10] Rust-only
    g.create_node(node_id="T")
    g.nodes["T"].threshold = theta
    for i in range(n_src):
        g.create_node(node_id=f"s{i}")
        g.create_synapse(f"s{i}", "T", weight=w)
    for nid, kw in (extra or {}).items():
        g.create_node(node_id=nid, **kw)
    return g, g._stab_receiver()


def _rs(g, nid):
    """[2026-11-10] the receiver's state for one node: (rho, pred_ema, conf_ema, last_t), None if not seen."""
    return g.nodes.receiver_state([nid])[0]


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
    for _ in range(40):
        g.active_predictions.clear()
        g.timestep += 1
        _fire(g, p, ["s0", "s1", "s2"], [("T", 0.1)] * 3)
    rho_hi = _rs(g, "T")[0]
    assert rho_hi > 0.3
    g.nodes["T"].firing_rate_ema = 0.5
    for _ in range(400):
        g.active_predictions.clear()
        g.timestep += 1
        _fire(g, p, ["s0"], [("T", 0.01)])   # candidate, never predicted
    assert _rs(g, "T")[0] == pytest.approx(0.1)   # down to rho_min, not below


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
    c0 = _rs(g, "T")[2]
    g.timestep = pred.expires_at + 1
    g._cleanup_predictions()
    assert _weights(g, ["s0", "s1", "s2"]) == pytest.approx([0.1 - 0.05 * pred.confidence / 3] * 3)
    assert _rs(g, "T")[2] < c0


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


@pytest.mark.parametrize("kind", ["nodes", "native"])   # [2026-11-10] a dict graph keeps the Rust-only receiver off (test below)
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
    assert g.nodes.receiver_set_state("U1", p[0], g.timestep, pred_ema=1.0, last_t=g.timestep)
    g.timestep += 100
    _fire(g, p, ["s0", "s1", "s2"], [("T", 0.1)] * 3)
    assert [pr.target_node_id for pr in g.active_predictions.values() if pr.chain_depth == 1] == ["U1"]
    alpha = p[4]
    # exact, not approx (Grok review 2026-11-10): Rust's powf decay against CPython's float ** int, bit for bit
    assert _rs(g, "U1")[1] == 1.0 * (1 - alpha) ** 100 + alpha
    assert _rs(g, "U1")[3] == g.timestep


def test_passing_but_dropped_still_counts_as_an_episode():
    """S18short: the table cap is bookkeeping; a neuron that reached the predictive state counts toward its own rate."""
    g, p = _rg(n_src=0, prediction_max_active=0, prediction_receiver_max_active=0)
    g.create_node(node_id="a0")
    g.create_synapse("a0", "T", weight=0.3)
    _fire(g, p, ["a0"], [("T", 0.3)])
    assert g._recv_stats["dropped"] == 1
    assert _rs(g, "T")[1] == pytest.approx(p[4])


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


# ---------------------------------------------------------------------------
# (5) the native reads (spec §10: the pass cost > 2 % of a step in S18short2) — oracle: the Python path, same state
# ---------------------------------------------------------------------------

HAVE_RECV = (getattr(NF.ng_tract.SynapseStore, "receiver_round", 0) >= 1
             and getattr(getattr(NF.ng_tract, "NodeStore", object), "receiver_round", 0) >= 1)
needs_recv = pytest.mark.skipif(not HAVE_RECV, reason="ng_tract without the S18 receiver reads")


def _native_graph(seed):
    g = restore_as(NF, "native", H2.built(seed))
    g.config.update(dict(ARM, **RPE, **ON))
    return g


@needs_recv
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_native_out_rows_match_python(seed):
    g = _native_graph(seed)
    rng = random.Random(seed)
    ids = list(g.nodes.keys())
    for _ in range(5):
        fired = rng.sample(ids, min(40, len(ids)))
        want = set()
        for a in fired:
            for sid in g._outgoing.get(a, ()):
                syn = g.synapses.get(sid)
                if syn is not None and syn.synapse_type != NF.SynapseType.INHIBITORY and syn.weight > 0.0:
                    want.add((syn.post_node_id, sid, a, syn.weight))
        got = g.synapses.receiver_out_rows(fired)
        assert len(got) == len(set(got)) and set(got) == want


@needs_recv
@pytest.mark.parametrize("seed", [0, 1])
def test_native_reads_match_getters(seed):
    g = _native_graph(seed)
    ids = list(g.nodes.keys())[:200] + ["no-such-node"]
    got = g.nodes.receiver_reads(ids)
    assert got[-1] is None
    for nid, r in zip(ids[:-1], got[:-1]):
        n = g.nodes.get(nid)
        assert r == (n.threshold, n.voltage, n.intrinsic_excitability, n.refractory_remaining, n.firing_rate_ema,
                     n.resting_potential, bool(n.is_inhibitory))


# [2026-11-10] S18 Rust-only: the Python receiver path is deleted; the oracle is now the PRE-PORT engine (NG 4409133:
# the Python `_recv` state and loops over the same native reads), run beside the ported one. Exact equality, no rounding.
PREPORT_REV = os.environ.get("NG_S18_PREPORT_REV", "4409133")
PREMOD = H2._load(PREPORT_REV, "nf_s18_preport")


def _state_rows(g):
    """Every seen neuron's receiver state, by id: the pre-port `_recv` dict or the NodeStore columns."""
    if "_recv" in g.__dict__:
        st = g.__dict__["_recv"]
        return sorted((nid, float(st["rho"][i]), float(st["pred_ema"][i]), float(st["conf_ema"][i]), int(st["last_t"][i]))
                      for nid, i in st["idx"].items() if nid in g.nodes)
    if not hasattr(g.nodes, "receiver_state"):
        return []
    ids = list(g.nodes.keys())
    return sorted((nid,) + r for nid, r in zip(ids, g.nodes.receiver_state(ids)) if r is not None)


def _snapshot(g):
    preds = sorted((p.target_node_id, p.chain_depth, p.strength, p.confidence, p.source_node_id, p.drive, p.rho,
                    tuple(p.contributors), p.pre_charge_applied)
                   for p in g.active_predictions.values() if p.contributors is not None)
    volts = sorted((nid, n.voltage) for nid, n in g.nodes.items())
    s = {k: g._recv_stats[k] for k in ("dropped", "registered", "merged", "chains")}
    return preds, _state_rows(g), volts, s


@needs_recv
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rust_receiver_pass_equals_the_preport_python_exactly(seed):
    """Same graph, same pending input, same fired sets: the ported pass registers exactly what the pre-port Python
    pass did (every field, bit for bit) and leaves the same per-neuron state and voltages (pre-charge)."""
    out = []
    for mod in (PREMOD, NF):
        g = restore_as(mod, "native", H2.built(seed))
        g.config.update(dict(ARM, **RPE, **ON))
        rng = random.Random(100 + seed)
        ids = list(g.nodes.keys())
        for _ in range(6):
            fired = rng.sample(ids, min(60, len(ids)))
            for a in fired:
                for sid in g._outgoing.get(a, ()):
                    syn = g.synapses.get(sid)
                    if syn is not None:
                        _pend(g, syn.post_node_id, syn.weight * rng.uniform(0.2, 1.0), dt=rng.randint(1, 5))
            g._receiver_predictions(fired, g._stab_receiver())
            g.timestep += 1
        assert g._recv_stats["registered"] > 0 and g._recv_stats["chains"] >= 0
        out.append(_snapshot(g))
    assert out[0] == out[1]


@needs_recv
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("s19", [False, True])
def test_rust_receiver_whole_run_equals_the_preport_engine(seed, s19):
    """A twin run with the receiver on (and S19's corrected DiffPC, which sizes its plateau with S18's rho): the final
    checkpoint BYTES of the ported engine equal the pre-port engine's."""
    start = H2.built(seed)
    t = S.Pair(restore_as(PREMOD, "native", start), restore_as(NF, "native", start), seed)
    extra = {"diffpc_receiver_enabled": True} if s19 else {}
    t.op("keys", lambda g: g.config.update(dict(ARM, **RPE, **ON, **extra)))
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.b.g._recv_stats["registered"] > 0
    assert _state_rows(t.a.g) == _state_rows(t.b.g)
    assert ckpt(t.a.g) == ckpt(t.b.g)


def test_receiver_needs_the_native_stores(caplog):
    """[2026-11-10] no Python path (Josh 2026-11-03): on the Python node store the receiver logs ERROR once, stays off."""
    g = NF.Graph(dict(ON))
    with caplog.at_level(logging.ERROR):
        assert g._stab_receiver() is None and g._stab_receiver() is None
    errs = [r for r in caplog.records if r.levelno == logging.ERROR and "prediction_receiver_enabled" in r.getMessage()]
    assert len(errs) == 1 and "receiver_round" in errs[0].getMessage()


# ---------------------------------------------------------------------------
# (6) the effective threshold (S18cal: busy neurons predicted below chance — the stored threshold misses S5's brake)
# ---------------------------------------------------------------------------

HAVE_EFF = getattr(getattr(NF.ng_tract, "NodeStore", object), "receiver_round", 0) >= 2
needs_eff = pytest.mark.skipif(not HAVE_EFF, reason="ng_tract without the effective-threshold read (receiver_round 2)")


@needs_eff
def test_native_effective_threshold_is_stored_plus_ahp():
    g = _native_graph(0)
    g.config.update(delta_Ca=0.2, ca_split_slow_delta=0.2)   # calcium on, so firing charges the slow pool
    for nid in list(g.nodes.keys())[:40]:
        g.nodes.get(nid).voltage = 5.0
    for _ in range(6):
        g.step()
    on, ca, _ = g.nodes.s5_state()
    assert on and ca is not None and max(ca) > 0.0, "the slow pool has charge to test with"
    ids = list(g.nodes.keys())
    base = g.nodes.receiver_reads(ids)
    eff = g.nodes.receiver_reads(ids, ahp_gain=0.075)
    for b, e, c in zip(base, eff, ca):
        assert e[0] == pytest.approx(b[0] + 0.075 * c) and e[1:] == b[1:]
    pooled = [p[0] - b[0] for p, b in zip(g.nodes.receiver_reads(ids, pool=True), base)]
    assert min(pooled) == pytest.approx(max(pooled)) and min(pooled) >= 0.0   # one pool offset for every row
    with pytest.raises(ValueError):
        g.nodes.receiver_reads(ids, ahp_gain=-1.0)


@needs_eff
@pytest.mark.parametrize("ca_on", [True, False])
def test_receiver_pass_asks_for_the_effective_threshold(ca_on, monkeypatch):
    g = _native_graph(1)
    g.config["ca_split_enabled"] = ca_on
    seen = []
    # [2026-11-10] Rust-only: the candidate and chain scans take (ahp_gain, pool) as their own arguments
    for name, gi in (("receiver_candidates", 11), ("receiver_chain_hop", 11)):
        orig = getattr(NF.ng_tract, name)
        monkeypatch.setattr(NF.ng_tract, name, lambda *a, _o=orig, _gi=gi: (seen.append((a[_gi], a[_gi + 1])), _o(*a))[1])
    g._receiver_predictions(list(g.nodes.keys())[:30], g._stab_receiver())
    want = float(g.config.get("ca_split_ahp_gain", 0.075)) if ca_on else 0.0
    assert seen and all(a == pytest.approx(want) for a, _ in seen)
    assert all(p is False for _, p in seen)   # interneurons on: the detector adds no pool offset (divisive form)


# ---------------------------------------------------------------------------
# (7) a receiver miss grows no wiring (Josh 2026-11-01: wiring the predictor to every recent firer on a failed
#     prediction is "completely backwards … not brain-like"; a miss = weaken what converged + the error signal + tag
#     the context; learning what actually happened is the surprise side, a later design)
# ---------------------------------------------------------------------------

def _miss_graph(**cfg):
    g, p = _rg(three_factor_enabled=False, **cfg)
    for i in range(5):
        g.create_node(node_id=f"x{i}")   # fired recently, unrelated to the prediction
    return g, p


def _expire_with_recent_firers(g, pred):
    g.timestep = pred.expires_at + 1
    for i in range(5):
        g._recent_spikes.setdefault(f"x{i}", deque(maxlen=50)).append(g.timestep - 1)
    g._cleanup_predictions()


def test_receiver_miss_grows_no_synapse_but_keeps_tag_and_novelty():
    g, p = _miss_graph()
    _fire(g, p, ["s0", "s1", "s2"], [("T", 0.1)] * 3)
    (pred,) = _preds_on(g, "T")
    src_syns = [sid for sid in g._outgoing.get(pred.source_node_id, ())]
    for sid in src_syns:
        g.synapses.get(sid).inactive_steps = 0
    n_syn, novel0 = len(g.synapses), g._total_novel_sequences
    _expire_with_recent_firers(g, pred)
    assert len(g.synapses) == n_syn, "a receiver miss wires nothing"
    assert g._total_novel_sequences == novel0 + 1, "novelty still counted"
    assert all(g.synapses.get(sid).salience > 1.0 for sid in src_syns), "the context is still tagged"


def test_old_prediction_miss_still_explores():
    g, _p = _miss_graph()
    pred = NF.Prediction(source_node_id="s0", target_node_id="T", strength=0.5, confidence=0.5,
                         created_at=g.timestep, expires_at=g.timestep + 10)
    g.active_predictions[pred.prediction_id] = pred
    n_syn = len(g.synapses)
    _expire_with_recent_firers(g, pred)
    assert len(g.synapses) == n_syn + 5, "the source-side path explores as before (s0 -> each recent firer)"
