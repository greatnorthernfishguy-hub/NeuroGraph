# ---- Changelog ----
# [2026-10-28] Claude (lane stability-s13, #1136; Josh 2026-10-28 "Yes, go ahead with both") — S13: the symmetric reward
#   floor and per-source surprise
# What: (1) both S13 keys absent / explicitly False == the S12 NG branch (aa1f82d's module) exactly on the dict,
#       NodeStore-only and native paths with S10's RPE on (final checkpoint BYTES); not in DEFAULT_CONFIG; set keys that
#       cannot run (floor without three-factor; per-source without RPE; bad tau) log ERROR once and change nothing. (2) the
#       floor: a depression larger than the weight and the matching potentiation are capped alike, so a zero-mean reward
#       on a weak synapse nets 0 on average over sign orders (the hard floor nets > 0); a large weight is unchanged. (3) per-source surprise: a source
#       that predicts as well as it expects sends nothing even when another source fails often; a source's expectation
#       starts at its confidence and learns; the step's reward is the mean delta, once; nothing new is saved.
# Why:  dev-log 2026-10-28 S12 §5 (the S12cal ledger split: (a) mean x mean, (c) clip).
# How:  oracle via `git show aa1f82d:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S7b helpers.
# -------------------
"""#1136 round S13: the symmetric reward floor and per-source surprise."""
import logging
import os
import random

import msgpack
import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s7b as S7B        # noqa: E402

NF = H2.NF
BASE_REV = os.environ.get("NG_S13_BASE_REV", "aa1f82d")
CMOD = H2._load(BASE_REV, "nf_s12_commit_s13")
S13_SWITCHES = ("reward_floor_symmetric_enabled", "reward_rpe_per_source_enabled")
S13_PARAMS = ("reward_rpe_source_tau",)
ARM = S7B.S7B_ARM
RPE = {"three_factor_enabled": True, "reward_rpe_enabled": True}
ckpt = S.ckpt
restore_as = S.restore_as
HAVE_FLOOR = getattr(NF.ng_tract.SynapseStore, "reward_floor_round", 0) >= 1
needs_floor = pytest.mark.skipif(not HAVE_FLOOR, reason="ng_tract without the S13 symmetric floor")


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_s12_branch_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(CMOD, kind, start), restore_as(NF, kind, start), seed)
    t.op("keys", lambda g: g.config.update(dict(ARM, **RPE) if kind == "native" else RPE))
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in S13_SWITCHES})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S13_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


def test_keys_not_in_default_config():
    for k in S13_SWITCHES + S13_PARAMS:
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(CMOD.DEFAULT_CONFIG.items())


@pytest.mark.parametrize("case", ["floor_no_three_factor", "source_no_rpe", "source_zero_tau", "source_text_tau"])
def test_misconfigured_s13_logs_error_once_and_changes_nothing(case, caplog):
    base = dict(ARM, **RPE)
    if case == "floor_no_three_factor":
        base["three_factor_enabled"] = False
        extra, key = dict(reward_floor_symmetric_enabled=True), "reward_floor_symmetric_enabled"
    else:
        extra, key = dict(reward_rpe_per_source_enabled=True), "reward_rpe_per_source_enabled"
        if case == "source_no_rpe":
            base["reward_rpe_enabled"] = False
        elif case == "source_zero_tau":
            extra["reward_rpe_source_tau"] = 0
        else:
            extra["reward_rpe_source_tau"] = "slow"
    start = H2.built(0)
    t = S.Pair(restore_as(CMOD, "native", start), restore_as(NF, "native", start), 0)
    t.op("keys", lambda g: g.config.update(base))
    t.b.g.config.update(extra)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 12, sleep_every=True)
    errs = [r for r in caplog.records if r.levelno == logging.ERROR and key in r.getMessage()]
    assert len(errs) == 1, [r.getMessage() for r in errs]
    for k in extra:
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)


# ---------------------------------------------------------------------------
# (2) the symmetric floor
# ---------------------------------------------------------------------------

def _graph(m, cfg, weights=(0.5,), lr=0.1):
    g = m.Graph(dict(RPE, learning_rate=lr, **cfg), native_node_store=True, native_hyperedge_store=True)
    for i in range(len(weights) + 1):
        g.create_node(node_id=f"n{i}")
    syn = [g.create_synapse("n0", f"n{i + 1}", weight=w).synapse_id for i, w in enumerate(weights)]
    return g, syn


def _w(g, syn):
    return [g.synapses.get(sid).weight for sid in syn]


@needs_floor
def test_a_zero_mean_reward_on_a_weak_synapse_nets_zero_with_the_floor():
    """Over every order of two +/- commits (|dw| 0.04 each on a 0.02 synapse) the hard floor nets +0.02 on average (it
    swallows depression only); the symmetric cap is a fair walk: 0 on average (any one order can end lower: 0 absorbs)."""
    import itertools
    net = {}
    for name, cfg in (("hard", {}), ("symmetric", {"reward_floor_symmetric_enabled": True})):
        ends = []
        for signs in itertools.product((0.5, -0.5), repeat=2):
            g, syn = _graph(NF, cfg, weights=(0.02,))
            for strength in signs:   # trace -0.8 (an LTD tag): +0.5 -> dw -0.04, -0.5 -> dw +0.04
                g.synapses.get(syn[0]).eligibility_trace = -0.8
                g.inject_reward(strength)
            ends.append(_w(g, syn)[0] - 0.02)
        net[name] = sum(ends) / len(ends)
    assert net["hard"] == pytest.approx(0.02), "the floor rectifies a zero-mean reward upward"
    assert net["symmetric"] == pytest.approx(0.0, abs=1e-12)


@needs_floor
def test_the_floor_caps_both_directions_at_the_weight():
    for strength, want in ((0.5, 0.0), (-0.5, 0.04)):   # |dw| 0.04 > w 0.02: -> -0.02 / +0.02
        g, syn = _graph(NF, {"reward_floor_symmetric_enabled": True}, weights=(0.02,))
        g.synapses.get(syn[0]).eligibility_trace = -0.8
        g.inject_reward(strength)
        assert _w(g, syn)[0] == pytest.approx(want)


@needs_floor
def test_a_large_weight_is_unchanged_by_the_floor():
    out = []
    for cfg in ({}, {"reward_floor_symmetric_enabled": True}):
        g, syn = _graph(NF, cfg, weights=(0.5,))
        g.synapses.get(syn[0]).eligibility_trace = 0.4
        g.inject_reward(-0.3)
        out.append(_w(g, syn)[0])
    assert out[0] == out[1] == pytest.approx(0.5 - 0.4 * 0.3 * 0.1)


# ---------------------------------------------------------------------------
# (3) per-source surprise
# ---------------------------------------------------------------------------

def _src_graph(m=NF, tau=20):
    g, syn = _graph(m, {"reward_rpe_per_source_enabled": True, "reward_rpe_source_tau": tau}, weights=(0.5, 0.5, 0.5))
    for sid in syn:
        g.synapses.get(sid).eligibility_trace = 0.4
    return g, syn


def test_a_source_that_does_as_well_as_it_expects_sends_nothing():
    g, syn = _src_graph()
    g._s13_q = {"good": 0.9, "bad": 0.1}
    n0 = g._total_rewards_injected
    # "good" hits (0.9 expected) and "bad" misses (0.1 expected): both nearly as expected -> |mean delta| 0.1 x 0.5 = 0.05
    g._s13_source_rpe_broadcast([("good", 1.0, 0.9), ("bad", 0.0, 0.1)])
    assert g._total_rewards_injected == n0, "mean delta 0: nothing sent"
    # S10 judges the same step against ONE global expectation: a large signal from the mix alone
    b, _ = _graph(NF, {}, weights=(0.5,))
    b._s10_rpe_baseline = -0.5
    b._s10_rpe_broadcast([0.9, -0.1])
    assert b._total_rewards_injected == 1


def test_the_expectation_starts_at_the_confidence_and_learns():
    g, syn = _src_graph(tau=4)
    g._s13_source_rpe_broadcast([("a", 0.0, 0.8)])
    assert g._s13_q["a"] == pytest.approx(0.8 + (0.0 - 0.8) / 4)
    assert g._reward_history[-1]["strength"] == pytest.approx(0.5 * (0.0 - 0.8))


def test_the_step_reward_is_the_mean_delta_once():
    g, syn = _src_graph()
    g._s13_q = {"a": 0.5, "b": 0.5}
    n0 = g._total_rewards_injected
    g._s13_source_rpe_broadcast([("a", 1.0, 0.5), ("a", 1.0, 0.5), ("b", 1.0, 0.5)])
    assert g._total_rewards_injected == n0 + 1
    assert g._reward_history[-1]["strength"] > 0


def test_step_routes_kept_events_to_the_per_source_rule():
    g, syn = _src_graph()
    g._s13_q = {"a": 0.2}
    g._s10_events = [0.8]
    g._s13_events = [("a", 1.0, 0.8)]
    g._s10_rpe_baseline = 0.8   # S10 would see delta 0: only S13 sends
    n0 = g._total_rewards_injected
    g.step()
    assert g._total_rewards_injected == n0 + 1
    assert g._reward_history[-1]["strength"] == pytest.approx(0.5 * (1.0 - 0.2))
    assert "_s13_events" not in g.__dict__ and "_s10_events" not in g.__dict__
    assert g._s10_rpe_baseline == 0.8, "S10's global expectation is not used under S13"


def test_nothing_new_is_saved():
    g, syn = _src_graph()
    g._s13_source_rpe_broadcast([("a", 1.0, 0.5)])
    assert g._s13_q
    off, _ = _graph(NF, {}, weights=(0.5, 0.5, 0.5))
    d = msgpack.unpackb(ckpt(g), raw=False, strict_map_key=False)
    assert set(d) == set(msgpack.unpackb(ckpt(off), raw=False, strict_map_key=False))
    assert b"_s13" not in ckpt(g)
    back = restore_as(NF, "native", ckpt(g))
    assert "_s13_q" not in back.__dict__
