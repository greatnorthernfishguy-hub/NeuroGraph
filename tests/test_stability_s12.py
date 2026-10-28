# ---- Changelog ----
# [2026-10-28] Claude (lane stability-s12, #1136; Josh 2026-10-28 "I'll go with your recommendation of (a)" / "Go ahead
#   and do both of them") — S12: the RPE commit shares a fixed plasticity budget across the step's tagged synapses
# What: (1) the S12 key absent / explicitly False == the S11 NG branch (f6ed0de's module) exactly on the dict, NodeStore-only
#       and native paths, with S10's RPE on (final checkpoint BYTES); not in DEFAULT_CONFIG; set keys that cannot run (no
#       RPE, bad tau / cap) log ERROR once and change nothing. (2) the rule: twice the eligibility mass commits the same
#       total, not twice; a zero-mean delta whose good steps carry more trace nets ~0 (S10 nets positive); the gain is
#       capped; no trace = nothing to scale; trace_gain scales the synapse sweep only (hyperedge thresholds keep the
#       strength). (3) nothing new is saved (the running mass relearns after a load).
# Why:  dev-log 2026-10-27_stability-s10-rpe-reward §5.2 (the +412 reward residue over S11cal's waking segment).
# How:  oracle via `git show f6ed0de:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S7b helpers.
# -------------------
"""#1136 round S12: the trace-normalised RPE commit."""
import logging
import os
import random

import msgpack
import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s7b as S7B        # noqa: E402

NF = H2.NF
BASE_REV = os.environ.get("NG_S12_BASE_REV", "f6ed0de")
CMOD = H2._load(BASE_REV, "nf_s11_commit_s12")
S12_SWITCHES = ("reward_rpe_trace_norm_enabled",)
S12_PARAMS = ("reward_rpe_trace_norm_tau", "reward_rpe_trace_norm_cap")
ARM = S7B.S7B_ARM
RPE = {"three_factor_enabled": True, "reward_rpe_enabled": True}
ckpt = S.ckpt
restore_as = S.restore_as


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_s11_branch_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(CMOD, kind, start), restore_as(NF, kind, start), seed)
    t.op("keys", lambda g: g.config.update(dict(ARM, **RPE) if kind == "native" else RPE))
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in S12_SWITCHES})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S12_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


def test_keys_not_in_default_config():
    for k in S12_SWITCHES + S12_PARAMS:
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(CMOD.DEFAULT_CONFIG.items())


@pytest.mark.parametrize("case", ["no_rpe", "zero_tau", "text_tau", "cap_below_one"])
def test_misconfigured_s12_logs_error_once_and_changes_nothing(case, caplog):
    base = dict(ARM, **RPE)
    extra = dict(reward_rpe_trace_norm_enabled=True)
    if case == "no_rpe":
        base["reward_rpe_enabled"] = False
    elif case == "zero_tau":
        extra["reward_rpe_trace_norm_tau"] = 0
    elif case == "text_tau":
        extra["reward_rpe_trace_norm_tau"] = "slow"
    elif case == "cap_below_one":
        extra["reward_rpe_trace_norm_cap"] = 0.5
    start = H2.built(0)
    t = S.Pair(restore_as(CMOD, "native", start), restore_as(NF, "native", start), 0)
    t.op("keys", lambda g: g.config.update(base))
    t.b.g.config.update(extra)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 12, sleep_every=True)
    errs = [r for r in caplog.records if r.levelno == logging.ERROR
            and "reward_rpe_trace_norm_enabled" in r.getMessage()]
    assert len(errs) == 1, [r.getMessage() for r in errs]
    for k in extra:
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)


# ---------------------------------------------------------------------------
# (2) the rule
# ---------------------------------------------------------------------------

def _graph(m, on, n=8, lr=0.01):
    g = m.Graph(dict(RPE, learning_rate=lr), native_node_store=True, native_hyperedge_store=True)
    for i in range(n + 1):
        g.create_node(node_id=f"n{i}")
    syn = [g.create_synapse("n0", f"n{i}", weight=0.5).synapse_id for i in range(1, n + 1)]
    if on:
        g.config.update({"reward_rpe_trace_norm_enabled": True, "reward_rpe_trace_norm_tau": 1e9})
    return g, syn


def _tag(g, syn, k, trace=0.4):
    for i, sid in enumerate(syn):
        g.synapses.get(sid).eligibility_trace = trace if i < k else 0.0


def _wsum(g, syn):
    return sum(g.synapses.get(sid).weight for sid in syn)


def _commit(g, syn, k, o, b=0.0):
    """One S10 step with outcome o against expectation b, k synapses tagged; returns the step's total weight change."""
    _tag(g, syn, k)
    g._s10_rpe_baseline = b
    w0 = _wsum(g, syn)
    g._s10_rpe_broadcast([o])
    return _wsum(g, syn) - w0


def test_twice_the_tagged_mass_commits_the_same_total():
    a, sa = _graph(NF, True)
    b, sb = _graph(NF, True)
    a._s12_trace_ref = b._s12_trace_ref = 4 * 0.4
    da, db = _commit(a, sa, 4, 0.8), _commit(b, sb, 8, 0.8)
    assert da > 0 and db == pytest.approx(da)
    assert b._s12_last == pytest.approx((8 * 0.4, 4 * 0.4, 0.5))
    # S11 (no S12): twice the tags, twice the commit
    c, sc = _graph(CMOD, False)
    d, sd = _graph(CMOD, False)
    assert _commit(d, sd, 8, 0.8) == pytest.approx(2 * _commit(c, sc, 4, 0.8))


def test_a_zero_mean_delta_with_more_trace_on_good_steps_nets_zero():
    """The S11cal residue: better-than-expected steps carry more trace (here twice), so trace x delta skews positive."""
    net = {}
    for name, m, on in (("s11", CMOD, False), ("s12", NF, True)):
        g, syn = _graph(m, on)
        total = 0.0
        for i in range(40):
            good = i % 2 == 0
            total += _commit(g, syn, 8 if good else 4, 0.8 if good else -0.8)
        net[name] = total
    assert net["s11"] > 0.05, net
    assert abs(net["s12"]) < 1e-9, net


def test_the_running_mass_starts_at_the_first_and_learns():
    g, syn = _graph(NF, True)
    g.config["reward_rpe_trace_norm_tau"] = 4
    _commit(g, syn, 8, 0.8)
    assert g._s12_last == pytest.approx((3.2, 3.2, 1.0))      # the first step's gain is 1
    assert g._s12_trace_ref == pytest.approx(3.2)
    _commit(g, syn, 4, 0.8)
    assert g._s12_last == pytest.approx((1.6, 3.2, 2.0))
    assert g._s12_trace_ref == pytest.approx(3.2 + (1.6 - 3.2) / 4)


def test_the_gain_is_capped():
    g, syn = _graph(NF, True)
    g._s12_trace_ref = 100.0
    _commit(g, syn, 1, 0.8)
    assert g._s12_last[2] == 4.0
    g.config["reward_rpe_trace_norm_cap"] = 10
    _commit(g, syn, 1, 0.8)
    assert g._s12_last[2] == 10.0


def test_no_trace_is_nothing_to_scale():
    g, syn = _graph(NF, True)
    g._s12_trace_ref = 2.0
    assert _commit(g, syn, 0, 0.8) == 0.0
    assert g._s12_trace_ref == 2.0 and "_s12_last" not in g.__dict__


def test_trace_gain_scales_the_synapse_sweep_only():
    out = {}
    for gain in (1.0, 0.5):
        g, syn = _graph(NF, False)
        h = g.create_hyperedge({"n1", "n2"}, activation_threshold=0.6).hyperedge_id
        g.hyperedges[h].refractory_remaining = 2
        _tag(g, syn, 8)
        w0 = _wsum(g, syn)
        g.inject_reward(0.4, trace_gain=gain)
        out[gain] = (_wsum(g, syn) - w0, g.hyperedges[h].activation_threshold, g._reward_history[-1]["strength"])
    assert out[0.5][0] == pytest.approx(0.5 * out[1.0][0]) and out[1.0][0] > 0
    assert out[0.5][1] == out[1.0][1] < 0.6
    assert out[0.5][2] == out[1.0][2] == 0.4


def test_nothing_new_is_saved():
    g, syn = _graph(NF, True)
    _commit(g, syn, 8, 0.8)
    assert g._s12_trace_ref is not None
    off, _ = _graph(NF, False)
    d = msgpack.unpackb(ckpt(g), raw=False, strict_map_key=False)
    assert set(d) == set(msgpack.unpackb(ckpt(off), raw=False, strict_map_key=False))
    assert b"_s12" not in ckpt(g)   # (the reward_rpe_trace_norm_* config keys are saved, as every config key is)
    back = restore_as(NF, "native", ckpt(g))
    assert "_s12_trace_ref" not in back.__dict__ and "_s12_last" not in back.__dict__
