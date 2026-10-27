# ---- Changelog ----
# [2026-10-27] Claude (lane stability-s10, #1136; Josh 2026-10-27 "what's the most similar, conceptually, to how the brain
#   does it?" / "Yes, go ahead") — S10: the reward is a signed prediction error against expectation, once per step
# What: (1) the S10 key absent / explicitly False == the S9 NG branch (7274482's module) exactly on the dict, NodeStore-only
#       and native paths with the S7b arm's keys (final checkpoint BYTES); not in DEFAULT_CONFIG; set keys that cannot run
#       (no three-factor, S9 also on, bad tau) log ERROR once and change nothing. (2) the rule: a failed prediction no
#       longer broadcasts on its own; the step's outcomes are judged against the running expectation — worse than
#       expected depresses EVERY eligible trace (global), better potentiates, as expected sends nothing; a step that
#       resolves nothing sends nothing and leaves the baseline; over a steady stream the signal nets ~0. (3) nothing new
#       is saved (the baseline relearns after a load).
# Why:  dev-log 2026-10-27_stability-s10-rpe-reward §2.
# How:  oracle via `git show 7274482:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S7b helpers.
# -------------------
"""#1136 round S10: the signed (RPE) per-step reward."""
import logging
import os
import random

import msgpack
import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s7b as S7B        # noqa: E402

NF = H2.NF
BASE_REV = os.environ.get("NG_S10_BASE_REV", "7274482")
CMOD = H2._load(BASE_REV, "nf_s9_commit_s10")
S10_SWITCHES = ("reward_rpe_enabled",)
S10_PARAMS = ("reward_rpe_tau",)
ARM = S7B.S7B_ARM
ckpt = S.ckpt
restore_as = S.restore_as


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_s9_branch_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(CMOD, kind, start), restore_as(NF, kind, start), seed)
    if kind == "native":
        t.op("keys", lambda g: g.config.update(ARM))
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in S10_SWITCHES})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S10_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


def test_keys_not_in_default_config():
    for k in S10_SWITCHES + S10_PARAMS:
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(CMOD.DEFAULT_CONFIG.items())


@pytest.mark.parametrize("case", ["no_three_factor", "with_s9", "zero_tau", "text_tau"])
def test_misconfigured_s10_logs_error_once_and_changes_nothing(case, caplog):
    base = dict(ARM)
    extra = dict(reward_rpe_enabled=True)
    if case == "no_three_factor":
        base["three_factor_enabled"] = False
    elif case == "with_s9":
        if getattr(NF.ng_tract.SynapseStore, "reward_round", 0) < 1:
            pytest.skip("ng_tract without the S9 reward pass")
        base["reward_event_scoped_enabled"] = True   # both sides run S9; only b also asks for S10
    elif case == "zero_tau":
        extra["reward_rpe_tau"] = 0
    elif case == "text_tau":
        extra["reward_rpe_tau"] = "slow"
    start = H2.built(0)
    t = S.Pair(restore_as(CMOD, "native", start), restore_as(NF, "native", start), 0)
    t.op("keys", lambda g: g.config.update(base))
    t.b.g.config.update(extra)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 12, sleep_every=True)
    errs = [r for r in caplog.records if r.levelno == logging.ERROR and "reward_rpe_enabled" in r.getMessage()]
    assert len(errs) == 1, [r.getMessage() for r in errs]
    for k in extra:
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)


# ---------------------------------------------------------------------------
# (2) the rule
# ---------------------------------------------------------------------------

def _graph(m, on):
    g = m.Graph({"three_factor_enabled": True, "learning_rate": 0.1}, native_node_store=True, native_hyperedge_store=True)
    for n in "ABCD":
        g.create_node(node_id=n)
    syn = {k: g.create_synapse(k[0], k[1], weight=0.5).synapse_id for k in ("AB", "AC", "DC", "DB")}
    for sid in syn.values():
        g.synapses.get(sid).eligibility_trace = 0.4
    if on:
        g.config["reward_rpe_enabled"] = True
    return g, syn


def _w(g, syn):
    return {k: g.synapses.get(sid).weight for k, sid in syn.items()}


def _pred(m, a, b, conf=0.8):
    return m.Prediction(source_node_id=a, target_node_id=b, strength=0.5, confidence=conf)


def test_a_failed_prediction_no_longer_broadcasts_on_its_own():
    g, syn = _graph(NF, True)
    w0 = _w(g, syn)
    g._on_prediction_error(_pred(NF, "A", "B"), 0.1, {"C"})
    assert _w(g, syn) == w0
    assert g._s10_events == [-0.8]
    # the S9 branch's base broadcasts at once
    b, bs = _graph(CMOD, False)
    b._on_prediction_error(_pred(CMOD, "A", "B"), 0.1, {"C"})
    assert _w(b, bs)["DC"] != w0["DC"]


def test_worse_than_expected_depresses_every_eligible_trace():
    g, syn = _graph(NF, True)
    w0 = _w(g, syn)
    g._s10_rpe_baseline = 0.2
    g._s10_rpe_broadcast([-0.8])
    w1 = _w(g, syn)
    assert all(w1[k] < w0[k] for k in syn), "global and negative"
    assert g._s10_rpe_baseline == pytest.approx(0.2 + (-0.8 - 0.2) / 2000)


def test_better_than_expected_potentiates():
    g, syn = _graph(NF, True)
    w0 = _w(g, syn)
    g._s10_rpe_baseline = -0.5
    g._s10_rpe_broadcast([0.8])
    assert all(_w(g, syn)[k] > w0[k] for k in syn)


def test_as_expected_sends_nothing():
    g, syn = _graph(NF, True)
    w0, n0 = _w(g, syn), g._total_rewards_injected
    g._s10_rpe_broadcast([0.6])   # the first event sets the expectation: its own delta is 0
    assert g._s10_rpe_baseline == 0.6
    g._s10_rpe_broadcast([0.6, 0.6])
    assert _w(g, syn) == w0 and g._total_rewards_injected == n0


def test_a_step_that_resolves_nothing_sends_nothing_and_keeps_the_baseline():
    g, syn = _graph(NF, True)
    g._s10_rpe_baseline = 0.3
    n0 = g._total_rewards_injected
    g.step()
    assert g._s10_rpe_baseline == 0.3 and g._total_rewards_injected == n0
    assert "_s10_events" not in g.__dict__


def test_the_step_takes_the_kept_outcomes_once():
    g, syn = _graph(NF, True)
    g._s10_rpe_baseline = 0.5
    g._s10_events = [-0.8, -0.8]
    n0 = g._total_rewards_injected
    g.step()
    assert g._total_rewards_injected == n0 + 1
    assert g._reward_history[-1]["strength"] == pytest.approx(0.5 * 2 * (-0.8 - 0.5))
    assert "_s10_events" not in g.__dict__


def test_a_steady_stream_nets_about_zero():
    g, _ = _graph(NF, True)
    g.config["reward_rpe_tau"] = 200
    rng = random.Random(7)
    total = n = 0.0
    for _ in range(3000):
        ev = [(0.6 if rng.random() < 0.3 else -0.6) for _ in range(rng.randint(1, 8))]
        b = g.__dict__.get("_s10_rpe_baseline", ev[0])
        total += sum(o - b for o in ev)
        n += len(ev)
        g._s10_rpe_broadcast(ev)
    # the base's broadcast here would be +0.3 / failure on average (always positive); RPE: ~0 per event
    assert abs(total / n) < 0.02, total / n
    assert g._s10_rpe_baseline == pytest.approx(0.3 * 0.6 - 0.7 * 0.6, abs=0.06)


def test_nothing_new_is_saved():
    g, _ = _graph(NF, True)
    g._s10_rpe_broadcast([0.8, -0.8])
    assert g._s10_rpe_baseline is not None
    d = msgpack.unpackb(ckpt(g), raw=False, strict_map_key=False)
    off, _ = _graph(NF, False)
    assert set(d) == set(msgpack.unpackb(ckpt(off), raw=False, strict_map_key=False))
    assert b"_s10" not in ckpt(g)   # (the reward_rpe_* config keys themselves are saved, as every config key is)
    back = restore_as(NF, "native", ckpt(g))
    assert "_s10_rpe_baseline" not in back.__dict__, "the baseline relearns after a load"
