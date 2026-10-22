# ---- Changelog ----
# [2026-10-22] Claude (lane stability-s6, #1136 / #1139 / #1166; Josh 2026-10-22 "I think that we should have some form of
#   intrinsic spontaneous activity in NG, yes." + "And I concur with your recommendations for the next round, as well.")
#   — S6: the bounded breaker, intrinsic spontaneous activity (miniature PSPs), the quiet hold on synaptic up-scaling
# What: (1) every S6 key absent / explicitly False == the combination commit (f28afe5's module) exactly on the dict,
#       NodeStore-only and native paths, with the whole combination candidate on (S..S5 + round I tuned + recall brakes;
#       every op of the H2 drive workload incl. sweeps + a sleep; final checkpoint BYTES); not in DEFAULT_CONFIG; set keys
#       that cannot run log ERROR once and change nothing. (2) the breaker: with the key, a silent mind still trips (the
#       event) but no node is multiplied x1.2 / x0.9; only under-target nodes move, by the homeostatic rate, and none
#       reaches the cap; without it the global boost is unchanged. (3) spontaneous activity: an unstimulated graph fires
#       with it and is silent without it; deterministic per seed; minis that tip nothing over leave the input sensor at 0
#       (a node's own background is not afferent input); the activity is regulated (homeostasis sees its spikes).
#       (4) the quiet hold: through a quiet phase up-scaling is held (counted) for nodes with no recent afferent input
#       and still granted to a node receiving input. (5) nothing new is saved: a checkpoint written with every S6 key on
#       loads in the combination module and has the combination's key set.
# Why:  hyperedge-store design §2.3 S6 rows; dev-log 2026-10-22 (whole-mind proof).
# How:  oracle via `git show f28afe5:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S5 / I helpers.
# -------------------
"""#1136 round S6: the bounded breaker, spontaneous activity, the quiet hold."""
import logging
import os
import random

import msgpack
import numpy as np
import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s5 as S5          # noqa: E402
from tests import test_interneurons_i as I         # noqa: E402

NF = H2.NF
COMBO_REV = os.environ.get("NG_S6_COMBO_REV", "f28afe5")
CMOD = H2._load(COMBO_REV, "nf_combo_commit_s6")
HAVE_S6 = (getattr(NF.ng_tract.NodeStore, "stab_round", 0) >= 7
           and getattr(NF.ng_tract.SynapseStore, "stab_round", 0) >= 6)
needs_s6 = pytest.mark.skipif(not HAVE_S6, reason="ng_tract without the S6 kernels")
S6_SWITCHES = ("breaker_homeo_enabled", "spontaneous_activity_enabled", "syn_scale_quiet_hold_enabled")
CAND = dict(S5.S5_FULL, **I.I2, interneuron_decay=0.85, inh_g_decay=0.85, interneuron_coincidence=1,
            recall_brakes_enabled=True)
S6_FULL = dict(CAND, breaker_homeo_enabled=True, spontaneous_activity_enabled=True, syn_scale_quiet_hold_enabled=True)
ckpt = S.ckpt
restore_as = S.restore_as


# ---------------------------------------------------------------------------
# (1) identities
# ---------------------------------------------------------------------------

@needs_s6
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_combination_commit_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(CMOD, kind, start), restore_as(NF, kind, start), seed)
    if kind == "native":
        t.op("keys", lambda g: g.config.update(CAND))
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in S6_SWITCHES})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S6_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_s6
def test_keys_not_in_default_config():
    for k in S6_SWITCHES + ("spont_release_p", "spont_quantal", "spont_seed", "syn_scale_quiet_window"):
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(CMOD.DEFAULT_CONFIG.items())


@needs_s6
@pytest.mark.parametrize("case", ["quiet_without_s", "quiet_without_input", "bad_p", "bad_q", "bad_window"])
def test_misconfigured_s6_keys_log_error_once_and_change_nothing(case, caplog):
    cfg, extra = CAND, {}
    if case == "quiet_without_s":
        cfg = {k: v for k, v in CAND.items() if k != "syn_scale_evidence_enabled"}
        extra = dict(syn_scale_quiet_hold_enabled=True)
    elif case == "quiet_without_input":
        cfg = {k: v for k, v in CAND.items() if k not in ("node_homeo_input_evidence_enabled",
                                                          "node_homeo_hopeless_enabled")}
        extra = dict(syn_scale_quiet_hold_enabled=True)
    else:
        extra = {"bad_p": dict(spontaneous_activity_enabled=True, spont_release_p=1.5),
                 "bad_q": dict(spontaneous_activity_enabled=True, spont_quantal=-1.0),
                 "bad_window": dict(syn_scale_quiet_hold_enabled=True, syn_scale_quiet_window=0)}[case]
    start = H2.built(0)
    t = S.Pair(restore_as(CMOD, "native", start), restore_as(NF, "native", start), 0)
    t.op("keys", lambda g: g.config.update(cfg))
    t.b.g.config.update(extra)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 12, sleep_every=True)
    errs = [r for r in caplog.records if "stability set (#1136)" in r.getMessage() and r.levelno == logging.ERROR]
    assert len(errs) == 1, [r.getMessage() for r in errs]
    for k in extra:
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_s6
def test_s6_keys_on_a_dict_backed_graph_log_error_and_stay_off(caplog):
    start = H2.built(0)
    t = S.Pair(restore_as(CMOD, "dict", start), restore_as(NF, "dict", start), 0)
    t.b.g.config.update({k: True for k in S6_SWITCHES})
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 10)
    assert any("S6-round" in r.getMessage() for r in caplog.records)
    for k in S6_SWITCHES:
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)


# ---------------------------------------------------------------------------
# helpers: a quiet native graph with the candidate keys
# ---------------------------------------------------------------------------

def _graph(seed, cfg):
    g = restore_as(NF, "native", H2.built(seed))
    g.config.update(cfg)
    if g.config.get("interneuron_recruit_enabled"):
        g._interneurons_recruit(g._stability_params().irecruit)
    return g


def _silence(g, steps):
    fired = 0
    for _ in range(steps):
        fired += len(g.step().fired_node_ids)
    return fired


def _cols(g, names):
    _, *vals = g.nodes.columns(list(names), with_ids=True)
    return [np.asarray(v, dtype=float) for v in vals]


# ---------------------------------------------------------------------------
# (2) the bounded breaker
# ---------------------------------------------------------------------------

@needs_s6
@pytest.mark.parametrize("bounded", [False, True])
def test_breaker_keeps_its_trigger_and_is_bounded_by_homeostasis(bounded):
    cfg = dict(CAND, zero_fire_breaker_steps=40, zero_fire_alert_steps=10)
    if bounded:
        cfg["breaker_homeo_enabled"] = True
    g = _graph(0, cfg)
    # silence the mind: every voltage to rest, thresholds high enough that nothing fires for a while
    for nid in list(g.nodes):
        nd = g.nodes[nid]
        nd.voltage = 0.0
        nd.threshold = 3.0
    g._delay_buffer.clear()
    trips = []
    g.register_event_handler("zero_fire_breaker_tripped", lambda **k: trips.append(k))
    x0, t0, ema0 = _cols(g, ("intrinsic_excitability", "threshold", "firing_rate_ema"))
    _silence(g, 45)
    x1, t1, _ = _cols(g, ("intrinsic_excitability", "threshold", "firing_rate_ema"))
    assert len(trips) == 1, "the trigger fires either way"
    if not bounded:
        assert np.allclose(x1, np.minimum(x0 * 1.2, 5.0))
        assert np.allclose(t1, np.maximum(0.01, t0 * 0.9))
        return
    st = g._stability_params()
    assert st.breaker is not None
    thr_rate, exc_rate = st.breaker[0], st.breaker[1]
    # no global multiply: the largest relative move is a few homeostatic steps, not x1.2 / x0.9
    assert (x1 / x0).max() < 1.0 + 4 * exc_rate * 2 + 1e-9
    assert (t0 - t1).max() <= 4 * thr_rate * 6 + 1e-9
    assert (x1 >= x0 - 1e-15).all() and (t1 <= t0 + 1e-15).all(), "a lift only lifts"
    assert (x1 < 5.0).all() or (x0[x1 >= 5.0] >= 5.0).all(), "never pushed onto the cap"
    moved = (x1 > x0) | (t1 < t0)
    assert moved.any()
    s6 = g.nodes.s6_state()
    assert s6["lift_calls"] >= 6 and s6["lift_thr_rows"] > 0


@needs_s6
def test_breaker_lift_touches_only_under_target_nodes():
    g = _graph(1, dict(CAND, breaker_homeo_enabled=True))
    rule = next(r for r in g._plasticity_rules if isinstance(r, NF.HomeostaticRule))
    ids = list(g.nodes)
    over = set(ids[: len(ids) // 2])
    for nid in ids:
        g.nodes[nid].firing_rate_ema = 0.5 if nid in over else 0.0
    st = g._stability_params()
    # with slow on the decision reads the long rate: set it from the fast one through one EMA update with alpha 1
    g.nodes.update_firing_ema([n for n in ids if n in over], 0.0, 1.0)
    x0, t0 = _cols(g, ("intrinsic_excitability", "threshold"))
    g.nodes.breaker_lift(rule._degree_targets, float(rule.target_firing_rate), st.breaker[0], st.breaker[1],
                         st.breaker[3], slow=st.breaker[4], prop=st.breaker[5])
    x1, t1 = _cols(g, ("intrinsic_excitability", "threshold"))
    for k, nid in enumerate(ids):
        if nid in over:
            assert x1[k] == x0[k] and t1[k] == t0[k]
        else:
            assert x1[k] >= x0[k] and t1[k] <= t0[k]


# ---------------------------------------------------------------------------
# (3) spontaneous activity
# ---------------------------------------------------------------------------

@needs_s6
def test_spontaneous_activity_gives_a_resting_level_without_any_input():
    quiet = dict(CAND)
    off = _graph(2, quiet)
    on = _graph(2, dict(quiet, spontaneous_activity_enabled=True, spont_release_p=0.05, spont_quantal=1.0))
    for g in (off, on):
        for nid in list(g.nodes):
            g.nodes[nid].voltage = 0.0
        g._delay_buffer.clear()
    f_off = _silence(off, 60)
    f_on = _silence(on, 60)
    assert f_off == 0
    assert f_on > 0
    s6 = on.nodes.s6_state()
    assert s6["sp_on"] and s6["minis_total"] > 0 and s6["sp_fired_total"] > 0


@needs_s6
def test_spontaneous_release_is_deterministic_per_seed():
    cfg = dict(CAND, spontaneous_activity_enabled=True, spont_release_p=0.05, spont_quantal=1.0)
    a, b, c = _graph(0, cfg), _graph(0, cfg), _graph(0, dict(cfg, spont_seed=7))
    # identical random / uuid streams per op (sprouting draws from them), as the identity tests
    sa, sb, sc = H2.Side(a, 0), H2.Side(b, 0), H2.Side(c, 0)
    fa = [sa(lambda g: list(g.step().fired_node_ids)) for _ in range(30)]
    fb = [sb(lambda g: list(g.step().fired_node_ids)) for _ in range(30)]
    fc = [sc(lambda g: list(g.step().fired_node_ids)) for _ in range(30)]
    assert fa == fb
    assert ckpt(a) == ckpt(b)
    assert fa != fc


@needs_s6
def test_subthreshold_minis_are_not_afferent_input():
    # q tiny: the minis move voltages but tip nothing over; the input sensor must stay where the off twin is
    cfg = dict(CAND)
    off = _graph(3, cfg)
    on = _graph(3, dict(cfg, spontaneous_activity_enabled=True, spont_release_p=0.5, spont_quantal=0.001))
    for g in (off, on):
        for nid in list(g.nodes):
            g.nodes[nid].voltage = 0.0
        g._delay_buffer.clear()
    _silence(off, 20)
    _silence(on, 20)
    v_off, = _cols(off, ("voltage",))
    v_on, = _cols(on, ("voltage",))
    assert not np.array_equal(v_off, v_on), "the minis did reach the membranes"
    _, _, _, in_off, fail_off = off.nodes.s2_state()
    _, _, _, in_on, fail_on = on.nodes.s2_state()
    assert in_on == in_off and fail_on == fail_off


@needs_s6
def test_spontaneous_spikes_are_regulated_like_any_other():
    # strong minis: nodes fire spontaneously; their firing-rate EMA (the homeostat's sensor) rises and the slow AHP pool
    # (the S5 brake) charges -- the spikes are ordinary spikes
    cfg = dict(CAND, spontaneous_activity_enabled=True, spont_release_p=0.2, spont_quantal=2.0)
    g = _graph(4, cfg)
    ema0, = _cols(g, ("firing_rate_ema",))
    _silence(g, 50)
    ema1, = _cols(g, ("firing_rate_ema",))
    on, ca, _ = g.nodes.s5_state()
    assert (ema1 > ema0).any()
    assert ca is not None and max(ca) > 0.0


# ---------------------------------------------------------------------------
# (4) the quiet hold
# ---------------------------------------------------------------------------

@needs_s6
def test_quiet_hold_holds_up_scaling_through_a_quiet_phase():
    # the interneuron layer off here: on this 70-node graph its pools cover every node, and its guard (inhibited is not
    # failing) then leaves no failure evidence at all, so S would never up-scale in either twin
    cfg = {k: v for k, v in CAND.items() if not k.startswith("interneuron")}
    cfg["syn_scale_max_up"] = 1.05
    rng = random.Random(5)
    a = _graph(5, cfg)
    # window 1 = "receiving afferent input on this very step": on this small, densely wired graph the paced node's
    # spikes reach most nodes within a few steps, so a longer window would count every node as driven
    b = _graph(5, dict(cfg, syn_scale_quiet_hold_enabled=True, syn_scale_quiet_window=1))
    ids = list(a.nodes)
    for _ in range(40):                                   # a driven phase: weak input that mostly FAILS (S2-1 evidence)
        cue = [(nid, rng.uniform(0.02, 0.08)) for nid in ids]
        for g in (a, b):
            for nid, c in cue:
                g.stimulate(nid, c)
            g.step()
    pace = ids[0]
    for _ in range(300):                                  # a quiet phase: one paced node keeps plasticity running
        for g in (a, b):
            g.stimulate(pace, 5.0)
            g.step()
    s6 = b.nodes.s6_state()
    assert s6["rd_on"] and s6["up_held_quiet"] > 0, repr(sorted(s6.items()))
    assert s6["up_granted"] > 0, "a node receiving input is still up-scaled"
    # (the twins' weights are not compared: once one up-factor is held, STDP and sprouting diverge the trajectories;
    # the kernel's own hold rule is pinned in Rust: node_store.rs s6_quiet_hold_on_up_scaling)


# ---------------------------------------------------------------------------
# (5) nothing new is saved
# ---------------------------------------------------------------------------

@needs_s6
def test_nothing_new_is_saved_and_the_combination_reads_the_file():
    cfg = dict(S6_FULL, spont_release_p=0.05, spont_quantal=1.0)
    g = _graph(6, cfg)
    for _ in range(30):
        g.step()
    b = ckpt(g)
    d = msgpack.unpackb(b, raw=False, strict_map_key=False)
    node_keys = set()
    for nd in d["nodes"].values():
        node_keys |= set(nd)
    assert not ({"sp_v", "last_in", "r_drv", "driven_rate"} & node_keys)
    h = restore_as(CMOD, "native", b)
    assert len(h.nodes) == len(g.nodes)
    h.step()
