# ---- Changelog ----
# [2026-10-20] Claude (lane stability-s5, #1136 round S5; Josh 2026-10-20 "OK, let's do your recs ... Yeah, the order
#   sounds right") — spike-frequency adaptation (#1149 calcium split), short-term synaptic depression (#1150) and
#   direction agreement: off == the S4 commit, S4 keys alone == S4, behaviour
# What: (1) every S5 key absent / explicitly False == the S4 commit (5ad8d30's module) exactly on the dict, NodeStore-only
#       and native paths (every op of the H2 drive workload + a sleep; final checkpoint BYTES); the S4 candidate's keys
#       alone (+ horizon 500, + orphan S5 parameters) == S4 bitwise; S5 keys that cannot run (agreement without slow,
#       the split with calcium off, out-of-range parameters, a dict-backed graph) log ERROR once and change nothing.
#       (2) the split: a node driven far above threshold every step fires at the refractory maximum without it and is
#       slowed (not silenced) with it, and the brake wears off once it quiets; today's channels are untouched off.
#       Depression: a synapse used on every fire delivers less and less (weight x resource) while its learned weight
#       is unchanged, a fresh synapse delivers the full weight, an inhibitory one is exempt. Agreement: moves are held
#       (and counted) when the fast rate disagrees. The full S5 set runs the workload without a declined pass, saves
#       Ca_slow (only when on) and round-trips; older readers step on the file.
# Why:  spec superpowers/specs/2026-10-17-rust-hyperedge-store-design.md §2.3 (S5 rows) / §3.2.
# How:  oracle loaded with `git show <rev>:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S3 / S4
#       tests' helpers reused; PYTHONHASHSEED fixed by the runner.
# -------------------
"""#1136 round S5: spike-frequency adaptation, short-term depression, direction agreement."""
import logging
import os
import random

import msgpack
import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s3 as S3          # noqa: E402
from tests import test_stability_s4 as S4          # noqa: E402

NF = H2.NF
S4_REV = os.environ.get("NG_S5_S4_REV", "5ad8d30")   # the S4 commit (5f92d89's neuro_foundation.py + its tests)
S4MOD = H2._load(S4_REV, "nf_s4_commit_s5")
HAVE_S5 = (S4.HAVE_S4 and getattr(NF.ng_tract.NodeStore, "stab_round", 0) >= 5
           and getattr(NF.ng_tract.SynapseStore, "stab_round", 0) >= 5)
needs_s5 = pytest.mark.skipif(not HAVE_S5, reason="ng_tract without the S5 kernels")

S5_SWITCHES = ("ca_split_enabled", "syn_std_enabled", "node_homeo_agree_enabled")
S5_PARAMS = {"ca_split_slow_tau": 150.0, "ca_split_ahp_gain": 0.05, "ca_split_slow_delta": 0.1,
             "syn_std_u": 0.3, "syn_std_tau": 80.0}
S4_FULL = S4.S4_FULL
S5_FULL = dict(S4_FULL, ca_split_enabled=True, syn_std_enabled=True, node_homeo_agree_enabled=True)
ckpt = S.ckpt
restore_as = S.restore_as


# ---------------------------------------------------------------------------
# (1) identities
# ---------------------------------------------------------------------------

@needs_s5
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_s4_commit_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(S4MOD, kind, start), restore_as(NF, kind, start), seed)
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in S5_SWITCHES})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S5_SWITCHES:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)
    assert not hasattr(t.b.g, "_stab")


@needs_s5
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("keys", ["s4_full", "s4_tau500", "s4_orphan_s5_params"])
def test_s4_keys_alone_equal_the_s4_commit_bitwise(seed, keys):
    cfg = {"s4_full": S4_FULL, "s4_tau500": dict(S4_FULL, syn_scale_rate_tau=500.0), "s4_orphan_s5_params": S4_FULL}[keys]
    start = H2.built(seed)
    t = S.Pair(restore_as(S4MOD, "native", start), restore_as(NF, "native", start), seed)
    t.op("keys", lambda g: g.config.update(cfg))
    if keys == "s4_orphan_s5_params":
        t.b.g.config.update(S5_PARAMS)
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in S5_PARAMS:
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_s5
@pytest.mark.parametrize("case", ["agree_no_slow", "split_calcium_off", "bad_tau", "bad_gain", "bad_u", "bad_std_tau"])
def test_misconfigured_s5_keys_log_error_once_and_change_nothing(case, caplog):
    cfg = S4_FULL
    if case == "agree_no_slow":
        extra = dict(node_homeo_agree_enabled=True)
        cfg = {k: v for k, v in S4_FULL.items() if k not in ("node_homeo_slow_enabled", "node_homeo_prop_enabled",
                                                              "node_homeo_hopeless_enabled")}
    elif case == "split_calcium_off":
        extra = dict(ca_split_enabled=True)
        cfg = dict(S4_FULL, delta_Ca=0.0)
    else:
        extra = {"bad_tau": dict(ca_split_enabled=True, ca_split_slow_tau=1.0),
                 "bad_gain": dict(ca_split_enabled=True, ca_split_ahp_gain=-0.1),
                 "bad_u": dict(syn_std_enabled=True, syn_std_u=0.0),
                 "bad_std_tau": dict(syn_std_enabled=True, syn_std_tau=0.0)}[case]
    start = H2.built(0)
    t = S.Pair(restore_as(S4MOD, "native", start), restore_as(NF, "native", start), 0)
    t.op("keys", lambda g: g.config.update(cfg))
    t.b.g.config.update(extra)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 15, sleep_every=True)
    errs = [r for r in caplog.records if "stability set (#1136)" in r.getMessage() and r.levelno == logging.ERROR]
    assert len(errs) == 1, [r.getMessage() for r in errs]
    for k in extra:
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)
    st = t.b.g._stab
    assert st is None or (st.ca_split is None and st.std is None and not st.agree)


@needs_s5
def test_s5_keys_on_a_dict_backed_graph_log_error_and_stay_off(caplog):
    kind = "dict"
    start = H2.built(0)
    t = S.Pair(restore_as(S4MOD, kind, start), restore_as(NF, kind, start), 0)
    on = dict(ca_split_enabled=True, syn_std_enabled=True)
    t.b.g.config.update(on)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 10, sleep_every=True)
    errs = [r for r in caplog.records if "stability set (#1136)" in r.getMessage() and r.levelno == logging.ERROR]
    assert errs
    for k in on:
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_s5
def test_s5_runs_on_the_node_store_only_path():
    """The NodeStore-only path (native nodes, Python hyperedges) carries the node / synapse kernels, as the S2-S4 node
    keys do: the split and depression run there."""
    g = restore_as(NF, "nodes", H2.built(0))
    g.config.update(ca_split_enabled=True, syn_std_enabled=True)
    for _ in range(10):
        g.step()
    assert g._stab.ca_split is not None and g._stab.std is not None
    assert g.nodes.s5_state()[0]


@needs_s5
def test_s5_keys_not_in_default_config_and_default_config_is_the_tips():
    for k in S5_SWITCHES + tuple(S5_PARAMS):
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(S.TIP.DEFAULT_CONFIG.items())


@needs_s5
@pytest.mark.parametrize("seed", [0, 3])
def test_full_s5_set_runs_the_workload_without_a_declined_pass_and_round_trips(seed):
    start = H2.built(seed)
    g = restore_as(NF, "native", start)
    g.config.update(S5_FULL)
    side = H2.Side(g, seed)

    class _T:
        n = side

        def op(self, name, fn):
            return side(fn)
    H2.drive(_T(), random.Random(seed), 25)
    g.sleep_cycle()
    assert g.hyperedges.pass_declines() == (0, 0, 0, 0)
    assert g._stab.ca_split is not None and g._stab.std == (0.2, 50.0) and g._stab.agree
    cs_on, col, agb = g.nodes.s5_state()
    assert cs_on and max(col) > 0.0
    x = g.synapses.std_resources(float(g.timestep))
    assert (x < 1.0).any() and (x > 0.0).all() and (x <= 1.0).all()
    b = ckpt(g)
    d1 = msgpack.unpackb(b, raw=False, strict_map_key=False)
    assert all("Ca_slow" in nd for nd in d1["nodes"].values())        # the slow pool persists (only when on)
    syn = next(iter(d1["synapses"].values()))
    assert len(syn) == 15                                              # depression is transient: the sacred 15 keys
    g2 = restore_as(NF, "native", b)
    assert g2.nodes.s5_state()[1] == col                               # Ca_slow restored exactly
    d2 = msgpack.unpackb(ckpt(g2), raw=False, strict_map_key=False)
    assert d1["nodes"] == d2["nodes"]
    for M, kind in ((S4MOD, "native"), (S4MOD, "dict"), (S.TIP, "dict")):   # older readers step on
        h = restore_as(M, kind, b)
        h.step()


@needs_s5
def test_s4_set_saves_no_ca_slow_key():
    g = restore_as(NF, "native", H2.built(0))
    g.config.update(S4_FULL)
    for _ in range(5):
        g.step()
    d = msgpack.unpackb(ckpt(g), raw=False, strict_map_key=False)
    assert not any("Ca_slow" in nd for nd in d["nodes"].values())


# ---------------------------------------------------------------------------
# (2) behaviour
# ---------------------------------------------------------------------------

def _tonic(cfg, steps, drive=5.0, quiet=0):
    """y0 is driven far above its threshold every step (a node in a reverberating core): it fires at the refractory
    maximum unless something brakes it; then `quiet` steps without drive."""
    g, y = S._homeo_graph(dict({"co_activation_window": 0}, **cfg))
    d = y[0]
    fires = []
    for k in range(steps + quiet):
        S._pace(g, k)
        if k < steps:
            g.stimulate(d, drive)
        r = g.step()
        fires.append(d in r.fired_node_ids)
    return g, d, fires


@needs_s5
def test_split_slows_a_max_rate_node_gradually_and_the_brake_wears_off():
    gain = 2.0       # the drive (5 x excitability) is far above threshold: the brake must be able to reach it
    cfg = {"ca_split_enabled": True, "ca_split_ahp_gain": gain, "ca_split_slow_tau": 100.0}
    _, _, f_off = _tonic({}, 600)
    g, d, f_on = _tonic(cfg, 600, quiet=300)
    early_off, late_off = sum(f_off[:60]), sum(f_off[500:600])
    late_on = sum(f_on[500:600])
    assert late_off >= 33                                    # off: the refractory maximum (1 in 3) throughout
    assert sum(f_on[:30]) >= 9                               # on: starts at the maximum (no brake yet)
    assert 0 < late_on < 0.8 * late_off                      # slowed, not switched off
    # graded: the inter-spike interval grows smoothly as the brake builds (no jump to silence)
    cs_on, col, _ = g.nodes.s5_state()
    assert cs_on
    row = list(g.nodes.keys()).index(d)
    # after 300 quiet steps (3 tau) the brake is < 10 % of what it was at the end of the drive
    brake_now = gain * col[row]
    brake_drive = gain * 0.2 * (late_on / 100.0) * 100.0     # ~ gain x delta x rate x tau
    assert brake_now < 0.1 * brake_drive


@needs_s5
def test_split_off_keeps_todays_channels_and_on_uses_g_can_alone_on_the_fast_pool():
    # the fast pool alone: a node that fired once has Ca_i = 0.2 and the next steps add g_CaN x Ca_i (not g_net)
    for on in (False, True):
        g, y = S._homeo_graph({"ca_split_enabled": True} if on else {})
        d = y[0]
        g.nodes[d].threshold = 0.5
        g.stimulate(d, 1.0)
        g.step()                                             # fires: Ca_i 0.2, voltage reset to resting 0
        v0 = g.nodes[d].voltage
        g.step()
        dv = g.nodes[d].voltage - v0 * g.config["decay_rate"]
        g_net = g.config["g_CaN"] - g.config["g_AHP"]
        want = (g.config["g_CaN"] if on else g_net) * 0.2
        assert dv == pytest.approx(want, rel=1e-9)


@needs_s5
def test_depression_spends_the_resource_keeps_the_weight_and_spares_fresh_and_inhibitory_synapses():
    g, y = S._homeo_graph({"syn_std_enabled": True, "syn_std_u": 0.5, "syn_std_tau": 1000.0})
    a, b, c = y[0], y[1], y[2]
    s_ab = g.create_synapse(a, b, weight=0.1).synapse_id
    s_ac = g.create_synapse(a, c, weight=0.1, synapse_type=NF.SynapseType.INHIBITORY).synapse_id
    w0 = g.synapses[s_ab].weight
    rows = []
    for k in range(4):
        rows.append(g.synapses.propagation_rows([s_ab, s_ac], True, (0.5, 1000.0, float(10 + k))))
    delivered = [r[0][1] for r in rows]
    assert delivered[0] == w0                                # fresh: the full weight
    assert delivered[1] < delivered[0] and delivered[2] < delivered[1] and delivered[3] < delivered[2]
    assert all(r[1][1] == 0.1 for r in rows)                 # inhibitory: exempt
    assert g.synapses[s_ab].weight == w0                     # the learned weight is untouched
    # in step(): a driven presynaptic node depletes its outgoing synapse; the weight stays the same
    for k in range(30):
        S._pace(g, k)
        g.stimulate(a, 5.0)
        g.step()
    x = g.synapses.std_resources(float(g.timestep))
    ks = list(g.synapses.keys())
    assert x[ks.index(s_ab)] < 0.5 and x[ks.index(s_ac)] == 1.0


@needs_s5
def test_depression_is_off_in_recall_sweeps():
    g, y = S._homeo_graph({"syn_std_enabled": True})
    s_ab = g.create_synapse(y[0], y[1], weight=0.5).synapse_id
    for _ in range(3):
        g.prime_and_propagate(node_ids=[y[0]], currents=[5.0], steps=3)
    x = g.synapses.std_resources(float(g.timestep))
    assert x[list(g.synapses.keys()).index(s_ab)] == 1.0


@needs_s5
def test_agreement_holds_the_silent_ex_core_and_counts_blocked_moves():
    """y1's long rate says 'far over target' (an ex-core node) but it is silent now: without agreement S4 keeps raising
    its threshold; with it the threshold holds and the held moves are counted."""
    def hot(cfg):
        g, y = S._homeo_graph(dict({"node_homeo_evidence_enabled": True, "node_homeo_input_evidence_enabled": True,
                                    "node_homeo_slow_enabled": True, "node_homeo_prop_enabled": True}, **cfg))
        d = y[1]
        g.nodes[d].firing_rate_ema = 0.5                      # r_long seeds from this at the first homeostatic pass
        g.nodes[d].threshold = 0.85
        S._pace(g, 0)
        g.step()
        g.nodes[d].firing_rate_ema = 0.0                      # ...and now it is silent
        for k in range(1, 60):
            S._pace(g, k)
            g.step()
        return g, d
    g4, d = hot({})
    g5, _ = hot({"node_homeo_agree_enabled": True})
    assert g4.nodes[d].threshold > 0.85 + 1e-6               # S4: the lagging long rate keeps pushing it down
    assert g5.nodes[d].threshold <= 0.85 + 0.002             # agreement: held after the fast rate fell
    assert sum(g5.nodes.s5_state()[2]) > 0
