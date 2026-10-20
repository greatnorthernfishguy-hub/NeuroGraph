# ---- Changelog ----
# [2026-10-19] Claude (lane stability-s2, #1136 round S2; Josh 2026-10-19 "Yeah, let's do all 4 now") — the four S2
#   corrections: off == the S commit, S keys alone == the S commit, one test set per mechanism
# What: (1) every S2 key absent / explicitly False == the S commit (04e99e1's module) exactly, on the dict, NodeStore-only
#       and native paths (every op of the H2 drive workload + a sleep, strictly; the final checkpoint BYTES); and with
#       ONLY the S keys set, the native path == the S commit bitwise (bytes included). An S2 key without its
#       prerequisite (input evidence without N / S, settle without S, the C2 leak without C2) changes nothing.
#       (2) Evidence by input received: an episodically cued node counts as driven and is up-regulated (S's mean-voltage
#       rule leaks it instead); a node that never receives input is never wound up. (3) Settle: S's scaling gives no
#       factor inside the deadband and approaches it proportionally, bounded. (4) #1144: a C2-raised threshold leaks back
#       to its own baseline when the assembly is quiet, never below it; a never-raised one is untouched; without the
#       leak it stays (S). (5) Burst inhibition: scales with the burst, decays fast, bounded, absent at steady activity;
#       a weak input that fires at normal activity is held off right after a burst. (6) Round trip + older readers;
#       the pre_fire-handler path.
# Why:  spec superpowers/specs/2026-10-17-rust-hyperedge-store-design.md §2.3 (S2 rows) / §3.2.
# How:  oracles loaded with `git show <rev>:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S tests'
#       Pair / drive / restore helpers reused; PYTHONHASHSEED fixed by the runner.
# -------------------
"""#1136 round S2: evidence by input received, scaling that settles, the C2 leak (#1144), burst inhibition."""
import logging
import os
import random

import msgpack
import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402

NF = H2.NF
S_REV = os.environ.get("NG_S2_S_REV", "04e99e1")   # the S commit (275807f's neuro_foundation.py + its tests)
SMOD = H2._load(S_REV, "nf_s_commit_s2")
TIP = S.TIP
HAVE_S2 = S.HAVE_S and hasattr(NF.ng_tract.NodeStore, "observe_inputs") and hasattr(NF.ng_tract.HyperedgeStore, "c2_bases")
needs_s2 = pytest.mark.skipif(not HAVE_S2, reason="ng_tract without the S2 kernels")

S2_SWITCHES = ("node_homeo_input_evidence_enabled", "syn_scale_settle_enabled", "he_threshold_leak_enabled",
               "burst_inhibition_enabled")
S2_KEYS = S2_SWITCHES + ("node_homeo_input_min", "node_homeo_input_alpha", "node_homeo_input_rate_min",
                         "node_homeo_fail_alpha", "node_homeo_fail_min", "syn_scale_deadband", "he_threshold_leak_rate",
                         "burst_inh_gain", "burst_inh_decay", "burst_inh_max", "burst_inh_base_alpha")
S2_FULL = dict(S.FULL, **{k: True for k in S2_SWITCHES})
ckpt = S.ckpt
restore_as = S.restore_as


def _nodes_of(b):
    return msgpack.unpackb(b, raw=False, strict_map_key=False)


# ---------------------------------------------------------------------------
# (1) identities: off == the S commit; S keys alone == the S commit; S2 keys without prerequisites == S
# ---------------------------------------------------------------------------

@needs_s2
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_s_commit_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(SMOD, kind, start), restore_as(NF, kind, start), seed)
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in S2_SWITCHES})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    if how == "explicit_false":
        for k in S2_SWITCHES:
            t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)
    assert not hasattr(t.b.g, "_stab")


@needs_s2
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("extra", ["none", "orphan_s2_params"])
def test_s_keys_alone_equal_the_s_commit_bitwise(seed, extra):
    """With only the S keys set, the S2 module makes exactly S's native calls (bytes included). `orphan_s2_params`:
    S2 PARAMETER keys without their switches change nothing either."""
    start = H2.built(seed)
    t = S.Pair(restore_as(SMOD, "native", start), restore_as(NF, "native", start), seed)
    t.op("keys", lambda g: g.config.update(S.FULL))
    if extra == "orphan_s2_params":
        t.b.g.config.update({"node_homeo_input_min": 0.3, "syn_scale_deadband": 0.5, "he_threshold_leak_rate": 0.4,
                             "burst_inh_gain": 99.0})
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    for k in ("node_homeo_input_min", "syn_scale_deadband", "he_threshold_leak_rate", "burst_inh_gain"):
        t.b.g.config.pop(k, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_s2
@pytest.mark.parametrize("seed", [0, 1])
def test_s2_switches_without_their_prerequisites_change_nothing(seed):
    """Input evidence without N / S, settle without S, the C2 leak without C2: S exactly (here: H + C1 + C3 only)."""
    base = {"he_habituation_enabled": True, "he_inject_respects_refractory": True, "he_target_unlearn_enabled": True}
    start = H2.built(seed)
    t = S.Pair(restore_as(SMOD, "native", start), restore_as(NF, "native", start), seed)
    t.op("keys", lambda g: g.config.update(base))
    t.b.g.config.update({"node_homeo_input_evidence_enabled": True, "syn_scale_settle_enabled": True,
                         "he_threshold_leak_enabled": True})
    S.drive(t, random.Random(seed), 20, sleep_every=True)
    for k in ("node_homeo_input_evidence_enabled", "syn_scale_settle_enabled", "he_threshold_leak_enabled"):
        t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_s2
def test_s2_keys_not_in_default_config_and_default_config_is_the_tips():
    for k in S2_KEYS:
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(TIP.DEFAULT_CONFIG.items())


@needs_s2
@pytest.mark.parametrize("seed", [0, 3])
def test_full_s2_set_runs_the_workload_without_a_declined_pass(seed):
    start = H2.built(seed)
    g = restore_as(NF, "native", start)
    g.config.update(S2_FULL)
    side = H2.Side(g, seed)

    class _T:
        n = side

        def op(self, name, fn):
            return side(fn)
    H2.drive(_T(), random.Random(seed), 25)
    g.sleep_cycle()
    assert g.hyperedges.pass_declines() == (0, 0, 0, 0)
    in_on, inh_g, inh_b, a, f = g.nodes.s2_state()
    assert in_on and a is not None and 0.0 <= inh_g <= 1.0
    restore_as(NF, "native", ckpt(g))


# ---------------------------------------------------------------------------
# (2) evidence by input received
# ---------------------------------------------------------------------------

def _cued_run(cfg, steps=400, cue_every=40):
    g, y = S._homeo_graph(dict({"node_homeo_evidence_enabled": True, "node_homeo_leak": 0.1}, **cfg))
    cued, never = y[0], y[1]
    for nid in (cued, never):
        g.nodes[nid].intrinsic_excitability = 2.0     # some gain to keep or lose
        g.nodes[nid].threshold = 0.85
        g.nodes[nid].firing_rate_ema = 0.001          # quiet (under target), not silent
    peak_never = 0.0
    for k in range(steps):
        S._pace(g, k)
        if k % cue_every == 0:
            g.stimulate(cued, 0.1)                     # a weak episodic cue: delivered, fails to fire it
        g.step()
        peak_never = max(peak_never, g.nodes[never].intrinsic_excitability)
    return g, cued, never, peak_never


@needs_s2
def test_an_episodically_cued_node_is_driven_and_a_never_reached_one_is_not():
    g, cued, never, peak = _cued_run({"node_homeo_input_evidence_enabled": True})
    C, N0 = g.nodes[cued], g.nodes[never]
    assert C.intrinsic_excitability > 2.0 and C.threshold < 0.85          # driven but failing: up-regulated
    assert N0.intrinsic_excitability < 1.05 and N0.threshold == pytest.approx(0.85)   # never reached: leaks to baseline
    assert peak <= 2.0                                                      # and is never wound up on the way
    in_on, _, _, a, f = g.nodes.s2_state()
    ids = list(g.nodes.keys())
    i, j = ids.index(cued), ids.index(never)
    assert in_on and a[i] >= 0.0005 and f[i] >= 0.5 and a[j] < 0.0005 and f[j] == 0.0
    assert not g.nodes.stab_columns()[0]                                    # S's v_ema is not switched on (or saved)
    # S's mean-voltage rule reads the same cued node as undriven and takes its gain
    g0, cued0, _, _ = _cued_run({})
    assert g0.nodes[cued0].intrinsic_excitability < 1.05


@needs_s2
def test_a_node_whose_input_fires_it_holds_its_gain():
    g, y = S._homeo_graph({"node_homeo_evidence_enabled": True, "node_homeo_leak": 0.1,
                           "node_homeo_input_evidence_enabled": True,
                           "co_activation_window": 0})   # no sprouting: the cue is the node's ONLY input
    r = y[0]
    g.nodes[r].intrinsic_excitability = 3.0
    g.nodes[r].threshold = 0.5
    held = None
    for k in range(400):
        S._pace(g, k)
        if k % 40 == 0:
            g.stimulate(r, 0.6)                        # 0.6 x x > threshold throughout: the cue fires it (succeeds)
        g.step()
        if k == 41:
            # until its first observed input (k = 40; switch-on at k = 0 only seeds the baseline) it is unafferented
            # and leaks; from then on it is afferented and its input succeeds
            held = (g.nodes[r].intrinsic_excitability, g.nodes[r].threshold)
            assert 1.0 < held[0] < 3.0
    assert (g.nodes[r].intrinsic_excitability, g.nodes[r].threshold) == held   # afferented + responsive: holds


@needs_s2
def test_input_evidence_columns_saved_only_when_on_and_read_by_older_readers():
    g, _ = S._homeo_graph({"node_homeo_evidence_enabled": True, "node_homeo_input_evidence_enabled": True})
    for k in range(10):
        S._pace(g, k)
        g.step()
    d = _nodes_of(ckpt(g))
    assert all("input_ema" in nd and "input_fail_ema" in nd and "v_ema" not in nd and len(nd) == 21
               for nd in d["nodes"].values())
    g2 = restore_as(NF, "native", ckpt(g))
    assert g2.nodes.s2_state()[3] == g.nodes.s2_state()[3] and g2.nodes.s2_state()[4] == g.nodes.s2_state()[4]
    for M, kind in ((SMOD, "native"), (TIP, "native"), (TIP, "dict")):
        restore_as(M, kind, ckpt(g)).step()
    g0, _ = S._homeo_graph({"node_homeo_evidence_enabled": True})
    S._pace(g0, 0)
    g0.step()
    assert all("input_ema" not in nd for nd in _nodes_of(ckpt(g0))["nodes"].values())


@needs_s2
def test_pre_fire_handler_step_still_observes_inputs():
    g, y = S._homeo_graph({"node_homeo_evidence_enabled": True, "node_homeo_input_evidence_enabled": True})
    g._event_handlers.setdefault("pre_fire", []).append(
        lambda *a, **k: 0.0)
    for k in range(5):
        S._pace(g, k)
        g.stimulate(y[0], 0.1)
        g.step()
    in_on, _, _, a, _ = g.nodes.s2_state()
    assert in_on and a[list(g.nodes.keys()).index(y[0])] > 0.0


# ---------------------------------------------------------------------------
# (3) S settles at the set point it restores
# ---------------------------------------------------------------------------

@needs_s2
def test_scaling_settles_inside_the_deadband_and_approaches_it_proportionally():
    g, y = S._homeo_graph({"syn_scale_evidence_enabled": True, "node_homeo_evidence_enabled": True,
                           "node_homeo_input_evidence_enabled": True})
    S._pace(g, 0)
    g.step()
    d = _nodes_of(ckpt(g))
    rates = {"y0": 0.05, "y1": 0.045, "y2": 0.2, "y3": 0.0001}   # at target, in the band, far over, far under
    for nid, r in rates.items():
        d["nodes"][nid]["syn_rate_ema"] = r
        d["nodes"][nid]["input_ema"] = 0.01
        d["nodes"][nid]["input_fail_ema"] = 0.9                    # afferented and failing (evidence for up)
    g2 = restore_as(NF, "native", msgpack.packb(d, use_bin_type=True))
    args = ({}, 0.05, 0.1, 1.03, 0.97, 0.5)
    s = g2.nodes.evidence_scales(*args, input=(0.0005, 0.5))                 # S's rule, input evidence
    st = g2.nodes.evidence_scales(*args, input=(0.0005, 0.5), deadband=0.2)  # S2: settles
    assert s["y1"] > 1.0 and "y1" not in st and "y0" not in st              # S winds past the band; S2 holds
    assert st["y2"] == max((0.25 * 1.2) ** 0.1, 0.97) and st["y3"] == 1.03  # outside: toward the band, bounded
    s_old = g2.nodes.evidence_scales(*args)                                  # S's call, S's answer (mean voltage)
    assert s_old == SMOD_scales(g2, args)


def SMOD_scales(g, args):
    return g.nodes.evidence_scales(*args, input=None, deadband=None)


# ---------------------------------------------------------------------------
# (4) #1144: C2's raised thresholds leak back to the assembly's own baseline
# ---------------------------------------------------------------------------

def _c2_run(leak):
    cfg = {"he_threshold_homeo_enabled": True, "he_rate_target": 0.05, "he_threshold_homeo_rate": 0.01}
    if leak:
        cfg.update({"he_threshold_leak_enabled": True, "he_threshold_leak_rate": 0.05})
    g, x = S.tiny(cfg, n=8)
    hot = g.create_hyperedge({x[0], x[1]}, activation_threshold=0.3).hyperedge_id
    rare = g.create_hyperedge({x[4], x[5]}, activation_threshold=0.3).hyperedge_id
    g.hyperedges[hot].recent_activation_ema = 0.3
    g._plasticity_rules = []
    for _ in range(100):                                 # over-active: the threshold climbs
        g.stimulate(x[0], 2.0)
        g.stimulate(x[1], 2.0)
        g.step()
    raised = g.hyperedges[hot].activation_threshold
    lows = []
    for _ in range(600):                                 # quiet: activity falls below target
        g.stimulate(x[7], 2.0)                           # (something fires every step)
        g.step()
        lows.append(g.hyperedges[hot].activation_threshold)
    return g, hot, rare, raised, lows


@needs_s2
def test_c2_raised_threshold_leaks_back_to_its_own_baseline_never_below():
    g, hot, rare, raised, lows = _c2_run(leak=True)
    assert raised > 0.5
    assert lows[-1] == pytest.approx(0.3, abs=1e-3) and min(lows) >= 0.3    # back to its own baseline, never below
    assert all(b <= a + 1e-15 for a, b in zip(lows, lows[1:]))             # monotone while quiet
    assert g.hyperedges[rare].activation_threshold == 0.3                  # never raised: untouched
    assert g.hyperedges.c2_bases() == [(g.hyperedges[hot].activation_threshold, 0.3)]
    d = _nodes_of(ckpt(g))
    assert d["hyperedges"][hot]["c2_threshold_base"] == 0.3 and "c2_threshold_base" not in d["hyperedges"][rare]
    g2 = restore_as(NF, "native", ckpt(g))
    assert g2.hyperedges.c2_bases() == g.hyperedges.c2_bases()
    for M, kind in ((SMOD, "native"), (TIP, "native"), (TIP, "dict")):
        restore_as(M, kind, ckpt(g)).step()


@needs_s2
def test_c2_without_the_leak_is_the_s_ratchet():
    g, hot, rare, raised, lows = _c2_run(leak=False)
    assert lows[-1] == raised and g.hyperedges.c2_bases() == []
    assert all("c2_threshold_base" not in h for h in _nodes_of(ckpt(g))["hyperedges"].values())


@needs_s2
def test_c2_leak_reactivates_a_silenced_assembly():
    """The S failure: an assembly C2 drove to 1.0 never fired again. With the leak it comes back to fire on input
    that reaches its baseline fraction."""
    cfg = {"he_threshold_homeo_enabled": True, "he_rate_target": 0.05, "he_threshold_homeo_rate": 0.02,
           "he_threshold_leak_enabled": True, "he_threshold_leak_rate": 0.02}
    g, x = S.tiny(cfg, n=8)
    h = g.create_hyperedge({x[0], x[1], x[2], x[3]}, activation_threshold=0.5).hyperedge_id
    g.hyperedges[h].recent_activation_ema = 0.3
    g._plasticity_rules = []
    for _ in range(100):
        for i in range(4):
            g.stimulate(x[i], 2.0)
        g.step()
    assert g.hyperedges[h].activation_threshold > 0.75
    for _ in range(600):
        g.stimulate(x[7], 2.0)
        g.step()
    for i in range(3):
        g.stimulate(x[i], 2.0)                          # 3 of 4 members: above the 0.5 baseline, below the raised one
    fired = False
    for _ in range(3):
        fired |= h in g.step().fired_hyperedge_ids
    assert fired


# ---------------------------------------------------------------------------
# (5) a burst recruits inhibition
# ---------------------------------------------------------------------------

def _inh_graph(on, n=20):
    cfg = {"orphan_node_grace_period": 10 ** 9, "three_factor_enabled": False, "default_threshold": 0.85}
    if on:
        cfg.update({"burst_inhibition_enabled": True, "burst_inh_gain": 10.0, "burst_inh_decay": 0.5,
                    "burst_inh_max": 1.0, "burst_inh_base_alpha": 0.01})
    g = NF.Graph(cfg, native_node_store=True, native_hyperedge_store=True)
    ids = [g.create_node(node_id=f"z{i:02d}").node_id for i in range(n)]
    for i in range(3):
        g.create_node(node_id=f"p{i}")
    g._plasticity_rules = []
    return g, ids


@needs_s2
def test_burst_inhibition_scales_decays_and_is_bounded_and_spares_normal_activity():
    out = {}
    for on in (False, True):
        g, z = _inh_graph(on)
        gs = []
        for k in range(30):                              # steady low activity: one pacemaker per step
            S._pace(g, k)
            g.step()
            gs.append(g.nodes.s2_state()[1])
        assert max(gs) < 1e-12                           # steady activity recruits no inhibition
        g.nodes[z[0]].voltage = 0.0
        g.stimulate(z[0], 0.9 / 0.97)                    # a weak input at normal activity: fires (0.9 >= 0.85)
        normal = z[0] in g.step().fired_node_ids
        for i in range(1, 16):                           # a burst: 15 of 23 nodes at once
            g.stimulate(z[i], 2.0)
        g.step()
        g_burst = g.nodes.s2_state()[1]
        g.stimulate(z[19], 0.9 / 0.97)                   # the same weak input to a fresh node right after the burst
        held = z[19] not in g.step().fired_node_ids
        g_after = [g.nodes.s2_state()[1]]
        for k in range(4):
            S._pace(g, k)
            g.step()
            g_after.append(g.nodes.s2_state()[1])
        out[on] = (normal, g_burst, held, g_after)
    n_off, gb_off, held_off, _ = out[False]
    n_on, gb_on, held_on, after = out[True]
    assert n_off and n_on                                # normal-activity response intact
    assert gb_off == 0.0 and not held_off                # off: nothing pushes back
    assert gb_on == 1.0 and held_on                      # on: the burst (excess ~0.8 x 10) is bounded at g_max
    assert all(a <= 1.0 for a in after) and after[-1] < 0.1   # decays fast (x 0.5 per step)


@needs_s2
def test_burst_inhibition_scales_with_the_size_of_the_burst():
    peaks = []
    for burst in (3, 6, 12):
        g, z = _inh_graph(True, n=40)
        g.config["burst_inh_max"] = 100.0
        for k in range(30):
            S._pace(g, k)
            g.step()
        for i in range(burst):
            g.stimulate(z[i], 2.0)
        g.step()
        peaks.append(g.nodes.s2_state()[1])
    assert 0.0 < peaks[0] < peaks[1] < peaks[2]


@needs_s2
def test_burst_inhibition_with_a_pre_fire_handler_logs_error_once_and_stays_off(caplog):
    g, z = _inh_graph(True)
    g._event_handlers.setdefault("pre_fire", []).append(lambda *a, **k: 0.0)
    with caplog.at_level(logging.ERROR):
        for k in range(3):
            S._pace(g, k)
            g.step()
    assert sum("burst_inhibition_enabled" in r.getMessage() for r in caplog.records) == 1
    assert g.nodes.s2_state()[1] == 0.0
