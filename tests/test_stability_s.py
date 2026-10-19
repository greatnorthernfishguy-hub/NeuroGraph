# ---- Changelog ----
# [2026-10-19] Claude (lane stability-s, #1136 phase S) — the stability set: off == 11d3898, the H oracle, one test set per
#   mechanism (C1 / H / N / S / C2 / C3 / #1134), the dict-path gate
# What: (1) every key absent (and every key explicitly False) == the trial tip 11d3898 exactly, on the dict path, the
#       NodeStore-only path and the native path (both stores): every operation of the H2 drive workload + a sleep compared
#       strictly (fired ids, voltages bitwise, every hyperedge field, pass state, events) and the final checkpoint BYTES.
#       (2) The H oracle — the 31 reference tests of cc-laptop-he-habituation-20261016 (c98f66c) retargeted to the native
#       path: the reference module (dict path, its members handed the store's order) is the oracle and H alone must match
#       it bitwise, step and both sweep modes, 6 seeds; plus the reference's own behaviour tests run on the native path
#       (gain, fade vs rare, first injection = gain, EMA before the update, no-HEs on == off, round trip, keys).
#       (3) C1 / C2 / C3 / N / S / #1134 behaviour tests (no Python oracle by design, spec §3.2 / Q2). (4) The dict-backed
#       graph: a set key logs ERROR once and stays off.
# Why:  spec superpowers/specs/2026-10-17-rust-hyperedge-store-design.md §3.1 / §3.2 / §4 S.
# How:  oracles loaded with `git show <rev>:neuro_foundation.py` under their own names; the H2 test helpers (OSet, Side,
#       state digests) reused; PYTHONHASHSEED fixed by the runner.
# -------------------
"""#1136 phase S: the stability set (C1, H, N, S, C2, C3) + #1134, Rust only."""
import logging
import os
import random
import struct
import tempfile

import msgpack
import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402  (helpers; loads its own ac9dd38 oracle)

NF = H2.NF
_load = H2._load
TIP_REV = os.environ.get("NG_S_TIP_REV", "11d3898")
REF_REV = os.environ.get("NG_S_REF_REV", "c98f66c")       # cc-laptop-he-habituation-20261016 (the H oracle)
TIP = _load(TIP_REV, "nf_tip_s")
REF = _load(REF_REV, "nf_ref_he_hab_s")
HAVE_S = hasattr(NF.ng_tract, "NodeStore") and hasattr(NF.ng_tract.NodeStore, "evidence_scales") \
    and hasattr(NF.ng_tract.HyperedgeStore, "release_weak_targets")
needs_s = pytest.mark.skipif(not HAVE_S, reason="ng_tract without the S-phase kernels")

S_KEYS = ("he_habituation_enabled", "he_habituation_half", "he_habituation_power", "he_inject_respects_refractory",
          "he_threshold_homeo_enabled", "he_rate_target", "he_threshold_homeo_rate", "he_target_unlearn_enabled",
          "he_target_unlearn_d", "he_target_unlearn_p", "he_target_unlearn_floor", "node_homeo_evidence_enabled",
          "node_homeo_v_floor", "node_homeo_leak", "node_homeo_v_alpha", "syn_scale_evidence_enabled",
          "syn_scale_rate_tau", "syn_scale_max_up", "syn_scale_max_down", "homeostatic_degree_targets_persist")
SWITCHES = ("he_habituation_enabled", "he_inject_respects_refractory", "he_threshold_homeo_enabled",
            "he_target_unlearn_enabled", "node_homeo_evidence_enabled", "syn_scale_evidence_enabled",
            "homeostatic_degree_targets_persist")
FULL = {k: True for k in SWITCHES}
fb = H2.fb


def graph_of(M, kind, config=None):
    if kind == "native":
        g = M.Graph(config, native_node_store=True, native_hyperedge_store=True)
        assert g._native_he_passes
        return g
    if kind == "nodes":
        return M.Graph(config, native_node_store=True, native_hyperedge_store=False)
    return M.Graph(config, native_node_store=False, native_hyperedge_store=False) if M is not REF else M.Graph(config)


def restore_as(M, kind, b):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        with open(p, "wb") as f:
            f.write(b)
        g = graph_of(M, kind)
        g.restore(p)
        return g


def members_of(g, m, hid, he):
    return m.member_list(hid) if getattr(g, "_native_he", False) else list(he.member_nodes)


def state(g):
    return (
        tuple((nid, fb(x.voltage), x.refractory_remaining, fb(x.intrinsic_excitability), fb(x.threshold),
               fb(x.firing_rate_ema)) for nid, x in g.nodes.items()),
        tuple((hid, H2.he_state(he, members_of(g, g.hyperedges, hid, he))) for hid, he in g.hyperedges.items()),
        tuple((hid, H2.he_state(he, members_of(g, g._archived_hyperedges, hid, he)))
              for hid, he in g._archived_hyperedges.items()),
        tuple(sorted((k, tuple(sorted(v))) for k, v in g._node_hyperedges.items())),
        tuple((pid, ps.hyperedge_id, tuple(sorted(ps.predicted_targets)), fb(ps.prediction_strength))
              for pid, ps in g._active_predictions.items()),
        tuple(g._he_last_fired_step.items()),
        tuple((h, tuple(c.items())) for h, c in g._he_output_candidates.items()),
        g._prediction_counter,
        tuple(sorted((sid, fb(s.weight)) for sid, s in g.synapses.items())),
    )


class Pair:
    """Two graphs (oracle A, candidate B) in lock-step; each op applied to both with identical random / uuid streams;
    state compared strictly after every op. `hand_order` = give A's hyperedges B's native member order (OSet)."""

    def __init__(self, ga, gb, seed, hand_order=False):
        self.a, self.b = H2.Side(ga, seed), H2.Side(gb, seed)
        self.hand = hand_order
        self.ops = 0
        self.sync()

    def sync(self):
        if self.hand:
            H2.sync_order(self.a.g, self.b.g)

    def op(self, name, fn):
        ra, rb = self.a(fn), self.b(fn)
        self.sync()
        assert ra == rb, f"{name}: result {ra!r} != {rb!r}"
        assert self.a.log == self.b.log, f"{name}: events differ {H2._first_diff(tuple(self.a.log), tuple(self.b.log))}"
        x, y = state(self.a.g), state(self.b.g)
        assert x == y, f"{name} (op {self.ops}): state differs at {H2._first_diff(x, y)}"
        self.a.log.clear()
        self.b.log.clear()
        self.ops += 1
        return rb


def drive(t, rng, rounds, sleep_every=None):
    """The H2 workload (step + sweeps + rewards + discovery + removal + consolidation / split / shed) + sleep_cycle."""
    class _T:
        n = t.b

        def op(self, name, fn):
            return t.op(name, fn)
    H2.drive(_T(), rng, rounds)
    if sleep_every:
        t.op("sleep", lambda g: {k: v for k, v in g.sleep_cycle().items() if k != "seconds"})


def ckpt(g):
    return H2.ckpt_bytes(g)


# ---------------------------------------------------------------------------
# (1) every key absent / explicitly False == 11d3898, bytes and behaviour, on every path
# ---------------------------------------------------------------------------

@needs_s
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_trial_tip_exactly(kind, seed, how):
    start = H2.built(seed)
    t = Pair(restore_as(TIP, kind, start), restore_as(NF, kind, start), seed)
    if how == "explicit_false":
        t.b.g.config.update({k: False for k in SWITCHES})
    drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    if how == "explicit_false":
        for k in SWITCHES:
            t.b.g.config.pop(k)
    assert ckpt(t.a.g) == ckpt(t.b.g)                     # byte for byte, config included
    assert not hasattr(t.b.g, "_stab")                     # no new Graph state while off


@needs_s
def test_keys_not_in_default_config_and_default_config_is_the_tips():
    for k in S_KEYS:
        assert k not in NF.DEFAULT_CONFIG and k not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(TIP.DEFAULT_CONFIG.items())


# ---------------------------------------------------------------------------
# (2) the H oracle: the reference module (dict path) == the native path with H alone
# ---------------------------------------------------------------------------

HAB = {"he_habituation_enabled": True}


@needs_s
@pytest.mark.parametrize("seed", list(range(6)))
def test_h_alone_native_equals_the_reference_bitwise(seed):
    """The reference's own 'native parity with switch on', retargeted: its module is the oracle, every op compared."""
    start = H2.built(seed)
    t = Pair(restore_as(REF, "dict", start), restore_as(NF, "native", start), seed, hand_order=True)
    t.op("keys", lambda g: g.config.update(HAB))
    drive(t, random.Random(seed), 30)
    assert t.ops > 80
    assert t.b.g.hyperedges.pass_declines() == (0, 0, 0, 0)
    assert H2.ckpt_cmp(ckpt(t.a.g)) == H2.ckpt_cmp(ckpt(t.b.g))


@needs_s
@pytest.mark.parametrize("cfg", [{"he_habituation_half": 0.07, "he_habituation_power": 3.0},
                                 {"he_habituation_power": 1.0}, {"he_habituation_half": 0.0}])
def test_h_gain_parameters_equal_the_reference(cfg):
    start = H2.built(1)
    t = Pair(restore_as(REF, "dict", start), restore_as(NF, "native", start), 1, hand_order=True)
    t.op("keys", lambda g: g.config.update(dict(HAB, **cfg)))
    drive(t, random.Random(1), 12)


@needs_s
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_h_off_native_equals_the_reference_off(seed):
    start = H2.built(seed)
    t = Pair(restore_as(REF, "dict", start), restore_as(NF, "native", start), seed, hand_order=True)
    t.op("keys", lambda g: g.config.update({"he_habituation_enabled": False}))
    drive(t, random.Random(seed), 15)


def _two_assemblies(M, kind, on):
    g = graph_of(M, kind, {"he_habituation_enabled": on, "he_experience_threshold": 1, "decay_rate": 0.97,
                           "default_threshold": 0.85, "three_factor_enabled": False, "orphan_node_grace_period": 10 ** 9})
    L = [g.create_node(node_id=f"L{i}").node_id for i in range(4)]
    R = [g.create_node(node_id=f"R{i}").node_id for i in range(4)]
    t = g.create_node(node_id="Rt").node_id
    hl = g.create_hyperedge(set(L), activation_threshold=0.5, output_targets=list(L), output_weight=1.0).hyperedge_id
    hr = g.create_hyperedge(set(R), activation_threshold=0.5, output_targets=[t], output_weight=1.0).hyperedge_id
    for h in (hl, hr):
        he = g.hyperedges[h]
        he.is_learnable = False
        he.activation_count = 10
        he.pattern_completion_strength = 0.6
    g._plasticity_rules = [r for r in g._plasticity_rules if not isinstance(r, M.HyperedgePlasticityRule)]
    return g, L, R, t, hl, hr


@needs_s
def test_self_reigniting_assembly_fades_and_rare_one_keeps_full_drive_native():
    traces = {}
    for M, kind in ((REF, "dict"), (NF, "native")):
        for on in (False, True):
            g, L, R, t, hl, hr = _two_assemblies(M, kind, on)
            for nid in L:
                g.stimulate(nid, 2.0)
            lf = [hl in g.step().fired_hyperedge_ids for _ in range(200)]
            g.stimulate(R[0], 2.0)
            g.stimulate(R[1], 2.0)
            before = {nid: g.nodes[nid].voltage for nid in (R[2], R[3], t)}
            assert hr in g.step().fired_hyperedge_ids
            for nid in (R[2], R[3]):
                assert g.nodes[nid].voltage == pytest.approx(before[nid] * 0.97 + 0.6 * g.nodes[nid].intrinsic_excitability)
            traces[(kind, on)] = (lf, [fb(g.nodes[n].voltage) for n in L + R + [t]])
    assert traces[("dict", True)] == traces[("native", True)] and traces[("dict", False)] == traces[("native", False)]
    off, on = traces[("native", False)][0], traces[("native", True)][0]
    assert sum(off[100:]) >= 30 and sum(on[100:]) == 0 and sum(on[:20]) >= 2


@needs_s
@pytest.mark.parametrize("write_mode", [None, False, True])
def test_first_injection_is_the_references_bitwise(write_mode):
    out = {}
    for M, kind in ((REF, "dict"), (NF, "native")):
        for on in (False, True):
            g = graph_of(M, kind, {"he_habituation_enabled": on, "he_experience_threshold": 1,
                                   "three_factor_enabled": False, "orphan_node_grace_period": 10 ** 9})
            m = [g.create_node(node_id=f"m{i}").node_id for i in range(4)]
            t = g.create_node(node_id="t").node_id
            h = g.create_hyperedge(set(m), activation_threshold=0.5, output_targets=[t], output_weight=1.0).hyperedge_id
            he = g.hyperedges[h]
            he.activation_count = 10
            he.pattern_completion_strength = 0.5
            he.recent_activation_ema = 0.2
            g._plasticity_rules = []
            for nid in m[2:] + [t]:
                g.nodes[nid].threshold = 1e9
            if write_mode is None:
                g.stimulate(m[0], 2.0)
                g.stimulate(m[1], 2.0)
                assert h in g.step().fired_hyperedge_ids
            else:
                g.prime_and_propagate([m[0], m[1]], [2.0, 2.0], steps=1, write_mode=write_mode)
            out[(kind, on)] = (fb(g.nodes[m[2]].voltage), fb(g.nodes[t].voltage), he.refractory_remaining)
    assert out[("dict", True)] == out[("native", True)] and out[("dict", False)] == out[("native", False)]
    if write_mode is None:
        gain = 1.0 / (1.0 + (0.2 / 0.05) ** 2)
        c0, o0 = (struct.unpack("<d", x)[0] for x in out[("native", False)][:2])
        c1, o1 = (struct.unpack("<d", x)[0] for x in out[("native", True)][:2])
        assert c1 == pytest.approx(c0 * gain) and o1 == pytest.approx(o0 * gain)


@needs_s
def test_step_gain_reads_the_ema_before_this_steps_update_native():
    vs = {}
    for M, kind in ((REF, "dict"), (NF, "native")):
        g = graph_of(M, kind, {"he_habituation_enabled": True, "he_experience_threshold": 1})
        m = [g.create_node(node_id=f"m{i}").node_id for i in range(4)]
        h = g.create_hyperedge(set(m[:3]), activation_threshold=0.5, output_targets=[m[3]]).hyperedge_id
        g.hyperedges[h].recent_activation_ema = 0.1
        g.nodes[m[3]].threshold = 1e9
        for nid in m[:3]:
            g.stimulate(nid, 2.0)
        g.step()
        vs[kind] = (fb(g.nodes[m[3]].voltage), fb(g.hyperedges[h].recent_activation_ema))
    assert vs["dict"] == vs["native"]
    assert struct.unpack("<d", vs["native"][1])[0] > 0.1


@needs_s
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_hyperedges_switch_on_equals_off_native(seed):
    def strip(b):
        g = restore_as(NF, "native", b)
        for hid in list(g.hyperedges):
            g.remove_hyperedge(hid)
        return ckpt(g)
    start = strip(H2.built(seed))
    t = Pair(restore_as(NF, "native", start), restore_as(NF, "native", start), seed)
    t.b.g.config.update(HAB)
    drive(t, random.Random(seed), 15)


@needs_s
def test_switch_on_changes_behaviour_somewhere_native():
    start = H2.built(0)
    a, b = restore_as(NF, "native", start), restore_as(NF, "native", start)
    b.config.update(HAB)
    la, lb = [], []
    for g, log in ((a, la), (b, lb)):
        rng = random.Random(0)
        for _ in range(60):
            for nid in rng.sample(sorted(g.nodes), 8):
                g.stimulate(nid, 2.0)
            log.append((tuple(g.step().fired_node_ids), tuple(fb(n.voltage) for n in g.nodes.values())))
    assert la != lb


@needs_s
@pytest.mark.parametrize("persist", [True, False])
def test_round_trip_keeps_keys_ema_and_homeostat_with_1134(persist):
    """Round trip with the whole set on: keys, every HE / node field, v_ema / syn_rate_ema, C3 weights survive; a
    restore is a fixed point (two restores continue identically, bitwise); and with #1134's persistence the restored
    homeostat continues like the in-memory one (thresholds bitwise over the next steps), while without it the
    degree targets are lost and the thresholds diverge (the reference could only assert the fixed point). The saved
    member lists are sorted (#1140), so whole-state bitwise continuation vs the IN-MEMORY graph is not expected:
    the restored graph sums its members in sorted order."""
    start = H2.built(1)
    cfg = dict(FULL, he_habituation_half=0.07, he_habituation_power=3.0)
    if not persist:
        cfg.pop("homeostatic_degree_targets_persist")
    g = restore_as(NF, "native", start)
    g.config.update(cfg)
    rng = random.Random(5)
    for _ in range(40):
        for nid in rng.sample(sorted(g.nodes), 8):
            g.stimulate(nid, 2.0)
        g.step()
    mid = ckpt(g)
    g2, g3 = restore_as(NF, "native", mid), restore_as(NF, "native", mid)
    for k, v in cfg.items():
        assert g2.config[k] == v
    assert state(g2)[0] == state(g)[0]
    assert g2.nodes.stab_columns() == g.nodes.stab_columns()
    assert g2.hyperedges.target_weights_all() == g.hyperedges.target_weights_all()
    hr = lambda x: next(r for r in x._plasticity_rules if isinstance(r, NF.HomeostaticRule))   # noqa: E731
    assert hr(g)._degree_targets
    assert (hr(g2)._degree_targets == hr(g)._degree_targets) is persist
    thr = lambda x: tuple(fb(n.threshold) for n in x.nodes.values())   # noqa: E731
    same = True
    for _ in range(6):
        g.step()
        g2.step()
        same = same and thr(g) == thr(g2)
    assert same is persist
    t = Pair(g3, restore_as(NF, "native", mid), 9)
    drive(t, random.Random(9), 10)


# ---------------------------------------------------------------------------
# (3) the new mechanisms (no Python oracle by design)
# ---------------------------------------------------------------------------

def tiny(cfg=None, n=6):
    g = NF.Graph(dict({"three_factor_enabled": False, "orphan_node_grace_period": 10 ** 9, "he_experience_threshold": 1},
                      **(cfg or {})), native_node_store=True, native_hyperedge_store=True)
    ids = [g.create_node(node_id=f"x{i}").node_id for i in range(n)]
    return g, ids


@needs_s
@pytest.mark.parametrize("where", ["step", "sweep_write", "sweep_read"])
def test_c1_a_refractory_target_receives_nothing(where):
    got = {}
    for on in (False, True):
        g, x = tiny({"he_inject_respects_refractory": on})
        h = g.create_hyperedge({x[0], x[1]}, activation_threshold=0.5, output_targets=[x[2], x[3]]).hyperedge_id
        g.hyperedges[h].pattern_completion_strength = 0.0
        g._plasticity_rules = []
        for nid in (x[2], x[3]):
            g.nodes[nid].threshold = 1e9
        g.nodes[x[2]].refractory_remaining = 5         # x2 refractory, x3 not
        if where == "step":
            g.stimulate(x[0], 2.0)
            g.stimulate(x[1], 2.0)
            g.step()
            got[on] = (g.nodes[x[2]].voltage, g.nodes[x[3]].voltage)
        else:
            r = g.prime_and_propagate([x[0], x[1]], [2.0, 2.0], steps=1, write_mode=(where == "sweep_write"))
            got[on] = (g.nodes[x[2]].voltage, g.nodes[x[3]].voltage) if where == "sweep_write" else None
    if where == "sweep_read":
        return
    assert got[False][0] > 0 and got[True][0] == 0.0           # shunted while refractory
    assert got[True][1] == got[False][1] > 0                   # a ready target gets exactly today's push


@needs_s
def test_c2_an_overactive_assembly_threshold_climbs_a_rare_one_does_not():
    g, x = tiny({"he_threshold_homeo_enabled": True, "he_rate_target": 0.05, "he_threshold_homeo_rate": 0.01}, n=8)
    hot = g.create_hyperedge({x[0], x[1]}, activation_threshold=0.3).hyperedge_id
    rare = g.create_hyperedge({x[4], x[5]}, activation_threshold=0.3).hyperedge_id
    fixed = g.create_hyperedge({x[0], x[1]}, activation_threshold=0.3, is_learnable=False).hyperedge_id
    g.hyperedges[hot].recent_activation_ema = 0.3
    g.hyperedges[fixed].recent_activation_ema = 0.3
    g._plasticity_rules = []
    for _ in range(200):
        g.stimulate(x[0], 2.0)
        g.stimulate(x[1], 2.0)
        g.step()
    assert g.hyperedges[hot].activation_threshold > 0.3
    assert g.hyperedges[hot].activation_threshold <= 1.0
    assert g.hyperedges[rare].activation_threshold == 0.3
    assert g.hyperedges[fixed].activation_threshold == 0.3
    # upward only: a threshold above the cap is never pulled down to it
    g.hyperedges[hot].activation_threshold = 1.5
    g.step()
    assert g.hyperedges[hot].activation_threshold == 1.5


@needs_s
def test_c3_self_explained_target_decays_and_is_released_in_sleep_independent_one_holds():
    g, x = tiny({"he_target_unlearn_enabled": True, "he_inject_respects_refractory": True,
                 "he_target_unlearn_d": 0.1, "he_target_unlearn_p": 0.2, "he_target_unlearn_floor": 0.3}, n=8)
    h = g.create_hyperedge({x[0], x[1]}, activation_threshold=0.5, output_targets=[x[2], x[3]]).hyperedge_id
    g.hyperedges[h].pattern_completion_strength = 0.0
    g._plasticity_rules = []
    g.nodes[x[2]].threshold = 1e9                       # never fires: nothing but the HE's own push
    for k in range(40):
        if k % 5 == 0:
            g.stimulate(x[0], 2.0)
            g.stimulate(x[1], 2.0)
        if k % 5 == 2:
            g.stimulate(x[3], 5.0)                       # x3 fires on its own at lag 2: independent evidence
        g.step()
    n_rows, ws = g.hyperedges.target_weights_all()
    w = dict(zip(g.hyperedges[h].output_targets, ws))
    assert n_rows == 1 and w[x[2]] < 0.3 and w[x[3]] >= 0.9
    b = ckpt(g)
    d = msgpack.unpackb(b, raw=False, strict_map_key=False)
    assert set(d["hyperedges"][h]["output_target_weights"]) == {x[2], x[3]}
    rec = g.sleep_cycle()
    assert rec["he_targets_released"] == 1
    assert list(g.hyperedges[h].output_targets) == [x[3]]
    # the saved column round-trips; old readers ignore it (the tip's dict path reads named keys only)
    g2 = restore_as(NF, "native", ckpt(g))
    assert g2.hyperedges.target_weights_all() == g.hyperedges.target_weights_all()
    g3 = restore_as(TIP, "dict", ckpt(g))
    assert list(g3.hyperedges[h].output_targets) == [x[3]]


@needs_s
def test_c3_weight_scales_injection_and_off_rows_save_twenty_keys():
    g, x = tiny({"he_target_unlearn_enabled": True, "he_target_unlearn_d": 0.25}, n=5)
    h = g.create_hyperedge({x[0], x[1]}, activation_threshold=0.5, output_targets=[x[2]]).hyperedge_id
    other = g.create_hyperedge({x[3], x[4]}, activation_threshold=0.5).hyperedge_id
    g.hyperedges[h].pattern_completion_strength = 0.0
    g._plasticity_rules = []
    g.nodes[x[2]].threshold = 1e9
    pushes = []
    for _ in range(3):
        v0 = g.nodes[x[2]].voltage
        g.stimulate(x[0], 2.0)
        g.stimulate(x[1], 2.0)
        g.step()
        pushes.append(g.nodes[x[2]].voltage - v0 * g.config["decay_rate"])
        for _ in range(3):
            g.step()
            g.nodes[x[2]].voltage = 0.0
    assert pushes[0] == pytest.approx(1.0) and pushes[1] == pytest.approx(0.75) and pushes[2] == pytest.approx(0.5)
    d = msgpack.unpackb(ckpt(g), raw=False, strict_map_key=False)
    assert "output_target_weights" in d["hyperedges"][h] and len(d["hyperedges"][h]) == 21
    assert "output_target_weights" not in d["hyperedges"][other] and len(d["hyperedges"][other]) == 20


def _homeo_graph(cfg, n=4):
    """Nodes y0.. plus three pacemakers p0..p2 (one stimulated per step, so some node fires on EVERY step: step() runs
    the plasticity rules, homeostasis included, only on a step where something fired)."""
    base = {"target_firing_rate": 0.05, "scaling_interval": 5, "degree_sensitivity": 0.0, "default_threshold": 0.85,
            "orphan_node_grace_period": 10 ** 9, "three_factor_enabled": False}
    g = NF.Graph(dict(base, **cfg), native_node_store=True, native_hyperedge_store=True)
    ids = [g.create_node(node_id=f"y{i}").node_id for i in range(n)]
    for i in range(3):
        g.create_node(node_id=f"p{i}")
    return g, ids


def _pace(g, k):
    g.stimulate(f"p{k % 3}", 2.0)


@needs_s
def test_n_undriven_node_relaxes_to_baseline_driven_but_failing_rises_neither_hits_a_rail():
    g, y = _homeo_graph({"node_homeo_evidence_enabled": True, "node_homeo_leak": 0.1, "node_homeo_v_floor": 0.5})
    loaded, driven, fresh = y[0], y[1], y[2]
    g.nodes[loaded].intrinsic_excitability = 5.0         # today's rail load: cap + floor, no input
    g.nodes[loaded].threshold = 0.01
    g.nodes[driven].threshold = 4.0                      # depolarized (v -> 2.5 x excitability) but failing to fire
    g.nodes[fresh].firing_rate_ema = 0.0                 # silent + undriven: today x(1 + 5 rate) every interval
    peak = 0.0
    for k in range(300):
        _pace(g, k)
        g.stimulate(driven, 0.075)
        g.step()
        peak = max(peak, g.nodes[fresh].intrinsic_excitability, g.nodes[loaded].intrinsic_excitability if k > 50 else 0)
    L, D, F = (g.nodes[n] for n in (loaded, driven, fresh))
    assert L.intrinsic_excitability < 1.05 and L.threshold > 0.8         # unwound toward (1.0, 0.85)
    assert F.intrinsic_excitability == pytest.approx(1.0) and F.threshold == pytest.approx(0.85, abs=1e-6)
    assert peak < 4.99                                                    # no rail without evidence
    assert D.intrinsic_excitability > 1.2 and D.threshold < 4.0           # evidence: up-regulated
    v_on, _, v, _ = g.nodes.stab_columns()
    assert v_on and v[1] > 0.5 * D.threshold
    # today's rule on the same graph: the silent undriven node winds up to the rail
    g0, y0 = _homeo_graph({})
    for k in range(300):
        _pace(g0, k)
        g0.step()
    assert g0.nodes[y0[2]].intrinsic_excitability >= 4.99 and g0.nodes[y0[2]].threshold < 0.6   # winding to the rails


@needs_s
def test_n_over_firing_still_damps_and_v_ema_is_saved_only_when_on():
    g, y = _homeo_graph({"node_homeo_evidence_enabled": True})
    g.nodes[y[0]].threshold = 0.05
    for k in range(100):
        _pace(g, k)
        g.stimulate(y[0], 1.0)
        g.step()
    assert g.nodes[y[0]].threshold > 0.05 and g.nodes[y[0]].intrinsic_excitability < 1.0
    d = msgpack.unpackb(ckpt(g), raw=False, strict_map_key=False)
    assert all("v_ema" in nd and len(nd) == 20 for nd in d["nodes"].values())
    g0, _ = _homeo_graph({})
    g0.step()
    d0 = msgpack.unpackb(ckpt(g0), raw=False, strict_map_key=False)
    assert all(len(nd) == 19 for nd in d0["nodes"].values())
    # restore keeps the column on and its values; the tip's readers ignore it
    g2 = restore_as(NF, "native", ckpt(g))
    assert g2.nodes.stab_columns()[2] == g.nodes.stab_columns()[2]
    restore_as(TIP, "dict", ckpt(g))
    restore_as(TIP, "native", ckpt(g))


@needs_s
def test_s_bounded_evidence_gated_scaling():
    cfg = {"syn_scale_evidence_enabled": True, "syn_scale_max_up": 1.03, "syn_scale_max_down": 0.97,
           "syn_scale_rate_tau": 50.0, "node_homeo_v_floor": 0.5, "strength_budget_enabled": False}
    g, y = _homeo_graph(cfg, n=6)
    src, quiet, driven, hot = y[0], y[1], y[2], y[3]
    for post in (quiet, driven, hot):
        g.create_synapse(src, post, weight=0.5)
    g.nodes[src].threshold = 1e9                         # the source never fires: weights move only by scaling
    g.nodes[driven].threshold = 4.0                      # depolarized but failing (v -> 2.5 x excitability)
    g.nodes[hot].threshold = 0.05
    w = lambda post: next(s.weight for s in g.synapses.values() if s.post_node_id == post)  # noqa: E731
    hist = []
    for k in range(200):
        _pace(g, k)
        g.stimulate(driven, 0.075)
        g.stimulate(hot, 1.0)
        g.nodes[quiet].firing_rate_ema = 0.001           # quiet, but the long rate is not silent
        g.step()
        hist.append((w(quiet), w(driven), w(hot)))
    assert w(quiet) == 0.5                               # undriven quiet node: never up-scaled
    assert w(driven) > 0.5 and w(hot) < 0.5
    for (q0, d0, h0), (q1, d1, h1) in zip(hist, hist[1:]):
        assert d1 <= d0 * 1.03 + 1e-12 and h1 >= h0 * 0.97 - 1e-12  # bounded per interval
    _, r_on, _, rl = g.nodes.stab_columns()
    assert r_on and rl is not None
    d = msgpack.unpackb(ckpt(g), raw=False, strict_map_key=False)
    assert all("syn_rate_ema" in nd for nd in d["nodes"].values())


@needs_s
def test_1134_degree_targets_persisted_only_with_the_key():
    g = restore_as(NF, "native", H2.built(2))
    rng = random.Random(2)
    for _ in range(30):
        for nid in rng.sample(sorted(g.nodes), 8):
            g.stimulate(nid, 2.0)
        g.step()
    assert "homeostatic_degree_targets" not in msgpack.unpackb(ckpt(g), raw=False, strict_map_key=False)
    g.config["homeostatic_degree_targets_persist"] = True
    d = msgpack.unpackb(ckpt(g), raw=False, strict_map_key=False)
    hr = next(r for r in g._plasticity_rules if isinstance(r, NF.HomeostaticRule))
    assert d["homeostatic_degree_targets"] == hr._degree_targets and hr._degree_targets
    for kind in ("dict", "native"):
        g2 = restore_as(NF, kind, ckpt(g))
        assert next(r for r in g2._plasticity_rules if isinstance(r, NF.HomeostaticRule))._degree_targets \
            == hr._degree_targets


# ---------------------------------------------------------------------------
# (4) the dict-backed graph: ERROR once, stays off
# ---------------------------------------------------------------------------

@needs_s
@pytest.mark.parametrize("kind", ["dict", "nodes"])
def test_dict_backed_graph_logs_error_once_and_stays_off(kind, caplog):
    start = H2.built(3)
    t = Pair(restore_as(TIP, kind, start), restore_as(NF, kind, start), 3)
    keys = {k: True for k in SWITCHES if k != "homeostatic_degree_targets_persist"}
    if kind == "nodes":   # N / S run on the NodeStore alone; only the hyperedge keys need the native passes
        keys = {k: True for k in ("he_habituation_enabled", "he_inject_respects_refractory",
                                  "he_threshold_homeo_enabled", "he_target_unlearn_enabled")}
    t.b.g.config.update(keys)
    with caplog.at_level(logging.ERROR, logger=NF.logger.name):
        drive(t, random.Random(3), 8, sleep_every=True)
    errs = [r for r in caplog.records if "stability set (#1136)" in r.getMessage()]
    assert len(errs) == (2 if kind == "dict" else 1)
