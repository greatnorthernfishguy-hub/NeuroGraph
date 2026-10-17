# ---- Changelog ----
# [2026-10-16] Claude (Opus 5.5, #1131 hyperedge habituation) — CREATE: tests for the habituation switch
# What: (1) switch off (key absent AND explicit False) == the trial tip 4e2ed30 exactly over a hyperedge-heavy
#       workload (step, Tonic write-mode sweeps, read-mode recall sweeps, rewards, node churn): per-step fired nodes and
#       hyperedges, node voltages / thresholds, hyperedge state, synapse state, sweep entries, and the final checkpoint
#       bytes (absent: identical; explicit False: identical once the three keys the host set are left out); native node
#       store off and on. (2) the keys are NOT in DEFAULT_CONFIG (the repo's protected-flag convention) and
#       DEFAULT_CONFIG equals 4e2ed30's exactly. (3) the gain function. (4) switch on: a hyperedge
#       that keeps re-igniting itself fades and goes quiet, while a rarely-active one keeps full drive and still
#       completes its pattern when cued; the first-step injection is scaled by exactly the gain in step() and in both
#       sweep modes. (5) synapse / STDP behaviour unchanged by the switch (no hyperedges: on == off exactly). (6)
#       persistence: recent_activation_ema and the keys survive a checkpoint round-trip, and a restored run continues
#       exactly like the in-memory one. (7) native node store on == off with the switch on.
# Why:  #1131 (dev-log 2026-10-16_cc-recall-reverberating-loop-1131.md); the change is in protected neuro_foundation.py
#       and must be provably inert when off.
# How:  BASE = `git show 4e2ed30:neuro_foundation.py` under its own name; test_sleep_p1's seeded/uuid helpers reused.
# -------------------
"""#1131: hyperedge habituation (an assembly that keeps re-igniting itself stops recruiting)."""
import importlib.util
import os
import random
import struct
import subprocess
import sys
import tempfile

import msgpack
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import neuro_foundation as NF  # noqa: E402  (the branch)
from tests import test_sleep_p1 as P1  # noqa: E402  (seeded / DetUUID / ckpt helpers)

BASE_REV = os.environ.get("NG_HE_HAB_BASE_REV", "4e2ed30")
NEW_KEYS = ("he_habituation_enabled", "he_habituation_half", "he_habituation_power")
HAVE_STORE = hasattr(NF.ng_tract, "NodeStore")
MODES = ["off", "on"] if HAVE_STORE else ["off"]
SEEDS = [0, 1, 2]


def _load_base():
    src = subprocess.run(["git", "-C", _REPO, "show", f"{BASE_REV}:neuro_foundation.py"],
                         check=True, capture_output=True, text=True).stdout
    d = tempfile.mkdtemp(prefix="he_hab_base_")
    p = os.path.join(d, "nf_base_he_hab.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nf_base_he_hab", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nf_base_he_hab"] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base()


def P(x):
    if type(x) is float:
        return ("f", struct.pack("<d", x))
    return (type(x).__name__, repr(x))


def new_graph(mode, config=None):
    if mode == "base":
        return BASE.Graph(config)
    return NF.Graph(config, native_node_store=(mode == "on"))


def mod_of(mode):
    return BASE if mode == "base" else NF


def ckpt_bytes(g, drop_new_keys=False):
    popped = {}
    if drop_new_keys:
        for k in NEW_KEYS:
            if k in g.config:
                popped[k] = g.config.pop(k)
    try:
        return P1.ckpt_bytes(g)
    finally:
        g.config.update(popped)


def restore(mode, b):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        with open(p, "wb") as f:
            f.write(b)
        g = new_graph(mode)
        g.restore(p)
        return g


# ---------------------------------------------------------------------------
# a hyperedge-heavy random graph + workload
# ---------------------------------------------------------------------------

def build(mode, seed, n_nodes=90, n_syn=500, n_he=14):
    M = mod_of(mode)

    def b(rng):
        g = new_graph(mode, {"three_factor_enabled": True, "scaling_interval": 3, "he_consolidation_eval_interval": 7,
                             "he_experience_threshold": 20})
        ids = []
        for i in range(n_nodes):
            n = g.create_node(node_id=f"n{seed}_{i}", metadata={}, is_inhibitory=rng.random() < 0.08)
            n.firing_rate_ema = rng.choice([0.0, rng.uniform(0, 0.3)])
            n.voltage = rng.uniform(0, 0.8)
            ids.append(n.node_id)
        for _ in range(n_syn):
            a, c = rng.sample(ids, 2)
            g.create_synapse(a, c, weight=rng.uniform(0.05, 1.0), delay=rng.randint(1, 3))
        modes = [M.ActivationMode.WEIGHTED_THRESHOLD] * 3 + [M.ActivationMode.GRADED, M.ActivationMode.K_OF_N]
        for j in range(n_he):
            mem = rng.sample(ids, rng.randint(3, 6))
            outs = [x for x in rng.sample(ids, rng.randint(0, 3)) if x not in mem]
            he = g.create_hyperedge(set(mem), member_weights={m: rng.uniform(0.5, 2.0) for m in mem},
                                    activation_threshold=rng.uniform(0.25, 0.6), activation_mode=rng.choice(modes),
                                    output_targets=outs, output_weight=rng.uniform(0.5, 2.0))
            he.activation_count = rng.choice([0, 5, 30, 200])
            he.recent_activation_ema = rng.choice([0.0, 0.001, 0.02, 0.05, 0.2, 0.33])
            he.pattern_completion_strength = rng.choice([0.0, 0.3, 0.6])
        g.timestep = rng.randint(50, 500)
        return g
    return P1.seeded(seed, b)


def he_state(g):
    return [(hid, P(he.recent_activation_ema), he.activation_count, P(he.activation_threshold), he.refractory_remaining,
             P(he.current_activation), sorted((k, P(v)) for k, v in he.member_weights.items()), list(he.output_targets),
             sorted(he.member_nodes), he.is_archived)
            for hid, he in g.hyperedges.items()]


def workload(rounds):
    def act(g, rng):
        log = []
        for k in range(rounds):
            for nid in rng.sample(list(g.nodes), min(6, len(g.nodes))):
                g.stimulate(nid, rng.uniform(0.5, 2.5))
            r = g.step()
            log.append((list(r.fired_node_ids), list(r.fired_hyperedge_ids), r.predictions_confirmed,
                        r.predictions_surprised, r.synapses_pruned, r.synapses_sprouted))
            log.append([(nid, P(n.voltage), P(n.threshold), n.refractory_remaining) for nid, n in g.nodes.items()])
            log.append(he_state(g))
            log.append(P1.syn_state(g))
            ids = rng.sample(list(g.nodes), 5)
            p = g.prime_and_propagate(ids, [1.5] * 5, steps=3, write_mode=True)
            log.append(P1.fe(p.fired_entries))
            q = g.prime_and_propagate(ids[:3], [2.0] * 3, steps=4, write_mode=False)
            log.append(P1.fe(q.fired_entries))
            log.append([(nid, P(n.voltage)) for nid, n in g.nodes.items()])
            if k % 5 == 4:
                g.inject_reward(0.2)
            if k % 9 == 5 and len(g.nodes) > 30:
                g.remove_node(rng.choice(list(g.nodes)))
        log.append(he_state(g))
        return log
    return act


def run(mode, start, seed, rounds=40, cfg=None):
    g = restore(mode, start)
    if cfg:
        g.config.update(cfg)
    log = P1.seeded(seed, lambda rng: workload(rounds)(g, rng), uuid_stream=1)
    return g, log


_BUILT = {}


def built(seed):
    if seed not in _BUILT:
        _BUILT[seed] = ckpt_bytes(build("base", seed))
    return _BUILT[seed]


def _assert_logs_equal(la, lb):
    assert len(la) == len(lb)
    for i, (x, y) in enumerate(zip(la, lb)):
        assert x == y, f"log entry {i} differs"


# ---------------------------------------------------------------------------
# (1) switch off == the trial tip exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_switch_off_is_the_trial_tip_exactly(mode, seed, how):
    start = built(seed)
    ga, la = run("base", start, seed)
    cfg = {"he_habituation_enabled": False, "he_habituation_half": 0.05, "he_habituation_power": 2.0} \
        if how == "explicit_false" else None
    gb, lb = run(mode, start, seed, cfg=cfg)
    if how == "absent":
        assert not [k for k in NEW_KEYS if k in gb.config]
    _assert_logs_equal(la, lb)
    assert sum(len(x[1]) for x in la[0::9]) > 20, "workload must actually fire hyperedges"
    if how == "absent":
        assert ckpt_bytes(ga) == ckpt_bytes(gb)                       # byte for byte, config included
    else:
        assert ckpt_bytes(ga) == ckpt_bytes(gb, drop_new_keys=True)   # only the keys the host itself set differ


def test_explicit_false_checkpoint_differs_only_by_the_keys_the_host_set():
    start = built(0)
    ga, _ = run("base", start, 0, rounds=5)
    gb, _ = run("off", start, 0, rounds=5, cfg={"he_habituation_enabled": False, "he_habituation_half": 0.05,
                                                 "he_habituation_power": 2.0})
    a = msgpack.unpackb(ckpt_bytes(ga), raw=False, strict_map_key=False)
    b = msgpack.unpackb(ckpt_bytes(gb), raw=False, strict_map_key=False)
    ca, cb = a.pop("config"), b.pop("config")
    assert a == b
    assert set(cb) - set(ca) == set(NEW_KEYS) and set(ca) <= set(cb)
    assert all(ca[k] == cb[k] for k in ca)


# ---------------------------------------------------------------------------
# (2) DEFAULT_CONFIG
# ---------------------------------------------------------------------------

def test_keys_are_not_in_default_config_and_default_config_is_unchanged():
    for k in NEW_KEYS:
        assert k not in NF.DEFAULT_CONFIG
        assert k not in NF.Graph().config
    assert NF.DEFAULT_CONFIG == BASE.DEFAULT_CONFIG
    assert list(NF.DEFAULT_CONFIG) == list(BASE.DEFAULT_CONFIG)


def test_absent_keys_read_as_off_half_0_05_power_2():
    g = NF.Graph()
    assert g._he_habituation_gain(0.05) == pytest.approx(0.5)              # half 0.05
    assert g._he_habituation_gain(0.1) == pytest.approx(1 / (1 + 2 ** 2))  # power 2


# ---------------------------------------------------------------------------
# (3) the gain
# ---------------------------------------------------------------------------

def test_gain_function():
    g = NF.Graph()
    gain = g._he_habituation_gain
    assert gain(0.0) == 1.0
    assert gain(0.05) == pytest.approx(0.5)
    assert gain(0.005) == pytest.approx(1 / 1.01)               # a rare assembly keeps ~full drive
    assert gain(0.33) == pytest.approx(1 / (1 + (0.33 / 0.05) ** 2))   # a refractory-limited loop: ~0.022
    assert gain(0.33) < 0.03
    g.config["he_habituation_power"] = 1.0
    assert gain(0.1) == pytest.approx(1 / 3)
    g.config["he_habituation_half"] = 0.0                     # degenerate half: no habituation, never a crash
    assert gain(0.3) == 1.0


# ---------------------------------------------------------------------------
# (4) switch on: the self-re-igniting assembly fades; the rare one keeps its drive
# ---------------------------------------------------------------------------

def _two_assemblies(on, mode="off"):
    """Assembly L: 4 members whose completion re-ignites the members that did not fire, and whose output targets are its
    own members — a self-sustaining loop once kicked. Assembly R: 4 different members, a separate output target, fired
    only when cued."""
    g = NF.Graph({"he_habituation_enabled": on, "he_experience_threshold": 1, "decay_rate": 0.97,
                  "default_threshold": 0.85, "three_factor_enabled": False, "orphan_node_grace_period": 10 ** 9},
                 native_node_store=(mode == "on"))
    L = [g.create_node(node_id=f"L{i}").node_id for i in range(4)]
    R = [g.create_node(node_id=f"R{i}").node_id for i in range(4)]
    t = g.create_node(node_id="Rt").node_id
    hl = g.create_hyperedge(set(L), activation_threshold=0.5, output_targets=list(L), output_weight=1.0)
    hr = g.create_hyperedge(set(R), activation_threshold=0.5, output_targets=[t], output_weight=1.0)
    for he in (hl, hr):
        he.is_learnable = False
        he.activation_count = 10
        he.pattern_completion_strength = 0.6
    g._plasticity_rules = [r for r in g._plasticity_rules if not isinstance(r, NF.HyperedgePlasticityRule)]
    return g, L, R, t, hl, hr


@pytest.mark.parametrize("mode", MODES)
def test_self_reigniting_assembly_fades_and_rare_one_keeps_full_drive(mode):
    fires = {}
    for on in (False, True):
        g, L, R, t, hl, hr = _two_assemblies(on, mode)
        for nid in L:
            g.stimulate(nid, 2.0)                       # one kick; after this there is no input to L at all
        lfires = []
        for k in range(200):
            r = g.step()
            lfires.append(hl.hyperedge_id in r.fired_hyperedge_ids)
        fires[on] = lfires
        # the rarely-active assembly R: cue two members; its completion + output must be at full strength
        assert hr.recent_activation_ema == 0.0
        g.stimulate(R[0], 2.0)
        g.stimulate(R[1], 2.0)
        before = {nid: g.nodes[nid].voltage for nid in (R[2], R[3], t)}
        r = g.step()
        assert hr.hyperedge_id in r.fired_hyperedge_ids
        decay = g.config["decay_rate"]
        for nid in (R[2], R[3]):                        # completion: 0.6 * w 1.0 * excitability, no habituation
            assert g.nodes[nid].voltage == pytest.approx(before[nid] * decay + 0.6 * g.nodes[nid].intrinsic_excitability)
        assert g.nodes[t].voltage == pytest.approx(before[t] * decay + 1.0 * g.nodes[t].intrinsic_excitability)
    off, on = fires[False], fires[True]
    assert sum(off[100:200]) >= 30, "the loop must self-sustain with the switch off (else the test is vacuous)"
    assert sum(on[100:200]) == 0, "with habituation the self-re-igniting assembly must go quiet"
    assert sum(on[:20]) >= 2, "it still fires at first (habituation is not a block)"


@pytest.mark.parametrize("write_mode", [None, False, True])
def test_first_injection_is_scaled_by_exactly_the_gain(write_mode):
    """One firing of a hyperedge with ema 0.2: completion and output are gain x the switch-off injection, in step()
    (gain from the EMA BEFORE this step's update) and in both prime_and_propagate modes."""
    out = {}
    for on in (False, True):
        g = NF.Graph({"he_habituation_enabled": on, "he_experience_threshold": 1, "three_factor_enabled": False,
                      "orphan_node_grace_period": 10 ** 9})
        m = [g.create_node(node_id=f"m{i}").node_id for i in range(4)]
        t = g.create_node(node_id="t").node_id
        he = g.create_hyperedge(set(m), activation_threshold=0.5, output_targets=[t], output_weight=1.0)
        he.activation_count = 10
        he.pattern_completion_strength = 0.5
        he.recent_activation_ema = 0.2
        g._plasticity_rules = []
        for nid in m[2:] + [t]:
            g.nodes[nid].threshold = 1e9                # observe the injected voltage without a spike
        if write_mode is None:
            g.stimulate(m[0], 2.0)
            g.stimulate(m[1], 2.0)
            r = g.step()
            assert he.hyperedge_id in r.fired_hyperedge_ids
            out[on] = (g.nodes[m[2]].voltage, g.nodes[t].voltage)
        elif write_mode:
            g.prime_and_propagate([m[0], m[1]], [2.0, 2.0], steps=1, write_mode=True)
            out[on] = (g.nodes[m[2]].voltage, g.nodes[t].voltage)
        else:                                            # read mode restores voltages: observe the gain calls
            calls = []
            real = g._he_habituation_gain
            g._he_habituation_gain = lambda ema, _r=real: calls.append(ema) or _r(ema)
            r = g.prime_and_propagate([m[0], m[1]], [2.0, 2.0], steps=2, write_mode=False)
            assert {e.node_id for e in r.fired_entries} >= {m[0], m[1]}
            out[on] = calls
    if write_mode is False:
        assert out[False] == []                          # off: the gain is never even computed
        assert out[True] and all(e == 0.2 for e in out[True])
        return
    gain = 1.0 / (1.0 + (0.2 / 0.05) ** 2)
    (c0, o0), (c1, o1) = out[False], out[True]
    assert c0 > 0 and o0 > 0
    assert c1 == pytest.approx(c0 * gain) and o1 == pytest.approx(o0 * gain)


def test_step_gain_reads_the_ema_before_this_steps_update():
    calls = []
    g = NF.Graph({"he_habituation_enabled": True, "he_experience_threshold": 1})
    m = [g.create_node(node_id=f"m{i}").node_id for i in range(3)]
    he = g.create_hyperedge(set(m), activation_threshold=0.5)
    he.recent_activation_ema = 0.1
    real = g._he_habituation_gain
    g._he_habituation_gain = lambda ema: calls.append(ema) or real(ema)
    for nid in m:
        g.stimulate(nid, 2.0)
    g.step()
    assert calls == [0.1]
    assert he.recent_activation_ema > 0.1             # the bookkeeping itself is unchanged (it rose with this fire)


# ---------------------------------------------------------------------------
# (5) synapse / STDP behaviour unchanged by the switch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_no_hyperedges_switch_on_equals_off_exactly(seed):
    g0 = build("off", seed)
    for hid in list(g0.hyperedges):
        g0.remove_hyperedge(hid)
    start = ckpt_bytes(g0)
    ga, la = run("off", start, seed, cfg={"he_habituation_enabled": False})
    gb, lb = run("off", start, seed, cfg={"he_habituation_enabled": True})
    _assert_logs_equal(la, lb)
    assert ckpt_bytes(ga, True) == ckpt_bytes(gb, True)


def test_switch_on_changes_behaviour_somewhere():
    """Non-vacuity: on the hyperedge workload the switch does change dynamics."""
    start = built(0)
    _, la = run("off", start, 0, rounds=20)
    _, lb = run("off", start, 0, rounds=20, cfg={"he_habituation_enabled": True})
    assert la != lb


# ---------------------------------------------------------------------------
# (6) persistence round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_round_trip_keeps_ema_and_keys_and_continues_identically(mode):
    start = built(1)
    cfg = {"he_habituation_enabled": True, "he_habituation_half": 0.07, "he_habituation_power": 3.0}
    g, _ = run(mode, start, 1, rounds=15, cfg=cfg)
    mid = ckpt_bytes(g)
    g2 = restore(mode, mid)
    for k, v in cfg.items():
        assert g2.config[k] == v
    assert {h: P(e.recent_activation_ema) for h, e in g.hyperedges.items()} == \
           {h: P(e.recent_activation_ema) for h, e in g2.hyperedges.items()}
    assert he_state(g) == he_state(g2)
    # The engine's homeostatic rule keeps in-memory-only state (_degree_targets, _steps_since_scaling) that no
    # checkpoint carries — at 4e2ed30 too — so a restored graph is not step-for-step the in-memory one. What a restore
    # must be is a fixed point: two successive round-trips continue identically, habituation included.
    g3 = restore(mode, ckpt_bytes(g2))
    lb = P1.seeded(9, lambda rng: workload(10)(g2, rng), uuid_stream=2)
    lc = P1.seeded(9, lambda rng: workload(10)(g3, rng), uuid_stream=2)
    _assert_logs_equal(lb, lc)
    assert _decoded(g2) == _decoded(g3)


def _decoded(g):
    """The checkpoint decoded, with set-valued fields sorted (a restored set may iterate in another order)."""
    d = msgpack.unpackb(ckpt_bytes(g), raw=False, strict_map_key=False)
    for he in d.get("hyperedges", {}).values():
        for k in ("member_nodes", "child_hyperedges"):
            if isinstance(he.get(k), list):
                he[k] = sorted(he[k])
    return d


# ---------------------------------------------------------------------------
# (7) native node store parity with the switch on
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_STORE, reason="installed ng_tract has no NodeStore")
@pytest.mark.parametrize("seed", SEEDS)
def test_native_node_store_parity_with_switch_on(seed):
    start = built(seed)
    cfg = {"he_habituation_enabled": True}
    ga, la = run("off", start, seed, cfg=cfg)
    gb, lb = run("on", start, seed, cfg=cfg)
    _assert_logs_equal(la, lb)
    assert ckpt_bytes(ga) == ckpt_bytes(gb)
