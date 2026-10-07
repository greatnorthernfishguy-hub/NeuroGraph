# ---- Changelog ----
# [2026-10-06] Claude (lane sleep-p1) — D15 (sleep-phase spec): the BASE engine gets D15's one named intended change
#   (tests/d15_intended.apply_d15_rule: drop pre.pred_weights[post] with the last pre->post synapse) so this file keeps checking
#   everything else bitwise; with D15 switched off the file passes unchanged (SLEEP_P1.md §6).
# [2026-10-06] Claude (lane nodestore-p2b) — CREATE: P2b equivalence (native SynapseStore.stdp_pass vs fallback vs trial tip)
# What: (1) exp exactness through the shipped binary: stdp_pass dw == the loop's math.exp expression, bitwise, over
#       every integer dt in a range past exp's underflow for each configured tau (the full 4.6M-result gate is
#       ~/.cache/p2b/exp_gate.py; this is its fast in-suite slice).
#       (2) per-step dw: STDPRule.apply on edge graphs — trial tip ef78c67 (dict of Node) vs this branch OFF (dict;
#       stdp_pass declines -> _stdp_python) vs ON native (stdp_pass) vs ON fallback (_stdp_python over NodeRefs);
#       every synapse's weight / eligibility / peak / last_update_time / max_weight bitwise (struct.pack('<d')).
#       Edge rows: dt == 0 (half-strength LTP), dt < 0 and > 0 on both passes (later-firing partners), soft
#       saturation (w at / near / above max_weight), NaN / +-inf last_spike_time, NaN / inf weights, diffpc_layer
#       outside 0..2, missing pre / post nodes, tombstoned and re-added ids, stale synapse ids in the adjacency,
#       duplicate fired ids, three_factor on and off, max_weight == 0 (ZeroDivisionError at the same point with the
#       same partial commit), a fired id with no node (KeyError).
#       (3) declines touch nothing; the node-row cache survives add / remove / compaction / clear.
#       (4) spec §5.2 item 5 whole runs: 30 rounds x the trial flag matrix (+ a pre_fire-handler set) x 6 seeds x
#       {ON native, OFF fallback}, every log entry (incl. per-step synapse state bitwise), snapshots and final
#       checkpoint bytes against the trial tip; ON asserts _stdp_python never ran.
# Why:  spec superpowers/specs/2026-10-05-native-node-store-design.md §4 P2b, §5.2 items 4-5, §10 item 7.
# How:  BASE modules are imported from git (ef78c67) under their own names. Fixed PYTHONHASHSEED (set by the harness).
# -------------------
"""P2b native STDP pass: bit-equivalence against the Python fallback and the trial tip."""
import importlib.util
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import types
import uuid

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import neuro_foundation as NF  # noqa: E402  (the branch)
import tonic_thread as TT  # noqa: E402
import activation_persistence as AP  # noqa: E402

BASE_REV = os.environ.get("NG_NODESTORE_P2B_BASE_REV", "ef78c67")   # trial tip this branch was cut from
HAVE_STORE = hasattr(NF.ng_tract, "NodeStore")
HAVE_P2B = hasattr(getattr(NF.ng_tract, "SynapseStore", None), "stdp_pass")
needs_p2b = pytest.mark.skipif(not (HAVE_STORE and HAVE_P2B), reason="ng_tract without NodeStore / stdp_pass")


def _load_base(fname, modname):
    src = subprocess.run(["git", "-C", _REPO, "show", f"{BASE_REV}:{fname}"],
                         check=True, capture_output=True, text=True).stdout
    d = tempfile.mkdtemp(prefix="p2b_base_")
    p = os.path.join(d, modname + ".py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(modname, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base("neuro_foundation.py", "nf_base_p2b")
from tests.d15_intended import apply_d15_rule  # noqa: E402  [2026-10-06] sleep-p1: D15's named intended change
apply_d15_rule(BASE)
BASE_TT = _load_base("tonic_thread.py", "tonic_thread_base_p2b")
BASE_AP = _load_base("activation_persistence.py", "activation_persistence_base_p2b")
MODES = ["off", "on"] if HAVE_STORE else ["off"]


def new_graph(mode, config=None):
    if mode == "base":
        return BASE.Graph(config)
    g = NF.Graph(config, native_node_store=mode.startswith("on"))
    assert isinstance(g.nodes, dict) == (not mode.startswith("on"))
    return g


def mod_of(mode):
    return BASE if mode == "base" else NF


def P(x):
    """Bitwise identity of a value: floats by their IEEE bytes, everything else by (type, repr)."""
    if type(x) is float:
        return ("f", struct.pack("<d", x))
    return (type(x).__name__, repr(x))


def syn_state(g):
    return [(sid, P(s.weight), P(s.eligibility_trace), P(s.peak_weight), P(s.last_update_time), P(s.max_weight))
            for sid, s in g.synapses.items()]


def ckpt_bytes(g):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        g.checkpoint(p)
        with open(p, "rb") as f:
            return f.read()


def restore(mode, b):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        with open(p, "wb") as f:
            f.write(b)
        g = new_graph(mode)
        g.restore(p)
        return g


class DetUUID:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def __call__(self):
        return uuid.UUID(int=self.rng.getrandbits(128), version=4)


def seeded(seed, fn, uuid_stream=0):
    saved = uuid.uuid4
    uuid.uuid4 = DetUUID(seed * 1000 + uuid_stream)
    random.seed(seed)
    try:
        return fn(random.Random(seed))
    finally:
        uuid.uuid4 = saved


NAN = float("nan")
INF = float("inf")
CURV = [[1.0] * 3 for _ in range(3)]


# ---------------------------------------------------------------------------
# (1) exp exactness through the shipped binary (fast slice of ~/.cache/p2b/exp_gate.py)
# ---------------------------------------------------------------------------

@needs_p2b
@pytest.mark.parametrize("tau", [10.0, 15.0, 20.0, 25.0, 30.0])
def test_stdp_pass_exp_is_math_exp_bitwise(tau):
    n = int(746 * tau) + 64                       # every integer dt up to past exp's underflow to 0.0
    g = new_graph("on")
    g.create_node(node_id="P")
    ts = 10 * n
    for k in range(n):
        g.create_node(node_id=f"q{k}").last_spike_time = float(ts - (k + 1))
        g.synapses.append(f"i{k}", f"q{k}", "P", weight=0.0, max_weight=1.0)
        g.synapses.append(f"o{k}", "P", f"q{k}", weight=0.0, max_weight=1.0)
    inc = {"P": {f"i{k}" for k in range(n)}}
    out = {"P": {f"o{k}" for k in range(n)}}
    assert g.synapses.stdp_pass(g.nodes, ["P"], ts, 1.0, 1.0, tau, tau, 1.0, True, CURV, inc, out) is True
    e = g.synapses.eligibility_copy()
    for k in range(n):
        dt = float(ts) - float(ts - (k + 1))
        assert P(float(e[2 * k])) == P(0.0 + 1.0 * math.exp(-dt / tau) * 1.0 * max(1.0, 0.0) * 1.0), (tau, dt)
        assert P(float(e[2 * k + 1])) == P(0.0 + -1.0 * math.exp(-dt / tau) * 1.0 * 1.0), (tau, -dt)


# ---------------------------------------------------------------------------
# (2) per-step dw on edge graphs: trial tip vs OFF vs ON native vs ON fallback
# ---------------------------------------------------------------------------

LAYERS = [0, 1, 2, 2, -1, 3, 7, 2 ** 40]


def build_edge_graph(seed, n=70, n_syn=520, mw_zero=False):
    rng = random.Random(seed)
    g = new_graph("base")
    ts0 = 500
    for i in range(n):
        node = g.create_node(node_id=f"s{seed}_{i}", metadata={"i": i})
        node.last_spike_time = rng.choice([-INF, -INF, float(ts0), float(ts0), float(ts0 - 1), float(ts0 + 3),
                                           float(ts0 - rng.randint(1, 400)), float(ts0 + rng.randint(1, 50)),
                                           NAN, INF, float(rng.randint(-5, 5))])
        node.diffpc_layer = rng.choice(LAYERS)
    ids = list(g.nodes)
    made = 0
    while made < n_syn:
        a, b = rng.sample(ids, 2)
        mw = rng.choice([5.0, 5.0, 1.0, 1e-300, INF, NAN, 2.5])
        s = g.create_synapse(a, b, weight=0.1, max_weight=mw)
        s.weight = rng.choice([0.0, 0.1, 1e-300, rng.uniform(0, 5), mw, mw * 0.999999999, 5.000000000000001, 6.0,
                               NAN, INF, -0.0])
        s.peak_weight = rng.choice([s.weight, 0.0, 5.0])
        s.eligibility_trace = rng.choice([0.0, -0.0, rng.uniform(-0.5, 0.5)])
        made += 1
    if mw_zero:
        for s in rng.sample(list(g.synapses.values()), 6):
            s.max_weight = rng.choice([0.0, -0.0])
    g.timestep = ts0
    return g


def edge_mutations(g, seed):
    """Applied identically after restore in every mode: missing pre/post nodes (rows whose synapses stay),
    tombstoned + re-added ids, a stale synapse id in the adjacency."""
    rng = random.Random(seed + 31)
    ids = list(g.nodes)
    for nid in rng.sample(ids, 4):                 # node gone, its synapses and adjacency entries stay
        del g.nodes[nid]
    for nid in rng.sample([i for i in ids if i in g.nodes], 3):
        g.remove_node(nid)
    for nid in sorted(rng.sample([i for i in ids if i not in g.nodes], 3)):
        node = g.create_node(node_id=nid, metadata={"readded": True})
        node.last_spike_time = 499.0
        node.diffpc_layer = 1
    live = [i for i in g.nodes]
    g._incoming.setdefault(live[0], set()).add("ghost-synapse-id")
    g._outgoing.setdefault(live[1], set()).add("ghost-synapse-id-2")
    return g


def fired_lists(g, seed):
    rng = random.Random(seed + 7)
    ids = list(g.nodes)
    out = []
    for k in range(4):
        f = rng.sample(ids, len(ids) // 3)
        out.append(f + f[:2])                      # duplicates: the pass repeats on the updated weights
    return out


_EDGE = {}


def edge_bytes(seed, mw_zero=False):
    if (seed, mw_zero) not in _EDGE:
        _EDGE[(seed, mw_zero)] = ckpt_bytes(build_edge_graph(seed, mw_zero=mw_zero))
    return _EDGE[(seed, mw_zero)]


def rule_of(mode, seed):
    return mod_of(mode).STDPRule(tau_plus=[10.0, 20.0, 15.0][seed % 3], tau_minus=[10.0, 20.0, 25.0][seed % 3],
                                 A_plus=1.2, A_minus=1.4, learning_rate=0.03)


def apply_once(mode, rule, g, fired, ts):
    if mode == "on_fb":
        NF._stdp_python(rule, g, fired, ts)
    else:
        rule.apply(g, fired, ts)


def edge_run(mode, seed, mw_zero=False, extra_fired=None):
    g = edge_mutations(restore("base" if mode == "base" else mode, edge_bytes(seed, mw_zero)), seed)
    g.config["three_factor_enabled"] = bool(seed % 2)
    rule = rule_of(mode, seed)
    log = []
    for k, fired in enumerate(fired_lists(g, seed)):
        if extra_fired and k == 2:
            fired = fired[:5] + [extra_fired] + fired[5:]
        try:
            apply_once(mode, rule, g, fired, 500 + k)
            log.append(("ok", syn_state(g)))
        except Exception as e:  # noqa: BLE001 — the exception and the partial commit are part of the contract
            log.append((type(e).__name__, syn_state(g)))
    return log, ckpt_bytes(g)


EDGE_MODES = ["off", "on", "on_fb"] if HAVE_STORE else ["off"]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("mode", EDGE_MODES)
def test_stdp_apply_edge_rows_vs_trial_tip(mode, seed, monkeypatch):
    calls = []
    orig = NF._stdp_python
    monkeypatch.setattr(NF, "_stdp_python", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    base = edge_run("base", seed)
    got = edge_run(mode, seed)
    assert [x[0] for x in got[0]] == ["ok"] * 4
    assert len({repr(x[1]) for x in base[0]}) == 4             # every apply changed something (non-vacuous)
    for i, (x, y) in enumerate(zip(base[0], got[0])):
        assert x == y, f"apply #{i} differs"
    assert got[1] == base[1]
    if mode == "on" and HAVE_P2B:
        assert calls == [], "the native pass declined"
    elif mode == "off":
        assert len(calls) == 4


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("mode", EDGE_MODES)
def test_stdp_apply_raises_at_the_same_point(mode, seed):
    """max_weight == 0 on an LTP / dt == 0 row: ZeroDivisionError after the same partial commit; a fired id with
    no node: KeyError after the passes before it."""
    for kw in ({"mw_zero": True}, {"extra_fired": "no-such-node"}):
        assert edge_run(mode, seed, **kw) == edge_run("base", seed, **kw)
    assert any(t != "ok" for t, _ in edge_run("base", seed, mw_zero=True)[0])
    assert edge_run("base", seed, extra_fired="no-such-node")[0][2][0] == "KeyError"


@needs_p2b
@pytest.mark.parametrize("seed", range(3))
def test_three_way_native_fallback_and_bool_contract(seed):
    """Same ON graph pair: stdp_pass vs _stdp_python over NodeRefs, then bitwise; stdp_pass returns exactly True."""
    ga = edge_mutations(restore("on", edge_bytes(seed)), seed)
    gb = edge_mutations(restore("on", edge_bytes(seed)), seed)
    ra, rb = rule_of("on", seed), rule_of("on", seed)
    for g in (ga, gb):
        g.config["three_factor_enabled"] = bool(seed % 2)
    for k, fired in enumerate(fired_lists(ga, seed)):
        r = ga.synapses.stdp_pass(ga.nodes, fired, 600 + k, ra.A_plus, ra.A_minus, ra.tau_plus, ra.tau_minus,
                                  ra.learning_rate, bool(seed % 2), NF._GSG_CURVATURE_TABLE, ga._incoming, ga._outgoing)
        assert r is True
        NF._stdp_python(rb, gb, fired, 600 + k)
        assert syn_state(ga) == syn_state(gb)


# ---------------------------------------------------------------------------
# (3) declines touch nothing; the node-row cache under churn
# ---------------------------------------------------------------------------

@needs_p2b
def test_declines_touch_nothing():
    g = edge_mutations(restore("on", edge_bytes(2)), 2)
    fired = fired_lists(g, 2)[0]
    before = syn_state(g)
    T = NF._GSG_CURVATURE_TABLE
    ok = dict(node_store=g.nodes, fired_ids=fired, timestep=600, a_plus=1.0, a_minus=1.2, tau_plus=20.0,
              tau_minus=20.0, learning_rate=0.01, three_factor=False, curvature_table=T,
              incoming=g._incoming, outgoing=g._outgoing)
    bad = [
        dict(a_plus=1), dict(a_minus=True), dict(learning_rate=None), dict(tau_plus=0.0), dict(tau_minus=-20.0),
        dict(tau_plus=NAN), dict(three_factor=1), dict(three_factor=None), dict(timestep=600.0), dict(timestep=True),
        dict(curvature_table=[[1.0] * 3] * 2), dict(curvature_table=[[1, 1.0, 1.0]] * 3),
        dict(fired_ids=tuple(fired)), dict(fired_ids=fired + [5]), dict(fired_ids=fired + ["no-such-node"]),
        dict(incoming=types.MappingProxyType(g._incoming)), dict(outgoing=None),
        dict(node_store={nid: n for nid, n in g.nodes.items()}),
    ]
    for b in bad:
        kw = dict(ok, **b)
        assert g.synapses.stdp_pass(*kw.values()) is False, b
        assert syn_state(g) == before, b
    # a non-str member in an adjacency set
    g._incoming[fired[0]] = set(g._incoming.get(fired[0], set())) | {7}
    assert g.synapses.stdp_pass(*ok.values()) is False
    g._incoming[fired[0]].discard(7)
    # an exact-Python overflow value (an int last_spike_time) on any node
    victim = list(g.nodes)[-1]
    g.nodes[victim].last_spike_time = 3
    assert g.synapses.stdp_pass(*ok.values()) is False
    assert syn_state(g) == before
    g.nodes[victim].last_spike_time = 3.0
    assert g.synapses.stdp_pass(*ok.values()) is True
    assert syn_state(g) != before


@needs_p2b
def test_node_row_cache_survives_churn():
    """Interleave node removal (tombstones), re-creation, mass removal (compaction: layout_epoch) and a synapse-store
    clear() + reload between native passes; each pass must equal the fallback on a twin graph."""
    ga = restore("on", edge_bytes(4))
    gb = restore("on", edge_bytes(4))
    ra, rb = rule_of("on", 4), rule_of("on", 4)
    rng = random.Random(9)
    for k in range(10):
        fired = rng.sample(list(ga.nodes), len(ga.nodes) // 3)
        ra.apply(ga, fired, 700 + k)
        NF._stdp_python(rb, gb, fired, 700 + k)
        assert syn_state(ga) == syn_state(gb), k
        if k == 2:
            for nid in rng.sample(list(ga.nodes), 3):
                for g in (ga, gb):
                    del g.nodes[nid]
        if k == 4:
            ids = list(ga.nodes)
            for nid in ids[: len(ids) // 2]:
                for g in (ga, gb):
                    del g.nodes[nid]            # > 1/4 tombstones -> compaction
            for nid in ids[: len(ids) // 2: 3]:
                for g in (ga, gb):
                    g.create_node(node_id=nid).last_spike_time = 703.0
        if k == 6:
            for g in (ga, gb):
                saved = g.synapses.to_checkpoint_msgpack()
                g.synapses.clear()
                g.synapses.bulk_load_msgpack(saved)


@needs_p2b
def test_native_pass_engages_in_step(monkeypatch):
    calls = []
    monkeypatch.setattr(NF, "_stdp_python", lambda *a, **k: calls.append(1))
    g = new_graph("on")
    for i in range(30):
        g.create_node(node_id=f"a{i}")
    for i in range(30):
        g.create_synapse(f"a{i}", f"a{(i + 1) % 30}", weight=0.5)
        g.create_synapse(f"a{i}", f"a{(i + 7) % 30}", weight=0.5)
    fired = 0
    for _ in range(6):
        for nid in list(g.nodes)[::3]:
            g.stimulate(nid, 2.0)
        fired += len(g.step().fired_node_ids)
    assert fired > 0
    assert calls == [], calls


# ---------------------------------------------------------------------------
# (4) spec §5.2 item 5: whole runs (P2a's workload + per-step synapse state bitwise), vs the trial tip
# ---------------------------------------------------------------------------

LONG = "turn text — Ünïcødé 𝔘 " * 30


def build_random_graph(mode, seed, n_nodes=120, n_syn=900):
    def build(rng):
        g = new_graph(mode, {"three_factor_enabled": rng.random() < 0.5, "scaling_interval": 3,
                             "grace_period": rng.choice([50, 5000]),
                             "inactivity_threshold": rng.choice([40, 1000]),
                             "sprout_degree_cap": rng.choice([0, 30]),
                             "orphan_node_grace_period": rng.choice([0, 5, 25])})
        ids = []
        texts = [LONG + str(k) for k in range(6)]
        for i in range(n_nodes):
            meta = rng.choice([{}, {}, {"constitutional": True}, {"provenance": "syl_authored"},
                               {"provenance": "cc_emergent"}, {"provenance": "cc_authored"},
                               {"_forest_content": rng.choice(texts), "kind": "tree"},
                               {"poincare_dir": bytes(rng.getrandbits(8) for _ in range(64)),
                                "nested": [1, [2.5, None], {"k": "v"}]}])
            n = g.create_node(node_id=f"n{seed}_{i}", metadata=dict(meta), is_inhibitory=rng.random() < 0.1)
            n.firing_rate_ema = rng.choice([0.0, 0.0, rng.uniform(0, 0.3)])
            if rng.random() < 0.4:
                n.last_spike_time = float(rng.randint(0, 120))
                for t in range(rng.randint(0, 130)):
                    n.spike_history.append(float(t))
            if rng.random() < 0.2:
                n.pred_weights[f"n{seed}_{rng.randrange(n_nodes)}"] = rng.uniform(-1, 1)
            n.diffpc_layer = rng.choice([0, 0, 1, 2])
            if rng.random() < 0.1:
                n.manifold_type = "spherical"
            n.Ca_i = rng.choice([0.0, rng.uniform(0, 2)])
            n.intrinsic_excitability = rng.choice([1.0, rng.uniform(0.5, 1.5)])
            n.refractory_period = rng.choice([2, 2, 0, 1, 3])
            ids.append(n.node_id)
        for nid in rng.sample(ids, 8):
            g.remove_node(nid)
        for nid in sorted(rng.sample([i for i in ids if i not in g.nodes], 3)):
            g.create_node(node_id=nid, metadata={"readded": True})
        live = list(g.nodes)
        made = 0
        while made < n_syn:
            a, b = rng.sample(live, 2)
            s = g.create_synapse(a, b, weight=rng.uniform(0.0, 1.2), delay=rng.randint(1, 4),
                                 synapse_type=rng.choice([mod_of(mode).SynapseType.EXCITATORY] * 4
                                                         + [mod_of(mode).SynapseType.INHIBITORY]))
            s.low_weight_steps = rng.choice([0, 1, 49, 50, 51, rng.randint(0, 6000)])
            s.inactive_steps = rng.choice([0, 39, 40, 41, rng.randint(0, 3000)])
            s.creation_time = float(rng.randint(-6000, 100))
            s.eligibility_trace = rng.choice([0.0, rng.uniform(-0.4, 0.4)])
            made += 1
        for sid in rng.sample(list(g.synapses.keys()), n_syn // 10):
            g.remove_synapse(sid)
        g.timestep = rng.randint(100, 9000)
        return g
    return seeded(seed, build)


SEEDS = list(range(6))
_BUILT = {}


def built_bytes(seed):
    if seed not in _BUILT:
        _BUILT[seed] = ckpt_bytes(build_random_graph("base", seed))
    return _BUILT[seed]


_OFF = {"prune_protected_faint_links": False, "strength_budget_enabled": False}
_BUDGET = {"strength_budget_enabled": True, "strength_budget_out": 2.0,
           "strength_budget_in": 2.5, "strength_budget_interval": 2}
FLAGS = {
    "absent": {},
    "off": dict(_OFF),
    "lifeline": dict(_OFF, prune_protected_faint_links=True),
    "budget": dict(_OFF, **_BUDGET),
    "lifeline_budget": dict(_BUDGET, prune_protected_faint_links=True, last_link_grace_steps=4),
}


def fe(entries):
    return [(e.node_id, e.firing_step, P(e.voltage_at_fire), e.source_distance, e.was_predicted)
            for e in entries]


def tonic_scans(g, tt_mod, cycle):
    fake = types.SimpleNamespace(_graph=g, _config=tt_mod.TonicConfig(), _cycle_count=cycle,
                                 _focus_fatigue={nid: 0.01 for nid in list(g.nodes)[::5]})
    he_index = {nid: 1 for nid in list(g.nodes)[::3]}
    a = tt_mod.TonicThread._read_active_nodes(fake, he_index)
    b = tt_mod.TonicThread._read_recent_spikes(fake)
    return [(k, P(v)) for k, v in a], [(k, P(v)) for k, v in b]


def ap_capture(g, ap_mod):
    from ces_config import CESConfig
    cap = ap_mod.ActivationPersistence(CESConfig()).capture(g)
    return [(nid, sorted((k, P(v)) for k, v in st.items() if k != "timestamp")) for nid, st in cap.items()]


def workload(rounds, tt_mod, ap_mod, prefire):
    def act(g, rng):
        log, snaps = [], []
        if prefire:
            g.register_event_handler("pre_fire", lambda node_id: 0.05 if node_id[-1] in "13579" else -0.05)
        for k in range(rounds):
            for nid in rng.sample(list(g.nodes), min(8, len(g.nodes))):
                g.stimulate(nid, rng.uniform(0.5, 3.0))
            r = g.step()
            log.append((list(r.fired_node_ids), r.synapses_pruned, r.synapses_sprouted))
            log.append([(nid, P(n.voltage), P(n.threshold), P(n.Ca_i), n.refractory_remaining)
                        for nid, n in g.nodes.items()])
            log.append(syn_state(g))                                                   # P2b: STDP's output
            g.config["tonic_ages_substrate"] = 1
            ids = rng.sample(list(g.nodes), min(6, len(g.nodes)))
            p = g.prime_and_propagate(ids, [1.5] * len(ids), steps=3, write_mode=True)    # Tonic tick (STDP too)
            log.append(fe(p.fired_entries))
            log.append(syn_state(g))
            log.append(tonic_scans(g, tt_mod, k))
            q = g.prime_and_propagate(ids[:3], [2.0] * len(ids[:3]), steps=4, write_mode=False)  # recall
            log.append(fe(q.fired_entries))
            if k % 3 == 1:
                n = g.create_node(node_id=f"new{k}", metadata={"_forest_content": LONG, "k": k})
                n.voltage = 0.25
                for other in rng.sample(list(g.nodes), 2):
                    if other != n.node_id:
                        g.create_synapse(n.node_id, other, weight=0.3)
            if k % 7 == 3 and len(g.nodes) > 20:
                g.remove_node(rng.choice(list(g.nodes)))
            if k % 5 == 2:
                for j in range(3):
                    iso = g.create_node(node_id=f"iso{k}_{j}", metadata={"k": k})
                    iso.creation_time = int(g.timestep) - 10_000
            if k == 15:
                for nid in sorted(rng.sample(list(g.nodes), len(g.nodes) // 3)):
                    g.remove_node(nid)
            if k % 5 == 4:
                g.inject_reward(0.2)
                g.inject_reward(-0.1, scope=set(rng.sample(list(g.nodes), 10)))
                log.append(g._collect_orphan_nodes())
                log.append(ap_capture(g, ap_mod))
                log.append(repr(g.get_telemetry()))
            if k % 10 == 9:
                log.append(g.compete_protected_links(2, 10))
                log.append(sorted(g.sleep_downscale(0.9).items()))
                snaps.append(ckpt_bytes(g))
        log.append(list(g.nodes))
        return log, snaps
    return act


def run(mode, start_bytes, act, seed):
    g = restore(mode, start_bytes)
    out = seeded(seed, lambda rng: act(g, rng), uuid_stream=1)
    return ckpt_bytes(g), out


RUN_FLAGS = list(FLAGS) + ["prefire"]


@pytest.mark.parametrize("flags", RUN_FLAGS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_whole_run(mode, seed, flags, monkeypatch):
    prefire = flags == "prefire"
    calls = []
    orig = NF._stdp_python
    monkeypatch.setattr(NF, "_stdp_python", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    cfg = FLAGS["off"] if prefire else FLAGS[flags]

    def mk(tt_mod, ap_mod):
        act0 = workload(30, tt_mod, ap_mod, prefire)

        def act(g, rng):
            g.config.update(cfg)
            return act0(g, rng)
        return act
    start = built_bytes(seed)
    a, (la, sa) = run("base", start, mk(BASE_TT, BASE_AP), seed)
    b, (lb, sb) = run(mode, start, mk(TT, AP), seed)
    assert len(la) == len(lb)
    for i, (x, y) in enumerate(zip(la, lb)):
        assert x == y, f"log entry {i} differs"
    assert sa == sb
    assert a == b
    if mode == "on" and HAVE_P2B:
        assert calls == [], f"native STDP declined {len(calls)}x"
    elif mode == "off":
        assert len(calls) >= 30
