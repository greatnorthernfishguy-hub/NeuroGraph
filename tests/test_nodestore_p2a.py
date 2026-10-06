# ---- Changelog ----
# [2026-10-06] Claude (lane nodestore-p2a) — CREATE: P2a equivalence (native NodeStore node passes vs fallback vs trial tip)
# What: (1) per-method: every P2a NodeStore method against its module-level Python fallback run over the SAME store's
#       NodeRefs and over a dict of Node, bitwise (struct.pack('<d') per float, so NaN / ±0.0 / ±inf compare exactly),
#       on graphs carrying the spec §5.2 item 4 edge rows that apply to node passes; declines (non-float parameters,
#       exact-Python overflow values) touch nothing and leave the fallback's result; columns() contract.
#       (2) step() / HomeostaticRule against the TRIAL TIP (git show BASE_REV:neuro_foundation.py) on those edge graphs.
#       (3) Tonic scans, Tonic features, activation capture, get_telemetry against the trial tip's files.
#       (4) spec §5.2 item 5 whole runs: 30 rounds x the trial's flag matrix (+ a pre_fire-handler set) x 6 seeds x
#       {native ON, fallback OFF}: per-step fired-id order, per-step voltages bitwise, every log entry, snapshots
#       and the final checkpoint bytes, each against the trial tip.
# Why:  spec superpowers/specs/2026-10-05-native-node-store-design.md §4 P2a, §5.2 items 4-5 (the 60/60 bar of
#       RUST_HOTPATHS_ONTO_S4.md).
# How:  BASE modules are imported from git under their own names; the branch is this worktree. ON cases skip when
#       the installed ng_tract has no NodeStore; the native-method asserts skip when its NodeStore has no P2a methods
#       (then ON exercises the fallbacks over NodeRefs — run this file under such a wheel too). Fixed PYTHONHASHSEED.
# -------------------
"""P2a native node passes: bit-equivalence against the Python fallbacks and the trial tip."""
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

BASE_REV = os.environ.get("NG_NODESTORE_P2A_BASE_REV", "6a85357")   # trial tip this branch was cut from
HAVE_STORE = hasattr(NF.ng_tract, "NodeStore")
HAVE_P2A = HAVE_STORE and hasattr(NF.ng_tract.NodeStore, "decay_voltages")
needs_store = pytest.mark.skipif(not HAVE_STORE, reason="ng_tract without NodeStore")
needs_p2a = pytest.mark.skipif(not HAVE_P2A, reason="ng_tract NodeStore without the P2a methods")


def _load_base(fname, modname):
    src = subprocess.run(["git", "-C", _REPO, "show", f"{BASE_REV}:{fname}"],
                         check=True, capture_output=True, text=True).stdout
    d = tempfile.mkdtemp(prefix="p2a_base_")
    p = os.path.join(d, modname + ".py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(modname, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base("neuro_foundation.py", "nf_base_p2a")
BASE_TT = _load_base("tonic_thread.py", "tonic_thread_base_p2a")
BASE_AP = _load_base("activation_persistence.py", "activation_persistence_base_p2a")
MODES = ["off", "on"] if HAVE_STORE else ["off"]


def new_graph(mode, config=None):
    if mode == "base":
        return BASE.Graph(config)
    g = NF.Graph(config, native_node_store=(mode == "on"))
    assert isinstance(g.nodes, dict) == (mode != "on")
    return g


def mod_of(mode):
    return BASE if mode == "base" else NF


def P(x):
    """Bitwise identity of a value: floats by their IEEE bytes, everything else by (type, repr)."""
    if type(x) is float:
        return ("f", struct.pack("<d", x))
    return (type(x).__name__, repr(x))


FIELDS = ("voltage", "threshold", "resting_potential", "refractory_remaining", "refractory_period",
          "last_spike_time", "firing_rate_ema", "intrinsic_excitability", "Ca_i", "diffpc_layer",
          "pred_error_ema", "creation_time", "is_inhibitory", "manifold_type")


def node_state(g):
    out = []
    for nid, n in g.nodes.items():
        out.append((nid, tuple(P(getattr(n, a)) for a in FIELDS),
                    tuple(P(v) for v in n.spike_history.to_list()), n.spike_history.capacity))
    return out


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


# ---------------------------------------------------------------------------
# Edge graphs (spec §5.2 item 4, the rows that apply to node passes)
# ---------------------------------------------------------------------------

NAN = float("nan")
INF = float("inf")
EDGE_F = [0.0, -0.0, NAN, INF, -INF, 5e-324, 1e-300, 1e-9, 1.0000000000000002e-9, 0.85, 0.8499999999999999,
          2.0, -1.5, 4.999999999999999, 5.0, 5.000000000000001, 1e308]
EDGE_EMA = [0.0, -0.0, 1e-10, 1e-9, 9.999999999999999e-10, NAN, 0.05, 0.06, 0.0400000001, 0.07, 1.0, INF, -0.3]
EDGE_I = [0, 1, 2, -1, 3, 127, 128, 255, 256, 65535, 65536, -129, 2 ** 40, 2 ** 62]


def build_edge_graph(mode, seed, n=90):
    rng = random.Random(seed)
    g = new_graph(mode, {"scaling_interval": 2})
    for i in range(n):
        node = g.create_node(node_id=f"e{seed}_{i}", metadata={"i": i})
        node.voltage = rng.choice(EDGE_F)
        node.threshold = rng.choice(EDGE_F + [0.85] * 6)
        node.resting_potential = rng.choice([0.0, 0.0, -0.0, 0.1, NAN, -0.2])
        node.refractory_remaining = rng.choice([0, 0, 0] + EDGE_I)
        node.refractory_period = rng.choice([2, 2] + EDGE_I)
        node.last_spike_time = rng.choice([-INF, -INF, 0.0, 17.0, NAN, INF])
        node.firing_rate_ema = rng.choice(EDGE_EMA)
        node.intrinsic_excitability = rng.choice([1.0, 0.1, 0.09999999999999999, 5.0, 4.9, NAN, 0.0, -1.0])
        node.Ca_i = rng.choice([0.0, 1e-9, 1.0000000000000002e-9, NAN, 4.9, 5.0, 6.0, -1.0, 0.3])
        cap = rng.choice([100, 100, 7, 1, 0])
        if cap != 100:
            node.spike_history = NF.RingBuffer(cap)
        for t in range(rng.choice([0, 0, 3, 7, 100, 130])):
            node.spike_history.append(float(t))
        if rng.random() < 0.1:
            node.manifold_type = "spherical"
    ids = list(g.nodes)
    for nid in rng.sample(ids, 9):                 # tombstones; then re-add 3 removed ids (they go to the end)
        g.remove_node(nid)
    for nid in sorted(rng.sample([i for i in ids if i not in g.nodes], 3)):
        node = g.create_node(node_id=nid, metadata={"readded": True})
        node.voltage = 3.0
    return g


def _churn(g, seed):
    """Tombstones in the live store, then re-adds (dict order: a re-added id goes to the end)."""
    rng = random.Random(seed + 991)
    ids = list(g.nodes)
    for nid in rng.sample(ids, 7):
        g.remove_node(nid)
    for nid in sorted(rng.sample([i for i in ids if i not in g.nodes], 2)):
        node = g.create_node(node_id=nid, metadata={"readded2": True})
        node.voltage = 1.25
        node.Ca_i = 0.5
    return g


def edge_pair(seed):
    """Equal edge graphs, as a live graph gets them (restored from a checkpoint the trial tip wrote, then
    churned): ON (native store) twice, and the dict path."""
    b = _edge_bytes(seed)
    return tuple(_churn(restore(m, b), seed) for m in ("on", "on", "off"))


# The P2a passes: (native call, fallback call) on a graph. Fallback = the module-level original loop.
def _targets(g, seed):
    rng = random.Random(seed + 77)
    t = {nid: rng.choice([0.05, 0.0, 0.1, NAN, 1e-12]) for nid in g.nodes if rng.random() < 0.7}
    t["ghost-id"] = 0.3                            # a key with no node: ignored
    return t


PASSES = {
    "decay_voltages": (lambda g, s: g.nodes.decay_voltages(0.97),
                       lambda g, s: NF._decay_voltages_python(g.nodes, 0.97)),
    "calcium_currents": (lambda g, s: g.nodes.calcium_currents(0.06 - 0.04, 0.9),
                         lambda g, s: NF._calcium_currents_python(g.nodes, 0.06 - 0.04, 0.9)),
    "fire": (lambda g, s: g.nodes.fire(_fire_ids(g, s), 1234, 0.2),
             lambda g, s: NF._fire_python(g.nodes, {}, _fire_ids(g, s), 1234, 0.2)),
    "fire_no_ca": (lambda g, s: g.nodes.fire(_fire_ids(g, s), 2 ** 53 + 1, None),
                   lambda g, s: NF._fire_python(g.nodes, {}, _fire_ids(g, s), 2 ** 53 + 1, 0.0)),
    "decrement_refractory": (lambda g, s: g.nodes.decrement_refractory(set(_fire_ids(g, s))),
                             lambda g, s: NF._decrement_refractory_python(g.nodes, set(_fire_ids(g, s)))),
    "update_firing_ema": (lambda g, s: g.nodes.update_firing_ema(set(_fire_ids(g, s)) | {"ghost", 5}, 0.01),
                          lambda g, s: NF._update_firing_ema_python(g.nodes, set(_fire_ids(g, s)) | {"ghost", 5}, 0.01)),
    "adapt_thresholds": (lambda g, s: g.nodes.adapt_thresholds(_targets(g, s), 0.05, 0.001, 5.0),
                         lambda g, s: NF._adapt_thresholds_python(g.nodes, _targets(g, s), 0.05, 0.001, 5.0)),
}


def _fire_ids(g, seed):
    ids = list(g.nodes)
    rng = random.Random(seed + 5)
    picked = rng.sample(ids, len(ids) // 3)
    return picked + picked[:2]                     # duplicates repeat, exactly as the loop would


@needs_p2a
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("name", list(PASSES))
def test_pass_native_vs_fallback(name, seed):
    native, fallback = PASSES[name]
    g_nat, g_fb, g_dict = edge_pair(seed)
    assert node_state(g_nat) == node_state(g_fb) == node_state(g_dict)
    assert native(g_nat, seed) is True
    fallback(g_fb, seed)
    fallback(g_dict, seed)
    assert node_state(g_nat) == node_state(g_fb)
    assert node_state(g_nat) == node_state(g_dict)
    assert ckpt_bytes(g_nat) == ckpt_bytes(g_dict)


@needs_p2a
@pytest.mark.parametrize("seed", range(4))
def test_detect_fired_native_vs_fallback(seed):
    g_nat, g_fb, g_dict = edge_pair(seed)
    got = g_nat.nodes.detect_fired()
    assert got == NF._detect_fired_python(g_fb.nodes, {}) == NF._detect_fired_python(g_dict.nodes, {})
    assert got == NF._detect_fired_python(g_dict.nodes, {"pre_fire": []})     # empty handler list = none
    assert node_state(g_nat) == node_state(g_dict)


@needs_p2a
@pytest.mark.parametrize("seed", range(4))
def test_adapt_excitability_native_vs_fallback(seed):
    g_nat, g_fb, g_dict = edge_pair(seed)
    t = _targets(g_nat, seed)
    ratios = g_nat.nodes.adapt_excitability(t, 0.05, 0.01)
    scales = {nid: r ** 0.1 for nid, r in ratios.items()}
    fb = NF._adapt_excitability_python(g_fb.nodes, t, 0.05, 0.01, 0.1)
    dd = NF._adapt_excitability_python(g_dict.nodes, t, 0.05, 0.01, 0.1)
    assert list(scales) == list(fb) == list(dd)
    assert [P(v) for v in scales.values()] == [P(v) for v in fb.values()] == [P(v) for v in dd.values()]
    assert node_state(g_nat) == node_state(g_fb) == node_state(g_dict)


@needs_p2a
def test_float_expression_order_matters_and_is_kept():
    # a voltage / decay pair where (v*d) + ((1-d)*r) differs from fma or reassociation: the store must equal Python
    g = new_graph("on")
    vals = [(0.1, 0.30000000000000004), (1e16, 1.0), (0.7, -0.7), (1.0 / 3.0, 2.0 / 3.0)]
    for k, (v, r) in enumerate(vals):
        n = g.create_node(node_id=f"x{k}")
        n.voltage, n.resting_potential = v, r
    for d in (0.97, 0.1, 1.0 / 3.0):
        g.nodes.decay_voltages(d)
        for k, (v, r) in enumerate(vals):
            v = v * d + (1.0 - d) * r
            vals[k] = (v, r)
            assert struct.pack("<d", g.nodes[f"x{k}"].voltage) == struct.pack("<d", v)


@needs_p2a
def test_declines_touch_nothing_and_fallback_matches():
    g, _, gd = edge_pair(9)
    before = node_state(g)
    # non-float parameters: the loop's arithmetic on other types is Python's to define
    assert g.nodes.decay_voltages(1) is False
    assert g.nodes.calcium_currents(0, 0.9) is False
    assert g.nodes.fire(list(g.nodes)[:3], 5.0, None) is False
    assert g.nodes.fire(list(g.nodes)[:3], 5, 1) is False
    assert g.nodes.fire(["no-such-node"], 5, None) is False      # the loop raises KeyError: let it
    assert g.nodes.fire([b"bytes-id"], 5, None) is False
    assert g.nodes.update_firing_ema(set(), 0) is False
    assert g.nodes.adapt_thresholds({}, 0.05, 0.001, 5) is False
    assert g.nodes.adapt_thresholds({list(g.nodes)[0]: 1}, 0.05, 0.001, 5.0) is False   # an int target of a node
    # (an int under a key that names no node is never read — by the loop either — so that pass proceeds)
    assert g.nodes.adapt_thresholds(types.MappingProxyType({}), 0.05, 0.001, 5.0) is False
    assert g.nodes.adapt_excitability({}, 0.05, 1) is None
    assert node_state(g) == before
    # an exact-Python overflow value in a touched field (an int voltage): decline, then the fallback == dict path
    nid = list(g.nodes)[4]
    g.nodes[nid].voltage = 3
    gd.nodes[nid].voltage = 3
    assert g.nodes.decay_voltages(0.97) is False
    assert g.nodes.detect_fired() is None
    assert g.nodes.columns(["voltage"]) is None
    assert g.nodes.columns(["threshold"]) is not None             # other fields still copy
    NF._decay_voltages_python(g.nodes, 0.97)
    NF._decay_voltages_python(gd.nodes, 0.97)
    assert node_state(g) == node_state(gd)
    # int in refractory: only the passes that touch it decline
    g2, _, _ = edge_pair(10)
    r0 = list(g2.nodes)[0]
    g2.nodes[r0].refractory_remaining = 2.5
    assert g2.nodes.decrement_refractory(set()) is False
    assert g2.nodes.decay_voltages(0.97) is True


@needs_p2a
def test_columns_contract():
    g, _, gd = edge_pair(3)
    names = ["voltage", "threshold", "refractory_remaining", "firing_rate_ema", "creation_time", "Ca_i"]
    ids, *arrs = g.nodes.columns(names)
    assert ids == list(gd.nodes)
    for name, a in zip(names, arrs):
        assert str(a.dtype) == ("int64" if name in ("refractory_remaining", "creation_time") else "float64")
        assert [P(v) for v in a.tolist()] == [P(getattr(n, name)) for n in gd.nodes.values()]
    some = list(gd.nodes)[::-7]
    sid, sv = g.nodes.columns(["voltage"], ids=some)
    assert sid == some and [P(v) for v in sv.tolist()] == [P(gd.nodes[k].voltage) for k in some]
    none_ids, _ = g.nodes.columns(["voltage"], with_ids=False)
    assert none_ids is None
    with pytest.raises(ValueError):
        g.nodes.columns(["metadata"])
    with pytest.raises(KeyError):
        g.nodes.columns(["voltage"], ids=["nope"])
    snap = g.nodes.columns(["voltage"])[1]
    snap[:] = 123.0                                              # a copy: the store is untouched
    assert g.nodes[ids[0]].voltage != 123.0 or gd.nodes[ids[0]].voltage == 123.0


@needs_p2a
def test_no_python_runs_under_the_borrow_reentrancy():
    # a fired-id iterable whose iteration re-enters the store: inputs are read before borrowing
    g, _, _ = edge_pair(2)

    made = []

    class Reenter:
        def __iter__(self):
            g.nodes.columns(["voltage"])
            made.append(f"made-during-iter-{len(made)}")
            g.create_node(node_id=made[-1])
            yield list(g.nodes)[0]

    assert g.nodes.decrement_refractory(Reenter()) is True
    assert g.nodes.update_firing_ema(Reenter(), 0.01) is True
    assert made and all(m in g.nodes for m in made)


# ---------------------------------------------------------------------------
# step() / HomeostaticRule on edge graphs vs the trial tip
# ---------------------------------------------------------------------------

def _edge_bytes(seed):
    return ckpt_bytes(build_edge_graph("base", seed))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("prefire", [False, True])
@pytest.mark.parametrize("mode", MODES)
def test_step_on_edge_graph_vs_trial_tip(mode, seed, prefire):
    start = _edge_bytes(seed)
    outs = {}
    for m in ("base", mode):
        g = restore(m, start)
        if prefire:
            g.register_event_handler("pre_fire", lambda node_id: 0.125 if node_id.endswith("1") else -0.0625)

        def act(rng, g=g):
            log = []
            for k in range(8):
                for nid in list(g.nodes)[k::9]:
                    g.stimulate(nid, 0.7)
                r = g.step()
                log.append((list(r.fired_node_ids), [P(n.voltage) for n in g.nodes.values()]))
            return log
        log = seeded(seed, act, uuid_stream=2)
        outs[m] = (log, node_state(g), ckpt_bytes(g))
    assert outs[mode][0] == outs["base"][0]
    assert outs[mode][1] == outs["base"][1]
    assert outs[mode][2] == outs["base"][2]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("mode", MODES)
def test_homeostatic_rule_vs_trial_tip(mode, seed):
    start = _edge_bytes(seed)
    states = {}
    for m in ("base", mode):
        g = restore(m, start)
        rule = mod_of(m).HomeostaticRule(target_firing_rate=0.05, scaling_interval=2, degree_sensitivity=0.4)
        ids = list(g.nodes)
        for k in range(5):
            rule.apply(g, ids[k::4], 100 + k)
        states[m] = (node_state(g), sorted(rule._degree_targets.items()), ckpt_bytes(g))
    assert states[mode][0] == states["base"][0]
    assert [(k, P(v)) for k, v in states[mode][1]] == [(k, P(v)) for k, v in states["base"][1]]
    assert states[mode][2] == states["base"][2]


# ---------------------------------------------------------------------------
# readers: Tonic scans, Tonic features, activation capture, telemetry vs the trial tip
# ---------------------------------------------------------------------------

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


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("mode", MODES)
def test_readers_vs_trial_tip(mode, seed):
    start = _edge_bytes(seed)
    gb, gm = restore("base", start), restore(mode, start)
    for k in range(3):
        for g in (gb, gm):
            for nid in list(g.nodes)[k::5]:
                g.stimulate(nid, 1.1)
            seeded(seed * 10 + k, lambda rng, g=g: g.step(), uuid_stream=3)
        assert tonic_scans(gm, TT, k) == tonic_scans(gb, BASE_TT, k)
        assert ap_capture(gm, AP) == ap_capture(gb, BASE_AP)
        tb, tm = gb.get_telemetry(), gm.get_telemetry()
        assert P(tm.global_firing_rate) == P(tb.global_firing_rate) and repr(tm) == repr(tb)


@pytest.mark.parametrize("mode", MODES)
def test_tonic_features_vs_trial_tip(mode):
    torch = pytest.importorskip("torch")
    pytest.importorskip("surgery.tonic_brain")
    import tonic_engine as TE
    BASE_TE = _load_base("tonic_engine.py", "tonic_engine_base_p2a")
    start = _edge_bytes(1)
    feats = {}
    for m, te in (("base", BASE_TE), (mode, TE)):
        g = restore(m, start)
        seeded(1, lambda rng, g=g: g.step(), uuid_stream=4)
        fake = types.SimpleNamespace(_graph=g, _identity_embedding_tensor=lambda: torch.zeros(768))
        f = te.TonicEngine._extract_graph_features_for_model(fake)
        if f is None:
            pytest.skip("GraphFeatures unavailable (Elmer surgery / torch import failed)")
        feats[m] = ({k: getattr(f, k) for k in f.__dataclass_fields__} if hasattr(f, "__dataclass_fields__")
                    else f._asdict())
    a, b = feats["base"], feats[mode]
    assert list(a) == list(b)
    for k in a:
        assert torch.equal(a[k], b[k]) or (torch.isnan(a[k]).any() and torch.equal(torch.nan_to_num(a[k]), torch.nan_to_num(b[k]))), k


# ---------------------------------------------------------------------------
# spec §5.2 item 5: whole runs (the P1 workload + Tonic scans + telemetry, per-step voltages bitwise)
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
            g.config["tonic_ages_substrate"] = 1
            ids = rng.sample(list(g.nodes), min(6, len(g.nodes)))
            p = g.prime_and_propagate(ids, [1.5] * len(ids), steps=3, write_mode=True)    # Tonic tick
            log.append(fe(p.fired_entries))
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


_FALLBACKS = ("_decay_voltages_python", "_calcium_currents_python", "_detect_fired_python", "_fire_python",
              "_decrement_refractory_python", "_update_firing_ema_python", "_adapt_thresholds_python",
              "_adapt_excitability_python")


@pytest.mark.parametrize("flags", RUN_FLAGS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_whole_run(mode, seed, flags, monkeypatch):
    prefire = flags == "prefire"
    fallback_calls = {}
    for name in _FALLBACKS:     # count (and still run) every fallback the BRANCH takes
        monkeypatch.setattr(NF, name, (lambda nm, o: (lambda *a, **k: (
            fallback_calls.__setitem__(nm, fallback_calls.get(nm, 0) + 1), o(*a, **k))[1]))(name, getattr(NF, name)))
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
    if mode == "on" and HAVE_P2A:   # the native passes really ran: no fallback but the pre_fire-handler one
        assert fallback_calls == ({"_detect_fired_python": 30} if prefire else {}), fallback_calls
    elif mode == "off":
        assert fallback_calls.get("_decay_voltages_python") == 30


@needs_p2a
def test_native_methods_actually_engage(monkeypatch):
    """Guard against a silent fallback: on a plain ON graph every P2a site takes the native method."""
    g = new_graph("on", {"scaling_interval": 2})
    for i in range(30):
        n = g.create_node(node_id=f"a{i}")
        n.voltage = 2.0 if i % 3 == 0 else 0.1
        n.Ca_i = 0.5 if i % 4 == 0 else 0.0
    calls = []
    for name in ("_decay_voltages_python", "_calcium_currents_python", "_detect_fired_python", "_fire_python",
                 "_decrement_refractory_python", "_update_firing_ema_python", "_adapt_thresholds_python",
                 "_adapt_excitability_python"):
        monkeypatch.setattr(NF, name, (lambda nm: (lambda *a, **k: calls.append(nm)))(name))
    fired = 0
    for _ in range(6):
        fired += len(g.step().fired_node_ids)
        for nid in list(g.nodes)[::3]:
            g.stimulate(nid, 2.0)
    assert fired > 0
    assert calls == [], calls


@needs_p2a
def test_degree_target_lookup_revalidates_layout():
    """The per-row dict lookups run with no borrow held; Python code they trigger (a key's __eq__) may
    change the store. Then the pass must decline (touch nothing) — the caller runs the Python loop."""
    g, _, _ = edge_pair(5)
    victim = list(g.nodes)[3]

    class Collider:
        def __hash__(self):
            return hash(victim)

        def __eq__(self, other):
            if "made-in-eq" not in g.nodes:
                g.create_node(node_id="made-in-eq")
            return False

    d = {Collider(): 0.3}
    d.update({nid: 0.05 for nid in list(g.nodes)[::2]})
    before = node_state(g)
    assert g.nodes.adapt_thresholds(d, 0.05, 0.001, 5.0) is False
    assert "made-in-eq" in g.nodes
    assert node_state(g)[:-1] == before                       # nothing touched but the node __eq__ made
    assert g.nodes.adapt_excitability(d, 0.05, 0.01) is not None   # no layout change this time: proceeds
