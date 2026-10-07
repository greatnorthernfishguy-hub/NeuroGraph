# [2026-10-06] Claude (lane sleep-p1) — D15 (sleep-phase spec): the BASE engine gets D15's one named intended change
#   (tests/d15_intended.apply_d15_rule: drop pre.pred_weights[post] with the last pre->post synapse) so this file keeps checking
#   everything else bitwise; with D15 switched off the file passes unchanged (SLEEP_P1.md §6).
# [2026-10-05] Claude (lane nodestore-p1) — equivalence: native node store (P1) ON / OFF vs the TRIAL TIP's code
"""P1 native node store: byte- and trace-equivalence against the trial tip.

BASE = the WHOLE neuro_foundation.py of the trial tip (BASE_REV, read from git, imported as a separate module;
its Graph.nodes is a dict of Node). BRANCH = this worktree, run twice: OFF (native_node_store=False -> the dict
path) and ON (native_node_store=True -> ng_tract.NodeStore; skipped when the installed wheel has no NodeStore).

Every run uses the same `random` seed and deterministic uuid4. Compared bit-for-bit: per-step fired-id order,
prime/recall fired entries, return values, activation_persistence captures, INCREMENTAL captures,
extract_subgraph, and full msgpack checkpoints (bytes). Covered: graphs BUILT in each mode (create_node /
remove_node / re-add of a removed id), whole runs over the trial's flag matrix (lifeline pruning on/off,
strength budget on/off, want-hub competition via compete_protected_links, sleep_downscale) with node
create / remove / orphan sweep in the run, and cross-mode compatibility (a checkpoint written ON is read OFF
and by the BASE module, and vice versa, each re-saving identical bytes).

Optional real-checkpoint pass: NG_NODESTORE_CKPT=<a COPY of a checkpoint> (the live plugin dir is refused).
Run with a fixed PYTHONHASHSEED (set iteration order is part of what is compared across processes).
"""
import gc
import hashlib
import importlib.util
import os
import random
import subprocess
import sys
import tempfile
import uuid

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import neuro_foundation as NF  # noqa: E402  (the branch)

BASE_REV = os.environ.get("NG_NODESTORE_BASE_REV", "0595221")   # trial tip this branch was cut from
HAVE_NATIVE = hasattr(NF.ng_tract, "NodeStore")
LIVE_DIR = "/.claude/plugins/neurograph/"


def _load_base():
    src = subprocess.run(["git", "-C", _REPO, "show", f"{BASE_REV}:neuro_foundation.py"],
                         check=True, capture_output=True, text=True).stdout
    d = tempfile.mkdtemp(prefix="nf_base_nodestore_")
    p = os.path.join(d, "nf_base_nodestore.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nf_base_nodestore", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nf_base_nodestore"] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base()
from tests.d15_intended import apply_d15_rule  # noqa: E402  [2026-10-06] sleep-p1: D15's named intended change
apply_d15_rule(BASE)
MODES = ["off", "on"] if HAVE_NATIVE else ["off"]


def new_graph(mode, config=None):
    if mode == "base":
        return BASE.Graph(config)
    g = NF.Graph(config, native_node_store=(mode == "on"))
    assert g._native_nodes == (mode == "on")
    assert isinstance(g.nodes, dict) == (mode != "on")
    return g


def mod_of(mode):
    return BASE if mode == "base" else NF


class DetUUID:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def __call__(self):
        return uuid.UUID(int=self.rng.getrandbits(128), version=4)


def ckpt_bytes(g, mode=NF.CheckpointMode.FULL):
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


def seeded(seed, fn, uuid_stream=0):
    saved = uuid.uuid4
    uuid.uuid4 = DetUUID(seed * 1000 + uuid_stream)   # distinct id streams for build vs run (no id reuse)
    random.seed(seed)
    try:
        return fn(random.Random(seed))
    finally:
        uuid.uuid4 = saved


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
            ids.append(n.node_id)
        for nid in rng.sample(ids, 8):            # node removal + re-add of the same id (dict order semantics)
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


def built_bytes(mode, seed):
    key = (mode, seed)
    if key not in _BUILT:
        _BUILT[key] = ckpt_bytes(build_random_graph(mode, seed))
    return _BUILT[key]


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
    return [(e.node_id, e.firing_step, e.voltage_at_fire, e.source_distance, e.was_predicted)
            for e in entries]


def ap_capture(g):
    from activation_persistence import ActivationPersistence
    from ces_config import CESConfig
    cap = ActivationPersistence(CESConfig()).capture(g)
    return {nid: {k: v for k, v in st.items() if k != "timestamp"} for nid, st in cap.items()}  # wall clock


def workload(rounds):
    def act(g, rng):
        log, snaps = [], []
        for k in range(rounds):
            for nid in rng.sample(list(g.nodes), min(8, len(g.nodes))):
                g.stimulate(nid, rng.uniform(0.5, 3.0))
            r = g.step()
            log.append((list(r.fired_node_ids), r.synapses_pruned, r.synapses_sprouted))
            g.config["tonic_ages_substrate"] = 1
            ids = rng.sample(list(g.nodes), min(6, len(g.nodes)))
            p = g.prime_and_propagate(ids, [1.5] * len(ids), steps=3, write_mode=True)    # Tonic tick
            log.append(fe(p.fired_entries))
            q = g.prime_and_propagate(ids[:3], [2.0] * len(ids[:3]), steps=4, write_mode=False)  # recall
            log.append(fe(q.fired_entries))
            if k % 3 == 1:                                     # node create (+ wiring) mid-run
                n = g.create_node(node_id=f"new{k}", metadata={"_forest_content": LONG, "k": k})
                n.voltage = 0.25
                for other in rng.sample(list(g.nodes), 2):
                    if other != n.node_id:
                        g.create_synapse(n.node_id, other, weight=0.3)
            if k % 7 == 3 and len(g.nodes) > 20:              # explicit removal mid-run
                g.remove_node(rng.choice(list(g.nodes)))
            if k % 5 == 2:                                     # isolated old nodes: the orphan sweep removes them
                for j in range(3):
                    iso = g.create_node(node_id=f"iso{k}_{j}", metadata={"k": k})
                    iso.creation_time = int(g.timestep) - 10_000
            if k == 15:                                        # mass removal: > 1/4 tombstones -> compaction (ON)
                for nid in sorted(rng.sample(list(g.nodes), len(g.nodes) // 3)):
                    g.remove_node(nid)
            if k % 5 == 4:
                g.inject_reward(0.2)
                g.inject_reward(-0.1, scope=set(rng.sample(list(g.nodes), 10)))
                log.append(g._collect_orphan_nodes())
                log.append(sorted(ap_capture(g).items()))
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


# ---------------------------------------------------------------------------
# 1. OFF == base, ON == base: graphs built through the mapping API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_built_graph_bytes(seed, mode):
    assert built_bytes(mode, seed) == built_bytes("base", seed)


# ---------------------------------------------------------------------------
# 2. whole runs: every flag combination, OFF and ON against the trial tip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flags", list(FLAGS))
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_whole_run(mode, seed, flags):
    act0 = workload(30)

    def act(g, rng):
        g.config.update(FLAGS[flags])
        return act0(g, rng)
    start = built_bytes("base", seed)
    a, (la, sa) = run("base", start, act, seed)
    b, (lb, sb) = run(mode, start, act, seed)
    assert la == lb
    assert sa == sb
    assert a == b


# ---------------------------------------------------------------------------
# 3. other node-reading capture paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS[:3])
@pytest.mark.parametrize("mode", MODES)
def test_incremental_subgraph_telemetry(seed, mode):
    def act(g, rng):
        for k in range(4):
            for nid in rng.sample(list(g.nodes), 6):
                g.stimulate(nid, 2.0)
            g.step()
        inc = g.capture_checkpoint(NF.CheckpointMode.INCREMENTAL if g.__class__ is NF.Graph
                                   else BASE.CheckpointMode.INCREMENTAL, detach=True)
        sub = g.extract_subgraph(set(rng.sample(list(g.nodes), 30)) | {"ghost"})
        sub["extraction_metadata"]["missing_nodes"].sort()
        import msgpack
        return (msgpack.packb(inc, use_bin_type=True), msgpack.packb(sub, use_bin_type=True),
                repr(g.get_telemetry()))
    start = built_bytes("base", seed)
    a, ra = run("base", start, act, seed)
    b, rb = run(mode, start, act, seed)
    assert ra == rb
    assert a == b


# ---------------------------------------------------------------------------
# 4. cross-mode compatibility: written ON -> read OFF / BASE, written OFF / BASE -> read ON
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_NATIVE, reason="ng_tract without NodeStore")
@pytest.mark.parametrize("seed", SEEDS)
def test_cross_mode_round_trips(seed):
    def churn(g, rng):
        for _ in range(5):
            for nid in rng.sample(list(g.nodes), 6):
                g.stimulate(nid, 2.5)
            g.step()
        return None
    on_bytes, _ = run("on", built_bytes("base", seed), churn, seed)
    off_bytes, _ = run("off", built_bytes("base", seed), churn, seed)
    assert on_bytes == off_bytes
    for reader in ("base", "off", "on"):
        assert ckpt_bytes(restore(reader, on_bytes)) == on_bytes, reader      # ON-written, re-saved by each
        assert ckpt_bytes(restore("on", ckpt_bytes(restore(reader, on_bytes)))) == on_bytes


# ---------------------------------------------------------------------------
# 5. NodeRef contract inside the engine
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_NATIVE, reason="ng_tract without NodeStore")
def test_engine_noderef_contract():
    g = new_graph("on")
    n = g.create_node(node_id="a", metadata={"x": 1})
    assert isinstance(n, NF.ng_tract.NodeRef) and n.node_id == "a"
    n.voltage = 0.5
    assert g.nodes["a"].voltage == 0.5                  # create_node hands back the live view
    g.create_node(node_id="b")
    held = g.nodes["a"]
    g.remove_node("a")
    with pytest.raises(KeyError):                       # D3
        held.voltage
    g.create_node(node_id="a")
    with pytest.raises(KeyError):                       # never follows a re-created id
        held.voltage
    assert list(g.nodes) == ["b", "a"]                  # D1: re-added id goes to the end


def test_opt_in_resolution():
    prev = NF.set_native_node_store_default(False)
    try:
        assert isinstance(NF.Graph().nodes, dict)                                  # default OFF
        assert isinstance(NF.Graph(native_node_store=False).nodes, dict)
        assert NF.set_native_node_store_default(True) is False                    # host switch; returns previous
        assert isinstance(NF.Graph().nodes, dict) == (not HAVE_NATIVE)             # needs the wheel
        assert isinstance(NF.Graph(native_node_store=False).nodes, dict)           # explicit keyword wins
        NF.set_native_node_store_default(False)
        assert isinstance(NF.Graph().nodes, dict)
        assert "native_node_store" not in NF.Graph(native_node_store=True).config  # never saved in the checkpoint
    finally:
        NF.set_native_node_store_default(prev)


# ---------------------------------------------------------------------------
# 6. optional: a COPY of the real checkpoint (sequential, memory-bounded)
# ---------------------------------------------------------------------------

CKPT = os.environ.get("NG_NODESTORE_CKPT")


@pytest.mark.skipif(not CKPT, reason="set NG_NODESTORE_CKPT to a checkpoint COPY")
@pytest.mark.parametrize("mode", MODES)
def test_checkpoint_copy_resave(mode):
    assert LIVE_DIR not in os.path.abspath(CKPT), "refusing the live checkpoint"
    with open(CKPT, "rb") as f:
        raw = f.read()
    g = restore(mode, raw)
    out = ckpt_bytes(g)
    del g
    gc.collect()
    base = restore("base", raw)
    ref = ckpt_bytes(base)
    del base
    gc.collect()
    rep = os.environ.get("NG_NODESTORE_CKPT_REPORT")
    if rep:
        import json
        with open(rep, "a") as f:
            f.write(json.dumps({"mode": mode, "ckpt_sha256": hashlib.sha256(raw).hexdigest(),
                                "resave_sha256": hashlib.sha256(out).hexdigest(),
                                "base_resave_sha256": hashlib.sha256(ref).hexdigest(),
                                "identical": out == ref}) + "\n")
    assert out == ref
