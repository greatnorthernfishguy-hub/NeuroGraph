# [2026-10-04] Claude (overnight Rust review) — equivalence: native hot paths vs the original Python loops
# [2026-10-05] Claude (lane rust-hotpaths-onto-s4) — carried over unchanged from the overnight lane (5ac8c3a, 52355c1); its oracles are d0ee8cc bodies; passes on this branch (60 passed); the trial-tip oracle suite is test_rust_hotpaths_onto_s4.py
"""Graph-level equivalence tests for the native SynapseStore hot paths.

The ORACLES below are the original Python bodies from neuro_foundation.py at
origin/cc-laptop-trial-s4-20261002 (d0ee8cc), copied verbatim. Each test builds
TWIN graphs from the same checkpoint bytes, runs the oracle on one and the
current (native) code on the other, then requires the two full msgpack
checkpoints to be BYTE-IDENTICAL (every node, synapse column, hyperedge, counter).

Needs the ng_tract wheel from ng-tract-rs branch cc-laptop-rust-hotpaths-20261004.
Optional real-checkpoint pass: set NG_HOTPATH_CKPT to a COPY of a checkpoint
(the test refuses the live plugin checkpoint path and never writes to it).
"""
import itertools
import math
import os
import random
import sys
import tempfile
import types
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neuro_foundation as nf  # noqa: E402

if not hasattr(nf.ng_tract.SynapseStore, "advance_low_weight_and_collect_prune"):
    pytest.skip("ng_tract wheel without the hot-path methods", allow_module_level=True)


# ---------------------------------------------------------------------------
# ORACLES — the ORIGINAL method bodies, taken verbatim from git (d0ee8cc, the
# branch point of this review branch) and compiled against THIS module's globals,
# so they see the very same classes/constants/ng_tract store as the new code.
# ---------------------------------------------------------------------------

ORIG_REV = "d0ee8cc"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORACLE_METHODS = {
    "Graph": ["_prune_synapses", "inject_reward", "step", "prime_and_propagate",
              "_sprout_synapses", "get_telemetry", "extract_subgraph", "_diffpc_step",
              "_find_synapse", "_generate_predictions_from_node"],
    "HomeostaticRule": ["apply", "_refresh_degree_targets"],
    "STDPRule": ["apply", "_apply_dw"],
}


def _load_oracles():
    import ast
    import subprocess
    import textwrap
    try:
        src = subprocess.run(["git", "-C", _REPO, "show", f"{ORIG_REV}:neuro_foundation.py"],
                             check=True, capture_output=True, text=True).stdout
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"original source {ORIG_REV} unavailable: {exc}", allow_module_level=True)
    tree = ast.parse(src)
    out = {}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in ORACLE_METHODS:
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name in ORACLE_METHODS[cls.name]:
                    code = textwrap.dedent(ast.get_source_segment(src, fn))
                    ns = dict(nf.__dict__)
                    exec(compile(code, f"<{ORIG_REV}:{cls.name}.{fn.name}>", "exec"), ns)
                    out[(cls.name, fn.name)] = ns[fn.name]
    missing = [(c, m) for c, ms in ORACLE_METHODS.items() for m in ms if (c, m) not in out]
    assert not missing, missing
    return out


ORACLES = _load_oracles()


def install_oracles(g):
    """Make graph `g` (and its rule objects) run the ORIGINAL code paths."""
    for (cls, name), fn in ORACLES.items():
        if cls == "Graph":
            setattr(g, name, types.MethodType(fn, g))
    for r in g._plasticity_rules:
        for (cls, name), fn in ORACLES.items():
            if cls != "Graph" and type(r).__name__ == cls:
                setattr(r, name, types.MethodType(fn, r))


def oracle_adjacency(graph):
    out, inc = {}, {}
    for sid in graph.synapses.keys():
        ref = graph.synapses[sid]
        out.setdefault(ref.pre_node_id, set()).add(sid)
        inc.setdefault(ref.post_node_id, set()).add(sid)
    return out, inc


def oracle_subgraph_synapses(graph, node_ids):
    extracted = {}
    for sid, syn in graph.synapses.items():
        if syn.pre_node_id in node_ids and syn.post_node_id in node_ids:
            extracted[sid] = graph.synapses.serialize_one(sid)
    return extracted


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

LIVE_DIR = "/.claude/plugins/neurograph/checkpoints/"


def ckpt_bytes(g):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        g.checkpoint(p)
        with open(p, "rb") as f:
            return f.read()


def restore_from_bytes(b):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        with open(p, "wb") as f:
            f.write(b)
        g = nf.Graph()
        g.restore(p)
        return g


def build_random_graph(seed, n_nodes=120, n_syn=900):
    rng = random.Random(seed)
    g = nf.Graph({"three_factor_enabled": True, "scaling_interval": 3,
                  "grace_period": rng.choice([50, 5000]),
                  "inactivity_threshold": rng.choice([40, 1000]),
                  "sprout_degree_cap": rng.choice([0, 30])})
    ids = []
    for i in range(n_nodes):
        meta = rng.choice([{}, {}, {}, {"constitutional": True},
                           {"provenance": "syl_authored"}, {"provenance": "cc_emergent"},
                           {"provenance": "cc_authored"}])
        n = g.create_node(node_id=f"n{seed}_{i}", metadata=dict(meta),
                          is_inhibitory=rng.random() < 0.1)
        n.firing_rate_ema = rng.choice([0.0, 0.0, rng.uniform(0, 0.3)])
        ids.append(n.node_id)
    made = 0
    while made < n_syn:
        a, b = rng.sample(ids, 2)
        s = g.create_synapse(a, b, weight=rng.uniform(0.0, 1.2), delay=rng.randint(1, 4),
                             synapse_type=rng.choice([nf.SynapseType.EXCITATORY] * 4
                                                     + [nf.SynapseType.INHIBITORY]))
        s.low_weight_steps = rng.choice([0, 1, 49, 50, 51, rng.randint(0, 6000)])
        s.inactive_steps = rng.choice([0, 39, 40, 41, rng.randint(0, 3000)])
        s.salience = rng.choice([1.0, 1.0, 1.7, rng.uniform(1, 4)])
        s.creation_time = float(rng.randint(-6000, 100))
        s.peak_weight = rng.uniform(0.0, 0.5)
        s.eligibility_trace = rng.choice([0.0, 1e-10, -1e-10, rng.uniform(-0.4, 0.4)])
        if rng.random() < 0.02:
            s.weight = rng.choice([0.0, -0.0, 0.005, 5.0])
        made += 1
    for sid in rng.sample(list(g.synapses.keys()), n_syn // 10):  # swap-remove churn
        g.remove_synapse(sid)
    g.timestep = rng.randint(100, 9000)
    return g


class DetUUID:
    """Deterministic uuid4 so twin runs mint identical node/synapse ids."""
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def __call__(self):
        return uuid.UUID(int=self.rng.getrandbits(128), version=4)


def run_twins(base_bytes, action, seed=0):
    """Restore twins, run `action(graph, rng)` with oracles on A and native on B,
    with identical random / uuid streams; return both checkpoints' bytes."""
    out = []
    for use_oracle in (True, False):
        g = restore_from_bytes(base_bytes)
        if use_oracle:
            install_oracles(g)
        saved_uuid = nf.uuid.uuid4
        nf.uuid.uuid4 = DetUUID(seed)
        random.seed(seed)
        try:
            ret = action(g, random.Random(seed))
        finally:
            nf.uuid.uuid4 = saved_uuid
        out.append((ckpt_bytes(g), ret, g))
    return out


# ---------------------------------------------------------------------------
# tests on randomized graphs
# ---------------------------------------------------------------------------

SEEDS = list(range(12))


@pytest.mark.parametrize("seed", SEEDS)
def test_prune_twins(seed):
    base = ckpt_bytes(build_random_graph(seed))

    def act(g, rng):
        return [g._prune_synapses() for _ in range(3)]
    (a, ra, _), (b, rb, _) = run_twins(base, act, seed)
    assert ra == rb
    assert a == b


@pytest.mark.parametrize("seed", SEEDS)
def test_inject_reward_twins(seed):
    base = ckpt_bytes(build_random_graph(seed))

    def act(g, rng):
        nodes = list(g.nodes)
        g.inject_reward(0.3)
        g.inject_reward(-0.2, scope=set(rng.sample(nodes, 10)))
        g.inject_reward(0.1, scope=set())
        g.inject_reward(1, scope=frozenset(rng.sample(nodes, 40)) | {"ghost"})
    (a, _, _), (b, _, _) = run_twins(base, act, seed)
    assert a == b


@pytest.mark.parametrize("seed", SEEDS)
def test_homeostatic_scaling_twins(seed):
    base = ckpt_bytes(build_random_graph(seed))

    def act(g, rng):
        hr = next(r for r in g._plasticity_rules if isinstance(r, nf.HomeostaticRule))
        nodes = list(g.nodes)
        for k in range(7):  # scaling_interval=3 -> the scaling branch runs twice
            hr.apply(g, rng.sample(nodes, 15), g.timestep + k)
    (a, _, _), (b, _, _) = run_twins(base, act, seed)
    assert a == b


@pytest.mark.parametrize("seed", SEEDS)
def test_restore_adjacency_and_subgraph(seed):
    g = restore_from_bytes(ckpt_bytes(build_random_graph(seed)))
    out, inc = oracle_adjacency(g)
    nonempty_out = {k: v for k, v in g._outgoing.items() if v}
    nonempty_inc = {k: v for k, v in g._incoming.items() if v}
    assert nonempty_out == out and nonempty_inc == inc
    for k in out:  # identical insertion history -> identical iteration order
        assert list(g._outgoing[k]) == list(out[k])
    for k in inc:
        assert list(g._incoming[k]) == list(inc[k])
    sub = set(random.Random(seed).sample(list(g.nodes), 50))
    assert g.extract_subgraph(sub)["synapses"] == oracle_subgraph_synapses(g, sub)


@pytest.mark.parametrize("seed", SEEDS[:4])
def test_telemetry_weights_identical(seed):
    import numpy as np
    g = build_random_graph(seed)
    w_old = [s.weight for s in list(g.synapses.values())]
    tel = g.get_telemetry()
    assert tel.mean_weight == float(np.mean(w_old))
    assert tel.std_weight == float(np.std(w_old))
    assert tel == ORACLES[("Graph", "get_telemetry")](g)
    assert nf.Graph().get_telemetry().mean_weight == 0.0  # empty store branch


@pytest.mark.parametrize("seed", SEEDS[:8])
def test_full_workload_twins(seed):
    """Interleaved step() + Tonic write-mode prime_and_propagate, end to end."""
    base = ckpt_bytes(build_random_graph(seed))

    def act(g, rng):
        g.config["tonic_ages_substrate"] = 1
        nodes = list(g.nodes)
        log = []
        for k in range(25):
            for nid in rng.sample(nodes, 8):
                if nid in g.nodes:
                    g.stimulate(nid, rng.uniform(0.5, 3.0))
            r = g.step()
            log.append((r.fired_node_ids, r.synapses_pruned, r.synapses_sprouted))
            live = list(g.nodes)
            ids = rng.sample(live, min(6, len(live)))
            p = g.prime_and_propagate(ids, [1.5] * len(ids), steps=3, write_mode=True)
            log.append([(e.node_id, e.firing_step, e.voltage_at_fire, e.source_distance)
                        for e in p.fired_entries])
            q = g.prime_and_propagate(ids[:3], [2.0] * len(ids[:3]), steps=4, write_mode=False)
            log.append([(e.node_id, e.firing_step, e.voltage_at_fire, e.source_distance)
                        for e in q.fired_entries])
        return log
    (a, la, _), (b, lb, _) = run_twins(base, act, seed)
    assert la == lb
    assert a == b


# ---------------------------------------------------------------------------
# optional: the real checkpoint COPY
# ---------------------------------------------------------------------------

CKPT = os.environ.get("NG_HOTPATH_CKPT")


@pytest.mark.skipif(not CKPT, reason="set NG_HOTPATH_CKPT to a checkpoint COPY")
def test_checkpoint_copy_twins():
    assert LIVE_DIR not in os.path.abspath(CKPT), "refusing the live checkpoint"
    with open(CKPT, "rb") as f:
        base = f.read()

    def act(g, rng):
        res = {}
        # adjacency rebuilt by the native path must equal the per-SynapseRef rebuild
        out, inc = oracle_adjacency(g)
        res["adj"] = ({k: v for k, v in g._outgoing.items() if v} == out
                      and {k: v for k, v in g._incoming.items() if v} == inc)
        res["prune"] = [g._prune_synapses() for _ in range(2)]
        g.inject_reward(0.05)
        g.inject_reward(0.05, scope=set(rng.sample(list(g.nodes), 200)))
        hr = next(r for r in g._plasticity_rules if isinstance(r, nf.HomeostaticRule))
        hr._steps_since_scaling = hr.scaling_interval
        hr.apply(g, rng.sample(list(g.nodes), 50), g.timestep)
        res["tel"] = (g.get_telemetry().mean_weight, g.get_telemetry().std_weight)
        # a short live-like workload on the real topology: step() + Tonic write-mode
        # propagation (incl. its prune) + a read-mode recall
        g.config["tonic_ages_substrate"] = 1
        trace = []
        for _ in range(int(os.environ.get("NG_HOTPATH_CKPT_TICKS", "3"))):
            for nid in rng.sample(list(g.nodes), 30):
                g.stimulate(nid, 2.0)
            r = g.step()
            trace.append((len(r.fired_node_ids), r.synapses_pruned, r.synapses_sprouted))
            ids = rng.sample(list(g.nodes), 12)
            p = g.prime_and_propagate(ids, [1.5] * 12, steps=3, write_mode=True)
            q = g.prime_and_propagate(ids[:6], [1.0] * 6, steps=3, write_mode=False)
            trace.append([(e.node_id, e.firing_step, e.source_distance) for e in p.fired_entries])
            trace.append([(e.node_id, e.firing_step, e.source_distance) for e in q.fired_entries])
        res["trace"] = trace
        return res
    (a, ra, ga), (b, rb, gb) = run_twins(base, act, 7)
    assert ra["adj"] is True and rb["adj"] is True
    assert ra == rb
    assert a == b
