# [2026-10-06] Claude (lane sleep-p1) — D15 (sleep-phase spec): the BASE engine gets D15's one named intended change
#   (tests/d15_intended.apply_d15_rule: drop pre.pred_weights[post] with the last pre->post synapse) so this file keeps checking
#   everything else bitwise; with D15 switched off the file passes unchanged (SLEEP_P1.md §6).
# [2026-10-05] Claude (lane rust-hotpaths-onto-s4) — equivalence: this branch's native hot paths vs the TRIAL TIP's code
"""Bit-for-bit equivalence of the native SynapseStore hot paths against the trial tip.

BASE = the WHOLE neuro_foundation.py of the trial tip (pinned BASE_REV, read from git and
imported as a separate module). BRANCH = this worktree's neuro_foundation.py. Twin graphs
are restored from the SAME checkpoint bytes into each module, the same action runs on each
(same `random` seed, deterministic uuid4), and the full msgpack checkpoints must be
BYTE-IDENTICAL, plus identical return values / per-step traces.

Covered: every converted site (prune with lifeline off / on / grace variants / competing
mode, inject_reward, homeostatic scaling, STDP 2- and 3-factor, step() propagation,
prime_and_propagate BFS + propagate in read and write mode, sprouting, telemetry,
extract_subgraph, restore adjacency incl. set iteration order), whole runs under every
flag combination (lifeline on/off x strength budget on/off), and the same on the
PYTHON FALLBACK path (a store proxy that hides the native batch methods).

Optional real-checkpoint pass: NG_ONTO_S4_CKPT=<a COPY of a checkpoint>. The live plugin
directory is refused; the copy is only ever read.
"""
import gc
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

BASE_REV = os.environ.get("NG_ONTO_S4_BASE_REV", "5246b63")   # trial tip this branch was cut from

NATIVE = ("advance_low_weight_and_collect_prune", "apply_eligibility_reward", "apply_stdp_dw",
          "bfs_hop_distances", "creation_time_copy", "endpoint_triples", "post_ids_of",
          "pre_ids_of", "propagation_rows", "scale_weights_by_post_node", "stdp_reads")
if not all(hasattr(NF.ng_tract.SynapseStore, m) for m in NATIVE):
    pytest.skip("ng_tract wheel without the hot-path methods", allow_module_level=True)


def _load_base():
    try:
        src = subprocess.run(["git", "-C", _REPO, "show", f"{BASE_REV}:neuro_foundation.py"],
                             check=True, capture_output=True, text=True).stdout
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"base source {BASE_REV} unavailable: {exc}", allow_module_level=True)
    d = tempfile.mkdtemp(prefix="nf_base_s4_")
    p = os.path.join(d, "nf_base_s4.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nf_base_s4", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nf_base_s4"] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base()
from tests.d15_intended import apply_d15_rule  # noqa: E402  [2026-10-06] sleep-p1: D15's named intended change
apply_d15_rule(BASE)
LIVE_DIR = "/.claude/plugins/neurograph/checkpoints/"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class HideNative:
    """Store proxy WITHOUT the native batch methods -> the branch takes its Python fallbacks."""
    HIDDEN = set(NATIVE) | {"weights_copy"}

    def __init__(self, store):
        object.__setattr__(self, "_s", store)

    def __getattr__(self, name):
        if name in HideNative.HIDDEN:
            raise AttributeError(name)
        return getattr(self._s, name)

    def __len__(self):
        return len(self._s)

    def __contains__(self, k):
        return k in self._s

    def __getitem__(self, k):
        return self._s[k]

    def __setitem__(self, k, v):
        self._s[k] = v

    def __delitem__(self, k):
        del self._s[k]

    def __iter__(self):
        return iter(self._s)


def ckpt_bytes(g):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        g.checkpoint(p)
        with open(p, "rb") as f:
            return f.read()


def restore(mod, b):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        with open(p, "wb") as f:
            f.write(b)
        g = mod.Graph()
        g.restore(p)
        return g


class DetUUID:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def __call__(self):
        return uuid.UUID(int=self.rng.getrandbits(128), version=4)


def run_one(mod, base_bytes, action, seed, fallback=False, keep=False):
    g = restore(mod, base_bytes)
    if fallback:
        g.synapses = HideNative(g.synapses)
    saved = uuid.uuid4
    uuid.uuid4 = DetUUID(seed)
    random.seed(seed)
    try:
        ret = action(mod, g, random.Random(seed))
    finally:
        uuid.uuid4 = saved
    out = (ckpt_bytes(g), ret)
    return out + ((g,) if keep else ())


def twins(base_bytes, action, seed, fallback=False):
    a = run_one(BASE, base_bytes, action, seed)
    b = run_one(NF, base_bytes, action, seed, fallback=fallback)
    return a, b


def build_random_graph(seed, n_nodes=120, n_syn=900):
    """Built with the BRANCH module; only its checkpoint bytes are used."""
    rng = random.Random(seed)
    g = NF.Graph({"three_factor_enabled": rng.random() < 0.5, "scaling_interval": 3,
                  "grace_period": rng.choice([50, 5000]),
                  "inactivity_threshold": rng.choice([40, 1000]),
                  "sprout_degree_cap": rng.choice([0, 30])})
    ids = []
    for i in range(n_nodes):
        meta = rng.choice([{}, {}, {}, {}, {"constitutional": True},
                           {"provenance": "syl_authored"}, {"provenance": "cc_emergent"},
                           {"provenance": "cc_authored"}])
        n = g.create_node(node_id=f"n{seed}_{i}", metadata=dict(meta),
                          is_inhibitory=rng.random() < 0.1)
        n.firing_rate_ema = rng.choice([0.0, 0.0, rng.uniform(0, 0.3)])
        if rng.random() < 0.4:
            n.last_spike_time = float(rng.randint(0, 120))
        ids.append(n.node_id)
    made = 0
    while made < n_syn:
        a, b = rng.sample(ids, 2)
        s = g.create_synapse(a, b, weight=rng.uniform(0.0, 1.2), delay=rng.randint(1, 4),
                             synapse_type=rng.choice([NF.SynapseType.EXCITATORY] * 4
                                                     + [NF.SynapseType.INHIBITORY]))
        s.low_weight_steps = rng.choice([0, 1, 49, 50, 51, rng.randint(0, 6000)])
        s.inactive_steps = rng.choice([0, 39, 40, 41, rng.randint(0, 3000)])
        s.salience = rng.choice([1.0, 1.0, 1.7, rng.uniform(1, 4)])
        s.creation_time = float(rng.randint(-6000, 100))
        s.peak_weight = rng.uniform(0.0, 0.5)
        s.eligibility_trace = rng.choice([0.0, 1e-10, -1e-10, rng.uniform(-0.4, 0.4)])
        if rng.random() < 0.02:
            s.weight = rng.choice([0.0, -0.0, 0.005, 5.0])
        if rng.random() < 0.03:   # pre-existing last-link stamps (some stale, some fresh)
            s.metadata = {"last_link_since": rng.randint(0, 120)}
        made += 1
    for sid in rng.sample(list(g.synapses.keys()), n_syn // 10):  # swap-remove churn
        g.remove_synapse(sid)
    g.timestep = rng.randint(100, 9000)
    return g


_OFF = {"prune_protected_faint_links": False, "strength_budget_enabled": False}
_BUDGET = {"strength_budget_enabled": True, "strength_budget_out": 2.0,
           "strength_budget_in": 2.5, "strength_budget_interval": 2}
# Every named combination sets BOTH switches explicitly (a real checkpoint carries its own
# config; the live CC graph has both ON). "absent" leaves the restored config as it is.
FLAGS = {
    "absent": {},
    "off": dict(_OFF),
    "lifeline": dict(_OFF, prune_protected_faint_links=True),
    "lifeline_grace3": dict(_OFF, prune_protected_faint_links=True, last_link_grace_steps=3),
    "lifeline_grace0": dict(_OFF, prune_protected_faint_links=True, last_link_grace_steps=0),
    "budget": dict(_OFF, **_BUDGET),
    "lifeline_budget": dict(_BUDGET, prune_protected_faint_links=True, last_link_grace_steps=4),
}

SEEDS = list(range(8))
_BASE_CACHE = {}


def base_bytes(seed):
    if seed not in _BASE_CACHE:
        _BASE_CACHE[seed] = ckpt_bytes(build_random_graph(seed))
    return _BASE_CACHE[seed]


def fe(entries):
    return [(e.node_id, e.firing_step, e.voltage_at_fire, e.source_distance, e.was_predicted)
            for e in entries]


# ---------------------------------------------------------------------------
# per-site
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("flags", list(FLAGS))
@pytest.mark.parametrize("seed", SEEDS)
def test_prune_site(seed, flags, fallback):
    def act(mod, g, rng):
        g.config.update(FLAGS[flags])
        out = []
        for k in range(6):
            rep = {} if k % 2 else None
            out.append((g._prune_synapses(report=rep), rep))
            g.timestep += 2
        return out
    (a, ra), (b, rb) = twins(base_bytes(seed), act, seed, fallback)
    assert ra == rb
    assert a == b


@pytest.mark.parametrize("flags", ["absent", "lifeline"])
@pytest.mark.parametrize("seed", SEEDS)
def test_competing_mode_unchanged(seed, flags):
    def act(mod, g, rng):
        g.config.update(FLAGS[flags])
        recs = []
        for _ in range(3):
            r = g.compete_protected_links(2, 15)
            recs.append(r)
            g._prune_synapses()
        return recs
    (a, ra), (b, rb) = twins(base_bytes(seed), act, seed)
    assert ra == rb
    assert a == b


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_inject_reward_site(seed, fallback):
    def act(mod, g, rng):
        nodes = list(g.nodes)
        g.inject_reward(0.3)
        g.inject_reward(-0.2, scope=set(rng.sample(nodes, 10)))
        g.inject_reward(0.1, scope=set())
        g.inject_reward(1, scope=frozenset(rng.sample(nodes, 40)) | {"ghost"})
        g.inject_reward(0.7, scope=list(rng.sample(nodes, 20)))
    (a, _), (b, _) = twins(base_bytes(seed), act, seed, fallback)
    assert a == b


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_homeostatic_scaling_site(seed, fallback):
    def act(mod, g, rng):
        hr = next(r for r in g._plasticity_rules if isinstance(r, mod.HomeostaticRule))
        nodes = list(g.nodes)
        for k in range(7):  # scaling_interval=3 -> the scaling branch runs twice
            hr.apply(g, rng.sample(nodes, 15), g.timestep + k)
    (a, _), (b, _) = twins(base_bytes(seed), act, seed, fallback)
    assert a == b


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("three_factor", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_stdp_site(seed, three_factor, fallback):
    def act(mod, g, rng):
        g.config["three_factor_enabled"] = three_factor
        st = next(r for r in g._plasticity_rules if isinstance(r, mod.STDPRule))
        nodes = list(g.nodes)
        for k in range(4):
            fired = rng.sample(nodes, 20)
            for nid in fired:
                g.nodes[nid].last_spike_time = float(g.timestep + k)
            st.apply(g, fired, g.timestep + k)
    (a, _), (b, _) = twins(base_bytes(seed), act, seed, fallback)
    assert a == b


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_step_and_sprout_site(seed, fallback):
    def act(mod, g, rng):
        nodes = list(g.nodes)
        log = []
        for k in range(12):
            for nid in rng.sample(nodes, 10):
                if nid in g.nodes:
                    g.stimulate(nid, rng.uniform(0.5, 3.0))
            r = g.step()
            log.append((list(r.fired_node_ids), r.synapses_pruned, r.synapses_sprouted))
        log.append(g._sprout_synapses(rng.sample(list(g.nodes), 12)))
        return log
    (a, ra), (b, rb) = twins(base_bytes(seed), act, seed, fallback)
    assert ra == rb
    assert a == b


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_prime_and_propagate_site(seed, fallback):
    def act(mod, g, rng):
        log = []
        for steps in (0, 1, 2, 3, 5):
            for write_mode, age in ((False, 0), (True, 0), (True, 1)):
                g.config["tonic_ages_substrate"] = age
                ids = rng.sample(list(g.nodes), 6)
                p = g.prime_and_propagate(ids, [1.5] * len(ids), steps=steps, write_mode=write_mode)
                log.append(fe(p.fired_entries))
        return log
    (a, ra), (b, rb) = twins(base_bytes(seed), act, seed, fallback)
    assert ra == rb
    assert a == b


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("seed", SEEDS[:4])
def test_telemetry_and_subgraph_site(seed, fallback):
    def act(mod, g, rng):
        t = g.get_telemetry()
        sub = set(rng.sample(list(g.nodes), 50))
        return (repr(t), g.extract_subgraph(sub))
    (a, ra), (b, rb) = twins(base_bytes(seed), act, seed, fallback)
    assert ra == rb
    assert a == b
    assert repr(BASE.Graph().get_telemetry()) == repr(NF.Graph().get_telemetry())


@pytest.mark.parametrize("seed", SEEDS)
def test_restore_adjacency_site(seed):
    b = base_bytes(seed)
    ga, gb = restore(BASE, b), restore(NF, b)
    for attr in ("_outgoing", "_incoming"):
        da, db = getattr(ga, attr), getattr(gb, attr)
        assert list(da) == list(db)                      # key insertion order
        for k in da:
            assert list(da[k]) == list(db[k])            # set iteration order


def test_restore_adjacency_fallback_helper():
    g = restore(NF, base_bytes(0))
    assert NF._endpoint_triples_python(g.synapses) == list(g.synapses.endpoint_triples())


# ---------------------------------------------------------------------------
# whole runs: every flag combination, native and fallback
# ---------------------------------------------------------------------------

def _workload(rounds):
    def act(mod, g, rng):
        log = []
        snaps = []
        for k in range(rounds):
            for nid in rng.sample(list(g.nodes), min(8, len(g.nodes))):
                g.stimulate(nid, rng.uniform(0.5, 3.0))
            r = g.step()
            log.append((list(r.fired_node_ids), r.synapses_pruned, r.synapses_sprouted))
            g.config["tonic_ages_substrate"] = 1
            ids = rng.sample(list(g.nodes), min(6, len(g.nodes)))
            p = g.prime_and_propagate(ids, [1.5] * len(ids), steps=3, write_mode=True)   # Tonic tick
            log.append(fe(p.fired_entries))
            q = g.prime_and_propagate(ids[:3], [2.0] * len(ids[:3]), steps=4, write_mode=False)  # recall
            log.append(fe(q.fired_entries))
            if k % 5 == 4:
                g.inject_reward(0.2)
                g.inject_reward(-0.1, scope=set(rng.sample(list(g.nodes), 10)))
            if k % 10 == 9:
                log.append(g.compete_protected_links(2, 10))
                log.append(sorted(g.sleep_downscale(0.9).items()))
                snaps.append(ckpt_bytes(g))
        return log, snaps
    return act


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("flags", ["absent", "off", "lifeline", "budget", "lifeline_budget"])
@pytest.mark.parametrize("seed", SEEDS[:6])
def test_whole_run(seed, flags, fallback):
    act0 = _workload(30)

    def act(mod, g, rng):
        g.config.update(FLAGS[flags])
        return act0(mod, g, rng)
    (a, (la, sa)), (b, (lb, sb)) = twins(base_bytes(seed), act, seed, fallback)
    assert la == lb
    assert sa == sb
    assert a == b


# ---------------------------------------------------------------------------
# optional: a COPY of the real checkpoint (sequential twins to bound memory)
# ---------------------------------------------------------------------------

CKPT = os.environ.get("NG_ONTO_S4_CKPT")


@pytest.mark.skipif(not CKPT, reason="set NG_ONTO_S4_CKPT to a checkpoint COPY")
@pytest.mark.parametrize("flags", os.environ.get("NG_ONTO_S4_CKPT_FLAGS", "absent").split(","))
def test_checkpoint_copy(flags):
    assert LIVE_DIR not in os.path.abspath(CKPT), "refusing the live checkpoint"
    with open(CKPT, "rb") as f:
        raw = f.read()
    ticks = int(os.environ.get("NG_ONTO_S4_CKPT_TICKS", "3"))

    def act(mod, g, rng):
        g.config.update(FLAGS[flags])
        res = {"prune": [g._prune_synapses() for _ in range(2)]}
        g.inject_reward(0.05)
        g.inject_reward(0.05, scope=set(rng.sample(list(g.nodes), 200)))
        hr = next(r for r in g._plasticity_rules if isinstance(r, mod.HomeostaticRule))
        hr._steps_since_scaling = hr.scaling_interval
        hr.apply(g, rng.sample(list(g.nodes), 50), g.timestep)
        res["tel"] = repr(g.get_telemetry())
        g.config["tonic_ages_substrate"] = 1
        trace = []
        for _ in range(ticks):
            for nid in rng.sample(list(g.nodes), 30):
                g.stimulate(nid, 2.0)
            r = g.step()
            trace.append((list(r.fired_node_ids), r.synapses_pruned, r.synapses_sprouted))
            ids = rng.sample(list(g.nodes), 12)
            p = g.prime_and_propagate(ids, [1.5] * 12, steps=3, write_mode=True)
            q = g.prime_and_propagate(ids[:6], [1.0] * 6, steps=3, write_mode=False)
            trace.append(fe(p.fired_entries))
            trace.append(fe(q.fired_entries))
        res["trace"] = trace
        return res

    a, ra = run_one(BASE, raw, act, 7)
    gc.collect()
    b, rb = run_one(NF, raw, act, 7)
    gc.collect()
    rep_path = os.environ.get("NG_ONTO_S4_CKPT_REPORT")
    if rep_path:   # evidence for the review doc (sizes, digests, what the run actually did)
        import hashlib
        import json
        with open(rep_path, "a") as f:
            f.write(json.dumps({
                "flags": flags, "ticks": ticks, "ckpt_sha256": hashlib.sha256(raw).hexdigest(),
                "base_out_sha256": hashlib.sha256(a).hexdigest(), "branch_out_sha256": hashlib.sha256(b).hexdigest(),
                "out_bytes": [len(a), len(b)], "prune": ra["prune"], "tel": ra["tel"],
                "steps": [t for t in ra["trace"] if isinstance(t, tuple) and len(t) == 3 and isinstance(t[1], int)
                          for t in [(len(t[0]), t[1], t[2])]],
                "fired_entries_per_pp": [len(t) for t in ra["trace"] if isinstance(t, list)],
                "identical": a == b and ra == rb}) + "\n")
    assert ra == rb
    assert a == b


# ---------------------------------------------------------------------------
# non-protected sites: tonic_engine._extract_graph_features_for_model, rpc metrics weights
# ---------------------------------------------------------------------------

def _base_tonic_features_fn():
    import ast
    import textwrap
    import tonic_engine
    src = subprocess.run(["git", "-C", _REPO, "show", f"{BASE_REV}:tonic_engine.py"],
                         check=True, capture_output=True, text=True).stdout
    tree = ast.parse(src)
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "TonicEngine":
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "_extract_graph_features_for_model":
                    ns = dict(tonic_engine.__dict__)
                    exec(compile(textwrap.dedent(ast.get_source_segment(src, fn)),
                                 f"<{BASE_REV}:tonic_engine>", "exec"), ns)
                    return tonic_engine, ns[fn.name]
    raise AssertionError("base _extract_graph_features_for_model not found")


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("seed", SEEDS[:4])
def test_tonic_features_site(seed, fallback):
    torch = pytest.importorskip("torch")
    pytest.importorskip("surgery.tonic_brain")
    import types
    tonic_engine, base_fn = _base_tonic_features_fn()
    g = restore(NF, base_bytes(seed))
    if seed % 2:   # fewer than 200 synapses
        for sid in list(g.synapses.keys())[150:]:
            g.remove_synapse(sid)
    zero = torch.zeros(768)
    fake = types.SimpleNamespace(_graph=g, _identity_embedding_tensor=lambda: zero)
    fa = base_fn(fake)
    if fallback:
        g.synapses = HideNative(g.synapses)
    fb = tonic_engine.TonicEngine._extract_graph_features_for_model(fake)
    for f in fa.__dataclass_fields__:
        va, vb = getattr(fa, f), getattr(fb, f)
        assert va.dtype == vb.dtype and va.shape == vb.shape, f
        assert torch.equal(va, vb), f


@pytest.mark.parametrize("seed", SEEDS[:4])
def test_rpc_metrics_weights_site(seed):
    import numpy as np
    g = restore(NF, base_bytes(seed))
    old = [s.weight for s in g.synapses.values()]
    new = g.synapses.weights_copy()
    assert list(new) == old
    assert float(np.mean(new)) == float(np.mean(old)) and float(np.std(new)) == float(np.std(old))
