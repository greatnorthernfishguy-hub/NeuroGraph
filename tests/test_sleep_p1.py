# ---- Changelog ----
# [2026-10-07] Claude (lane sleep-observe) — test_purge_is_never_called_by_the_engine: by intent, the engine's one
#   sleep_cycle call is sleep_observe's shadow.sleep_cycle() (the real sleep on a private copy; never the live graph).
# [2026-10-07] Claude (lane sleep-prearm) — workload(..., compete=True): the want-hub competition call every 10 rounds can
#   be switched off (compete=False) or replaced (a callable), for whole runs with the disuse sleep on, where the engine now
#   REFUSES compete_protected_links (#1066). Default unchanged: every existing caller runs the same workload as before.
# [2026-10-06] Claude (lane sleep-p1) — CREATE: sleep phase P1 + D15 equivalence and invariants
# What: (1) flag ABSENT whole runs vs the trial tip de8b214 (P2b workload: step, write-mode Tonic ticks with aging,
#       recall, node churn, rewards, the want-hub competition, sleep_downscale, snapshots; per-step synapse state
#       bitwise; final checkpoint bytes), dict and native node store, 5 config flag sets:
#         - "p1_only": the branch with D15 switched back to the base removal function == de8b214 EXACTLY;
#         - full branch == de8b214 + a test shim that applies D15's one named change (drop pre.pred_weights[post]
#           when the last pre->post synapse goes) — so D15 changes nothing else.
#       (2) flag ON (structural_plasticity_in_sleep): step() / the Tonic tail == the trial tip with ONLY those two
#       removal calls skipped; a periodic sleep_cycle == the trial tip's step-8 removal pair run once.
#       (3) sleep_cycle at a given state removes exactly the synapses / nodes the per-step path removes at that state
#       (same predicates, lifelines, last-link holds), same counters after, same checkpoint bytes; record + event.
#       (4) D15: no dangling pred_weights after removal (synapse, duplicate pair, node, prune, whole run); the purge's
#       counts, idempotence and untouched values; the intended change (a re-created pair starts from the 0.5 prior).
# Why:  spec superpowers/specs/2026-10-06-sleep-phase-design.md §8 P1 proof bar.
# How:  BASE neuro_foundation is imported from git (de8b214) under its own name. Fixed PYTHONHASHSEED (harness).
# -------------------
"""Sleep phase P1 (sleep_cycle, structural_plasticity_in_sleep) and D15 (pred_weights consistency)."""
import importlib.util
import logging
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

BASE_REV = os.environ.get("NG_SLEEP_P1_BASE_REV", "de8b214")   # trial tip this branch was cut from
HAVE_STORE = hasattr(NF.ng_tract, "NodeStore")
MODES = ["off", "on"] if HAVE_STORE else ["off"]
KEY = "structural_plasticity_in_sleep"


def _load_base(fname, modname):
    src = subprocess.run(["git", "-C", _REPO, "show", f"{BASE_REV}:{fname}"],
                         check=True, capture_output=True, text=True).stdout
    d = tempfile.mkdtemp(prefix="sleep_p1_base_")
    p = os.path.join(d, modname + ".py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(modname, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base("neuro_foundation.py", "nf_base_sleep_p1")


def new_graph(mode, config=None):
    if mode == "base":
        return BASE.Graph(config)
    return NF.Graph(config, native_node_store=(mode == "on"))


def mod_of(mode):
    return BASE if mode == "base" else NF


def P(x):
    if type(x) is float:
        return ("f", struct.pack("<d", x))
    return (type(x).__name__, repr(x))


def syn_state(g):
    return [(sid, P(s.weight), P(s.eligibility_trace), P(s.peak_weight), P(s.last_update_time), P(s.max_weight),
             s.low_weight_steps, s.inactive_steps) for sid, s in g.synapses.items()]


def pw_state(g):
    return [(nid, sorted((k, P(v)) for k, v in n.pred_weights.items())) for nid, n in g.nodes.items()]


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
        return fn(random.Random(seed + 77))
    finally:
        uuid.uuid4 = saved


def dangling(g):
    """(node, key) pairs whose key has no node->key synapse or names a missing node (the D15 invariant)."""
    out = []
    for nid, n in g.nodes.items():
        if not n.pred_weights:
            continue
        targets = {g.synapses[s].post_node_id for s in g._outgoing.get(nid, ()) if s in g.synapses}
        out.extend((nid, k) for k in n.pred_weights if k not in targets or k not in g.nodes)
    return out


# ---------------------------------------------------------------------------
# graph builder + workload (the P2b whole-run workload, plus DiffPC-consistent pred_weights)
# ---------------------------------------------------------------------------

LONG = "turn text — Ünïcødé 𝔘 " * 30


def build_random_graph(mode, seed, n_nodes=120, n_syn=900):
    def build(rng):
        g = new_graph(mode, {"three_factor_enabled": rng.random() < 0.5, "scaling_interval": 3,
                             "grace_period": rng.choice([50, 5000]),
                             "inactivity_threshold": rng.choice([40, 1000, float("inf")]),
                             "sprout_degree_cap": rng.choice([0, 30]),
                             "orphan_node_grace_period": rng.choice([0, 5, 25])})
        ids = []
        for i in range(n_nodes):
            meta = rng.choice([{}, {}, {"constitutional": True}, {"provenance": "syl_authored"},
                               {"provenance": "cc_emergent"}, {"provenance": "cc_authored"},
                               {"_forest_content": LONG + str(i % 5), "kind": "tree"}])
            n = g.create_node(node_id=f"n{seed}_{i}", metadata=dict(meta), is_inhibitory=rng.random() < 0.1)
            n.firing_rate_ema = rng.choice([0.0, 0.0, rng.uniform(0, 0.3)])
            if rng.random() < 0.4:
                n.last_spike_time = float(rng.randint(0, 120))
            n.diffpc_layer = rng.choice([0, 1, 1, 2, 2])
            n.Ca_i = rng.choice([0.0, rng.uniform(0, 2)])
            ids.append(n.node_id)
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
            if rng.random() < 0.5:                                  # a DiffPC-written entry (consistent)
                g.nodes[a].pred_weights[b] = rng.uniform(0, 1)
            made += 1
        for nid in rng.sample(ids, 6):                              # pre-existing DANGLING entries (old removals)
            g.nodes[nid].pred_weights[rng.choice(ids)] = rng.uniform(0, 1)
            g.nodes[nid].pred_weights[f"gone{seed}_{nid}"] = 0.25
        g.timestep = rng.randint(100, 9000)
        return g
    return seeded(seed, build)


SEEDS = list(range(4))
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


def workload(rounds, sleep_every=0, sleep_fn=None, compete=True):
    """sleep_fn(g) -> comparable record of one sleep; called every `sleep_every` rounds when given.
    compete: True = g.compete_protected_links(2, 10) every 10 rounds (the default, unchanged); False = skipped; a callable
    = called instead (its return value is logged)."""
    def act(g, rng):
        log, snaps = [], []
        for k in range(rounds):
            for nid in rng.sample(list(g.nodes), min(8, len(g.nodes))):
                g.stimulate(nid, rng.uniform(0.5, 3.0))
            r = g.step()
            log.append((list(r.fired_node_ids), r.synapses_pruned, r.synapses_sprouted, r.diffpc_ternary_spikes))
            log.append([(nid, P(n.voltage), P(n.threshold), P(n.Ca_i), n.refractory_remaining)
                        for nid, n in g.nodes.items()])
            log.append(syn_state(g))
            log.append(pw_state(g))
            g.config["tonic_ages_substrate"] = 1
            ids = rng.sample(list(g.nodes), min(6, len(g.nodes)))
            p = g.prime_and_propagate(ids, [1.5] * len(ids), steps=3, write_mode=True)    # Tonic tick (ages, tail)
            log.append(fe(p.fired_entries))
            log.append(syn_state(g))
            q = g.prime_and_propagate(ids[:3], [2.0] * len(ids[:3]), steps=4, write_mode=False)
            log.append(fe(q.fired_entries))
            if k % 3 == 1:
                n = g.create_node(node_id=f"new{k}", metadata={"_forest_content": LONG, "k": k})
                n.voltage = 0.25
                for other in rng.sample(list(g.nodes), 2):
                    if other != n.node_id:
                        g.create_synapse(n.node_id, other, weight=0.3)
            if k % 4 == 2:                                          # re-create a removed-then-sprouted pair shape
                a, b = rng.sample(list(g.nodes), 2)
                sid = g.create_synapse(a, b, weight=0.05).synapse_id
                g.nodes[a].pred_weights[b] = 0.9
                g.remove_synapse(sid)
                g.create_synapse(a, b, weight=0.05)
            if k % 7 == 3 and len(g.nodes) > 20:
                g.remove_node(rng.choice(list(g.nodes)))
            if k % 5 == 2:
                for j in range(3):
                    iso = g.create_node(node_id=f"iso{k}_{j}", metadata={"k": k})
                    iso.creation_time = int(g.timestep) - 10_000
            if k == 15:
                for nid in sorted(rng.sample(list(g.nodes), len(g.nodes) // 4)):
                    g.remove_node(nid)
            if k % 5 == 4:
                g.inject_reward(0.2)
                log.append(type(g)._collect_orphan_nodes(g))       # class call: never an instance patch
                log.append(repr(g.get_telemetry()))
            if sleep_fn is not None and sleep_every and k % sleep_every == sleep_every - 1:
                log.append(sleep_fn(g))
            if k % 10 == 9:
                if compete is True:
                    log.append(g.compete_protected_links(2, 10))
                elif callable(compete):
                    log.append(compete(g))
                log.append(sorted(g.sleep_downscale(0.9).items()))
                snaps.append(ckpt_bytes(g))
        log.append(list(g.nodes))
        return log, snaps
    return act


def run(mode, start_bytes, act, seed, prep=None):
    g = restore(mode, start_bytes)
    if prep is not None:
        prep(g)
    out = seeded(seed, lambda rng: act(g, rng), uuid_stream=1)
    return ckpt_bytes(g), out


def d15_shim(g):
    """Applies to a BASE graph the one change D15 names: drop pre.pred_weights[post] once no pre->post synapse is left."""
    orig = g._remove_synapse_internal

    def wrapped(sid):
        s = g.synapses.get(sid)
        pair = (s.pre_node_id, s.post_node_id) if s is not None else None
        orig(sid)
        if pair is not None:
            n = g.nodes.get(pair[0])
            if n is not None and pair[1] in n.pred_weights and g._find_synapse(*pair) is None:
                del n.pred_weights[pair[1]]
    g._remove_synapse_internal = wrapped


def p1_only(g):
    """The branch graph with D15 switched back off (the base removal function): P1 alone."""
    g._remove_synapse_internal = types.MethodType(BASE.Graph._remove_synapse_internal, g)


def compare(la, lb, sa, sb, a, b):
    assert len(la) == len(lb)
    for i, (x, y) in enumerate(zip(la, lb)):
        assert x == y, f"log entry {i} differs"
    assert sa == sb
    assert a == b


# ---------------------------------------------------------------------------
# (1) flag absent: the trial tip, exactly (P1 alone) and up to D15's named change (full branch)
# ---------------------------------------------------------------------------

def _cfg_prep(cfg, extra=None):
    def prep(g):
        g.config.update(cfg)
        if extra is not None:
            extra(g)
    return prep


@pytest.mark.parametrize("flags", list(FLAGS))
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_flag_absent_p1_alone_is_the_trial_tip_exactly(mode, seed, flags):
    start = built_bytes(seed)
    act = workload(30)
    a, (la, sa) = run("base", start, act, seed, _cfg_prep(FLAGS[flags]))
    b, (lb, sb) = run(mode, start, act, seed, _cfg_prep(FLAGS[flags], p1_only))
    compare(la, lb, sa, sb, a, b)


@pytest.mark.parametrize("flags", list(FLAGS))
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_flag_absent_full_branch_is_the_trial_tip_plus_only_the_d15_change(mode, seed, flags):
    start = built_bytes(seed)
    act = workload(30)
    a, (la, sa) = run("base", start, act, seed, _cfg_prep(FLAGS[flags], d15_shim))
    b, (lb, sb) = run(mode, start, act, seed, _cfg_prep(FLAGS[flags]))
    compare(la, lb, sa, sb, a, b)
    # non-vacuous: without the shim the trial tip keeps entries D15 drops
    c, (lc, _) = run("base", start, act, seed, _cfg_prep(FLAGS[flags]))
    assert c != b


def test_flag_absent_step_calls_both_removal_functions(monkeypatch):
    assert KEY not in NF.DEFAULT_CONFIG
    g = restore("off", built_bytes(0))
    calls = []
    monkeypatch.setattr(g, "_prune_synapses", lambda **kw: (calls.append("prune"), 0)[1])
    monkeypatch.setattr(g, "_collect_orphan_nodes", lambda: (calls.append("collect"), 0)[1])
    g.step()
    g.config["tonic_ages_substrate"] = 1
    g.prime_and_propagate(list(g.nodes)[:4], [1.5] * 4, steps=2, write_mode=True)
    assert calls == ["prune", "collect", "prune", "collect"]


# ---------------------------------------------------------------------------
# (2) flag on: step() / Tonic tail skip exactly the removal pair; sleep_cycle is the step-8 pair run once
# ---------------------------------------------------------------------------

def base_without_wake_removal(g):
    """The trial tip with ONLY step 8's and the Tonic tail's removal calls skipped (instance patches; explicit
    class-level calls and the competing-mode prune still run the real functions)."""
    real_prune = g._prune_synapses
    g._prune_synapses = lambda **kw: real_prune(**kw) if kw else 0
    g._collect_orphan_nodes = lambda: 0


def base_sleep(g):
    pruned = type(g)._prune_synapses(g)
    collected = type(g)._collect_orphan_nodes(g)
    g._total_pruned += pruned
    return (pruned, collected, len(g.synapses), len(g.nodes))


def branch_sleep(g):
    r = g.sleep_cycle()
    assert r["in_sleep_mode"] is True
    return (r["pruned"], r["nodes_collected"], r["synapses_after"], r["nodes_after"])


@pytest.mark.parametrize("flags", list(FLAGS))
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_flag_on_wake_skips_only_removal_and_sleep_cycle_is_the_step8_pair(mode, seed, flags):
    start = built_bytes(seed)
    cfg = dict(FLAGS[flags], **{KEY: True})

    def both(extra):
        def prep(g):
            g.config.update(cfg)
            extra(g)
        return prep
    a, (la, sa) = run("base", start, workload(30, 6, base_sleep), seed,
                      both(lambda g: (d15_shim(g), base_without_wake_removal(g))))
    b, (lb, sb) = run(mode, start, workload(30, 6, branch_sleep), seed, both(lambda g: None))
    compare(la, lb, sa, sb, a, b)
    # wake never removed anything on the branch (step-level counts are in the log: entry 0 of every round)
    assert all(e[1] == 0 for e in lb if isinstance(e, tuple) and len(e) == 4 and isinstance(e[0], list))


def test_flag_on_wake_still_sprouts_and_ages(monkeypatch):
    g = restore("off", built_bytes(1))
    g.config[KEY] = True
    seen = []
    g.register_event_handler("pruned", lambda **kw: seen.append(("pruned", kw)))
    g.register_event_handler("nodes_collected", lambda **kw: seen.append(("nodes_collected", kw)))
    sprouted = 0
    inactive0 = [s.inactive_steps for s in g.synapses.values()]
    for k in range(10):
        for nid in list(g.nodes)[k::7][:12]:
            g.stimulate(nid, 3.0)
        r = g.step()
        assert r.synapses_pruned == 0
        sprouted += r.synapses_sprouted
    g.config["tonic_ages_substrate"] = 1
    g.prime_and_propagate(list(g.nodes)[:6], [2.0] * 6, steps=3, write_mode=True)
    assert seen == []
    assert sprouted > 0
    assert any(s.inactive_steps > i for s, i in zip(g.synapses.values(), inactive0))


# ---------------------------------------------------------------------------
# (3) at one state: sleep_cycle removes exactly what the per-step path removes
# ---------------------------------------------------------------------------

def _removal_probe(g):
    rec = {"nodes": []}
    g.register_event_handler("nodes_collected", lambda **kw: rec["nodes"].extend(kw.get("node_ids", [])))
    return rec


@pytest.mark.parametrize("grace", [50, 5000])
@pytest.mark.parametrize("flags", ["off", "lifeline", "lifeline_budget"])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_sleep_cycle_removes_exactly_the_per_step_set(mode, seed, flags, grace):
    # advance a few wake steps on the trial tip so last-link stamps and counters exist, then fork the state
    def warm(g):
        g.config.update(FLAGS[flags])
        g.config["grace_period"] = grace
        g.config["last_link_grace_steps"] = 3
        for k in range(6):
            for nid in list(g.nodes)[k::9][:10]:
                g.stimulate(nid, 2.5)
            g.step()
        g.config[KEY] = True
    gb = restore("base", built_bytes(seed))
    seeded(seed, lambda rng: warm(gb))
    state = ckpt_bytes(gb)
    base = restore("base", state)
    br = restore(mode, state)
    for g in (base, br):
        g.config.update(FLAGS[flags])
        g.config.update({"grace_period": grace, "last_link_grace_steps": 3, KEY: True})
    pb, pr = _removal_probe(base), _removal_probe(br)
    syn0, nodes0 = set(base.synapses.keys()), set(base.nodes)
    assert syn0 == set(br.synapses.keys()) and nodes0 == set(br.nodes)
    ends = {sid: (s.pre_node_id, s.post_node_id) for sid, s in base.synapses.items()}
    pruned = base._prune_synapses()                       # the trial tip's step-8 removal pair
    collected = base._collect_orphan_nodes()
    base._total_pruned += pruned
    events = []
    br.register_event_handler("sleep_cycle", lambda **kw: events.append(kw))
    rec = br.sleep_cycle()
    gone = syn0 - set(base.synapses.keys())
    assert syn0 - set(br.synapses.keys()) == gone
    assert pr["nodes"] == pb["nodes"]
    assert (rec["pruned"], rec["nodes_collected"]) == (pruned, collected)
    assert rec["synapses_before"] == len(syn0) and rec["synapses_after"] == len(br.synapses)
    assert rec["nodes_before"] == len(nodes0) and rec["nodes_after"] == len(br.nodes)
    assert len(events) == 1 and events[0]["pruned"] == pruned
    assert syn_state(br) == syn_state(base)               # counters (low_weight_steps) advanced identically
    # pred_weights: the branch == the trial tip minus exactly D15's entries (a removed pair with no synapse left)
    still = {(s.pre_node_id, s.post_node_id) for s in br.synapses.values()}
    cut = {ends[sid] for sid in gone} - still
    for nid, n in base.nodes.items():
        want = {k: v for k, v in n.pred_weights.items() if (nid, k) not in cut}
        assert sorted((k, P(v)) for k, v in br.nodes[nid].pred_weights.items()) == sorted((k, P(v)) for k, v in want.items())
        for k in [k for k in n.pred_weights if (nid, k) in cut]:
            del n.pred_weights[k]
    assert ckpt_bytes(br) == ckpt_bytes(base)


def test_sleep_cycle_is_not_vacuous_somewhere():
    g = restore("off", built_bytes(2))
    g.config.update({"grace_period": 50, "orphan_node_grace_period": 0, KEY: True})
    r = g.sleep_cycle()
    assert r["pruned"] > 0


def test_sleep_cycle_logs_one_info_line(caplog):
    g = restore("off", built_bytes(0))
    with caplog.at_level(logging.INFO, logger=NF.logger.name):
        g.sleep_cycle()
    lines = [r for r in caplog.records if r.getMessage().startswith("sleep_cycle:")]
    assert len(lines) == 1 and "in_sleep_mode=False" in lines[0].getMessage()


# ---------------------------------------------------------------------------
# (4) D15
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_remove_synapse_drops_the_entry_only_with_the_last_parallel_synapse(mode):
    g = new_graph(mode)
    for n in ("a", "b", "c"):
        g.create_node(node_id=n)
    s1 = g.create_synapse("a", "b").synapse_id
    s2 = g.create_synapse("a", "b").synapse_id
    s3 = g.create_synapse("a", "c").synapse_id
    g.nodes["a"].pred_weights.update({"b": 0.7, "c": 0.3})
    g.remove_synapse(s1)
    assert dict(g.nodes["a"].pred_weights) == {"b": 0.7, "c": 0.3}
    g.remove_synapse(s2)
    assert dict(g.nodes["a"].pred_weights) == {"c": 0.3}
    g._remove_synapse_internal(s3)
    assert dict(g.nodes["a"].pred_weights) == {}
    g._remove_synapse_internal("missing")                 # still no KeyError


@pytest.mark.parametrize("mode", MODES)
def test_remove_node_drops_every_in_neighbours_entry(mode):
    g = new_graph(mode)
    for n in ("a", "b", "x", "y"):
        g.create_node(node_id=n)
    g.create_synapse("a", "x")
    g.create_synapse("b", "x")
    g.create_synapse("b", "y")
    g.create_synapse("x", "y")
    g.nodes["a"].pred_weights["x"] = 0.6
    g.nodes["b"].pred_weights.update({"x": 0.4, "y": 0.8})
    g.nodes["x"].pred_weights["y"] = 0.5
    g.remove_node("x")
    assert dict(g.nodes["a"].pred_weights) == {}
    assert dict(g.nodes["b"].pred_weights) == {"y": 0.8}
    assert dangling(g) == []


def test_the_intended_trace_change_a_recreated_pair_starts_from_the_prior():
    """The one named behaviour change: DiffPC reads pred_weights.get(post, 0.5); the trial tip kept the old value
    across remove + re-sprout, D15 drops it with the last synapse."""
    out = {}
    for mode in ("base", "off"):
        g = new_graph(mode)
        g.create_node(node_id="p")
        g.create_node(node_id="q")
        sid = g.create_synapse("p", "q").synapse_id
        g.nodes["p"].pred_weights["q"] = 0.93
        g.remove_synapse(sid)
        g.create_synapse("p", "q")
        out[mode] = g.nodes["p"].pred_weights.get("q", 0.5)
    assert out == {"base": 0.93, "off": 0.5}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", SEEDS[:2])
def test_whole_run_keeps_pred_weights_consistent_after_the_purge(mode, seed):
    g = restore(mode, built_bytes(seed))
    g.config.update(FLAGS["lifeline_budget"])
    assert dangling(g)                                     # the builder plants dangling entries
    rec = g.purge_dangling_pred_weights()
    assert dangling(g) == [] and rec["entries_removed"] > 0
    bad = []

    def act(gr, rng):
        for k in range(25):
            for nid in rng.sample(list(gr.nodes), min(8, len(gr.nodes))):
                gr.stimulate(nid, rng.uniform(0.5, 3.0))
            gr.step()
            bad.extend(dangling(gr))
            gr.config["tonic_ages_substrate"] = 1
            ids = rng.sample(list(gr.nodes), min(6, len(gr.nodes)))
            gr.prime_and_propagate(ids, [1.5] * len(ids), steps=3, write_mode=True)
            bad.extend(dangling(gr))
            if k % 6 == 5 and len(gr.nodes) > 20:
                for nid in rng.sample(list(gr.nodes), 5):
                    gr.remove_node(nid)
                bad.extend(dangling(gr))
            if k % 8 == 7:
                gr.sleep_cycle()
                bad.extend(dangling(gr))
    seeded(seed, lambda rng: act(g, rng), uuid_stream=3)
    assert bad == []
    assert sum(len(n.pred_weights) for n in g.nodes.values()) > 0   # DiffPC kept writing (non-vacuous)


@pytest.mark.parametrize("mode", MODES)
def test_purge_counts_idempotence_and_untouched_values(mode):
    g = new_graph(mode)
    for n in ("a", "b", "c", "d"):
        g.create_node(node_id=n)
    g.create_synapse("a", "b")
    g.create_synapse("c", "a")
    g.nodes["a"].pred_weights.update({"b": 0.123456789, "c": 0.5, "zz": 0.1})   # c: no a->c synapse; zz: no node
    g.nodes["c"].pred_weights.update({"a": 0.25})
    g.nodes["d"].pred_weights.update({"a": 0.9})                                # d has no synapse at all
    rec = g.purge_dangling_pred_weights()
    assert {k: rec[k] for k in ("entries_before", "entries_removed", "removed_key_node_missing", "entries_after",
                                "nodes_touched", "nodes_with_pred_weights")} == {
        "entries_before": 5, "entries_removed": 3, "removed_key_node_missing": 1, "entries_after": 2,
        "nodes_touched": 2, "nodes_with_pred_weights": 3}
    assert dict(g.nodes["a"].pred_weights) == {"b": 0.123456789}
    assert dict(g.nodes["c"].pred_weights) == {"a": 0.25}
    assert dict(g.nodes["d"].pred_weights) == {}
    again = g.purge_dangling_pred_weights()
    assert again["entries_removed"] == 0 and again["entries_after"] == 2


def test_purge_is_never_called_by_the_engine():
    import re
    src = open(os.path.join(_REPO, "neuro_foundation.py")).read()
    assert set(re.findall(r"(\w+)\.purge_dangling_pred_weights\(", src)) <= {"Graph"}   # changelog mention only
    # [2026-10-07] by intent (lane sleep-observe): Graph.sleep_observe runs the real sleep on its private SHADOW copy
    # (shadow.sleep_cycle()); the engine still never calls sleep_cycle on a live graph.
    assert set(re.findall(r"(\w+)\.sleep_cycle\(", src)) <= {"Graph", "shadow"}
    for m in re.finditer(r"shadow\.sleep_cycle\(", src):
        assert re.findall(r"^    def (\w+)\(", src[:m.start()], re.M)[-1] == "sleep_observe"
    assert len(re.findall(r"def purge_dangling_pred_weights\(", src)) == 1
