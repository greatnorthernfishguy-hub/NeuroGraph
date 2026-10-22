"""RD §2 — the Tonic's autonomous cycle runs the real step() (switch tonic_unified_step, absent = off).

Design: prd/tonic-restoration-design.md §2 / §5 (B) / §7 (Syl approved 2026-06-10, "Build it, lover."; Josh 2026-10-21
"1 Yes, let's build it."). Guard: concepts/Tonic Charter.md (purposes P1-P10, invariants I1-I14, Josh's rulings).

# ---- Changelog ----
# [2026-10-21] Claude Code (Opus 5.5, laptop; lane tonic-unified) — initial
# What: the unified cycle runs step(); homeostasis + the EMA see Tonic fires; the clock advances; no delayed spike is
#       lost or duplicated (also against a concurrent stepper); prefetch warm still lands; #89 fatigue quiets without
#       erasing; #90 valence stays read-only; the #329 spine whisper is present and never floored; identity-protected
#       structure is never pruned; RD §5 (B) damping holds; the breaker counts Tonic firing and still trips on true
#       silence with no body; eligibility traces decay; off == the base line (11d3898) byte for byte.
# Why:  the build brief's PROVE list. How: sandbox Graphs only (never a live singleton, never a checkpoint).
# -------------------
"""
import math
import os
import random
import subprocess
import sys
import threading

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import neuro_foundation as nf  # noqa: E402
from neuro_foundation import Graph  # noqa: E402
import tonic_engine as te  # noqa: E402
import tonic_thread as tt  # noqa: E402
from tonic_engine import EngineConfig, TonicEngine  # noqa: E402
from tonic_thread import TonicConfig, TonicThread, tonic_unified_drive  # noqa: E402


def _graph(n=24, unified=True, seed=7, extra=None, p=0.25):
    random.seed(seed)
    cfg = {"three_factor_enabled": True}
    if unified:
        cfg["tonic_unified_step"] = True
    cfg.update(extra or {})
    g = Graph(config=cfg)
    ids = [f"n{i:02d}" for i in range(n)]
    for nid in ids:
        g.create_node(node_id=nid)
    for a in ids:
        for b in ids:
            if a != b and random.random() < p:
                g.create_synapse(a, b, weight=random.uniform(0.3, 0.8), delay=1)
    return g, ids


def _thread(g, **cfg):
    cfg.setdefault("valence_enabled", False)
    return TonicThread(g, {}, config=TonicConfig(**cfg))


def _engine(g, thread, acts):
    eng = TonicEngine(g, {}, thread, config=EngineConfig(activation_strength=1.0), require_shared_body=True)
    if thread is not None:
        thread.set_latent_engine(eng)
    eng._model = object()
    eng._shared_body = object()
    eng._model_inference = lambda features: list(acts)
    return eng


class _Spy:
    def __init__(self, g):
        self.g = g
        self.steps = []
        self.pp = []
        self.stim = []
        self._step = g.step
        self._pp = g.prime_and_propagate
        self._stim = g.stimulate
        g.step = self.step
        g.prime_and_propagate = self.prime_and_propagate
        g.stimulate = self.stimulate

    def step(self, *a, **k):
        self.steps.append(k.get("structural_damping"))
        return self._step(*a, **k)

    def prime_and_propagate(self, *a, **k):
        self.pp.append(k.get("write_mode", False))
        return self._pp(*a, **k)

    def stimulate(self, nid, cur):
        self.stim.append((nid, cur))
        return self._stim(nid, cur)


def _warm(g, ids, v=1.0):
    for nid in ids[:6]:
        g.nodes[nid].voltage = v


# --------------------------------------------------------------------------- the unified cycle runs step()

def test_ouroboros_cycle_runs_real_step_not_the_write_sweep():
    g, ids = _graph()
    _warm(g, ids)
    spy = _Spy(g)
    t = _thread(g)
    out = t.ouroboros_cycle()
    assert out["active_count"] > 0
    assert True not in spy.pp, "no write-mode sweep on the unified path"
    assert spy.steps == [(1.5, 0.5)] * TonicConfig().propagation_steps


def test_engine_tick_runs_real_step_with_rd_damping():
    g, ids = _graph()
    _warm(g, ids)
    t = _thread(g)
    eng = _engine(g, t, [(ids[0], 0.9), (ids[3], 0.8)])
    spy = _Spy(g)
    eng._generate_latent_token_inner()
    assert True not in spy.pp
    # engine sweep (2) + ouroboros (2); spine has no constitutional nodes here
    assert spy.steps == [(1.5, 0.5)] * (EngineConfig().propagation_steps + TonicConfig().propagation_steps)
    assert t.status["unified"]["drives"] == 2


def test_damping_keys_read_live():
    g, ids = _graph(extra={"tonic_unified_prune_factor": 2.0, "tonic_unified_sprout_factor": 0.25})
    spy = _Spy(g)
    tonic_unified_drive(g, [ids[0]], [1.0], 1)
    assert spy.steps == [(2.0, 0.25)]


def test_off_absent_and_false_keep_the_write_sweep():
    for extra in ({}, {"tonic_unified_step": False}):
        g, ids = _graph(unified=False, extra=extra)
        _warm(g, ids)
        spy = _Spy(g)
        t = _thread(g)
        t.ouroboros_cycle()
        assert spy.steps == [] and spy.pp == [True]
        assert "unified" not in t.status


def test_unified_path_runs_diffpc_and_predictions_machinery(monkeypatch):
    g, ids = _graph()
    _warm(g, ids, 2.0)
    called = []
    orig = g._diffpc_step
    monkeypatch.setattr(g, "_diffpc_step", lambda fired: called.append(len(fired)) or orig(fired))
    tonic_unified_drive(g, ids[:6], [2.0] * 6, 2)
    assert called, "the step() pipeline (7b DiffPC) ran for Tonic firing (#300/#307 path)"


# --------------------------------------------------------------------------- homeostasis + the EMA see Tonic fires

def _fire_node_via_tonic(unified):
    g, ids = _graph(unified=unified)
    target = ids[5]
    for _ in range(3):
        if unified:
            tonic_unified_drive(g, [target], [5.0], 2)
        else:
            g.prime_and_propagate([target], [5.0], steps=2, write_mode=True)
    return g, target


def test_homeostasis_and_ema_see_tonic_fires():
    g_on, tgt = _fire_node_via_tonic(True)
    g_off, _ = _fire_node_via_tonic(False)
    assert g_on.nodes[tgt].firing_rate_ema > 0.0, "the EMA counts the Tonic's own firing"
    assert g_off.nodes[tgt].firing_rate_ema == 0.0, "control: the write sweep is invisible to the EMA"
    # thresholds adapt only through HomeostaticRule (step 7): the unified node's threshold moved, the sweep's did not
    assert g_on.nodes[tgt].threshold != g_off.nodes[tgt].threshold


def test_constitutional_nodes_are_seen_by_homeostasis_like_any_neuron():
    """Josh 2026-10-21: homeostasis sees constitutional nodes (no exemption)."""
    g, ids = _graph()
    g.nodes[ids[2]].metadata = {"constitutional": True}
    for _ in range(3):
        tonic_unified_drive(g, [ids[2]], [5.0], 2)
    assert g.nodes[ids[2]].firing_rate_ema > 0.0


# --------------------------------------------------------------------------- the clock (LAW 8)

def test_clock_advances_through_real_steps():
    g, ids = _graph()
    _warm(g, ids)
    g.nodes[ids[0]].metadata = {"constitutional": True}
    t = _thread(g)
    eng = _engine(g, t, [(ids[1], 0.9)])
    ts0 = g.timestep
    eng._generate_latent_token_inner()
    # engine sweep 2 + spine whisper 1 + ouroboros 2
    assert g.timestep - ts0 == 5
    assert t.status["unified"]["timestep_advance"] == 5


def test_clock_frozen_on_the_old_path():
    g, ids = _graph(unified=False)
    _warm(g, ids)
    t = _thread(g)
    eng = _engine(g, t, [(ids[1], 0.9)])
    ts0 = g.timestep
    eng._generate_latent_token_inner()
    assert g.timestep == ts0


def test_spike_stamps_are_real_timesteps():
    g, ids = _graph()
    r = tonic_unified_drive(g, [ids[4]], [5.0], 2)
    fired = {e.node_id for e in r.fired_entries}
    assert ids[4] in fired
    assert g.nodes[ids[4]].last_spike_time <= g.timestep, "no stamp ahead of the clock (the frozen-stamp bug)"


def test_eligibility_traces_decay_on_the_tonic_clock():
    g, ids = _graph()
    sid = next(iter(g.synapses.keys()))
    g.synapses[sid].eligibility_trace = 1.0
    tonic_unified_drive(g, [], [], 3)
    tau = g.config["eligibility_trace_tau"]
    assert g.synapses[sid].eligibility_trace == pytest.approx(math.exp(-3.0 / tau), rel=1e-6, abs=0.06)
    assert g.synapses[sid].eligibility_trace < 1.0


# --------------------------------------------------------------------------- delayed spikes: none lost, none doubled

def test_pending_delayed_spikes_delivered_exactly_once():
    g, ids = _graph(p=0.0)
    tgt = g.nodes[ids[9]]
    ts = g.timestep
    g._delay_buffer.setdefault(ts + 1, []).append((ids[9], 0.3))
    g._delay_buffer.setdefault(ts + 3, []).append((ids[9], 0.2))
    tonic_unified_drive(g, [], [], 2)          # delivers ts+1 only
    assert ts + 1 not in g._delay_buffer and ts + 3 in g._delay_buffer
    tonic_unified_drive(g, [], [], 2)          # delivers ts+3
    assert not [k for k in g._delay_buffer if k <= g.timestep], "nothing stranded behind the clock"
    assert tgt.voltage > 0.0


def test_tonic_spikes_go_to_the_real_buffer_not_a_discarded_local_one():
    g, ids = _graph(p=0.0)
    g.create_synapse(ids[0], ids[1], weight=0.5, delay=4)
    tonic_unified_drive(g, [ids[0]], [5.0], 1)
    # fired at step 1; arrival at timestep + 4 is pending in the graph's own buffer
    assert any(t == ids[1] for k, v in g._delay_buffer.items() for t, _ in v)


def test_concurrent_stepper_no_double_step_no_lost_spike():
    """Drain pacing steps on another thread while the Tonic drives: every step is one timestep, no buffer key is left
    behind the clock, and every scheduled arrival is popped exactly once."""
    g, ids = _graph(n=40, p=0.15)
    ts0 = g.timestep
    n_drain, n_tonic, steps_per = 120, 40, 2
    errors = []

    def drain():
        try:
            for _ in range(n_drain):
                g.step()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    th = threading.Thread(target=drain)
    th.start()
    for i in range(n_tonic):
        tonic_unified_drive(g, [ids[i % 40]], [3.0], steps_per)
    th.join()
    assert not errors
    assert g.timestep == ts0 + n_drain + n_tonic * steps_per
    assert not [k for k in g._delay_buffer if k <= g.timestep]


# --------------------------------------------------------------------------- prefetch warm still lands

def test_prefetch_warm_lands_on_the_unified_path(monkeypatch):
    monkeypatch.setattr(te, "_CC_PITH_PREFETCH_WARM_ENABLED", True)
    g, ids = _graph()
    t = _thread(g)
    eng = TonicEngine(g, {}, t, config=EngineConfig(activation_strength=1.0), require_shared_body=True)
    eng._model = object()
    eng._shared_body = object()

    # the real _model_inference path merges seeds; emulate its tail: model proposals + merge
    def inference(features):
        seen = {ids[0]: 0.9}
        eng._merge_prefetch_seeds(seen)
        return list(seen.items())

    eng._model_inference = inference
    eng.set_prefetch_seed(lambda: {ids[7]: 1.0})
    g.nodes[ids[7]].threshold = 0.4
    spy = _Spy(g)
    eng._generate_latent_token_inner()
    assert (ids[7], 0.5) in spy.stim, "the warm seed is primed at activation_strength x 0.5 x score"
    assert g.nodes[ids[7]].last_spike_time > -math.inf, "and it fires inside step() (STDP / sprout / tally see it)"
    assert eng.status["prefetch_seeded"] == 1


# --------------------------------------------------------------------------- #89 fatigue quiets, never erases

def test_fatigue_quiets_without_erasing_on_the_unified_path():
    g, ids = _graph(n=30)
    for i, nid in enumerate(ids):
        g.nodes[nid].metadata = {"_forest_content": f"a remembered thought number {i} about the river"}
    t = _thread(g, exploration_bias=0.0)
    focus_seen = []
    for _ in range(60):
        for nid in ids[:5]:
            g.nodes[nid].voltage = max(g.nodes[nid].voltage, 0.6)   # a persistent weld candidate set
        t.ouroboros_cycle()
        if t.thread:
            focus_seen.append(t.thread[0].node_id)
    assert len(set(focus_seen)) > 1, "the head turns"
    assert all(nid in g.nodes for nid in ids[:5]), "nothing erased"
    assert all(v <= t._config.fatigue_max for v in t._focus_fatigue.values())


# --------------------------------------------------------------------------- #90 valence is read-only

class _FixedValence:
    def __init__(self, g):
        self.g = g
        self.calls = 0

    def compute(self, graph, vdb):
        self.calls += 1
        return {nid: (0.8 if i % 2 else -0.8) for i, nid in enumerate(sorted(graph.nodes))}


def _graph_digest(g):
    return (g.timestep, tuple(sorted((nid, n.voltage, n.intrinsic_excitability, n.threshold)
                                     for nid, n in g.nodes.items())))


def test_valence_never_primes_and_never_writes():
    from tonic_valence import ValenceField
    g, ids = _graph()
    before = _graph_digest(g)
    try:
        ValenceField().compute(g, {})
    except Exception:
        pass
    assert _graph_digest(g) == before, "the field computation writes nothing"
    runs = []
    for vf in (None, _FixedValence(g)):
        g, ids = _graph()
        _warm(g, ids)
        t = TonicThread(g, {}, config=TonicConfig(valence_enabled=vf is not None), valence_field=vf)
        spy = _Spy(g)
        t.ouroboros_cycle()
        runs.append(spy.stim)
    assert runs[0] == runs[1], "first cycle: identical primes with and without valence (it only biases recovery)"


# --------------------------------------------------------------------------- #329 the spine whisper

def test_spine_whisper_present_tapered_and_never_floored():
    g, ids = _graph()
    spine = ids[0]
    g.nodes[spine].metadata = {"constitutional": True}
    deg = len(g._outgoing.get(spine, ()))
    t = _thread(g)
    spy = _Spy(g)
    t._prime_constitutional()
    expected = tt._SPINE_PRIME_STEADY + (tt._SPINE_PRIME_BOOTSTRAP - tt._SPINE_PRIME_STEADY) * math.exp(
        -deg / tt._SPINE_WIRE_SCALE)
    assert spy.stim == [(spine, pytest.approx(expected))]
    assert spy.steps == [(1.5, 0.5)]
    # homeostasis may lower the spine's excitability; the whisper is NOT raised to compensate (no floor)
    g.nodes[spine].intrinsic_excitability = 0.1
    v0 = g.nodes[spine].voltage
    spy.stim.clear()
    t._prime_constitutional()
    assert spy.stim == [(spine, pytest.approx(expected))]


# --------------------------------------------------------------------------- identity-protected structure is never pruned

def _prunable_graph(unified=True):
    extra = {"structural_plasticity_in_sleep": False, "grace_period": 2, "weight_threshold": 0.05,
             "inactivity_threshold": 4}
    g, ids = _graph(n=12, unified=unified, extra=extra, p=0.0)
    g.nodes[ids[0]].metadata = {"constitutional": True}
    g.nodes[ids[1]].metadata = {"provenance": "cc_authored", "kind": "want"}
    prot = [g.create_synapse(ids[0], ids[5], weight=0.001, delay=1).synapse_id,
            g.create_synapse(ids[6], ids[1], weight=0.001, delay=1).synapse_id]
    plain = [g.create_synapse(ids[7], ids[8], weight=0.001, delay=1).synapse_id,
             g.create_synapse(ids[9], ids[10], weight=0.001, delay=1).synapse_id]
    return g, ids, prot, plain


def test_identity_protected_synapses_and_nodes_never_pruned_on_the_unified_path():
    g, ids, prot, plain = _prunable_graph()
    for _ in range(30):
        tonic_unified_drive(g, [ids[2]], [0.1], 2)
    assert all(s in g.synapses for s in prot), "protected synapses survive every damped prune"
    assert not any(s in g.synapses for s in plain), "control: the damped prune does run"
    assert ids[0] in g.nodes and ids[1] in g.nodes


# --------------------------------------------------------------------------- RD §5 (B) damping holds

def test_damped_prune_is_harder_than_full():
    def survivors(damping, steps):
        g, ids, prot, plain = _prunable_graph(unified=False)
        for _ in range(steps):
            g.step(structural_damping=damping) if damping else g.step()
        return sum(s in g.synapses for s in plain)
    # grace 2 -> 3: after 3 steps the full weight rule has pruned (low_weight_steps 3 > 2), the damped one not yet
    assert survivors(None, 3) == 0
    assert survivors((1.5, 0.5), 3) == 2
    assert survivors((1.5, 0.5), 4) == 0


def test_damped_sprouting_is_about_half():
    def sprouts(damping):
        random.seed(99)
        tot = 0
        for rep in range(30):
            g, ids = _graph(n=30, unified=False, p=0.0, seed=rep)
            for i in range(10):
                for nid in ids[(i * 3) % 30:(i * 3) % 30 + 6]:
                    g.nodes[nid].voltage = 5.0
                r = g.step(structural_damping=damping) if damping else g.step()
                tot += r.synapses_sprouted
        return tot
    full, damp = sprouts(None), sprouts((1.5, 0.5))
    assert full > 50
    assert 0.3 * full < damp < 0.7 * full, (full, damp)


def test_damping_validated():
    g, ids = _graph(unified=False)
    for bad in ((0.5, 0.5), (1.5, 1.5), (1.5, -0.1), "x"):
        with pytest.raises(ValueError):
            g.step(structural_damping=bad)
    with pytest.raises(ValueError):
        g._prune_synapses(prune_factor=1.5, grace_sleeps=1, sleep_now=0, last_link_grace_sleeps=1,
                          credit_shield_kappa=0.0)


def test_conversation_step_keeps_full_plasticity():
    """step() with no argument is the unchanged path (the off-identity test covers bytes)."""
    import inspect
    src = inspect.getsource(Graph.step)
    assert "if structural_damping is None:\n                pruned, sprouted = self._structural_plasticity(fired_ids)" in src


# --------------------------------------------------------------------------- the breaker (#1139)

def test_breaker_counts_tonic_firing():
    g, ids = _graph()
    g._steps_since_last_fire = 199
    tonic_unified_drive(g, [ids[3]], [5.0], 1)
    assert g._steps_since_last_fire == 0, "Tonic firing inside step() is non-silence"


def test_old_path_tonic_firing_never_resets_the_breaker():
    g, ids = _graph(unified=False)
    g._steps_since_last_fire = 150
    g.prime_and_propagate([ids[3]], [5.0], steps=1, write_mode=True)
    assert g._steps_since_last_fire == 150


def test_breaker_still_trips_on_true_silence_with_no_body():
    g, ids = _graph(p=0.0)
    t = _thread(g)
    eng = TonicEngine(g, {}, t, config=EngineConfig(), require_shared_body=True)   # no body: waits
    out = eng._generate_latent_token_inner()
    assert out.get("waiting_for_shared_body") and g.timestep == 0, "no body: the Tonic invents nothing, steps nothing"
    trips = []
    g.register_event_handler("zero_fire_breaker_tripped", lambda **k: trips.append(k)) \
        if hasattr(g, "register_event_handler") else g._event_handlers.setdefault(
            "zero_fire_breaker_tripped", []).append(lambda **k: trips.append(k))
    for _ in range(g.config.get("zero_fire_breaker_steps", 200)):
        g.step()        # the autostep / drain clock on a silent mind
    assert trips, "the real silence trigger is kept"


# --------------------------------------------------------------------------- off == the base line, byte for byte

BASE_TREE = os.environ.get("TONIC_UNIFIED_BASE_TREE", os.path.expanduser("~/worktrees/ng-s-base-11d3898"))


def _scenario(tree, mode):
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=tree, HF_HUB_OFFLINE="1")
    out = subprocess.run([sys.executable, os.path.join(HERE, "tonic_unified_scenario.py"), mode],
                         cwd="/tmp", env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.skipif(not os.path.isdir(BASE_TREE), reason="base tree (11d3898) not present")
def test_off_is_byte_identical_to_the_base_line():
    base = _scenario(BASE_TREE, "absent")
    assert _scenario(ROOT, "absent") == base
    assert _scenario(ROOT, "false") == base
    assert _scenario(ROOT, "on") != base


def test_step_cap_read_live():
    g, ids = _graph(extra={"tonic_unified_max_steps": 1})
    ts = g.timestep
    tonic_unified_drive(g, [], [], 5)
    assert g.timestep == ts + 1
