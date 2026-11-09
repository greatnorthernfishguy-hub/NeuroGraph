# ---- Changelog ----
# [2026-10-21] Claude (lane rebaseline, #1136 / #1151; Josh 2026-10-21 "Let's go with your suggestion" + "Agreed?" on recall
#   feeling the S5 brakes) — recall_brakes_enabled: read-mode recall sweeps feel the live brakes, read-only
# What: (1) the key absent / explicitly False == the round-I commit (53a6d9f's module) exactly on the dict, NodeStore-only
#       and native paths, with the S5 + round-I candidate keys on (every op of the H2 drive workload incl. read and write
#       sweeps + a sleep; final checkpoint BYTES); not in DEFAULT_CONFIG. (2) read-only: twins with the key off / on, the
#       same ops; every step(), write sweep and the final checkpoint are identical, and each braked read sweep leaves
#       every node column, the slow calcium, g_inh, the I -> E weights, the depression resources and the synapse weights
#       bit-identical. (3) the native AHP + depression brakes == the round-I harness's Python method (threshold + gain x
#       Ca_slow, weight x resource, applied and restored around the sweep) entry by entry. (4) behaviour: an adapted
#       node and an inhibited node are held back in the sweep with the key on and fire with it off; depletion lowers
#       the delivered current; write mode never feels the brakes. (5) a graph without the recall-round stores logs
#       ERROR once and stays off.
# Why:  dev-log #1131 Parts 10-12; interneuron design Q5 / risk 10.
# How:  oracle via `git show 53a6d9f:neuro_foundation.py` (tests.test_hyperedge_store_h2._load); the S / S5 / I helpers.
# -------------------
"""#1136 / #1151 rebaseline: recall feels the brakes."""
import hashlib
import logging
import os
import random

import numpy as np
import pytest

from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s5 as S5          # noqa: E402
from tests import test_interneurons_i as I         # noqa: E402

NF = H2.NF
I_REV = os.environ.get("NG_RB_I_REV", "53a6d9f")
IMOD = H2._load(I_REV, "nf_i_commit_rb")
HAVE_RB = (I.HAVE_I and getattr(NF.ng_tract.NodeStore, "recall_round", 0) >= 1
           and getattr(NF.ng_tract.SynapseStore, "recall_round", 0) >= 1)
needs_rb = pytest.mark.skipif(not HAVE_RB, reason="ng_tract without the recall-brake kernels")
KEY = "recall_brakes_enabled"
CAND = dict(S5.S5_FULL, **I.I2, interneuron_decay=0.85, inh_g_decay=0.85, interneuron_coincidence=1)
ckpt = S.ckpt
restore_as = S.restore_as


def deep(g):
    """every piece of state a read sweep could touch, as bytes"""
    h = hashlib.sha256()
    cols = ["voltage", "threshold", "intrinsic_excitability", "firing_rate_ema", "refractory_remaining", "Ca_i"]
    try:
        _, *vals = g.nodes.columns(cols, with_ids=True)
    except Exception:
        cols = cols[:5]
        _, *vals = g.nodes.columns(cols, with_ids=True)
    for v in vals:
        h.update(np.asarray(v, dtype=float).tobytes())
    on, ca, agb = g.nodes.s5_state()
    h.update(repr((on, agb)).encode())
    if ca is not None:
        h.update(np.asarray(ca, dtype=float).tobytes())
    if g.nodes.interneurons_exists():
        gg, w, fires, _ = g.nodes.interneuron_arrays()
        for v in (gg, w, fires):
            h.update(np.asarray(v, dtype=float).tobytes())
        h.update(repr(sorted(g.nodes.interneuron_census(0.01).items())).encode())
        vm = g.nodes.v_mark_column()
        if vm is not None:
            h.update(np.asarray(vm, dtype=float).tobytes())
    h.update(np.asarray(g.synapses.std_resources(float(g.timestep)), dtype=float).tobytes())
    h.update(np.asarray(g.synapses.weights_copy(), dtype=float).tobytes())
    return h.hexdigest()


def _sweep(g, rng_seed, write=False):
    rng = random.Random(rng_seed)
    ids = rng.sample(list(g.nodes), min(10, len(g.nodes)))
    r = g.prime_and_propagate(ids, [rng.uniform(0.5, 3.0) for _ in ids], steps=3, write_mode=write)
    return [(e.node_id, e.firing_step, float(e.voltage_at_fire).hex(), e.source_distance) for e in r.fired_entries]


def _cand(seed, steps=30):
    g, _ = I._grown(seed, CAND)
    rng = random.Random(seed)
    for _ in range(steps):
        for nid in rng.sample(list(g.nodes), 12):
            g.stimulate(nid, rng.uniform(0.5, 3.0))
        g.step()
    return g


# ---------------------------------------------------------------------------
# (1) identity when off
# ---------------------------------------------------------------------------

@needs_rb
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["dict", "nodes", "native"])
@pytest.mark.parametrize("how", ["absent", "explicit_false"])
def test_off_is_the_round_i_commit_exactly(kind, seed, how):
    start = H2.built(seed)
    t = S.Pair(restore_as(IMOD, kind, start), restore_as(NF, kind, start), seed)
    if kind == "native":
        t.op("keys", lambda g: g.config.update(CAND))
    if how == "explicit_false":
        t.b.g.config[KEY] = False
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    assert t.ops > 60
    t.b.g.config.pop(KEY, None)
    assert ckpt(t.a.g) == ckpt(t.b.g)


@needs_rb
def test_key_not_in_default_config():
    assert KEY not in NF.DEFAULT_CONFIG and KEY not in NF.Graph().config
    assert list(NF.DEFAULT_CONFIG.items()) == list(IMOD.DEFAULT_CONFIG.items())


# ---------------------------------------------------------------------------
# (2) read-only: on == off for the mind; each braked sweep restores everything bit-exactly
# ---------------------------------------------------------------------------

@needs_rb
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_braked_sweeps_are_read_only_and_the_mind_steps_identically(seed):
    start = ckpt(_cand(seed, steps=5))
    a, b = H2.Side(restore_as(NF, "native", start), seed), H2.Side(restore_as(NF, "native", start), seed)
    rng = random.Random(seed)

    def step(g, cue):
        for nid, c in cue:
            g.stimulate(nid, c)
        return list(g.step().fired_node_ids)

    for _ in range(25):                                     # rebuild the transient brakes (g_inh, resources, calcium)
        cue = [(nid, rng.uniform(0.5, 3.0)) for nid in rng.sample(list(a.g.nodes), 12)]
        assert a(lambda g: step(g, cue)) == b(lambda g: step(g, cue))
    b.g.config[KEY] = True
    assert b.g._recall_brakes(b.g._stability_params()) is not None
    differ = 0
    for k in range(20):
        da, db = deep(a.g), deep(b.g)
        assert da == db
        sa, sb = a(lambda g: _sweep(g, 1000 * seed + k)), b(lambda g: _sweep(g, 1000 * seed + k))
        assert deep(a.g) == da and deep(b.g) == db, "a read sweep changed the mind"
        differ += sa != sb
        if k % 5 == 4:                                       # write mode never feels them
            assert a(lambda g: _sweep(g, 7 + k, write=True)) == b(lambda g: _sweep(g, 7 + k, write=True))
        cue = [(nid, rng.uniform(0.5, 3.0)) for nid in rng.sample(list(a.g.nodes), 12)]
        assert a(lambda g: step(g, cue)) == b(lambda g: step(g, cue))
    b.g.config.pop(KEY)
    assert ckpt(a.g) == ckpt(b.g)
    assert differ > 0, "the brakes never changed a sweep on this graph"


# ---------------------------------------------------------------------------
# (3) native AHP + depression == the round-I harness's Python method
# ---------------------------------------------------------------------------

def _harness_braked(g, fn):
    """i1/scripts/i_settle.py _brakes_on / _brakes_off, verbatim in effect"""
    st = g._stability_params()
    undo_t, undo_w = [], []
    on, col, _ = g.nodes.s5_state()
    gain = float(st.ca_split[3])
    for nid, ca in zip(g.nodes.keys(), col):
        if ca > 0.0 and gain > 0.0:
            nd = g.nodes[nid]
            undo_t.append((nid, nd.threshold))
            nd.threshold = nd.threshold + gain * ca
    xr = np.asarray(g.synapses.std_resources(float(g.timestep)))
    for sid, x in zip(g.synapses.keys(), xr):
        if x < 1.0:
            w = g.synapses.get_weight(sid)
            undo_w.append((sid, w))
            g.synapses.set_weight(sid, w * float(x))
    try:
        return fn()
    finally:
        for nid, th in undo_t:
            g.nodes[nid].threshold = th
        for sid, w in undo_w:
            g.synapses.set_weight(sid, w)


@needs_rb
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_native_ahp_and_depression_equal_the_harness_method(seed):
    cfg = dict(CAND)
    for k in I.I_SWITCHES:
        cfg.pop(k, None)
    a = restore_as(NF, "native", H2.built(seed))
    a.config.update(cfg)
    rng = random.Random(seed)
    for _ in range(30):
        for nid in rng.sample(list(a.nodes), 12):
            a.stimulate(nid, rng.uniform(0.5, 3.0))
        a.step()
    assert not a.nodes.interneurons_exists()
    for k in range(10):
        want = _harness_braked(a, lambda: _sweep(a, k))
        a.config[KEY] = True
        got = _sweep(a, k)
        a.config.pop(KEY)
        # the harness rounds weight x resource into the weight column and back; the kernel multiplies in place —
        # the same product, so the fired entries (ids, steps, voltages) agree exactly
        assert got == want


# ---------------------------------------------------------------------------
# (4) behaviour
# ---------------------------------------------------------------------------

def _chain(cfg):
    g = NF.Graph(dict(cfg), native_node_store=True, native_hyperedge_store=True)
    for n in ("a", "b"):
        g.create_node(n)
    g.create_synapse("a", "b", weight=2.0)
    return g


@needs_rb
def test_an_adapted_node_is_held_back_in_the_sweep_only_with_the_key():
    g = _chain(dict(S5.S5_FULL))
    st = g._stability_params()
    assert st is not None and st.ca_split is not None
    for _ in range(40):                       # b fires repeatedly -> its slow calcium builds up
        g.stimulate("a", 5.0)
        g.stimulate("b", 5.0)
        g.step()
    ca = dict(zip(g.nodes.keys(), g.nodes.s5_state()[1]))
    gain = float(st.ca_split[3])
    assert gain * ca["b"] > 0.01
    nd = g.nodes["b"]
    nd.voltage, nd.refractory_remaining = nd.resting_potential, 0      # a clean start for the probe
    d = g.config["decay_rate"]
    # crosses the threshold, not the AHP-raised one
    cur = (nd.threshold + 0.5 * gain * ca["b"]) / (d * nd.intrinsic_excitability)
    off = {e.node_id for e in g.prime_and_propagate(["b"], [cur], steps=1).fired_entries}
    g.config[KEY] = True
    on = {e.node_id for e in g.prime_and_propagate(["b"], [cur], steps=1).fired_entries}
    assert "b" in off and "b" not in on


@needs_rb
def test_depletion_lowers_the_delivered_current_read_only():
    cfg = {k: v for k, v in S5.S5_FULL.items()}
    g = _chain(cfg)
    for _ in range(10):
        g.stimulate("a", 5.0)
        g.step()
    sid = next(iter(g._outgoing["a"]))
    x = float(np.asarray(g.synapses.std_resources(float(g.timestep)))[list(g.synapses.keys()).index(sid)])
    assert x < 1.0
    w = g.synapses.get_weight(sid)
    rows = g.synapses.propagation_rows([sid], False, std_read=float(g.timestep))
    assert rows[0][1] == w * x
    assert float(np.asarray(g.synapses.std_resources(float(g.timestep)))[list(g.synapses.keys()).index(sid)]) == x
    with pytest.raises(ValueError):
        g.synapses.propagation_rows([sid], False, std=(0.2, 50.0, 1.0), std_read=1.0)


@needs_rb
@pytest.mark.parametrize("seed", [0, 1])
def test_an_inhibited_node_is_held_back_in_the_sweep(seed):
    g = _cand(seed, steps=40)
    gg = dict(zip(g.nodes.keys(), g.nodes.interneuron_arrays()[0]))
    held = [n for n, x in gg.items() if x > 0.5]
    if not held:
        pytest.skip("no strongly inhibited node on this seed")
    n = held[0]
    nd = g.nodes[n]
    nd.voltage, nd.refractory_remaining = nd.resting_potential, 0      # a clean start for the probe
    g.config["ca_split_ahp_gain"] = 0.0          # isolate the inhibition
    d = g.config["decay_rate"]
    # the prime's drive crosses the threshold undivided but not once divided by (1 + g_inh)
    cur = nd.threshold * (1.0 + 0.5 * gg[n]) / (d * nd.intrinsic_excitability)
    off = {e.node_id for e in g.prime_and_propagate([n], [cur], steps=1).fired_entries}
    g.config[KEY] = True
    on = {e.node_id for e in g.prime_and_propagate([n], [cur], steps=1).fired_entries}
    assert n in off and n not in on
    g.config["interneuron_form"] = "offset"      # the ablation form: threshold + g_inh in the sweep too
    assert n not in {e.node_id for e in g.prime_and_propagate([n], [cur], steps=1).fired_entries}


# ---------------------------------------------------------------------------
# (5) misconfiguration
# ---------------------------------------------------------------------------

@needs_rb
def test_dict_graph_logs_error_once_and_stays_off(caplog):
    start = H2.built(0)
    t = S.Pair(restore_as(IMOD, "dict", start), restore_as(NF, "dict", start), 0)
    t.b.g.config.update(dict(S5.S5_FULL, **{KEY: True}))
    t.a.g.config.update(S5.S5_FULL)
    with caplog.at_level(logging.ERROR):
        S.drive(t, random.Random(0), 8, sleep_every=True)
    errs = [r for r in caplog.records if KEY in r.getMessage() and r.levelno == logging.ERROR]
    assert len(errs) == 1, [r.getMessage() for r in errs]
    t.b.g.config.pop(KEY)
    assert ckpt(t.a.g) == ckpt(t.b.g)
