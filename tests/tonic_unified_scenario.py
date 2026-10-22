"""A deterministic Tonic scenario, run as a script, that prints a digest of everything the Tonic touches.

Used by tests/test_tonic_unified.py to prove "switch off == the base line" across code trees: the test runs this
same file under PYTHONPATH=<base tree> and PYTHONPATH=<this tree> (PYTHONHASHSEED=0) and compares the digests.
It only imports public names that exist on both trees (Graph, TonicThread, TonicConfig, TonicEngine, EngineConfig).

argv[1]: "absent" (no switch key), "false" (tonic_unified_step=False), or "on" (tonic_unified_step=True).

# ---- Changelog ----
# [2026-10-21] Claude Code (Opus 5.5, laptop; lane tonic-unified) — initial
# What: builds a seeded 60-node graph (three-factor, sprout tally on, wake prune on, a constitutional node and an
#       authored want), a TonicThread (valence off) and a TonicEngine whose model inference is a fixed function of the
#       tick, runs 12 engine ticks (each: the engine sweep + ouroboros_cycle) + 6 conversation step()s, and prints a
#       sha256 over nodes, synapses, the delay buffer, the clock, predictions, the thread and the fatigue table.
# Why:  RD §2 build brief: "with the switch off, behaviour and bytes are identical to the current live line".
# How:  stdout = one JSON line {"digest": ..., "timestep": ..., "fired": ...}.
# -------------------
"""
import hashlib
import json
import math
import random
import sys

from neuro_foundation import Graph
from tonic_thread import TonicThread, TonicConfig
from tonic_engine import TonicEngine, EngineConfig


def build(mode):
    random.seed(1234)
    cfg = {
        "three_factor_enabled": True,
        "sprout_tally_enabled": True, "sprout_tally_slots": 8, "sprout_tally_theta": 2.0,
        "sprout_tally_horizon_steps": 50.0,
        "co_activation_window": 5,
    }
    if mode == "false":
        cfg["tonic_unified_step"] = False
    elif mode == "on":
        cfg["tonic_unified_step"] = True
    g = Graph(config=cfg)
    ids = [f"n{i:02d}" for i in range(60)]
    for nid in ids:
        md = {}
        if nid == "n00":
            md = {"constitutional": True}
        elif nid == "n01":
            md = {"provenance": "cc_authored", "kind": "want"}
        g.create_node(node_id=nid, metadata=md)
    for i, a in enumerate(ids):
        for b in random.sample(ids, 4):
            if a != b and g._find_synapse(a, b) is None:
                g.create_synapse(a, b, weight=random.uniform(0.05, 0.9), delay=random.randint(1, 4))
    return g, ids


def digest(g, thread):
    h = hashlib.sha256()
    for nid in sorted(g.nodes.keys()):
        n = g.nodes[nid]
        h.update(repr((nid, n.voltage, n.threshold, n.intrinsic_excitability, n.firing_rate_ema,
                       n.last_spike_time, n.refractory_remaining)).encode())
    rows = sorted((s.pre_node_id, s.post_node_id, s.weight, s.eligibility_trace, s.inactive_steps,
                   s.low_weight_steps, s.delay) for _sid, s in g.synapses.items())   # synapse ids are uuids
    h.update(repr(rows).encode())
    h.update(repr(sorted((k, sorted(v)) for k, v in g._delay_buffer.items())).encode())
    h.update(repr((g.timestep, g._steps_since_last_fire, g._total_predictions, g._total_pruned,
                   g._total_sprouted, len(g._active_predictions))).encode())
    h.update(repr([(it.node_id, it.activity) for it in thread.thread]).encode())
    h.update(repr(sorted(thread._focus_fatigue.items())).encode())
    h.update(repr(sorted((k, sorted(v)) for k, v in g._recent_spikes.items())).encode())
    return h.hexdigest()


def main():
    mode = sys.argv[1]
    g, ids = build(mode)
    thread = TonicThread(g, {}, config=TonicConfig(valence_enabled=False))
    eng = TonicEngine(g, {}, thread, config=EngineConfig(activation_strength=1.0), require_shared_body=True)
    thread.set_latent_engine(eng)
    eng._model = object()
    eng._shared_body = object()
    tick = {"i": 0}

    def fake_inference(features):
        i = tick["i"]
        tick["i"] += 1
        return [(ids[(i * 7 + k * 5) % len(ids)], 0.6 + 0.1 * k) for k in range(4)]

    eng._model_inference = fake_inference
    fired = 0
    for t in range(12):
        out = eng._generate_latent_token_inner()
        fired += out.get("fired", 0)
        if t % 2 == 1:
            g.stimulate(ids[(t * 3) % len(ids)], 1.2)
            g.step()
    print(json.dumps({"digest": digest(g, thread), "timestep": g.timestep, "fired": fired}))


if __name__ == "__main__":
    main()
