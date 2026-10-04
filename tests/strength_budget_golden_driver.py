# ---- Changelog ----
# [2026-10-04] Claude (lane strength-budget) — NEW golden driver: strength budget OFF must be byte-identical to the base
# What: pinned to ONE checkout, builds the want-hub seeded graph (want_hub_golden_driver.build_graph: a constitutional rim,
#   authored wants, plain nodes, every prune boundary), drives N real step() calls with a seeded stimulus schedule (so
#   STDP / homeostasis / structural plasticity / prune all run, and every plasticity rule's apply() is reached), then
#   checkpoints FULL and prints one JSON record (state digest, per-step results, checkpoint sha256).
# Why:  spec 2026-10-04 "off by default so Syl and every other NG are unchanged" — the proof is the same graph through the
#   base checkout and the branch checkout, compared on outputs AND checkpoint bytes (the want-hub lane's method).
# How:  separate process per checkout (sys.path pinned, P379-style void if neuro_foundation resolves elsewhere);
#   deterministic uuid4 counter, random/numpy seeded, PYTHONHASHSEED fixed by the caller. --config-json lets a test pass
#   explicit OFF settings (e.g. strength_budget_enabled False) to show they also change nothing but the saved config.
# -------------------
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.realpath(__file__))


def _run(checkout, ckpt_path, steps, config_json):
    checkout = os.path.realpath(checkout)
    sys.path.insert(0, checkout)
    sys.path.insert(1, HERE)
    random.seed(0)
    import numpy
    numpy.random.seed(0)
    import want_hub_golden_driver as drv
    drv.install_deterministic_uuid()
    import neuro_foundation as nf
    nf_file = os.path.realpath(nf.__file__)
    if os.path.dirname(nf_file) != checkout:
        print(json.dumps({"void": True, "reason": "neuro_foundation resolved to %s, not %s" % (nf_file, checkout)}))
        return 3
    extra = json.loads(config_json) if config_json else {}
    g, roles = drv.build_graph(nf, seed=31, extra_config=extra)
    rng = random.Random(7)
    targets = [roles["rim"]] + roles["wants"] + roles["plain"]
    per_step = []
    for _ in range(steps):
        for nid in rng.sample(targets, 6):
            g.stimulate(nid, 2.0)
        r = g.step()
        per_step.append([r.timestep, sorted(r.fired_node_ids), r.synapses_pruned, r.synapses_sprouted])
    g.checkpoint(ckpt_path)
    weights = sorted((sid, repr(s.weight)) for sid, s in g.synapses.items())
    rec = {
        "void": False,
        "neuro_foundation_file": nf_file,
        "git_rev": drv.git_rev(checkout),
        "rules": [type(r).__name__ for r in g._plasticity_rules],
        "steps": steps,
        "per_step": per_step,
        "synapses_after": len(g.synapses),
        "weights_sha": __import__("hashlib").sha256(json.dumps(weights).encode()).hexdigest(),
        "state_digest": drv.state_digest(g),
        "checkpoint_sha256": drv.sha256_file(ckpt_path),
    }
    print(json.dumps(rec, sort_keys=True))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--ckpt", required=True, help="scratch temp .msgpack path (never a live path)")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--config-json", default="")
    a = ap.parse_args()
    sys.exit(_run(a.checkout, a.ckpt, a.steps, a.config_json))
