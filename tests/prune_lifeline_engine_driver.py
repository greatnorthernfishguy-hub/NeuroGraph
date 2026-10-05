# ---- Changelog ----
# [2026-10-04] Claude (lane prune-lifeline) — NEW driver: the want-hub engine + a default prune, pinned to ONE checkout
# What: builds want_hub_golden_driver.build_graph (seed, leaves), runs compete_protected_links(topk, budget) and then one
#   default _prune_synapses() + _collect_orphan_nodes(), checkpoints FULL, prints one JSON record (engine record, removed ids,
#   state digest, checkpoint sha256).
# Why:  lane prune-lifeline — "flag off = byte-identical" must also hold for the engine path (compete_protected_links reads
#   the flag), not only for step().
# How:  separate process per checkout (sys.path pinned, void if neuro_foundation resolves elsewhere); deterministic uuid.
# -------------------
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.realpath(__file__))


def _run(checkout, ckpt_path, config_json, topk, budget):
    checkout = os.path.realpath(checkout)
    sys.path.insert(0, checkout)
    sys.path.insert(1, HERE)
    random.seed(0)
    import want_hub_golden_driver as drv
    drv.install_deterministic_uuid()
    import neuro_foundation as nf
    nf_file = os.path.realpath(nf.__file__)
    if os.path.dirname(nf_file) != checkout:
        print(json.dumps({"void": True, "reason": "neuro_foundation resolved to %s, not %s" % (nf_file, checkout)}))
        return 3
    g, _roles = drv.build_graph(nf, seed=31, extra_config=json.loads(config_json) if config_json else {}, n_leaf=12)
    before = set(g.synapses.keys())
    rec_engine = g.compete_protected_links(topk, budget)
    after_engine = set(g.synapses.keys())
    pruned = g._prune_synapses()
    g._collect_orphan_nodes()
    g.checkpoint(ckpt_path)
    print(json.dumps({
        "void": False, "neuro_foundation_file": nf_file, "git_rev": drv.git_rev(checkout),
        "engine": rec_engine, "engine_removed": sorted(before - after_engine), "default_pruned": pruned,
        "synapses_after": sorted(g.synapses.keys()), "nodes_after": sorted(g.nodes),
        "state_digest": drv.state_digest(g), "checkpoint_sha256": drv.sha256_file(ckpt_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--ckpt", required=True, help="scratch temp .msgpack path (never a live path)")
    ap.add_argument("--config-json", default="")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--budget", type=int, default=40)
    a = ap.parse_args()
    sys.exit(_run(a.checkout, a.ckpt, a.config_json, a.topk, a.budget))
