# [2026-10-05] Claude (lane rust-hotpaths-onto-s4) — before/after timings on a checkpoint COPY (not a test; run by hand)
"""Time the hot paths of ONE neuro_foundation (the trial tip from git = "base", or this branch)
on a COPY of a checkpoint. Run once per side, in separate processes (one graph in memory):

    python tests/rust_hotpaths_onto_s4_timings.py base   /tmp/copy/main.msgpack
    python tests/rust_hotpaths_onto_s4_timings.py branch /tmp/copy/main.msgpack

Each operation runs REPS times on the same restored graph, in the same order and with the same
random seed on both sides, so base and branch see the same state sequence. Prints JSON:
median / min ms per operation. Refuses the live plugin checkpoint directory; only reads the copy.
"""
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import uuid

LIVE_DIR = "/.claude/plugins/neurograph/checkpoints/"
BASE_REV = os.environ.get("NG_ONTO_S4_BASE_REV", "5246b63")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPS = int(os.environ.get("NG_ONTO_S4_REPS", "5"))


def load(side):
    sys.path.insert(0, REPO)
    if side == "branch":
        import neuro_foundation as mod
        return mod
    import importlib.util
    src = subprocess.run(["git", "-C", REPO, "show", f"{BASE_REV}:neuro_foundation.py"],
                         check=True, capture_output=True, text=True).stdout
    p = os.path.join(tempfile.mkdtemp(prefix="nf_base_t_"), "nf_base_s4.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nf_base_s4", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nf_base_s4"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    side, ckpt = sys.argv[1], sys.argv[2]
    assert side in ("base", "branch")
    assert LIVE_DIR not in os.path.abspath(ckpt), "refusing the live checkpoint"
    mod = load(side)
    g = mod.Graph()
    t0 = time.perf_counter()
    g.restore(ckpt)
    restore_ms = (time.perf_counter() - t0) * 1e3
    rng_u = random.Random(11)
    uuid.uuid4 = lambda: uuid.UUID(int=rng_u.getrandbits(128), version=4)
    random.seed(11)
    rng = random.Random(11)
    nodes = list(g.nodes)
    g.config["tonic_ages_substrate"] = 1
    out = {"side": side, "nodes": len(g.nodes), "synapses": len(g.synapses),
           "restore_ms": round(restore_ms, 1)}

    def timeit(name, fn):
        ts = []
        for _ in range(REPS):
            t = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t) * 1e3)
        out[name] = {"median_ms": round(statistics.median(ts), 1), "min_ms": round(min(ts), 1)}

    def do_step():
        for nid in rng.sample(nodes, 30):
            if nid in g.nodes:
                g.stimulate(nid, 2.0)
        g.step()

    def tonic_tick():
        ids = rng.sample(list(g.nodes), 12)
        g.prime_and_propagate(ids, [1.5] * 12, steps=3, write_mode=True)

    def recall():
        ids = rng.sample(list(g.nodes), 6)
        g.prime_and_propagate(ids, [1.0] * 6, steps=3, write_mode=False)

    live_cfg = {k: g.config.get(k) for k in ("prune_protected_faint_links", "strength_budget_enabled")}
    out["checkpoint_config"] = live_cfg
    g.config["prune_protected_faint_links"] = False
    timeit("prune_lifeline_off", lambda: g._prune_synapses())
    g.config["prune_protected_faint_links"] = True
    timeit("prune_lifeline_on", lambda: g._prune_synapses())
    g.config.update(live_cfg)   # the rest runs with the checkpoint's own config (live CC: both ON)
    timeit("inject_reward_global", lambda: g.inject_reward(0.05))
    timeit("step", do_step)
    timeit("tonic_tick_write_pp", tonic_tick)
    timeit("recall_read_pp", recall)
    timeit("get_telemetry", lambda: g.get_telemetry())
    print(json.dumps(out))


if __name__ == "__main__":
    main()
