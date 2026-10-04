# ---- Changelog ----
# [2026-10-04] Claude (lane strength-budget) — NEW dry-run tool for the strength budget / sleep downscaling
# What: loads a COPY of a checkpoint into a Graph, and for each candidate (budget_out, budget_in) runs ONE
#   apply_strength_budget pass + ONE existing _prune_synapses pass, then (separately, from a fresh load) ONE
#   sleep_downscale(factor) + prune pass; prints a JSON record per scenario (synapse counts, prunes vs a
#   no-budget control prune, below-threshold counts, top-10 out-degree/out-strength nodes, protected-node
#   strongest-link check). Feeds STRENGTH_BUDGET_DRYRUN.md.
# Why:  spec 2026-10-04 — "Final values chosen from a dry run, not guessed".
# How:  refuses any path under ~/.claude/plugins/ or ~/NeuroGraph/data (copy first); never checkpoints anything.
# -------------------
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import neuro_foundation as nf  # noqa: E402


def load(path):
    g = nf.Graph()
    g.restore(path)
    return g


def sums(g):
    out = {n: 0.0 for n in g.nodes}
    inc = {n: 0.0 for n in g.nodes}
    outdeg = {n: 0 for n in g.nodes}
    for _sid, s in g.synapses.items():
        out[s.pre_node_id] = out.get(s.pre_node_id, 0.0) + s.weight
        inc[s.post_node_id] = inc.get(s.post_node_id, 0.0) + s.weight
        outdeg[s.pre_node_id] = outdeg.get(s.pre_node_id, 0) + 1
    return out, inc, outdeg


def pct(vals, q):
    v = sorted(vals)
    return v[min(len(v) - 1, int(q * len(v)))] if v else 0.0


def strongest(g, nid, d):
    idx = g._outgoing if d == "out" else g._incoming
    ids = idx.get(nid) or ()
    if not ids:
        return None, None
    sid = min(ids, key=lambda x: (-g.synapses[x].weight, x))
    return sid, g.synapses[sid].weight


def snapshot(g):
    out, inc, outdeg = sums(g)
    wt = g.config["weight_threshold"]
    prot = set(g._strength_protected_ids())
    below = below_unprot = 0
    total = 0.0
    for _sid, s in g.synapses.items():
        total += s.weight
        if s.weight < wt:
            below += 1
            if s.pre_node_id not in prot and s.post_node_id not in prot:
                below_unprot += 1
    top_deg = sorted(g.nodes, key=lambda n: (-outdeg.get(n, 0), n))[:10]
    top_str = sorted(g.nodes, key=lambda n: (-out.get(n, 0.0), n))[:10]
    return {
        "synapses": len(g.synapses), "total_weight": total, "below_threshold": below,
        "below_threshold_unprotected": below_unprot,
        "out_sum_p50_p99_max": [pct(out.values(), .5), pct(out.values(), .99), max(out.values(), default=0)],
        "in_sum_p50_p99_max": [pct(inc.values(), .5), pct(inc.values(), .99), max(inc.values(), default=0)],
        "top_outdeg": [[n, outdeg.get(n, 0), round(out.get(n, 0.0), 4)] for n in top_deg],
        "top_outstr": [[n, outdeg.get(n, 0), round(out.get(n, 0.0), 4)] for n in top_str],
        "_out": out, "_outdeg": outdeg,
    }


def prot_check(g, before):
    """before: {(nid, d): (sid, w)} — every protected node's strongest link must end >= min(w, floor)."""
    floor = 2 * g.config["weight_threshold"]
    bad = []
    for (nid, d), (_sid, w) in before.items():
        _s2, w2 = strongest(g, nid, d)
        if w2 is None or w2 < min(w, floor):
            bad.append([nid, d, w, w2])
    return bad


def scenario(path, label, fn):
    g = load(path)
    prot = g._strength_protected_ids()
    pre = snapshot(g)
    before = {}
    for nid in prot:
        for d in ("out", "in"):
            sid, w = strongest(g, nid, d)
            if sid is not None:
                before[(nid, d)] = (sid, w)
    t0 = time.perf_counter()
    res = fn(g)
    t_pass = time.perf_counter() - t0
    mid = snapshot(g)
    t0 = time.perf_counter()
    pruned = g._prune_synapses()
    t_prune = time.perf_counter() - t0
    post = snapshot(g)
    watch = {}
    for n in ("constitutional::rim::choice_clause", "cc:want::4625485108d2e9be"):
        if n in g.nodes:
            watch[n] = {"before": [pre["_outdeg"].get(n, 0), round(pre["_out"].get(n, 0.0), 4)],
                        "after": [post["_outdeg"].get(n, 0), round(post["_out"].get(n, 0.0), 4)],
                        "strongest_out_after": strongest(g, n, "out")[1], "strongest_in_after": strongest(g, n, "in")[1]}
    for d in (pre, mid, post):
        d.pop("_out"), d.pop("_outdeg")
    return {"label": label, "result": res, "pass_seconds": round(t_pass, 3), "prune_seconds": round(t_prune, 3),
            "pruned": pruned, "before": pre, "after_pass": mid, "after_prune": post, "watch": watch,
            "protected_nodes": len(prot), "protected_links_checked": len(before),
            "protected_violations": prot_check(g, before)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a COPY of a checkpoint (never the live file)")
    ap.add_argument("--out", nargs="+", default=["5", "10", "20"])
    ap.add_argument("--in", dest="inb", default="5.4")
    ap.add_argument("--sleep", type=float, default=0.98)
    a = ap.parse_args()
    rp = os.path.realpath(a.ckpt)
    for bad in (os.path.expanduser("~/.claude/plugins/"), os.path.expanduser("~/NeuroGraph/data")):
        if rp.startswith(os.path.realpath(bad)):
            raise SystemExit("refusing a live path: %s (copy it first)" % rp)
    print(json.dumps({"ng_tract": nf.ng_tract.__file__,
                      "native": hasattr(nf.ng_tract.SynapseStore, "normalize_strength")}), flush=True)
    print(json.dumps(scenario(rp, "control: prune only", lambda g: None)), flush=True)
    inb = None if a.inb.lower() == "none" else float(a.inb)
    for o in a.out:
        ob = None if o.lower() == "none" else float(o)
        print(json.dumps(scenario(rp, "budget out=%s in=%s" % (ob, inb),
                                  lambda g, ob=ob: g.apply_strength_budget(ob, inb))), flush=True)
    print(json.dumps(scenario(rp, "sleep_downscale(%s)" % a.sleep, lambda g: g.sleep_downscale(a.sleep))), flush=True)


if __name__ == "__main__":
    main()
