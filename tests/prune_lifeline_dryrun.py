# ---- Changelog ----
# [2026-10-04] Claude (lane prune-lifeline) — NEW dry-run tool: what normal pruning would take from protected nodes
# What: loads a COPY of a checkpoint, applies the laptop's CC_SNN_CONFIG prune settings (weight_threshold, grace_period,
#   inactivity_threshold, initial_sprouting_weight — passed on the command line), runs ONE apply_strength_budget(out, in)
#   pass, then classifies every synapse touching a protected node under the lifeline rule WITHOUT mutating anything:
#   lifeline / eligible NOW (by rule: weight-with-dwell-met, activity, age) / eligible ONCE the weight rule's dwell elapses
#   (weight < threshold, weights held static) / kept. Then runs ONE real _prune_synapses() with the flag ON and checks the
#   prediction, per protected node before/after degree, the lifeline guarantee, and unprotected partners left with zero
#   synapses. A flag-OFF control prune on a fresh load gives the baseline. Prints JSON; feeds PRUNE_LIFELINE_DRYRUN.md.
# Why:  lane prune-lifeline step 6 (Josh ruling 2026-10-04).
# How:  refuses any path under ~/.claude/plugins/ or ~/NeuroGraph/data (copy first); never checkpoints anything.
# -------------------
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import neuro_foundation as nf  # noqa: E402

if os.path.dirname(os.path.realpath(nf.__file__)) != REPO:
    raise SystemExit("neuro_foundation resolved to %s, not %s" % (nf.__file__, REPO))

FLAG = "prune_protected_faint_links"


def load(path, cfg):
    g = nf.Graph()
    g.restore(path)
    g.config.update(cfg)
    return g


def degrees(g, nids):
    return {n: [len(g._outgoing.get(n) or ()), len(g._incoming.get(n) or ())] for n in nids}


def classify(g):
    """Mirror of _prune_synapses' default-path predicates for the flag-ON rule; mutates nothing."""
    wt, grace = g.config["weight_threshold"], g.config["grace_period"]
    inact, init_w = g.config["inactivity_threshold"], g.config["initial_sprouting_weight"]
    prot = set(g._strength_protected_ids())
    life = g._protected_lifelines()
    out = {"touching_protected": 0, "lifeline": 0, "now_weight": 0, "now_activity": 0, "now_age": 0,
           "dwell_weight": 0, "kept": 0, "unprotected_now": 0, "unprotected_dwell": 0}
    now_ids, dwell_ids = set(), set()
    lws_hist = {}
    for sid, s in g.synapses.items():
        touches = s.pre_node_id in prot or s.post_node_id in prot
        if touches:
            out["touching_protected"] += 1
            if sid in life:
                out["lifeline"] += 1
                continue
        age = g.timestep - s.creation_time
        low = s.weight < wt
        if low and s.low_weight_steps + 1 > grace:
            kind = "now_weight"
        elif s.inactive_steps > inact * s.salience:
            kind = "now_activity"
        elif age > grace and s.peak_weight < 2.0 * init_w:
            kind = "now_age"
        elif low:
            kind = "dwell_weight"
        else:
            kind = "kept"
        if touches:
            out[kind] += 1
            if low and kind == "dwell_weight":
                b = s.low_weight_steps
                b = "0" if b == 0 else "1-99" if b < 100 else "100-999" if b < 1000 else "1000-4999" if b < 5000 else ">=5000"
                lws_hist[b] = lws_hist.get(b, 0) + 1
        else:
            if kind.startswith("now"):
                out["unprotected_now"] += 1
            elif kind == "dwell_weight":
                out["unprotected_dwell"] += 1
        (now_ids if kind.startswith("now") else dwell_ids if kind == "dwell_weight" else set()).add(sid)
    out["dwell_low_weight_steps_hist"] = lws_hist
    return out, now_ids, dwell_ids, prot, life


def projected_degrees(g, nids, gone):
    res = {}
    for n in nids:
        o = sum(1 for s in (g._outgoing.get(n) or ()) if s not in gone)
        i = sum(1 for s in (g._incoming.get(n) or ()) if s not in gone)
        res[n] = [o, i]
    return res


def stranded_partners(g, prot, gone):
    """Unprotected nodes that have >=1 synapse to a protected node now, and would have ZERO synapses if `gone` left."""
    cand = set()
    for p in prot:
        for idx, end in ((g._outgoing, "post_node_id"), (g._incoming, "pre_node_id")):
            for sid in idx.get(p) or ():
                cand.add(getattr(g.synapses[sid], end))
    cand -= prot
    n = 0
    hyper = 0
    for c in cand:
        inc = set(g._outgoing.get(c) or ()) | set(g._incoming.get(c) or ())
        if inc and inc <= gone:
            n += 1
            if g._node_hyperedges.get(c):
                hyper += 1
    return {"partners": len(cand), "left_with_zero_synapses": n, "of_which_in_a_hyperedge": hyper}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a COPY of a checkpoint (never the live file)")
    ap.add_argument("--out-budget", type=float, default=10.0)
    ap.add_argument("--in-budget", type=float, default=5.4)
    ap.add_argument("--cfg-json", default=json.dumps({"weight_threshold": 0.01, "grace_period": 5000,
                                                       "inactivity_threshold": 1000, "initial_sprouting_weight": 0.1}))
    a = ap.parse_args()
    rp = os.path.realpath(a.ckpt)
    for bad in (os.path.expanduser("~/.claude/plugins/"), os.path.expanduser("~/NeuroGraph/data")):
        if rp.startswith(os.path.realpath(bad)):
            raise SystemExit("refusing a live path: %s (copy it first)" % rp)
    cfg = json.loads(a.cfg_json)
    print(json.dumps({"ng_tract": nf.ng_tract.__file__, "native_normalize": hasattr(nf.ng_tract.SynapseStore, "normalize_strength"),
                      "cfg": cfg}), flush=True)

    # control: flag OFF, same budget pass, one prune
    g = load(rp, dict(cfg, **{FLAG: False}))
    ckpt_cfg = {k: g.config.get(k) for k in ("weight_threshold", "grace_period", "inactivity_threshold", "initial_sprouting_weight")}
    n0 = len(g.synapses)
    g.apply_strength_budget(a.out_budget, a.in_budget)
    t0 = time.perf_counter()
    pruned_off = g._prune_synapses()
    print(json.dumps({"control_flag_off": {"synapses": n0, "pruned": pruned_off,
                                            "prune_seconds": round(time.perf_counter() - t0, 3)}}), flush=True)
    del g

    g = load(rp, dict(cfg, **{FLAG: True}))
    prot0 = g._strength_protected_ids()
    meta = {n: {"constitutional": bool((g.nodes[n].metadata or {}).get("constitutional")),
                "provenance": (g.nodes[n].metadata or {}).get("provenance"),
                "label": str((g.nodes[n].metadata or {}).get("label") or (g.nodes[n].metadata or {}).get("text") or "")[:60]}
            for n in prot0}
    deg_before = degrees(g, prot0)
    n_before = len(g.synapses)
    budget = g.apply_strength_budget(a.out_budget, a.in_budget)
    t0 = time.perf_counter()
    cls, now_ids, dwell_ids, prot, life = classify(g)
    t_cls = time.perf_counter() - t0
    proj_now = projected_degrees(g, prot0, now_ids)
    proj_dwell = projected_degrees(g, prot0, now_ids | dwell_ids)
    strand_now = stranded_partners(g, prot, now_ids)
    strand_dwell = stranded_partners(g, prot, now_ids | dwell_ids)
    t0 = time.perf_counter()
    life_seconds = None
    t1 = time.perf_counter()
    g._protected_lifelines()
    life_seconds = round(time.perf_counter() - t1, 4)
    t0 = time.perf_counter()
    pruned_on = g._prune_synapses()
    t_prune = time.perf_counter() - t0
    deg_after = degrees(g, prot0)
    missing_life = sorted(s for s in life if s not in g.synapses)
    lifeline_violations = []
    for n in prot0:
        for d in (0, 1):
            if deg_before[n][d] > 0 and deg_after[n][d] == 0:
                lifeline_violations.append([n, "out" if d == 0 else "in"])
    orphans_before = sum(1 for n in g.nodes if not g._outgoing.get(n) and not g._incoming.get(n))
    rec = {
        "checkpoint_config_before_override": ckpt_cfg,
        "timestep": g.timestep, "synapses_before": n_before, "budget_pass": budget,
        "classification_flag_on": cls, "classify_seconds": round(t_cls, 3),
        "lifelines": len(life), "lifeline_query_seconds": life_seconds,
        "predicted_now_total": len(now_ids), "predicted_dwell_total": len(dwell_ids),
        "pruned_flag_on": pruned_on, "prune_seconds": round(t_prune, 3),
        "prediction_matches": pruned_on == len(now_ids),
        "lifelines_missing_after_prune": missing_life, "lifeline_violations": lifeline_violations,
        "stranded_partners_now": strand_now, "stranded_partners_after_dwell": strand_dwell,
        "zero_degree_nodes_after_prune_before_orphan_sweep": orphans_before,
        "protected": {n: {"meta": meta[n], "before": deg_before[n], "after_prune_now": deg_after[n],
                          "projected_now": proj_now[n], "projected_after_dwell": proj_dwell[n]} for n in prot0},
    }
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
