# ---- Changelog ----
# [2026-09-30] Claude Sonnet 5.5 (Z12 builder, dispatch #12011, ADDENDUM 3 tests fold) — driver additions for le-036 / checker-029 C1, C5, C6, C2, C3:
#   (C1) --ids random: engine-minted synapse ids come from a SEEDED random uuid4-shaped stream, so id order != creation order != items() order
#        (the counter ids hid an unconditional default-path sort-by-id, mutant M07a); (C6) every record carries ng_tract_file / ng_tract_version
#        (the native SynapseStore is an installed wheel, not a file of either checkout); (C5) ref_order_key counts a link once per distinct
#        endpoint (a self-loop is ONE link) and inject_self_loop() builds a restored-style self-loop (create_synapse refuses them);
#        (C2) bad_calls gains order_key entries that are not mutually comparable / not tuples. No existing scenario or comparison changed.
# [2026-09-30] Claude Sonnet 5.5 (Z12 worker, lane want-hub-competition-d, dispatch #11135) — want-hub (d) test driver
# What: NEW helper for tests/test_want_hub_competition.py — NOT a test module (no test_ prefix, no pytest import).
#   (1) build_graph(): the ONE seeded synthetic-graph builder shared by test G (run as a script, once per checkout)
#   and by tests A/K/R (imported in-process); (2) state_digest(): the full post-state hash test G compares;
#   (3) a script entry point that, pinned to ONE checkout, builds the graph, drives a prune scenario, and prints a JSON record.
# Why: plan-005 sec 2.6 test G — the SAME seeded graph through base e4ebf982 AND the branch as TWO SEPARATE CHECKOUTS in two
#   SEPARATE processes (two checkouts' neuro_foundation in one interpreter would return the first import from sys.modules
#   and pass vacuously — P379). Synthetic graphs ONLY: no checkpoint, no vectors, no Syl file, no tract, no embed/TID call.
# How: sys.path[0] is pinned to --checkout BEFORE neuro_foundation is imported; the run is VOID (exit 3) if the imported
#   module is not inside that checkout. uuid.uuid4 is replaced by a counter so synapse ids are identical across processes
#   (the engine mints them with uuid4; without this no two runs could ever be compared). PYTHONHASHSEED is pinned by the
#   caller (set iteration order, e.g. list(he.member_nodes) in the serializer, depends on it).
# -------------------
import argparse
import collections
import hashlib
import json
import os
import random
import subprocess
import sys
import uuid

TIMESTEP = 10_000


def install_deterministic_uuid():
    """Replace uuid.uuid4 with a counter so engine-minted ids are reproducible. Returns the counter box."""
    box = {"n": 0}

    def _det_uuid4():
        box["n"] += 1
        return uuid.UUID(int=box["n"])

    uuid.uuid4 = _det_uuid4
    return box


def install_seeded_random_uuid(seed=20260930):
    """C1: engine-minted ids become SEEDED random uuid4-shaped values (deterministic across processes, but NOT ascending in
    creation order — like the uuid4 ids of a real graph). Returns the Random so a caller can see it is stateful."""
    rng = random.Random(seed)
    uuid.uuid4 = lambda: uuid.UUID(int=rng.getrandbits(128), version=4)
    return rng


def ng_tract_info():
    """C6: where the native SynapseStore comes from and its installed version (an installed wheel, NOT a checkout file)."""
    import importlib.metadata
    import ng_tract
    try:
        ver = importlib.metadata.version("ng_tract")
    except Exception as exc:  # recorded, never hidden
        ver = "UNKNOWN(%s)" % type(exc).__name__
    return os.path.realpath(ng_tract.__file__), ver


# ---------------------------------------------------------------------------
# The seeded synthetic graph (same builder for every checkout)
# ---------------------------------------------------------------------------
_WEIGHTS = [0.0, 0.005, 0.0099, 0.01, 0.0100001, 0.05, 0.3, 0.9]
_INACTIVE = [0, 10, 999, 1000, 1001, 1500, 2000, 2001, 5000]
_SALIENCE = [1.0, 1.0, 1.0, 2.0, 1.5]
_PEAK = [0.05, 0.1, 0.19, 0.2, 0.21, 0.5]
_AGE = [0, 100, 5000, 5001, 9000]
_LWS = [0, 0, 1, 4999, 5000, 5001, 10]


def build_graph(nf, seed, n_want=6, n_plain=40, n_syn=600, tonic_ages=False, extra_config=None, n_leaf=0):
    """Seeded graph: one constitutional rim node, n_want authored wants, n_plain ordinary nodes.

    Returns (graph, roles) with roles = {'rim': id, 'wants': [ids], 'plain': [ids]}. Synapse kinds present:
    rim<->want, rim<->plain, want->plain, plain->want, want<->want, plain<->plain. Field values are drawn from menus
    that include every predicate boundary in _prune_synapses (weight == weight_threshold, low_weight_steps around
    grace, inactive_steps == inactivity*salience, age == grace, peak_weight == 2*initial, salience != 1).
    Caller must have installed the deterministic uuid first.
    """
    rng = random.Random(seed)
    cfg = dict(extra_config or {})
    if tonic_ages:
        cfg["tonic_ages_substrate"] = 1
    g = nf.Graph(cfg)
    rim = "constitutional::rim::choice_clause"
    g.create_node(node_id=rim, metadata={"constitutional": True})
    wants = []
    for i in range(n_want):
        prov = "syl_authored" if i % 2 == 0 else "cc_authored"
        wid = "want::%d" % i
        g.create_node(node_id=wid, metadata={"provenance": prov})
        wants.append(wid)
    plain = []
    for i in range(n_plain):
        pid = "n%d" % i
        meta = {"provenance": "syl_emergent"} if i % 7 == 0 else {}
        g.create_node(node_id=pid, metadata=meta)
        plain.append(pid)
    g.timestep = TIMESTEP  # nodes were created at 0: every node is far past orphan_node_grace_period

    used = set()
    pairs = []

    def _add(a, b):
        if a != b and (a, b) not in used:
            used.add((a, b))
            pairs.append((a, b))

    for w in wants:
        _add(rim, w)
        _add(w, rim)
    for p in plain[:8]:
        _add(rim, p)
        _add(p, rim)
    for a in wants:
        for b in wants:
            if rng.random() < 0.4:
                _add(a, b)
    while len(pairs) < n_syn:
        kind = rng.random()
        if kind < 0.35:
            _add(rng.choice(wants), rng.choice(plain))
        elif kind < 0.6:
            _add(rng.choice(plain), rng.choice(wants))
        else:
            _add(rng.choice(plain), rng.choice(plain))

    # Leaf partners (tests A/K/R only): ordinary nodes whose EVERY incident synapse touches a want, so the caller-side
    # last-link rule (plan sec 4A.3) has something to hold back.
    leaves = []
    g.timestep = 0  # created at timestep 0 like every other node, so they are past orphan_node_grace_period
    for i in range(n_leaf):
        lid = "leaf::%d" % i
        g.create_node(node_id=lid, metadata={})
        leaves.append(lid)
    g.timestep = TIMESTEP
    for i, lid in enumerate(leaves):
        _add(rng.choice(wants), lid)
        if i % 2:
            _add(lid, rng.choice(wants))

    for (a, b) in pairs:
        syn = g.create_synapse(a, b, rng.choice(_WEIGHTS))
        syn.weight = rng.choice(_WEIGHTS)
        syn.peak_weight = rng.choice(_PEAK)
        syn.low_weight_steps = rng.choice(_LWS)
        syn.inactive_steps = rng.choice(_INACTIVE)
        syn.salience = rng.choice(_SALIENCE)
        syn.creation_time = float(TIMESTEP - rng.choice(_AGE))
        if rng.random() < 0.2:
            g._synapse_confirmation_history.setdefault(syn.synapse_id, collections.deque([True, False, True], maxlen=10))
    g._dirty_synapses.update(g.synapses.keys())
    return g, {"rim": rim, "wants": wants, "plain": plain, "leaves": leaves}


def force_worst(g, sids):
    """Put synapses in the WORST state for survival (plan sec 2.6/R-1): every prune predicate would take them."""
    grace = g.config["grace_period"]
    for sid in sids:
        s = g.synapses[sid]
        s.weight = 0.0
        s.low_weight_steps = grace + 1
        s.inactive_steps = 10 ** 6
        s.creation_time = 0.0
        s.peak_weight = 0.0
        s.salience = 1.0


# ---------------------------------------------------------------------------
# Reference (test-side) implementation of the CALLER's job, written from plan-005 sec 2.1-2.3, 4A.2, 4A.3.
# It exists so tests can pin what the orchestrator must build; it is NOT engine code and is never imported by it.
# ---------------------------------------------------------------------------
class RefSets(object):
    pass


def is_constitutional(g, nid):
    return bool((g.nodes[nid].metadata or {}).get("constitutional"))


def protected_wants(g):
    return sorted(n for n in g.nodes if g._is_identity_protected(n) and not is_constitutional(g, n))


def frozen_rim(g):
    """F: every synapse whose pre or post node is constitutional (plan sec 2.1)."""
    return {sid for sid, s in g.synapses.items() if is_constitutional(g, s.pre_node_id) or is_constitutional(g, s.post_node_id)}


def rank_key(g, sid):
    """Sec 2.3: weight desc -> peak_weight desc -> inactive_steps asc -> synapse_id asc (string compare)."""
    s = g.synapses[sid]
    return (-s.weight, -s.peak_weight, s.inactive_steps, sid)


def guaranteed(g, K, F):
    """G: per protected non-constitutional node, its K strongest non-F outgoing AND K strongest non-F incoming."""
    G = set()
    for p in protected_wants(g):
        for idx in (g._outgoing, g._incoming):
            ids = [sid for sid in idx.get(p, ()) if sid not in F]
            G.update(sorted(ids, key=lambda x: rank_key(g, x))[:K])
    return G


def arena(g, F):
    """Non-F synapses touching a protected non-constitutional node."""
    out = set()
    for sid, s in g.synapses.items():
        if sid in F:
            continue
        if g._is_identity_protected(s.pre_node_id) or g._is_identity_protected(s.post_node_id):
            out.add(sid)
    return out


def last_link_set(g, competing0):
    """Sec 4A.3: for each UNPROTECTED partner whose every incident synapse is in the competing set, hold back the
    strongest (sec 2.3 order) one. Conservative: no eligibility knowledge."""
    held = set()
    for nid in g.nodes:
        if g._is_identity_protected(nid):
            continue
        inc = set(g._outgoing.get(nid, ())) | set(g._incoming.get(nid, ()))
        if inc and inc <= competing0:
            held.add(sorted(inc, key=lambda x: rank_key(g, x))[0])
    return held


def ref_sets(g, K):
    r = RefSets()
    r.F = frozen_rim(g)
    r.G = guaranteed(g, K, r.F)
    r.arena = arena(g, r.F)
    competing0 = r.arena - r.G
    r.last = last_link_set(g, competing0)
    r.competing = competing0 - r.last
    r.excluded = r.F | r.G | r.last
    return r


def ref_order_key(g, competing):
    """Sec 4A.2 (Exec P409): the static HEIGHT key. Per want, competing links stalest-first (inactive desc, weight asc,
    id asc) get rank r; height = c_w - r. A want<->want link takes the LARGER of its two endpoint heights (plan flags
    this as an open policy point; the larger is what the plan's numbers use). key = (-height, -inactive, weight, id)."""
    by_want = collections.defaultdict(list)
    for sid in competing:
        s = g.synapses[sid]
        # C5: a link counts ONCE per distinct endpoint (a self-loop is one link of its want), as plan 4A.2 "c_w = w's number of competing links"
        for nid in {s.pre_node_id, s.post_node_id}:
            if g._is_identity_protected(nid) and not is_constitutional(g, nid):
                by_want[nid].append(sid)
    height = {}
    for _w, ids in by_want.items():
        ids.sort(key=lambda x: (-g.synapses[x].inactive_steps, g.synapses[x].weight, x))
        c = len(ids)
        for r, sid in enumerate(ids):
            height[sid] = max(height.get(sid, c - r), c - r)
    return {sid: (-height[sid], -g.synapses[sid].inactive_steps, g.synapses[sid].weight, sid) for sid in competing}


def ref_eligible(g, sid):
    """The three predicates of _prune_synapses as they behave at runtime, read-only (no counter mutation).
    Weight rule sees low_weight_steps AFTER the +1; the identity skip is NOT applied (the caller decides membership)."""
    c = g.config
    s = g.synapses[sid]
    if s.weight < c["weight_threshold"]:
        if s.low_weight_steps + 1 > c["grace_period"]:
            return True
    if s.inactive_steps > c["inactivity_threshold"] * s.salience:
        return True
    age = g.timestep - s.creation_time
    return bool(age > c["grace_period"] and s.peak_weight < 2.0 * c["initial_sprouting_weight"])


def good_kwargs(g, K, B):
    s = ref_sets(g, K)
    return s, dict(competing_ids=set(s.competing), excluded_ids=set(s.excluded), max_removals=B,
                   order_key=ref_order_key(g, s.competing), report={})


def bad_calls(g, K, B):
    """Each entry is ONE defect on top of an otherwise valid competing call: (label, kwargs)."""
    s, ok = good_kwargs(g, K, B)
    rim_id = sorted(s.F)[0]
    some = sorted(s.competing)[0]

    def with_(**kw):
        d = dict(ok)
        d["competing_ids"] = set(ok["competing_ids"])
        d["excluded_ids"] = set(ok["excluded_ids"])
        d["order_key"] = dict(ok["order_key"])
        d["report"] = {}
        d.update(kw)
        return d

    calls = []
    both = with_()
    both["excluded_ids"].add(some)
    calls.append(("competing id also excluded", both))
    const = with_()
    const["competing_ids"].add(rim_id)
    const["excluded_ids"].discard(rim_id)
    const["order_key"][rim_id] = (0, 0, 0.0, rim_id)
    calls.append(("competing id touches a constitutional node", const))
    absent = with_()
    absent["competing_ids"].add("no-such-synapse")
    absent["order_key"]["no-such-synapse"] = (0, 0, 0.0, "no-such-synapse")
    calls.append(("competing id absent from self.synapses", absent))
    calls.append(("competing without excluded", with_(excluded_ids=None)))
    calls.append(("excluded without competing", with_(competing_ids=None)))
    missing = with_()
    del missing["order_key"][some]
    calls.append(("order_key lacks an entry for a competing id", missing))
    calls.append(("order_key None in competing mode", with_(order_key=None)))
    calls.append(("order_key omitted in competing mode", {k: v for k, v in with_().items() if k != "order_key"}))
    calls.append(("max_removals omitted in competing mode", {k: v for k, v in with_().items() if k != "max_removals"}))
    for bad in (None, 0, -1, -3, 5.0, "5"):
        calls.append(("max_removals=%r in competing mode" % (bad,), with_(max_removals=bad)))
    # C2 (le-036 N3-1): order_key VALUES that cannot be sorted together. The poisoned entry belongs to an ELIGIBLE competitor, so a
    # validator that only looks at keys lazily (after the loop) would reach the sort with it and raise TypeError AFTER every competitor's
    # low_weight_steps moved. Each must be a ValueError raised BEFORE the loop, state unchanged.
    elig = sorted(x for x in ok["competing_ids"] if ref_eligible(g, x))
    assert len(elig) >= 2, "C2 fixture needs >= 2 eligible competitors"
    mixed = with_()
    mixed["order_key"][elig[0]] = ("not-a-number", 1, 1.0, elig[0])
    calls.append(("order_key values not mutually comparable (str vs number)", mixed))
    scalar = with_()
    scalar["order_key"][elig[1]] = 7
    calls.append(("order_key entry is not a tuple", scalar))
    short = with_()
    short["order_key"][elig[0]] = (0, 0)
    calls.append(("order_key entries are tuples of different length", short))
    return calls


def inject_self_loop(nf, g, nid, weight=0.0, **fields):
    """C5: a restored-style self-loop. Graph.create_synapse refuses pre == post ("Self-connections not allowed"), but a synapse can
    arrive in a restored checkpoint; this builds it exactly the way create_synapse does, minus the refusal. Returns its id."""
    syn = nf.Synapse(pre_node_id=nid, post_node_id=nid, weight=weight, max_weight=g.config["max_weight"], delay=1,
                     synapse_type=nf.SynapseType.EXCITATORY, creation_time=float(g.timestep),
                     last_update_time=float(g.timestep), peak_weight=weight)
    sid = syn.synapse_id
    with g._step_lock:
        g.synapses[sid] = syn
        g._outgoing[nid].add(sid)
        g._incoming[nid].add(sid)
        g._dirty_synapses.add(sid)
    for k, v in fields.items():
        setattr(g.synapses[sid], k, v)
    return sid


def _synapse_row(sid, s):
    return [sid, s.pre_node_id, s.post_node_id, repr(s.weight), repr(s.peak_weight), s.low_weight_steps,
            s.inactive_steps, repr(s.salience), repr(s.creation_time)]


def state_digest(g, events=None):
    """sha256 over the full post-state test G compares (plan-005 sec 2.6/G): every synapse field IN store items() order,
    _dirty_synapses, _synapse_confirmation_history, the node set, timestep, and the `pruned` event count/timestep."""
    doc = {
        "timestep": g.timestep,
        "synapses_in_items_order": [_synapse_row(sid, s) for sid, s in g.synapses.items()],
        "dirty_synapses": sorted(g._dirty_synapses),
        "confirmation_history": sorted((k, list(v)) for k, v in g._synapse_confirmation_history.items()),
        "nodes": sorted(g.nodes.keys()),
        "pruned_events": events if events is not None else None,
    }
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def instrument(g):
    """Record removal ORDER (wraps the instance's _remove_synapse_internal) and `pruned` events. Returns (order, events)."""
    order, events = [], []
    orig = g._remove_synapse_internal

    def _wrap(sid):
        order.append(sid)
        return orig(sid)

    g._remove_synapse_internal = _wrap
    g.register_event_handler("pruned", lambda **kw: events.append([kw.get("count"), kw.get("timestep")]))
    return order, events


NEW_PARAMS = ("competing_ids", "excluded_ids", "max_removals", "order_key", "report")


def new_api_present(nf):
    import inspect
    try:
        params = inspect.signature(nf.Graph._prune_synapses).parameters
    except (TypeError, ValueError):
        return False
    return all(p in params and params[p].kind is inspect.Parameter.KEYWORD_ONLY for p in NEW_PARAMS)


def git_rev(checkout):
    return subprocess.run(["git", "-C", checkout, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------------
# Script entry: ONE checkout, ONE scenario, ONE JSON record on stdout (last line)
# ---------------------------------------------------------------------------
SCENARIOS = ("door_a", "door_b", "direct_defaults", "direct_explicit_none")


def _run(checkout, scenario, ckpt_path, ids="counter"):
    checkout = os.path.realpath(checkout)
    sys.path.insert(0, checkout)
    random.seed(0)
    try:
        import numpy
        numpy.random.seed(0)
    except Exception:
        pass
    if ids == "random":
        install_seeded_random_uuid()      # C1: uuid4-shaped ids NOT ascending in creation order
    else:
        install_deterministic_uuid()
    import neuro_foundation as nf
    nf_file = os.path.realpath(nf.__file__)
    if os.path.dirname(nf_file) != checkout:
        print(json.dumps({"void": True, "reason": "neuro_foundation resolved to %s, not %s" % (nf_file, checkout)}))
        return 3
    g, _roles = build_graph(nf, seed=11 if scenario != "door_b" else 12, tonic_ages=(scenario == "door_b"))
    order, events = instrument(g)
    before_ids = list(g.synapses.keys())
    ret = None
    variant = "ok"
    if scenario == "door_a":
        ret = list(g._structural_plasticity([]))
    elif scenario == "door_b":
        res = g.prime_and_propagate(["n0"], [2.0], steps=1, write_mode=True)
        ret = [res.nodes_primed, res.steps_run]
    elif scenario == "direct_defaults":
        ret = g._prune_synapses()
    elif scenario == "direct_explicit_none":
        if not new_api_present(nf):
            variant = "unavailable: _prune_synapses has no keyword-only %s" % (list(NEW_PARAMS),)
            ret = None
        else:
            ret = g._prune_synapses(competing_ids=None, excluded_ids=None, max_removals=None, order_key=None, report=None)
    else:
        raise SystemExit("unknown scenario %r" % scenario)
    del g._remove_synapse_internal  # drop the recording wrapper before serializing
    g.checkpoint(ckpt_path)
    rec = {
        "void": False,
        "scenario": scenario,
        "variant": variant,
        "ids": ids,
        "creation_order_sorted": before_ids == sorted(before_ids),
        "neuro_foundation_file": nf_file,
        "ng_tract_file": ng_tract_info()[0],
        "ng_tract_version": ng_tract_info()[1],
        "git_rev": git_rev(checkout),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "new_api_present": new_api_present(nf),
        "synapses_before": len(before_ids),
        "synapses_after": len(g.synapses),
        "return": ret,
        "removal_order": order,
        "pruned_events": events,
        "state_digest": state_digest(g, events),
        "checkpoint_sha256": sha256_file(ckpt_path),
    }
    print(json.dumps(rec, sort_keys=True))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--scenario", required=True, choices=SCENARIOS)
    ap.add_argument("--ckpt", required=True, help="scratch temp .msgpack path (never a live path)")
    ap.add_argument("--ids", choices=("counter", "random"), default="counter",
                    help="synapse id stream: counter (creation order == id order) or seeded random uuid4-shaped (C1)")
    a = ap.parse_args()
    sys.exit(_run(a.checkout, a.scenario, a.ckpt, a.ids))
