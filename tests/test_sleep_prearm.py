# ---- Changelog ----
# [2026-10-07] Claude (lane sleep-prearm) — CREATE: sleep phase pre-arming (spec §8 P3 prerequisites)
# What: (1) every new key absent: the branch == the trial tip fe6538b exactly (the P1/P2 whole-run workload; per-step
#       synapse state bitwise; final checkpoint bytes), with and without structural_plasticity_in_sleep + P1 sleeps, and
#       with the disuse sleep ON but unchunked (the one-hold P2 path, incl. the migration's shared row helper).
#       (2) chunked disuse sleep (sleep_clearance_chunk_seconds): the SAME decisions as the one-hold sleep — whole runs
#       and a built graph: removed ids in order, counters, stamps, records (minus timing), checkpoint bytes; holds
#       bounded and the lock really released between them (another thread gets in); one "pruned" event per hold, under
#       the lock, summing to the record (#1051's ledger counts exactly); validation refuses before touching anything;
#       steps running concurrently between holds never remove and leave the indexes consistent.
#       (3) #1066: compete_protected_links refuses while disuse owns low_weight_steps; keys absent it runs as before.
#       (4) _prune_synapses(defer_removal=...) is refused outside the sleep-unit clearance / without a report.
# Why:  spec superpowers/specs/2026-10-06-sleep-phase-design.md §8 P3 ("Before arming") and risk row "_step_lock hold";
#       punch list #1066.
# How:  BASE = `git show fe6538b:neuro_foundation.py` under its own name; the P1 test module's builder/workload reused.
# -------------------
"""Sleep phase pre-arming: chunked clearance + migration, #1066 refusal, keys-absent equivalence."""
import importlib.util
import os
import subprocess
import sys
import tempfile
import threading
import time

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import neuro_foundation as NF  # noqa: E402  (the branch)
from tests import test_sleep_p1 as P1  # noqa: E402  (builder, workload, helpers)

BASE_REV = os.environ.get("NG_SLEEP_PREARM_BASE_REV", "fe6538b")


def _load_base():
    src = subprocess.run(["git", "-C", _REPO, "show", f"{BASE_REV}:neuro_foundation.py"],
                         check=True, capture_output=True, text=True).stdout
    d = tempfile.mkdtemp(prefix="sleep_prearm_base_")
    p = os.path.join(d, "nf_base_sleep_prearm.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nf_base_sleep_prearm", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nf_base_sleep_prearm"] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base()
MODES = P1.MODES
DISUSE = {"sleep_disuse_enabled": True, "structural_plasticity_in_sleep": True, "sleep_downscale_d0": 0.1,
          "sleep_downscale_h": 0.05, "sleep_weight_grace_sleeps": 1, "sleep_last_link_grace_sleeps": 2,
          "sleep_credit_shield_kappa": 0.015}
CHUNK_KEYS = ("sleep_clearance_chunk_seconds", "sleep_clearance_chunk_gap_seconds")


def restore(mode, b):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        with open(p, "wb") as f:
            f.write(b)
        g = BASE.Graph() if mode == "base" else NF.Graph(native_node_store=(mode == "on"))
        g.restore(p)
        return g


def run(mode, start, act, seed, prep=None):
    g = restore(mode, start)
    if prep is not None:
        prep(g)
    out = P1.seeded(seed, lambda rng: act(g, rng), uuid_stream=1)
    return P1.ckpt_bytes(g), out


def _prep(cfg, purge=False):
    def prep(g):
        g.config.update(cfg)
        if purge:
            g.purge_dangling_pred_weights()
    return prep


_TIMING = ("seconds", "seconds_parts", "lock_holds", "chunked", "chunk", "decided")


def rec_cmp(r):
    """A sleep record without its timing / chunking fields (they differ by construction)."""
    return sorted((k, repr(v)) for k, v in r.items() if k not in _TIMING)


# ---------------------------------------------------------------------------
# (1) keys absent: fe6538b exactly
# ---------------------------------------------------------------------------

def test_new_keys_are_not_in_default_config():
    for k in CHUNK_KEYS:
        assert k not in NF.DEFAULT_CONFIG


def p1_sleep(g):
    r = g.sleep_cycle()
    return (r["pruned"], r["nodes_collected"], r["synapses_after"], r["nodes_after"], r["in_sleep_mode"])


@pytest.mark.parametrize("flags", ["absent", "lifeline_budget"])
@pytest.mark.parametrize("seed", P1.SEEDS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("in_sleep", [False, True])
def test_keys_absent_is_fe6538b_exactly(mode, seed, flags, in_sleep):
    start = P1.built_bytes(seed)
    cfg = dict(P1.FLAGS[flags])
    if in_sleep:
        cfg["structural_plasticity_in_sleep"] = True
    act = P1.workload(30, 6, p1_sleep)                     # incl. compete_protected_links every 10 rounds (works)
    a, (la, sa) = run("base", start, act, seed, _prep(cfg))
    b, (lb, sb) = run(mode, start, act, seed, _prep(cfg))
    P1.compare(la, lb, sa, sb, a, b)


def disuse_rec(g):
    return rec_cmp(g.sleep_cycle())


@pytest.mark.parametrize("seed", P1.SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_disuse_unchunked_is_fe6538b_exactly(mode, seed):
    """The one-hold disuse sleep (chunk key absent) is the P2 path unchanged (the migration now goes through the shared
    row helper). compete_protected_links is left out on both sides (#1066 refuses it on the branch, by intent)."""
    start = P1.built_bytes(seed)
    cfg = dict(P1.FLAGS["lifeline_budget"], **DISUSE)
    act = P1.workload(30, 3, disuse_rec, compete=False)
    a, (la, sa) = run("base", start, act, seed, _prep(cfg, purge=True))
    b, (lb, sb) = run(mode, start, act, seed, _prep(cfg, purge=True))
    P1.compare(la, lb, sa, sb, a, b)


# ---------------------------------------------------------------------------
# (2) the chunked disuse sleep: same decisions, bounded holds
# ---------------------------------------------------------------------------

def chunked_sleep(hold, gap=0.0, into=None):
    """Sets the chunk keys ONLY around the call, so config (and every snapshot) is the same as the unchunked run."""
    def fn(g):
        g.config.update({"sleep_clearance_chunk_seconds": hold, "sleep_clearance_chunk_gap_seconds": gap})
        try:
            r = g.sleep_cycle()
        finally:
            for k in CHUNK_KEYS:
                g.config.pop(k, None)
        assert r["chunked"] is True
        if into is not None:
            into.append(r)
        return rec_cmp(r)
    return fn


@pytest.mark.parametrize("flags", ["lifeline", "lifeline_budget"])
@pytest.mark.parametrize("seed", P1.SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_chunked_whole_run_equals_one_hold(mode, seed, flags):
    start = P1.built_bytes(seed)
    cfg = dict(P1.FLAGS[flags], **DISUSE)
    recs = []
    a, (la, sa) = run(mode, start, P1.workload(30, 3, disuse_rec, compete=False), seed, _prep(cfg, purge=True))
    # hold 1e-9 s: every hold ends at the first 32-row check -> the most holds possible
    b, (lb, sb) = run(mode, start, P1.workload(30, 3, chunked_sleep(1e-9, 0.0, recs), compete=False), seed,
                      _prep(cfg, purge=True))
    P1.compare(la, lb, sa, sb, a, b)
    assert len(recs) == 10 and recs[0]["migrated"]
    assert sum(r["pruned"] for r in recs) > 0, "vacuous: nothing removed"
    # hold 1e-9: every hold ends at its first 32-row clock check, so a sleep that decided n removals took ceil(n/32) holds
    assert all(len(r["lock_holds"]["remove"]) == (r["decided"] + 31) // 32 for r in recs)
    assert len(recs[0]["lock_holds"]["migrate"]) > 1


def big_graph(mode="off", n=60, per=40, seed=3):
    """n nodes, ~n*per synapses, most faint, a constitutional hub and a want: enough rows for many 32-row holds.
    Seeded uuid4 (P1.seeded), so two builds have the same synapse ids."""
    return P1.seeded(seed, lambda rng: _big_graph(mode, n, per, rng))


def _big_graph(mode, n, per, rng):
    g = NF.Graph({"prune_protected_faint_links": True, "orphan_node_grace_period": 0, "three_factor_enabled": True},
                 native_node_store=(mode == "on"))
    for i in range(n):
        g.create_node(node_id=f"n{i:03d}")
    g.create_node(node_id="CC", metadata={"constitutional": True})
    g.create_node(node_id="W", metadata={"provenance": "syl_authored"})
    ids = list(g.nodes)
    for i in range(n * per):
        a, b = rng.sample(ids, 2)
        s = g.create_synapse(a, b, weight=rng.choice([0.0005, 0.002, 0.004, 0.008, 0.02, 0.3, 1.1]))
        s.low_weight_steps = rng.randint(0, 6000)
        s.eligibility_trace = rng.choice([0.0, 0.0, rng.uniform(-0.3, 0.3)])
        if rng.random() < 0.05:
            s.metadata = {"last_link_since": 10, "k": i}
        if rng.random() < 0.3:
            g.nodes[a].pred_weights[b] = 0.4
    g.timestep = 70_000
    g.config.update(DISUSE)
    return g


@pytest.mark.parametrize("mode", MODES)
def test_chunked_built_graph_same_removal_set_order_counters_and_bytes(mode):
    seen = {}
    for label, hold in (("one", None), ("chunked", 1e-9)):
        g = big_graph(mode)
        removed = []
        g.register_event_handler("pruned", lambda count=0, **_: removed.append(count))
        order = []
        orig = g._remove_synapse_internal
        g._remove_synapse_internal = lambda sid, _o=orig: (order.append(sid), _o(sid))[1]
        recs = []
        for _ in range(3):
            if hold is not None:
                g.config.update({"sleep_clearance_chunk_seconds": hold, "sleep_clearance_chunk_gap_seconds": 0.0})
            recs.append(g.sleep_cycle())
            for k in CHUNK_KEYS:
                g.config.pop(k, None)
        seen[label] = (order, [rec_cmp(r) for r in recs], P1.syn_state(g), P1.pw_state(g), P1.ckpt_bytes(g),
                       sum(removed), recs)
    one, ch = seen["one"], seen["chunked"]
    assert one[0] == ch[0] and len(one[0]) > 500
    assert one[1:5] == ch[1:5]
    assert one[5] == ch[5] == sum(r["pruned"] for r in ch[6])
    assert max(len(r["lock_holds"]["remove"]) for r in ch[6]) > 10


def test_chunked_holds_bounded_lock_released_and_events_under_lock():
    g = big_graph(n=80, per=60)
    g.config.update({"sleep_weight_grace_sleeps": 0})          # the first (migrating) sleep already clears
    hold = 0.003
    g.config.update({"sleep_clearance_chunk_seconds": hold, "sleep_clearance_chunk_gap_seconds": 0.002})
    events = []

    def on_pruned(count=0, **_):
        events.append((count, g._step_lock._is_owned(), len(g.synapses)))
    g.register_event_handler("pruned", on_pruned)
    before = len(g.synapses)
    got_in = []
    stop = threading.Event()

    def other():                                               # another thread that wants the lock
        while not stop.is_set():
            with g._step_lock:
                got_in.append(time.perf_counter())
            time.sleep(0.0005)
    t = threading.Thread(target=other)
    t.start()
    try:
        t0 = time.perf_counter()
        r = g.sleep_cycle()
        t1 = time.perf_counter()
    finally:
        stop.set()
        t.join()
    assert r["pruned"] > 1000 and len(r["lock_holds"]["remove"]) > 1
    assert all(owned for _, owned, _ in events)
    assert sum(c for c, _, _ in events) == r["pruned"]
    cum = 0
    for c, _, n_after in events:                               # every event is exact at the moment it fires
        cum += c
        assert n_after == before - cum
    assert sum(1 for x in got_in if t0 < x < t1) >= 2, "the other thread never got the lock during the sleep"
    # each hold checks the clock every 32 rows: generous slack for a loaded machine
    assert max(r["lock_holds"]["remove"]) < hold + 0.5
    assert max(r["lock_holds"]["migrate"]) < hold + 0.5


def test_guardian_ledger_counts_a_chunked_sleep_exactly():
    import checkpoint_guardian as CG
    g = big_graph()
    g.config.update({"sleep_weight_grace_sleeps": 0, "sleep_clearance_chunk_seconds": 1e-9,
                     "sleep_clearance_chunk_gap_seconds": 0.0})
    led = CG.RemovalLedger().attach(g)
    r = g.sleep_cycle()
    s = led.snapshot()
    assert s["synapses"] == r["pruned"] > 0 and s["sleep_synapses"] == r["pruned"] and s["sleep_cycles"] == 1
    assert 2 <= s["prune_events"] <= len(r["lock_holds"]["remove"])
    assert s["handler_errors"] == 0


def test_steps_between_holds_never_remove_and_indexes_stay_consistent():
    g = big_graph(n=60, per=50)
    g.config.update({"sleep_weight_grace_sleeps": 0, "sleep_clearance_chunk_seconds": 0.002,
                     "sleep_clearance_chunk_gap_seconds": 0.003})
    step_pruned = []
    stop = threading.Event()
    errors = []

    def stepper():
        try:
            while not stop.is_set():
                for nid in list(g.nodes)[:10]:
                    g.stimulate(nid, 2.0)
                step_pruned.append(g.step().synapses_pruned)
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
    t = threading.Thread(target=stepper)
    t.start()
    try:
        r = g.sleep_cycle()
    finally:
        stop.set()
        t.join()
    assert errors == []
    assert step_pruned and all(p == 0 for p in step_pruned)
    assert r["pruned"] > 0
    for nid, sids in list(g._outgoing.items()) + list(g._incoming.items()):
        assert all(s in g.synapses for s in sids)
    assert P1.dangling(g) == []


@pytest.mark.parametrize("key,val", [("sleep_clearance_chunk_seconds", 0.0), ("sleep_clearance_chunk_seconds", -1.0),
                                     ("sleep_clearance_chunk_seconds", float("nan")),
                                     ("sleep_clearance_chunk_seconds", True), ("sleep_clearance_chunk_seconds", "0.2"),
                                     ("sleep_clearance_chunk_gap_seconds", None),
                                     ("sleep_clearance_chunk_gap_seconds", -0.1),
                                     ("sleep_clearance_chunk_gap_seconds", float("inf"))])
def test_chunk_validation_refuses_before_touching_anything(key, val):
    g = big_graph(n=10, per=5)
    g.config.update({"sleep_clearance_chunk_seconds": 0.2, "sleep_clearance_chunk_gap_seconds": 0.01})
    g.config[key] = val
    before = P1.ckpt_bytes(g)
    with pytest.raises(ValueError):
        g.sleep_cycle()
    assert P1.ckpt_bytes(g) == before and "sleep_low_weight_unit" not in g.config


def test_defer_removal_is_refused_outside_the_sleep_unit_clearance():
    g = big_graph(n=10, per=5)
    with pytest.raises(ValueError):
        g._prune_synapses(defer_removal=True)
    with pytest.raises(ValueError):
        g._prune_synapses(grace_sleeps=1, sleep_now=1, last_link_grace_sleeps=2, credit_shield_kappa=0.0,
                          defer_removal=True)                  # no report dict


def test_one_info_line_reports_the_holds(caplog):
    import logging
    g = big_graph(n=20, per=20)
    g.config.update({"sleep_clearance_chunk_seconds": 0.2, "sleep_clearance_chunk_gap_seconds": 0.0})
    with caplog.at_level(logging.INFO, logger=NF.logger.name):
        r = g.sleep_cycle()
    lines = [x.getMessage() for x in caplog.records if "sleep_cycle(disuse, chunked)" in x.getMessage()]
    assert len(lines) == 1 and "lock holds" in lines[0] and str(r["pruned"]) in lines[0]


# ---------------------------------------------------------------------------
# (3) #1066
# ---------------------------------------------------------------------------

def want_graph():
    g = NF.Graph({"orphan_node_grace_period": 0})
    for i in range(6):
        g.create_node(node_id=f"p{i}")
    g.create_node(node_id="W", metadata={"provenance": "syl_authored"})
    for i in range(6):
        s = g.create_synapse("W", f"p{i}", weight=0.001)
        s.low_weight_steps = 10_000
        g.create_synapse(f"p{i}", f"p{(i + 1) % 6}", weight=0.5)   # partners stay wired: no last-link hold
    return g


def test_compete_refuses_under_disuse_and_with_the_sleep_marker_and_touches_nothing():
    g = want_graph()
    g.config.update(DISUSE)
    before = P1.ckpt_bytes(g)
    with pytest.raises(ValueError, match="#1066"):
        g.compete_protected_links(1, 5)
    assert P1.ckpt_bytes(g) == before
    g2 = want_graph()
    g2.config["sleep_low_weight_unit"] = "sleeps"
    with pytest.raises(ValueError, match="#1066"):
        g2.compete_protected_links(1, 5)
    g2.config.pop("sleep_low_weight_unit")                     # back in step units: it runs again
    assert g2.compete_protected_links(1, 5)["removed"] >= 1
