# ---- Changelog ----
# [2026-10-07] Claude (lane sleep-observe) — CREATE: sleep phase P3 observe mode (Graph.sleep_observe)
# What: (1) keys absent / observe never called: the branch == the trial tip 1cb9706 exactly (the P1/P2 whole-run workload,
#       both node stores; P1 sleeps; the disuse sleep one-hold and chunked; per-step state bitwise; checkpoint bytes).
#       (2) observe WRITES NOTHING: checkpoint bytes, every synapse column, stamps, pred_weights, config, the dirty sets,
#       adjacency, _total_pruned, the fair-chance latch, the live event handlers (no pruned / nodes_collected /
#       sleep_cycle; the #1051 RemovalLedger unchanged), on the P1 path, the disuse path (migrating, migrated, chunked),
#       with config overrides, with a live instance patch, both node stores.
#       (3) observe == real: from the same state, projection 1's removal order, collected nodes, shield-held ids, record
#       (minus timing) and the post-downscale weight of EVERY synapse equal what the real sleep_cycle then does, over
#       several wake/sleep cycles (migration, tags, the first big clearance, steady state), chunked and one-hold; a
#       k-sleep projection == k consecutive real sleeps; an unarmed graph + overrides == the armed graph's real sleep.
#       (4) the live lock is held ONCE, briefly; another thread steps during the projection; validation refuses before
#       touching anything; one INFO line, the shadow's own sleep line is not INFO.
# Why:  spec superpowers/specs/2026-10-06-sleep-phase-design.md §8 P3 (observe first, nothing written) + the lane brief.
# How:  BASE = `git show 1cb9706:neuro_foundation.py`; builders / workload from tests/test_sleep_p1.py and
#       tests/test_sleep_prearm.py.
# -------------------
"""Sleep phase P3 observe mode: the real sleep on a private shadow; nothing written to the live graph."""
import copy
import importlib.util
import logging
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
from tests import test_sleep_p1 as P1  # noqa: E402
from tests import test_sleep_prearm as PA  # noqa: E402

BASE_REV = os.environ.get("NG_SLEEP_OBSERVE_BASE_REV", "1cb9706")


def _load_base():
    src = subprocess.run(["git", "-C", _REPO, "show", f"{BASE_REV}:neuro_foundation.py"],
                         check=True, capture_output=True, text=True).stdout
    d = tempfile.mkdtemp(prefix="sleep_observe_base_")
    p = os.path.join(d, "nf_base_sleep_observe.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nf_base_sleep_observe", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nf_base_sleep_observe"] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base()
MODES = P1.MODES
DISUSE = dict(PA.DISUSE)
CHUNK = {"sleep_clearance_chunk_seconds": 1e-9, "sleep_clearance_chunk_gap_seconds": 0.0}
_TIMING = PA._TIMING


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


# ---------------------------------------------------------------------------
# (1) observe never called: 1cb9706 exactly
# ---------------------------------------------------------------------------

def test_observe_adds_no_default_config_key():
    assert not [k for k in NF.DEFAULT_CONFIG if "observe" in k]


@pytest.mark.parametrize("seed", P1.SEEDS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", ["p1_in_sleep", "disuse", "disuse_chunked"])
def test_without_observe_is_1cb9706_exactly(mode, seed, variant):
    start = P1.built_bytes(seed)
    if variant == "p1_in_sleep":
        cfg = dict(P1.FLAGS["lifeline_budget"], structural_plasticity_in_sleep=True)
        act = P1.workload(30, 6, PA.p1_sleep)
        purge = False
    else:
        cfg = dict(P1.FLAGS["lifeline_budget"], **DISUSE)
        if variant == "disuse_chunked":
            cfg.update(CHUNK)
        act = P1.workload(30, 3, PA.disuse_rec, compete=False)
        purge = True
    a, (la, sa) = run("base", start, act, seed, _prep(cfg, purge))
    b, (lb, sb) = run(mode, start, act, seed, _prep(cfg, purge))
    P1.compare(la, lb, sa, sb, a, b)


# ---------------------------------------------------------------------------
# helpers: a full picture of the live graph's state, and a recording real sleep
# ---------------------------------------------------------------------------

def full_state(g):
    """Everything sleep can write, read from the live graph (and the checkpoint bytes)."""
    md = [(sid, repr(sorted((s.metadata or {}).items())), s.low_weight_steps, s.inactive_steps, P1.P(s.salience))
          for sid, s in g.synapses.items()]
    nodes = [(nid, repr(sorted((n.metadata or {}).items()) if isinstance(n.metadata, dict) else n.metadata))
             for nid, n in g.nodes.items()]
    fc = getattr(g, "_fair_chance_cfg", None)
    return {
        "ckpt": P1.ckpt_bytes(g),
        "syn": P1.syn_state(g),
        "md": md,
        "pw": P1.pw_state(g),
        "nodes": nodes,
        "config": repr(sorted((k, repr(v)) for k, v in g.config.items())),
        "dirty": (sorted(g._dirty_nodes), sorted(g._dirty_synapses), sorted(g._dirty_hyperedges)),
        "adj": (sorted((k, sorted(v)) for k, v in g._outgoing.items()), sorted((k, sorted(v)) for k, v in g._incoming.items())),
        "nhe": sorted((k, sorted(v)) for k, v in g._node_hyperedges.items()),
        "hist": sorted(g._synapse_confirmation_history),
        "recent": sorted(g._recent_spikes),
        "total_pruned": g._total_pruned,
        "fair_chance": None if fc is None else repr(sorted((k, v) for k, v in fc.items() if k != "clock")),
        "handlers": {k: list(v) for k, v in g._event_handlers.items()},
        "timestep": g.timestep,
    }


class Events:
    def __init__(self, g):
        self.seen = []
        for ev in ("pruned", "nodes_collected", "sleep_cycle"):
            g.register_event_handler(ev, lambda _ev=ev, **kw: self.seen.append(_ev))


def real_sleep(g):
    """The real sleep_cycle, recording what the observe projection is compared with: removal order, the weight each
    removed synapse had when removed (= post-downscale), collected nodes, and the post-downscale weight of every synapse."""
    order = []
    orig = NF.Graph._remove_synapse_internal

    def rec_remove(sid, _g=g):
        s = _g.synapses.get(sid)
        if s is not None:
            order.append((sid, float(s.weight)))
        orig(_g, sid)
    nodes_before = set(g.nodes.keys())
    g._remove_synapse_internal = rec_remove
    try:
        r = g.sleep_cycle()
    finally:
        del g._remove_synapse_internal
    post = {sid: float(s.weight) for sid, s in g.synapses.items()}
    for sid, w in order:
        post[sid] = w
    return {"record": r, "removed": [sid for sid, _ in order], "collected": sorted(nodes_before - set(g.nodes.keys())),
            "post": post, "held": list(r.get("shield_held_ids", []) or [])}


def assert_projection_equals_real(proj, real):
    assert proj["removed_ids"] == real["removed"]
    assert proj["collected_ids_all"] == real["collected"]
    assert proj["shield_held_ids"] == real["held"]
    assert proj["would_remove"] == real["record"]["pruned"]
    assert proj["would_collect"] == real["record"]["nodes_collected"]
    a, b = proj["post_downscale_weights"], real["post"]
    assert a.keys() == b.keys()
    assert all(P1.P(a[k]) == P1.P(b[k]) for k in a)            # bitwise
    want = {k: repr(v) for k, v in real["record"].items() if k not in _TIMING and k != "shield_held_ids"}
    got = {k: repr(v) for k, v in proj["record"].items() if k not in _TIMING}
    want.pop("downscale_native", None), got.pop("downscale_native", None)
    assert got == want


def wake(g, rng, steps=4):
    for _ in range(steps):
        for nid in rng.sample(list(g.nodes), min(10, len(g.nodes))):
            g.stimulate(nid, rng.uniform(0.5, 3.0))
        g.step()
    ids = rng.sample(list(g.nodes), min(6, len(g.nodes)))
    g.prime_and_propagate(ids, [1.5] * len(ids), steps=3, write_mode=True)
    if rng.random() < 0.5:
        g.inject_reward(0.2)


# ---------------------------------------------------------------------------
# (2) observe writes nothing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", ["disuse_unmigrated", "disuse_migrated", "disuse_chunked", "p1", "overrides"])
def test_observe_writes_nothing(mode, variant):
    import checkpoint_guardian as CG
    g = PA.big_graph(mode)
    g.enable_fair_chance_window(5, heartbeat_max_age_s=1e-9, clock=time.monotonic)   # stale: the latch is open
    overrides = None
    if variant == "disuse_migrated":
        g.sleep_cycle()                                        # migrates + tags (real), then observe from there
    elif variant == "disuse_chunked":
        g.config.update(CHUNK)
    elif variant == "p1":
        for k in DISUSE:
            g.config.pop(k, None)
        g.config["structural_plasticity_in_sleep"] = True
    elif variant == "overrides":
        for k in DISUSE:
            g.config.pop(k, None)                              # an unarmed graph; the shadow gets the armed keys
        overrides = dict(DISUSE, **CHUNK)
    ev = Events(g)
    led = CG.RemovalLedger().attach(g)
    before = full_state(g)
    led0 = led.snapshot()
    r = g.sleep_observe(config_overrides=overrides, detail=True)
    assert r["would_remove_total"] > 0, "vacuous: the projection removes nothing"
    after = full_state(g)
    for k in before:
        assert before[k] == after[k], k
    assert ev.seen == []
    assert led.snapshot() == led0
    assert g._fair_chance_cfg["stale_logged"] is False


def test_observe_ignores_a_live_instance_patch_and_never_calls_it():
    g = PA.big_graph()
    called = []
    orig = g._remove_synapse_internal
    g._remove_synapse_internal = lambda sid, _o=orig: (called.append(sid), _o(sid))[1]
    g.config["sleep_weight_grace_sleeps"] = 0                  # the first projection already clears
    before = full_state(g)
    r = g.sleep_observe(sleeps=1)
    assert r["projections"][0]["would_remove"] > 0 and called == []
    assert "_remove_synapse_internal" in r["instance_overrides_not_used"]
    after = full_state(g)
    assert {k: v for k, v in before.items() if k != "handlers"} == {k: v for k, v in after.items() if k != "handlers"}


def test_observe_never_touches_the_vector_drop_handler():
    g = PA.big_graph()
    g.config.update({"sleep_weight_grace_sleeps": 0, "sleep_last_link_grace_sleeps": 0})
    for i in range(5):                                         # orphans-to-be: one faint link each
        g.create_node(node_id=f"lonely{i}").creation_time = 1
        g.create_synapse(f"lonely{i}", "n000", weight=0.0001)
    dropped = []
    g.register_event_handler("nodes_collected", lambda node_ids=(), **_: dropped.extend(node_ids))
    r = g.sleep_observe(sleeps=1, detail=True)
    assert r["projections"][0]["would_collect"] >= 5 and dropped == []
    assert all(f"lonely{i}" in g.nodes for i in range(5))


# ---------------------------------------------------------------------------
# (3) observe == real
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("grace", [0, 2])
def test_projection_one_equals_the_next_real_sleep_over_wake_sleep_cycles(mode, chunked, grace):
    """Cycles of (wake, observe, real sleep) on a built graph: every projection 1 == the real sleep that follows it.
    grace 2 = the laptop's G: migration, two tag-only sleeps, then the first big clearance, then steady state."""
    import random
    g = PA.big_graph(mode, n=70, per=45)
    g.config["sleep_weight_grace_sleeps"] = grace
    if chunked:
        g.config.update(CHUNK)
    rng = random.Random(5)
    cleared = []
    for cycle in range(7):
        if cycle:
            P1.seeded(100 + cycle, lambda _r: wake(g, rng))
        obs = g.sleep_observe(sleeps=1, detail=True)
        real = real_sleep(g)
        assert_projection_equals_real(obs["projections"][0], real)
        cleared.append(real["record"]["pruned"])
    first = next(i for i, c in enumerate(cleared) if c)
    assert first == grace                                     # sleep G+1 (index G): migration tags, the count must exceed G
    assert cleared[first] > 500, cleared                      # the big first clearance
    assert any(c for c in cleared[first + 1:]), cleared       # and later (steady-state) sleeps


@pytest.mark.parametrize("mode", MODES)
def test_k_sleep_projection_equals_k_consecutive_real_sleeps(mode):
    g = PA.big_graph(mode, n=60, per=40)
    g.config["sleep_weight_grace_sleeps"] = 2
    obs = g.sleep_observe(detail=True)                        # auto: unmigrated -> G + 1 = 3 sleeps
    assert obs["sleeps_projected"] == 3 and obs["first_clearance"] == 3
    obs4 = g.sleep_observe(sleeps=4, detail=True)
    for k in range(4):
        real = real_sleep(g)
        assert_projection_equals_real(obs4["projections"][k], real)
        if k < 3:                                             # the auto (3-sleep) projection too
            assert_projection_equals_real(obs["projections"][k], real)
    after = g.sleep_observe(detail=True)                      # migrated now: auto = 1 (the next real sleep)
    assert after["sleeps_projected"] == 1


@pytest.mark.parametrize("chunked", [False, True])
def test_unarmed_graph_with_overrides_equals_the_armed_real_sleep(chunked):
    """The daemon's observe mode: the live graph keeps its wake removal off the sleep path (keys absent); the projection
    gets the armed keys as overrides. It must equal the real sleep of the same graph once those keys are armed."""
    g = PA.big_graph(n=60, per=40)
    armed = {k: g.config.pop(k) for k in list(DISUSE)}
    armed["sleep_weight_grace_sleeps"] = 0
    if chunked:
        armed.update(CHUNK)
    obs = g.sleep_observe(sleeps=2, config_overrides=armed, detail=True)
    g.config.update(armed)
    for k in range(2):
        assert_projection_equals_real(obs["projections"][k], real_sleep(g))
    assert obs["projections"][0]["would_remove"] > 0


def test_the_projection_record_reports_what_judging_forgetting_needs():
    g = PA.big_graph(n=60, per=40)
    g.config["sleep_weight_grace_sleeps"] = 0
    for s in list(g.synapses.values())[:200]:
        s.peak_weight = 0.9                                     # some once-strong links among the faint ones
    r = g.sleep_observe(sleeps=1, sample=7)
    p = r["projections"][0]
    assert r["observe"] and r["path"] == "disuse" and r["lock_holds"]["count"] == 1
    assert sum(p["weight_bands_before"]) == r["start"]["synapses"]
    assert sum(p["weight_bands_after"]) == r["start"]["synapses"] - p["would_remove"]
    assert sum(p["removed_by_weight_band_before"]) == sum(p["removed_by_peak_band"]) == p["would_remove"]
    assert len(p["strongest_removed"]) == 7
    ws = [x["w_before"] for x in p["strongest_removed"]]
    assert ws == sorted(ws, reverse=True)
    assert {"synapse_id", "pre", "post", "w_before", "w_at_removal", "peak", "trace"} == set(p["strongest_removed"][0])
    prot = {x["node_id"]: x for x in p["protected"]}
    assert set(prot) == {"CC", "W"}
    assert all(x["lifelines_intact"] for x in prot.values()) and p["lifelines_removed"] == []
    assert p["removed_peak_ge_0_5"] > 0
    assert isinstance(p["low_weight_counts_after"], dict) and all(int(k) > 0 for k in p["low_weight_counts_after"])
    assert sum(p["low_weight_counts_after"].values()) >= p["shield_held"]          # held links keep their count
    import json
    json.dumps(r)                                              # serialisable as is (the daemon writes it)


# ---------------------------------------------------------------------------
# (4) lock, validation, logging
# ---------------------------------------------------------------------------

def test_live_lock_held_once_briefly_and_free_during_the_projection():
    g = PA.big_graph(n=80, per=60)
    g.config["sleep_weight_grace_sleeps"] = 0
    acquisitions = []
    real_lock = g._step_lock

    class CountingLock:
        def __enter__(self):
            real_lock.acquire()
            acquisitions.append(time.perf_counter())
            return self

        def __exit__(self, *a):
            real_lock.release()

        def __getattr__(self, n):
            return getattr(real_lock, n)
    g._step_lock = CountingLock()
    steps, stop, errors = [], threading.Event(), []

    def stepper():
        try:
            while not stop.is_set():
                with real_lock:
                    steps.append(time.perf_counter())
                time.sleep(0.0005)
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
    t = threading.Thread(target=stepper)
    t.start()
    try:
        t0 = time.perf_counter()
        r = g.sleep_observe(sleeps=2)
        t1 = time.perf_counter()
    finally:
        stop.set()
        t.join()
    g._step_lock = real_lock
    assert errors == [] and len(acquisitions) == 1
    assert r["lock_holds"]["snapshot"] < 0.5
    t_proj = t0 + r["seconds_parts"]["shadow"]
    assert sum(1 for x in steps if t_proj < x < t1) >= 2, "the live lock was not free during the projection"


def test_concurrent_steps_during_the_projection_are_harmless():
    g = PA.big_graph(n=60, per=50)
    g.config["sleep_weight_grace_sleeps"] = 0
    stop, errors = threading.Event(), []

    def stepper():
        try:
            while not stop.is_set():
                for nid in list(g.nodes)[:10]:
                    g.stimulate(nid, 2.0)
                g.step()
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
    t = threading.Thread(target=stepper)
    t.start()
    try:
        r = g.sleep_observe(sleeps=2)
    finally:
        stop.set()
        t.join()
    assert errors == [] and r["projections"][0]["would_remove"] > 0
    for nid, sids in list(g._outgoing.items()) + list(g._incoming.items()):
        assert all(s in g.synapses for s in sids)


@pytest.mark.parametrize("kw", [{"sleeps": 0}, {"sleeps": 17}, {"sleeps": True}, {"sleeps": 1.0}, {"sample": -1},
                                {"sample": 2.0}, {"config_overrides": [("a", 1)]}, {"config_overrides": {1: 2}},
                                {"config_overrides": {"sleep_downscale_d0": 2.0}},
                                {"config_overrides": {"sleep_credit_shield_kappa": None}},
                                {"config_overrides": {"structural_plasticity_in_sleep": False}},
                                {"config_overrides": {"sleep_clearance_chunk_seconds": 0.0}}])
def test_validation_refuses_and_touches_nothing(kw):
    g = PA.big_graph(n=10, per=5)
    before = full_state(g)
    with pytest.raises(ValueError):
        g.sleep_observe(**kw)
    assert full_state(g) == before


def test_one_info_line_and_the_shadow_sleep_line_is_not_info(caplog):
    g = PA.big_graph(n=20, per=20)
    g.config.update(CHUNK)
    with caplog.at_level(logging.DEBUG, logger=NF.logger.name):
        r = g.sleep_observe(sleeps=3)
    info = [x.getMessage() for x in caplog.records if x.levelno >= logging.INFO]
    assert len([m for m in info if m.startswith("sleep_observe:")]) == 1
    assert not [m for m in info if m.startswith("sleep_cycle")]
    dbg = [x for x in caplog.records if x.getMessage().startswith("sleep_cycle(disuse, chunked)")]
    assert len(dbg) == 3 and all(x.levelno == logging.DEBUG for x in dbg)
    assert str(r["would_remove_total"]) in [m for m in info if m.startswith("sleep_observe:")][0]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=NF.logger.name):
        g.sleep_cycle()                                        # the real sleep still logs INFO
    assert [x for x in caplog.records if x.getMessage().startswith("sleep_cycle(disuse, chunked)")
            and x.levelno == logging.INFO]
