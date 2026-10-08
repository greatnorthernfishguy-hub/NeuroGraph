# ---- Changelog ----
# [2026-10-08] Claude (lane sprout-1050) — CREATE: #1050 co-firing tally sprouting — equivalence, rules, native == fallback
# What: (1) keys ABSENT: the branch == the trial tip 4de1166 exactly over the P1 whole-run workload (step, Tonic ticks,
#       recall, node churn, rewards, competition, downscale, snapshots; per-step synapse state bitwise; final checkpoint
#       bytes), with and without the P1 / P2 sleep keys. (2) the tally's rules on hand-built graphs: one coincidence
#       never sprouts, nor does one burst (an episode counts once); repeated co-firing does, in the STDP direction; K bounds the partner table; only retracted slots
#       are replaced; connected pairs never take a slot; the rails (10 per call, sprout_degree_cap, an existing pair);
#       the Tonic's write-mode tail feeds the tally; surprise-driven sprouting feeds it too (a sprout only on repetition,
#       born as today's surprise sprout), and is unchanged with the keys absent; a sleep empties it; sleep_observe never touches the live tally;
#       a restore starts cold; validation raises. (3) native SynapseStore.cofire_tally_update == the Python fallback,
#       bit for bit: random call sequences (crossings + full state) and whole runs (sprout set and order, every
#       synapse field, checkpoint bytes, tally state after every round), with the sleep on.
# Why:  spec superpowers/specs/2026-10-06-sleep-phase-design.md §7 / D12 (#1050; Josh approved starting 2026-10-08).
# How:  BASE = `git show 4de1166:neuro_foundation.py` under its own name; the P1 test module's builder / workload reused.
#       The fallback is forced by monkeypatching NF._cofire_tally_native (module-level dispatch).
# -------------------
"""#1050: sprouting from repeated co-firing (the co-firing tally)."""
import importlib.util
import logging
import math
import os
import random
import struct
import subprocess
import sys
import tempfile

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import neuro_foundation as NF  # noqa: E402  (the branch)
from tests import test_sleep_p1 as P1  # noqa: E402  (builder, workload, helpers)

BASE_REV = os.environ.get("NG_SPROUT_1050_BASE_REV", "4de1166")
NATIVE = hasattr(NF.ng_tract.SynapseStore, "cofire_tally_update")
needs_native = pytest.mark.skipif(not NATIVE, reason="installed ng_tract has no cofire_tally_update")


def _load_base():
    src = subprocess.run(["git", "-C", _REPO, "show", f"{BASE_REV}:neuro_foundation.py"],
                         check=True, capture_output=True, text=True).stdout
    d = tempfile.mkdtemp(prefix="sprout_1050_base_")
    p = os.path.join(d, "nf_base_sprout_1050.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nf_base_sprout_1050", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nf_base_sprout_1050"] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base()
MODES = P1.MODES
TALLY = {"sprout_tally_enabled": True, "sprout_tally_slots": 4, "sprout_tally_theta": 2.5,
         "sprout_tally_horizon_steps": 40}
DISUSE = {"sleep_disuse_enabled": True, "structural_plasticity_in_sleep": True, "sleep_downscale_d0": 0.1,
          "sleep_downscale_h": 0.05, "sleep_weight_grace_sleeps": 2, "sleep_last_link_grace_sleeps": 2,
          "sleep_credit_shield_kappa": 0.015}
NEW_KEYS = ("sprout_tally_enabled", "sprout_tally_slots", "sprout_tally_theta", "sprout_tally_horizon_steps")


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
    return P1.ckpt_bytes(g), out, g


def _prep(cfg):
    return lambda g: g.config.update(cfg)


def tstate(g):
    """Tally snapshot with floats as exact bytes."""
    return [(o, [None if x is None else (x[0], struct.pack("<d", x[1]), x[2]) for x in tbl])
            for o, tbl in g.cofire_tally_state()]


# ---------------------------------------------------------------------------
# (1) keys absent: the trial tip exactly
# ---------------------------------------------------------------------------

def test_new_keys_are_not_in_default_config():
    for k in NEW_KEYS:
        assert k not in NF.DEFAULT_CONFIG


def p1_sleep(g):
    r = g.sleep_cycle()
    return (r["pruned"], r["nodes_collected"], r["synapses_after"], r["nodes_after"], r["in_sleep_mode"],
            "tally_retracted" in r)


@pytest.mark.parametrize("flags", ["absent", "lifeline_budget"])
@pytest.mark.parametrize("seed", P1.SEEDS[:3])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sleep", ["none", "p1", "disuse"])
def test_keys_absent_is_the_trial_tip_exactly(mode, seed, flags, sleep):
    start = P1.built_bytes(seed)
    cfg = dict(P1.FLAGS[flags])
    if sleep == "p1":
        cfg["structural_plasticity_in_sleep"] = True
    elif sleep == "disuse":
        cfg.update(DISUSE)
    act = P1.workload(30, 6 if sleep != "none" else 0, p1_sleep, compete=(sleep != "disuse"))
    a, (la, sa), _ = run("base", start, act, seed, _prep(cfg))
    b, (lb, sb), gb = run(mode, start, act, seed, _prep(cfg))
    P1.compare(la, lb, sa, sb, a, b)
    assert "_cofire_tally" not in gb.__dict__ and "_sprout_tally_stats" not in gb.__dict__
    assert gb.cofire_tally_state() == []


def test_sprout_delay_is_the_moved_body():
    """The delay rule moved out of _sprout_synapses' loop verbatim (same draw, same geodesic override)."""
    import inspect
    src = inspect.getsource(NF.Graph._sprout_delay)
    base_src = inspect.getsource(BASE.Graph._sprout_synapses)
    body = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("_") and "=" in ln]
    for ln in body:
        assert ln in [x.strip() for x in base_src.splitlines()], ln


# ---------------------------------------------------------------------------
# (2) rules on hand-built graphs
# ---------------------------------------------------------------------------

def graph(n=12, mode="off", **cfg):
    g = NF.Graph(dict(TALLY, **cfg), native_node_store=(mode == "on"))
    for i in range(n):
        g.create_node(node_id=f"n{i:02d}")
    return g


def spike(g, nid, t):
    g._recent_spikes.setdefault(nid, NF.deque(maxlen=20)).append(t)


def cofire(g, earlier, now_fired, t=None):
    """earlier fired at t-1, now_fired fire at t: one _sprout_synapses call at timestep t."""
    t = g.timestep if t is None else t
    g.timestep = t
    for e in earlier:
        spike(g, e, t - 1)
    for f in now_fired:
        spike(g, f, t)
    return g._sprout_synapses(list(now_fired))


def edges(g):
    return sorted((s.pre_node_id, s.post_node_id) for s in g.synapses.values())


@pytest.fixture(params=["native", "fallback"])
def impl(request, monkeypatch):
    if request.param == "native" and not NATIVE:
        pytest.skip("installed ng_tract has no cofire_tally_update")
    if request.param == "fallback":
        monkeypatch.setattr(NF, "_cofire_tally_native", lambda store: None)
    return request.param


def test_one_coincidence_never_sprouts_repetition_does_in_the_stdp_direction(impl):
    g = graph(sprout_tally_theta=2.4, sprout_tally_horizon_steps=50)
    assert cofire(g, ["n01"], ["n00"], t=100) == 0
    assert edges(g) == []
    assert cofire(g, ["n01"], ["n00"], t=110) == 0       # a second occasion: 1*lam^10 + 1 = 1.82 < 2.4
    assert cofire(g, ["n01"], ["n00"], t=111) == 0       # the same burst (gap <= co_activation_window): + 0
    assert cofire(g, ["n01"], ["n00"], t=112) == 0
    assert cofire(g, ["n01"], ["n00"], t=120) == 1       # the third occasion: 1.75*lam^8 + 1 = 2.49 >= 2.4
    assert edges(g) == [("n01", "n00")]                  # earlier -> later
    assert g._sprout_tally_stats["sprouted"] == 1
    assert all(x is None for _, tbl in g.cofire_tally_state() for x in tbl)   # the slot was emptied


def test_decay_lets_far_apart_coincidences_fade(impl):
    g = graph(sprout_tally_theta=2.5, sprout_tally_horizon_steps=5)
    for t in (100, 140, 180, 220):
        assert cofire(g, ["n01"], ["n00"], t=t) == 0
    assert edges(g) == []


def test_partner_table_is_bounded_and_only_retracted_slots_are_replaced(impl):
    g = graph(sprout_tally_slots=2, sprout_tally_theta=3.0, sprout_tally_horizon_steps=50)
    cofire(g, ["n01", "n02", "n03"], ["n00"], t=100)
    st = dict(g.cofire_tally_state())
    assert [x[0] for x in st["n00"]] == ["n01", "n02"]          # first come (equal recency: dict order)
    cofire(g, ["n03"], ["n00"], t=110)                        # nothing retracted yet: n03 finds no slot
    assert [x[0] for x in dict(g.cofire_tally_state())["n00"]] == ["n01", "n02"]
    cofire(g, ["n03"], ["n00"], t=200)                        # both retracted (e^-2 < e^-1): lowest index first
    assert [x[0] for x in dict(g.cofire_tally_state())["n00"]] == ["n03", "n02"]


def test_most_recent_candidates_come_first(impl):
    g = graph(sprout_tally_slots=1, sprout_tally_theta=3.0)
    g.timestep = 100
    spike(g, "n05", 96)   # older
    spike(g, "n06", 99)   # most recent
    spike(g, "n00", 100)
    g._sprout_synapses(["n00"])
    assert dict(g.cofire_tally_state())["n00"][0][0] == "n06"


def test_connected_pairs_never_take_a_slot(impl):
    g = graph()
    g.create_synapse("n00", "n01", weight=0.3)     # either direction counts
    g.create_synapse("n02", "n00", weight=0.3)
    cofire(g, ["n01", "n02", "n03"], ["n00"], t=100)
    assert [x[0] for x in dict(g.cofire_tally_state())["n00"] if x] == ["n03"]


def test_rails_cap_degree_and_existing(impl):
    # 12 pairs cross in one call; the 10-per-call cap stops two
    g = graph(n=30, sprout_tally_slots=1, sprout_tally_theta=1.5, sprout_tally_horizon_steps=50)
    pairs = [(f"n{i:02d}", f"n{i + 12:02d}") for i in range(12)]
    for t in (100, 107):
        g.timestep = t
        for e, _ in pairs:
            spike(g, e, t - 1)
        for _, f in pairs:
            spike(g, f, t)
        n = g._sprout_synapses([f for _, f in pairs])
    assert n == 10
    assert g._sprout_tally_stats["blocked_cap"] >= 2
    # degree cap: a saturated ordinary hub neither sprouts nor receives
    g2 = graph(n=8, sprout_degree_cap=2, sprout_tally_theta=1.5)
    g2.create_synapse("n00", "n05", weight=0.3)
    g2.create_synapse("n00", "n06", weight=0.3)
    cofire(g2, ["n01"], ["n00"], t=100)
    assert cofire(g2, ["n01"], ["n00"], t=107) == 0
    assert g2._sprout_tally_stats["blocked_degree"] == 1


def test_tonic_write_mode_feeds_the_tally_not_a_direct_sprout(impl):
    g = graph(n=6, sprout_tally_theta=2.5)
    g.create_synapse("n00", "n01", weight=3.0)
    g.timestep = 50
    spike(g, "n04", 49)                          # a node that fired in the last steps
    before = len(g.synapses)
    g.prime_and_propagate(["n00"], [5.0], steps=2, write_mode=True)
    assert len(g.synapses) == before             # a single co-firing: no sprout
    assert g._sprout_tally_stats["calls"] >= 1
    assert any(tbl for _, tbl in g.cofire_tally_state())


def test_sleep_empties_the_tally_and_reports_it(impl, caplog):
    g = graph(**DISUSE)
    cofire(g, ["n01", "n02"], ["n00"], t=100)
    assert sum(1 for _, tbl in g.cofire_tally_state() for x in tbl if x) == 2
    with caplog.at_level(logging.INFO, logger=NF.logger.name):
        rec = g.sleep_cycle()
    assert rec["tally_retracted"] == 2
    assert sum(1 for _, tbl in g.cofire_tally_state() for x in tbl if x) == 0
    assert any("sprout tally: sleep retracted 2" in r.getMessage() for r in caplog.records)
    # the P1 sleep (disuse off) retracts too
    g2 = graph(structural_plasticity_in_sleep=True)
    cofire(g2, ["n01"], ["n00"], t=100)
    assert g2.sleep_cycle()["tally_retracted"] == 1


def test_sleep_observe_never_touches_the_live_tally(impl):
    g = graph(**DISUSE)
    cofire(g, ["n01", "n02"], ["n00"], t=100)
    before = tstate(g)
    g.sleep_observe(sleeps=1)
    assert tstate(g) == before


def test_restore_starts_cold(impl, tmp_path):
    g = graph()
    cofire(g, ["n01"], ["n00"], t=100)
    p = str(tmp_path / "c.msgpack")
    g.checkpoint(p)
    g.restore(p)
    assert all(x is None for _, tbl in g.cofire_tally_state() for x in tbl)
    assert g.config["sprout_tally_enabled"] is True        # the keys persist like every config key


def test_surprise_feeds_the_tally_and_sprouts_only_on_repetition_born_as_today(impl):
    g = graph(n=8, sprout_tally_theta=2.4, sprout_tally_horizon_steps=50)
    g.timestep = 100
    pred = NF.Prediction(source_node_id="n00", target_node_id="n01", strength=0.8, confidence=0.5)
    g._surprise_exploration(pred, {"n01", "n02", "n03"})          # one violated expectation: no shotgun
    assert edges(g) == []
    held = dict(g.cofire_tally_state())
    assert held["n02"][0][0] == "n00" and held["n03"][0][0] == "n00"    # owner = what fired instead, partner = source
    g.timestep = 110
    g._surprise_exploration(pred, {"n02"})
    assert edges(g) == []
    g.timestep = 120
    g._surprise_exploration(pred, {"n02"})                         # third occasion n02 instead of n01
    assert edges(g) == [("n00", "n02")]
    s = g._find_synapse("n00", "n02")
    assert s.weight == g.config["surprise_sprouting_weight"]
    assert s.metadata["creation_mode"] == "surprise_driven" and s.metadata["expected_target"] == "n01"
    assert s.salience == min(1.0 + 0.8 * 0.5 * 4.0, g.config["he_salience_max"])
    assert g._sprout_tally_stats["surprise_sprouted"] == 1 and g._sprout_tally_stats["surprise_calls"] == 3


def test_surprise_keys_absent_still_sprouts_today(monkeypatch):
    g = NF.Graph()
    for i in range(4):
        g.create_node(node_id=f"n{i:02d}")
    pred = NF.Prediction(source_node_id="n00", target_node_id="n01", strength=0.8, confidence=0.5)
    g._surprise_exploration(pred, {"n01", "n02", "n03"})
    assert edges(g) == [("n00", "n02"), ("n00", "n03")]
    assert g.cofire_tally_state() == []


@pytest.mark.parametrize("bad", [{"sprout_tally_slots": None}, {"sprout_tally_slots": 0},
                                 {"sprout_tally_slots": 2.0}, {"sprout_tally_theta": 1.0},
                                 {"sprout_tally_theta": float("nan")}, {"sprout_tally_horizon_steps": 0},
                                 {"sprout_tally_horizon_steps": True}])
def test_validation_raises_before_anything_is_touched(bad):
    g = graph(**bad)
    with pytest.raises(ValueError):
        cofire(g, ["n01"], ["n00"], t=100)
    assert g.cofire_tally_state() == [] and len(g.synapses) == 0


# ---------------------------------------------------------------------------
# (3) native == Python fallback, bit for bit
# ---------------------------------------------------------------------------

@needs_native
@pytest.mark.parametrize("seed", range(6))
def test_native_core_equals_python_fallback_on_random_call_sequences(seed):
    rng = random.Random(seed)
    nodes = [f"x{i:03d}" for i in range(40)]
    store = NF.ng_tract.SynapseStore()
    store.set_synapse_type_class(NF.SynapseType)
    store.set_synapse_class(NF.Synapse)
    conn = set()
    for _ in range(60):
        a, b = rng.sample(nodes, 2)
        s = NF.Synapse(pre_node_id=a, post_node_id=b, weight=0.1)
        store[s.synapse_id] = s
        conn.add((a, b))
    tally = {"k": 0, "tables": {}}
    t = 0
    for call in range(400):
        t += rng.choice([0, 1, 1, 2, 5, 30])
        k = 3 if call < 250 else 4                            # a K change empties both
        fired = rng.sample(nodes, rng.randint(1, 8))
        cands = [x for x in rng.sample(nodes, rng.randint(0, 25)) if x not in fired]
        th, lam, fl = 2.2 + (seed % 3) * 0.4, math.exp(-1 / 15), math.exp(-1)
        gap = seed % 3 + 1
        a = store.cofire_tally_update(fired, cands, t, k, th, lam, fl, gap)
        b = NF._cofire_tally_update_python(tally, fired, cands, t, k, th, lam, fl, gap,
                                           lambda o, p: (o, p) in conn or (p, o) in conn)
        assert [tuple(x) for x in a] == b
        nat = store.cofire_tally_state()
        py = sorted(tally["tables"].items())
        assert [(o, [None if x is None else (x[0], struct.pack("<d", x[1]), x[2]) for x in tbl]) for o, tbl in nat] == \
               [(o, [None if x is None else (x[0], struct.pack("<d", x[1]), x[2]) for x in tbl]) for o, tbl in py]


def tally_workload(rounds):
    """P1-style rounds with the tally on: a fixed group stimulated on separate occasions (two rounds every 6: repeated
    co-firing episodes) + random noise, a Tonic write tick, a recall, node churn, rewards, a disuse sleep every
    15 rounds (the sleep empties the tally); the tally state after every round."""
    def act(g, rng):
        log = []
        hot = sorted(rng.sample(list(g.nodes), 10))
        for k in range(rounds):
            if k % 6 < 2:
                for nid in (hot[:5] if k % 6 == 0 else hot[5:]):
                    if nid in g.nodes:
                        g.stimulate(nid, 2.5)
            for nid in rng.sample(list(g.nodes), min(6, len(g.nodes))):
                g.stimulate(nid, rng.uniform(0.5, 3.0))
            r = g.step()
            log.append((list(r.fired_node_ids), r.synapses_pruned, r.synapses_sprouted))
            log.append(P1.syn_state(g))
            ids = rng.sample(list(g.nodes), min(6, len(g.nodes)))
            p = g.prime_and_propagate(ids, [1.5] * len(ids), steps=3, write_mode=True)
            log.append(P1.fe(p.fired_entries))
            q = g.prime_and_propagate(ids[:3], [2.0] * 3, steps=4, write_mode=False)
            log.append(P1.fe(q.fired_entries))
            if k % 4 == 2:
                n = g.create_node(node_id=f"new{k}")
                g.create_synapse(n.node_id, rng.choice([x for x in g.nodes if x != n.node_id]), weight=0.3)
            if k % 7 == 3 and len(g.nodes) > 20:
                g.remove_node(rng.choice(sorted(g.nodes)))
            if k % 5 == 4:
                g.inject_reward(0.2)
            if k % 15 == 14:
                rec = g.sleep_cycle()
                log.append((rec["pruned"], rec["nodes_collected"], rec.get("tally_retracted")))
            log.append(tstate(g))
            log.append(sorted(g._sprout_tally_stats.items()) if "_sprout_tally_stats" in g.__dict__ else None)
        log.append(sorted((s.pre_node_id, s.post_node_id, sid) for sid, s in g.synapses.items()))
        return log, []
    return act


@needs_native
@pytest.mark.parametrize("seed", P1.SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_whole_run_native_equals_fallback(mode, seed, monkeypatch):
    start = P1.built_bytes(seed)
    cfg = dict(TALLY, **DISUSE)
    act = tally_workload(30)
    a, (la, _), ga = run(mode, start, act, seed, _prep(cfg))
    monkeypatch.setattr(NF, "_cofire_tally_native", lambda store: None)
    b, (lb, _), gb = run(mode, start, act, seed, _prep(cfg))
    assert "_cofire_tally" not in ga.__dict__ and "_cofire_tally" in gb.__dict__   # the two paths really ran
    P1.compare(la, lb, [], [], a, b)
    assert ga._sprout_tally_stats["sprouted"] > 0                                   # not vacuous


@pytest.mark.parametrize("seed", P1.SEEDS[:2])
def test_whole_run_tally_on_is_deterministic_and_sprouts_less(seed):
    start = P1.built_bytes(seed)
    cfg = dict(TALLY, **DISUSE)
    act = tally_workload(30)
    a, (la, _), ga = run("off", start, act, seed, _prep(cfg))
    b, (lb, _), _ = run("off", start, act, seed, _prep(cfg))
    P1.compare(la, lb, [], [], a, b)
    c, (lc, _), gc = run("off", start, act, seed, _prep(DISUSE))
    sprouted_tally = sum(x[2] for x in la if isinstance(x, tuple) and len(x) == 3 and isinstance(x[0], list))
    sprouted_today = sum(x[2] for x in lc if isinstance(x, tuple) and len(x) == 3 and isinstance(x[0], list))
    assert 0 < sprouted_tally < sprouted_today
