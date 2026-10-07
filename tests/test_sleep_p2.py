# ---- Changelog ----
# [2026-10-07] Claude (lane sleep-prearm) — by intent (#1066): the whole runs with disuse ON no longer call
#   compete_protected_links (the engine refuses it under disuse); they assert the refusal instead (P1.workload compete=).
# [2026-10-07] Claude (lane sleep-p2) — CREATE: sleep phase P2 (disuse) equivalence, rules and invariants
# What: (1) keys ABSENT: the branch == the P1 tip 149fa1f exactly over the P1 whole-run workload (step, Tonic ticks
#       with aging, recall, node churn, rewards, want-hub competition, sleep_downscale, snapshots; per-step synapse
#       state bitwise; final checkpoint bytes), with and without structural_plasticity_in_sleep + P1 sleep_cycle.
#       (2) the disuse sleep's rules on hand-built graphs: validation touches nothing; migration (counters -> 0,
#       stamps dropped, first sleep only tags); G in sleeps; activity + age clauses off by argument; the
#       strength-aware, salience-armored downscale (formula, weight-only, protected floor); native == Python fallback
#       bitwise; the D14 shield (kappa, negative traces, re-test); lifelines; last-link grace in sleeps; the
#       step-path marker drop; record / event / log line; D15 consistency.
#       (3) whole runs with disuse on: deterministic; lifelines of every protected node survive every sleep; no
#       dangling pred_weights.
# Why:  spec superpowers/specs/2026-10-06-sleep-phase-design.md §3, §8 P2, D1-D4, D11, D14.
# How:  BASE = `git show 149fa1f:neuro_foundation.py` under its own name; the P1 test module's builder/workload reused.
# -------------------
"""Sleep phase P2: strength-aware downscale, sleep-unit clearance, D14 shield, migration."""
import importlib.util
import logging
import math
import os
import random
import subprocess
import sys
import tempfile

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import neuro_foundation as NF  # noqa: E402  (the branch)
from tests import test_sleep_p1 as P1  # noqa: E402  (builder, workload, helpers)

BASE_REV = os.environ.get("NG_SLEEP_P2_BASE_REV", "149fa1f")
NATIVE_AWARE = hasattr(NF.ng_tract.SynapseStore, "scale_strength_aware")


def _load_base():
    src = subprocess.run(["git", "-C", _REPO, "show", f"{BASE_REV}:neuro_foundation.py"],
                         check=True, capture_output=True, text=True).stdout
    d = tempfile.mkdtemp(prefix="sleep_p2_base_")
    p = os.path.join(d, "nf_base_sleep_p2.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nf_base_sleep_p2", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nf_base_sleep_p2"] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base()
MODES = P1.MODES
DISUSE = {"sleep_disuse_enabled": True, "structural_plasticity_in_sleep": True, "sleep_downscale_d0": 0.1,
          "sleep_downscale_h": 0.05, "sleep_weight_grace_sleeps": 1, "sleep_last_link_grace_sleeps": 2,
          "sleep_credit_shield_kappa": 0.15}
NEW_KEYS = ("sleep_disuse_enabled", "sleep_downscale_d0", "sleep_downscale_h", "sleep_weight_grace_sleeps",
            "sleep_last_link_grace_sleeps", "sleep_credit_shield_kappa", "sleep_low_weight_unit",
            "sleep_cycles_completed")


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


def _prep(cfg):
    return lambda g: g.config.update(cfg)


# ---------------------------------------------------------------------------
# (1) keys absent: the P1 tip exactly
# ---------------------------------------------------------------------------

def test_new_keys_are_not_in_default_config():
    for k in NEW_KEYS:
        assert k not in NF.DEFAULT_CONFIG


def p1_sleep(g):
    r = g.sleep_cycle()
    return (r["pruned"], r["nodes_collected"], r["synapses_after"], r["nodes_after"], r["in_sleep_mode"])


@pytest.mark.parametrize("flags", ["absent", "lifeline", "lifeline_budget"])
@pytest.mark.parametrize("seed", P1.SEEDS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("in_sleep", [False, True])
def test_keys_absent_is_the_p1_tip_exactly(mode, seed, flags, in_sleep):
    start = P1.built_bytes(seed)
    cfg = dict(P1.FLAGS[flags])
    if in_sleep:
        cfg["structural_plasticity_in_sleep"] = True
    act = P1.workload(30, 6, p1_sleep)
    a, (la, sa) = run("base", start, act, seed, _prep(cfg))
    b, (lb, sb) = run(mode, start, act, seed, _prep(cfg))
    P1.compare(la, lb, sa, sb, a, b)


def test_step_path_drops_the_sleep_unit_marker_and_nothing_else():
    g = restore("off", P1.built_bytes(0))
    g.config["sleep_low_weight_unit"] = "sleeps"
    g.step()
    assert "sleep_low_weight_unit" not in g.config
    g2 = restore("off", P1.built_bytes(0))
    g2.config["structural_plasticity_in_sleep"] = True
    g2.config["sleep_low_weight_unit"] = "sleeps"
    g2.step()                                    # in sleep mode step() runs no prune: the marker stays
    assert g2.config["sleep_low_weight_unit"] == "sleeps"


# ---------------------------------------------------------------------------
# (2) the disuse sleep on hand-built graphs
# ---------------------------------------------------------------------------

def small_graph(mode="off", **cfg):
    g = NF.Graph(dict({"prune_protected_faint_links": True, "orphan_node_grace_period": 0}, **cfg),
                 native_node_store=(mode == "on"))
    for i in range(8):
        g.create_node(node_id=f"n{i}")
    g.create_node(node_id="P", metadata={"constitutional": True})
    g.timestep = 50_000
    return g


def syn(g, a, b, w, **attrs):
    s = g.create_synapse(a, b, weight=w)
    for k, v in attrs.items():
        setattr(s, k, v)
    return s.synapse_id


def enable(g, **over):
    g.config.update(DISUSE)
    g.config.update(over)


@pytest.mark.parametrize("missing", ["sleep_downscale_d0", "sleep_downscale_h", "sleep_weight_grace_sleeps",
                                     "sleep_last_link_grace_sleeps", "sleep_credit_shield_kappa",
                                     "structural_plasticity_in_sleep"])
def test_validation_refuses_before_touching_anything(missing):
    g = small_graph()
    syn(g, "n0", "n1", 0.001, low_weight_steps=9)
    enable(g)
    g.config.pop(missing)
    before = P1.ckpt_bytes(g)
    with pytest.raises(ValueError):
        g.sleep_cycle()
    assert P1.ckpt_bytes(g) == before


@pytest.mark.parametrize("key,val", [("sleep_downscale_d0", 1.5), ("sleep_downscale_d0", -0.1),
                                     ("sleep_downscale_h", 0.0), ("sleep_downscale_h", float("inf")),
                                     ("sleep_weight_grace_sleeps", -1), ("sleep_weight_grace_sleeps", 1.0),
                                     ("sleep_weight_grace_sleeps", True), ("sleep_credit_shield_kappa", -0.1),
                                     ("sleep_credit_shield_kappa", float("nan"))])
def test_validation_rejects_bad_values(key, val):
    g = small_graph()
    enable(g, **{key: val})
    with pytest.raises(ValueError):
        g.sleep_cycle()
    assert "sleep_low_weight_unit" not in g.config


def test_migration_resets_counters_and_stamps_and_the_first_sleep_only_tags():
    g = small_graph()
    a = syn(g, "n0", "n1", 0.001, low_weight_steps=4999)
    b = syn(g, "n1", "n2", 0.5, low_weight_steps=7)
    c = syn(g, "n2", "n3", 0.002, low_weight_steps=3)
    g.synapses[c].metadata = {"last_link_since": 49_000, "x": 1}
    syn(g, "n3", "n2", 0.6)
    syn(g, "n0", "n4", 0.9)                      # n0 / n1 are not left linkless (no last-link hold)
    enable(g, sleep_downscale_d0=0.0)
    r = g.sleep_cycle()
    assert r["migrated"] is True and r["lws_reset"] == 3 and r["stamps_cleared"] == 1
    assert r["pruned"] == 0 and r["sleep_index"] == 1
    assert g.synapses[a].low_weight_steps == 1 and g.synapses[b].low_weight_steps == 0
    assert g.synapses[c].metadata == {"x": 1}
    assert g.config["sleep_low_weight_unit"] == "sleeps" and g.config["sleep_cycles_completed"] == 1
    r2 = g.sleep_cycle()
    assert r2["migrated"] is False and r2["sleep_index"] == 2
    assert a not in g.synapses and c not in g.synapses and b in g.synapses


def test_marker_dropped_by_a_step_prune_re_migrates_next_sleep():
    g = small_graph()
    a = syn(g, "n0", "n1", 0.001)
    syn(g, "n1", "n0", 0.9)
    enable(g, sleep_downscale_d0=0.0)
    g.sleep_cycle()
    assert g.synapses[a].low_weight_steps == 1
    g.config["structural_plasticity_in_sleep"] = False
    g._prune_synapses()                         # a step-unit pass: counter now mixed
    assert "sleep_low_weight_unit" not in g.config
    g.config["structural_plasticity_in_sleep"] = True
    r = g.sleep_cycle()
    assert r["migrated"] is True and r["pruned"] == 0 and g.synapses[a].low_weight_steps == 1


@pytest.mark.parametrize("G", [0, 1, 2, 3])
def test_weight_grace_counts_sleeps(G):
    g = small_graph()
    a = syn(g, "n0", "n1", 0.001)
    syn(g, "n1", "n0", 0.9)
    enable(g, sleep_downscale_d0=0.0, sleep_weight_grace_sleeps=G, sleep_credit_shield_kappa=0.0)
    for k in range(1, 6):
        g.sleep_cycle()
        assert (a in g.synapses) == (k <= G), (G, k)


def test_a_link_lifted_between_sleeps_resets_its_count():
    g = small_graph()
    a = syn(g, "n0", "n1", 0.001)
    syn(g, "n1", "n0", 0.9)
    enable(g, sleep_downscale_d0=0.0)
    g.sleep_cycle()
    g.synapses[a].weight = 0.02                  # wake lifted it
    g.sleep_cycle()
    assert g.synapses[a].low_weight_steps == 0
    g.synapses[a].weight = 0.001
    g.sleep_cycle()
    assert a in g.synapses                        # tagged again, not cleared
    g.sleep_cycle()
    assert a not in g.synapses


def test_activity_and_age_clauses_are_off_by_argument():
    g = small_graph(inactivity_threshold=10, grace_period=100)
    a = syn(g, "n0", "n1", 0.05, inactive_steps=10_000, creation_time=0.0, peak_weight=0.05)
    syn(g, "n1", "n0", 0.9)
    enable(g, sleep_downscale_d0=0.0)
    for _ in range(4):
        g.sleep_cycle()
    assert a in g.synapses
    # control: the default path would remove it at once (activity + age clauses)
    g2 = small_graph(inactivity_threshold=10, grace_period=100)
    a2 = syn(g2, "n0", "n1", 0.05, inactive_steps=10_000, creation_time=0.0, peak_weight=0.05)
    syn(g2, "n1", "n0", 0.9)
    g2._prune_synapses()
    assert a2 not in g2.synapses


def _expect(w, sal, d0, h):
    s = sal if sal > 1.0 else 1.0
    d = d0 * h / (h + w) / s
    return w * (1.0 - d)


@pytest.mark.parametrize("mode", MODES)
def test_downscale_formula_weight_only_and_salience_armor(mode):
    g = small_graph(mode)
    cases = [(0.0, 1.0), (0.005, 1.0), (0.05, 1.0), (0.5, 1.0), (4.9, 1.0), (0.005, 3.0), (0.5, 2.0)]
    ids = []
    for i, (w, sal) in enumerate(cases):
        ids.append(syn(g, f"n{i}", f"n{(i + 1) % 8}", w, salience=sal, eligibility_trace=0.03 * i,
                       peak_weight=1.0, low_weight_steps=0))
    other = {sid: (g.synapses[sid].eligibility_trace, g.synapses[sid].peak_weight, g.synapses[sid].salience,
                   g.synapses[sid].delay) for sid in ids}
    r = g.sleep_downscale_strength_aware(0.2, 0.05)
    for sid, (w, sal) in zip(ids, cases):
        assert g.synapses[sid].weight == _expect(w, sal, 0.2, 0.05)
        s = g.synapses[sid]
        assert (s.eligibility_trace, s.peak_weight, s.salience, s.delay) == other[sid]
    assert r["synapses_scaled"] == len(cases) - 1          # 0.0 stays 0.0
    # strength-aware: the relative loss falls with strength; salience halves/thirds it
    loss = {(w, sal): 1 - g.synapses[sid].weight / w for sid, (w, sal) in zip(ids, cases) if w > 0}
    assert loss[(0.005, 1.0)] > loss[(0.05, 1.0)] > loss[(0.5, 1.0)] > loss[(4.9, 1.0)]
    assert math.isclose(loss[(0.005, 3.0)], loss[(0.005, 1.0)] / 3, rel_tol=1e-12)


@pytest.mark.parametrize("mode", MODES)
def test_downscale_keeps_the_protected_strongest_link_floor(mode):
    g = small_graph(mode)
    best = syn(g, "P", "n1", 0.015)
    worse = syn(g, "P", "n2", 0.004)
    inn = syn(g, "n3", "P", 0.3)
    r = g.sleep_downscale_strength_aware(1.0, 0.05)
    assert g.synapses[best].weight == 0.015                  # min(pre-pass 0.015, floor 0.02): restored
    assert g.synapses[inn].weight == _expect(0.3, 1.0, 1.0, 0.05)   # above its guard min(0.3, 0.02): untouched
    assert g.synapses[worse].weight < 0.004
    assert r["clamped"] == 1


def _random_store_graph(seed, mode):
    rng = random.Random(seed)
    g = NF.Graph(native_node_store=(mode == "on"))
    for i in range(60):
        g.create_node(node_id=f"r{i}", metadata=({"constitutional": True} if i < 3 else {}))
    for _ in range(700):
        a, b = rng.sample(range(60), 2)
        s = g.create_synapse(f"r{a}", f"r{b}", weight=rng.choice([0.0, rng.uniform(0, 0.02), rng.uniform(0, 5)]))
        s.salience = rng.choice([1.0, 1.0, rng.uniform(1, 5), 0.7])
    return g


@pytest.mark.skipif(not NATIVE_AWARE, reason="installed ng_tract has no scale_strength_aware")
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("d0,h", [(0.1, 0.05), (0.3, 0.02), (1.0, 0.5), (0.0, 0.05), (0.05, 1e-6)])
def test_native_equals_python_fallback_bitwise(seed, d0, h):
    g1, g2 = _random_store_graph(seed, "off"), _random_store_graph(seed, "off")
    prot = g1._strength_protected_ids()
    r1 = dict(g1.synapses.scale_strength_aware(d0, h, prot, 0.02))
    r2 = NF._scale_strength_aware_python(g2.synapses, g2._outgoing, g2._incoming, d0, h, prot, 0.02)
    assert r1 == r2
    assert [P1.P(g1.synapses.get_weight(s)) for s in g1.synapses.keys()] == \
           [P1.P(g2.synapses.get_weight(s)) for s in g2.synapses.keys()]


@pytest.mark.skipif(not NATIVE_AWARE, reason="installed ng_tract has no scale_strength_aware")
def test_engine_uses_the_native_pass_when_present():
    g = _random_store_graph(0, "off")
    assert g.sleep_downscale_strength_aware(0.1, 0.05)["native"] is True


def test_shield_holds_pending_credit_and_re_tests_next_sleep():
    g = small_graph()
    held = syn(g, "n0", "n1", 0.005, eligibility_trace=0.04)      # 0.005 + 0.15*0.04 = 0.011 >= 0.01
    neg = syn(g, "n1", "n2", 0.005, eligibility_trace=-0.5)       # pending depression: no shield
    small = syn(g, "n2", "n3", 0.005, eligibility_trace=0.01)     # 0.0065 < 0.01
    for a, b in (("n1", "n0"), ("n2", "n1"), ("n3", "n2")):
        syn(g, a, b, 0.9)
    enable(g, sleep_downscale_d0=0.0)
    g.sleep_cycle()
    r = g.sleep_cycle()
    assert held in g.synapses and neg not in g.synapses and small not in g.synapses
    assert r["shield_held_ids"] == [held] and r["shield_held"] == 1 and r["eligible"] == 3
    assert g.synapses[held].low_weight_steps == 2                 # keeps its count
    g.synapses[held].eligibility_trace = 0.001                    # credit decayed
    r3 = g.sleep_cycle()
    assert held not in g.synapses and r3["shield_held"] == 0


def test_shield_kappa_one_reward_floor_and_zero():
    for kappa, survives in ((0.015, False), (0.0, False), (0.15, True)):
        g = small_graph()
        a = syn(g, "n0", "n1", 0.005, eligibility_trace=0.04)
        syn(g, "n1", "n0", 0.9)
        enable(g, sleep_downscale_d0=0.0, sleep_credit_shield_kappa=kappa)
        g.sleep_cycle()
        g.sleep_cycle()
        assert (a in g.synapses) is survives, kappa


@pytest.mark.parametrize("mode", MODES)
def test_lifelines_survive_and_keep_their_count(mode):
    g = small_graph(mode)
    ll_out = syn(g, "P", "n1", 0.0001, low_weight_steps=0)
    ll_in = syn(g, "n2", "P", 0.0001)
    other = syn(g, "P", "n3", 0.00005)
    syn(g, "n1", "n2", 0.9)
    syn(g, "n3", "n4", 0.9)
    enable(g, sleep_downscale_d0=0.3)
    for _ in range(5):
        g.sleep_cycle()
    assert ll_out in g.synapses and ll_in in g.synapses and other not in g.synapses
    assert g.synapses[ll_out].weight == 0.0001 and g.synapses[ll_in].weight == 0.0001   # floored at pre-pass
    assert g.synapses[ll_out].low_weight_steps == 0


def test_last_link_grace_counts_sleeps_and_ignores_step_stamps():
    g = small_graph()
    only = syn(g, "n5", "n6", 0.001)             # n5 and n6 have no other link
    g.synapses[only].metadata = {"last_link_since": 1}           # an old STEP stamp (dropped by migration)
    syn(g, "n0", "n1", 0.9)
    enable(g, sleep_downscale_d0=0.0, sleep_last_link_grace_sleeps=2)
    alive = []
    for k in range(1, 7):
        r = g.sleep_cycle()
        alive.append(only in g.synapses)
        if only in g.synapses and k >= 2:
            assert g.synapses[only].metadata.get("last_link_since_sleep") == 2
    # sleep 1 tags; sleep 2: eligible, stamped (since=2), held; sleep 3: 3-2 < 2 held; sleep 4: expired -> removed
    assert alive == [True, True, True, False, False, False]
    assert "last_link_since" not in (g.synapses.get(only).metadata if only in g.synapses else {})


def test_last_link_grace_zero_disables_the_hold():
    g = small_graph()
    only = syn(g, "n5", "n6", 0.001)
    syn(g, "n0", "n1", 0.9)
    enable(g, sleep_downscale_d0=0.0, sleep_last_link_grace_sleeps=0)
    g.sleep_cycle()
    g.sleep_cycle()
    assert only not in g.synapses


def test_record_event_and_one_info_line(caplog):
    g = small_graph()
    a = syn(g, "n0", "n1", 0.001)
    syn(g, "n1", "n0", 0.9)
    events = []
    g.register_event_handler("sleep_cycle", lambda **kw: events.append(kw))
    enable(g)
    with caplog.at_level(logging.INFO, logger=NF.logger.name):
        g.sleep_cycle()
        r = g.sleep_cycle()
    lines = [x for x in caplog.records if x.getMessage().startswith("sleep_cycle(disuse)")]
    assert len(lines) == 2
    assert len(events) == 2 and events[1]["pruned"] == r["pruned"] == 1 and a not in g.synapses
    for k in ("disuse", "sleep_index", "params", "migrated", "lws_reset", "stamps_cleared", "downscaled",
              "downscale_clamped", "downscale_native", "eligible", "shield_held", "shield_held_ids",
              "last_link_held", "below_threshold_after", "seconds", "synapses_before", "synapses_after"):
        assert k in r
    assert r["params"] == {"d0": 0.1, "h": 0.05, "G": 1, "LL": 2, "kappa": 0.15}


def test_sleep_unit_arguments_are_refused_in_competing_mode_and_alone():
    g = small_graph()
    with pytest.raises(ValueError):
        g._prune_synapses(sleep_now=1)
    with pytest.raises(ValueError):
        g._prune_synapses(grace_sleeps=1, sleep_now=1, last_link_grace_sleeps=2, credit_shield_kappa=0.1,
                          competing_ids=[], excluded_ids=[], max_removals=1, order_key={})


def test_disuse_clearance_keeps_pred_weights_consistent():
    g = small_graph()
    a = syn(g, "n0", "n1", 0.001)
    syn(g, "n1", "n0", 0.9)
    g.nodes["n0"].pred_weights["n1"] = 0.7
    enable(g)
    g.sleep_cycle()
    g.sleep_cycle()
    assert a not in g.synapses and "n1" not in g.nodes["n0"].pred_weights
    assert P1.dangling(g) == [] or all(k.startswith("gone") for _, k in P1.dangling(g))


# ---------------------------------------------------------------------------
# (3) whole runs with disuse on
# ---------------------------------------------------------------------------

def compete_refused(g):
    """#1066 (sleep-prearm): under the disuse sleep the want-hub competition refuses before touching anything."""
    with pytest.raises(ValueError, match="#1066"):
        g.compete_protected_links(2, 10)
    return "compete-refused"


def disuse_sleep(record_into):
    def fn(g):
        prot = g._strength_protected_ids()
        ll_before = g._protected_lifelines(prot)
        deg = {p: (len(g._outgoing.get(p, ())), len(g._incoming.get(p, ()))) for p in prot}
        r = g.sleep_cycle()
        assert r["disuse"] is True
        record_into.append((ll_before, {p: (len(g._outgoing.get(p, ())), len(g._incoming.get(p, ()))) for p in prot},
                            deg))
        return (r["pruned"], r["nodes_collected"], r["synapses_after"], r["shield_held"], r["last_link_held"],
                r["downscaled"], r["sleep_index"])
    return fn


@pytest.mark.parametrize("seed", P1.SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_whole_run_disuse_on_lifelines_and_determinism(mode, seed):
    start = P1.built_bytes(seed)
    cfg = dict(P1.FLAGS["lifeline_budget"], **DISUSE)
    rec1, rec2 = [], []

    def prep(g):                     # the builder seeds pre-existing dangling entries (old removals): purge them first
        g.config.update(cfg)
        g.purge_dangling_pred_weights()
    a, (la, sa) = run(mode, start, P1.workload(30, 3, disuse_sleep(rec1), compete=compete_refused), seed, prep)
    b, (lb, sb) = run(mode, start, P1.workload(30, 3, disuse_sleep(rec2), compete=compete_refused), seed, prep)
    P1.compare(la, lb, sa, sb, a, b)
    assert len(rec1) == 10
    g = restore(mode, a)
    for ll_before, deg_after, deg_before in rec1:
        for p, (o, i) in deg_after.items():
            # a protected node that had a link in a direction keeps one (its lifeline) through the sleep
            ob, ib = deg_before[p]
            assert (o >= 1 or ob == 0) and (i >= 1 or ib == 0), p
    assert P1.dangling(g) == []
    assert g.config["sleep_cycles_completed"] == 10


@pytest.mark.parametrize("seed", P1.SEEDS[:2])
def test_whole_run_disuse_removes_and_is_not_vacuous(seed):
    start = P1.built_bytes(seed)
    cfg = dict(P1.FLAGS["lifeline"], **DISUSE)
    out = []
    run("off", start, P1.workload(30, 3, lambda g: out.append(g.sleep_cycle()) or 0, compete=compete_refused), seed,
        _prep(cfg))
    assert sum(r["pruned"] for r in out) > 0 and sum(r["downscaled"] for r in out) > 0
    assert out[0]["migrated"] and not any(r["migrated"] for r in out[1:])
