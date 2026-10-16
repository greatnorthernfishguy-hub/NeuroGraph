# ---- Changelog ----
# [2026-10-15] Claude Code (Opus 5.5, lane nglite-sleep) — new file: NGLite.sleep() rule tests (#1114 Phase 1)
# What: the rule's behaviour on the Python core (and on the Rust core when the installed ng_tract has
#   NGLiteCore.sleep): faint vs established downscale, evidence armor incl. contested links, grace counted in sleeps
#   (and reset when a link is lifted), orphan collection after the same grace, constitutional (rim) nodes and their
#   synapses frozen, the epoch (increments, persists through save/load, read via sleep_state / get_stats, absent from
#   a never-slept save), weight-only (last_updated untouched), the INFO line, parameter validation, and the at-cap
#   eviction that no longer takes the newest; Commons.sleep / sleep_state pass-throughs; persist_cc_commons's lock.
# Why: design ~/docs/superpowers/specs/2026-10-15-ng-lite-sleep-design.md §4 (unit tests for the rule). Python ==
#   Rust bit-for-bit is proven in ng-tract-rs tests/test_ng_lite_core.py.
# How: tiny NG-Lites (embedding_dim 8, raw comparisons, no receptor layer); states built with record_outcome, plus
#   direct synapse edits on the Python core where a weight has to sit at an exact value.
# -------------------
import logging
import math
import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ng_lite  # noqa: E402
from ng_lite import NGLite, NGLiteSynapse  # noqa: E402

CFG = {"embedding_dim": 8, "mean_centre": False, "receptor_layer_enabled": False}


def vec(seed):
    v = np.random.default_rng(seed).standard_normal(8).astype(np.float32)
    return v / np.linalg.norm(v)


def make(cfg=None, rust=False):
    """rust=True: the Rust core (ng_tract exports NGLiteCore only when the switch was set at ITS import, so the
    class is put in place around the constructor, as ng-tract-rs tests/test_ng_lite_core.py does)."""
    env = os.environ.get("NG_LITE_RUST_CORE")
    nt = sys.modules.get("ng_tract")
    had = nt is not None and hasattr(nt, "NGLiteCore")
    try:
        if rust:
            import ng_tract
            nt = ng_tract
            had = hasattr(nt, "NGLiteCore")
            nt.NGLiteCore = nt._NGLiteCore
            os.environ["NG_LITE_RUST_CORE"] = "1"
        else:
            os.environ.pop("NG_LITE_RUST_CORE", None)
        ng = NGLite("t", {**CFG, **(cfg or {})})
    finally:
        if rust and not had:
            delattr(nt, "NGLiteCore")
        if env is None:
            os.environ.pop("NG_LITE_RUST_CORE", None)
        else:
            os.environ["NG_LITE_RUST_CORE"] = env
    assert (ng._core is not None) == rust
    return ng


def _rust_available():
    try:
        import ng_tract
    except ImportError:
        return False
    core = getattr(ng_tract, "_NGLiteCore", None)
    return core is not None and hasattr(core, "sleep")


CORES = ["python"] + (["rust"] if _rust_available() else [])


def put(ng, src_seed, target, w, success=1, failure=0, variance_m2=0.0):
    """A synapse at an exact weight (Python core): one real deposit, then the fields set."""
    node = ng.find_or_create_node(vec(src_seed))
    ng.record_outcome(vec(src_seed), target, True)
    s = ng.synapses[(node.node_id, target)]
    s.weight, s.success_count, s.failure_count = w, success, failure
    s.welford_count, s.welford_m2 = success + failure, variance_m2
    return node.node_id


def w_of(ng, src, tgt):
    s = ng.get_synapse(src, tgt)
    return None if s is None else s.weight


# ---------------------------------------------------------------- the rule

def test_faint_link_loses_about_d0_established_about_d0_h_over_w():
    ng = make()
    a = put(ng, 1, "faint", 0.004)
    b = put(ng, 2, "strong", 0.9)
    ng.sleep(now=1.0)
    d0, h = ng_lite.DEFAULT_CONFIG["sleep_d0"], ng_lite.DEFAULT_CONFIG["sleep_h"]
    assert w_of(ng, a, "faint") == 0.004 * (1.0 - d0 * h / (h + 0.004))
    assert w_of(ng, b, "strong") == 0.9 * (1.0 - d0 * h / (h + 0.9))
    faint_loss, strong_loss = 1 - w_of(ng, a, "faint") / 0.004, 1 - w_of(ng, b, "strong") / 0.9
    assert faint_loss > 0.9 * d0                               # a faint link (w << h) loses ~d0
    assert strong_loss < 0.5 * faint_loss                      # an established link loses far less (~d0*h/(h+w))


def test_armor_grows_with_outcomes_and_a_contested_link_counts_its_evidence():
    ng = make()
    one = put(ng, 1, "one", 0.5, success=1)
    many = put(ng, 2, "many", 0.5, success=9)
    # contested: evidence on both sides (variance > 0.002, 0.15 <= w <= 0.85), same outcome count as "many"
    cont = put(ng, 3, "cont", 0.5, success=5, failure=4, variance_m2=0.05 * 9)
    assert ng.synapses[(cont, "cont")].is_contested
    ng.sleep(now=1.0)
    loss = {k: 1 - w_of(ng, s, k) / 0.5 for s, k in ((one, "one"), (many, "many"), (cont, "cont"))}
    assert loss["many"] < loss["one"] and loss["cont"] == loss["many"]
    armor = 1.0 + 1.0 * math.log(9.0)
    d0, h = ng_lite.DEFAULT_CONFIG["sleep_d0"], ng_lite.DEFAULT_CONFIG["sleep_h"]
    assert w_of(ng, many, "many") == 0.5 * (1.0 - d0 * h / (h + 0.5) / armor)
    # the cap
    assert ng._sleep_armor(NGLiteSynapse(success_count=10 ** 6)) == 5.0


def test_grace_is_counted_in_sleeps_not_deposits_and_a_lifted_link_starts_over():
    ng = make()
    a = put(ng, 1, "faint", 0.005)
    b = put(ng, 2, "lifted", 0.005)
    for _ in range(50):                                   # deposits elsewhere do not advance any grace
        ng.record_outcome(vec(9), "other", True)
    r1 = ng.sleep(now=1.0)
    assert r1["faint"] == 2 and r1["cleared"] == 0
    ng.synapses[(b, "lifted")].weight = 0.5               # wake lifted it above the threshold
    r2 = ng.sleep(now=2.0)
    assert ng.get_synapse(b, "lifted").metadata.get("sleep_faint") is None
    assert r2["cleared"] == 0 and ng.get_synapse(a, "faint").metadata["sleep_faint"] == 2
    r3 = ng.sleep(now=3.0)                                # grace 2: below at three sleeps in a row -> cleared
    assert r3["cleared"] == 1 and ng.get_synapse(a, "faint") is None
    assert ng.get_synapse(b, "lifted") is not None


def test_grace_zero_clears_at_the_first_sleep_below():
    ng = make({"sleep_grace_sleeps": 0})
    a = put(ng, 1, "faint", 0.005)
    assert ng.sleep(now=1.0)["cleared"] == 1 and ng.get_synapse(a, "faint") is None


def test_orphan_node_is_removed_after_the_same_grace():
    ng = make()
    a = put(ng, 1, "faint", 0.005)
    n_before = ng.node_count()
    reps = [ng.sleep(now=float(k)) for k in range(6)]
    assert reps[2]["cleared"] == 1                        # its last link goes at sleep 3
    # orphaned in that same sleep (orphan sleep 1), sleep 4 = 2, sleep 5 = 3 > grace -> collected: like a link,
    # three sleeps in a row without a link
    assert [r["orphans_collected"] for r in reps] == [0, 0, 0, 0, 1, 0]
    assert [r["orphans_waiting"] for r in reps][2:4] == [1, 1]
    assert ng.node_count() == n_before - 1 and ng.get_node(a) is None
    assert all(h in ng.nodes for h in ng._embedding_cache)


def test_constitutional_nodes_and_their_synapses_are_frozen():
    rim = [{"embedding": [float(x) for x in vec(42)], "description": "rim"}]
    ng = make({"constitutional_embeddings": rim})
    rim_node = next(n for n in ng.nodes.values() if n.constitutional)
    # a synapse from a rim node (old states can carry them; record_outcome never creates one)
    ng.synapses[(rim_node.node_id, "x")] = NGLiteSynapse(rim_node.node_id, "x", weight=0.001)
    for k in range(5):
        r = ng.sleep(now=float(k))
        assert r["frozen_nodes"] == 1 and r["frozen_synapses"] == 1
    assert ng.synapses[(rim_node.node_id, "x")].weight == 0.001
    assert ng.synapses[(rim_node.node_id, "x")].metadata == {}
    assert rim_node.constitutional and rim_node.embedding_hash in ng.nodes   # never collected as an orphan


def test_sleep_touches_weight_only():
    ng = make()
    ng.record_outcome(vec(1), "t", True, metadata={"text": "x"})
    before = {k: (s.last_updated, s.success_count, s.activation_count, s.welford_m2, s.metadata.get("last_context"))
              for k, s in ng.synapses.items()}
    ng.sleep(now=5.0)
    after = {k: (s.last_updated, s.success_count, s.activation_count, s.welford_m2, s.metadata.get("last_context"))
             for k, s in ng.synapses.items()}
    assert before == after


# ---------------------------------------------------------------- the epoch, the report

@pytest.mark.parametrize("core", CORES)
def test_epoch_increments_persists_and_is_on_the_read_api(tmp_path, core):
    ng = make(rust=core == "rust")
    ng.record_outcome(vec(1), "t", True)
    assert ng.sleep_state() == {"epoch": 0, "last_sleep_at": None}
    assert "sleep" not in ng.export_state()                # a never-slept state saves as before
    ng.sleep(now=100.0)
    ng.sleep(now=200.5)
    assert ng.sleep_state() == {"epoch": 2, "last_sleep_at": 200.5}
    st = ng.get_stats()
    assert st["sleep_epoch"] == 2 and st["last_sleep_at"] == 200.5
    p = ng.save(str(tmp_path / "s.msgpack"))
    for c2 in CORES:
        ng2 = make(rust=c2 == "rust")
        assert ng2.load(p)
        assert ng2.sleep_state() == {"epoch": 2, "last_sleep_at": 200.5}
        assert ng2.sleep(now=300.0)["epoch"] == 3


@pytest.mark.parametrize("core", CORES)
def test_report_and_info_line_every_sleep(caplog, core):
    ng = make(rust=core == "rust")
    with caplog.at_level(logging.INFO, logger="ng_lite"):
        r = ng.sleep(now=1.0)
        ng.sleep(now=2.0)
    lines = [m for m in caplog.messages if m.startswith("NG-Lite sleep [t]")]
    assert len(lines) == 2 and "epoch=2" in lines[1]
    for k in ("epoch", "last_sleep_at", "synapses_before", "synapses_after", "nodes_before", "nodes_after",
              "downscaled", "faint", "cleared", "frozen_synapses", "orphans_waiting", "orphans_collected",
              "frozen_nodes", "core", "seconds"):
        assert k in r
    assert r["core"] == core


@pytest.mark.parametrize("bad", [{"sleep_h": 0.0}, {"sleep_d0": 1.5}, {"sleep_armor_max": 0.5},
                                 {"sleep_grace_sleeps": -1}, {"sleep_d0": "x"}])
def test_bad_parameters_are_refused_and_nothing_changes(bad):
    ng = make(bad)
    ng.record_outcome(vec(1), "t", True)
    before = ng.export_state()["synapses"]
    with pytest.raises(ValueError, match=next(iter(bad))):
        ng.sleep()
    assert ng.export_state()["synapses"] == before and ng.sleep_state()["epoch"] == 0


def test_sleep_parameters_are_tunable_and_clamped():
    ng = make()
    assert ng.update_tunable("sleep_d0", 0.9)["new_value"] == 0.5
    assert ng.update_tunable("sleep_grace_sleeps", 3)["new_value"] == 3.0
    for k in ("sleep_d0", "sleep_h", "sleep_grace_sleeps", "sleep_armor_gain", "sleep_armor_max"):
        assert k in ng.get_tunables() and k in ng_lite.DEFAULT_CONFIG
    assert (ng_lite.DEFAULT_CONFIG["sleep_d0"], ng_lite.DEFAULT_CONFIG["sleep_h"],
            ng_lite.DEFAULT_CONFIG["sleep_grace_sleeps"]) == (0.2, 0.5, 2)   # Josh 2026-10-16, from the Commons harness


# ---------------------------------------------------------------- caps

def test_at_cap_synapse_eviction_is_not_newest_first():
    ng = make({"max_synapses": 3})
    ng.record_outcome(vec(1), "a", True)
    ng.record_outcome(vec(1), "b", True)
    ng.sleep(now=1.0)                                     # a and b downscaled
    ng.record_outcome(vec(1), "c", True)                  # newest, fresh 0.575
    ng.record_outcome(vec(1), "d", True)                  # at cap: evicts the weakest old one, never c
    targets = [s.target_id for s in ng.list_synapses()]
    assert targets == ["b", "c", "d"]


def test_at_cap_eviction_prefers_the_longest_faint_then_armored_strength():
    ng = make({"max_synapses": 4})
    src = put(ng, 1, "strong_armored", 0.3, success=20)
    put(ng, 1, "faint_long", 0.005)
    put(ng, 1, "plain", 0.2)
    ng.sleep(now=1.0)
    ng.sleep(now=2.0)                                     # faint_long: 2 faint sleeps, still in grace
    put(ng, 1, "faint_new", 0.002)                        # weaker, but not faint for any sleep yet
    ng.record_outcome(vec(1), "e", True)                  # at cap
    left = {s.target_id for s in ng.list_synapses(source_id=src)}
    assert "faint_long" not in left and {"faint_new", "strong_armored", "plain", "e"} <= left
    ng.record_outcome(vec(1), "f", True)                  # next: lowest w * armor -> faint_new
    left = {s.target_id for s in ng.list_synapses(source_id=src)}
    assert "faint_new" not in left and "strong_armored" in left


def test_at_cap_node_eviction_takes_an_orphan_before_a_linked_node():
    ng = make({"max_nodes": 3})
    ng.find_or_create_node(vec(1))                        # orphan (no synapse)
    ng.record_outcome(vec(2), "t2", True)
    ng.record_outcome(vec(3), "t3", True)
    ng.sleep(now=1.0)
    ng.record_outcome(vec(4), "t4", True)                 # at cap: the orphan goes, not a linked or the new node
    kept = {s.target_id for s in ng.list_synapses()}
    assert kept == {"t2", "t3", "t4"} and ng.node_count() == 3


# ---------------------------------------------------------------- Commons, persist

def test_commons_sleep_and_sleep_state_pass_through(tmp_path):
    from commons import Commons
    c = Commons(config=CFG)
    c.deposit(vec(1), "cc:x", metadata={"text": "x"})
    assert c.sleep_state() == {"epoch": 0, "last_sleep_at": None}
    r = c.sleep(now=10.0)
    assert r["epoch"] == 1 and c.sleep_state() == {"epoch": 1, "last_sleep_at": 10.0}
    c.persist(str(tmp_path / "c.msgpack"))
    c2 = Commons(config=CFG)
    c2.restore(str(tmp_path / "c.msgpack"))
    assert c2.sleep_state()["epoch"] == 1 and c2.bucket_recent(limit=5)[0][0] == "cc:x"


def test_persist_cc_commons_serializes_writers(tmp_path, monkeypatch):
    import cc_ng_organism as org
    inside, peak = [0], [0]
    gate = threading.Lock()

    class Slow:
        def persist(self, path):
            with gate:
                inside[0] += 1
                peak[0] = max(peak[0], inside[0])
            threading.Event().wait(0.05)
            with gate:
                inside[0] -= 1

    monkeypatch.setattr(org, "_cc_commons", Slow())
    ts = [threading.Thread(target=org.persist_cc_commons, args=(str(tmp_path),)) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert peak[0] == 1


def test_rust_core_without_sleep_fails_loud_and_reads_stay_truthful(caplog):
    """A pre-#1114 ng_tract wheel with the new ng_lite.py (install order): sleep raises naming the wheel, logged at
    ERROR once; get_stats / sleep_state report the core's truth (it cannot sleep, so epoch 0)."""
    class OldCore:
        def get_stats(self):
            return {k: 0 for k in ("node_count", "synapse_count", "max_nodes", "max_synapses", "memory_estimate_bytes",
                                   "total_outcomes", "total_successes", "success_rate", "avg_synapse_weight",
                                   "embedding_cache_size")}
    ng = make()
    ng._core = OldCore()
    with caplog.at_level(logging.ERROR, logger="ng_lite"):
        for _ in range(2):
            with pytest.raises(RuntimeError, match="predates #1114"):
                ng.sleep()
    assert sum("predates #1114" in m for m in caplog.messages) == 1
    assert ng.sleep_state() == {"epoch": 0, "last_sleep_at": None}
    assert ng.get_stats()["sleep_epoch"] == 0 and ng.get_stats()["last_sleep_at"] is None
