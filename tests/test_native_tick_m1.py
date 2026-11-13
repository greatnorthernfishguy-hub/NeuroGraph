# ---- Changelog ----
# [2026-11-13] DudeMan CC (Claude Fable 5.2, laptop) -- native tick piece 1 (punch #1221): the engine against itself
#   before the port.
# What: the engine on this branch (predictions, confirmations and misses in ng_tract.PredictionBook) beside the engine
#       at ec3c1f5 (the same passes in Python), loaded from git: twin runs, final checkpoints byte for byte.
# Why:  plan docs/superpowers/plans/2026-11-13-native-tick-m1-prediction-miss-path.md §3 and §7.
# How:  the oracle is `git show ec3c1f5:neuro_foundation.py` (tests.test_hyperedge_store_h2._load). Two differences are
#       deliberate and are levelled here so that everything else must be equal: (1) a pair of nodes joined by two
#       synapses resolves to the lowest row (the oracle's walk of a set of ids follows the process's str hash order:
#       punch #1223), so the oracle gets the same rule; (2) #1222's three extra keys on an open receiver prediction are
#       switched off for the byte comparison and tested on their own below.
#       [2026-11-13, first version] before the engine was edited this file ran the book through a shim that replaced
#       two methods of an unedited twin (15 cases byte-identical); the engine edit replaced the shim.
# -------------------
import os
import random
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import neuro_foundation as NF  # noqa: E402
from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_sleep_p1 as P1              # noqa: E402
from tests import test_sprout_1050 as T1050        # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s18 as S18        # noqa: E402

HAVE_BOOK = hasattr(NF.ng_tract, "PredictionBook")
pytestmark = pytest.mark.skipif(not HAVE_BOOK, reason="installed ng_tract has no PredictionBook")

PREPORT_REV = os.environ.get("NG_NATIVE_TICK_M1_PREPORT_REV", "ec3c1f5")
PRE = H2._load(PREPORT_REV, "nf_native_tick_m1_preport") if HAVE_BOOK else None


def _find_first(self, pre_id, post_id):
    sid = self.synapses.find_first(pre_id, post_id)
    return self.synapses.get(sid) if sid is not None else None


@pytest.fixture(autouse=True)
def _levelled(monkeypatch):
    if PRE is not None:
        monkeypatch.setattr(PRE.Graph, "_find_synapse", _find_first)
    monkeypatch.setattr(NF.Graph, "_PB_SAVE_RECEIVER_PART", False)


def _restore(mod, b):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        with open(p, "wb") as f:
            f.write(b)
        g = mod.Graph(native_node_store=True)
        g.restore(p)
        return g


# ---------------------------------------------------------------------------
# (1) receiver on (S18), with and without the corrected DiffPC (S19)
# ---------------------------------------------------------------------------

@S18.needs_recv
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("s19", [False, True])
def test_receiver_whole_run_equals_the_preport_engine(seed, s19):
    start = H2.built(seed)
    t = S.Pair(S18.restore_as(PRE, "native", start), S18.restore_as(NF, "native", start), seed)
    extra = {"diffpc_receiver_enabled": True} if s19 else {}
    t.op("keys", lambda g: g.config.update(dict(S18.ARM, **S18.RPE, **S18.ON, **extra)))
    S.drive(t, random.Random(seed), 25, sleep_every=True)
    g = t.b.g
    assert g._recv_stats["registered"] > 0
    assert g._total_predictions_confirmed + g._total_predictions_errors > 0          # not vacuous
    assert S18._state_rows(t.a.g) == S18._state_rows(t.b.g)
    assert S18.ckpt(t.a.g) == S18.ckpt(t.b.g)


# ---------------------------------------------------------------------------
# (2) source-side with the co-firing tally (the live daemon's path), with and without listeners
# ---------------------------------------------------------------------------

def _run(mod, start, act, seed, cfg, heard):
    g = _restore(mod, start)
    g.config.update(cfg)
    log = T1050._heard(g) if heard else []
    out = P1.seeded(seed, lambda rng: act(g, rng), uuid_stream=1)
    return P1.ckpt_bytes(g), out, log, g


@T1050.needs_window
@pytest.mark.parametrize("seed", P1.SEEDS)
@pytest.mark.parametrize("heard", [False, True])
@pytest.mark.parametrize("tally", [True, False])
def test_source_side_whole_run_equals_the_preport_engine(seed, heard, tally):
    start = P1.built_bytes(seed)
    cfg = dict(T1050.DISUSE, prediction_threshold=0.5)   # the workload's weights predict (default bar 3.0)
    if tally:
        cfg.update(T1050.TALLY)
    act = T1050.tally_workload(30) if tally else _plain_workload(30)
    a, (la, _), ea, ga = _run(PRE, start, act, seed, cfg, heard)
    b, (lb, _), eb, gb = _run(NF, start, act, seed, cfg, heard)
    assert gb._total_predictions_errors == ga._total_predictions_errors > 0          # misses really happened
    assert ea == eb and (len(ea) > 0) == heard                                       # every event, with its ids
    assert ga._novel_sequence_log == list(gb._novel_sequence_log)
    P1.compare(la, lb, [], [], a, b)


def _plain_workload(rounds):
    """The #1050 workload without its tally reads (the no-tally path wires every node of the fired window)."""
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
            if k % 7 == 3 and len(g.nodes) > 20:
                g.remove_node(rng.choice(sorted(g.nodes)))
            if k % 5 == 4:
                g.inject_reward(0.2)
        log.append(sorted((s.pre_node_id, s.post_node_id, sid) for sid, s in g.synapses.items()))
        return log, []
    return act


# ---------------------------------------------------------------------------
# (3) a save in mid-flight restores into the same run
# ---------------------------------------------------------------------------

@S18.needs_recv
@pytest.mark.parametrize("seed", [0, 1])
def test_restore_mid_flight_equals_the_preport_engine(seed):
    start = H2.built(seed)
    t = S.Pair(S18.restore_as(PRE, "native", start), S18.restore_as(NF, "native", start), seed)
    t.op("keys", lambda g: g.config.update(dict(S18.ARM, **S18.RPE, **S18.ON)))
    S.drive(t, random.Random(seed), 12)
    mid_a, mid_b = S18.ckpt(t.a.g), S18.ckpt(t.b.g)
    assert mid_a == mid_b and len(t.b.g.active_predictions) > 0
    t2 = S.Pair(S18.restore_as(PRE, "native", mid_a), S18.restore_as(NF, "native", mid_b), seed + 50)
    t2.op("keys", lambda g: g.config.update(dict(S18.ARM, **S18.RPE, **S18.ON)))
    S.drive(t2, random.Random(seed + 50), 12, sleep_every=True)
    assert S18.ckpt(t2.a.g) == S18.ckpt(t2.b.g)


# ---------------------------------------------------------------------------
# (4) #1222: an open receiver prediction survives a save / restore as a receiver prediction
# ---------------------------------------------------------------------------

@S18.needs_recv
def test_receiver_prediction_keeps_its_receiver_part_across_a_restore(monkeypatch):
    monkeypatch.setattr(NF.Graph, "_PB_SAVE_RECEIVER_PART", True)
    start = H2.built(0)
    g = S18.restore_as(NF, "native", start)
    g.config.update(dict(S18.ARM, **S18.RPE, **S18.ON))
    rng = random.Random(3)
    for _ in range(12):
        for nid in rng.sample(list(g.nodes.keys()), 8):
            g.stimulate(nid, 2.5)
        g.step()
        if any(p.contributors for p in g.active_predictions.values()):
            break
    before = {pid: p for pid, p in g.active_predictions.items() if p.contributors}
    assert before, "the run produced no open receiver prediction"
    g2 = S18.restore_as(NF, "native", S18.ckpt(g))
    after = dict(g2.active_predictions.items())
    for pid, p in before.items():
        q = after[pid]
        assert q.contributors == p.contributors and q.contributors_raw == p.contributors_raw
        assert (q.drive, q.rho, q.source_node_id, q.strength, q.confidence) == \
               (p.drive, p.rho, p.source_node_id, p.strength, p.confidence)
    S18.restore_as(PRE, "native", S18.ckpt(g))            # the older loader still reads the file


# ---------------------------------------------------------------------------
# (5) the registration calls: merge and the pending index follow the engine's own arithmetic
# ---------------------------------------------------------------------------

def test_receiver_merge_and_pending_index_follow_the_engine():
    g = NF.Graph()
    b = NF.ng_tract.PredictionBook()
    P = NF.Prediction
    p1 = P(target_node_id="T", strength=0.4, confidence=0.5, created_at=1, expires_at=11, drive=0.3, rho=0.2)
    syns1 = {"s1": ("a", 0.25), "s2": ("b", 0.5)}
    g._receiver_set_contributors(p1, syns1)
    src = b.insert("p1", None, "T", 0.4, 0.5, 1, 11, 0, None, 0.0, ([(k, v[0], v[1]) for k, v in syns1.items()], 0.3, 0.2))
    assert src == p1.source_node_id and b.contributors("p1") == p1.contributors
    b.insert("p0", "x", "U", 0.1, 0.1, 1, 11)                       # a source-side prediction: not pending
    b.insert("p2", None, "T", 0.2, 0.2, 1, 11, 0, None, 0.0, ([("s9", "z", 1.0)], 0.1, 0.1))
    assert b.receiver_pending() == [("T", "p2")]                    # the LAST receiver prediction on a target
    new = {"s2": ("b", 0.75), "s3": ("c", 0.9)}
    merged = dict(p1.contributors_raw)
    merged.update(new)
    g._receiver_set_contributors(p1, merged)
    p1.strength, p1.confidence = max(p1.strength, 0.3), max(p1.confidence, 0.9)
    assert b.receiver_merge("p1", [(k, v[0], v[1]) for k, v in new.items()], 0.3, 0.9)
    (ids, nums, rest) = b.get("p1")
    assert ids == ("p1", p1.source_node_id, "T") and nums[:2] == (p1.strength, p1.confidence)
    assert b.contributors("p1") == p1.contributors and rest[2:] == (True, 0.3, 0.2)
    assert not b.receiver_merge("p0", [], 1.0, 1.0) and not b.receiver_merge("nope", [], 1.0, 1.0)
    assert b.targets() == ["T", "U", "T"] and b.targets(True) == ["T", "T"] and len(b) == 3
